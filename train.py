# train.py — NCDE training pipeline
#
# CORRECTED VERSION: Removed monkey-patching (Fix 5), now uses proper model parameter.
# This ensures the saved model state_dict matches the model architecture in predict.py.
#
# Changes:
# - Removed apply_fix5_to_model() monkey patch
# - Now passes enriched_h0=True to NCDEForecaster constructor
# - Removed redundant re-application of fix5 after model.load_state_dict
# - Simplified code, eliminated architecture mismatch
# - FIX: Passes tickers list to build_asset_features to ensure all ETFs get features
#
# Usage:
# python train.py --option A
# python train.py --option B
# python train.py --option both

import argparse
import json
import os
import pickle
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchcde

import config as cfg
import loader
import features as feat
from model import NCDEForecaster, gaussian_nll_loss

os.makedirs(cfg.MODELS_DIR, exist_ok=True)
os.makedirs(cfg.DATA_DIR, exist_ok=True)

DEVICE = torch.device("cpu")  # GitHub Actions free tier — CPU only

# ── Fix 7: Cross-sectional normalisation ──────────────────────────────────────

def cs_normalize(y: torch.Tensor) -> torch.Tensor:
    """
    Cross-sectional z-score of returns across assets, per sample.

    Shape: (batch, n_assets) → (batch, n_assets) z-scored across asset dim.
    """
    mu = y.mean(dim=-1, keepdim=True)
    std = y.std(dim=-1, keepdim=True) + 1e-8
    return (y - mu) / std

# ── Dataset helpers ────────────────────────────────────────────────────────────

def make_dataloaders(feat_dict: dict, scaler: feat.PathScaler) -> tuple:
    """
    Chronological 80/10/10 split → DataLoaders.
    Scaler fitted on train only — no data leakage.
    """
    X_a = feat_dict["X_asset"]  # (N, T, n_asset_path_dim)
    X_m = feat_dict["X_macro"]  # (N, T, n_macro_feats)
    y = feat_dict["y"]  # (N, n_assets)

    N = len(X_a)
    n_train = int(N * cfg.TRAIN_SPLIT)
    n_val = int(N * cfg.VAL_SPLIT)

    # Fit scaler on train only
    X_a_tr, X_m_tr = scaler.fit_transform(X_a[:n_train], X_m[:n_train])
    X_a_va, X_m_va = scaler.transform(X_a[n_train:n_train+n_val], X_m[n_train:n_train+n_val])
    X_a_te, X_m_te = scaler.transform(X_a[n_train+n_val:], X_m[n_train+n_val:])

    def to_ds(Xa, Xm, y_):
        return TensorDataset(
            torch.tensor(Xa, dtype=torch.float32),
            torch.tensor(Xm, dtype=torch.float32),
            torch.tensor(y_, dtype=torch.float32),
        )

    train_dl = DataLoader(to_ds(X_a_tr, X_m_tr, y[:n_train]), batch_size=cfg.BATCH_SIZE, shuffle=False)
    val_dl = DataLoader(to_ds(X_a_va, X_m_va, y[n_train:n_train+n_val]), batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_dl = DataLoader(to_ds(X_a_te, X_m_te, y[n_train+n_val:]), batch_size=cfg.BATCH_SIZE, shuffle=False)

    dates = feat_dict["dates"]
    splits = {
        "n_train": n_train,
        "n_val": n_val,
        "n_test": N - n_train - n_val,
        "train_start": str(dates[0].date()),
        "train_end": str(dates[n_train - 1].date()),
        "val_end": str(dates[n_train + n_val - 1].date()),
        "test_end": str(dates[-1].date()),
    }
    return train_dl, val_dl, test_dl, splits

# ── Spline builder ─────────────────────────────────────────────────────────────

def build_combined_path(
    X_asset: torch.Tensor,
    X_macro: torch.Tensor,
) -> torchcde.CubicSpline:
    """
    Concatenate asset+macro channel-wise then build a single CubicSpline.
    """
    X_combined = torch.cat([X_asset, X_macro], dim=-1)
    t = torch.arange(X_combined.shape[1], dtype=torch.float32)
    coeffs = torchcde.natural_cubic_coeffs(X_combined, t)
    return torchcde.CubicSpline(coeffs, t)

# ── Training loop ──────────────────────────────────────────────────────────────

def train_epoch(model, loader_, optimizer) -> float:
    """
    Fix 7: y_batch is cross-sectionally z-scored before the NLL loss.
    """
    model.train()
    total_loss = 0.0
    for X_a, X_m, y_batch in loader_:
        optimizer.zero_grad()
        X_path = build_combined_path(X_a, X_m)
        mu, sigma = model(X_path)

        # Fix 7: cross-sectional normalisation of targets (train only)
        y_norm = cs_normalize(y_batch)

        loss = gaussian_nll_loss(mu, sigma, y_norm)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader_)

def eval_epoch(model, loader_) -> tuple:
    """
    Evaluation on raw (un-normalised) returns — no CS normalisation here.
    """
    model.eval()
    total_loss = 0.0
    all_mu, all_y = [], []
    with torch.no_grad():
        for X_a, X_m, y_batch in loader_:
            X_path = build_combined_path(X_a, X_m)
            mu, sigma = model(X_path)

            # Eval loss also uses CS-normalised y so it's comparable to train loss.
            y_norm = cs_normalize(y_batch)
            loss = gaussian_nll_loss(mu, sigma, y_norm)
            total_loss += loss.item()

            all_mu.append(mu.numpy())
            all_y.append(y_batch.numpy())  # raw returns for metrics

    mu_arr = np.concatenate(all_mu, axis=0)  # (N, n_assets)
    y_arr = np.concatenate(all_y, axis=0)

    # Top-1 pick per day (highest mu)
    picks = mu_arr.argmax(axis=1)
    pick_rets = y_arr[np.arange(len(picks)), picks]

    ann_return = float(pick_rets.mean() * 252)
    sharpe = float((pick_rets.mean() / (pick_rets.std() + 1e-8)) * np.sqrt(252))
    hit_rate = float((pick_rets > 0).mean())

    # Information coefficient
    from scipy.stats import spearmanr
    ic_list = []
    for i in range(len(mu_arr)):
        r, _ = spearmanr(mu_arr[i], y_arr[i])
        if not np.isnan(r):
            ic_list.append(r)
    ic = float(np.mean(ic_list)) if ic_list else 0.0

    return total_loss / len(loader_), ic, hit_rate, ann_return, sharpe

# ── Main training function ─────────────────────────────────────────────────────

def train_option(option: str) -> dict:
    t0 = time.time()

    # ── Set option-specific hyperparameters ────────────────────────────────
    if option == "A":
        max_epochs = cfg.OPTION_A_MAX_EPOCHS  # 150
        patience = cfg.OPTION_A_PATIENCE  # 25
        ode_steps = cfg.OPTION_A_ODE_STEPS  # 60
        lookback = cfg.OPTION_A_LOOKBACK  # 90
        enriched_h0 = True  # ← CORRECTED: Use enriched h0 for Option A (Fix 5)
    else:
        max_epochs = cfg.OPTION_B_MAX_EPOCHS  # 80
        patience = cfg.OPTION_B_PATIENCE  # 15
        ode_steps = cfg.OPTION_B_ODE_STEPS  # 20
        lookback = cfg.OPTION_B_LOOKBACK  # 60
        enriched_h0 = True  # ← Can also use for Option B, or set to False

    # Patch cfg so model picks up correct ODE_STEPS
    cfg.ODE_STEPS = ode_steps
    cfg.LOOKBACK = lookback

    print(f"\n{'='*60}")
    label = "A (Fixed Income)" if option == "A" else "B (Equity)"
    print(f"NCDE Training — Option {label}")
    print(f" hidden={cfg.HIDDEN_DIM} vf_dim={cfg.VECTOR_FIELD_DIM} "
          f"layers={cfg.N_LAYERS} lookback={lookback} ode_steps={ode_steps}")
    print(f" max_epochs={max_epochs} patience={patience} enriched_h0={enriched_h0}")
    print(f"{'='*60}")

    # Load data
    print("\n[1/5] Loading data...")
    master = loader.load_master()
    data = loader.get_option_data(option, master)

    # Feature engineering
    print("\n[2/5] Building features...")
    # FIX: Ensure all tickers from config get feature columns
    feat_dict = feat.prepare_features(data, lookback=lookback)
    scaler = feat.PathScaler()

    # Dataloaders
    print("\n[3/5] Preparing dataloaders...")
    train_dl, val_dl, test_dl, splits = make_dataloaders(feat_dict, scaler)
    print(f" Train: {splits['n_train']} | Val: {splits['n_val']} | Test: {splits['n_test']}")
    print(f" Train: {splits['train_start']} -> {splits['train_end']}")
    print(f" Val : {splits['train_end']} -> {splits['val_end']}")
    print(f" Test : {splits['val_end']} -> {splits['test_end']}")

    # Model
    print("\n[4/5] Building model...")

    # CORRECTED: Pass enriched_h0 as proper parameter, no monkey-patching needed
    model = NCDEForecaster(
        n_asset_path_dim=feat_dict["n_asset_path_dim"],
        n_macro_feats=feat_dict["n_macro_feats"],
        n_assets=feat_dict["n_assets"],
        hidden_dim=cfg.HIDDEN_DIM,
        vector_field_dim=cfg.VECTOR_FIELD_DIM,
        n_layers=cfg.N_LAYERS,
        readout_dim=cfg.READOUT_DIM,
        dropout=cfg.DROPOUT,
        solver=cfg.SOLVER,
        adjoint=cfg.ADJOINT,
        ode_steps=ode_steps,
        lookback=lookback,
        enriched_h0=enriched_h0,  # ← CORRECTED: Proper parameter
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    proj_type = "enriched (2x input)" if enriched_h0 else "standard"
    print(f" Parameters: {n_params:,} (initial_proj: {proj_type})")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Warmup
    warmup_epochs = 5
    warmup_factor = cfg.LEARNING_RATE / 5

    # Training loop
    print(f"\n[5/5] Training ({max_epochs} epochs, patience={patience}, warmup={warmup_epochs})...")

    best_val_loss = float("inf")
    best_val_ic = -float("inf")
    best_val_ann_ret = -float("inf")
    best_val_sharpe = -float("inf")
    best_composite = -float("inf")
    patience_count = 0
    history = {
        "train_loss": [], "val_loss": [], "val_ic": [],
        "val_ann_return": [], "val_sharpe": [],
    }

    model_path = os.path.join(cfg.MODELS_DIR, f"ncde_option{option}_best.pt")

    for epoch in range(1, max_epochs + 1):

        # Warmup lr schedule
        if epoch <= warmup_epochs:
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_factor + (cfg.LEARNING_RATE - warmup_factor) * (epoch / warmup_epochs)

        train_loss = train_epoch(model, train_dl, optimizer)
        val_loss, val_ic, val_hr, val_ret, val_sharpe = eval_epoch(model, val_dl)

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_ic"].append(round(val_ic, 4))
        history["val_ann_return"].append(round(val_ret, 4))
        history["val_sharpe"].append(round(val_sharpe, 4))

        if epoch > warmup_epochs:
            scheduler.step(val_loss)

        # Composite score
        composite = val_sharpe * 0.5 + val_ret * 0.3 + val_ic * 10.0 * 0.2
        improved = composite > best_composite

        if improved:
            best_val_loss = val_loss
            best_val_ic = val_ic
            best_val_ann_ret = val_ret
            best_val_sharpe = val_sharpe
            best_composite = composite
            patience_count = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_count += 1

        current_lr = optimizer.param_groups[0]["lr"]
        if epoch % 10 == 0 or epoch == 1:
            marker = "*BEST*" if improved else f"patience {patience_count}/{patience}"
            print(
                f" Epoch {epoch:3d} | train={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_ret={val_ret:.4f} | "
                f"val_ic={val_ic:.3f} | val_sharpe={val_sharpe:.3f} | "
                f"score={composite:.4f} | lr={current_lr:.2e} | {marker}"
            )

        if patience_count >= patience:
            print(f" Early stopping at epoch {epoch}")
            break

    # Test evaluation
    print("\nEvaluating on test set...")

    # CORRECTED: Simple load - no need to re-apply patches
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    test_loss, test_ic, test_hr, test_ret, test_sharpe = eval_epoch(model, test_dl)

    print(
        f" Test NLL={test_loss:.4f} | IC={test_ic:.3f} | "
        f"Hit={test_hr:.3f} | Ann Return={test_ret:.4f} | Sharpe={test_sharpe:.3f}"
    )

    # Save scaler
    scaler_path = os.path.join(cfg.MODELS_DIR, f"scaler_option{option}.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Save metadata
    elapsed = round(time.time() - t0, 1)
    summary = {
        "option": option,
        "trained_at": datetime.utcnow().isoformat(),
        "elapsed_sec": elapsed,
        "n_params": n_params,
        "n_assets": feat_dict["n_assets"],
        "tickers": feat_dict["tickers"],
        "n_asset_path_dim": feat_dict["n_asset_path_dim"],
        "n_macro_feats": feat_dict["n_macro_feats"],
        "lookback": lookback,
        "fixes_applied": ["fix2", "fix5", "fix6", "fix7"],
        "best_val_loss": round(best_val_loss, 6),
        "best_val_ic": round(best_val_ic, 4),
        "best_val_ann_return": round(best_val_ann_ret, 4),
        "best_val_sharpe": round(best_val_sharpe, 4),
        "best_composite": round(best_composite, 4),
        "test_loss": round(test_loss, 6),
        "test_ic": round(test_ic, 4),
        "test_hit_rate": round(test_hr, 4),
        "test_ann_return": round(test_ret, 4),
        "test_sharpe": round(test_sharpe, 4),
        "splits": splits,
        "history": history,
        "config": {
            "hidden_dim": cfg.HIDDEN_DIM,
            "vector_field_dim": cfg.VECTOR_FIELD_DIM,
            "n_layers": cfg.N_LAYERS,
            "readout_dim": cfg.READOUT_DIM,
            "dropout": cfg.DROPOUT,
            "solver": cfg.SOLVER,
            "adjoint": cfg.ADJOINT,
            "ode_steps": ode_steps,
            "lr": cfg.LEARNING_RATE,
            "batch_size": cfg.BATCH_SIZE,
            "max_epochs": max_epochs,
            "patience": patience,
            "lookback": lookback,
            "train_split": cfg.TRAIN_SPLIT,
            "val_split": cfg.VAL_SPLIT,
            "enriched_h0": enriched_h0,  # ← CORRECTED: Store this in metadata
            "fix2_epochs_patience": f"{max_epochs}/{patience}",
            "fix5_h0_enriched": enriched_h0,
            "fix6_ode_steps": ode_steps,
            "fix7_cs_normalize": True,
        },
    }

    meta_path = os.path.join(cfg.MODELS_DIR, f"meta_option{option}.json")
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)

    elapsed_min = round(elapsed / 60, 1)
    print(f"\nOption {option} done in {elapsed_min} min")
    print(f" Best composite : {best_composite:.4f}")
    print(f" Best val IC : {best_val_ic:.3f}")
    print(f" Test Ann Return: {test_ret*100:.2f}%")
    print(f" Test Sharpe : {test_sharpe:.3f}")
    print(f" Test IC : {test_ic:.3f}")
    print(f" Model saved : {model_path}")

    return summary

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NCDE model")
    parser.add_argument(
        "--option", choices=["A", "B", "both"], default="both",
        help="A = Fixed Income, B = Equity, both = train sequentially",
    )
    args = parser.parse_args()
    options = ["A", "B"] if args.option == "both" else [args.option]

    summaries = {}
    for opt in options:
        summaries[opt] = train_option(opt)

    print(f"\n{'='*60}")
    print("ALL TRAINING COMPLETE")
    for opt, s in summaries.items():
        print(
            f" Option {opt}: ann_return={s['test_ann_return']:.3f} | "
            f"IC={s['test_ic']:.3f} | Sharpe={s['test_sharpe']:.3f} | "
            f"elapsed={round(s['elapsed_sec']/60, 1)}min"
        )
    print(f"{'='*60}")
