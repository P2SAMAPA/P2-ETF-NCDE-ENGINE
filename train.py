# train.py — NCDE training pipeline
#
# Trains Option A (Fixed Income) and Option B (Equity) NCDE models.
# Output: models/ncde_option{A|B}_best.pt + meta + scaler
#
# Usage:
#   python train.py --option A
#   python train.py --option B
#   python train.py --option both

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
os.makedirs(cfg.DATA_DIR,   exist_ok=True)

DEVICE = torch.device("cpu")   # GitHub Actions free tier — CPU only


# ── Dataset helpers ────────────────────────────────────────────────────────────

def make_dataloaders(feat_dict: dict, scaler: feat.PathScaler) -> tuple:
    """
    Chronological 80/10/10 split → DataLoaders.
    Scaler fitted on train only — no data leakage.
    """
    X_a = feat_dict["X_asset"]   # (N, T, n_asset_path_dim)
    X_m = feat_dict["X_macro"]   # (N, T, n_macro_feats)
    y   = feat_dict["y"]         # (N, n_assets)
    N   = len(X_a)

    n_train = int(N * cfg.TRAIN_SPLIT)
    n_val   = int(N * cfg.VAL_SPLIT)

    # Fit scaler on train only
    X_a_tr, X_m_tr = scaler.fit_transform(X_a[:n_train],          X_m[:n_train])
    X_a_va, X_m_va = scaler.transform(    X_a[n_train:n_train+n_val], X_m[n_train:n_train+n_val])
    X_a_te, X_m_te = scaler.transform(    X_a[n_train+n_val:],    X_m[n_train+n_val:])

    def to_ds(Xa, Xm, y_):
        return TensorDataset(
            torch.tensor(Xa, dtype=torch.float32),
            torch.tensor(Xm, dtype=torch.float32),
            torch.tensor(y_,  dtype=torch.float32),
        )

    train_dl = DataLoader(to_ds(X_a_tr, X_m_tr, y[:n_train]),          batch_size=cfg.BATCH_SIZE, shuffle=False)
    val_dl   = DataLoader(to_ds(X_a_va, X_m_va, y[n_train:n_train+n_val]), batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_dl  = DataLoader(to_ds(X_a_te, X_m_te, y[n_train+n_val:]),    batch_size=cfg.BATCH_SIZE, shuffle=False)

    dates = feat_dict["dates"]
    splits = {
        "n_train":     n_train,
        "n_val":       n_val,
        "n_test":      N - n_train - n_val,
        "train_start": str(dates[0].date()),
        "train_end":   str(dates[n_train - 1].date()),
        "val_end":     str(dates[n_train + n_val - 1].date()),
        "test_end":    str(dates[-1].date()),
    }
    return train_dl, val_dl, test_dl, splits


# ── Spline builder (per batch) ─────────────────────────────────────────────────

def _build_paths(X_asset: torch.Tensor, X_macro: torch.Tensor):
    """Build torchcde cubic spline paths from raw batch tensors."""
    t = torch.arange(X_asset.shape[1], dtype=torch.float32)
    asset_coeffs = torchcde.natural_cubic_coeffs(X_asset, t)
    macro_coeffs = torchcde.natural_cubic_coeffs(X_macro, t)
    asset_path   = torchcde.CubicSpline(asset_coeffs, t)
    macro_path   = torchcde.CubicSpline(macro_coeffs, t)
    return asset_path, macro_path


# ── Training loop ──────────────────────────────────────────────────────────────

def train_epoch(model, loader_, optimizer) -> float:
    model.train()
    total_loss = 0.0
    for X_a, X_m, y_batch in loader_:
        optimizer.zero_grad()
        asset_path, macro_path = _build_paths(X_a, X_m)
        mu, sigma = model(asset_path, macro_path)
        loss = gaussian_nll_loss(mu, sigma, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader_)


def eval_epoch(model, loader_) -> tuple:
    """Returns (avg_nll, ic, hit_rate, ann_return)."""
    model.eval()
    total_loss = 0.0
    all_mu, all_y = [], []

    with torch.no_grad():
        for X_a, X_m, y_batch in loader_:
            asset_path, macro_path = _build_paths(X_a, X_m)
            mu, sigma = model(asset_path, macro_path)
            loss = gaussian_nll_loss(mu, sigma, y_batch)
            total_loss += loss.item()
            all_mu.append(mu.numpy())
            all_y.append(y_batch.numpy())

    mu_arr = np.concatenate(all_mu, axis=0)    # (N, n_assets)
    y_arr  = np.concatenate(all_y,  axis=0)

    # Top-1 pick per day (highest mu) — simple signal quality metrics
    picks     = mu_arr.argmax(axis=1)
    pick_rets = y_arr[np.arange(len(picks)), picks]

    ann_return = float(pick_rets.mean() * 252)
    sharpe     = float((pick_rets.mean() / (pick_rets.std() + 1e-8)) * np.sqrt(252))
    hit_rate   = float((pick_rets > 0).mean())

    # Information coefficient — rank correlation between mu and actual returns
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
    print(f"\n{'='*60}")
    print(f"NCDE Training — Option {'A (Fixed Income)' if option == 'A' else 'B (Equity)'}")
    print(f"{'='*60}")

    # Load data
    print("\n[1/5] Loading data...")
    master = loader.load_master()
    data   = loader.get_option_data(option, master)

    # Feature engineering
    print("\n[2/5] Building features...")
    feat_dict = feat.prepare_features(data, lookback=cfg.LOOKBACK)
    scaler    = feat.PathScaler()

    # Dataloaders
    print("\n[3/5] Preparing dataloaders...")
    train_dl, val_dl, test_dl, splits = make_dataloaders(feat_dict, scaler)
    print(f"  Train: {splits['n_train']} | Val: {splits['n_val']} | Test: {splits['n_test']}")
    print(f"  Train: {splits['train_start']} -> {splits['train_end']}")
    print(f"  Val  : {splits['train_end']}  -> {splits['val_end']}")
    print(f"  Test : {splits['val_end']}    -> {splits['test_end']}")

    # Model
    print("\n[4/5] Building model...")
    model = NCDEForecaster(
        n_asset_path_dim = feat_dict["n_asset_path_dim"],
        n_macro_feats    = feat_dict["n_macro_feats"],
        n_assets         = feat_dict["n_assets"],
        hidden_dim       = cfg.HIDDEN_DIM,
        n_layers         = cfg.N_LAYERS,
        readout_dim      = cfg.READOUT_DIM,
        dropout          = cfg.DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training loop
    print(f"\n[5/5] Training ({cfg.MAX_EPOCHS} epochs, patience={cfg.PATIENCE})...")
    best_val_loss      = float("inf")
    best_val_ic        = -float("inf")
    best_val_ann_ret   = -float("inf")
    best_val_sharpe    = -float("inf")
    patience_count     = 0
    history = {"train_loss": [], "val_loss": [], "val_ic": [], "val_ann_return": [], "val_sharpe": []}

    model_path = os.path.join(cfg.MODELS_DIR, f"ncde_option{option}_best.pt")

    for epoch in range(1, cfg.MAX_EPOCHS + 1):
        train_loss                                   = train_epoch(model, train_dl, optimizer)
        val_loss, val_ic, val_hr, val_ret, val_sharpe = eval_epoch(model, val_dl)

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_ic"].append(round(val_ic, 4))
        history["val_ann_return"].append(round(val_ret, 4))
        history["val_sharpe"].append(round(val_sharpe, 4))

        scheduler.step(val_loss)

        improved = val_ret > best_val_ann_ret
        if improved:
            best_val_loss    = val_loss
            best_val_ic      = val_ic
            best_val_ann_ret = val_ret
            best_val_sharpe  = val_sharpe
            patience_count   = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_count += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | train={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | val_ret={val_ret:.4f} | "
                  f"val_ic={val_ic:.3f} | val_sharpe={val_sharpe:.3f} | "
                  f"{'*BEST*' if improved else f'patience {patience_count}/{cfg.PATIENCE}'}")

        if patience_count >= cfg.PATIENCE:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Test evaluation
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    test_loss, test_ic, test_hr, test_ret, test_sharpe = eval_epoch(model, test_dl)
    print(f"  Test NLL={test_loss:.4f} | IC={test_ic:.3f} | "
          f"Hit={test_hr:.3f} | Ann Return={test_ret:.4f} | Sharpe={test_sharpe:.3f}")

    # Save scaler
    scaler_path = os.path.join(cfg.MODELS_DIR, f"scaler_option{option}.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Save metadata
    elapsed = round(time.time() - t0, 1)
    summary = {
        "option":              option,
        "trained_at":          datetime.utcnow().isoformat(),
        "elapsed_sec":         elapsed,
        "n_params":            n_params,
        "n_assets":            feat_dict["n_assets"],
        "tickers":             feat_dict["tickers"],
        "n_asset_path_dim":    feat_dict["n_asset_path_dim"],
        "n_macro_feats":       feat_dict["n_macro_feats"],
        "lookback":            cfg.LOOKBACK,
        "best_val_loss":       round(best_val_loss,    6),
        "best_val_ic":         round(best_val_ic,      4),
        "best_val_ann_return": round(best_val_ann_ret, 4),
        "best_val_sharpe":     round(best_val_sharpe,  4),
        "test_loss":           round(test_loss,  6),
        "test_ic":             round(test_ic,    4),
        "test_hit_rate":       round(test_hr,    4),
        "test_ann_return":     round(test_ret,   4),
        "test_sharpe":         round(test_sharpe, 4),
        "splits":              splits,
        "history":             history,
        "config": {
            "hidden_dim":       cfg.HIDDEN_DIM,
            "vector_field_dim": cfg.VECTOR_FIELD_DIM,
            "n_layers":         cfg.N_LAYERS,
            "readout_dim":      cfg.READOUT_DIM,
            "dropout":          cfg.DROPOUT,
            "solver":           cfg.SOLVER,
            "adjoint":          cfg.ADJOINT,
            "lr":               cfg.LEARNING_RATE,
            "batch_size":       cfg.BATCH_SIZE,
            "max_epochs":       cfg.MAX_EPOCHS,
            "patience":         cfg.PATIENCE,
            "lookback":         cfg.LOOKBACK,
            "train_split":      cfg.TRAIN_SPLIT,
            "val_split":        cfg.VAL_SPLIT,
        },
    }

    meta_path = os.path.join(cfg.MODELS_DIR, f"meta_option{option}.json")
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOption {option} done in {elapsed}s")
    print(f"  Best val Ann Return : {best_val_ann_ret*100:.2f}%")
    print(f"  Best val IC         : {best_val_ic:.3f}")
    print(f"  Test Ann Return     : {test_ret*100:.2f}%")
    print(f"  Test Sharpe         : {test_sharpe:.3f}")
    print(f"  Model saved         : {model_path}")

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
        print(f"  Option {opt}: ann_return={s['test_ann_return']:.3f} | "
              f"IC={s['test_ic']:.3f} | Sharpe={s['test_sharpe']:.3f}")
    print(f"{'='*60}")
