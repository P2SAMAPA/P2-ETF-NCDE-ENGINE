# train.py — NCDE training pipeline
#
# Trains Option A (Fixed Income) and Option B (Equity) NCDE models.
# Each option now uses its own architecture + training config from config.py.
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

DEVICE = torch.device("cpu")


# ── Dataset helpers ────────────────────────────────────────────────────────────

def make_dataloaders(feat_dict: dict, scaler: feat.PathScaler, ocfg: dict) -> tuple:
    """
    Chronological 80/10/10 split → DataLoaders.
    Scaler fitted on train only. Uses per-option batch_size.
    """
    X_a = feat_dict["X_asset"]
    X_m = feat_dict["X_macro"]
    y   = feat_dict["y"]

    N       = len(X_a)
    n_train = int(N * cfg.TRAIN_SPLIT)
    n_val   = int(N * cfg.VAL_SPLIT)

    X_a_tr, X_m_tr = scaler.fit_transform(X_a[:n_train],              X_m[:n_train])
    X_a_va, X_m_va = scaler.transform(    X_a[n_train:n_train+n_val], X_m[n_train:n_train+n_val])
    X_a_te, X_m_te = scaler.transform(    X_a[n_train+n_val:],        X_m[n_train+n_val:])

    def to_ds(Xa, Xm, y_):
        return TensorDataset(
            torch.tensor(Xa, dtype=torch.float32),
            torch.tensor(Xm, dtype=torch.float32),
            torch.tensor(y_, dtype=torch.float32),
        )

    bs = ocfg["batch_size"]
    train_dl = DataLoader(to_ds(X_a_tr, X_m_tr, y[:n_train]),              batch_size=bs, shuffle=False)
    val_dl   = DataLoader(to_ds(X_a_va, X_m_va, y[n_train:n_train+n_val]), batch_size=bs, shuffle=False)
    test_dl  = DataLoader(to_ds(X_a_te, X_m_te, y[n_train+n_val:]),        batch_size=bs, shuffle=False)

    dates  = feat_dict["dates"]
    splits = {
        "n_train":    n_train,
        "n_val":      n_val,
        "n_test":     N - n_train - n_val,
        "train_start": str(dates[0].date()),
        "train_end":   str(dates[n_train - 1].date()),
        "val_end":     str(dates[n_train + n_val - 1].date()),
        "test_end":    str(dates[-1].date()),
    }
    return train_dl, val_dl, test_dl, splits


# ── Spline builder ─────────────────────────────────────────────────────────────

def build_combined_path(X_asset: torch.Tensor, X_macro: torch.Tensor) -> torchcde.CubicSpline:
    X_combined = torch.cat([X_asset, X_macro], dim=-1)
    t          = torch.arange(X_combined.shape[1], dtype=torch.float32)
    coeffs     = torchcde.natural_cubic_coeffs(X_combined, t)
    return torchcde.CubicSpline(coeffs, t)


# ── Training loop ──────────────────────────────────────────────────────────────

def train_epoch(model, loader_, optimizer, grad_clip: float) -> float:
    model.train()
    total_loss = 0.0
    for X_a, X_m, y_batch in loader_:
        optimizer.zero_grad()
        X_path    = build_combined_path(X_a, X_m)
        mu, sigma = model(X_path)
        loss      = gaussian_nll_loss(mu, sigma, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader_)


def eval_epoch(model, loader_) -> tuple:
    model.eval()
    total_loss = 0.0
    all_mu, all_y = [], []

    with torch.no_grad():
        for X_a, X_m, y_batch in loader_:
            X_path    = build_combined_path(X_a, X_m)
            mu, sigma = model(X_path)
            loss      = gaussian_nll_loss(mu, sigma, y_batch)
            total_loss += loss.item()
            all_mu.append(mu.numpy())
            all_y.append(y_batch.numpy())

    mu_arr = np.concatenate(all_mu, axis=0)
    y_arr  = np.concatenate(all_y,  axis=0)

    picks     = mu_arr.argmax(axis=1)
    pick_rets = y_arr[np.arange(len(picks)), picks]

    ann_return = float(pick_rets.mean() * 252)
    sharpe     = float((pick_rets.mean() / (pick_rets.std() + 1e-8)) * np.sqrt(252))
    hit_rate   = float((pick_rets > 0).mean())

    from scipy.stats import spearmanr
    ic_list = []
    for i in range(len(mu_arr)):
        r, _ = spearmanr(mu_arr[i], y_arr[i])
        if not np.isnan(r):
            ic_list.append(r)
    ic = float(np.mean(ic_list)) if ic_list else 0.0

    return total_loss / len(loader_), ic, hit_rate, ann_return, sharpe


def composite_score(nll: float, ic: float, ann_ret: float, sharpe: float) -> float:
    """
    Composite score for early stopping. Blends NLL, IC, and Sharpe to avoid
    firing early on the noisy val_ann_return signal.

    Weights deliberately favour IC (directional signal quality) because:
    - NLL can improve by just shrinking sigma (not improving direction)
    - IC is the most stable indicator of genuine ranking ability
    - Sharpe is noisier than IC on a 450-sample val set but still useful
    """
    return 0.35 * (-nll) + 0.40 * (ic * 5.0) + 0.25 * (sharpe * 0.5)


def get_warmup_lr(epoch: int, base_lr: float, warmup_epochs: int) -> float:
    if epoch <= warmup_epochs:
        return base_lr * (epoch / warmup_epochs)
    return base_lr


# ── Main training function ─────────────────────────────────────────────────────

def train_option(option: str) -> dict:
    t0    = time.time()
    ocfg  = cfg.OPTION_CONFIGS[option]

    print(f"\n{'='*60}")
    print(f"NCDE Training — Option {'A (Fixed Income)' if option == 'A' else 'B (Equity)'}")
    print(f"  hidden={ocfg['hidden_dim']} vf_dim={ocfg['vector_field_dim']} "
          f"layers={ocfg['n_layers']} lookback={ocfg['lookback']} "
          f"ode_steps={ocfg['ode_steps']}")
    print(f"{'='*60}")

    print("\n[1/5] Loading data...")
    master = loader.load_master()
    data   = loader.get_option_data(option, master)

    print("\n[2/5] Building features...")
    feat_dict = feat.prepare_features(data, lookback=ocfg["lookback"])
    scaler    = feat.PathScaler()

    print("\n[3/5] Preparing dataloaders...")
    train_dl, val_dl, test_dl, splits = make_dataloaders(feat_dict, scaler, ocfg)
    print(f"  Train: {splits['n_train']} | Val: {splits['n_val']} | Test: {splits['n_test']}")
    print(f"  Train: {splits['train_start']} -> {splits['train_end']}")
    print(f"  Val  : {splits['train_end']} -> {splits['val_end']}")
    print(f"  Test : {splits['val_end']} -> {splits['test_end']}")

    print("\n[4/5] Building model...")
    model = NCDEForecaster(
        n_asset_path_dim = feat_dict["n_asset_path_dim"],
        n_macro_feats    = feat_dict["n_macro_feats"],
        n_assets         = feat_dict["n_assets"],
        hidden_dim       = ocfg["hidden_dim"],
        vector_field_dim = ocfg["vector_field_dim"],
        n_layers         = ocfg["n_layers"],
        readout_dim      = ocfg["readout_dim"],
        dropout          = ocfg["dropout"],
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=ocfg["learning_rate"],
        weight_decay=ocfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=ocfg["lr_scheduler_patience"],
        min_lr=5e-7,
    )

    warmup_epochs = ocfg["warmup_epochs"]
    max_epochs    = ocfg["max_epochs"]
    patience_cfg  = ocfg["patience"]
    grad_clip     = ocfg["grad_clip"]
    base_lr       = ocfg["learning_rate"]

    print(f"\n[5/5] Training ({max_epochs} epochs, patience={patience_cfg}, warmup={warmup_epochs})...")

    best_score       = -float("inf")
    best_val_loss    = float("inf")
    best_val_ic      = -float("inf")
    best_val_ann_ret = -float("inf")
    best_val_sharpe  = -float("inf")
    patience_count   = 0

    history = {
        "train_loss": [], "val_loss": [], "val_ic": [],
        "val_ann_return": [], "val_sharpe": [], "composite_score": [],
    }

    model_path = os.path.join(cfg.MODELS_DIR, f"ncde_option{option}_best.pt")

    for epoch in range(1, max_epochs + 1):

        # LR warmup
        current_lr = get_warmup_lr(epoch, base_lr, warmup_epochs)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        train_loss                              = train_epoch(model, train_dl, optimizer, grad_clip)
        val_loss, val_ic, val_hr, val_ret, val_sharpe = eval_epoch(model, val_dl)

        score = composite_score(val_loss, val_ic, val_ret, val_sharpe)

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_ic"].append(round(val_ic, 4))
        history["val_ann_return"].append(round(val_ret, 4))
        history["val_sharpe"].append(round(val_sharpe, 4))
        history["composite_score"].append(round(score, 6))

        if epoch > warmup_epochs:
            scheduler.step(val_loss)

        improved = score > best_score
        if improved:
            best_score       = score
            best_val_loss    = val_loss
            best_val_ic      = val_ic
            best_val_ann_ret = val_ret
            best_val_sharpe  = val_sharpe
            patience_count   = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_count += 1

        if epoch % 10 == 0 or epoch == 1:
            actual_lr = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:3d} | train={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_ret={val_ret:.4f} | "
                f"val_ic={val_ic:.3f} | val_sharpe={val_sharpe:.3f} | "
                f"score={score:.4f} | lr={actual_lr:.2e} | "
                f"{'*BEST*' if improved else f'patience {patience_count}/{patience_cfg}'}"
            )

        if patience_count >= patience_cfg:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Test set evaluation
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    test_loss, test_ic, test_hr, test_ret, test_sharpe = eval_epoch(model, test_dl)
    print(
        f"  Test NLL={test_loss:.4f} | IC={test_ic:.3f} | "
        f"Hit={test_hr:.3f} | Ann Return={test_ret:.4f} | Sharpe={test_sharpe:.3f}"
    )

    scaler_path = os.path.join(cfg.MODELS_DIR, f"scaler_option{option}.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

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
        "lookback":            ocfg["lookback"],
        "best_val_loss":       round(best_val_loss, 6),
        "best_val_ic":         round(best_val_ic, 4),
        "best_val_ann_return": round(best_val_ann_ret, 4),
        "best_val_sharpe":     round(best_val_sharpe, 4),
        "best_composite_score": round(best_score, 6),
        "test_loss":           round(test_loss, 6),
        "test_ic":             round(test_ic, 4),
        "test_hit_rate":       round(test_hr, 4),
        "test_ann_return":     round(test_ret, 4),
        "test_sharpe":         round(test_sharpe, 4),
        "splits":              splits,
        "history":             history,
        "config":              ocfg,
    }

    meta_path = os.path.join(cfg.MODELS_DIR, f"meta_option{option}.json")
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)

    elapsed_min = elapsed / 60
    print(f"\nOption {option} done in {elapsed_min:.1f} min")
    print(f"  Best composite : {best_score:.4f}")
    print(f"  Best val IC    : {best_val_ic:.3f}")
    print(f"  Test Ann Return: {test_ret*100:.2f}%")
    print(f"  Test Sharpe    : {test_sharpe:.3f}")
    print(f"  Test IC        : {test_ic:.3f}")
    return summary


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--option", choices=["A", "B", "both"], default="both",
        help="A = Fixed Income, B = Equity, both = train sequentially",
    )
    args    = parser.parse_args()
    options = ["A", "B"] if args.option == "both" else [args.option]

    summaries = {}
    for opt in options:
        summaries[opt] = train_option(opt)

    print(f"\n{'='*60}")
    print("ALL TRAINING COMPLETE")
    for opt, s in summaries.items():
        print(
            f"  Option {opt}: ann_return={s['test_ann_return']:.3f} | "
            f"IC={s['test_ic']:.3f} | Sharpe={s['test_sharpe']:.3f} | "
            f"elapsed={s['elapsed_sec']/60:.1f}min"
        )
    print(f"{'='*60}")
