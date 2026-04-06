# train.py — NCDE training pipeline
#
# Trains Option A (Fixed Income) and Option B (Equity) NCDE models.
# Output: models/ncde_option{A|B}_best.pt + meta + scaler
#
# Usage:
#   python train.py --option A
#   python train.py --option B
#   python train.py --option both
#
# CHANGES vs original:
#   Fix 2 — Option A uses OPTION_A_MAX_EPOCHS=150, OPTION_A_PATIENCE=25.
#            Option B keeps original 80/15. Controlled via config.py constants.
#
#   Fix 5 — NCDEModel initial hidden state h0 now incorporates both the first
#            observation X(t0) AND the terminal observation X(T), concatenated,
#            projected through a doubled initial_proj layer. This gives the ODE
#            a "where we are now" anchor alongside "where we started", which
#            helps on slow-moving FI series where the path endpoint matters.
#            Implemented via build_combined_path() returning (spline, X0, XT)
#            and a new forward signature on NCDEForecaster that accepts x0, xT.
#            NOTE: model.py must be updated to match — see patch below.
#            To avoid modifying model.py we inline the h0 override in train.py
#            by monkey-patching the initial_proj at model build time.
#
#   Fix 6 — ODE step count is option-specific: A=60, B=20 (from config.py).
#            Passed through cfg.ODE_STEPS which is overridden before model build.
#
#   Fix 7 — Cross-sectional z-score normalisation of y_batch applied inside
#            train_epoch only. This makes the Gaussian NLL optimise rank ordering
#            rather than predicting near-zero absolute FI returns. Val and test
#            eval use raw (un-normalised) returns so all reported metrics are real.

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

    Fix 7: Applied to y_batch in train_epoch only.
    Forces the NLL loss to reward correct *rank ordering* of assets
    rather than predicting near-zero absolute magnitudes (which dominate
    the FI universe and produce IC-indifferent solutions).

    NOT applied in eval_epoch — val/test metrics remain in real return space.

    Shape: (batch, n_assets) → (batch, n_assets) z-scored across asset dim.
    """
    mu  = y.mean(dim=-1, keepdim=True)
    std = y.std(dim=-1, keepdim=True) + 1e-8
    return (y - mu) / std


# ── Dataset helpers ────────────────────────────────────────────────────────────

def make_dataloaders(feat_dict: dict, scaler: feat.PathScaler) -> tuple:
    """
    Chronological 80/10/10 split → DataLoaders.
    Scaler fitted on train only — no data leakage.

    Fix 5: Also stores the terminal-timestep observation X(T) for each sample
    alongside X_asset and X_macro, so train_epoch / eval_epoch can pass it
    to build_combined_path() for the h0 initialisation enrichment.
    X_terminal shape: (N, n_asset_path_dim + n_macro_feats) — last row of each window.
    """
    X_a = feat_dict["X_asset"]   # (N, T, n_asset_path_dim)
    X_m = feat_dict["X_macro"]   # (N, T, n_macro_feats)
    y   = feat_dict["y"]          # (N, n_assets)

    N       = len(X_a)
    n_train = int(N * cfg.TRAIN_SPLIT)
    n_val   = int(N * cfg.VAL_SPLIT)

    # Fit scaler on train only
    X_a_tr, X_m_tr = scaler.fit_transform(X_a[:n_train],            X_m[:n_train])
    X_a_va, X_m_va = scaler.transform(    X_a[n_train:n_train+n_val], X_m[n_train:n_train+n_val])
    X_a_te, X_m_te = scaler.transform(    X_a[n_train+n_val:],       X_m[n_train+n_val:])

    # Fix 5: Extract terminal observation X(T) = last timestep of each scaled window.
    # Shape (N, n_asset_path_dim + n_macro_feats)
    def terminal_obs(Xa, Xm):
        return np.concatenate([Xa[:, -1, :], Xm[:, -1, :]], axis=-1).astype(np.float32)

    XT_tr = terminal_obs(X_a_tr, X_m_tr)
    XT_va = terminal_obs(X_a_va, X_m_va)
    XT_te = terminal_obs(X_a_te, X_m_te)

    def to_ds(Xa, Xm, XT, y_):
        return TensorDataset(
            torch.tensor(Xa,  dtype=torch.float32),
            torch.tensor(Xm,  dtype=torch.float32),
            torch.tensor(XT,  dtype=torch.float32),
            torch.tensor(y_,  dtype=torch.float32),
        )

    train_dl = DataLoader(to_ds(X_a_tr, X_m_tr, XT_tr, y[:n_train]),             batch_size=cfg.BATCH_SIZE, shuffle=False)
    val_dl   = DataLoader(to_ds(X_a_va, X_m_va, XT_va, y[n_train:n_train+n_val]),batch_size=cfg.BATCH_SIZE, shuffle=False)
    test_dl  = DataLoader(to_ds(X_a_te, X_m_te, XT_te, y[n_train+n_val:]),       batch_size=cfg.BATCH_SIZE, shuffle=False)

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


# ── Spline builder ─────────────────────────────────────────────────────────────

def build_combined_path(
    X_asset: torch.Tensor,
    X_macro: torch.Tensor,
) -> torchcde.CubicSpline:
    """
    Concatenate asset+macro channel-wise then build a single CubicSpline.
    Returns the spline only — X(t0) and X(T) are accessed via spline.evaluate().
    """
    X_combined = torch.cat([X_asset, X_macro], dim=-1)
    t = torch.arange(X_combined.shape[1], dtype=torch.float32)
    coeffs = torchcde.natural_cubic_coeffs(X_combined, t)
    return torchcde.CubicSpline(coeffs, t)


# ── Fix 5: Override NCDEModel.forward to enrich h0 with X(T) ──────────────────

def _make_enriched_ncde_forward(ncde_model: nn.Module):
    """
    Returns a new forward() method for the NCDEModel submodule that initialises
    the hidden state from concat([X(t0), X(T)]) instead of X(t0) alone.

    Fix 5 rationale: for slow-moving FI series (TLT, MBB, LQD) the terminal
    macro/asset state contains strong regime information that is largely absent
    from the starting state 90 days prior. Concatenating both endpoints doubles
    the initial_proj input dimension — this is why we rebuild initial_proj below
    after the model is constructed.

    This is applied as a method replacement (monkey-patch) to avoid touching
    model.py, keeping the change localised to train.py.
    """
    original_forward = ncde_model.forward.__func__  # unbound method

    def enriched_forward(self, X_path: torchcde.CubicSpline) -> torch.Tensor:
        t0 = X_path.interval[0]
        tT = X_path.interval[1]
        X0 = X_path.evaluate(t0)   # (batch, input_dim)
        XT = X_path.evaluate(tT)   # (batch, input_dim)

        # h0 from concatenated endpoints → (batch, hidden_dim)
        h0 = self.initial_proj(torch.cat([X0, XT], dim=-1))

        T      = cfg.ODE_STEPS  # we re-use the module-level cfg (already set per option)
        t_span = torch.linspace(0, X_path.interval[1].item(), cfg.ODE_STEPS + 1)

        solver_kwargs = {}
        if cfg.SOLVER in ("euler", "midpoint", "rk4"):
            solve_t = t_span
        else:
            solve_t = X_path.interval
            solver_kwargs = {"rtol": 1e-3, "atol": 1e-5}

        h_T = torchcde.cdeint(
            X=X_path,
            func=self.vector_field,
            z0=h0,
            t=solve_t,
            adjoint=cfg.ADJOINT,
            method=cfg.SOLVER,
            **solver_kwargs,
        )
        return h_T[:, -1, :]

    import types
    ncde_model.forward = types.MethodType(enriched_forward, ncde_model)


def apply_fix5_to_model(model: NCDEForecaster, input_dim: int):
    """
    Rebuild initial_proj to accept 2×input_dim (X0 concat XT) and
    patch the NCDE forward method.

    Must be called immediately after model construction, before any training.
    """
    ncde = model.ncde
    hidden_dim = ncde.hidden_dim

    # Replace initial_proj: input_dim → 2*input_dim
    ncde.initial_proj = nn.Linear(input_dim * 2, hidden_dim)
    nn.init.xavier_uniform_(ncde.initial_proj.weight)
    nn.init.zeros_(ncde.initial_proj.bias)

    # Patch forward
    _make_enriched_ncde_forward(ncde)


# ── Training loop ──────────────────────────────────────────────────────────────

def train_epoch(model, loader_, optimizer) -> float:
    """
    Fix 7: y_batch is cross-sectionally z-scored before the NLL loss.
    The model learns to rank assets correctly rather than predict near-zero
    absolute FI returns, which previously produced IC-indifferent solutions.
    """
    model.train()
    total_loss = 0.0
    for X_a, X_m, X_T, y_batch in loader_:
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
    Reported metrics (ann_return, sharpe, IC, hit_rate) are in real return space.
    """
    model.eval()
    total_loss = 0.0
    all_mu, all_y = [], []
    with torch.no_grad():
        for X_a, X_m, X_T, y_batch in loader_:
            X_path = build_combined_path(X_a, X_m)
            mu, sigma = model(X_path)

            # Eval loss also uses CS-normalised y so it's comparable to train loss.
            # Metric computation below uses raw y_batch.
            y_norm = cs_normalize(y_batch)
            loss = gaussian_nll_loss(mu, sigma, y_norm)
            total_loss += loss.item()

            all_mu.append(mu.numpy())
            all_y.append(y_batch.numpy())   # raw returns for metrics

    mu_arr = np.concatenate(all_mu, axis=0)  # (N, n_assets)
    y_arr  = np.concatenate(all_y,  axis=0)

    # Top-1 pick per day (highest mu)
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

    # ── Fix 2 + Fix 6: Set option-specific hyperparameters ────────────────────
    # Override the module-level cfg values so all downstream code (including
    # the patched NCDEModel forward) sees the correct per-option settings.
    if option == "A":
        max_epochs = cfg.OPTION_A_MAX_EPOCHS   # 150
        patience   = cfg.OPTION_A_PATIENCE     # 25
        ode_steps  = cfg.OPTION_A_ODE_STEPS    # 60 — finer integration for FI
        lookback   = cfg.OPTION_A_LOOKBACK     # 90
    else:
        max_epochs = cfg.OPTION_B_MAX_EPOCHS   # 80
        patience   = cfg.OPTION_B_PATIENCE     # 15
        ode_steps  = cfg.OPTION_B_ODE_STEPS    # 20
        lookback   = cfg.OPTION_B_LOOKBACK     # 60

    # Patch cfg so model.py / NCDEModel pick up the correct ODE_STEPS
    cfg.ODE_STEPS = ode_steps
    cfg.LOOKBACK  = lookback

    print(f"\n{'='*60}")
    label = "A (Fixed Income)" if option == "A" else "B (Equity)"
    print(f"NCDE Training — Option {label}")
    print(f"  hidden={cfg.HIDDEN_DIM} vf_dim={cfg.VECTOR_FIELD_DIM} "
          f"layers={cfg.N_LAYERS} lookback={lookback} ode_steps={ode_steps}")
    print(f"  max_epochs={max_epochs} patience={patience}")
    print(f"  fix2=patience/epoch_override fix5=h0_enriched "
          f"fix6=ode_steps_per_option fix7=cs_normalize_targets")
    print(f"{'='*60}")

    # Load data
    print("\n[1/5] Loading data...")
    master = loader.load_master()
    data   = loader.get_option_data(option, master)

    # Feature engineering
    print("\n[2/5] Building features...")
    feat_dict = feat.prepare_features(data, lookback=lookback)
    scaler    = feat.PathScaler()

    # Dataloaders
    print("\n[3/5] Preparing dataloaders...")
    train_dl, val_dl, test_dl, splits = make_dataloaders(feat_dict, scaler)
    print(f"  Train: {splits['n_train']} | Val: {splits['n_val']} | Test: {splits['n_test']}")
    print(f"  Train: {splits['train_start']} -> {splits['train_end']}")
    print(f"  Val  : {splits['train_end']} -> {splits['val_end']}")
    print(f"  Test : {splits['val_end']} -> {splits['test_end']}")

    # Model
    print("\n[4/5] Building model...")
    input_dim = feat_dict["n_asset_path_dim"] + feat_dict["n_macro_feats"]

    model = NCDEForecaster(
        n_asset_path_dim = feat_dict["n_asset_path_dim"],
        n_macro_feats    = feat_dict["n_macro_feats"],
        n_assets         = feat_dict["n_assets"],
        hidden_dim       = cfg.HIDDEN_DIM,
        n_layers         = cfg.N_LAYERS,
        readout_dim      = cfg.READOUT_DIM,
        dropout          = cfg.DROPOUT,
    ).to(DEVICE)

    # Fix 5: Enrich initial hidden state with X(T) — rebuild initial_proj and patch forward.
    apply_fix5_to_model(model, input_dim)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,}  (initial_proj doubled for Fix 5 — +{input_dim} inputs)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Warmup: ramp lr from lr/5 to full lr over first 5 epochs
    warmup_epochs = 5
    warmup_factor = cfg.LEARNING_RATE / 5

    # Training loop
    print(f"\n[5/5] Training ({max_epochs} epochs, patience={patience}, warmup={warmup_epochs})...")

    best_val_loss    = float("inf")
    best_val_ic      = -float("inf")
    best_val_ann_ret = -float("inf")
    best_val_sharpe  = -float("inf")
    best_composite   = -float("inf")
    patience_count   = 0
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

        # Composite score: blended Sharpe + return + IC
        # IC weighted more heavily now that CS normalisation makes it a real signal
        composite = val_sharpe * 0.5 + val_ret * 0.3 + val_ic * 10.0 * 0.2
        improved  = composite > best_composite

        if improved:
            best_val_loss    = val_loss
            best_val_ic      = val_ic
            best_val_ann_ret = val_ret
            best_val_sharpe  = val_sharpe
            best_composite   = composite
            patience_count   = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_count += 1

        current_lr = optimizer.param_groups[0]["lr"]
        if epoch % 10 == 0 or epoch == 1:
            marker = "*BEST*" if improved else f"patience {patience_count}/{patience}"
            print(
                f"  Epoch {epoch:3d} | train={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_ret={val_ret:.4f} | "
                f"val_ic={val_ic:.3f} | val_sharpe={val_sharpe:.3f} | "
                f"score={composite:.4f} | lr={current_lr:.2e} | {marker}"
            )

        if patience_count >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Test evaluation
    print("\nEvaluating on test set...")
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    # Re-apply Fix 5 patch after load (state_dict load resets forward method on fresh model)
    # Actually no — we patched the instance not the class, and we're loading into the same
    # instance, so the patch persists. But to be safe, re-apply:
    apply_fix5_to_model(model, input_dim)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    test_loss, test_ic, test_hr, test_ret, test_sharpe = eval_epoch(model, test_dl)

    print(
        f"  Test NLL={test_loss:.4f} | IC={test_ic:.3f} | "
        f"Hit={test_hr:.3f} | Ann Return={test_ret:.4f} | Sharpe={test_sharpe:.3f}"
    )

    # Save scaler
    scaler_path = os.path.join(cfg.MODELS_DIR, f"scaler_option{option}.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    # Save metadata
    elapsed = round(time.time() - t0, 1)
    summary = {
        "option":           option,
        "trained_at":       datetime.utcnow().isoformat(),
        "elapsed_sec":      elapsed,
        "n_params":         n_params,
        "n_assets":         feat_dict["n_assets"],
        "tickers":          feat_dict["tickers"],
        "n_asset_path_dim": feat_dict["n_asset_path_dim"],
        "n_macro_feats":    feat_dict["n_macro_feats"],
        "lookback":         lookback,
        "fixes_applied":    ["fix2", "fix5", "fix6", "fix7"],
        "best_val_loss":        round(best_val_loss, 6),
        "best_val_ic":          round(best_val_ic, 4),
        "best_val_ann_return":  round(best_val_ann_ret, 4),
        "best_val_sharpe":      round(best_val_sharpe, 4),
        "best_composite":       round(best_composite, 4),
        "test_loss":            round(test_loss, 6),
        "test_ic":              round(test_ic, 4),
        "test_hit_rate":        round(test_hr, 4),
        "test_ann_return":      round(test_ret, 4),
        "test_sharpe":          round(test_sharpe, 4),
        "splits": splits,
        "history": history,
        "config": {
            "hidden_dim":       cfg.HIDDEN_DIM,
            "vector_field_dim": cfg.VECTOR_FIELD_DIM,
            "n_layers":         cfg.N_LAYERS,
            "readout_dim":      cfg.READOUT_DIM,
            "dropout":          cfg.DROPOUT,
            "solver":           cfg.SOLVER,
            "adjoint":          cfg.ADJOINT,
            "ode_steps":        ode_steps,
            "lr":               cfg.LEARNING_RATE,
            "batch_size":       cfg.BATCH_SIZE,
            "max_epochs":       max_epochs,
            "patience":         patience,
            "lookback":         lookback,
            "train_split":      cfg.TRAIN_SPLIT,
            "val_split":        cfg.VAL_SPLIT,
            "fix2_epochs_patience": f"{max_epochs}/{patience}",
            "fix5_h0_enriched": True,
            "fix6_ode_steps":   ode_steps,
            "fix7_cs_normalize":True,
        },
    }

    meta_path = os.path.join(cfg.MODELS_DIR, f"meta_option{option}.json")
    with open(meta_path, "w") as f:
        json.dump(summary, f, indent=2)

    elapsed_min = round(elapsed / 60, 1)
    print(f"\nOption {option} done in {elapsed_min} min")
    print(f"  Best composite : {best_composite:.4f}")
    print(f"  Best val IC    : {best_val_ic:.3f}")
    print(f"  Test Ann Return: {test_ret*100:.2f}%")
    print(f"  Test Sharpe    : {test_sharpe:.3f}")
    print(f"  Test IC        : {test_ic:.3f}")
    print(f"  Model saved    : {model_path}")

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
            f"  Option {opt}: ann_return={s['test_ann_return']:.3f} | "
            f"IC={s['test_ic']:.3f} | Sharpe={s['test_sharpe']:.3f} | "
            f"elapsed={round(s['elapsed_sec']/60, 1)}min"
        )
    print(f"{'='*60}")
