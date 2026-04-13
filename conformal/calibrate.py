# conformal/calibrate.py — Conformal calibration for NCDE forecasts
#
# THEORY (split conformal prediction, Angelopoulos & Bates 2022):
# ─────────────────────────────────────────────────────────────────
# We use the NCDE val set as the *calibration set* (never used to train weights).
# For each (mu_i, sigma_i, y_i) in the cal set the nonconformity score is:
#
#   s_i  =  |y_i - mu_i| / sigma_i        (normalised absolute residual)
#
# At coverage level alpha (e.g. 0.90), the conformal quantile is:
#
#   q̂  =  ceil( (n+1)(1-alpha) ) / n  -th  empirical quantile of {s_i}
#
# At test time the prediction interval for ETF j is:
#
#   [mu_j - q̂·sigma_j ,  mu_j + q̂·sigma_j]
#
# This interval has *marginal* coverage ≥ 1-alpha over any exchangeable
# draw from the same distribution — a finite-sample guarantee.
#
# We compute q̂ for three alpha levels: 0.90, 0.80, 0.70.
# Results saved to  models/conformal_option{A|B}.json
#
# Usage:
#   python -m conformal.calibrate --option both
#
# Requirements: existing trained model + scaler in models/
# Does NOT retrain — reads the same .pt weights predict.py uses.

import argparse
import json
import os
import sys
import pickle
from datetime import datetime

import numpy as np
import torch
import torchcde

# ── Make sure repo root is on path when run as module ─────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
import loader
import features as feat
from predict import load_model, build_combined_path

DEVICE = torch.device("cpu")

ALPHA_LEVELS = [0.90, 0.80, 0.70]   # coverage targets


# ── Collect val-set residuals ─────────────────────────────────────────────────

def collect_calibration_scores(option: str) -> dict:
    """
    Run the frozen NCDE over the *validation slice* (same 80/10/10 split
    as training) and compute per-ETF normalised residuals.

    Returns
    -------
    {
      "tickers": [...],
      "scores":  list of shape (n_cal_samples, n_assets)   # s_i per ETF
    }
    """
    print(f"\n[calibrate] Collecting calibration scores for Option {option}...")

    # ── Load frozen model ────────────────────────────────────────────────────
    model, meta, scaler = load_model(option)
    model.eval()

    tickers  = meta["tickers"]
    lookback = meta["config"]["lookback"]
    n_train  = meta["splits"]["n_train"]
    n_val    = meta["splits"]["n_val"]

    # ── Rebuild full feature matrix (same as train.py) ───────────────────────
    print("[calibrate]   Loading master dataset...")
    master = loader.load_master()
    data   = loader.get_option_data(option, master)

    feat_dict = feat.prepare_features(data, lookback=lookback)

    X_asset_all = feat_dict["X_asset"]   # (N, T, d_asset)
    X_macro_all = feat_dict["X_macro"]   # (N, T, d_macro)
    y_all       = feat_dict["y"]         # (N, n_assets)   actual next-day returns

    # ── Val slice — indices [n_train : n_train + n_val] ─────────────────────
    X_a_val = X_asset_all[n_train : n_train + n_val]
    X_m_val = X_macro_all[n_train : n_train + n_val]
    y_val   = y_all       [n_train : n_train + n_val]

    n_cal = len(X_a_val)
    print(f"[calibrate]   Calibration samples: {n_cal}  (val set)")

    if n_cal < 30:
        raise ValueError(
            f"Only {n_cal} calibration samples — too few for reliable conformal bounds. "
            "Check your data range / split settings."
        )

    # ── Scale with the *train-fitted* scaler (no leakage) ────────────────────
    X_a_s, X_m_s = scaler.transform(X_a_val, X_m_val)

    # ── Batch inference ──────────────────────────────────────────────────────
    BATCH = 64
    all_mu    = []
    all_sigma = []

    with torch.no_grad():
        for start in range(0, n_cal, BATCH):
            end   = min(start + BATCH, n_cal)
            X_a_b = torch.tensor(X_a_s[start:end], dtype=torch.float32)
            X_m_b = torch.tensor(X_m_s[start:end], dtype=torch.float32)
            X_path = build_combined_path(X_a_b, X_m_b)
            mu_b, sigma_b = model(X_path)
            all_mu.append(mu_b.numpy())
            all_sigma.append(sigma_b.numpy())

    mu_arr    = np.concatenate(all_mu,    axis=0)   # (n_cal, n_assets)
    sigma_arr = np.concatenate(all_sigma, axis=0)   # (n_cal, n_assets)

    # ── Nonconformity scores: |y - mu| / sigma ────────────────────────────────
    scores = np.abs(y_val - mu_arr) / (sigma_arr + 1e-8)   # (n_cal, n_assets)

    print(f"[calibrate]   Score stats — mean={scores.mean():.3f}  "
          f"p50={np.median(scores):.3f}  p90={np.percentile(scores, 90):.3f}")

    return {
        "tickers": tickers,
        "scores":  scores.tolist(),   # JSON-serialisable
        "n_cal":   n_cal,
        "val_start": meta["splits"]["train_end"],
        "val_end":   meta["splits"]["val_end"],
    }


# ── Compute quantile thresholds ───────────────────────────────────────────────

def compute_quantiles(scores_dict: dict) -> dict:
    """
    For each alpha level and each ETF compute the conformal quantile q̂.
    Also compute a *pooled* q̂ across all ETFs (used as fallback).
    """
    scores  = np.array(scores_dict["scores"])   # (n_cal, n_assets)
    tickers = scores_dict["tickers"]
    n_cal   = scores.shape[0]

    quantiles = {}
    for alpha in ALPHA_LEVELS:
        # Finite-sample correction: use ceil((n+1)(1-alpha)) / n quantile
        # This gives ≥ (1-alpha) marginal coverage (Vovk et al., 2005).
        level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
        level = min(level, 1.0)   # clamp — can exceed 1 at tiny n

        per_etf = {}
        for j, ticker in enumerate(tickers):
            q = float(np.quantile(scores[:, j], level))
            per_etf[ticker] = round(q, 6)

        pooled = float(np.quantile(scores.ravel(), level))

        quantiles[str(alpha)] = {
            "per_etf": per_etf,
            "pooled":  round(pooled, 6),
            "level_used": round(float(level), 6),
        }

    return quantiles


# ── Empirical coverage diagnostics ───────────────────────────────────────────

def empirical_coverage(scores_dict: dict, quantiles: dict) -> dict:
    """
    Measure how often the interval [mu ± q̂·sigma] actually contained y
    on the calibration set itself (should be ≥ 1-alpha by construction).
    """
    scores  = np.array(scores_dict["scores"])
    tickers = scores_dict["tickers"]

    coverage = {}
    for alpha_str, q_info in quantiles.items():
        alpha = float(alpha_str)
        per_etf_cov = {}
        for j, ticker in enumerate(tickers):
            q = q_info["per_etf"][ticker]
            covered = float((scores[:, j] <= q).mean())
            per_etf_cov[ticker] = round(covered, 4)

        pooled_q   = q_info["pooled"]
        pooled_cov = float((scores.ravel() <= pooled_q).mean())

        coverage[alpha_str] = {
            "per_etf": per_etf_cov,
            "pooled":  round(pooled_cov, 4),
            "target":  round(1 - alpha, 4),
        }

    return coverage


# ── Save ──────────────────────────────────────────────────────────────────────

def save_conformal(option: str, scores_dict: dict, quantiles: dict, coverage: dict):
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    out = {
        "option":       option,
        "calibrated_at": datetime.utcnow().isoformat(),
        "n_cal":        scores_dict["n_cal"],
        "val_start":    scores_dict["val_start"],
        "val_end":      scores_dict["val_end"],
        "tickers":      scores_dict["tickers"],
        "alpha_levels": ALPHA_LEVELS,
        "quantiles":    quantiles,
        "coverage":     coverage,
    }
    path = os.path.join(cfg.MODELS_DIR, f"conformal_option{option}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[calibrate] Saved → {path}")
    return out


# ── Upload to HF ──────────────────────────────────────────────────────────────

def upload_conformal(option: str):
    if not cfg.HF_TOKEN:
        print("[calibrate] No HF_TOKEN — skipping upload.")
        return
    try:
        from huggingface_hub import HfApi
        api  = HfApi(token=cfg.HF_TOKEN)
        path = os.path.join(cfg.MODELS_DIR, f"conformal_option{option}.json")
        with open(path, "rb") as f:
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=f"conformal/conformal_option{option}.json",
                repo_id=cfg.HF_SIGNALS_REPO,
                repo_type="dataset",
                commit_message=f"Update conformal calibration Option {option}",
            )
        print(f"[calibrate] Uploaded conformal_option{option}.json → {cfg.HF_SIGNALS_REPO}")
    except Exception as e:
        print(f"[calibrate] WARNING: Upload failed: {e}")


# ── Coverage summary printer ──────────────────────────────────────────────────

def print_summary(option: str, quantiles: dict, coverage: dict):
    print(f"\n{'─'*55}")
    print(f"Conformal calibration summary — Option {option}")
    print(f"{'─'*55}")
    for alpha_str in sorted(quantiles.keys(), reverse=True):
        q_info = quantiles[alpha_str]
        cov    = coverage[alpha_str]
        target = cov["target"]
        achieved = cov["pooled"]
        status = "✓" if achieved >= target - 0.01 else "✗"
        print(f"  α={float(alpha_str):.0%}  target≥{target:.0%}  "
              f"achieved={achieved:.1%}  {status}  "
              f"(pooled q̂={q_info['pooled']:.3f})")
    print()


# ── CLI entry point ───────────────────────────────────────────────────────────

def calibrate_option(option: str):
    scores_dict = collect_calibration_scores(option)
    quantiles   = compute_quantiles(scores_dict)
    coverage    = empirical_coverage(scores_dict, quantiles)
    save_conformal(option, scores_dict, quantiles, coverage)
    upload_conformal(option)
    print_summary(option, quantiles, coverage)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute conformal quantiles from NCDE val-set residuals"
    )
    parser.add_argument("--option", choices=["A", "B", "both"], default="both")
    args = parser.parse_args()

    options = ["A", "B"] if args.option == "both" else [args.option]
    for opt in options:
        calibrate_option(opt)

    print("[calibrate] All done.")
