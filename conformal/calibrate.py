# conformal/calibrate.py — Conformal calibration for NCDE forecasts
#
# THEORY (split conformal prediction, Angelopoulos & Bates 2022):
# ─────────────────────────────────────────────────────────────────
# Two scoring modes are supported (--score_type flag):
#
#   normalised (default):
#     s_i = |y_i - mu_i| / sigma_i
#     q̂ ends up in "sigma units" — interval = [mu ± q̂·sigma]
#     Useful when sigma is a reliable relative confidence ranking.
#     PROBLEM: if sigma >> actual daily returns (NCDE outputs sigma ~1.0
#     but returns are ~0.01), q̂ collapses to near zero and the interval
#     becomes a trivial sliver around mu.
#
#   absolute (recommended for NCDE):
#     s_i = |y_i - mu_i|
#     q̂ is directly in return space (e.g. q̂ = 0.015 = 1.5% interval half-width)
#     Interval = [mu - q̂,  mu + q̂]  — constant width per coverage level,
#     independent of sigma. Gives genuinely interpretable return ranges.
#     The sigma is still used for ETF ranking; conformal only controls interval width.
#
# Coverage guarantee holds for both modes:
#   P(y ∈ interval) ≥ 1-alpha  (finite-sample, distribution-free)
#
# Usage:
#   python -m conformal.calibrate --option both --score_type absolute
#   python -m conformal.calibrate --option both --score_type normalised
#
# Output: models/conformal_option{A|B}.json
# Uploaded to HF: conformal/conformal_option{A|B}.json

import argparse
import json
import os
import sys
import pickle
from datetime import datetime

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
import loader
import features as feat
from predict import load_model, build_combined_path

DEVICE     = torch.device("cpu")
ALPHA_LEVELS = [0.90, 0.80, 0.70]

SCORE_TYPES = ("normalised", "absolute")


# ── Val-set residuals ─────────────────────────────────────────────────────────

def collect_calibration_scores(option: str, score_type: str) -> dict:
    """
    Run the frozen NCDE over the val slice and compute nonconformity scores.

    normalised : s = |y - mu| / sigma   (sigma-normalised residual)
    absolute   : s = |y - mu|           (raw return residual, in return units)

    Returns dict with scores array (n_cal, n_assets) and metadata.
    """
    print(f"\n[calibrate] Option {option} | score_type={score_type}")

    model, meta, scaler = load_model(option)
    model.eval()

    tickers  = meta["tickers"]
    lookback = meta["config"]["lookback"]
    n_train  = meta["splits"]["n_train"]
    n_val    = meta["splits"]["n_val"]

    print("[calibrate]   Loading master dataset...")
    master    = loader.load_master()
    data      = loader.get_option_data(option, master)
    feat_dict = feat.prepare_features(data, lookback=lookback)

    X_asset_all = feat_dict["X_asset"]
    X_macro_all = feat_dict["X_macro"]
    y_all       = feat_dict["y"]

    X_a_val = X_asset_all[n_train : n_train + n_val]
    X_m_val = X_macro_all[n_train : n_train + n_val]
    y_val   = y_all       [n_train : n_train + n_val]

    n_cal = len(X_a_val)
    print(f"[calibrate]   Calibration samples: {n_cal}  (val set)")

    if n_cal < 30:
        raise ValueError(
            f"Only {n_cal} calibration samples — too few for reliable conformal bounds."
        )

    X_a_s, X_m_s = scaler.transform(X_a_val, X_m_val)

    BATCH     = 64
    all_mu    = []
    all_sigma = []

    with torch.no_grad():
        for start in range(0, n_cal, BATCH):
            end    = min(start + BATCH, n_cal)
            X_a_b  = torch.tensor(X_a_s[start:end], dtype=torch.float32)
            X_m_b  = torch.tensor(X_m_s[start:end], dtype=torch.float32)
            X_path = build_combined_path(X_a_b, X_m_b)
            mu_b, sigma_b = model(X_path)
            all_mu.append(mu_b.numpy())
            all_sigma.append(sigma_b.numpy())

    mu_arr    = np.concatenate(all_mu,    axis=0)   # (n_cal, n_assets)
    sigma_arr = np.concatenate(all_sigma, axis=0)   # (n_cal, n_assets)

    # ── Nonconformity scores ──────────────────────────────────────────────────
    residuals = np.abs(y_val - mu_arr)              # (n_cal, n_assets)

    if score_type == "normalised":
        scores = residuals / (sigma_arr + 1e-8)
        score_label = "|y - μ| / σ"
    else:  # absolute
        scores = residuals
        score_label = "|y - μ|  (return units)"

    print(f"[calibrate]   Score ({score_label}):")
    print(f"              mean={scores.mean():.5f}  "
          f"p50={np.median(scores):.5f}  "
          f"p90={np.percentile(scores, 90):.5f}  "
          f"p95={np.percentile(scores, 95):.5f}")

    # Sanity check for normalised mode — warn if sigma >> residuals
    if score_type == "normalised":
        sigma_mean = sigma_arr.mean()
        resid_mean = residuals.mean()
        if sigma_mean > resid_mean * 10:
            print(
                f"[calibrate]   WARNING: sigma_mean={sigma_mean:.4f} is "
                f"{sigma_mean/resid_mean:.0f}x larger than residual_mean={resid_mean:.5f}. "
                f"This will produce q̂ ≈ 0 and trivially narrow intervals. "
                f"Consider using --score_type absolute instead."
            )

    return {
        "tickers":   tickers,
        "scores":    scores.tolist(),
        "score_type": score_type,
        "score_label": score_label,
        "n_cal":     n_cal,
        "val_start": meta["splits"]["train_end"],
        "val_end":   meta["splits"]["val_end"],
        # diagnostics
        "score_mean": round(float(scores.mean()),  6),
        "score_p50":  round(float(np.median(scores)), 6),
        "score_p90":  round(float(np.percentile(scores, 90)), 6),
    }


# ── Quantile thresholds ───────────────────────────────────────────────────────

def compute_quantiles(scores_dict: dict) -> dict:
    """
    For each alpha and each ETF compute the conformal quantile q̂.
    Also compute a pooled q̂ across all ETFs (used as fallback).

    For score_type=absolute, q̂ is in return units (e.g. 0.015 = 1.5%).
    For score_type=normalised, q̂ is in sigma units.
    """
    scores     = np.array(scores_dict["scores"])   # (n_cal, n_assets)
    tickers    = scores_dict["tickers"]
    n_cal      = scores.shape[0]
    score_type = scores_dict["score_type"]

    quantiles = {}
    for alpha in ALPHA_LEVELS:
        # Finite-sample corrected level (Vovk et al. 2005)
        level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
        level = min(level, 1.0)

        per_etf = {}
        for j, ticker in enumerate(tickers):
            q = float(np.quantile(scores[:, j], level))
            per_etf[ticker] = round(q, 6)

        pooled = float(np.quantile(scores.ravel(), level))

        quantiles[str(alpha)] = {
            "per_etf":    per_etf,
            "pooled":     round(pooled, 6),
            "level_used": round(float(level), 6),
            "score_type": score_type,
            # human-readable interpretation
            "interpretation": (
                f"interval half-width = q̂ × σ  (q̂ in σ-units)"
                if score_type == "normalised"
                else f"interval half-width = q̂  (q̂ = {round(pooled*100,3):.3f}% return)"
            ),
        }

    return quantiles


# ── Empirical coverage ────────────────────────────────────────────────────────

def empirical_coverage(scores_dict: dict, quantiles: dict) -> dict:
    scores  = np.array(scores_dict["scores"])
    tickers = scores_dict["tickers"]

    coverage = {}
    for alpha_str, q_info in quantiles.items():
        per_etf_cov = {}
        for j, ticker in enumerate(tickers):
            q       = q_info["per_etf"][ticker]
            covered = float((scores[:, j] <= q).mean())
            per_etf_cov[ticker] = round(covered, 4)

        pooled_q   = q_info["pooled"]
        pooled_cov = float((scores.ravel() <= pooled_q).mean())

        coverage[alpha_str] = {
            "per_etf": per_etf_cov,
            "pooled":  round(pooled_cov, 4),
            "target":  round(1 - float(alpha_str), 4),
        }

    return coverage


# ── Save ──────────────────────────────────────────────────────────────────────

def save_conformal(option: str, scores_dict: dict,
                   quantiles: dict, coverage: dict) -> dict:
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    out = {
        "option":         option,
        "calibrated_at":  datetime.utcnow().isoformat(),
        "score_type":     scores_dict["score_type"],
        "score_label":    scores_dict["score_label"],
        "n_cal":          scores_dict["n_cal"],
        "val_start":      scores_dict["val_start"],
        "val_end":        scores_dict["val_end"],
        "tickers":        scores_dict["tickers"],
        "alpha_levels":   ALPHA_LEVELS,
        "score_stats": {
            "mean": scores_dict["score_mean"],
            "p50":  scores_dict["score_p50"],
            "p90":  scores_dict["score_p90"],
        },
        "quantiles":  quantiles,
        "coverage":   coverage,
    }
    path = os.path.join(cfg.MODELS_DIR, f"conformal_option{option}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[calibrate] Saved → {path}")
    return out


# ── Upload ────────────────────────────────────────────────────────────────────

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
        print(f"[calibrate] Uploaded → {cfg.HF_SIGNALS_REPO}")
    except Exception as e:
        print(f"[calibrate] WARNING: Upload failed: {e}")


# ── Summary printer ───────────────────────────────────────────────────────────

def print_summary(option: str, scores_dict: dict,
                  quantiles: dict, coverage: dict):
    score_type = scores_dict["score_type"]
    unit       = "σ-units" if score_type == "normalised" else "return units"

    print(f"\n{'─'*60}")
    print(f"Conformal calibration — Option {option}  [{score_type}]")
    print(f"{'─'*60}")
    print(f"  Score stats: mean={scores_dict['score_mean']:.5f}  "
          f"p50={scores_dict['score_p50']:.5f}  "
          f"p90={scores_dict['score_p90']:.5f}")
    print()

    for alpha_str in sorted(quantiles.keys(), reverse=True):
        q_info   = quantiles[alpha_str]
        cov      = coverage[alpha_str]
        target   = cov["target"]
        achieved = cov["pooled"]
        pooled_q = q_info["pooled"]
        status   = "✓" if achieved >= target - 0.01 else "✗"

        if score_type == "absolute":
            q_display = f"±{pooled_q*100:.3f}% return"
        else:
            q_display = f"q̂={pooled_q:.4f} σ-units"

        print(f"  α={float(alpha_str):.0%}  "
              f"target≥{target:.0%}  achieved={achieved:.1%}  {status}  "
              f"({q_display})")
    print()


# ── Main calibration function ─────────────────────────────────────────────────

def calibrate_option(option: str, score_type: str = "absolute"):
    """
    Run full calibration pipeline for one option.
    Default score_type is 'absolute' — gives interpretable return-space intervals.
    """
    scores_dict = collect_calibration_scores(option, score_type)
    quantiles   = compute_quantiles(scores_dict)
    coverage    = empirical_coverage(scores_dict, quantiles)
    save_conformal(option, scores_dict, quantiles, coverage)
    upload_conformal(option)
    print_summary(option, scores_dict, quantiles, coverage)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute conformal quantiles from NCDE val-set residuals.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Score types:
  absolute   (recommended) — s = |y - mu|
             q̂ is in return units, e.g. q̂=0.015 means ±1.5% interval.
             Use this when NCDE sigma >> actual daily returns.

  normalised (original)    — s = |y - mu| / sigma
             q̂ is in sigma-units — interval = mu ± q̂·sigma.
             Use this only when sigma is calibrated to return scale.
""",
    )
    parser.add_argument("--option",
                        choices=["A", "B", "both"], default="both")
    parser.add_argument("--score_type",
                        choices=SCORE_TYPES, default="absolute",
                        help="absolute (return-space) or normalised (sigma-space). "
                             "Default: absolute")
    args = parser.parse_args()

    options = ["A", "B"] if args.option == "both" else [args.option]
    for opt in options:
        calibrate_option(opt, args.score_type)

    print("[calibrate] All done.")
