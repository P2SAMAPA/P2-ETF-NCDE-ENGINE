# conformal/predict_conformal.py — Wrap NCDE signals with conformal intervals
#
# Reads  : models/conformal_option{A|B}.json  (calibration thresholds)
#           models/latest_signals.json         (raw NCDE mu/sigma)
#
# Produces: models/latest_signals_conformal.json  (uploaded to HF separately)
#
# Schema per ETF (conformal_forecasts):
#   {
#     "mu":         <same as NCDE>,
#     "sigma":      <same as NCDE>,
#     "confidence": <same as NCDE>,
#     "intervals": {
#       "0.9": {"lo": float, "hi": float, "width": float},
#       "0.8": {"lo": float, "hi": float, "width": float},
#       "0.7": {"lo": float, "hi": float, "width": float},
#     },
#     "q_hat": {"0.9": float, "0.8": float, "0.7": float}
#   }
#
# The top_pick remains the highest-mu ETF (conformal wrapping adds intervals,
# it does not change the point-forecast ranking).
#
# Usage:
#   python -m conformal.predict_conformal --option both

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

# ── Repo root on path ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg

HF_CONFORMAL_REPO = cfg.HF_SIGNALS_REPO   # same signals repo, different path


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_ncde_signals() -> dict:
    """Load the latest NCDE signals from local models/ directory."""
    path = os.path.join(cfg.MODELS_DIR, "latest_signals.json")
    if not os.path.exists(path):
        # Try downloading from HF
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=cfg.HF_SIGNALS_REPO,
                filename="signals/latest_signals.json",
                repo_type="dataset",
                token=cfg.HF_TOKEN or None,
                local_dir=cfg.MODELS_DIR,
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            raise FileNotFoundError(
                f"Could not find latest_signals.json locally or on HF: {e}"
            )
    with open(path) as f:
        return json.load(f)


def load_conformal_params(option: str) -> dict:
    """Load conformal quantiles from local models/ or HF."""
    local = os.path.join(cfg.MODELS_DIR, f"conformal_option{option}.json")
    if not os.path.exists(local):
        try:
            from huggingface_hub import hf_hub_download
            local = hf_hub_download(
                repo_id=cfg.HF_SIGNALS_REPO,
                filename=f"conformal/conformal_option{option}.json",
                repo_type="dataset",
                token=cfg.HF_TOKEN or None,
                local_dir=cfg.MODELS_DIR,
                local_dir_use_symlinks=False,
            )
        except Exception as e:
            raise FileNotFoundError(
                f"Conformal params not found for Option {option}. "
                f"Run  python -m conformal.calibrate --option {option}  first.\n{e}"
            )
    with open(local) as f:
        return json.load(f)


# ── Core wrapping logic ───────────────────────────────────────────────────────

def wrap_signal(ncde_signal: dict, conformal_params: dict) -> dict:
    """
    Take a raw NCDE signal dict for one option and produce a conformal-wrapped
    version with guaranteed prediction intervals.

    The original signal is untouched — we add fields on top.
    """
    if not ncde_signal:
        return {}

    forecasts = ncde_signal.get("forecasts", {})
    quantiles  = conformal_params["quantiles"]   # {"0.9": {...}, "0.8": ..., "0.7": ...}
    tickers    = conformal_params["tickers"]

    conformal_forecasts = {}
    for ticker in tickers:
        if ticker not in forecasts:
            continue

        mu    = forecasts[ticker]["mu"]
        sigma = forecasts[ticker]["sigma"]
        conf  = forecasts[ticker]["confidence"]

        intervals = {}
        q_hat     = {}
        for alpha_str, q_info in quantiles.items():
            # Use per-ETF q̂ if available, else fall back to pooled
            q = q_info["per_etf"].get(ticker, q_info["pooled"])
            half_width = q * sigma
            intervals[alpha_str] = {
                "lo":    round(mu - half_width, 6),
                "hi":    round(mu + half_width, 6),
                "width": round(2 * half_width,  6),
            }
            q_hat[alpha_str] = round(q, 6)

        conformal_forecasts[ticker] = {
            "mu":         mu,
            "sigma":      sigma,
            "confidence": conf,
            "intervals":  intervals,
            "q_hat":      q_hat,
        }

    # ── Interval-aware pick: highest mu (point forecast unchanged) ─────────
    top_pick       = ncde_signal["top_pick"]
    top_mu         = ncde_signal["top_mu"]
    top_confidence = ncde_signal["top_confidence"]

    # Width of the 90% interval for the top pick (informational)
    top_interval_90 = conformal_forecasts.get(top_pick, {}).get(
        "intervals", {}
    ).get("0.9", {})

    return {
        # ── Provenance ───────────────────────────────────────────────────
        "option":              ncde_signal.get("option"),
        "option_name":         ncde_signal.get("option_name"),
        "signal_date":         ncde_signal.get("signal_date"),
        "last_data_date":      ncde_signal.get("last_data_date"),
        "generated_at":        datetime.utcnow().isoformat(),
        "ncde_generated_at":   ncde_signal.get("generated_at"),

        # ── Point forecast (unchanged from NCDE) ─────────────────────────
        "top_pick":            top_pick,
        "top_mu":              top_mu,
        "top_confidence":      top_confidence,
        "top_interval_90":     top_interval_90,

        # ── Conformal metadata ────────────────────────────────────────────
        "alpha_levels":        conformal_params["alpha_levels"],
        "n_cal":               conformal_params["n_cal"],
        "cal_period":          f"{conformal_params['val_start']} → {conformal_params['val_end']}",
        "calibrated_at":       conformal_params["calibrated_at"],
        "coverage_diagnostics": conformal_params["coverage"],

        # ── Per-ETF conformal forecasts ───────────────────────────────────
        "conformal_forecasts": conformal_forecasts,

        # ── Carry through NCDE extras ─────────────────────────────────────
        "regime_context":      ncde_signal.get("regime_context"),
        "macro_stress":        ncde_signal.get("macro_stress"),
        "test_ann_return":     ncde_signal.get("test_ann_return"),
        "test_sharpe":         ncde_signal.get("test_sharpe"),
        "test_ic":             ncde_signal.get("test_ic"),
        "model_n_params":      ncde_signal.get("model_n_params"),
        "actual_return":       ncde_signal.get("actual_return"),
        "hit":                 ncde_signal.get("hit"),
    }


# ── Save + upload ─────────────────────────────────────────────────────────────

def save_conformal_signals(sig_A=None, sig_B=None):
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)

    combined = {
        "generated_at": datetime.utcnow().isoformat(),
        "option_A": sig_A,
        "option_B": sig_B,
    }

    local_path = os.path.join(cfg.MODELS_DIR, "latest_signals_conformal.json")
    with open(local_path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"[predict_conformal] Saved → {local_path}")

    # Upload to HF
    if not cfg.HF_TOKEN:
        print("[predict_conformal] No HF_TOKEN — skipping upload.")
        return

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=cfg.HF_TOKEN)
        with open(local_path, "rb") as f:
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo="conformal/latest_signals_conformal.json",
                repo_id=cfg.HF_SIGNALS_REPO,
                repo_type="dataset",
                commit_message=f"Update conformal signals ({combined['generated_at']})",
            )
        print(f"[predict_conformal] Uploaded → {cfg.HF_SIGNALS_REPO}/conformal/")
    except Exception as e:
        print(f"[predict_conformal] WARNING: Upload failed: {e}")

    # Also save per-option history records
    for sig, opt_label in [(sig_A, "A"), (sig_B, "B")]:
        if not sig:
            continue
        _update_conformal_history(sig, opt_label)


def _update_conformal_history(sig: dict, option: str):
    """Append today's conformal signal summary to signal_history_conformal_{opt}.json."""
    history_path = os.path.join(cfg.MODELS_DIR, f"signal_history_conformal_{option}.json")

    # Load existing history from HF if not local
    history = []
    if os.path.exists(history_path):
        with open(history_path) as f:
            history = json.load(f)
    else:
        try:
            from huggingface_hub import hf_hub_download
            dl = hf_hub_download(
                repo_id=cfg.HF_SIGNALS_REPO,
                filename=f"conformal/signal_history_conformal_{option}.json",
                repo_type="dataset",
                token=cfg.HF_TOKEN or None,
                local_dir=cfg.MODELS_DIR,
                local_dir_use_symlinks=False,
            )
            with open(dl) as f:
                history = json.load(f)
        except Exception:
            history = []

    record = {
        "signal_date":    sig["signal_date"],
        "top_pick":       sig["top_pick"],
        "top_mu":         sig["top_mu"],
        "top_confidence": sig["top_confidence"],
        "generated_at":   sig["generated_at"],
        "interval_90_lo": sig.get("top_interval_90", {}).get("lo"),
        "interval_90_hi": sig.get("top_interval_90", {}).get("hi"),
        "interval_90_width": sig.get("top_interval_90", {}).get("width"),
        "actual_return":  sig.get("actual_return"),
        "hit":            sig.get("hit"),
        "interval_covered": None,  # backfilled next run
    }

    existing_dates = {r["signal_date"] for r in history}
    if record["signal_date"] not in existing_dates:
        history.append(record)
    else:
        # Backfill actual_return / hit / interval_covered if now known
        for r in history:
            if r["signal_date"] == record["signal_date"]:
                if record["actual_return"] is not None:
                    r["actual_return"] = record["actual_return"]
                    r["hit"] = record["hit"]
                    lo = r.get("interval_90_lo")
                    hi = r.get("interval_90_hi")
                    if lo is not None and hi is not None:
                        r["interval_covered"] = bool(lo <= record["actual_return"] <= hi)
                break

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    if not cfg.HF_TOKEN:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=cfg.HF_TOKEN)
        with open(history_path, "rb") as f:
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=f"conformal/signal_history_conformal_{option}.json",
                repo_id=cfg.HF_SIGNALS_REPO,
                repo_type="dataset",
                commit_message=f"Update conformal history Option {option} ({record['signal_date']})",
            )
        print(f"[predict_conformal] Uploaded conformal history Option {option}")
    except Exception as e:
        print(f"[predict_conformal] WARNING: History upload failed: {e}")


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_signal_summary(sig: dict):
    if not sig:
        return
    opt   = sig.get("option", "?")
    pick  = sig.get("top_pick", "?")
    mu    = sig.get("top_mu", 0.0)
    iv90  = sig.get("top_interval_90", {})
    lo    = iv90.get("lo", "?")
    hi    = iv90.get("hi", "?")
    date  = sig.get("signal_date", "?")
    n_cal = sig.get("n_cal", 0)
    print(f"  Option {opt}: {pick}  μ={mu:.4f}  "
          f"90% CI=[{lo:.4f}, {hi:.4f}]  "
          f"for {date}  (n_cal={n_cal})")


# ── Entry point ───────────────────────────────────────────────────────────────

def run_option(option: str, ncde_raw: dict) -> dict:
    key = f"option_{option}"
    ncde_sig = ncde_raw.get(key) or {}
    if not ncde_sig:
        print(f"[predict_conformal] No NCDE signal for Option {option} — skipping.")
        return {}
    params = load_conformal_params(option)
    wrapped = wrap_signal(ncde_sig, params)
    print_signal_summary(wrapped)
    return wrapped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Wrap NCDE signals with conformal prediction intervals"
    )
    parser.add_argument("--option", choices=["A", "B", "both"], default="both")
    args = parser.parse_args()

    print("[predict_conformal] Loading NCDE signals...")
    ncde_raw = load_ncde_signals()

    options = ["A", "B"] if args.option == "both" else [args.option]
    sig_A = sig_B = None

    if "A" in options:
        sig_A = run_option("A", ncde_raw)
    if "B" in options:
        sig_B = run_option("B", ncde_raw)

    save_conformal_signals(sig_A, sig_B)
    print("[predict_conformal] Done.")
