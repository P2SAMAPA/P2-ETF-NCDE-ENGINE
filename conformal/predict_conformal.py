# conformal/predict_conformal.py — Wrap NCDE signals with conformal intervals
#
# SELF-HEALING: if conformal_option{X}.json is missing (first run or after
# a model retrain), this script auto-runs calibration inline before wrapping.
# No separate "run calibrate first" step is required.
#
# Reads  : models/conformal_option{A|B}.json   (calibration thresholds)
#           HF signals/latest_signals.json       (raw NCDE mu/sigma)
#
# Writes : models/latest_signals_conformal.json
#           HF conformal/latest_signals_conformal.json
#           HF conformal/signal_history_conformal_{A|B}.json
#
# Usage:
#   python -m conformal.predict_conformal --option both

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_ncde_signals() -> dict:
    """Load latest NCDE signals — local first, then HF."""
    local = os.path.join(cfg.MODELS_DIR, "latest_signals.json")
    if os.path.exists(local):
        with open(local) as f:
            return json.load(f)
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=cfg.HF_SIGNALS_REPO,
            filename="signals/latest_signals.json",
            repo_type="dataset",
            token=cfg.HF_TOKEN or None,
            local_dir=cfg.MODELS_DIR,
        )
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not find latest_signals.json locally or on HF: {e}"
        )


def load_conformal_params(option: str) -> dict | None:
    """
    Return conformal params dict, or None if not found anywhere.
    Never raises — caller decides what to do when None is returned.
    """
    local = os.path.join(cfg.MODELS_DIR, f"conformal_option{option}.json")
    if os.path.exists(local):
        with open(local) as f:
            return json.load(f)
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=cfg.HF_SIGNALS_REPO,
            filename=f"conformal/conformal_option{option}.json",
            repo_type="dataset",
            token=cfg.HF_TOKEN or None,
            local_dir=cfg.MODELS_DIR,
        )
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None   # triggers auto-calibration in caller


# ── Auto-calibration ──────────────────────────────────────────────────────────

def ensure_calibrated(option: str) -> dict:
    """
    Return conformal params. If missing, run calibration first.
    Raises FileNotFoundError with a clear message if no model exists.
    """
    params = load_conformal_params(option)
    if params is not None:
        print(f"[predict_conformal] Conformal params loaded for Option {option}.")
        return params

    print(f"[predict_conformal] No conformal params for Option {option} — "
          f"running calibration now (this is normal on first run)...")

    model_path = os.path.join(cfg.MODELS_DIR, f"ncde_option{option}_best.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model at {model_path}. "
            f"Run  python train.py --option {option}  first, "
            f"then this script will calibrate automatically."
        )

    from conformal.calibrate import calibrate_option
    calibrate_option(option)

    params = load_conformal_params(option)
    if params is None:
        raise RuntimeError(
            f"Calibration ran but conformal_option{option}.json still not found."
        )
    return params


# ── Core wrapping logic ───────────────────────────────────────────────────────

def wrap_signal(ncde_signal: dict, conformal_params: dict) -> dict:
    """
    Wrap a raw NCDE signal with conformal prediction intervals.
    Point forecast (mu, top_pick ranking) is completely unchanged.
    """
    forecasts = ncde_signal.get("forecasts", {})
    quantiles = conformal_params["quantiles"]
    tickers   = conformal_params["tickers"]

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
            q    = q_info["per_etf"].get(ticker, q_info["pooled"])
            half = q * sigma
            intervals[alpha_str] = {
                "lo":    round(mu - half, 6),
                "hi":    round(mu + half, 6),
                "width": round(2 * half,  6),
            }
            q_hat[alpha_str] = round(q, 6)

        conformal_forecasts[ticker] = {
            "mu":         mu,
            "sigma":      sigma,
            "confidence": conf,
            "intervals":  intervals,
            "q_hat":      q_hat,
        }

    top_pick        = ncde_signal["top_pick"]
    top_mu          = ncde_signal["top_mu"]
    top_confidence  = ncde_signal["top_confidence"]
    top_interval_90 = (conformal_forecasts
                       .get(top_pick, {})
                       .get("intervals", {})
                       .get("0.9", {}))

    return {
        "option":               ncde_signal.get("option"),
        "option_name":          ncde_signal.get("option_name"),
        "signal_date":          ncde_signal.get("signal_date"),
        "last_data_date":       ncde_signal.get("last_data_date"),
        "generated_at":         datetime.utcnow().isoformat(),
        "ncde_generated_at":    ncde_signal.get("generated_at"),
        "top_pick":             top_pick,
        "top_mu":               top_mu,
        "top_confidence":       top_confidence,
        "top_interval_90":      top_interval_90,
        "alpha_levels":         conformal_params["alpha_levels"],
        "n_cal":                conformal_params["n_cal"],
        "cal_period":           (f"{conformal_params['val_start']} → "
                                 f"{conformal_params['val_end']}"),
        "calibrated_at":        conformal_params["calibrated_at"],
        "coverage_diagnostics": conformal_params["coverage"],
        "conformal_forecasts":  conformal_forecasts,
        # carry-through from NCDE
        "regime_context":       ncde_signal.get("regime_context"),
        "macro_stress":         ncde_signal.get("macro_stress"),
        "test_ann_return":      ncde_signal.get("test_ann_return"),
        "test_sharpe":          ncde_signal.get("test_sharpe"),
        "test_ic":              ncde_signal.get("test_ic"),
        "model_n_params":       ncde_signal.get("model_n_params"),
        "actual_return":        ncde_signal.get("actual_return"),
        "hit":                  ncde_signal.get("hit"),
    }


# ── Save + upload ─────────────────────────────────────────────────────────────

def save_and_upload(sig_A=None, sig_B=None):
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)

    # Never write {} — use None for options that didn't run
    combined = {
        "generated_at": datetime.utcnow().isoformat(),
        "option_A":     sig_A or None,
        "option_B":     sig_B or None,
    }

    local_path = os.path.join(cfg.MODELS_DIR, "latest_signals_conformal.json")
    with open(local_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"[predict_conformal] Saved locally → {local_path}")

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
        print(f"[predict_conformal] Uploaded → {cfg.HF_SIGNALS_REPO}")
    except Exception as e:
        print(f"[predict_conformal] WARNING: upload failed: {e}")

    for sig, opt in [(sig_A, "A"), (sig_B, "B")]:
        if sig:
            _update_conformal_history(sig, opt)


def _update_conformal_history(sig: dict, option: str):
    history_path = os.path.join(
        cfg.MODELS_DIR, f"signal_history_conformal_{option}.json"
    )
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
            )
            with open(dl) as f:
                history = json.load(f)
        except Exception:
            history = []

    iv90   = sig.get("top_interval_90", {})
    record = {
        "signal_date":       sig["signal_date"],
        "top_pick":          sig["top_pick"],
        "top_mu":            sig["top_mu"],
        "top_confidence":    sig["top_confidence"],
        "generated_at":      sig["generated_at"],
        "interval_90_lo":    iv90.get("lo"),
        "interval_90_hi":    iv90.get("hi"),
        "interval_90_width": iv90.get("width"),
        "actual_return":     sig.get("actual_return"),
        "hit":               sig.get("hit"),
        "interval_covered":  None,   # backfilled next run
    }

    existing = {r["signal_date"] for r in history}
    if record["signal_date"] not in existing:
        history.append(record)
    else:
        for r in history:
            if r["signal_date"] == record["signal_date"]:
                if record["actual_return"] is not None:
                    r["actual_return"] = record["actual_return"]
                    r["hit"] = record["hit"]
                    lo = r.get("interval_90_lo")
                    hi = r.get("interval_90_hi")
                    if lo is not None and hi is not None:
                        r["interval_covered"] = bool(
                            lo <= record["actual_return"] <= hi
                        )
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
                commit_message=(f"Update conformal history Option {option} "
                                f"({record['signal_date']})"),
            )
        print(f"[predict_conformal] Uploaded history Option {option}")
    except Exception as e:
        print(f"[predict_conformal] WARNING: history upload failed: {e}")


# ── Per-option runner ─────────────────────────────────────────────────────────

def run_option(option: str, ncde_raw: dict) -> dict | None:
    """
    Returns a fully-wrapped conformal signal dict, or None.
    Never returns {} — callers can rely on truthiness check.
    """
    key      = f"option_{option}"
    ncde_sig = ncde_raw.get(key)

    # Treat missing / None / {} / incomplete dicts the same way
    if not ncde_sig or not isinstance(ncde_sig, dict) or "top_pick" not in ncde_sig:
        print(f"[predict_conformal] NCDE Option {option} signal is absent or "
              f"incomplete — skipping.")
        return None

    try:
        params = ensure_calibrated(option)
    except FileNotFoundError as e:
        print(f"[predict_conformal] Option {option} skipped — {e}")
        return None
    except Exception as e:
        print(f"[predict_conformal] Option {option} calibration error: {e}")
        return None

    wrapped = wrap_signal(ncde_sig, params)
    iv90    = wrapped.get("top_interval_90", {})
    print(f"[predict_conformal]   Option {option}: {wrapped['top_pick']}  "
          f"μ={wrapped['top_mu']:.4f}  "
          f"90% CI=[{iv90.get('lo','?')}, {iv90.get('hi','?')}]  "
          f"for {wrapped['signal_date']}  (n_cal={wrapped['n_cal']})")
    return wrapped


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Wrap NCDE signals with conformal prediction intervals. "
            "Auto-calibrates on first run if conformal params are missing."
        )
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

    save_and_upload(sig_A, sig_B)
    print("[predict_conformal] Done.")
