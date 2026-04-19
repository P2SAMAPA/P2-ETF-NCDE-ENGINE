# conformal/predict_conformal.py — Wrap NCDE signals with conformal intervals
#
# FIX: Resolved filename collision between calibration params and signal output.
#
#   BEFORE (broken):
#     calibrate.py  saved params  → models/conformal_option{A|B}.json
#     predict_conformal.py saved signal → models/conformal_option{A|B}.json  ← SAME NAME
#     Result: next run downloads the signal file, finds no "quantiles" key → KeyError
#
#   AFTER (fixed):
#     calibration params  → models/conformal_params_option{A|B}.json   (new name)
#     signal output       → models/conformal_signal_option{A|B}.json   (new name)
#     HF paths mirror the same distinction
#
# Supports two score types (determined by what calibrate.py used):
#
#   absolute   (recommended):
#     interval = [mu - q̂,  mu + q̂]
#     q̂ is in return units — directly interpretable (e.g. ±1.5%)
#
#   normalised (original):
#     interval = [mu - q̂·sigma,  mu + q̂·sigma]
#     q̂ is in sigma-units — only meaningful if sigma ≈ return scale
#
# The score_type is read from the calibration JSON automatically.
# Self-healing: auto-calibrates (absolute mode) if params missing.
#
# Also writes models/latest_signals_conformal.json — the combined file
# expected by the CI verification step.
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


# ── Filename helpers ──────────────────────────────────────────────────────────
# Centralised so every function uses the same names consistently.

def _params_local(option: str) -> str:
    """Local path for calibration params (has 'quantiles' key)."""
    return os.path.join(cfg.MODELS_DIR, f"conformal_params_option{option}.json")

def _params_hf(option: str) -> str:
    """HF repo path for calibration params."""
    return f"conformal/conformal_params_option{option}.json"

def _signal_local(option: str) -> str:
    """Local path for conformal-wrapped signal output."""
    return os.path.join(cfg.MODELS_DIR, f"conformal_signal_option{option}.json")

def _signal_hf(option: str) -> str:
    """HF repo path for conformal-wrapped signal output."""
    return f"conformal/conformal_signal_option{option}.json"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_ncde_signal(option: str) -> dict:
    """Load raw NCDE signal for a single option."""
    local = os.path.join(cfg.MODELS_DIR, f"signal_{option}.json")
    if os.path.exists(local):
        with open(local) as f:
            return json.load(f)
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=cfg.HF_SIGNALS_REPO,
            filename=f"signals/signal_{option}.json",
            repo_type="dataset",
            token=cfg.HF_TOKEN or None,
            local_dir=cfg.MODELS_DIR,
            local_dir_use_symlinks=False,
        )
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not find signal_{option}.json locally or on HF: {e}"
        )


def load_conformal_params(option: str) -> dict | None:
    """
    Load calibration params (the file with 'quantiles').
    Uses the new _params_* names to avoid collision with signal output.
    Also accepts the old name conformal_option{X}.json as a fallback
    so existing calibrations are not wasted.
    """
    # 1. New canonical name
    local = _params_local(option)
    if os.path.exists(local):
        with open(local) as f:
            data = json.load(f)
        if "quantiles" in data:
            return data
        # File exists but is the old signal output (no quantiles) — ignore it
        print(f"[predict_conformal] WARNING: {local} has no 'quantiles' key "
              f"— likely old signal file. Will try HF or re-calibrate.")

    # 2. Old name still on disk (calibrate.py hasn't been re-run yet)
    old_local = os.path.join(cfg.MODELS_DIR, f"conformal_option{option}.json")
    if os.path.exists(old_local):
        with open(old_local) as f:
            data = json.load(f)
        if "quantiles" in data:
            print(f"[predict_conformal] Found legacy params at {old_local} — "
                  f"copying to {local} for future runs.")
            os.makedirs(cfg.MODELS_DIR, exist_ok=True)
            with open(local, "w") as f:
                json.dump(data, f, indent=2)
            return data

    # 3. Try HF (new name first, then old)
    for hf_path in [_params_hf(option),
                    f"conformal/conformal_option{option}.json"]:
        try:
            from huggingface_hub import hf_hub_download
            dl = hf_hub_download(
                repo_id=cfg.HF_SIGNALS_REPO,
                filename=hf_path,
                repo_type="dataset",
                token=cfg.HF_TOKEN or None,
                local_dir=cfg.MODELS_DIR,
                local_dir_use_symlinks=False,
            )
            with open(dl) as f:
                data = json.load(f)
            if "quantiles" in data:
                # Save under canonical name for next time
                with open(local, "w") as f:
                    json.dump(data, f, indent=2)
                return data
        except Exception:
            continue

    return None


# ── Auto-calibration ──────────────────────────────────────────────────────────

def ensure_calibrated(option: str) -> dict:
    params = load_conformal_params(option)
    if params is not None:
        score_type = params.get("score_type", "unknown")
        print(f"[predict_conformal] Params loaded for Option {option} "
              f"(score_type={score_type}).")
        return params

    print(f"[predict_conformal] No conformal params for Option {option} — "
          f"auto-calibrating with score_type=absolute...")

    model_path = os.path.join(cfg.MODELS_DIR, f"ncde_option{option}_best.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model at {model_path}. "
            f"Run  python train.py --option {option}  first."
        )

    from conformal.calibrate import calibrate_option
    calibrate_option(option, score_type="absolute")

    params = load_conformal_params(option)
    if params is None:
        raise RuntimeError(
            f"Calibration ran but conformal params for Option {option} "
            f"still not found. Check calibrate.py output."
        )
    return params


# ── Core wrapping logic ───────────────────────────────────────────────────────

def _compute_interval(mu: float, sigma: float, q: float,
                      score_type: str) -> tuple[float, float]:
    """
    absolute  : half-width = q           (q is in return units, e.g. 0.015 = 1.5%)
    normalised: half-width = q * sigma   (q is in sigma units)
    """
    half = q if score_type == "absolute" else q * sigma
    return mu - half, mu + half


def wrap_signal(ncde_signal: dict, conformal_params: dict) -> dict:
    """
    Wrap a raw NCDE signal with conformal prediction intervals.
    Point forecast (mu, top_pick ranking) is completely unchanged.
    """
    forecasts  = ncde_signal.get("forecasts", {})
    quantiles  = conformal_params["quantiles"]   # KeyError impossible after fix
    tickers    = conformal_params["tickers"]
    score_type = conformal_params.get("score_type", "normalised")

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
            q      = q_info["per_etf"].get(ticker, q_info["pooled"])
            lo, hi = _compute_interval(mu, sigma, q, score_type)
            intervals[alpha_str] = {
                "lo":    round(lo,      6),
                "hi":    round(hi,      6),
                "width": round(hi - lo, 6),
            }
            q_hat[alpha_str] = round(q, 6)

        conformal_forecasts[ticker] = {
            "mu":         mu,
            "sigma":      sigma,
            "confidence": conf,
            "intervals":  intervals,
            "q_hat":      q_hat,
            "score_type": score_type,
        }

    top_pick       = ncde_signal["top_pick"]
    top_mu         = ncde_signal["top_mu"]
    top_confidence = ncde_signal["top_confidence"]
    top_interval_90 = (conformal_forecasts
                       .get(top_pick, {})
                       .get("intervals", {})
                       .get("0.9", {}))

    pooled_q_90 = quantiles.get("0.9", {}).get("pooled", 0)
    if score_type == "absolute":
        q_label = f"±{pooled_q_90*100:.3f}% return  (pooled 90%)"
    else:
        q_label = f"q̂={pooled_q_90:.4f} σ-units  (pooled 90%)"

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
        "score_type":           score_type,
        "q_label":              q_label,
        "alpha_levels":         conformal_params["alpha_levels"],
        "n_cal":                conformal_params["n_cal"],
        "cal_period":           (f"{conformal_params['val_start']} → "
                                 f"{conformal_params['val_end']}"),
        "calibrated_at":        conformal_params["calibrated_at"],
        "coverage_diagnostics": conformal_params["coverage"],
        "score_stats":          conformal_params.get("score_stats", {}),
        "conformal_forecasts":  conformal_forecasts,
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

def save_conformal_signal(sig_A: dict | None, sig_B: dict | None):
    """
    Save individual conformal signal files and the combined
    latest_signals_conformal.json expected by the CI verification step.
    """
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)

    hf_uploads = []   # list of (local_path, repo_path) to upload

    # ── Individual option files ───────────────────────────────────────────────
    for sig, option in [(sig_A, "A"), (sig_B, "B")]:
        if sig is None:
            continue
        local = _signal_local(option)
        with open(local, "w") as f:
            json.dump(sig, f, indent=2)
        print(f"[predict_conformal] Saved → {local}")
        hf_uploads.append((local, _signal_hf(option)))
        _update_conformal_history(sig, option)

    # ── Combined file for CI verification ────────────────────────────────────
    combined = {
        "generated_at": datetime.utcnow().isoformat(),
        "option_A":     sig_A,
        "option_B":     sig_B,
    }
    combined_local = os.path.join(cfg.MODELS_DIR, "latest_signals_conformal.json")
    with open(combined_local, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"[predict_conformal] Saved → {combined_local}")
    hf_uploads.append((combined_local,
                       "conformal/latest_signals_conformal.json"))

    # ── Upload all to HF ──────────────────────────────────────────────────────
    if not cfg.HF_TOKEN:
        print("[predict_conformal] No HF_TOKEN — skipping upload.")
        return

    try:
        from huggingface_hub import HfApi
        api = HfApi(token=cfg.HF_TOKEN)
        for local_path, repo_path in hf_uploads:
            with open(local_path, "rb") as f:
                api.upload_file(
                    path_or_fileobj=f,
                    path_in_repo=repo_path,
                    repo_id=cfg.HF_SIGNALS_REPO,
                    repo_type="dataset",
                    commit_message=f"Update {os.path.basename(repo_path)} "
                                   f"({combined['generated_at'][:10]})",
                )
            print(f"[predict_conformal] Uploaded → {repo_path}")
    except Exception as e:
        print(f"[predict_conformal] WARNING: HF upload failed: {e}")


# ── History ───────────────────────────────────────────────────────────────────

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
                local_dir_use_symlinks=False,
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
        "score_type":        sig.get("score_type", "unknown"),
        "interval_90_lo":    iv90.get("lo"),
        "interval_90_hi":    iv90.get("hi"),
        "interval_90_width": iv90.get("width"),
        "actual_return":     sig.get("actual_return"),
        "hit":               sig.get("hit"),
        "interval_covered":  None,
    }

    existing = {r["signal_date"] for r in history}
    if record["signal_date"] not in existing:
        history.append(record)
    else:
        for r in history:
            if r["signal_date"] == record["signal_date"]:
                if record["actual_return"] is not None:
                    r["actual_return"]   = record["actual_return"]
                    r["hit"]             = record["hit"]
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

def run_option(option: str) -> dict | None:
    """Run conformal prediction for a single option."""
    try:
        ncde_sig = load_ncde_signal(option)
    except FileNotFoundError as e:
        print(f"[predict_conformal] NCDE Option {option} signal not found — "
              f"skipping: {e}")
        return None

    if not isinstance(ncde_sig, dict) or "top_pick" not in ncde_sig:
        print(f"[predict_conformal] NCDE Option {option} signal invalid — skipping.")
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
          f"({wrapped.get('score_type','?')})  "
          f"signal_date={wrapped['signal_date']}")
    return wrapped


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Wrap NCDE signals with conformal prediction intervals. "
            "Score type is read from calibration JSON automatically. "
            "Auto-calibrates with score_type=absolute if params are missing."
        )
    )
    parser.add_argument("--option", choices=["A", "B", "both"], default="both")
    args = parser.parse_args()

    options = ["A", "B"] if args.option == "both" else [args.option]

    sig_A = sig_B = None
    for opt in options:
        print(f"[predict_conformal] Processing Option {opt}...")
        sig = run_option(opt)
        if opt == "A":
            sig_A = sig
        else:
            sig_B = sig

    # Save everything (individual files + combined CI file) in one pass
    save_conformal_signal(sig_A, sig_B)

    print("[predict_conformal] Done.")
