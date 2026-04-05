# predict.py — Daily signal generation for P2-ETF-NCDE-ENGINE
#
# Loads trained NCDE model, runs inference on latest data,
# and uploads signal JSON + history to HF signals repo.
#
# Output schema per option:
#   signal_date, last_data_date, generated_at,
#   forecasts: {ticker: {mu, sigma, confidence}},
#   top_pick, top_mu, top_confidence,
#   regime_context, macro_stress,
#   actual_return, hit  (backfilled next run)
#
# Usage:
#   python predict.py --option both

import argparse
import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import torch
import torchcde
from huggingface_hub import HfApi, hf_hub_download

import config as cfg
import loader
import features as feat
from model import NCDEForecaster

DEVICE = torch.device("cpu")


# ── Helpers ────────────────────────────────────────────────────────────────────

def next_trading_day(from_date: str = None) -> str:
    nyse   = mcal.get_calendar("NYSE")
    base   = pd.Timestamp(from_date) if from_date else pd.Timestamp.today()
    sched  = nyse.schedule(
        start_date=base.strftime("%Y-%m-%d"),
        end_date=(base + pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
    )
    days   = mcal.date_range(sched, frequency="1D").normalize().tz_localize(None)
    future = [d for d in days if d > base]
    return str(future[0].date()) if future else str((base + pd.Timedelta(days=1)).date())


def _get_actual_return(pick: str, signal_date: str, master: pd.DataFrame) -> tuple:
    try:
        date = pd.Timestamp(signal_date)
        col  = f"{pick}_ret"
        if col in master.columns and date in master.index:
            ret = master.loc[date, col]
            if not np.isnan(ret):
                return float(ret), bool(ret > 0)
    except Exception:
        pass
    return None, None


def _build_splines(X_asset: np.ndarray, X_macro: np.ndarray):
    """Build torchcde cubic spline paths from (1, T, F) numpy arrays."""
    Xa = torch.tensor(X_asset, dtype=torch.float32)
    Xm = torch.tensor(X_macro, dtype=torch.float32)
    t  = torch.arange(Xa.shape[1], dtype=torch.float32)
    asset_path = torchcde.CubicSpline(torchcde.natural_cubic_coeffs(Xa, t), t)
    macro_path = torchcde.CubicSpline(torchcde.natural_cubic_coeffs(Xm, t), t)
    return asset_path, macro_path


# ── Model loader ───────────────────────────────────────────────────────────────

def load_model(option: str) -> tuple:
    model_path  = os.path.join(cfg.MODELS_DIR, f"ncde_option{option}_best.pt")
    meta_path   = os.path.join(cfg.MODELS_DIR, f"meta_option{option}.json")
    scaler_path = os.path.join(cfg.MODELS_DIR, f"scaler_option{option}.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run train.py --option {option} first."
        )

    with open(meta_path) as f:
        meta = json.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    model = NCDEForecaster(
        n_asset_path_dim = meta["n_asset_path_dim"],
        n_macro_feats    = meta["n_macro_feats"],
        n_assets         = meta["n_assets"],
        hidden_dim       = meta["config"]["hidden_dim"],
        n_layers         = meta["config"]["n_layers"],
        readout_dim      = meta["config"]["readout_dim"],
        dropout          = 0.0,   # no dropout at inference
    ).to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model, meta, scaler


# ── Inference ──────────────────────────────────────────────────────────────────

def _build_inference_tensors(option: str, master: pd.DataFrame, meta: dict) -> tuple:
    """Build scaled (1, T, F) arrays ready for spline construction."""
    data        = loader.get_option_data(option, master)
    lookback    = meta["config"]["lookback"]
    tickers     = meta["tickers"]

    asset_feat  = feat.build_asset_features(data["log_returns"], data["vol"])
    macro_feat  = feat.build_macro_features(data["macro"], data["macro_derived"])

    common_idx  = asset_feat.index.intersection(macro_feat.index)
    af          = asset_feat.reindex(common_idx).ffill().fillna(0.0)
    mf          = macro_feat.reindex(common_idx).ffill().fillna(0.0)

    # Build per-asset column indices (same logic as features.build_sequences)
    n_assets    = len(tickers)
    asset_col_indices = []
    for ticker in tickers:
        cols = [c for c in af.columns if c.startswith(ticker + "_")]
        idxs = [af.columns.get_loc(c) for c in cols]
        asset_col_indices.append(idxs)
    n_asset_feats = len(asset_col_indices[0])

    af_window   = af.iloc[-lookback:].values
    mf_window   = mf.iloc[-lookback:].values

    X_asset     = np.zeros((1, lookback, n_assets * n_asset_feats), dtype=np.float32)
    X_macro     = mf_window[np.newaxis, :, :]

    for a, col_idxs in enumerate(asset_col_indices):
        start = a * n_asset_feats
        X_asset[0, :, start:start + n_asset_feats] = af_window[:, col_idxs]

    last_data_date = str(af.index[-1].date())

    # Regime context
    latest_macro   = data["macro"].iloc[-1]
    regime_context = {
        "VIX":       round(float(latest_macro.get("VIX",       0)), 2),
        "T10Y2Y":    round(float(latest_macro.get("T10Y2Y",    0)), 3),
        "HY_SPREAD": round(float(latest_macro.get("HY_SPREAD", 0)), 2),
        "USD_INDEX": round(float(latest_macro.get("USD_INDEX", 0)), 2),
    }
    stress = None
    if "macro_stress_composite" in data["macro_derived"].columns:
        stress = round(float(data["macro_derived"]["macro_stress_composite"].iloc[-1]), 3)

    return X_asset, X_macro, last_data_date, regime_context, stress, tickers


def generate_signal(option: str, master: pd.DataFrame) -> dict:
    print(f"\n[predict] Generating signal for Option {option}...")
    model, meta, scaler = load_model(option)

    X_asset, X_macro, last_data_date, regime_context, stress, tickers = \
        _build_inference_tensors(option, master, meta)

    # Scale
    X_asset_s, X_macro_s = scaler.transform(X_asset, X_macro)

    # Build splines and run inference
    asset_path, macro_path = _build_splines(X_asset_s, X_macro_s)
    with torch.no_grad():
        mu, sigma = model(asset_path, macro_path)

    mu_arr    = mu.numpy()[0]      # (n_assets,)
    sigma_arr = sigma.numpy()[0]   # (n_assets,)

    # Confidence = 1 / sigma, normalised to sum to 1
    raw_conf   = 1.0 / (sigma_arr + 1e-8)
    confidence = raw_conf / raw_conf.sum()

    # Per-ETF forecast dict
    forecasts = {
        tickers[i]: {
            "mu":         round(float(mu_arr[i]),    6),
            "sigma":      round(float(sigma_arr[i]), 6),
            "confidence": round(float(confidence[i]), 4),
        }
        for i in range(len(tickers))
    }

    # Top pick = highest mu (signal), tiebreak by confidence
    top_idx        = int(np.argmax(mu_arr))
    top_pick       = tickers[top_idx]
    top_mu         = float(mu_arr[top_idx])
    top_confidence = float(confidence[top_idx])

    signal_date = next_trading_day(last_data_date)

    # Backfill actual return if signal_date already in master
    actual_return, hit = _get_actual_return(top_pick, signal_date, master)

    print(f"  Option {option}: top_pick={top_pick} | mu={top_mu:.4f} | "
          f"confidence={top_confidence:.2%} | for {signal_date}")

    return {
        "option":          option,
        "option_name":     "Fixed Income / Alts" if option == "A" else "Equity Sectors",
        "signal_date":     signal_date,
        "last_data_date":  last_data_date,
        "generated_at":    datetime.utcnow().isoformat(),
        "top_pick":        top_pick,
        "top_mu":          round(top_mu, 6),
        "top_confidence":  round(top_confidence, 4),
        "forecasts":       forecasts,
        "regime_context":  regime_context,
        "macro_stress":    stress,
        "trained_at":      meta["trained_at"],
        "test_ann_return": meta.get("test_ann_return", 0),
        "test_sharpe":     meta.get("test_sharpe", 0),
        "test_ic":         meta.get("test_ic", 0),
        "model_n_params":  meta["n_params"],
        "actual_return":   actual_return,
        "hit":             hit,
    }


# ── History + upload ───────────────────────────────────────────────────────────

def _load_remote_history(option: str) -> list:
    try:
        local_path = hf_hub_download(
            repo_id=cfg.HF_SIGNALS_REPO,
            filename=f"signals/signal_history_{option}.json",
            repo_type="dataset",
            token=cfg.HF_TOKEN or None,
            local_dir=cfg.MODELS_DIR,
            local_dir_use_symlinks=False,
        )
        with open(local_path) as f:
            return json.load(f)
    except Exception:
        return []


def update_signal_history(signal: dict, option: str) -> None:
    history = _load_remote_history(option)

    record = {
        "signal_date":    signal["signal_date"],
        "top_pick":       signal["top_pick"],
        "top_mu":         signal["top_mu"],
        "top_confidence": signal["top_confidence"],
        "generated_at":   signal["generated_at"],
        "actual_return":  signal.get("actual_return"),
        "hit":            signal.get("hit"),
    }

    existing_dates = {r["signal_date"] for r in history}
    if record["signal_date"] not in existing_dates:
        history.append(record)
        print(f"[predict] Appended record for {record['signal_date']}")
    else:
        print(f"[predict] Record for {record['signal_date']} already exists — skipping")

    history_path = os.path.join(cfg.MODELS_DIR, f"signal_history_{option}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    try:
        api = HfApi(token=cfg.HF_TOKEN)
        with open(history_path, "rb") as f:
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=f"signals/signal_history_{option}.json",
                repo_id=cfg.HF_SIGNALS_REPO,
                repo_type="dataset",
                commit_message=f"Update signal history Option {option} ({record['signal_date']})",
            )
        print(f"[predict] Uploaded history for Option {option}")
    except Exception as e:
        print(f"[predict] WARNING: Failed to upload history Option {option}: {e}")


def save_signals(sig_A=None, sig_B=None) -> None:
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)

    combined = {
        "generated_at": datetime.utcnow().isoformat(),
        "option_A":     sig_A,
        "option_B":     sig_B,
    }

    local_combined = os.path.join(cfg.MODELS_DIR, "latest_signals.json")
    with open(local_combined, "w") as f:
        json.dump(combined, f, indent=2)

    try:
        api = HfApi(token=cfg.HF_TOKEN)
        with open(local_combined, "rb") as f:
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo="signals/latest_signals.json",
                repo_id=cfg.HF_SIGNALS_REPO,
                repo_type="dataset",
                commit_message=f"Update latest signals ({combined['generated_at']})",
            )
        print("[predict] Uploaded latest_signals.json")
    except Exception as e:
        print(f"[predict] WARNING: Failed to upload latest_signals.json: {e}")

    # Upload individual signal files and update history
    for sig, name, opt in [(sig_A, "signal_A", "A"), (sig_B, "signal_B", "B")]:
        if sig:
            local = os.path.join(cfg.MODELS_DIR, f"{name}.json")
            with open(local, "w") as f:
                json.dump(sig, f, indent=2)
            try:
                with open(local, "rb") as f:
                    api.upload_file(
                        path_or_fileobj=f,
                        path_in_repo=f"signals/{name}.json",
                        repo_id=cfg.HF_SIGNALS_REPO,
                        repo_type="dataset",
                        commit_message=f"Update {name} signal",
                    )
                print(f"[predict] Uploaded {name}.json")
            except Exception as e:
                print(f"[predict] WARNING: Failed to upload {name}.json: {e}")
            update_signal_history(sig, opt)

    print("\n[predict] Done.")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NCDE daily signals")
    parser.add_argument("--option", choices=["A", "B", "both"], default="both")
    args = parser.parse_args()

    print("[predict] Loading master dataset...")
    master = loader.load_master()

    sig_A = sig_B = None

    if args.option in ("A", "both"):
        sig_A = generate_signal("A", master)
    if args.option in ("B", "both"):
        sig_B = generate_signal("B", master)

    save_signals(sig_A, sig_B)

    for sig, label in [(sig_A, "A"), (sig_B, "B")]:
        if sig:
            print(f"  Option {label}: {sig['top_pick']} on {sig['signal_date']} "
                  f"(mu={sig['top_mu']:.4f}, confidence={sig['top_confidence']:.2%})")
