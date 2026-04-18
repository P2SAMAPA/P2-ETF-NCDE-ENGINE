# predict.py — Daily signal generation for P2-ETF-NCDE-ENGINE
#
# SIMPLIFIED: Now saves only two separate files (signal_A.json and signal_B.json)
# Removed combined latest_signals.json to simplify the flow
#
# Output schema per option:
# signal_date, last_data_date, generated_at,
# forecasts: {ticker: {mu, sigma, confidence}},
# top_pick, top_mu, top_confidence,
# regime_context, macro_stress,
# actual_return, hit (backfilled next run)
#
# Usage:
# python predict.py --option both

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
    nyse = mcal.get_calendar("NYSE")
    base = pd.Timestamp(from_date) if from_date else pd.Timestamp.today()
    sched = nyse.schedule(
        start_date=base.strftime("%Y-%m-%d"),
        end_date=(base + pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
    )
    days = mcal.date_range(sched, frequency="1D").normalize().tz_localize(None)
    future = [d for d in days if d > base]
    return str(future[0].date()) if future else str((base + pd.Timedelta(days=1)).date())

def _get_actual_return(pick: str, signal_date: str, master: pd.DataFrame) -> tuple:
    try:
        date = pd.Timestamp(signal_date)
        col = f"{pick}_ret"
        if col in master.columns and date in master.index:
            ret = master.loc[date, col]
            if not np.isnan(ret):
                return float(ret), bool(ret > 0)
    except Exception:
        pass
    return None, None

def build_combined_path(X_asset: np.ndarray, X_macro: np.ndarray) -> torchcde.CubicSpline:
    """Concatenate asset+macro channel-wise then build a single CubicSpline."""
    Xa = torch.tensor(X_asset, dtype=torch.float32)
    Xm = torch.tensor(X_macro, dtype=torch.float32)
    X_combined = torch.cat([Xa, Xm], dim=-1)
    t = torch.arange(X_combined.shape[1], dtype=torch.float32)
    coeffs = torchcde.natural_cubic_coeffs(X_combined, t)
    return torchcde.CubicSpline(coeffs, t)

# ── Model loader ───────────────────────────────────────────────────────────────

def load_model(option: str) -> tuple:
    model_path = os.path.join(cfg.MODELS_DIR, f"ncde_option{option}_best.pt")
    meta_path = os.path.join(cfg.MODELS_DIR, f"meta_option{option}.json")
    scaler_path = os.path.join(cfg.MODELS_DIR, f"scaler_option{option}.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No trained model at {model_path}. Run train.py --option {option} first."
        )

    with open(meta_path) as f:
        meta = json.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    enriched_h0 = meta.get("config", {}).get("enriched_h0", False)
    if enriched_h0 is None:
        enriched_h0 = meta.get("config", {}).get("fix5_h0_enriched", False)

    print(f"[predict] Loading model for Option {option} (enriched_h0={enriched_h0})")

    model = NCDEForecaster(
        n_asset_path_dim=meta["n_asset_path_dim"],
        n_macro_feats=meta["n_macro_feats"],
        n_assets=meta["n_assets"],
        hidden_dim=meta["config"]["hidden_dim"],
        vector_field_dim=meta["config"]["vector_field_dim"],
        n_layers=meta["config"]["n_layers"],
        readout_dim=meta["config"]["readout_dim"],
        dropout=0.0,
        solver=meta["config"].get("solver", cfg.SOLVER),
        adjoint=meta["config"].get("adjoint", cfg.ADJOINT),
        ode_steps=meta["config"]["ode_steps"],
        lookback=meta["config"]["lookback"],
        enriched_h0=enriched_h0,
    ).to(DEVICE)

    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"[predict] Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"[predict] Model expects {meta['n_assets']} assets: {meta['tickers']}")

    return model, meta, scaler

# ── Inference ──────────────────────────────────────────────────────────────────

def _build_inference_tensors(option: str, master: pd.DataFrame, meta: dict) -> tuple:
    """Build scaled (1, T, F) arrays ready for spline construction."""
    tickers = meta["tickers"]
    lookback = meta["config"]["lookback"]
    
    data = loader.get_option_data(option, master)
    data["tickers"] = tickers

    asset_feat = feat.build_asset_features(data["log_returns"], data["vol"], tickers=tickers)
    macro_feat = feat.build_macro_features(data["macro"], data["macro_derived"])

    common_idx = asset_feat.index.intersection(macro_feat.index)
    af = asset_feat.reindex(common_idx).ffill().fillna(0.0)
    mf = macro_feat.reindex(common_idx).ffill().fillna(0.0)

    n_assets = len(tickers)
    asset_col_indices = []
    
    for ticker in tickers:
        cols = [c for c in af.columns if c.startswith(ticker + "_")]
        if len(cols) == 0:
            raise ValueError(f"No features found for ticker {ticker}")
        idxs = [af.columns.get_loc(c) for c in cols]
        asset_col_indices.append(idxs)

    n_asset_feats_per_ticker = [len(idxs) for idxs in asset_col_indices]
    if len(set(n_asset_feats_per_ticker)) > 1:
        print(f"[predict] Warning: Tickers have different feature counts")
        min_feats = min(n_asset_feats_per_ticker)
        asset_col_indices = [idxs[:min_feats] for idxs in asset_col_indices]
        n_asset_feats = min_feats
    else:
        n_asset_feats = n_asset_feats_per_ticker[0]

    af_window = af.iloc[-lookback:].values
    mf_window = mf.iloc[-lookback:].values

    X_asset = np.zeros((1, lookback, n_assets * n_asset_feats), dtype=np.float32)
    X_macro = mf_window[np.newaxis, :, :]

    for a, col_idxs in enumerate(asset_col_indices):
        start = a * n_asset_feats
        X_asset[0, :, start:start + n_asset_feats] = af_window[:, col_idxs]

    last_data_date = str(af.index[-1].date())

    latest_macro = data["macro"].iloc[-1]
    regime_context = {
        "VIX": round(float(latest_macro.get("VIX", 0)), 2),
        "T10Y2Y": round(float(latest_macro.get("T10Y2Y", 0)), 3),
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

    X_asset_s, X_macro_s = scaler.transform(X_asset, X_macro)

    X_path = build_combined_path(X_asset_s, X_macro_s)
    with torch.no_grad():
        mu, sigma = model(X_path)

    mu_arr = mu.numpy()[0]
    sigma_arr = sigma.numpy()[0]

    raw_conf = 1.0 / (sigma_arr + 1e-8)
    confidence = raw_conf / raw_conf.sum()

    forecasts = {
        tickers[i]: {
            "mu": round(float(mu_arr[i]), 6),
            "sigma": round(float(sigma_arr[i]), 6),
            "confidence": round(float(confidence[i]), 4),
        }
        for i in range(len(tickers))
    }

    top_idx = int(np.argmax(mu_arr))
    top_pick = tickers[top_idx]
    top_mu = float(mu_arr[top_idx])
    top_confidence = float(confidence[top_idx])

    signal_date = next_trading_day(last_data_date)

    actual_return, hit = _get_actual_return(top_pick, signal_date, master)

    print(f" Option {option}: top_pick={top_pick} | mu={top_mu:.4f} | "
          f"confidence={top_confidence:.2%} | for {signal_date}")

    return {
        "option": option,
        "option_name": "Fixed Income / Alts" if option == "A" else "Equity Sectors",
        "signal_date": signal_date,
        "last_data_date": last_data_date,
        "generated_at": datetime.utcnow().isoformat(),
        "top_pick": top_pick,
        "top_mu": round(top_mu, 6),
        "top_confidence": round(top_confidence, 4),
        "forecasts": forecasts,
        "regime_context": regime_context,
        "macro_stress": stress,
        "trained_at": meta["trained_at"],
        "test_ann_return": meta.get("test_ann_return", 0),
        "test_sharpe": meta.get("test_sharpe", 0),
        "test_ic": meta.get("test_ic", 0),
        "model_n_params": meta["n_params"],
        "actual_return": actual_return,
        "hit": hit,
    }

# ── History ────────────────────────────────────────────────────────────────────

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
        "signal_date": signal["signal_date"],
        "top_pick": signal["top_pick"],
        "top_mu": signal["top_mu"],
        "top_confidence": signal["top_confidence"],
        "generated_at": signal["generated_at"],
        "actual_return": signal.get("actual_return"),
        "hit": signal.get("hit"),
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

# ── Save signals ───────────────────────────────────────────────────────────────

def save_signal(signal: dict, option: str) -> None:
    """Save a single option signal to HF dataset repo."""
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)
    
    filename = f"signal_{option}.json"
    local_path = os.path.join(cfg.MODELS_DIR, filename)
    
    with open(local_path, "w") as f:
        json.dump(signal, f, indent=2)
    
    try:
        api = HfApi(token=cfg.HF_TOKEN)
        with open(local_path, "rb") as f:
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=f"signals/{filename}",
                repo_id=cfg.HF_SIGNALS_REPO,
                repo_type="dataset",
                commit_message=f"Update {filename} ({signal['signal_date']})",
            )
        print(f"[predict] Uploaded {filename}")
    except Exception as e:
        print(f"[predict] WARNING: Failed to upload {filename}: {e}")
    
    # Also update history
    update_signal_history(signal, option)

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NCDE daily signals")
    parser.add_argument("--option", choices=["A", "B", "both"], default="both")
    args = parser.parse_args()

    print("[predict] Loading master dataset...")
    master = loader.load_master()

    if args.option in ("A", "both"):
        sig_A = generate_signal("A", master)
        save_signal(sig_A, "A")
        print(f" Option A: {sig_A['top_pick']} on {sig_A['signal_date']} "
              f"(mu={sig_A['top_mu']:.4f}, confidence={sig_A['top_confidence']:.2%})")

    if args.option in ("B", "both"):
        sig_B = generate_signal("B", master)
        save_signal(sig_B, "B")
        print(f" Option B: {sig_B['top_pick']} on {sig_B['signal_date']} "
              f"(mu={sig_B['top_mu']:.4f}, confidence={sig_B['top_confidence']:.2%})")

    print("\n[predict] Done.")
