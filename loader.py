# loader.py — Loads source data from P2SAMAPA/p2-etf-deepm-data
#
# NCDE engine is a pure consumer of the shared dataset maintained by DeePM.
# No data download or upload happens here — only reads.

import os
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
import config as cfg


def _download(filename: str) -> str:
    """Download a file from the source HF dataset repo."""
    return hf_hub_download(
        repo_id=cfg.HF_SOURCE_REPO,
        filename=filename,
        repo_type="dataset",
        token=cfg.HF_TOKEN if cfg.HF_TOKEN else None,
        force_download=True,
    )


def _fix_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame has a proper timezone-naive DatetimeIndex."""
    for col in ["Date", "date", "DATE"]:
        if col in df.columns:
            df = df.set_index(col)
            break
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "Date"
    for col in list(df.columns):
        if isinstance(col, str) and col.lower() in ("date", "index", "level_0"):
            df = df.drop(columns=[col])
    return df.sort_index()


def load_master() -> pd.DataFrame:
    """Load the full aligned master parquet from DeePM dataset."""
    path = _download(cfg.FILE_MASTER)
    df = pd.read_parquet(path)
    df = _fix_index(df)
    print(f"[loader] Master: {df.shape}, {df.index[0].date()} -> {df.index[-1].date()}")
    return df


def get_option_data(option: str, master: pd.DataFrame) -> dict:
    """
    Extract Option A (FI) or Option B (Equity) data from master DataFrame.

    Returns dict with:
        tickers        — list of ETF tickers for this option
        benchmark      — benchmark ticker string
        prices         — Close prices per ticker
        returns        — simple daily returns
        log_returns    — log daily returns
        vol            — annualised realised volatility
        macro          — raw FRED macro features
        macro_derived  — engineered macro features
        benchmark_ret  — benchmark simple returns
        cash_rate      — daily T-bill rate series
    """
    if option == "A":
        tickers   = cfg.FI_ETFS
        benchmark = cfg.FI_BENCHMARK
    elif option == "B":
        tickers   = cfg.EQ_ETFS
        benchmark = cfg.EQ_BENCHMARK
    else:
        raise ValueError(f"option must be 'A' or 'B', got {option!r}")

    # Close prices
    price_cols = [f"{t}_Close" for t in tickers if f"{t}_Close" in master.columns]
    prices = master[price_cols].copy()
    prices.columns = [c.replace("_Close", "") for c in prices.columns]

    # Simple returns
    ret_cols = [f"{t}_ret" for t in tickers if f"{t}_ret" in master.columns]
    returns = master[ret_cols].copy()
    returns.columns = [c.replace("_ret", "") for c in returns.columns]

    # Log returns
    logret_cols = [f"{t}_logret" for t in tickers if f"{t}_logret" in master.columns]
    log_returns = master[logret_cols].copy()
    log_returns.columns = [c.replace("_logret", "") for c in log_returns.columns]

    # Volatility
    vol_cols = [f"{t}_vol" for t in tickers if f"{t}_vol" in master.columns]
    if vol_cols:
        vol = master[vol_cols].copy()
        vol.columns = [c.replace("_vol", "") for c in vol.columns]
    else:
        vol = log_returns.rolling(cfg.VOL_WINDOW).std() * np.sqrt(252)

    # Raw FRED macro
    macro_cols = [c for c in cfg.FRED_SERIES.keys() if c in master.columns]
    macro = master[macro_cols].copy()

    # Derived macro features
    derived_cols = [c for c in master.columns if any(
        c.startswith(p) for p in [
            "VIX_", "YC_", "DGS10_", "HY_", "IG_", "HY_IG",
            "credit_", "USD_", "OIL_", "TBILL_", "macro_stress",
        ]
    )]
    macro_derived = master[derived_cols].copy()

    # Benchmark returns
    bench_ret_col = f"{benchmark}_ret"
    if bench_ret_col in master.columns:
        benchmark_ret = master[bench_ret_col].rename(benchmark)
    else:
        benchmark_ret = master[f"{benchmark}_Close"].pct_change().rename(benchmark)

    # Cash rate (daily T-bill)
    if "TBILL_daily" in master.columns:
        cash_rate = master["TBILL_daily"]
    elif "DTB3" in master.columns:
        cash_rate = master["DTB3"] / 252 / 100
    else:
        cash_rate = pd.Series(0.0, index=master.index, name="TBILL_daily")

    # Drop rows with all-NaN prices
    valid_idx = prices.dropna(how="all").index
    prices        = prices.loc[valid_idx]
    returns       = returns.reindex(valid_idx)
    log_returns   = log_returns.reindex(valid_idx)
    vol           = vol.reindex(valid_idx)
    macro         = macro.reindex(valid_idx).ffill()
    macro_derived = macro_derived.reindex(valid_idx).ffill()
    benchmark_ret = benchmark_ret.reindex(valid_idx)
    cash_rate     = cash_rate.reindex(valid_idx).ffill().fillna(0.0)

    print(f"[loader] Option {option} ({len(tickers)} ETFs): "
          f"{len(prices)} days, {prices.index[0].date()} -> {prices.index[-1].date()}")

    return {
        "option":        option,
        "tickers":       tickers,
        "benchmark":     benchmark,
        "prices":        prices,
        "returns":       returns,
        "log_returns":   log_returns,
        "vol":           vol,
        "macro":         macro,
        "macro_derived": macro_derived,
        "benchmark_ret": benchmark_ret,
        "cash_rate":     cash_rate,
    }
