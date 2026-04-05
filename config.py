# config.py — Master configuration for P2-ETF-NCDE-ENGINE
#
# Neural Controlled Differential Equation engine for next-day ETF return forecasting.
# Reads source data from P2SAMAPA/p2-etf-deepm-data (maintained by DeePM repo).
# Writes signals/models to P2SAMAPA/p2-etf-ncde-engine-signals.

import os
from datetime import date

# ── HuggingFace ────────────────────────────────────────────────────────────────
HF_SOURCE_REPO  = os.environ.get("HF_SOURCE_REPO",  "P2SAMAPA/p2-etf-deepm-data")
HF_SIGNALS_REPO = os.environ.get("HF_SIGNALS_REPO", "P2SAMAPA/p2-etf-ncde-engine-signals")
HF_TOKEN        = os.environ.get("HF_TOKEN", "")

# ── Option A — Fixed Income / Alternatives ─────────────────────────────────────
FI_ETFS = [
    "TLT",  # 20+ Year Treasury Bond
    "LQD",  # Investment Grade Corporate Bond
    "HYG",  # High Yield Corporate Bond
    "VNQ",  # Real Estate (REITs)
    "GLD",  # Gold
    "SLV",  # Silver
    "PFF",  # Preferred Stock
    "MBB",  # Mortgage-Backed Securities
]
FI_BENCHMARK = "AGG"

# ── Option B — Equity Sectors ──────────────────────────────────────────────────
EQ_ETFS = [
    "SPY",  # S&P 500
    "QQQ",  # NASDAQ 100
    "XLK",  # Technology
    "XLF",  # Financials
    "XLE",  # Energy
    "XLV",  # Health Care
    "XLI",  # Industrials
    "XLY",  # Consumer Discretionary
    "XLP",  # Consumer Staples
    "XLU",  # Utilities
    "GDX",  # Gold Miners
    "XME",  # Metals & Mining
]
EQ_BENCHMARK = "SPY"

# ── Source dataset file paths (read from HF_SOURCE_REPO) ──────────────────────
FILE_MASTER        = "data/master.parquet"
FILE_ETF_OHLCV     = "data/etf_ohlcv.parquet"
FILE_ETF_RETURNS   = "data/etf_returns.parquet"
FILE_MACRO_FRED    = "data/macro_fred.parquet"
FILE_MACRO_DERIVED = "data/macro_derived.parquet"

# ── FRED macro column names (as they appear in master.parquet) ─────────────────
FRED_SERIES = {
    "VIX":       "VIXCLS",
    "T10Y2Y":    "T10Y2Y",
    "DGS10":     "DGS10",
    "DTB3":      "DTB3",
    "HY_SPREAD": "BAMLH0A0HYM2",
    "IG_SPREAD": "BAMLC0A0CM",
    "USD_INDEX": "DTWEXBGS",
    "WTI_OIL":   "DCOILWTICO",
}
MACRO_CORE = ["VIX", "T10Y2Y", "HY_SPREAD", "USD_INDEX", "DTB3"]

# ── Train / val / test / live split ───────────────────────────────────────────
# Full history from 2008 used. Chronological, no shuffle.
# 80% train / 10% val / 10% test — live period never touched during training.
TRAIN_SPLIT = 0.80
VAL_SPLIT   = 0.10
# test = remaining 0.10
TRAIN_END   = "2024-12-31"
LIVE_START  = "2025-01-01"

# ── Feature engineering ────────────────────────────────────────────────────────
LOOKBACK        = 30    # trading days — halved from 60; cuts ODE integration time ~50%
VOL_WINDOW      = 21    # realised vol window
ZSCORE_WINDOW   = 63    # rolling z-score window (~1 quarter)
RETURN_WINDOWS  = [1, 5, 21, 63]

# ── NCDE model architecture ───────────────────────────────────────────────────
HIDDEN_DIM       = 32   # reduced from 64; vector field MLP runs at every ODE step
VECTOR_FIELD_DIM = 64   # reduced from 128
N_LAYERS         = 2    # depth of vector field MLP
DROPOUT          = 0.1
SOLVER           = "euler"  # fixed-step; fast on CPU. Options: euler, midpoint, dopri5
ADJOINT          = False    # direct backprop faster than adjoint on CPU with small models
ODE_STEPS        = 10       # number of fixed steps for euler/midpoint solver

# ── Readout head ───────────────────────────────────────────────────────────────
# h(T) → MLP → (mu, log_sigma) per ETF
# confidence = 1 / sigma (normalised in predict.py)
READOUT_DIM      = 32   # reduced from 64

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE    = 64       # larger batch = fewer solver calls per epoch
MAX_EPOCHS    = 50       # 50 is sufficient; early stopping handles the rest
PATIENCE      = 8        # early stopping patience
LEARNING_RATE = 5e-4
WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 1.0

# ── Local runtime directories ──────────────────────────────────────────────────
MODELS_DIR = "models"
DATA_DIR   = "data"

# ── GitHub Actions schedule ────────────────────────────────────────────────────
# Daily predict runs at 23:00 UTC Mon-Fri (after DeePM data update at 22:00 UTC)
CRON_SCHEDULE = "0 23 * * 1-5"
