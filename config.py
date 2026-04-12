# config.py — Master configuration for P2-ETF-NCDE-ENGINE
#
# Neural Controlled Differential Equation engine for next-day ETF return forecasting.
# Reads source data from P2SAMAPA/p2-etf-deepm-data (maintained by DeePM repo).
# Writes signals/models to P2SAMAPA/p2-etf-ncde-engine-signals.
#
# CHANGES vs original:
#   Fix 2 — Option A gets more epochs (150) and higher patience (25) vs Option B (80/15).
#            FI universe is lower-signal; it needs more time to converge past noise.
#   Fix 6 — Option A gets ODE_STEPS=60 (vs 20 default) to halve integration step size
#            over the 90-day lookback. At 20 steps over 90 days each step = 4.5 days,
#            which is too coarse for rate-sensitive fixed income path dynamics.
#            Option B keeps ODE_STEPS=20 — equities converge fine at current resolution.

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
    "XLB",  # Basic Materials
    "XLRE", # Real Estate
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

TRAIN_SPLIT = 0.80
VAL_SPLIT   = 0.10
# test = remaining 0.10

TRAIN_END  = "2024-12-31"
LIVE_START = "2025-01-01"

# ── Feature engineering ────────────────────────────────────────────────────────

LOOKBACK        = 40   # default; overridden per-option in train.py (A=90, B=60)
VOL_WINDOW      = 21
ZSCORE_WINDOW   = 63
RETURN_WINDOWS  = [1, 5, 21, 63]

# ── NCDE model architecture ───────────────────────────────────────────────────

HIDDEN_DIM       = 48   # hidden state dimension
VECTOR_FIELD_DIM = 96   # intermediate dim inside vector field MLP
N_LAYERS         = 2    # depth of vector field MLP
DROPOUT          = 0.1
SOLVER           = "midpoint"
ADJOINT          = False

# Fix 6 — ODE_STEPS is now option-specific (set in train.py per option).
# This default is used only if train.py does not override it.
# Option A override → 60  (1.5 days/step over 90-day window)
# Option B override → 20  (3 days/step over 60-day window, sufficient for equity)
ODE_STEPS = 20

# ── Readout head ───────────────────────────────────────────────────────────────

READOUT_DIM = 48

# ── Training — shared defaults ─────────────────────────────────────────────────

BATCH_SIZE    = 32
LEARNING_RATE = 3e-4
WEIGHT_DECAY  = 1e-4
GRAD_CLIP     = 1.0

# Fix 2 — Option A gets its own MAX_EPOCHS and PATIENCE (set in train.py).
# These shared values are used as Option B defaults.
# Option A: MAX_EPOCHS=150, PATIENCE=25  ← more room to converge on noisy FI signals
# Option B: MAX_EPOCHS=80,  PATIENCE=15  ← unchanged from original
MAX_EPOCHS = 80
PATIENCE   = 15

# Per-option overrides (consumed by train.py — do not remove)
OPTION_A_MAX_EPOCHS = 150
OPTION_A_PATIENCE   = 25
OPTION_A_ODE_STEPS  = 60   # Fix 6 — finer integration for 90-day FI lookback
OPTION_A_LOOKBACK   = 90

OPTION_B_MAX_EPOCHS = 80
OPTION_B_PATIENCE   = 15
OPTION_B_ODE_STEPS  = 20
OPTION_B_LOOKBACK   = 60

# ── Local runtime directories ──────────────────────────────────────────────────

MODELS_DIR = "models"
DATA_DIR   = "data"

# ── GitHub Actions schedule ────────────────────────────────────────────────────

CRON_SCHEDULE = "0 23 * * 1-5"
