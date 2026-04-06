# config.py — Master configuration for P2-ETF-NCDE-ENGINE
#
# KEY DESIGN: Option A (Fixed Income) and Option B (Equity) have SEPARATE
# architecture and training configs. FI ETFs are macro-driven and low-volatility
# — they need a smaller, more regularised model with a longer lookback.
# Equity sectors are momentum/sentiment-driven and benefit from a larger model.
#
# Evidence from training logs:
#   Old small model (400k params, lookback=40):  A: IC=0.042, Sharpe=1.84  B: IC=0.004, Sharpe=1.14
#   New large model (2-3M params, lookback=60):  A: IC=-0.005, Sharpe=1.47 B: IC=0.045, Sharpe=1.48
#
# Conclusion: FI needs a compact model + longer lookback. Equity needs the large model.

import os

# ── HuggingFace ────────────────────────────────────────────────────────────────

HF_SOURCE_REPO  = os.environ.get("HF_SOURCE_REPO",  "P2SAMAPA/p2-etf-deepm-data")
HF_SIGNALS_REPO = os.environ.get("HF_SIGNALS_REPO", "P2SAMAPA/p2-etf-ncde-engine-signals")
HF_TOKEN        = os.environ.get("HF_TOKEN", "")

# ── ETF universes ──────────────────────────────────────────────────────────────

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

# ── Source dataset file paths ──────────────────────────────────────────────────

FILE_MASTER        = "data/master.parquet"
FILE_ETF_OHLCV     = "data/etf_ohlcv.parquet"
FILE_ETF_RETURNS   = "data/etf_returns.parquet"
FILE_MACRO_FRED    = "data/macro_fred.parquet"
FILE_MACRO_DERIVED = "data/macro_derived.parquet"

# ── FRED macro columns ─────────────────────────────────────────────────────────

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

# ── Train / val / test split ───────────────────────────────────────────────────

TRAIN_SPLIT = 0.80
VAL_SPLIT   = 0.10
TRAIN_END   = "2024-12-31"
LIVE_START  = "2025-01-01"

# ── Feature engineering ────────────────────────────────────────────────────────

VOL_WINDOW     = 21
ZSCORE_WINDOW  = 63
RETURN_WINDOWS = [1, 5, 21, 63]

# ── Per-option configs ─────────────────────────────────────────────────────────
#
# Option A — Fixed Income / Alts
# --------------------------------
# Bond/commodity returns are dominated by macro (rates, spreads, USD).
# Returns are smaller in magnitude and more regime-persistent (slow mean reversion).
# The large model (2.1M params) learned sigma well but lost directional signal
# entirely (IC=-0.005). The old compact model (400k) kept IC=0.042, Sharpe=1.84.
# Fix: stay compact, add regularisation, extend lookback to 90 days so the model
# sees a full Fed cycle phase rather than just 2 months.
#
# Option B — Equity Sectors
# --------------------------
# Equity returns driven by momentum, risk-on/off, cross-sector rotation.
# Richer path geometry benefits from more ODE capacity.
# New large model was a clear improvement: IC=0.045, Sharpe=1.475, Ann=44%.
# Keep that architecture, minor readout widening for 12-asset output.

OPTION_CONFIGS = {
    "A": {
        # Architecture — compact, well-regularised
        "hidden_dim":             64,   # sweet spot: old 48 was too small, new 96 overfit NLL
        "vector_field_dim":      128,   # modest upgrade from old 96
        "n_layers":                2,   # shallow: FI dynamics are lower-frequency
        "readout_dim":            96,   # fixes old bottleneck (48→24) without over-widening
        "dropout":              0.20,   # stronger regularisation than equity model

        # ODE
        "solver":          "midpoint",
        "adjoint":              False,
        "ode_steps":               30,  # 30 steps over 90d path = 1 step per 3 days; fine for FI

        # Features
        "lookback":                90,  # 3× old lookback — captures rate cycle phases

        # Training
        "batch_size":              32,
        "max_epochs":             130,
        "patience":                25,  # FI val metrics noisy; be patient
        "learning_rate":         3e-4,
        "weight_decay":          4e-4,  # stronger L2 — key to keeping IC positive
        "grad_clip":              0.5,
        "lr_scheduler_patience":   10,
        "warmup_epochs":            5,
    },

    "B": {
        # Architecture — larger, proven to work
        "hidden_dim":             96,
        "vector_field_dim":      256,
        "n_layers":                3,
        "readout_dim":           160,   # slightly wider than previous 128 for 12-asset output
        "dropout":              0.12,   # equity model wasn't overfitting; ease off dropout

        # ODE
        "solver":          "midpoint",
        "adjoint":              False,
        "ode_steps":               40,

        # Features
        "lookback":                60,  # equity momentum signals decay fast; 60d is right

        # Training
        "batch_size":              64,
        "max_epochs":             150,
        "patience":                22,
        "learning_rate":         5e-4,
        "weight_decay":          2e-4,
        "grad_clip":              0.5,
        "lr_scheduler_patience":    8,
        "warmup_epochs":            5,
    },
}


def get(option: str, key: str):
    """Retrieve a per-option config value. e.g. cfg.get('A', 'hidden_dim')"""
    return OPTION_CONFIGS[option][key]


# ── Flat aliases ───────────────────────────────────────────────────────────────
# Used by Streamlit footnote and any code that reads cfg.SOLVER etc. directly.
# Always reflects Option B (equity) values as the display default.

HIDDEN_DIM            = OPTION_CONFIGS["B"]["hidden_dim"]
VECTOR_FIELD_DIM      = OPTION_CONFIGS["B"]["vector_field_dim"]
N_LAYERS              = OPTION_CONFIGS["B"]["n_layers"]
READOUT_DIM           = OPTION_CONFIGS["B"]["readout_dim"]
DROPOUT               = OPTION_CONFIGS["B"]["dropout"]
SOLVER                = OPTION_CONFIGS["B"]["solver"]
ADJOINT               = OPTION_CONFIGS["B"]["adjoint"]
ODE_STEPS             = OPTION_CONFIGS["B"]["ode_steps"]
LOOKBACK              = OPTION_CONFIGS["B"]["lookback"]
BATCH_SIZE            = OPTION_CONFIGS["B"]["batch_size"]
MAX_EPOCHS            = OPTION_CONFIGS["B"]["max_epochs"]
PATIENCE              = OPTION_CONFIGS["B"]["patience"]
LEARNING_RATE         = OPTION_CONFIGS["B"]["learning_rate"]
WEIGHT_DECAY          = OPTION_CONFIGS["B"]["weight_decay"]
GRAD_CLIP             = OPTION_CONFIGS["B"]["grad_clip"]
LR_SCHEDULER_PATIENCE = OPTION_CONFIGS["B"]["lr_scheduler_patience"]

# ── Directories ────────────────────────────────────────────────────────────────

MODELS_DIR = "models"
DATA_DIR   = "data"

# ── GitHub Actions schedule ────────────────────────────────────────────────────

CRON_SCHEDULE = "0 23 * * 1-5"
