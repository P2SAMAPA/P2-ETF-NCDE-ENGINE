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

FILE_MASTER       = "data/master.parquet"
FILE_ETF_OHLCV    = "data/etf_ohlcv.parquet"
FILE_ETF_RETURNS  = "data/etf_returns.parquet"
FILE_MACRO_FRED   = "data/macro_fred.parquet"
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

TRAIN_END  = "2024-12-31"
LIVE_START = "2025-01-01"

# ── Feature engineering ────────────────────────────────────────────────────────

LOOKBACK       = 60   # 3 months of path — more path geometry for the NCDE to exploit
                      # (was 40; at 60 days the macro regime cycles become visible)
VOL_WINDOW     = 21   # realised vol window
ZSCORE_WINDOW  = 63   # rolling z-score window (~1 quarter)
RETURN_WINDOWS = [1, 5, 21, 63]

# ── NCDE model architecture ───────────────────────────────────────────────────
#
# Previous config: HIDDEN=48, VF_DIM=96, N_LAYERS=2 → ~20k params → underfitting
# New config: HIDDEN=96, VF_DIM=256, N_LAYERS=3 → ~120k params → appropriate capacity
#
# Rule of thumb for NCDEs: hidden_dim should be ~2-3x the input channel count.
# With ~30 input channels (asset path + macro), HIDDEN=96 is the minimum sensible.

HIDDEN_DIM        = 96   # hidden state dimension (was 48 — doubled)
VECTOR_FIELD_DIM  = 256  # intermediate dim inside vector field MLP (was 96 — 2.7x)
N_LAYERS          = 3    # depth of vector field MLP (was 2 — adds expressiveness)
DROPOUT           = 0.15 # slightly higher to regularise the larger model

# ODE solver settings
# midpoint is the right solver for CPU — keep it.
# ODE_STEPS: must be >= LOOKBACK/2 for meaningful path integration.
# Previously 20 steps over 40 timepoints = very coarse.
# Now 40 steps over 60 timepoints = still efficient but captures path curvature.
SOLVER    = "midpoint"  # 2nd-order, good accuracy/speed tradeoff on CPU
ADJOINT   = False       # direct backprop faster than adjoint on CPU
ODE_STEPS = 40          # integration steps over lookback window (was 20 — doubled)

# ── Readout head ───────────────────────────────────────────────────────────────
# h(T) → MLP → (mu, log_sigma) per ETF
# READOUT_DIM should be >= HIDDEN_DIM to avoid bottleneck before the output.
# Previously 48 (same as hidden) → bottleneck at the 48→24 layer.

READOUT_DIM = 128  # was 48; larger readout gives the model room to express
                   # cross-asset return structure

# ── Training ───────────────────────────────────────────────────────────────────
#
# Previous regime: 4-minute runs = model not actually training.
# Targets: ~45-90 min on GitHub Actions CPU. If it runs faster, the model
# is still not learning enough. Use the epoch count and loss curves to judge.
#
# Key changes vs previous config:
#   BATCH_SIZE: 32 → 64   (larger batches = more stable gradient on noisy daily returns)
#   MAX_EPOCHS: 80 → 150  (early stopping will still cut it short if needed)
#   PATIENCE: 15 → 20     (val_ann_return is noisy; need more patience)
#   LR: 3e-4 → 5e-4       (larger model needs faster initial learning)
#   WEIGHT_DECAY: 1e-4 → 2e-4  (slightly stronger L2 for larger model)
#   GRAD_CLIP: 1.0 → 0.5  (tighter clip — NCDE gradients can spike on path data)
#   LR scheduler: ReduceLROnPlateau patience 5 → 8

BATCH_SIZE    = 64    # was 32; larger batch = more stable gradient signal
MAX_EPOCHS    = 150   # was 80; let the model actually converge
PATIENCE      = 20    # was 15; val_ann_return is noisy, need more patience
LEARNING_RATE = 5e-4  # was 3e-4; faster warmup for larger model
WEIGHT_DECAY  = 2e-4  # was 1e-4; slightly stronger regularisation
GRAD_CLIP     = 0.5   # was 1.0; tighter clipping for NCDE stability

# LR scheduler patience (used in train.py ReduceLROnPlateau)
LR_SCHEDULER_PATIENCE = 8   # was 5; let loss plateau longer before halving LR

# ── Local runtime directories ──────────────────────────────────────────────────

MODELS_DIR = "models"
DATA_DIR   = "data"

# ── GitHub Actions schedule ────────────────────────────────────────────────────
# Daily predict runs at 23:00 UTC Mon-Fri (after DeePM data update at 22:00 UTC)

CRON_SCHEDULE = "0 23 * * 1-5"
