# P2-ETF-NCDE-ENGINE

**Neural Controlled Differential Equations for ETF Return Forecasting**

Continuous-time regime-aware signal engine for Fixed Income / Alternatives and Equity Sectors.
Outputs next-day return forecasts (μ) with calibrated uncertainty (σ) per ETF.

---

## What is this?

Standard DL engines (LSTM, Mamba, Transformer) treat time series as discrete sequences.
NCDEs instead model the *continuous path* of your data using controlled differential equations:

```
dh(t) = f(h(t), X(t)) dX(t)
```

Where:
- `h(t)` is the hidden state evolving continuously over time
- `X(t)` is a cubic spline interpolation of your ETF + macro features
- `f` is a learned vector field (MLP) — the neural part
- `dX(t)` is the "control" — macro series (VIX, T10Y2Y, spreads) drive the dynamics

**Why this matters for ETFs:**
- Macro data arrives irregularly — NCDEs handle this natively
- Regime shifts are captured as path geometry, not just point values
- Outputs calibrated (μ, σ) per ETF — confidence is principled, not heuristic

---

## Architecture

```
OHLCV + macro (from p2-etf-deepm-data)
        │
        ▼
   features.py          ← log returns, vol, momentum, macro derived
        │
        ▼
  cubic spline           ← torchcde.natural_cubic_coeffs()
   interpolation
        │
        ▼
   VectorField f         ← MLP: h → (hidden_dim × input_dim)
        │
        ▼
   ODE solver            ← torchdiffeq dopri5 + adjoint backprop
        │
        ▼
   h(T) terminal state
        │
        ▼
   ReadoutHead           ← MLP: h(T) → (μ, log_σ) per ETF
        │
        ▼
   (μ, σ) per ETF        ← Gaussian NLL trained
```

---

## Shared Dataset

Reads from `P2SAMAPA/p2-etf-deepm-data` (maintained by the DeePM repo).
Writes signals to `P2SAMAPA/p2-etf-ncde-engine-signals`.

**No data download or FRED API key required** — this engine is a pure consumer.

---

## ETF Universe

### Option A — Fixed Income / Alternatives (benchmark: AGG)

| Ticker | Description |
|--------|-------------|
| TLT    | 20+ Year Treasury Bond |
| LQD    | Investment Grade Corporate Bond |
| HYG    | High Yield Corporate Bond |
| VNQ    | Real Estate (REITs) |
| GLD    | Gold |
| SLV    | Silver |
| PFF    | Preferred Stock |
| MBB    | Mortgage-Backed Securities |

### Option B — Equity Sectors (benchmark: SPY)

| Ticker | Description |
|--------|-------------|
| SPY    | S&P 500 |
| QQQ    | NASDAQ 100 |
| XLK    | Technology |
| XLF    | Financials |
| XLE    | Energy |
| XLV    | Health Care |
| XLI    | Industrials |
| XLY    | Consumer Discretionary |
| XLP    | Consumer Staples |
| XLU    | Utilities |
| GDX    | Gold Miners |
| XME    | Metals & Mining |

---

## Train / Val / Test Split

| Split | Ratio | Notes |
|-------|-------|-------|
| Train | 80%   | 2008-01-01 → ~2022 |
| Val   | 10%   | Early stopping + hyperparam selection |
| Test  | 10%   | Held-out OOS evaluation |
| Live  | —     | 2025-01-01 → today, never touched during training |

---

## Setup

### Secrets required

| Secret | Purpose |
|--------|---------|
| `HF_TOKEN` | HuggingFace read/write token |

No FRED key needed — data is pre-built by the DeePM pipeline.

### 1. Validate the dataset

```bash
pip install -r requirements_train.txt
export HF_TOKEN=hf_...
python validate_dataset.py
```

### 2. Train models

```bash
# Via GitHub Actions (recommended):
Actions → Train NCDE Models → Run workflow → option: both

# Or locally:
python train.py --option both
```

Training takes ~30-90 minutes per option on CPU (GitHub Actions free tier).

### 3. Daily signals run automatically

`daily_predict.yml` runs at 23:00 UTC Mon-Fri (one hour after DeePM data update).

### 4. Streamlit dashboard

Connect `app.py` to Streamlit Community Cloud.
Set `HF_TOKEN` in Streamlit secrets.

---

## Output schema (signals/latest_signals.json)

```json
{
  "generated_at": "2025-06-01T23:05:00",
  "option_A": {
    "top_pick": "GLD",
    "top_mu": 0.0042,
    "top_confidence": 0.18,
    "forecasts": {
      "GLD": {"mu": 0.0042, "sigma": 0.0089, "confidence": 0.18},
      "TLT": {"mu": -0.0011, "sigma": 0.0120, "confidence": 0.12}
    },
    "regime_context": {"VIX": 18.4, "T10Y2Y": -0.12},
    "signal_date": "2025-06-02",
    "actual_return": null,
    "hit": null
  }
}
```

---

## Key papers

- Kidger et al., *Neural Controlled Differential Equations for Irregular Time Series* (NeurIPS 2020)
- Kidger et al., *Neural SDEs as Infinite-Dimensional GANs* (ICML 2021)
- `torchcde` library: https://github.com/patrick-kidger/torchcde

---

## Disclaimer

Research and educational purposes only. Not financial advice.
