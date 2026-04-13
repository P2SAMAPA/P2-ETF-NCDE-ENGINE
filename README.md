# P2-ETF-NCDE-ENGINE

**Neural Controlled Differential Equations for ETF Return Forecasting**  
with **Conformal Prediction Intervals** (guaranteed coverage)

Continuous-time regime-aware signal engine for Fixed Income / Alternatives and Equity Sectors.
Outputs next-day return forecasts (μ) with calibrated uncertainty (σ) per ETF —
and, optionally, *provably correct* prediction intervals via split conformal prediction.

---

## What is this?

Standard DL engines (LSTM, Mamba, Transformer) treat time series as discrete sequences.
NCDEs instead model the *continuous path* of your data using controlled differential equations:

```
dh(t) = f(h(t), X(t)) dX(t)
```

Where:

* `h(t)` is the hidden state evolving continuously over time
* `X(t)` is a cubic spline interpolation of your ETF + macro features
* `f` is a learned vector field (MLP) — the neural part
* `dX(t)` is the "control" — macro series (VIX, T10Y2Y, spreads) drive the dynamics

**Why this matters for ETFs:**

* Macro data arrives irregularly — NCDEs handle this natively
* Regime shifts are captured as path geometry, not just point values
* Outputs calibrated (μ, σ) per ETF — confidence is principled, not heuristic

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
        │
        ▼
   conformal/            ← split conformal wrapper (OPTIONAL — see below)
   calibrate.py          ← computes q̂ on val set  →  [μ ± q̂·σ] intervals
```

---

## Conformal Prediction Module  *(new — separate, non-invasive)*

> The existing NCDE engine is **completely unchanged**.  
> The conformal module reads the NCDE outputs and adds a layer on top.

### What is split conformal prediction?

The NCDE outputs (μ, σ) per ETF. The σ is Gaussian-NLL-trained and informative,
but there is no *guarantee* that `[μ − σ, μ + σ]` contains the true return with any
specific probability.

**Split conformal prediction** (Angelopoulos & Bates, 2022) turns this into a
*finite-sample, distribution-free* guarantee:

1. Use the val set (10% of history, never seen during training) as a **calibration set**.
2. For each calibration sample compute the **nonconformity score**:  
   `s = |y − μ| / σ`
3. At coverage level `1−α`, compute the conformal quantile:  
   `q̂ = ⌈(n+1)(1−α)⌉/n -th empirical quantile of {s₁, …, sₙ}`
4. Prediction interval for any new ETF:  `[μ − q̂·σ,  μ + q̂·σ]`

**Guarantee:** for any new exchangeable draw, `P(y ∈ interval) ≥ 1−α` holds
*without any distributional assumption* — even if the NCDE is mis-specified.

Three coverage levels are calibrated: **90% / 80% / 70%**.

### New files

```
conformal/
├── __init__.py              ← module docstring
├── calibrate.py             ← Step 1: compute q̂ from val-set residuals
├── predict_conformal.py     ← Step 2: wrap daily NCDE signals with intervals
└── app_conformal.py         ← Streamlit dashboard (NCDE vs NCDE+Conformal)
```

No existing file is modified.  
New HF paths: `conformal/conformal_option{A|B}.json` and  
`conformal/latest_signals_conformal.json` (same signals repo).

### Conformal signal schema

```json
{
  "option": "A",
  "signal_date": "2025-06-02",
  "top_pick": "GLD",
  "top_mu": 0.0042,
  "top_confidence": 0.18,
  "top_interval_90": {"lo": -0.0021, "hi": 0.0105, "width": 0.0126},
  "n_cal": 470,
  "cal_period": "2022-01-03 → 2023-06-30",
  "coverage_diagnostics": {
    "0.9": {"target": 0.9, "achieved": 0.912, "pooled": 0.912}
  },
  "conformal_forecasts": {
    "GLD": {
      "mu": 0.0042,
      "sigma": 0.0089,
      "confidence": 0.18,
      "q_hat": {"0.9": 1.42, "0.8": 1.05, "0.7": 0.82},
      "intervals": {
        "0.9": {"lo": -0.0084, "hi": 0.0168, "width": 0.0252},
        "0.8": {"lo": -0.0052, "hi": 0.0136, "width": 0.0188},
        "0.7": {"lo": -0.0031, "hi": 0.0115, "width": 0.0146}
      }
    }
  }
}
```

---

## Shared Dataset

Reads from `P2SAMAPA/p2-etf-deepm-data` (maintained by the DeePM repo).  
Writes NCDE signals to `P2SAMAPA/p2-etf-ncde-engine-signals`.  
Writes conformal signals to `P2SAMAPA/p2-etf-ncde-engine-signals/conformal/`.

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
| Val   | 10%   | Early stopping + conformal calibration set |
| Test  | 10%   | Held-out OOS evaluation |
| Live  | —     | 2025-01-01 → today, never touched during training |

> The conformal module reuses the **val set** as its calibration set — the same
> chronological slice that was held out during training.  No additional data is consumed.

---

## Setup

### Secrets required

| Secret    | Purpose |
|-----------|---------|
| `HF_TOKEN` | HuggingFace read/write token |

### 1. Validate the dataset

```bash
pip install -r requirements_train.txt
export HF_TOKEN=hf_...
python validate_dataset.py
```

### 2. Train the NCDE model

```bash
# Via GitHub Actions (recommended):
Actions → Train NCDE Models → Run workflow → option: both

# Or locally:
python train.py --option both
```

Training takes ~30–90 minutes per option on CPU (GitHub Actions free tier).

### 3. Calibrate the conformal wrapper  *(run once after training)*

```bash
python -m conformal.calibrate --option both
```

This reads the frozen model weights, runs inference on the val set, computes the
nonconformity scores, and saves `models/conformal_option{A|B}.json`.  
Upload to HF is automatic if `HF_TOKEN` is set.

Expected output:

```
  α=90%  target≥90%  achieved=91.2%  ✓  (pooled q̂=1.42)
  α=80%  target≥80%  achieved=82.1%  ✓  (pooled q̂=1.05)
  α=70%  target≥70%  achieved=71.8%  ✓  (pooled q̂=0.82)
```

### 4. Daily signals — NCDE + conformal

Daily predict workflow (GitHub Actions) runs automatically in sequence:

```
23:00  predict.py                     → signals/latest_signals.json
23:05  conformal/predict_conformal.py → conformal/latest_signals_conformal.json
```

Or locally:

```bash
python predict.py --option both
python -m conformal.predict_conformal --option both
```

### 5. Streamlit dashboards

**Original NCDE dashboard** (unchanged):

```bash
streamlit run app.py
```

**Conformal comparison dashboard** (new):

```bash
streamlit run conformal/app_conformal.py
```

Both use the same `HF_TOKEN` secret and can be deployed as separate Streamlit apps.

---

## GitHub Actions workflows

| Workflow                  | Schedule       | What it does |
|---------------------------|----------------|--------------|
| `train_ncde.yml`          | Manual trigger | Train Option A/B models |
| `daily_predict.yml`       | 23:00 UTC M–F  | Run `predict.py` → upload NCDE signals |
| `daily_conformal.yml`     | 23:05 UTC M–F  | Run `predict_conformal.py` → upload conformal signals |

`daily_conformal.yml` depends on `daily_predict.yml` completing first (use `needs:` or a 5-minute offset).

---

## Output schemas

### NCDE signal (`signals/latest_signals.json`)

```json
{
  "generated_at": "2025-06-01T23:05:00",
  "option_A": {
    "top_pick": "GLD",
    "top_mu": 0.0042,
    "top_confidence": 0.18,
    "forecasts": {
      "GLD": {"mu": 0.0042, "sigma": 0.0089, "confidence": 0.18}
    },
    "regime_context": {"VIX": 18.4, "T10Y2Y": -0.12},
    "signal_date": "2025-06-02",
    "actual_return": null,
    "hit": null
  }
}
```

### Conformal signal (`conformal/latest_signals_conformal.json`)

```json
{
  "generated_at": "2025-06-01T23:06:00",
  "option_A": {
    "top_pick": "GLD",
    "top_mu": 0.0042,
    "top_interval_90": {"lo": -0.0084, "hi": 0.0168, "width": 0.0252},
    "n_cal": 470,
    "cal_period": "2022-01-03 → 2023-06-30",
    "coverage_diagnostics": {"0.9": {"target": 0.9, "achieved": 0.912}},
    "conformal_forecasts": { ... }
  }
}
```

---

## Key papers

* Kidger et al., *Neural Controlled Differential Equations for Irregular Time Series* (NeurIPS 2020)
* Kidger et al., *Neural SDEs as Infinite-Dimensional GANs* (ICML 2021)
* Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction* (2022)  
  [arxiv.org/abs/2107.07511](https://arxiv.org/abs/2107.07511)
* Vovk, Gammerman & Shafer, *Algorithmic Learning in a Random World* (2005)
* `torchcde` library: https://github.com/patrick-kidger/torchcde

---

## Disclaimer

Research and educational purposes only. Not financial advice.
