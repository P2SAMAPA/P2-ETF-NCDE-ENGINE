# app.py — P2-ETF-NCDE-ENGINE Streamlit Dashboard
#
# Reads latest signals from P2SAMAPA/p2-etf-ncde-engine-signals HF dataset.
# Tab layout: Option A | Option B
# Shows: top pick + mu/confidence hero, per-ETF forecast bar chart,
#        regime context pills, signal history table.

import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from huggingface_hub import hf_hub_download
import pandas_market_calendars as mcal

import config as cfg

st.set_page_config(
    page_title="NCDE — ETF Signal Engine",
    page_icon="∂",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.stApp { background-color: #ffffff; }
.hero-card {
    background: #f0f4ff; border: 1px solid #d0d8f0;
    border-radius: 14px; padding: 28px 32px 22px 32px; margin-bottom: 24px;
}
.hero-ticker  { font-size: 64px; font-weight: 700; color: #1a1a2e; line-height: 1.1; }
.hero-mu      { font-size: 28px; font-weight: 500; color: #3a5bd9; margin-top: 6px; }
.hero-date    { font-size: 15px; color: #6b7280; margin-top: 8px; }
.hero-conf    { font-size: 14px; color: #3a5bd9; font-weight: 600;
                background: #e0e7ff; border-radius: 20px; padding: 3px 12px;
                display: inline-block; margin-top: 8px; }
.runner-up    { font-size: 18px; color: #374151; margin-top: 14px;
                padding-top: 14px; border-top: 1px solid #e5e7eb; }
.metric-row   { display:flex; gap:12px; margin:10px 0 16px 0; }
.metric-box   { flex:1; background:#fff; border:1px solid #e5e7eb;
                border-radius:10px; padding:14px 10px; text-align:center; }
.metric-label { font-size:12px; color:#6b7280; text-transform:uppercase;
                letter-spacing:.05em; margin-bottom:6px; }
.metric-value { font-size:24px; font-weight:600; color:#111827; }
.pos { color:#059669; } .neg { color:#dc2626; }
.pill { display:inline-block; padding:5px 14px; border-radius:20px;
        font-size:15px; font-weight:500; margin:3px 3px 3px 0; }
.pill-g { background:#d1fae5; color:#065f46; }
.pill-a { background:#fef3c7; color:#92400e; }
.pill-r { background:#fee2e2; color:#991b1b; }
.hit-line { font-size:16px; color:#374151; margin-bottom:10px; }
.fn { font-size:13px; color:#9ca3af; margin-top:8px; }
.sec-hdr { font-size:20px; font-weight:700; color:#1a1a2e; margin:28px 0 12px 0; }
</style>
""", unsafe_allow_html=True)

nyse = mcal.get_calendar("NYSE")


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_signals() -> dict:
    try:
        path = hf_hub_download(
            repo_id=cfg.HF_SIGNALS_REPO,
            filename="signals/latest_signals.json",
            repo_type="dataset",
            token=cfg.HF_TOKEN or None,
            force_download=True,
        )
        with open(path) as f:
            raw = json.load(f)
        return {
            "A": raw.get("option_A") or {},
            "B": raw.get("option_B") or {},
        }
    except Exception as e:
        st.error(f"Could not load signals: {e}")
        return {"A": {}, "B": {}}


@st.cache_data(ttl=3600)
def load_master() -> pd.DataFrame:
    try:
        path = hf_hub_download(
            repo_id=cfg.HF_SOURCE_REPO,
            filename=cfg.FILE_MASTER,
            repo_type="dataset",
            token=cfg.HF_TOKEN or None,
            force_download=True,
        )
        df = pd.read_parquet(path)
        for col in ["Date", "date"]:
            if col in df.columns:
                df = df.set_index(col)
                break
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df.sort_index()
    except Exception as e:
        st.error(f"Could not load master dataset: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_history(option: str) -> pd.DataFrame:
    try:
        path = hf_hub_download(
            repo_id=cfg.HF_SIGNALS_REPO,
            filename=f"signals/signal_history_{option}.json",
            repo_type="dataset",
            token=cfg.HF_TOKEN or None,
            force_download=True,
        )
        with open(path) as f:
            return pd.DataFrame(json.load(f))
    except Exception:
        return pd.DataFrame()


def next_trading_day(date: pd.Timestamp) -> pd.Timestamp:
    sched = nyse.schedule(start_date=date, end_date=date + pd.Timedelta(days=10))
    days  = sched.index[sched.index > date]
    return days[0] if len(days) > 0 else date + pd.Timedelta(days=1)


# ── UI helpers ─────────────────────────────────────────────────────────────────

def pill(label, val, lo, hi):
    cls = "pill-g" if val < lo else ("pill-r" if val > hi else "pill-a")
    return f'<span class="pill {cls}">{label}: {val}</span>'


def render_hero(signal: dict, master: pd.DataFrame):
    if not signal or "top_pick" not in signal:
        st.info("Signal not available yet — run the training workflow first.")
        return

    tickers   = cfg.FI_ETFS if signal.get("option") == "A" else cfg.EQ_ETFS
    forecasts = signal.get("forecasts", {})

    # Sort by mu descending
    ranked = sorted(
        [(t, forecasts[t]["mu"], forecasts[t]["confidence"])
         for t in tickers if t in forecasts],
        key=lambda x: x[1], reverse=True,
    )

    t1 = ranked[0] if len(ranked) > 0 else (signal["top_pick"], signal["top_mu"], signal["top_confidence"])
    t2 = ranked[1] if len(ranked) > 1 else None
    t3 = ranked[2] if len(ranked) > 2 else None

    next_day = str(next_trading_day(master.index[-1]).date()) \
        if not master.empty else signal.get("signal_date", "—")

    gen = signal.get("generated_at", "")
    try:
        gen = datetime.fromisoformat(gen).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pass

    runner = ""
    if t2:
        runner += f"<span style='color:#6b7280'>2nd:</span> <b>{t2[0]}</b> μ={t2[1]:.4f}"
    if t3:
        runner += f"&nbsp;&nbsp;<span style='color:#6b7280'>3rd:</span> <b>{t3[0]}</b> μ={t3[1]:.4f}"

    rc   = signal.get("regime_context", {})
    st_  = signal.get("macro_stress")
    pills = ""
    if rc.get("VIX"):        pills += pill("VIX",    rc["VIX"],       15,   25)
    if rc.get("T10Y2Y"):     pills += pill("T10Y2Y", rc["T10Y2Y"],  -0.5,  0.5)
    if rc.get("HY_SPREAD"):  pills += pill("HY Spr", rc["HY_SPREAD"], 300, 500)
    if st_ is not None:      pills += pill("Stress",  st_,           -0.5,  0.5)

    st.markdown(f"""
<div class="hero-card">
  <div class="hero-ticker">{t1[0]}</div>
  <div class="hero-mu">μ = {t1[1]:.4f}</div>
  <div class="hero-date">Signal for {next_day} &nbsp;·&nbsp; Generated {gen}</div>
  <div class="hero-conf">Confidence {t1[2]:.1%}</div>
  <div class="runner-up">{runner}</div>
  <div style="margin-top:16px">{pills}</div>
</div>
""", unsafe_allow_html=True)


def render_model_metrics(signal: dict):
    if not signal:
        return
    fp = lambda v: f"{v*100:.1f}%"
    c  = lambda v: "pos" if v >= 0 else "neg"
    r  = signal.get("test_ann_return", 0)
    s  = signal.get("test_sharpe",     0)
    ic = signal.get("test_ic",         0)
    st.markdown(f"""
<div class="metric-row">
  <div class="metric-box">
    <div class="metric-label">Test Ann Return</div>
    <div class="metric-value {c(r)}">{fp(r)}</div>
  </div>
  <div class="metric-box">
    <div class="metric-label">Test Sharpe</div>
    <div class="metric-value {c(s)}">{s:.2f}</div>
  </div>
  <div class="metric-box">
    <div class="metric-label">Test IC</div>
    <div class="metric-value {c(ic)}">{ic:.3f}</div>
  </div>
</div>
""", unsafe_allow_html=True)


def render_forecast_chart(signal: dict, key: str = ""):
    """Horizontal bar chart of μ per ETF with error bars (±σ)."""
    if not signal or "forecasts" not in signal:
        return

    forecasts = signal["forecasts"]
    tickers   = cfg.FI_ETFS if signal.get("option") == "A" else cfg.EQ_ETFS
    tickers   = [t for t in tickers if t in forecasts]

    mus    = [forecasts[t]["mu"]    for t in tickers]
    sigmas = [forecasts[t]["sigma"] for t in tickers]
    confs  = [forecasts[t]["confidence"] for t in tickers]
    colors = ["#3a5bd9" if t == signal["top_pick"] else "#9ca3af" for t in tickers]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=mus, y=tickers,
        orientation="h",
        marker_color=colors,
        error_x=dict(type="data", array=sigmas, visible=True, color="#d1d5db"),
        customdata=[[c] for c in confs],
        hovertemplate="<b>%{y}</b><br>μ = %{x:.4f}<br>σ = %{error_x.array:.4f}<br>conf = %{customdata[0]:.1%}<extra></extra>",
    ))
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#9ca3af")
    fig.update_layout(
        height=max(250, len(tickers) * 32),
        margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(title="Predicted next-day return (μ)", showgrid=True, gridcolor="#f3f4f6"),
        yaxis=dict(showgrid=False),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False}, key=f"fcst_{key}")


def render_history(hist_df: pd.DataFrame, master: pd.DataFrame):
    if hist_df.empty:
        st.info("Signal history will appear after the first prediction run.")
        return

    # Backfill actual returns from master
    if not master.empty and "actual_return" not in hist_df.columns:
        def get_ret(row):
            try:
                date = pd.Timestamp(row["signal_date"])
                col  = f"{row['top_pick']}_ret"
                if col in master.columns and date in master.index:
                    return master.loc[date, col]
            except Exception:
                pass
            return np.nan
        hist_df["actual_return"] = hist_df.apply(get_ret, axis=1)

    # Coerce actual_return to numeric so isnan checks are safe
    if "actual_return" in hist_df.columns:
        hist_df["actual_return"] = pd.to_numeric(hist_df["actual_return"], errors="coerce")

    if "hit" not in hist_df.columns and "actual_return" in hist_df.columns:
        hist_df["hit"] = hist_df["actual_return"].apply(
            lambda x: "✓" if (pd.notna(x) and x > 0) else ("✗" if pd.notna(x) else "—")
        )

    disp = hist_df.sort_values("signal_date", ascending=False).copy()

    col_map = {
        "signal_date":    "Date",
        "top_pick":       "Pick",
        "top_mu":         "μ",
        "top_confidence": "Confidence",
        "actual_return":  "Actual Return",
        "hit":            "Hit",
    }
    cols = [c for c in col_map if c in disp.columns]
    disp = disp[cols].rename(columns=col_map)

    if "Confidence" in disp.columns:
        disp["Confidence"] = disp["Confidence"].apply(lambda x: f"{x*100:.1f}%")

    if "Actual Return" in disp.columns:
        disp["Actual Return"] = pd.to_numeric(disp["Actual Return"], errors="coerce")
        disp["Actual Return"] = disp["Actual Return"].apply(
            lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—"
        )

    if "Hit" in disp.columns:
        hits  = (disp["Hit"] == "✓").sum()
        total = disp["Hit"].isin(["✓", "✗"]).sum()
        hr    = hits / total if total > 0 else 0
        st.markdown(
            f"<div class='hit-line'>Hit rate: <b>{hr:.1%}</b> &nbsp;({hits}/{total} signals)</div>",
            unsafe_allow_html=True,
        )

    st.dataframe(disp, use_container_width=True, hide_index=True,
                 column_config={
                     "Hit":  st.column_config.TextColumn(width="small"),
                     "Pick": st.column_config.TextColumn(width="small"),
                 })


def render_footnote(signal: dict):
    if not signal:
        return
    trained = signal.get("trained_at", "—")
    try:
        trained = datetime.fromisoformat(trained).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        pass
    n_params = signal.get("model_n_params", 0)
    st.markdown(
        f"<div class='fn'>Trained {trained} &nbsp;·&nbsp; "
        f"Params: {n_params:,} &nbsp;·&nbsp; "
        f"Solver: {cfg.SOLVER} &nbsp;·&nbsp; "
        f"Lookback: {cfg.LOOKBACK}d</div>",
        unsafe_allow_html=True,
    )


# ── Option renderer ────────────────────────────────────────────────────────────

def render_option(option: str, signals: dict, master: pd.DataFrame):
    signal  = signals.get(option, {})
    hist    = load_history(option)

    render_hero(signal, master)
    render_model_metrics(signal)

    st.markdown("<div class='sec-hdr'>ETF Forecasts (μ ± σ)</div>", unsafe_allow_html=True)
    st.caption("Bar = predicted next-day return (μ). Error bar = uncertainty (σ). Blue = top pick.")
    render_forecast_chart(signal, key=option)

    st.markdown("<div class='sec-hdr'>Signal History</div>", unsafe_allow_html=True)
    render_history(hist, master)
    render_footnote(signal)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    st.markdown(
        "<h2 style='margin-bottom:2px;color:#1a1a2e;font-size:34px;'>"
        "NCDE — Neural CDE ETF Signal Engine</h2>"
        "<p style='color:#6b7280;font-size:16px;margin-top:0;'>"
        "Continuous-time &nbsp;·&nbsp; Controlled Differential Equations "
        "&nbsp;·&nbsp; Macro control path &nbsp;·&nbsp; μ + σ forecasts</p>",
        unsafe_allow_html=True,
    )

    with st.spinner("Loading signals and data..."):
        signals = load_signals()
        master  = load_master()

    tab_a, tab_b = st.tabs([
        "📊 Option A — Fixed Income / Alts",
        "📈 Option B — Equity Sectors",
    ])

    with tab_a:
        render_option("A", signals, master)

    with tab_b:
        render_option("B", signals, master)

    st.markdown(
        "<div style='margin-top:40px;padding-top:16px;border-top:1px solid #e5e7eb;"
        "font-size:13px;color:#9ca3af;text-align:center;'>"
        "P2-ETF-NCDE-ENGINE &nbsp;·&nbsp; Research only &nbsp;·&nbsp; Not financial advice"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
