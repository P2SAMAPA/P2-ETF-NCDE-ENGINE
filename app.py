# app.py — P2-ETF-NCDE-ENGINE Streamlit Dashboard
#
# SIMPLIFIED: Now reads from separate signal_A.json and signal_B.json files
# Removed dependency on combined latest_signals.json
#
# Tabs:
#   📊 Option A — Fixed Income / Alts
#   📈 Option B — Equity Sectors
#   ∂̂  Conformal — FI / Commodities
#   ∂̂  Conformal — Equities

import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
.hero        { font-size: 3rem; font-weight: 700; margin-bottom: 0.2rem; }
.mu          { font-size: 1.5rem; font-weight: 600; color: #3a5bd9; }
.conf        { font-size: 1rem; color: #6b7280; }
.runner      { font-size: 0.95rem; color: #374151; margin-top: 0.5rem; }
.pill-g      { color: #059669; font-weight: 500; }
.pill-r      { color: #dc2626; font-weight: 500; }
.pill-a      { color: #6b7280; }
.metric-box  { border: 1px solid #e5e7eb; border-radius: 6px;
               padding: 0.5rem 0.75rem; margin-right: 0.5rem;
               display: inline-block; }
.metric-pos  { color: #059669; font-weight: 600; }
.metric-neg  { color: #dc2626; font-weight: 600; }
.badge-g     { background:#d1fae5; color:#065f46; border-radius:4px;
               padding:2px 8px; font-size:0.82rem; font-weight:600; }
.badge-r     { background:#fee2e2; color:#991b1b; border-radius:4px;
               padding:2px 8px; font-size:0.82rem; font-weight:600; }
.badge-y     { background:#fef3c7; color:#92400e; border-radius:4px;
               padding:2px 8px; font-size:0.82rem; font-weight:600; }
.badge-a     { background:#f3f4f6; color:#374151; border-radius:4px;
               padding:2px 8px; font-size:0.82rem; }
.section-hdr { font-weight:600; font-size:1rem; margin:1rem 0 0.3rem; }
</style>
""", unsafe_allow_html=True)

nyse = mcal.get_calendar("NYSE")


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_signal(option: str) -> dict:
    """Load signal for a single option (A or B) from separate file."""
    try:
        path = hf_hub_download(
            repo_id=cfg.HF_SIGNALS_REPO,
            filename=f"signals/signal_{option}.json",
            repo_type="dataset",
            token=cfg.HF_TOKEN or None,
            force_download=True,
        )
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Could not load signal_{option}.json: {e}")
        return {}

@st.cache_data(ttl=300)
def load_conformal_signal(option: str) -> dict:
    """Load conformal signal for a single option."""
    try:
        path = hf_hub_download(
            repo_id=cfg.HF_SIGNALS_REPO,
            filename=f"conformal/signal_{option}_conformal.json",
            repo_type="dataset",
            token=cfg.HF_TOKEN or None,
            force_download=True,
        )
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}

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

@st.cache_data(ttl=3600)
def load_conformal_history(option: str) -> pd.DataFrame:
    try:
        path = hf_hub_download(
            repo_id=cfg.HF_SIGNALS_REPO,
            filename=f"conformal/signal_history_conformal_{option}.json",
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


def _fmt_dt(s):
    try:
        return datetime.fromisoformat(s).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return s or "—"


def pill(label, val, lo, hi):
    cls = "pill-g" if val < lo else ("pill-r" if val > hi else "pill-a")
    return f'<span class="{cls}">{label}: {val}</span>'


# ═══════════════════════════════════════════════════════════════════════════════
# ORIGINAL NCDE TABS (A and B)
# ═══════════════════════════════════════════════════════════════════════════════

def render_hero(signal: dict, master: pd.DataFrame):
    if not signal or "top_pick" not in signal:
        st.info("Signal not available yet — run the training workflow first.")
        return

    tickers   = cfg.FI_ETFS if signal.get("option") == "A" else cfg.EQ_ETFS
    forecasts = signal.get("forecasts", {})
    ranked    = sorted(
        [(t, forecasts[t]["mu"], forecasts[t]["confidence"])
         for t in tickers if t in forecasts],
        key=lambda x: x[1], reverse=True,
    )
    t1 = ranked[0] if ranked else (signal["top_pick"], signal["top_mu"],
                                   signal["top_confidence"])
    t2 = ranked[1] if len(ranked) > 1 else None
    t3 = ranked[2] if len(ranked) > 2 else None

    next_day = (str(next_trading_day(master.index[-1]).date())
                if not master.empty else signal.get("signal_date", "—"))
    gen = _fmt_dt(signal.get("generated_at", ""))

    runner = ""
    if t2: runner += f"2nd: **{t2[0]}** μ={t2[1]:.4f}"
    if t3: runner += f"&nbsp;&nbsp;3rd: **{t3[0]}** μ={t3[1]:.4f}"

    rc  = signal.get("regime_context", {})
    st_ = signal.get("macro_stress")
    pills = ""
    if rc.get("VIX"):       pills += pill("VIX",    rc["VIX"],       15, 25)
    if rc.get("T10Y2Y"):    pills += pill("T10Y2Y", rc["T10Y2Y"], -0.5, 0.5)
    if rc.get("HY_SPREAD"): pills += pill("HY Spr", rc["HY_SPREAD"], 300, 500)
    if st_ is not None:     pills += pill("Stress",  st_,          -0.5, 0.5)

    st.markdown(f"""
<div class="hero">{t1[0]}</div>
<div class="mu">μ = {t1[1]:.4f}</div>
<div class="conf">Signal for {next_day} &nbsp;·&nbsp; Generated {gen}</div>
<div class="runner">Confidence {t1[2]:.1%}&nbsp;&nbsp;{runner}</div>
<div style="margin-top:0.5rem">{pills}</div>
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
<div style="margin:1rem 0">
  <span class="metric-box">Ann Return: <span class="metric-{c(r)}">{fp(r)}</span></span>
  <span class="metric-box">Sharpe: <span class="metric-{c(s)}">{s:.2f}</span></span>
  <span class="metric-box">IC: <span class="metric-{c(ic)}">{ic:.3f}</span></span>
</div>
""", unsafe_allow_html=True)


def render_forecast_chart(signal: dict, key: str = ""):
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
        x=mus, y=tickers, orientation="h",
        marker_color=colors,
        error_x=dict(type="data", array=sigmas, visible=True, color="#d1d5db"),
        customdata=[[c] for c in confs],
        hovertemplate="<b>%{y}</b><br>μ=%{x:.4f}<br>σ=%{error_x.array:.4f}"
                      "<br>conf=%{customdata[0]:.1%}",
    ))
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#9ca3af")
    fig.update_layout(
        height=max(250, len(tickers) * 32),
        margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(title="Predicted next-day return (μ)",
                   showgrid=True, gridcolor="#f3f4f6"),
        yaxis=dict(showgrid=False),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False}, key=f"fcst_{key}")


def render_history(hist_df: pd.DataFrame, master: pd.DataFrame):
    if hist_df.empty:
        st.info("Signal history will appear after the first prediction run.")
        return

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

    if "actual_return" in hist_df.columns:
        hist_df["actual_return"] = pd.to_numeric(
            hist_df["actual_return"], errors="coerce"
        )
    if "hit" not in hist_df.columns and "actual_return" in hist_df.columns:
        hist_df["hit"] = hist_df["actual_return"].apply(
            lambda x: "✓" if (pd.notna(x) and x > 0)
            else ("✗" if pd.notna(x) else "—")
        )

    disp    = hist_df.sort_values("signal_date", ascending=False).copy()
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
        disp["Actual Return"] = disp["Actual Return"].apply(
            lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—"
        )
    if "Hit" in disp.columns:
        hits  = (disp["Hit"] == "✓").sum()
        total = disp["Hit"].isin(["✓", "✗"]).sum()
        hr    = hits / total if total > 0 else 0
        st.markdown(
            f"<div style='margin-bottom:0.5rem'>"
            f"Hit rate: <b>{hr:.1%}</b> &nbsp;({hits}/{total} signals)</div>",
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
    trained  = _fmt_dt(signal.get("trained_at", "—"))
    n_params = signal.get("model_n_params", 0)
    st.markdown(
        f"<div style='font-size:0.8rem;color:#9ca3af;margin-top:1rem'>"
        f"Trained {trained} &nbsp;·&nbsp; Params: {n_params:,} &nbsp;·&nbsp; "
        f"Solver: {cfg.SOLVER} &nbsp;·&nbsp; Lookback: {cfg.LOOKBACK}d"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_ncde_option(option: str, signal: dict, master: pd.DataFrame):
    hist = load_history(option)
    render_hero(signal, master)
    render_model_metrics(signal)
    st.markdown("<hr style='margin:1.5rem 0 0.5rem'>", unsafe_allow_html=True)
    st.markdown("<b>ETF Forecasts (μ ± σ)</b>", unsafe_allow_html=True)
    st.caption("Bar = predicted next-day return (μ). Error bar = uncertainty (σ). "
               "Blue = top pick.")
    render_forecast_chart(signal, key=option)
    st.markdown("<hr style='margin:1.5rem 0 0.5rem'>", unsafe_allow_html=True)
    st.markdown("<b>Signal History</b>", unsafe_allow_html=True)
    render_history(hist, master)
    render_footnote(signal)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFORMAL TABS (C and D)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_conf_tickers(conf_sig: dict, option: str) -> list:
    all_t = cfg.FI_ETFS if option == "A" else cfg.EQ_ETFS
    fc    = conf_sig.get("conformal_forecasts", {})
    return [t for t in all_t if t in fc and "intervals" in fc[t]]


def render_conformal_hero(conf_sig: dict, option: str):
    if not conf_sig or "top_pick" not in conf_sig:
        st.info(
            "Conformal signal not available yet. "
            "The daily workflow auto-calibrates on first run after training. "
            "Check the Actions tab to confirm the conformal workflow ran successfully."
        )
        return

    pick  = conf_sig["top_pick"]
    mu    = conf_sig["top_mu"]
    conf_ = conf_sig["top_confidence"]
    date  = conf_sig.get("signal_date", "—")
    gen   = _fmt_dt(conf_sig.get("generated_at", ""))
    n_cal = conf_sig.get("n_cal", "?")
    cal_p = conf_sig.get("cal_period", "?")

    fc   = conf_sig.get("conformal_forecasts", {})
    iv90 = fc.get(pick, {}).get("intervals", {}).get("0.9", {})
    lo90 = iv90.get("lo")
    hi90 = iv90.get("hi")
    q_90 = fc.get(pick, {}).get("q_hat", {}).get("0.9", "?")

    iv_str = (f"[{lo90:.4f}, {hi90:.4f}]"
              if lo90 is not None and hi90 is not None else "—")

    if lo90 is not None and hi90 is not None:
        if lo90 > 0:
            sig_class = '<span class="badge-g">STRONG — entire 90% CI positive</span>'
        elif hi90 < 0:
            sig_class = '<span class="badge-r">AVOID — entire 90% CI negative</span>'
        else:
            sig_class = '<span class="badge-y">UNCERTAIN — CI crosses zero</span>'
    else:
        sig_class = ""

    rc  = conf_sig.get("regime_context", {})
    st_ = conf_sig.get("macro_stress")
    pills = ""
    if rc.get("VIX"):       pills += pill("VIX",    rc["VIX"],       15, 25) + "&nbsp;"
    if rc.get("T10Y2Y"):    pills += pill("T10Y2Y", rc["T10Y2Y"], -0.5, 0.5) + "&nbsp;"
    if rc.get("HY_SPREAD"): pills += pill("HY Spr", rc["HY_SPREAD"], 300, 500)
    if st_ is not None:     pills += "&nbsp;" + pill("Stress", st_,  -0.5, 0.5)

    ranked = sorted(
        [(t, d["mu"]) for t, d in fc.items()],
        key=lambda x: x[1], reverse=True,
    )
    runner = ""
    if len(ranked) > 1: runner += f"2nd: **{ranked[1][0]}** μ={ranked[1][1]:.4f}"
    if len(ranked) > 2: runner += f"&nbsp;&nbsp;3rd: **{ranked[2][0]}** μ={ranked[2][1]:.4f}"

    score_type = conf_sig.get("score_type", "normalised")
    if isinstance(q_90, float):
        if score_type == "absolute":
            q_str = f"±{q_90*100:.3f}% half-width"
        else:
            q_str = f"q̂={q_90:.3f} × σ"
    else:
        q_str = str(q_90)

    st.markdown(f"""
<div class="hero">{pick}</div>
<div class="mu">μ = {mu:.4f}</div>
<div class="conf">Signal for {date} &nbsp;·&nbsp; Generated {gen}</div>
<div class="runner">Confidence {conf_:.1%}&nbsp;&nbsp;{runner}</div>
<div style="margin-top:0.5rem">
  {sig_class}&nbsp;
  <span class="badge-a">90% CI {iv_str}</span>&nbsp;
  <span class="badge-a">{q_str}</span>
</div>
<div style="margin-top:0.4rem">
  <span class="badge-a">cal n={n_cal} &nbsp;·&nbsp; {cal_p}</span>
</div>
<div style="margin-top:0.4rem">{pills}</div>
""", unsafe_allow_html=True)


def render_signal_classification_chart(conf_sig: dict, option: str, alpha: str):
    tickers = _get_conf_tickers(conf_sig, option)
    if not tickers:
        st.info("No tickers with conformal interval data yet.")
        return

    conf_fc = conf_sig.get("conformal_forecasts", {})
    valid   = [t for t in tickers
               if alpha in conf_fc.get(t, {}).get("intervals", {})]
    if not valid:
        st.info(f"No interval data at this coverage level yet.")
        return

    valid = sorted(valid, key=lambda t: conf_fc[t]["mu"], reverse=True)

    mus   = [conf_fc[t]["mu"] for t in valid]
    iv_lo = [conf_fc[t]["intervals"][alpha]["lo"] for t in valid]
    iv_hi = [conf_fc[t]["intervals"][alpha]["hi"] for t in valid]
    top   = conf_sig.get("top_pick", "")

    dot_colors  = []
    line_colors = []
    for t, lo, hi in zip(valid, iv_lo, iv_hi):
        if t == top:
            dot_colors.append("#3a5bd9")
            line_colors.append("rgba(58,91,217,0.4)")
        elif lo > 0:
            dot_colors.append("#059669")
            line_colors.append("rgba(5,150,105,0.35)")
        elif hi < 0:
            dot_colors.append("#dc2626")
            line_colors.append("rgba(220,38,38,0.35)")
        else:
            dot_colors.append("#9ca3af")
            line_colors.append("rgba(156,163,175,0.25)")

    fig = go.Figure()

    for i, t in enumerate(valid):
        fig.add_trace(go.Scatter(
            x=[iv_lo[i], iv_hi[i]], y=[t, t],
            mode="lines",
            line=dict(color=line_colors[i], width=7),
            showlegend=False, hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=mus, y=valid, mode="markers",
        marker=dict(color=dot_colors, size=11,
                    line=dict(width=1.5, color="white")),
        customdata=list(zip(
            iv_lo, iv_hi,
            [conf_fc[t]["q_hat"].get(alpha, "?") for t in valid],
        )),
        hovertemplate=(
            "<b>%{y}</b><br>"
            "μ = %{x:.4f}<br>"
            "CI lo = %{customdata[0]:.4f}<br>"
            "CI hi = %{customdata[1]:.4f}<br>"
            "q̂ = %{customdata[2]:.3f}"
        ),
        showlegend=False,
    ))

    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#9ca3af")
    fig.update_layout(
        height=max(300, len(valid) * 36),
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(title="Return  (dot = μ,  bar = conformal CI)",
                   showgrid=True, gridcolor="#f3f4f6"),
        yaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False},
                    key=f"conf_dot_{option}_{alpha}")


def render_interval_comparison_chart(conf_sig: dict, ncde_sig: dict,
                                     alpha: str, option: str):
    if not conf_sig or not ncde_sig:
        return

    tickers = _get_conf_tickers(conf_sig, option)
    if not tickers:
        return

    ncde_fc = ncde_sig.get("forecasts", {})
    conf_fc = conf_sig.get("conformal_forecasts", {})
    valid   = [t for t in tickers
               if alpha in conf_fc.get(t, {}).get("intervals", {})
               and t in ncde_fc]
    if not valid:
        return

    ncde_2s   = [2 * ncde_fc[t]["sigma"]                          for t in valid]
    conf_w    = [conf_fc[t]["intervals"][alpha]["width"]            for t in valid]
    q_hats    = [conf_fc[t]["q_hat"].get(alpha, 1.0)               for t in valid]
    top_pick  = conf_sig.get("top_pick", "")
    score_type = conf_sig.get("score_type", "normalised")

    width_colors = ["#3a5bd9" if t == top_pick else "#6b7280" for t in valid]

    q_colors = []
    for t, q in zip(valid, q_hats):
        if t == top_pick:
            q_colors.append("#3a5bd9")
        elif score_type == "normalised" and q > 1.5:
            q_colors.append("#f59e0b")
        else:
            q_colors.append("#9ca3af")

    q_panel_title = (
        "q̂ per ETF  (red line = 1.0, perfectly calibrated)"
        if score_type == "normalised"
        else "q̂ per ETF  (return half-width in %, red line = pooled 90%)"
    )
    q_axis_label = "q̂" if score_type == "normalised" else "q̂  (return units)"

    q_ref_line = 1.0
    if score_type == "absolute":
        q_ref_line = conformal_params_pooled if (
            conformal_params_pooled := conf_sig.get(
                "coverage_diagnostics", {}
            ).get("0.9", {}).get("pooled", 1.0)
        ) else float(np.mean(q_hats)) if q_hats else 1.0

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"CI width — NCDE 2σ (grey) vs conformal {int(float(alpha)*100)}% (colour)",
            q_panel_title,
        ],
        horizontal_spacing=0.14,
    )

    fig.add_trace(go.Bar(
        name="NCDE 2σ", x=ncde_2s, y=valid, orientation="h",
        marker_color="#e5e7eb",
        hovertemplate="<b>%{y}</b><br>NCDE 2σ = %{x:.5f}",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        name=f"Conformal {int(float(alpha)*100)}% width",
        x=conf_w, y=valid, orientation="h",
        marker_color=width_colors, opacity=0.8,
        hovertemplate="<b>%{y}</b><br>Conformal width = %{x:.5f}",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        name="q̂ (miscalibration factor)",
        x=q_hats, y=valid, orientation="h",
        marker_color=q_colors,
        hovertemplate="<b>%{y}</b><br>q̂ = %{x:.3f}<extra></extra>",
    ), row=1, col=2)

    fig.add_vline(x=q_ref_line, line_width=1.5, line_dash="dash",
                  line_color="#ef4444", row=1, col=2)

    fig.update_layout(
        barmode="overlay",
        height=max(320, len(valid) * 36),
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="white", plot_bgcolor="white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.05,
                    xanchor="left", x=0),
        font=dict(size=11),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f3f4f6", row=1, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="#f3f4f6",
                     title_text=q_axis_label, row=1, col=2)
    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False},
                    key=f"conf_width_{option}_{alpha}")


def render_q_hat_table(conf_sig: dict, option: str):
    tickers = _get_conf_tickers(conf_sig, option)
    if not tickers:
        return

    conf_fc    = conf_sig.get("conformal_forecasts", {})
    score_type = conf_sig.get("score_type", "normalised")

    if score_type == "absolute":
        q_col_suffix = "% ret"
        caption = (
            "q̂ is the **interval half-width in return units** (e.g. 0.015 = ±1.5%). "
            "All ETFs share the same q̂ at each level — width is constant, not σ-scaled. "
            "CI = [μ − q̂,  μ + q̂]."
        )
    else:
        q_col_suffix = "σ-units"
        caption = (
            "q̂ > 1 → NCDE σ was **too tight** on this ETF (intervals inflated). "
            "q̂ < 1 → NCDE was over-cautious (intervals shrunk slightly). "
            "CI = [μ − q̂·σ,  μ + q̂·σ]."
        )

    rows = []
    for t in tickers:
        fc_t = conf_fc.get(t, {})
        q    = fc_t.get("q_hat",     {})
        ivs  = fc_t.get("intervals", {})

        def fmt_q(v):
            if score_type == "absolute":
                return round(v * 100, 4)
            return round(v, 4)

        rows.append({
            "ETF":                  t,
            "μ":                    round(fc_t.get("mu",    0), 5),
            "σ (NCDE)":             round(fc_t.get("sigma", 0), 5),
            f"q̂ 90% ({q_col_suffix})": fmt_q(q.get("0.9", 0)),
            f"q̂ 80% ({q_col_suffix})": fmt_q(q.get("0.8", 0)),
            f"q̂ 70% ({q_col_suffix})": fmt_q(q.get("0.7", 0)),
            "CI 90% lo":            round(ivs.get("0.9", {}).get("lo",    0), 5),
            "CI 90% hi":            round(ivs.get("0.9", {}).get("hi",    0), 5),
            "CI 90% width":         round(ivs.get("0.9", {}).get("width", 0), 5),
        })

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values("μ", ascending=False)
    st.caption(caption)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_coverage_diagnostics(conf_sig: dict):
    diag = (conf_sig or {}).get("coverage_diagnostics") or {}
    if not diag:
        return

    st.markdown('<div class="section-hdr">Calibration coverage diagnostics</div>',
                unsafe_allow_html=True)
    st.caption(
        "Empirical coverage on the val set (n_cal samples). "
        "By the conformal theorem, achieved must be ≥ target."
    )

    rows = []
    for alpha_str in sorted(diag.keys(), reverse=True):
        info     = diag[alpha_str]
        target   = info.get("target", 1 - float(alpha_str))
        achieved = info.get("pooled", 0)
        ok       = achieved >= target - 0.005
        rows.append({
            "Level":   f"{int(float(alpha_str)*100)}%",
            "Target":  f"≥ {target:.0%}",
            "Achieved (pooled)": f"{achieved:.1%}",
            "Status":  "✓ pass" if ok else "✗ fail",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=False, hide_index=True)


def render_conformal_history_table(hist_df: pd.DataFrame):
    if hist_df.empty:
        st.info("Conformal history will appear after the first run.")
        return

    disp = hist_df.sort_values("signal_date", ascending=False).copy()

    if "hit" in disp.columns:
        hits  = (disp["hit"] == True).sum()
        total = disp["hit"].notna().sum()
        hr    = hits / total if total > 0 else 0
        st.markdown(
            f"<div style='margin-bottom:0.4rem'>"
            f"Hit rate: <b>{hr:.1%}</b> ({hits}/{total})</div>",
            unsafe_allow_html=True,
        )

    if "interval_covered" in disp.columns:
        covered = disp["interval_covered"].dropna()
        if len(covered) > 0:
            cov_rate = covered.mean()
            badge    = "badge-g" if cov_rate >= 0.88 else "badge-r"
            st.markdown(
                f"<div style='margin-bottom:0.6rem'>"
                f"90% interval coverage: "
                f"<span class='{badge}'>{cov_rate:.1%}</span> "
                f"({int(covered.sum())}/{len(covered)} signals)"
                f"</div>",
                unsafe_allow_html=True,
            )

    col_map = {
        "signal_date":       "Date",
        "top_pick":          "Pick",
        "top_mu":            "μ",
        "top_confidence":    "Conf.",
        "interval_90_lo":    "CI lo",
        "interval_90_hi":    "CI hi",
        "interval_90_width": "Width",
        "actual_return":     "Actual Return",
        "hit":               "Hit",
        "interval_covered":  "CI covered",
    }
    cols = [c for c in col_map if c in disp.columns]
    disp = disp[cols].rename(columns=col_map)

    if "Conf." in disp.columns:
        disp["Conf."] = disp["Conf."].apply(
            lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—"
        )
    if "Actual Return" in disp.columns:
        disp["Actual Return"] = disp["Actual Return"].apply(
            lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—"
        )
    for col in ["Hit", "CI covered"]:
        if col in disp.columns:
            disp[col] = disp[col].apply(
                lambda x: "✓" if x is True else ("✗" if x is False else "—")
            )
    st.dataframe(disp, use_container_width=True, hide_index=True)


def render_conformal_option(option: str, ncde_signal: dict,
                             conf_signal: dict, master: pd.DataFrame):
    score_type = conf_signal.get("score_type", "normalised") if conf_signal else "normalised"

    render_conformal_hero(conf_signal, option)
    render_model_metrics(conf_signal)

    st.markdown("<hr style='margin:1.5rem 0 0.5rem'>", unsafe_allow_html=True)

    alpha_choice = st.radio(
        "Coverage level",
        options=["0.9", "0.8", "0.7"],
        index=0,
        horizontal=True,
        key=f"alpha_{option}",
        format_func=lambda x: f"{int(float(x)*100)}%",
    )

    st.markdown(
        f'<div class="section-hdr">'
        f'Signal classification — conformal {int(float(alpha_choice)*100)}% intervals'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Dot = μ. Bar = conformal CI. "
        "🔵 top pick &nbsp;·&nbsp; "
        "🟢 CI entirely positive (confident long) &nbsp;·&nbsp; "
        "🔴 CI entirely negative (avoid) &nbsp;·&nbsp; "
        "⚫ CI crosses zero (uncertain)."
    )
    render_signal_classification_chart(conf_signal, option, alpha_choice)

    st.markdown(
        '<div class="section-hdr">'
        'Interval width vs NCDE 2σ, and q̂ miscalibration factor'
        '</div>',
        unsafe_allow_html=True,
    )
    if score_type == "absolute":
        width_caption = (
            "Left: conformal CI width (coloured) vs raw NCDE 2σ (grey). "
            "In **absolute** mode all ETFs share the same q̂ so widths differ only by 2×q̂ vs 2σ. "
            "Right: q̂ in return units — the actual half-width of each interval."
        )
    else:
        width_caption = (
            "Left: how much wider the conformal interval is vs raw NCDE ±σ. "
            "Right: q̂ — the inflation factor per ETF. "
            "Red line = 1.0 (perfectly calibrated). Orange = q̂ > 1.5 (NCDE was overconfident here)."
        )
    st.caption(width_caption)
    render_interval_comparison_chart(conf_signal, ncde_signal, alpha_choice, option)

    st.markdown(
        '<div class="section-hdr">Full conformal table — μ, σ, q̂ at all levels</div>',
        unsafe_allow_html=True,
    )
    render_q_hat_table(conf_signal, option)

    st.markdown("<hr style='margin:1.5rem 0 0.5rem'>", unsafe_allow_html=True)
    render_coverage_diagnostics(conf_signal)

    st.markdown("<hr style='margin:1.5rem 0 0.5rem'>", unsafe_allow_html=True)
    st.markdown('<div class="section-hdr">Conformal signal history</div>',
                unsafe_allow_html=True)
    render_conformal_history_table(load_conformal_history(option))

    if conf_signal:
        st.markdown(
            f"<div style='font-size:0.8rem;color:#9ca3af;margin-top:1rem'>"
            f"Calibration set: {conf_signal.get('n_cal','?')} samples &nbsp;·&nbsp; "
            f"{conf_signal.get('cal_period','?')} &nbsp;·&nbsp; "
            f"Calibrated {_fmt_dt(conf_signal.get('calibrated_at',''))} &nbsp;·&nbsp; "
            f"Params: {conf_signal.get('model_n_params', 0):,}"
            f"</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    st.markdown(
        "<h2 style='margin-bottom:0.2rem'>∂ NCDE — Neural CDE ETF Signal Engine</h2>"
        "<p style='color:#6b7280;margin-top:0'>"
        "Continuous-time &nbsp;·&nbsp; Controlled Differential Equations "
        "&nbsp;·&nbsp; Macro control path &nbsp;·&nbsp; μ + σ forecasts"
        "&nbsp;·&nbsp; Conformal prediction intervals</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🔄 Refresh", help="Clear cache and reload signals"):
            st.cache_data.clear()
            st.rerun()

    with st.spinner("Loading signals and data..."):
        signal_A = load_signal("A")
        signal_B = load_signal("B")
        conformal_A = load_conformal_signal("A")
        conformal_B = load_conformal_signal("B")
        master = load_master()

    tab_a, tab_b, tab_ca, tab_cb = st.tabs([
        "📊 Option A — Fixed Income / Alts",
        "📈 Option B — Equity Sectors",
        "∂̂  Conformal — FI / Commodities",
        "∂̂  Conformal — Equities",
    ])

    with tab_a:
        render_ncde_option("A", signal_A, master)

    with tab_b:
        render_ncde_option("B", signal_B, master)

    with tab_ca:
        with st.expander("ℹ️ What is conformal prediction?", expanded=False):
            st.markdown("""
**Split conformal prediction** wraps the NCDE (μ, σ) into a guaranteed interval
`[μ − q̂·σ, μ + q̂·σ]` where **q̂** is derived from val-set residuals.

**Guarantee:** `P(true return ∈ interval) ≥ 1−α` on any future draw —
no distributional assumptions required.

**What you see here that you don't see in the NCDE tabs:**
- **Signal classification** — whether the *entire* CI is positive (confident long), negative (avoid), or straddles zero
- **q̂ per ETF** — the NCDE miscalibration factor. q̂ > 1.5 means the NCDE was significantly overconfident on that ETF during validation
- **Coverage diagnostics** — verifies the theoretical guarantee is met on the calibration set
- **CI covered** in history — tracks whether the actual return fell inside the interval
""")
        render_conformal_option("A", signal_A, conformal_A, master)

    with tab_cb:
        render_conformal_option("B", signal_B, conformal_B, master)

    st.markdown(
        "<hr style='margin:2rem 0 1rem'>"
        "<div style='text-align:center;font-size:0.8rem;color:#9ca3af'>"
        "P2-ETF-NCDE-ENGINE &nbsp;·&nbsp; Research only &nbsp;·&nbsp; Not financial advice"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
