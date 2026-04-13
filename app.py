# app.py — P2-ETF-NCDE-ENGINE Streamlit Dashboard
#
# Tabs:
#   📊 Option A — Fixed Income / Alts      (original NCDE, unchanged)
#   📈 Option B — Equity Sectors           (original NCDE, unchanged)
#   ∂̂  Conformal — FI / Commodities       (NCDE + conformal intervals)
#   ∂̂  Conformal — Equities               (NCDE + conformal intervals)

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
.hero        { font-size: 3rem; font-weight: 700; margin-bottom: 0.2rem; }
.mu          { font-size: 1.5rem; font-weight: 600; color: #3a5bd9; }
.conf        { font-size: 1rem; color: #6b7280; }
.runner      { font-size: 0.95rem; color: #374151; margin-top: 0.5rem; }
.pill-g      { color: #059669; font-weight: 500; }
.pill-r      { color: #dc2626; font-weight: 500; }
.pill-a      { color: #6b7280; }
.metric-box  { border: 1px solid #e5e7eb; border-radius: 6px;
               padding: 0.5rem 0.75rem; margin-right: 0.5rem; display: inline-block; }
.metric-pos  { color: #059669; font-weight: 600; }
.metric-neg  { color: #dc2626; font-weight: 600; }
.badge-g     { background:#d1fae5; color:#065f46; border-radius:4px;
               padding:2px 8px; font-size:0.82rem; font-weight:600; }
.badge-r     { background:#fee2e2; color:#991b1b; border-radius:4px;
               padding:2px 8px; font-size:0.82rem; font-weight:600; }
.badge-a     { background:#f3f4f6; color:#374151; border-radius:4px;
               padding:2px 8px; font-size:0.82rem; }
.section-hdr { font-weight: 600; font-size: 1rem; margin: 1rem 0 0.3rem; }
</style>
""", unsafe_allow_html=True)

nyse = mcal.get_calendar("NYSE")


# ── Data loading ───────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
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


@st.cache_data(ttl=300)
def load_conformal_signals() -> dict:
    try:
        path = hf_hub_download(
            repo_id=cfg.HF_SIGNALS_REPO,
            filename="conformal/latest_signals_conformal.json",
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
    except Exception:
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
    days = sched.index[sched.index > date]
    return days[0] if len(days) > 0 else date + pd.Timedelta(days=1)


def _fmt_dt(s):
    try:
        return datetime.fromisoformat(s).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return s or "—"


# ═══════════════════════════════════════════════════════════════════════════════
# ORIGINAL NCDE RENDERING (tabs A and B — unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

def pill(label, val, lo, hi):
    cls = "pill-g" if val < lo else ("pill-r" if val > hi else "pill-a")
    return f'<span class="{cls}">{label}: {val}</span>'


def render_hero(signal: dict, master: pd.DataFrame):
    if not signal or "top_pick" not in signal:
        st.info("Signal not available yet — run the training workflow first.")
        return

    tickers   = cfg.FI_ETFS if signal.get("option") == "A" else cfg.EQ_ETFS
    forecasts = signal.get("forecasts", {})

    ranked = sorted(
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
    if t2:
        runner += f"2nd: **{t2[0]}** μ={t2[1]:.4f}"
    if t3:
        runner += f"&nbsp;&nbsp;3rd: **{t3[0]}** μ={t3[1]:.4f}"

    rc = signal.get("regime_context", {})
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
        hovertemplate="<b>%{y}</b><br>μ = %{x:.4f}<br>σ = %{error_x.array:.4f}"
                      "<br>conf = %{customdata[0]:.1%}",
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
            lambda x: "✓" if (pd.notna(x) and x > 0) else ("✗" if pd.notna(x) else "—")
        )

    disp    = hist_df.sort_values("signal_date", ascending=False).copy()
    col_map = {
        "signal_date":   "Date",
        "top_pick":      "Pick",
        "top_mu":        "μ",
        "top_confidence":"Confidence",
        "actual_return": "Actual Return",
        "hit":           "Hit",
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


def render_ncde_option(option: str, signals: dict, master: pd.DataFrame):
    signal = signals.get(option, {})
    hist   = load_history(option)
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
# CONFORMAL RENDERING (tabs C and D — new)
# ═══════════════════════════════════════════════════════════════════════════════

def render_conformal_hero(conf_sig: dict, ncde_sig: dict):
    """Hero card for conformal tab — shows pick + 90% interval badge."""
    if not conf_sig or "top_pick" not in conf_sig:
        st.info(
            "Conformal signal not available yet. "
            "The workflow will auto-calibrate on the next run after training completes."
        )
        return

    pick   = conf_sig["top_pick"]
    mu     = conf_sig["top_mu"]
    conf_  = conf_sig["top_confidence"]
    date   = conf_sig.get("signal_date", "—")
    gen    = _fmt_dt(conf_sig.get("generated_at", ""))
    n_cal  = conf_sig.get("n_cal", "?")
    cal_p  = conf_sig.get("cal_period", "?")
    iv90   = conf_sig.get("top_interval_90", {})
    lo90   = iv90.get("lo")
    hi90   = iv90.get("hi")

    iv_str = (f"[{lo90:.4f}, {hi90:.4f}]"
              if lo90 is not None and hi90 is not None else "—")

    # Macro pills from NCDE signal
    rc    = conf_sig.get("regime_context", {})
    st_   = conf_sig.get("macro_stress")
    pills = ""
    if rc.get("VIX"):       pills += pill("VIX",    rc["VIX"],       15, 25)
    if rc.get("T10Y2Y"):    pills += pill("T10Y2Y", rc["T10Y2Y"], -0.5, 0.5)
    if rc.get("HY_SPREAD"): pills += pill("HY Spr", rc["HY_SPREAD"], 300, 500)
    if st_ is not None:     pills += pill("Stress",  st_,          -0.5, 0.5)

    # Ranked picks from conformal_forecasts
    fc = conf_sig.get("conformal_forecasts", {})
    ranked = sorted(
        [(t, d["mu"]) for t, d in fc.items()],
        key=lambda x: x[1], reverse=True,
    )
    runner = ""
    if len(ranked) > 1:
        runner += f"2nd: **{ranked[1][0]}** μ={ranked[1][1]:.4f}"
    if len(ranked) > 2:
        runner += f"&nbsp;&nbsp;3rd: **{ranked[2][0]}** μ={ranked[2][1]:.4f}"

    st.markdown(f"""
<div class="hero">{pick}</div>
<div class="mu">μ = {mu:.4f}</div>
<div class="conf">Signal for {date} &nbsp;·&nbsp; Generated {gen}</div>
<div class="runner">Confidence {conf_:.1%}&nbsp;&nbsp;{runner}</div>
<div style="margin-top:0.5rem">
  <span class="badge-g">90% CI {iv_str}</span>
  &nbsp;
  <span class="badge-a">cal n={n_cal} &nbsp;·&nbsp; {cal_p}</span>
</div>
<div style="margin-top:0.4rem">{pills}</div>
""", unsafe_allow_html=True)


def render_conformal_chart(conf_sig: dict, ncde_sig: dict,
                           alpha: str, option: str):
    """
    Overlay chart: NCDE ±σ (thin grey) vs conformal interval (thick coloured).
    Blue = top pick.
    """
    if not conf_sig or not ncde_sig:
        st.info("Need both NCDE and conformal signals to render comparison chart.")
        return

    # Convert alpha to string key (as it appears in JSON)
    try:
        alpha_float = float(alpha)
        alpha_key = str(alpha_float)
        alpha_pct = int(alpha_float * 100)
    except:
        alpha_key = "0.9"
        alpha_pct = 90
    
    tickers_all = cfg.FI_ETFS if option == "A" else cfg.EQ_ETFS
    ncde_fc  = ncde_sig.get("forecasts", {})
    conf_fc  = conf_sig.get("conformal_forecasts", {})
    
    # Build data arrays
    tickers = []
    mus = []
    sigmas = []
    iv_los = []
    iv_his = []
    
    for t in tickers_all:
        if t not in ncde_fc or t not in conf_fc:
            continue
            
        # Get NCDE values
        mu = ncde_fc[t].get("mu", 0.0)
        sigma = ncde_fc[t].get("sigma", 0.0)
        
        # Get conformal interval for this alpha level
        intervals = conf_fc[t].get("intervals", {})
        interval = intervals.get(alpha_key, {})
        
        lo = interval.get("lo", mu - sigma)
        hi = interval.get("hi", mu + sigma)
        
        tickers.append(t)
        mus.append(float(mu))
        sigmas.append(float(sigma))
        iv_los.append(float(lo))
        iv_his.append(float(hi))
    
    if not tickers:
        st.warning(f"No data available for {alpha_pct}% confidence level")
        return
    
    top_pick = conf_sig.get("top_pick", "")
    
    # Create simple colors - no transparency issues
    bar_colors = []
    for t in tickers:
        if t == top_pick:
            bar_colors.append("#3a5bd9")
        else:
            bar_colors.append("#9ca3af")
    
    # Create figure
    fig = go.Figure()
    
    # Add conformal interval trace (thick colored error bars)
    # Calculate error bar values
    error_plus = []
    error_minus = []
    for i in range(len(mus)):
        error_plus.append(iv_his[i] - mus[i])
        error_minus.append(mus[i] - iv_los[i])
    
    fig.add_trace(go.Bar(
        name=f"Conformal {alpha_pct}% CI",
        x=mus,
        y=tickers,
        orientation='h',
        marker_color=bar_colors,
        opacity=0.3,
        error_x=dict(
            type='data',
            symmetric=False,
            array=error_plus,
            arrayminus=error_minus,
            visible=True,
            color="#3a5bd9",
            thickness=6,
            width=10
        ),
        showlegend=True
    ))
    
    # Add NCDE raw sigma trace (thin grey error bars)
    fig.add_trace(go.Bar(
        name="NCDE ±σ (raw)",
        x=mus,
        y=tickers,
        orientation='h',
        marker_color=bar_colors,
        opacity=0.8,
        error_x=dict(
            type='data',
            array=sigmas,
            visible=True,
            color='#d1d5db',
            thickness=2,
            width=8
        ),
        showlegend=True
    ))
    
    # Add vertical line at zero
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#9ca3af")
    
    # Calculate height based on number of tickers
    chart_height = max(400, len(tickers) * 35)
    
    # Update layout
    fig.update_layout(
        barmode='overlay',
        height=chart_height,
        margin=dict(l=10, r=10, t=30, b=30),
        paper_bgcolor='white',
        plot_bgcolor='white',
        xaxis=dict(
            title="Predicted next-day return",
            showgrid=True,
            gridcolor='#f3f4f6',
            tickformat='.3f',
            zeroline=False
        ),
        yaxis=dict(
            showgrid=False,
            title=None
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0
        )
    )
    
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_coverage_diagnostics(conf_sig: dict):
    """Coverage diagnostics table — calibration-set empirical coverage vs target."""
    diag = (conf_sig or {}).get("coverage_diagnostics") or {}
    if not diag:
        return

    st.markdown('<div class="section-hdr">Calibration coverage diagnostics</div>',
                unsafe_allow_html=True)
    st.caption(
        "Empirical coverage on the val-set calibration samples. "
        "Coverage ≥ target is guaranteed by the conformal theorem."
    )

    rows = []
    for alpha_str in sorted(diag.keys(), reverse=True):
        info     = diag[alpha_str]
        target   = info.get("target", 1 - float(alpha_str))
        achieved = info.get("pooled", 0)
        ok       = achieved >= target - 0.005
        rows.append({
            "Level":            f"{int(float(alpha_str)*100)}%",
            "Target":           f"≥ {target:.0%}",
            "Achieved (pooled)": f"{achieved:.1%}",
            "Status":           "✓ pass" if ok else "✗ fail",
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=False, hide_index=True)


def render_conformal_history(hist_df: pd.DataFrame):
    if hist_df.empty:
        st.info("Conformal signal history will appear after the first run.")
        return

    disp = hist_df.sort_values("signal_date", ascending=False).copy()

    if "hit" in disp.columns:
        hits  = (disp["hit"] == True).sum()   # noqa
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
        "interval_90_lo":    "CI lo (90%)",
        "interval_90_hi":    "CI hi (90%)",
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


def render_conformal_option(option: str, ncde_signals: dict,
                            conf_signals: dict, master: pd.DataFrame):
    ncde_sig = ncde_signals.get(option, {})
    conf_sig = conf_signals.get(option, {})

    render_conformal_hero(conf_sig, ncde_sig)
    render_model_metrics(conf_sig)

    st.markdown("<hr style='margin:1.5rem 0 0.5rem'>", unsafe_allow_html=True)

    alpha_choice = st.radio(
        "Coverage level",
        options=["0.9", "0.8", "0.7"],
        index=0,
        horizontal=True,
        key=f"alpha_{option}",
        format_func=lambda x: f"{int(float(x)*100)}%",
    )

    st.markdown('<div class="section-hdr">ETF forecasts — NCDE vs conformal interval</div>',
                unsafe_allow_html=True)
    st.caption(
        "**Thick coloured error bar** = conformal guaranteed interval at selected level.  "
        "**Thin grey error bar** = raw NCDE ±σ. "
        "Blue = top pick. The conformal bar is always at least as wide as ±σ."
    )
    render_conformal_chart(conf_sig, ncde_sig, alpha_choice, option)

    st.markdown("<hr style='margin:1.5rem 0 0.5rem'>", unsafe_allow_html=True)
    render_coverage_diagnostics(conf_sig)

    st.markdown("<hr style='margin:1.5rem 0 0.5rem'>", unsafe_allow_html=True)
    st.markdown('<div class="section-hdr">Conformal signal history</div>',
                unsafe_allow_html=True)
    conf_hist = load_conformal_history(option)
    render_conformal_history(conf_hist)

    # Footer
    if conf_sig:
        n_cal   = conf_sig.get("n_cal", "?")
        cal_p   = conf_sig.get("cal_period", "?")
        cal_at  = _fmt_dt(conf_sig.get("calibrated_at", ""))
        n_par   = conf_sig.get("model_n_params", 0)
        st.markdown(
            f"<div style='font-size:0.8rem;color:#9ca3af;margin-top:1rem'>"
            f"Calibration set: {n_cal} samples &nbsp;·&nbsp; {cal_p} &nbsp;·&nbsp; "
            f"Calibrated {cal_at} &nbsp;·&nbsp; Params: {n_par:,}"
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
        signals          = load_signals()
        conformal_sigs   = load_conformal_signals()
        master           = load_master()

    (tab_a, tab_b,
     tab_conf_a, tab_conf_b) = st.tabs([
        "📊 Option A — Fixed Income / Alts",
        "📈 Option B — Equity Sectors",
        "∂̂  Conformal — FI / Commodities",
        "∂̂  Conformal — Equities",
    ])

    with tab_a:
        render_ncde_option("A", signals, master)

    with tab_b:
        render_ncde_option("B", signals, master)

    with tab_conf_a:
        with st.expander("ℹ️ What is conformal prediction?", expanded=False):
            st.markdown("""
**Split conformal prediction** turns the NCDE's (μ, σ) into an interval
`[μ − q̂·σ,  μ + q̂·σ]` with a **finite-sample, distribution-free** coverage guarantee.

**How q̂ is computed:**
1. Run the frozen model on the val set (10% holdout — never used in training).
2. For each sample, score `s = |y − μ| / σ` (normalised absolute residual).
3. At coverage 1−α, `q̂ = ⌈(n+1)(1−α)⌉/n`-th quantile of those scores.

**Guarantee:** `P(true return ∈ interval) ≥ 1−α` on any future exchangeable draw —
regardless of whether the NCDE is well-specified.

The point forecast (μ, pick ranking) is **completely unchanged**.
""")
        render_conformal_option("A", signals, conformal_sigs, master)

    with tab_conf_b:
        render_conformal_option("B", signals, conformal_sigs, master)

    st.markdown(
        "<hr style='margin:2rem 0 1rem'>"
        "<div style='text-align:center;font-size:0.8rem;color:#9ca3af'>"
        "P2-ETF-NCDE-ENGINE &nbsp;·&nbsp; Research only &nbsp;·&nbsp; Not financial advice"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
