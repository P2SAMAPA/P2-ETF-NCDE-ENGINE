# conformal/app_conformal.py — NCDE + Conformal Prediction Streamlit Dashboard
#
# Standalone Streamlit page.  Deploy alongside app.py or as its own app.
#
# Shows:
#   • Side-by-side view: NCDE only  vs  NCDE + Conformal
#   • Per-ETF interval chart (error bars grow with q̂·σ)
#   • Interval width ranking (which ETFs are most/least uncertain)
#   • Coverage diagnostics (does the cal-set coverage meet the guarantee?)
#   • Conformal history table with "interval_covered" column
#   • Tab A / Tab B layout identical to original app.py

import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from huggingface_hub import hf_hub_download

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg

st.set_page_config(
    page_title="NCDE + Conformal — ETF Signal Engine",
    page_icon="∂̂",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
.hero        { font-size: 2.8rem; font-weight: 700; margin-bottom: 0.1rem; }
.mu          { font-size: 1.4rem; font-weight: 600; color: #3a5bd9; }
.conf        { font-size: 0.95rem; color: #6b7280; }
.badge-g     { background:#d1fae5; color:#065f46; border-radius:4px;
               padding:2px 7px; font-size:0.8rem; font-weight:600; }
.badge-r     { background:#fee2e2; color:#991b1b; border-radius:4px;
               padding:2px 7px; font-size:0.8rem; font-weight:600; }
.badge-a     { background:#f3f4f6; color:#374151; border-radius:4px;
               padding:2px 7px; font-size:0.8rem; font-weight:600; }
.pill-g      { color:#059669; font-weight:500; }
.pill-r      { color:#dc2626; font-weight:500; }
.pill-a      { color:#6b7280; }
.cover-ok    { color:#059669; font-weight:600; }
.cover-bad   { color:#dc2626; font-weight:600; }
.section-hdr { font-weight:600; font-size:1rem; margin:1rem 0 0.3rem; }
</style>
""", unsafe_allow_html=True)


# ── Data loaders ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_ncde_signals() -> dict:
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
        return {"A": raw.get("option_A") or {}, "B": raw.get("option_B") or {}}
    except Exception as e:
        st.warning(f"Could not load NCDE signals: {e}")
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
        return {"A": raw.get("option_A") or {}, "B": raw.get("option_B") or {}}
    except Exception as e:
        st.warning(f"Could not load conformal signals: {e}")
        return {"A": {}, "B": {}}


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


@st.cache_data(ttl=3600)
def load_ncde_history(option: str) -> pd.DataFrame:
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


# ── UI helpers ────────────────────────────────────────────────────────────────

def _pill(label, val, lo, hi):
    cls = "pill-g" if val < lo else ("pill-r" if val > hi else "pill-a")
    return f'<span class="{cls}">{label}: {val}</span>'


def _fmt_dt(s):
    try:
        return datetime.fromisoformat(s).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return s or "—"


def render_hero_pair(ncde_sig: dict, conf_sig: dict):
    """Two-column hero: NCDE (left) vs NCDE + Conformal (right)."""
    col_l, col_r = st.columns(2, gap="large")

    # ── Left: raw NCDE ───────────────────────────────────────────────────────
    with col_l:
        st.markdown("#### NCDE — raw signal")
        if not ncde_sig or "top_pick" not in ncde_sig:
            st.info("NCDE signal not available.")
        else:
            pick  = ncde_sig["top_pick"]
            mu    = ncde_sig["top_mu"]
            conf  = ncde_sig["top_confidence"]
            date  = ncde_sig.get("signal_date", "—")
            gen   = _fmt_dt(ncde_sig.get("generated_at", ""))
            rc    = ncde_sig.get("regime_context", {})
            stress = ncde_sig.get("macro_stress")

            pills = ""
            if rc.get("VIX"):      pills += _pill("VIX",    rc["VIX"],      15, 25) + "&nbsp;&nbsp;"
            if rc.get("T10Y2Y"):   pills += _pill("T10Y2Y", rc["T10Y2Y"], -0.5, 0.5) + "&nbsp;&nbsp;"
            if rc.get("HY_SPREAD"):pills += _pill("HY Spr", rc["HY_SPREAD"], 300, 500)
            if stress is not None: pills += "&nbsp;&nbsp;" + _pill("Stress", stress, -0.5, 0.5)

            st.markdown(f"""
<div class="hero">{pick}</div>
<div class="mu">μ = {mu:.4f}</div>
<div class="conf">Signal for {date} &nbsp;·&nbsp; Generated {gen}</div>
<div class="conf">Confidence {conf:.1%}</div>
<div style="margin-top:0.4rem">{pills}</div>
""", unsafe_allow_html=True)

    # ── Right: conformal ─────────────────────────────────────────────────────
    with col_r:
        st.markdown("#### NCDE + Conformal")
        if not conf_sig or "top_pick" not in conf_sig:
            st.info("Conformal signal not available — run calibrate.py first.")
        else:
            pick  = conf_sig["top_pick"]
            mu    = conf_sig["top_mu"]
            conf_ = conf_sig["top_confidence"]
            date  = conf_sig.get("signal_date", "—")
            gen   = _fmt_dt(conf_sig.get("generated_at", ""))
            iv90  = conf_sig.get("top_interval_90", {})
            lo90  = iv90.get("lo")
            hi90  = iv90.get("hi")
            n_cal = conf_sig.get("n_cal", "?")
            cal_p = conf_sig.get("cal_period", "?")

            interval_str = (
                f"[{lo90:.4f}, {hi90:.4f}]" if lo90 is not None else "—"
            )

            st.markdown(f"""
<div class="hero">{pick}</div>
<div class="mu">μ = {mu:.4f}</div>
<div class="conf">Signal for {date} &nbsp;·&nbsp; Generated {gen}</div>
<div class="conf">Confidence {conf_:.1%}</div>
<div style="margin-top:0.4rem">
  <span class="badge-g">90% CI {interval_str}</span>
  &nbsp;
  <span class="badge-a">cal n={n_cal} &nbsp;·&nbsp; {cal_p}</span>
</div>
""", unsafe_allow_html=True)


# ── Interval chart ────────────────────────────────────────────────────────────

def render_interval_chart(ncde_sig: dict, conf_sig: dict, alpha: str, option: str):
    """
    Horizontal bar chart showing mu with NCDE σ error bar (grey)
    vs conformal interval (coloured).
    """
    if not ncde_sig or not conf_sig:
        st.info("Both NCDE and conformal signals required for comparison chart.")
        return

    tickers_all = cfg.FI_ETFS if option == "A" else cfg.EQ_ETFS
    ncde_fc  = ncde_sig.get("forecasts", {})
    conf_fc  = conf_sig.get("conformal_forecasts", {})
    tickers  = [t for t in tickers_all if t in ncde_fc and t in conf_fc]

    mus        = [ncde_fc[t]["mu"]    for t in tickers]
    sigma_ncde = [ncde_fc[t]["sigma"] for t in tickers]

    conf_lo = [conf_fc[t]["intervals"][alpha]["lo"]    for t in tickers]
    conf_hi = [conf_fc[t]["intervals"][alpha]["hi"]    for t in tickers]
    conf_w  = [(conf_fc[t]["intervals"][alpha]["hi"] -
                conf_fc[t]["intervals"][alpha]["lo"]) / 2 for t in tickers]

    top_pick = ncde_sig.get("top_pick", "")
    colors   = ["#3a5bd9" if t == top_pick else "#9ca3af" for t in tickers]

    fig = go.Figure()

    # Conformal intervals (filled error bars)
    fig.add_trace(go.Bar(
        name=f"Conformal {int(float(alpha)*100)}% CI",
        x=mus, y=tickers,
        orientation="h",
        marker_color=[c + "55" for c in colors],
        error_x=dict(
            type="data",
            symmetric=False,
            array=[h - m for h, m in zip(conf_hi, mus)],
            arrayminus=[m - l for m, l in zip(mus, conf_lo)],
            visible=True,
            color="#3a5bd9",
            thickness=3,
            width=6,
        ),
        customdata=[[conf_fc[t]["intervals"][alpha]["lo"],
                     conf_fc[t]["intervals"][alpha]["hi"],
                     conf_fc[t]["intervals"][alpha]["width"]] for t in tickers],
        hovertemplate=(
            "<b>%{y}</b><br>"
            f"μ = %{{x:.4f}}<br>"
            f"Conformal {int(float(alpha)*100)}% CI: [%{{customdata[0]:.4f}}, %{{customdata[1]:.4f}}]<br>"
            "Width: %{customdata[2]:.4f}"
        ),
    ))

    # NCDE raw σ (thin grey)
    fig.add_trace(go.Bar(
        name="NCDE σ (raw)",
        x=mus, y=tickers,
        orientation="h",
        marker_color=colors,
        error_x=dict(
            type="data", array=sigma_ncde, visible=True,
            color="#d1d5db", thickness=1.5, width=4,
        ),
        hovertemplate="<b>%{y}</b><br>μ = %{x:.4f}<br>σ = %{error_x.array:.4f}",
    ))

    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#9ca3af")

    fig.update_layout(
        barmode="overlay",
        height=max(280, len(tickers) * 34),
        margin=dict(l=0, r=0, t=8, b=0),
        paper_bgcolor="white", plot_bgcolor="white",
        xaxis=dict(
            title="Predicted next-day return",
            showgrid=True, gridcolor="#f3f4f6",
        ),
        yaxis=dict(showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        showlegend=True,
    )

    st.plotly_chart(fig, use_container_width=True,
                    config={"displayModeBar": False},
                    key=f"iv_chart_{option}_{alpha}")


# ── Interval width ranking ────────────────────────────────────────────────────

def render_width_ranking(conf_sig: dict, alpha: str, option: str):
    """Show which ETFs have the widest/narrowest conformal intervals."""
    if not conf_sig:
        return

    tickers_all = cfg.FI_ETFS if option == "A" else cfg.EQ_ETFS
    conf_fc = conf_sig.get("conformal_forecasts", {})
    rows = []
    for t in tickers_all:
        if t not in conf_fc:
            continue
        iv = conf_fc[t]["intervals"].get(alpha, {})
        rows.append({
            "ETF":    t,
            "μ":      round(conf_fc[t]["mu"], 5),
            "σ (NCDE)": round(conf_fc[t]["sigma"], 5),
            f"q̂ ({int(float(alpha)*100)}%)": round(conf_fc[t]["q_hat"].get(alpha, 0), 3),
            f"CI lo ({int(float(alpha)*100)}%)": round(iv.get("lo", 0), 5),
            f"CI hi ({int(float(alpha)*100)}%)": round(iv.get("hi", 0), 5),
            "Width": round(iv.get("width", 0), 5),
        })

    if not rows:
        return

    df = pd.DataFrame(rows).sort_values("Width", ascending=False)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ── Coverage diagnostics ──────────────────────────────────────────────────────

def render_coverage_diagnostics(conf_sig: dict):
    """Show calibration-set empirical coverage vs target for each alpha."""
    if not conf_sig:
        return

    diag = conf_sig.get("coverage_diagnostics") or {}
    if not diag:
        st.info("Coverage diagnostics not available.")
        return

    st.markdown('<div class="section-hdr">Calibration coverage diagnostics</div>',
                unsafe_allow_html=True)
    st.caption(
        "Empirical coverage on the **val set** (n_cal samples). "
        "Coverage ≥ target is guaranteed by the conformal theorem. "
        "If any row shows ✗, the calibration set may be too small."
    )

    rows = []
    for alpha_str in sorted(diag.keys(), reverse=True):
        info    = diag[alpha_str]
        target  = info.get("target", 1 - float(alpha_str))
        achieved = info.get("pooled", 0)
        ok      = achieved >= target - 0.005
        rows.append({
            "Coverage level": f"{int(float(alpha_str)*100)}%",
            "Target":   f"≥ {target:.0%}",
            "Achieved (pooled)": f"{achieved:.1%}",
            "Status":   "✓ pass" if ok else "✗ fail",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=False, hide_index=True)


# ── History table ─────────────────────────────────────────────────────────────

def render_conformal_history(hist_df: pd.DataFrame):
    if hist_df.empty:
        st.info("Conformal signal history will appear after the first run.")
        return

    disp = hist_df.sort_values("signal_date", ascending=False).copy()

    # Hit rate
    if "hit" in disp.columns:
        hits  = (disp["hit"] == True).sum()         # noqa
        total = disp["hit"].notna().sum()
        hr    = hits / total if total > 0 else 0
        st.markdown(
            f"<div style='margin-bottom:0.4rem'>"
            f"Hit rate: <b>{hr:.1%}</b> ({hits}/{total})</div>",
            unsafe_allow_html=True,
        )

    # Interval coverage rate
    if "interval_covered" in disp.columns:
        covered = disp["interval_covered"].dropna()
        if len(covered) > 0:
            cov_rate = covered.mean()
            badge = "badge-g" if cov_rate >= 0.88 else "badge-r"
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
    cols_present = [c for c in col_map if c in disp.columns]
    disp = disp[cols_present].rename(columns=col_map)

    for pct_col in ["Conf."]:
        if pct_col in disp.columns:
            disp[pct_col] = disp[pct_col].apply(
                lambda x: f"{x*100:.1f}%" if pd.notna(x) else "—"
            )
    if "Actual Return" in disp.columns:
        disp["Actual Return"] = disp["Actual Return"].apply(
            lambda x: f"{x*100:.2f}%" if pd.notna(x) else "—"
        )
    for bool_col in ["Hit", "CI covered"]:
        if bool_col in disp.columns:
            disp[bool_col] = disp[bool_col].apply(
                lambda x: "✓" if x is True else ("✗" if x is False else "—")
            )

    st.dataframe(disp, use_container_width=True, hide_index=True)


# ── Option renderer ───────────────────────────────────────────────────────────

def render_option(option: str, ncde_signals: dict, conf_signals: dict):
    ncde_sig = ncde_signals.get(option, {})
    conf_sig = conf_signals.get(option, {})

    render_hero_pair(ncde_sig, conf_sig)

    st.markdown("<hr style='margin:1.5rem 0 0.5rem'>", unsafe_allow_html=True)

    # Alpha selector
    alpha_choice = st.radio(
        "Coverage level for interval chart",
        options=["0.9", "0.8", "0.7"],
        index=0,
        horizontal=True,
        key=f"alpha_{option}",
        format_func=lambda x: f"{int(float(x)*100)}%",
    )

    st.markdown('<div class="section-hdr">ETF forecasts — NCDE vs conformal interval</div>',
                unsafe_allow_html=True)
    st.caption(
        "Blue bar = top pick. "
        "**Thick coloured error bar** = conformal guaranteed interval. "
        "**Thin grey error bar** = raw NCDE ±σ. "
        "The conformal interval is always wider — that is correct."
    )
    render_interval_chart(ncde_sig, conf_sig, alpha_choice, option)

    st.markdown('<div class="section-hdr">Interval width ranking (widest → narrowest)</div>',
                unsafe_allow_html=True)
    render_width_ranking(conf_sig, alpha_choice, option)

    st.markdown("<hr style='margin:1.5rem 0 0.5rem'>", unsafe_allow_html=True)
    render_coverage_diagnostics(conf_sig)

    st.markdown("<hr style='margin:1.5rem 0 0.5rem'>", unsafe_allow_html=True)
    st.markdown('<div class="section-hdr">Conformal signal history</div>',
                unsafe_allow_html=True)
    hist = load_conformal_history(option)
    render_conformal_history(hist)

    # NCDE comparison footnote
    ncde_hist = load_ncde_history(option)
    if not ncde_hist.empty and "hit" in ncde_hist.columns:
        ncde_hits  = (ncde_hist["hit"] == True).sum()       # noqa
        ncde_total = ncde_hist["hit"].notna().sum()
        if ncde_total > 0:
            st.markdown(
                f"<div style='font-size:0.8rem;color:#9ca3af;margin-top:0.5rem'>"
                f"NCDE baseline hit rate: {ncde_hits/ncde_total:.1%} "
                f"({ncde_hits}/{ncde_total} signals)</div>",
                unsafe_allow_html=True,
            )

    if conf_sig:
        n_cal    = conf_sig.get("n_cal", "?")
        cal_p    = conf_sig.get("cal_period", "?")
        cal_at   = _fmt_dt(conf_sig.get("calibrated_at", ""))
        gen_at   = _fmt_dt(conf_sig.get("generated_at", ""))
        n_params = conf_sig.get("model_n_params", 0)
        st.markdown(
            f"<div style='font-size:0.8rem;color:#9ca3af;margin-top:0.5rem'>"
            f"Calibration set: {n_cal} samples &nbsp;·&nbsp; {cal_p} &nbsp;·&nbsp; "
            f"Calibrated {cal_at} &nbsp;·&nbsp; Signal {gen_at} &nbsp;·&nbsp; "
            f"Params: {n_params:,}"
            f"</div>",
            unsafe_allow_html=True,
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.markdown(
        "<h2 style='margin-bottom:0.1rem'>∂̂ NCDE + Conformal Prediction</h2>"
        "<p style='color:#6b7280;margin-top:0'>"
        "NCDE point forecast &nbsp;+&nbsp; split conformal prediction intervals "
        "&nbsp;·&nbsp; marginal coverage guarantee &nbsp;·&nbsp; "
        "compare raw vs calibrated uncertainty</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()

    with st.expander("ℹ️ What is conformal prediction?", expanded=False):
        st.markdown("""
**Split conformal prediction** (Angelopoulos & Bates, 2022) turns any model's output
into an interval with a *finite-sample, distribution-free* coverage guarantee.

**How it works here:**
1. The NCDE val set (10% of history, never used in training) becomes the *calibration set*.
2. For each calibration sample we compute the nonconformity score:
   `s = |y − μ| / σ`  (how many σ-units was the NCDE off?)
3. At coverage level `1−α`, the conformal quantile is the `⌈(n+1)(1−α)⌉/n`-th
   empirical quantile of `{s₁, …, sₙ}`.
4. The prediction interval for a new ETF is: `[μ − q̂·σ,  μ + q̂·σ]`

**Guarantee:** On any new exchangeable draw, the true return falls inside the interval
with probability **≥ 1−α** — regardless of model misspecification.

**What changes:** Only the width of the interval (q̂ scales σ). The point forecast
(μ), top pick ranking, and all NCDE logic are completely untouched.
""")

    with st.spinner("Loading signals..."):
        ncde_signals = load_ncde_signals()
        conf_signals = load_conformal_signals()

    tab_a, tab_b = st.tabs([
        "📊 Option A — Fixed Income / Alts",
        "📈 Option B — Equity Sectors",
    ])

    with tab_a:
        render_option("A", ncde_signals, conf_signals)
    with tab_b:
        render_option("B", ncde_signals, conf_signals)

    st.markdown(
        "<hr style='margin:2rem 0 1rem'>"
        "<div style='text-align:center;font-size:0.8rem;color:#9ca3af'>"
        "P2-ETF-NCDE-ENGINE + Conformal &nbsp;·&nbsp; Research only &nbsp;·&nbsp; Not financial advice"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
