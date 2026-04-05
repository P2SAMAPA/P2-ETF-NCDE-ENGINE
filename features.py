# features.py — Feature engineering and continuous path builder for NCDE
#
# Key difference from DeePM features.py:
#   build_ncde_path() produces a torchcde-compatible cubic spline interpolation
#   of the multivariate time series, which is what the NCDE solver consumes.
#
# Input feature set per asset per day:
#   - Log returns: 1d, 5d, 21d, 63d
#   - Realised vol (21d annualised)
#   - Vol-scaled return (Sharpe-like signal)
#   - Cross-sectional momentum z-rank
#
# Control path (macro, shared across assets):
#   - Derived macro features from macro_derived
#   - These feed in as dX(t) — the "control" in the controlled DE

import numpy as np
import pandas as pd
import torch
import torchcde
from sklearn.preprocessing import RobustScaler
import config as cfg


# ── Per-asset time-series features ────────────────────────────────────────────

def build_asset_features(log_returns: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    """
    Build per-asset features concatenated into a flat DataFrame.
    Columns: {TICKER}_logret_1d, {TICKER}_logret_5d, ..., {TICKER}_vol, {TICKER}_mom_zrank
    """
    frames = []
    for ticker in log_returns.columns:
        lr = log_returns[ticker]
        v  = vol[ticker] if ticker in vol.columns else \
             lr.rolling(cfg.VOL_WINDOW).std() * np.sqrt(252)

        f = pd.DataFrame(index=lr.index)
        f[f"{ticker}_logret_1d"]  = lr
        f[f"{ticker}_logret_5d"]  = lr.rolling(5).sum()
        f[f"{ticker}_logret_21d"] = lr.rolling(21).sum()
        f[f"{ticker}_logret_63d"] = lr.rolling(63).sum()
        f[f"{ticker}_vol"]        = v
        f[f"{ticker}_vol_scaled"] = lr / (v.replace(0, np.nan) / np.sqrt(252) + 1e-8)
        frames.append(f)

    asset_features = pd.concat(frames, axis=1)

    # Cross-sectional momentum z-rank ([-1, 1])
    ret_1d_cols  = [c for c in asset_features.columns if c.endswith("_logret_1d")]
    ret_21d_cols = [c for c in asset_features.columns if c.endswith("_logret_21d")]

    for cols, suffix in [(ret_1d_cols, "mom1d_zrank"), (ret_21d_cols, "mom21d_zrank")]:
        rank_df = asset_features[cols].rank(axis=1, pct=True) * 2 - 1
        for orig_col in cols:
            ticker = orig_col.split("_")[0]
            asset_features[f"{ticker}_{suffix}"] = rank_df[orig_col]

    return asset_features.dropna(how="all")


def build_macro_features(macro: pd.DataFrame, macro_derived: pd.DataFrame) -> pd.DataFrame:
    """
    Select derived macro features to use as the NCDE control path.
    These are already z-scored and stationary from the DeePM pipeline.
    """
    derived_cols = [c for c in macro_derived.columns if c in [
        "VIX_zscore", "VIX_log", "VIX_chg1d",
        "YC_slope", "YC_slope_zscore", "YC_slope_chg",
        "DGS10_zscore", "DGS10_chg",
        "HY_spread_zscore", "HY_spread_chg",
        "IG_spread_zscore",
        "HY_IG_ratio_zscore",
        "credit_stress",
        "USD_zscore", "USD_chg",
        "OIL_zscore", "OIL_chg",
        "TBILL_daily",
        "macro_stress_composite",
    ]]
    macro_feat = macro_derived[derived_cols].copy()
    macro_feat.index.name = "Date"
    return macro_feat


# ── Continuous path builder (core NCDE ingredient) ────────────────────────────

def build_ncde_path(
    X: np.ndarray,
    t: torch.Tensor = None,
) -> torchcde.CubicSpline:
    """
    Fit a natural cubic spline to a sequence of observations.

    Args:
        X : np.ndarray of shape (batch, time, channels)
        t : optional 1D time tensor (defaults to 0, 1, ..., T-1)

    Returns:
        torchcde.CubicSpline — callable path object X(t) and dX/dt
        used directly by the NCDE solver.
    """
    X_t = torch.tensor(X, dtype=torch.float32)
    if t is None:
        t = torch.arange(X_t.shape[1], dtype=torch.float32)
    coeffs = torchcde.natural_cubic_coeffs(X_t, t)
    return torchcde.CubicSpline(coeffs, t)


# ── Sequence builder ───────────────────────────────────────────────────────────

def build_sequences(
    asset_features:  pd.DataFrame,
    macro_features:  pd.DataFrame,
    tickers:         list,
    lookback:        int,
    target_returns:  pd.DataFrame,
) -> tuple:
    """
    Build sliding-window sequences for NCDE training.

    Returns:
        X_asset  : np.ndarray (N, lookback, n_asset_feats * n_assets)
                   All asset features concatenated along feature dim.
                   The NCDE sees the full multivariate path jointly.
        X_macro  : np.ndarray (N, lookback, n_macro_feats)
                   Control path — macro series interpolated separately.
        y        : np.ndarray (N, n_assets) — next-day simple returns
        dates    : DatetimeIndex of prediction dates (length N)
    """
    common_idx = (
        asset_features.index
        .intersection(macro_features.index)
        .intersection(target_returns.index)
    ).sort_values()

    af = asset_features.reindex(common_idx).ffill().fillna(0.0)
    mf = macro_features.reindex(common_idx).ffill().fillna(0.0)
    tr = target_returns.reindex(common_idx)

    # Build per-asset column index map
    asset_col_indices = []
    for ticker in tickers:
        cols = [c for c in af.columns if c.startswith(ticker + "_")]
        idxs = [af.columns.get_loc(c) for c in cols]
        asset_col_indices.append(idxs)

    n_assets      = len(tickers)
    n_asset_feats = len(asset_col_indices[0])
    n_macro_feats = mf.shape[1]
    N             = len(common_idx) - lookback

    # Concatenate all asset features into one path vector per timestep
    # Shape: (N, lookback, n_assets * n_asset_feats)
    X_asset = np.zeros((N, lookback, n_assets * n_asset_feats), dtype=np.float32)
    X_macro = np.zeros((N, lookback, n_macro_feats),            dtype=np.float32)
    y       = np.zeros((N, n_assets),                           dtype=np.float32)

    af_arr = af.values
    mf_arr = mf.values
    tr_arr = tr.values

    for i in range(N):
        window = slice(i, i + lookback)
        for a, col_idxs in enumerate(asset_col_indices):
            start = a * n_asset_feats
            X_asset[i, :, start:start + n_asset_feats] = af_arr[window][:, col_idxs]
        X_macro[i] = mf_arr[window]
        y[i]       = tr_arr[i + lookback]

    y = np.nan_to_num(y, nan=0.0)
    dates = common_idx[lookback:]

    return X_asset, X_macro, y, dates


# ── Scaler ─────────────────────────────────────────────────────────────────────

class PathScaler:
    """
    RobustScaler fitted on training data only — no data leakage.
    Scales asset path and macro control path independently.
    Reshapes to (N*T, F) for fitting then back to (N, T, F).
    """

    def __init__(self):
        self.asset_scaler = RobustScaler()
        self.macro_scaler = RobustScaler()
        self._fitted      = False
        self._has_macro   = None

    def fit(self, X_asset: np.ndarray, X_macro: np.ndarray):
        N, T, Fa = X_asset.shape
        self.asset_scaler.fit(X_asset.reshape(-1, Fa))
        N, T, Fm = X_macro.shape
        self._has_macro = (Fm > 0)
        if self._has_macro:
            self.macro_scaler.fit(X_macro.reshape(-1, Fm))
        self._fitted = True
        return self

    def transform(self, X_asset: np.ndarray, X_macro: np.ndarray):
        N, T, Fa = X_asset.shape
        Xa = self.asset_scaler.transform(
            X_asset.reshape(-1, Fa)
        ).reshape(N, T, Fa).astype(np.float32)

        N, T, Fm = X_macro.shape
        if self._has_macro and Fm > 0:
            Xm = self.macro_scaler.transform(
                X_macro.reshape(-1, Fm)
            ).reshape(N, T, Fm).astype(np.float32)
        else:
            Xm = X_macro.astype(np.float32)

        return Xa, Xm

    def fit_transform(self, X_asset: np.ndarray, X_macro: np.ndarray):
        return self.fit(X_asset, X_macro).transform(X_asset, X_macro)


# ── Full pipeline ──────────────────────────────────────────────────────────────

def prepare_features(data: dict, lookback: int = None) -> dict:
    """
    Full feature preparation pipeline for one option (A or B).

    Args:
        data     : dict from loader.get_option_data()
        lookback : sequence length (default cfg.LOOKBACK)

    Returns dict with:
        X_asset, X_macro, y, dates,
        tickers, n_assets, n_asset_path_dim, n_macro_feats
    """
    lookback = lookback or cfg.LOOKBACK
    print(f"[features] Building features for Option {data['option']}...")

    asset_feat = build_asset_features(data["log_returns"], data["vol"])
    macro_feat = build_macro_features(data["macro"], data["macro_derived"])

    target_ret = data["returns"].reindex(
        data["returns"].index.intersection(asset_feat.index)
    )

    X_asset, X_macro, y, dates = build_sequences(
        asset_feat, macro_feat, data["tickers"], lookback, target_ret
    )

    print(f"[features] X_asset: {X_asset.shape}, X_macro: {X_macro.shape}, y: {y.shape}")

    return {
        "X_asset":          X_asset,
        "X_macro":          X_macro,
        "y":                y,
        "dates":            dates,
        "tickers":          data["tickers"],
        "n_assets":         len(data["tickers"]),
        "n_asset_path_dim": X_asset.shape[-1],   # n_assets * n_asset_feats_per_ticker
        "n_macro_feats":    X_macro.shape[-1],
        "macro_feat_names": list(macro_feat.columns),
    }
