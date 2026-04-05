# validate_dataset.py — Sanity check for P2-ETF-NCDE-ENGINE
#
# Validates that the source dataset loads correctly and has the
# expected columns and shape before training.
#
# Usage:
#   python validate_dataset.py

import sys
import numpy as np
import config as cfg
import loader
import features as feat


def validate():
    print("=" * 60)
    print("P2-ETF-NCDE-ENGINE — Dataset Validation")
    print("=" * 60)

    errors   = []
    warnings = []

    # 1. Load master
    print("\n[1/4] Loading master dataset...")
    try:
        master = loader.load_master()
        print(f"  OK — shape: {master.shape}")
        print(f"       range: {master.index[0].date()} -> {master.index[-1].date()}")
    except Exception as e:
        errors.append(f"Failed to load master: {e}")
        _report(errors, warnings)
        return

    if len(master) < 1000:
        errors.append(f"Master too small: {len(master)} rows (expected >1000)")

    # 2. Option A data
    print("\n[2/4] Validating Option A (Fixed Income)...")
    try:
        data_A = loader.get_option_data("A", master)
        for key in ["prices", "returns", "log_returns", "vol", "macro", "macro_derived"]:
            df = data_A[key]
            n_nan = df.isna().sum().sum() if hasattr(df, "isna") else 0
            print(f"  {key:15s}: shape={df.shape}, NaN={n_nan}")
        print(f"  Tickers: {data_A['tickers']}")
    except Exception as e:
        errors.append(f"Option A data error: {e}")

    # 3. Option B data
    print("\n[3/4] Validating Option B (Equity)...")
    try:
        data_B = loader.get_option_data("B", master)
        for key in ["prices", "returns", "log_returns", "vol", "macro", "macro_derived"]:
            df = data_B[key]
            n_nan = df.isna().sum().sum() if hasattr(df, "isna") else 0
            print(f"  {key:15s}: shape={df.shape}, NaN={n_nan}")
        print(f"  Tickers: {data_B['tickers']}")
    except Exception as e:
        errors.append(f"Option B data error: {e}")

    # 4. Feature pipeline
    print("\n[4/4] Validating feature pipeline (Option A sample)...")
    try:
        feat_dict = feat.prepare_features(data_A, lookback=cfg.LOOKBACK)
        print(f"  X_asset : {feat_dict['X_asset'].shape}")
        print(f"  X_macro : {feat_dict['X_macro'].shape}")
        print(f"  y       : {feat_dict['y'].shape}")
        print(f"  dates   : {feat_dict['dates'][0].date()} -> {feat_dict['dates'][-1].date()}")

        # Check for NaN/Inf in features
        if np.isnan(feat_dict["X_asset"]).any():
            warnings.append("X_asset contains NaN — check feature engineering")
        if np.isinf(feat_dict["X_asset"]).any():
            warnings.append("X_asset contains Inf — check vol scaling")
        if np.isnan(feat_dict["X_macro"]).any():
            warnings.append("X_macro contains NaN — check macro derived features")

        # Confirm spline builds without error (small batch)
        import torch
        import torchcde
        Xa = torch.tensor(feat_dict["X_asset"][:4], dtype=torch.float32)
        Xm = torch.tensor(feat_dict["X_macro"][:4], dtype=torch.float32)
        t  = torch.arange(Xa.shape[1], dtype=torch.float32)
        _ = torchcde.CubicSpline(torchcde.natural_cubic_coeffs(Xa, t), t)
        _ = torchcde.CubicSpline(torchcde.natural_cubic_coeffs(Xm, t), t)
        print("  Spline build: OK")

    except Exception as e:
        errors.append(f"Feature pipeline error: {e}")

    _report(errors, warnings)


def _report(errors, warnings):
    print("\n" + "=" * 60)
    if warnings:
        for w in warnings:
            print(f"  WARNING: {w}")
    if errors:
        for e in errors:
            print(f"  ERROR:   {e}")
        print("\nValidation FAILED.")
        sys.exit(1)
    else:
        print("  All checks passed.")
        print("\nValidation PASSED — ready to train.")


if __name__ == "__main__":
    validate()
