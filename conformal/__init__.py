# conformal/ — Conformal prediction wrapper for P2-ETF-NCDE-ENGINE
#
# Modules
# -------
# calibrate.py         — run once after training; computes q̂ on val set
# predict_conformal.py — run daily after predict.py; wraps signals with intervals
# app_conformal.py     — standalone Streamlit dashboard (NCDE vs NCDE+Conformal)
