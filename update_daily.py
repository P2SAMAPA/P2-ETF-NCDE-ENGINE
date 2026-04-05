# update_daily.py — P2-ETF-NCDE-ENGINE daily runner
#
# NCDE does NOT maintain any data — it reads from the DeePM dataset.
# This script simply runs predict.py to refresh signals after market close.
# Data is updated upstream by the DeePM daily_update workflow at 22:00 UTC.
# This workflow runs at 23:00 UTC (one hour later) to ensure fresh data.
#
# Usage:
#   python update_daily.py
#   python update_daily.py --option A

import argparse
import os
import sys
import logging
from datetime import datetime

import pandas_market_calendars as mcal
import pandas as pd

import config as cfg
import loader
from predict import generate_signal, save_signals

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("update_daily.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def is_trading_day() -> bool:
    """Return True if today is a NYSE trading day."""
    nyse  = mcal.get_calendar("NYSE")
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    sched = nyse.schedule(start_date=today, end_date=today)
    return not sched.empty


def run(option: str = "both") -> None:
    log.info("=" * 60)
    log.info(f"P2-ETF-NCDE DAILY UPDATE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    log.info("=" * 60)

    if not cfg.HF_TOKEN:
        log.error("HF_TOKEN not set — cannot upload signals.")
        sys.exit(1)

    if not is_trading_day():
        log.info("Not a NYSE trading day — skipping.")
        return

    log.info("Loading master dataset from DeePM source...")
    try:
        master = loader.load_master()
    except Exception as e:
        log.error(f"Failed to load master dataset: {e}")
        sys.exit(1)

    sig_A = sig_B = None

    if option in ("A", "both"):
        try:
            sig_A = generate_signal("A", master)
        except FileNotFoundError as e:
            log.warning(f"Option A model not found: {e}")
        except Exception as e:
            log.error(f"Option A signal generation failed: {e}")

    if option in ("B", "both"):
        try:
            sig_B = generate_signal("B", master)
        except FileNotFoundError as e:
            log.warning(f"Option B model not found: {e}")
        except Exception as e:
            log.error(f"Option B signal generation failed: {e}")

    if sig_A is None and sig_B is None:
        log.error("No signals generated — check models exist (run train.py first).")
        sys.exit(1)

    save_signals(sig_A, sig_B)
    log.info("Daily update complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", choices=["A", "B", "both"], default="both")
    args = parser.parse_args()
    run(args.option)
