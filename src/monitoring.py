"""
Step 7 — Monitoring framework.

"Establish monitoring framework to track forecast accuracy and trigger
model retraining when performance degrades beyond acceptable thresholds."

In production, this would run on a schedule (e.g. daily/weekly cron or
task scheduler) once real actuals become available for dates that were
previously forecast — compare forecast vs. actual, log it, and flag
sites whose recent accuracy has degraded past the project's MAPE target.

This module is demonstrated (see __main__) against Step 4's real holdout
data, since no live production feed exists yet — the logic is what
matters and is identical to what would run against real future actuals.
"""
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

MAPE_THRESHOLD = 15.0          # project's stated forecast-accuracy target
DEGRADATION_WINDOW_DAYS = 14   # how many recent days of actuals to check
LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "monitoring_log.jsonl"


def safe_mape(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = actual > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


def check_site_performance(site_id: str, actual: pd.Series, predicted: pd.Series,
                            threshold: float = MAPE_THRESHOLD) -> dict:
    """Core check: given recent actual-vs-forecast pairs for one site,
    compute live MAPE and decide whether retraining should be triggered."""
    mape = safe_mape(actual.values, predicted.values)
    degraded = (not np.isnan(mape)) and mape > threshold

    result = {
        "site_id": site_id,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "live_mape": mape,
        "threshold": threshold,
        "n_days_checked": len(actual),
        "degraded": bool(degraded),
        "action": "TRIGGER_RETRAIN" if degraded else "OK",
    }
    return result


def log_result(result: dict):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(result) + "\n")


def run_monitoring_check(forecasts_df: pd.DataFrame, actual_col: str = "actual",
                          pred_col: str = "rf_pred", threshold: float = MAPE_THRESHOLD) -> pd.DataFrame:
    """Run the check across all sites present in a forecasts-vs-actuals
    dataframe (same shape as model_forecasts_holdout.parquet from Step 4,
    or a live equivalent once deployed)."""
    results = []
    for site_id, g in forecasts_df.groupby("site_id"):
        recent = g.sort_values("date").tail(DEGRADATION_WINDOW_DAYS)
        result = check_site_performance(site_id, recent[actual_col], recent[pred_col], threshold)
        results.append(result)
        log_result(result)

    results_df = pd.DataFrame(results)
    n_triggered = (results_df["action"] == "TRIGGER_RETRAIN").sum()
    print(f"Monitoring check complete: {len(results_df)} sites checked, "
          f"{n_triggered} flagged for retraining (live MAPE > {threshold}%)")
    if n_triggered > 0:
        print("Sites flagged:")
        print(results_df[results_df.action == "TRIGGER_RETRAIN"][["site_id", "live_mape"]].to_string(index=False))
    return results_df


if __name__ == "__main__":
    # Demonstration against Step 4's real holdout data (data/processed/
    # model_forecasts_holdout.parquet) — in production this file would be
    # replaced by a live actual-vs-forecast feed, but the check logic below
    # is exactly what would run against it.
    data_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "model_forecasts_holdout.parquet"
    if not data_path.exists():
        print(f"ERROR: {data_path} not found. Run notebook 04_model_development.ipynb first.")
        sys.exit(1)

    df = pd.read_parquet(data_path)
    df["date"] = pd.to_datetime(df["date"])
    print(f"Demonstrating monitoring check against Step 4 holdout data ({len(df)} rows, "
          f"{df.site_id.nunique()} sites)\n")
    results = run_monitoring_check(df)
    print(f"\nFull results logged to {LOG_PATH}")
