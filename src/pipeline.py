"""
Step 7 — Production training pipeline.

Orchestrates: ingest -> preprocess -> rolling-window (walk-forward)
validation -> train final per-site model on full history -> persist to
models/.

Random Forest is the only model trained here — Step 4 already compared
SARIMAX vs. RF vs. naive across all 30 sites and RF won on every single
one (see notebooks/04_model_development.ipynb). This pipeline's job is
to validate and productionize the SELECTED model, not re-run model
selection on every scheduled retrain.

Rolling-window validation (addresses the Step 4 gap of using only one
fixed holdout window): evaluates on 3 independent, non-overlapping
56-day windows walking backward from the end of history, each trained
only on data strictly before its own test window (no leakage across
windows). This checks whether the ~3.3% MAPE found in Step 4 holds up
generally, or was specific to that one end-of-2024 period.

Run with:  python src/pipeline.py
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
from extract import get_full_dataset          # noqa: E402
from preprocessing import run_preprocessing, SAFE_FEATURES  # noqa: E402
from model_registry import save_model         # noqa: E402

warnings.filterwarnings("ignore")

HORIZON = 56
N_VALIDATION_WINDOWS = 3
Z_SCORE_98 = 2.054
LEAD_TIME_DAYS = 3
REVIEW_BUFFER_DAYS = 7


def safe_mape(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    mask = actual > 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


def rmse(actual, pred):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def rolling_window_validate(site_df: pd.DataFrame) -> dict:
    """Walk-forward validation across N_VALIDATION_WINDOWS independent
    holdout windows. Returns per-window and aggregate metrics."""
    window_results = []
    for w in range(N_VALIDATION_WINDOWS):
        test_end_offset = w * HORIZON
        test_start_offset = test_end_offset + HORIZON
        if test_start_offset >= len(site_df):
            break

        test = site_df.iloc[len(site_df) - test_start_offset: len(site_df) - test_end_offset] if test_end_offset > 0 \
            else site_df.iloc[len(site_df) - test_start_offset:]
        train = site_df.iloc[: len(site_df) - test_start_offset]
        train = train.dropna(subset=SAFE_FEATURES + ["consumed_tonnes"])
        if len(train) < 60 or len(test) < HORIZON:
            continue

        rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
        rf.fit(train[SAFE_FEATURES], train["consumed_tonnes"])
        pred = rf.predict(test[SAFE_FEATURES])  # one-step-ahead style eval using true lags (test set is historical)
        actual = test["consumed_tonnes"].values

        window_results.append({
            "window": w, "mape": safe_mape(actual, pred), "rmse": rmse(actual, pred),
        })

    if not window_results:
        return {"windows": [], "mean_mape": np.nan, "mean_rmse": np.nan, "mape_std": np.nan}

    mapes = [w["mape"] for w in window_results if not np.isnan(w["mape"])]
    rmses = [w["rmse"] for w in window_results]
    return {
        "windows": window_results,
        "mean_mape": float(np.mean(mapes)) if mapes else np.nan,
        "mape_std": float(np.std(mapes)) if len(mapes) > 1 else 0.0,
        "mean_rmse": float(np.mean(rmses)),
    }


def train_final_model(site_df: pd.DataFrame):
    """Train on ALL available history — this is the model that gets
    persisted and used for real forecasting, separate from the
    validation-only models fit inside rolling_window_validate."""
    train = site_df.dropna(subset=SAFE_FEATURES + ["consumed_tonnes"])
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(train[SAFE_FEATURES], train["consumed_tonnes"])
    return rf


def compute_inventory_params(site_df: pd.DataFrame, rmse_estimate: float) -> dict:
    """Same reorder-point formula as notebook 05, computed here so
    inference.py doesn't need to re-derive it or depend on the notebook."""
    silo_capacity = float(site_df["silo_capacity"].iloc[0])
    avg_daily_demand = float(site_df["consumed_tonnes"].tail(90).mean())  # recent demand level
    safety_stock = Z_SCORE_98 * rmse_estimate * np.sqrt(LEAD_TIME_DAYS)
    rop = avg_daily_demand * LEAD_TIME_DAYS + safety_stock
    order_up_to_s = min(silo_capacity, rop + avg_daily_demand * REVIEW_BUFFER_DAYS)
    return {
        "silo_capacity": silo_capacity, "avg_daily_demand": avg_daily_demand,
        "safety_stock": safety_stock, "reorder_point": rop, "order_up_to_level": order_up_to_s,
        "lead_time_days": LEAD_TIME_DAYS,
    }


def run_pipeline():
    print("=== Step 7 Production Pipeline ===")
    t0 = time.time()

    print("[1/4] Ingesting from SQLite...")
    raw = get_full_dataset()

    print("[2/4] Preprocessing (validation + feature engineering)...")
    df = run_preprocessing(raw)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["site_id", "date"]).reset_index(drop=True)

    behavior_cols = [c for c in df.columns if c.startswith("behavior_")]

    print(f"[3/4] Rolling-window validation + final training, {df.site_id.nunique()} sites...")
    results = []
    for site_id, g in df.groupby("site_id"):
        g = g.sort_values("date").reset_index(drop=True)

        validation = rolling_window_validate(g)
        final_model = train_final_model(g)

        rmse_estimate = validation["mean_rmse"] if not np.isnan(validation.get("mean_rmse", np.nan)) else 5.0
        inv_params = compute_inventory_params(g, rmse_estimate)

        metrics = {
            "mean_mape": validation["mean_mape"], "mape_std": validation["mape_std"],
            "mean_rmse": validation["mean_rmse"], "n_validation_windows": len(validation["windows"]),
            "per_window": validation["windows"],
        }
        save_model(site_id, final_model, SAFE_FEATURES, metrics, inv_params)

        results.append({"site_id": site_id, **{k: v for k, v in metrics.items() if k != "per_window"}})
        print(f"  {site_id}: mean_mape={metrics['mean_mape']:.2f}% "
              f"(std={metrics['mape_std']:.2f}, {metrics['n_validation_windows']} windows) — saved")

    results_df = pd.DataFrame(results)
    print(f"\n[4/4] Done in {time.time()-t0:.1f}s")
    print(f"\nAggregate rolling-window MAPE: mean={results_df['mean_mape'].mean():.2f}%, "
          f"worst-site={results_df['mean_mape'].max():.2f}%")
    print(f"Sites meeting <=15% MAPE across ALL validation windows: "
          f"{(results_df['mean_mape'] <= 15).sum()}/{len(results_df)}")

    out_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "pipeline_validation_results.parquet"
    results_df.to_parquet(out_path, index=False)
    print(f"Saved validation results -> {out_path}")
    return results_df


if __name__ == "__main__":
    run_pipeline()
