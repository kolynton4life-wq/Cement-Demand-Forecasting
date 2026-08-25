"""
Preprocessing — reusable functions refactored from notebooks
01_data_ingestion.ipynb and 03_feature_engineering.ipynb.

The notebooks remain the source of truth for exploration/documentation;
this module exists so `pipeline.py` and tests can call the same logic
programmatically without executing a notebook. Logic here is a direct
port — same column names, same lag windows, same leakage-safe feature
list — kept in sync deliberately, not reimplemented from scratch.
"""
import numpy as np
import pandas as pd

LAG_DAYS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]

# Columns mathematically derived from same-day consumed_tonnes — excluded
# from SAFE_FEATURES because using them as model inputs is target leakage.
LEAKY_COLUMNS = [
    "pour_shortfall_tonnes", "stockout_risk", "planned_actual_ratio",
    "inventory_turnover_ratio", "days_of_stock", "closing_inventory_tonnes",
    "closing_inventory_raw_tonnes", "overflow_tonnes", "silo_utilization_pct",
]

SAFE_FEATURES = [
    "planned_pour_tonnes", "opening_inventory_tonnes", "deliveries_tonnes",
    "rain_mm", "avg_temp_c", "silo_capacity",
    "heavy_rain_flag", "moderate_rain_flag", "frost_risk_flag", "weather_adjusted_planned_pour",
    "adherence_roll_mean_14d", "planned_pour_trust_adjusted",
    "consumed_lag_1d", "consumed_lag_7d", "consumed_lag_14d", "consumed_lag_28d",
    "consumed_roll_mean_7d", "consumed_roll_std_7d", "consumed_roll_mean_14d",
    "consumed_roll_std_14d", "consumed_roll_mean_28d", "consumed_roll_std_28d",
    "type_CEM_I", "type_CEM_II", "type_CEM_III",
]
NON_LAG_FEATURES = [c for c in SAFE_FEATURES if not c.startswith("consumed_")]


def validate_and_clean(df: pd.DataFrame, cap_inventory_at_capacity: bool = True) -> pd.DataFrame:
    """Port of notebook 01's validation logic: dedup check, balance
    equation check/correction, capacity capping, stockout flag."""
    df = df.copy()
    dup_keys = df.duplicated(subset=["date", "site_id", "cement_type"]).sum()
    assert dup_keys == 0, f"{dup_keys} duplicate (date, site_id, cement_type) rows found"

    non_negative_cols = ["consumed_tonnes", "planned_pour_tonnes", "opening_inventory_tonnes",
                          "deliveries_tonnes", "closing_inventory_tonnes", "rain_mm"]
    for c in non_negative_cols:
        df[c] = df[c].clip(lower=0)

    # Physical constraint: consumed_tonnes cannot exceed what was actually
    # available that day (opening + deliveries) — a recorded value above
    # that is a data error, not a real event, since you cannot consume
    # material you don't have. Caught by the pytest suite on adversarial
    # synthetic data (never triggers on the real MIG dataset — Phase 1
    # found 0 negative closing-inventory rows — but the pipeline must be
    # correct for messier data too, e.g. once real production data flows
    # in that may not be as clean as this synthetic set).
    available = df["opening_inventory_tonnes"] + df["deliveries_tonnes"]
    over_available = df["consumed_tonnes"] > available + 0.01  # same rounding tolerance as the balance-equation check below
    if over_available.any():
        df.loc[over_available, "consumed_tonnes"] = available[over_available]

    expected_closing = df["opening_inventory_tonnes"] + df["deliveries_tonnes"] - df["consumed_tonnes"]
    balance_diff = (df["closing_inventory_tonnes"] - expected_closing).abs()
    df.loc[balance_diff > 0.01, "closing_inventory_tonnes"] = expected_closing[balance_diff > 0.01]
    # expected_closing is now guaranteed >= 0 by the consumed_tonnes cap
    # above, so no separate post-hoc clip on closing_inventory is needed.

    df["overflow_tonnes"] = (df["closing_inventory_tonnes"] - df["silo_capacity"]).clip(lower=0)
    if cap_inventory_at_capacity:
        df["closing_inventory_raw_tonnes"] = df["closing_inventory_tonnes"]
        df["closing_inventory_tonnes"] = df[["closing_inventory_tonnes", "silo_capacity"]].min(axis=1)

    df["stockout_risk"] = df["consumed_tonnes"] < df["planned_pour_tonnes"]
    df["pour_shortfall_tonnes"] = (df["planned_pour_tonnes"] - df["consumed_tonnes"]).clip(lower=0)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Port of notebook 03's feature engineering. Grouping key is
    site_id ONLY (not site_id+cement_type) — each site logs exactly one
    row per calendar day; cement_type is a same-day attribute, not a
    parallel series. Grouping by both would fragment each site's
    continuous daily series (the bug caught and fixed during Step 4
    cross-review — see notebook 03 Section 1 for the full explanation).
    """
    df = df.sort_values(["site_id", "date"]).reset_index(drop=True)
    group = df.groupby("site_id")

    for lag in LAG_DAYS:
        df[f"consumed_lag_{lag}d"] = group["consumed_tonnes"].shift(lag)
    for window in ROLLING_WINDOWS:
        df[f"consumed_roll_mean_{window}d"] = (
            group["consumed_tonnes"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"consumed_roll_std_{window}d"] = (
            group["consumed_tonnes"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).std())
        )

    df["heavy_rain_flag"] = (df["rain_mm"] >= 15).astype(int)
    df["moderate_rain_flag"] = ((df["rain_mm"] >= 5) & (df["rain_mm"] < 15)).astype(int)
    df["frost_risk_flag"] = (df["avg_temp_c"] <= 5).astype(int)
    df["weather_adjusted_planned_pour"] = df["planned_pour_tonnes"] * np.where(
        df["heavy_rain_flag"] == 1, 0.3, np.where(df["moderate_rain_flag"] == 1, 0.85, 1.0)
    )

    df["planned_actual_ratio"] = df["consumed_tonnes"] / df["planned_pour_tonnes"].replace(0, np.nan)
    df["adherence_roll_mean_14d"] = (
        group["planned_actual_ratio"].transform(lambda s: s.shift(1).rolling(14, min_periods=3).mean())
    )
    df["planned_pour_trust_adjusted"] = (
        df["planned_pour_tonnes"] * df["adherence_roll_mean_14d"].fillna(df["adherence_roll_mean_14d"].median())
    )

    df["inventory_turnover_ratio"] = df["consumed_tonnes"] / df["opening_inventory_tonnes"].replace(0, np.nan)
    df["days_of_stock"] = df["closing_inventory_tonnes"] / df["consumed_roll_mean_7d"].replace(0, np.nan)
    df["silo_utilization_pct"] = df["closing_inventory_tonnes"] / df["silo_capacity"]

    df = pd.get_dummies(df, columns=["behavior", "cement_type"], prefix=["behavior", "type"], drop_first=False)
    for b_col in [c for c in df.columns if c.startswith("behavior_")]:
        b_name = b_col.replace("behavior_", "")
        df[f"heavy_rain_x_{b_name}"] = df["heavy_rain_flag"] * df[b_col]
        df[f"planned_pour_x_{b_name}"] = df["planned_pour_tonnes"] * df[b_col]

    return df


def run_preprocessing(raw_ops_sites_joined: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing entry point: ingestion validation -> feature
    engineering. Input must already be the Operations JOIN Sites result
    (see src/data/extract.py::get_full_dataset)."""
    df = raw_ops_sites_joined.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = validate_and_clean(df)
    df = engineer_features(df)
    return df
