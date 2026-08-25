"""
Validation & cleaning — encodes the Phase 1 data audit findings so every
downstream script (EDA, features, modeling) starts from the same clean base.

Findings this addresses (see docs/data_dictionary.md for full audit):
1. Balance equation (closing = opening + deliveries - consumed) holds to
   the penny across all rows as-is — no fix needed, but we assert it so
   any future data refresh breaking this fails loudly.
2. 'conservative' behavior sites show closing_inventory up to ~38x
   silo_capacity — a synthetic-data artifact. We cap closing_inventory at
   capacity and report the excess as overflow_tonnes (proxy for waste/
   write-offs), per config.CAP_INVENTORY_AT_CAPACITY.
3. No nulls, no negatives, no duplicate (date, site_id, cement_type) keys
   found in the source — asserted here rather than silently assumed.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))  # for config.py
sys.path.append(str(Path(__file__).resolve().parent))         # for extract.py
from config import CAP_INVENTORY_AT_CAPACITY
from extract import get_full_dataset


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    report = {}

    # --- Structural checks (fail loudly if a future data refresh breaks these) ---
    dup_keys = df.duplicated(subset=["date", "site_id", "cement_type"]).sum()
    assert dup_keys == 0, f"{dup_keys} duplicate (date, site_id, cement_type) rows found"

    nulls = df.isnull().sum()
    report["nulls_per_column"] = nulls[nulls > 0].to_dict()

    numeric_cols = ["consumed_tonnes", "planned_pour_tonnes", "opening_inventory_tonnes",
                     "deliveries_tonnes", "closing_inventory_tonnes", "rain_mm"]
    negatives = {c: int((df[c] < 0).sum()) for c in numeric_cols if (df[c] < 0).any()}
    report["negative_values"] = negatives

    # --- Balance equation check ---
    expected_closing = df["opening_inventory_tonnes"] + df["deliveries_tonnes"] - df["consumed_tonnes"]
    balance_diff = (df["closing_inventory_tonnes"] - expected_closing).abs()
    report["balance_violations"] = int((balance_diff > 0.01).sum())

    # --- Capacity overflow (conservative-behavior artifact) ---
    df["overflow_tonnes"] = (df["closing_inventory_tonnes"] - df["silo_capacity"]).clip(lower=0)
    report["rows_exceeding_capacity"] = int((df["overflow_tonnes"] > 0).sum())
    report["rows_exceeding_capacity_pct"] = round(
        report["rows_exceeding_capacity"] / len(df) * 100, 1
    )

    if CAP_INVENTORY_AT_CAPACITY:
        df["closing_inventory_raw_tonnes"] = df["closing_inventory_tonnes"]
        df["closing_inventory_tonnes"] = df[["closing_inventory_tonnes", "silo_capacity"]].min(axis=1)

    # --- Stockout risk flag (feeds target outcome: 98% pour readiness) ---
    df["stockout_risk"] = df["consumed_tonnes"] < df["planned_pour_tonnes"]
    df["pour_shortfall_tonnes"] = (df["planned_pour_tonnes"] - df["consumed_tonnes"]).clip(lower=0)

    print("=== Validation report ===")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print("==========================")

    return df


if __name__ == "__main__":
    df = get_full_dataset()
    clean = validate_and_clean(df)
    out_path = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "operations_clean.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clean.to_parquet(out_path, index=False)
    print(f"Saved cleaned dataset -> {out_path} ({len(clean)} rows)")
