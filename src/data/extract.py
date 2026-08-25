"""
Extraction — one function per logical source, plus a combined pull.
Keep raw extracts unmodified here; cleaning/validation lives in validate.py.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from connect import get_engine


def get_sites() -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql("SELECT * FROM Sites", engine)


def get_cement_types() -> pd.DataFrame:
    engine = get_engine()
    return pd.read_sql("SELECT * FROM CementTypes", engine)


def get_operations() -> pd.DataFrame:
    """Raw daily fact table: one row per (date, site_id, cement_type)."""
    engine = get_engine()
    df = pd.read_sql("SELECT * FROM Operations", engine)
    df["date"] = pd.to_datetime(df["date"])
    return df


def get_full_dataset() -> pd.DataFrame:
    """
    Operations joined with Sites (region, behavior) — the dataset
    you'll actually build features on. cement_type already lives on
    Operations, so CementTypes is just a lookup/reference table.
    """
    ops = get_operations()
    sites = get_sites()
    df = ops.merge(sites, on="site_id", how="left", suffixes=("", "_site"))
    # sanity: silo_capacity appears on both tables — confirm they agree
    mismatch = (df["silo_capacity"] != df["silo_capacity_site"]).sum()
    if mismatch:
        print(f"WARNING: {mismatch} rows have silo_capacity mismatch between "
              f"Operations and Sites tables — investigate before modeling.")
    df = df.drop(columns=["silo_capacity_site"])
    return df


if __name__ == "__main__":
    df = get_full_dataset()
    print(df.shape)
    print(df.head())
    print(df.dtypes)
