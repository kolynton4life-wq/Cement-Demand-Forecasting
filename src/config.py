"""
Central config — single source of truth for paths and constants.
Import this everywhere instead of hardcoding paths/values in scripts.
"""
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "raw" / "MIG_Cement_Records.db"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# --- Forecast settings ---
FORECAST_HORIZON_WEEKS = 8
TARGET_MAPE = 0.15          # 15%
TARGET_SERVICE_LEVEL = 0.98  # 98% pour readiness

# --- Known data caveat (documented in docs/data_dictionary.md) ---
# 'conservative' behavior sites show closing_inventory far exceeding
# silo_capacity (avg ~38x) — a synthetic-data artifact, not a real
# physical state. CAP_INVENTORY_AT_CAPACITY controls whether validate.py
# clips closing_inventory to silo_capacity and reports the excess as
# 'overflow_tonnes' (treated as write-off/waste for KPI purposes).
CAP_INVENTORY_AT_CAPACITY = True
