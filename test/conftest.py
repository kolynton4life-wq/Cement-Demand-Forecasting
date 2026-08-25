import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "data"))


@pytest.fixture
def sample_raw_df():
    """Small synthetic (site x date) dataframe matching the real schema,
    for fast unit tests that don't need the full 32,880-row dataset."""
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    rows = []
    for site_id, capacity, behavior in [("TEST_A", 200, "aggressive"), ("TEST_B", 300, "conservative")]:
        opening = 50.0
        for i, d in enumerate(dates):
            consumed = max(10 + 5 * np.sin(i / 7) + np.random.RandomState(i).normal(0, 2), 0)
            delivered = 30.0 if i % 10 == 0 else 0.0
            planned = consumed + np.random.RandomState(i + 1).normal(0, 3)
            closing = opening + delivered - consumed
            rows.append(dict(
                date=d, site_id=site_id, cement_type=["CEM_I", "CEM_II", "CEM_III"][i % 3],
                planned_pour_tonnes=max(planned, 0), consumed_tonnes=consumed,
                opening_inventory_tonnes=opening, deliveries_tonnes=delivered,
                closing_inventory_tonnes=closing, rain_mm=max(np.random.RandomState(i + 2).normal(3, 5), 0),
                avg_temp_c=np.random.RandomState(i + 3).normal(10, 5), silo_capacity=capacity,
                region="South", behavior=behavior,
            ))
            opening = closing
    return pd.DataFrame(rows)
