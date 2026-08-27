import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "dashboard"))
from scenario_engine import categorize_risk, classify_urgency, run_scenario, SCENARIO_PRESETS


def test_categorize_risk_stockout_takes_priority():
    row = pd.Series({"is_stockout": True, "silo_utilization_pct": 0.95})
    assert categorize_risk(row) == "Stockout"


def test_categorize_risk_overcapacity():
    row = pd.Series({"is_stockout": False, "silo_utilization_pct": 0.95})
    assert categorize_risk(row) == "Overcapacity"


def test_categorize_risk_low_stock():
    row = pd.Series({"is_stockout": False, "silo_utilization_pct": 0.05})
    assert categorize_risk(row) == "Low Stock"


def test_categorize_risk_normal():
    row = pd.Series({"is_stockout": False, "silo_utilization_pct": 0.40})
    assert categorize_risk(row) == "Normal"


def test_classify_urgency_emergency_on_negative_position():
    row = pd.Series({"ROP": 100.0, "inventory_position": -5.0})
    assert classify_urgency(row) == "Emergency Reorder"


def test_classify_urgency_place_reorder_near_rop():
    row = pd.Series({"ROP": 100.0, "inventory_position": 90.0})
    assert classify_urgency(row) == "Place Reorder"


def _baseline_scenario():
    demand = np.array([25.0] * 56)
    return run_scenario(
        forecast_demand=demand, silo_capacity=300, rop=100, order_up_to=250,
        opening_inventory=150, lead_time_days=3,
    )


def test_scenario_baseline_produces_sane_output():
    result = _baseline_scenario()
    assert len(result["series"]) == 56
    assert result["minimum_inventory"] >= 0, "inventory should never go negative (physical constraint)"
    assert result["ending_inventory"] <= 300


def test_scenario_demand_surge_increases_risk():
    demand = np.array([25.0] * 56)
    baseline = run_scenario(demand, 300, 100, 250, 150, 3)
    surge = run_scenario(demand, 300, 100, 250, 150, 3, **SCENARIO_PRESETS["Demand Surge"])
    assert surge["total_risk_days"] >= baseline["total_risk_days"], \
        "a demand surge should never produce LESS risk than baseline"
    assert surge["minimum_inventory"] <= baseline["minimum_inventory"]


def test_scenario_supply_disruption_increases_stockouts():
    demand = np.array([25.0] * 56)
    baseline = run_scenario(demand, 300, 100, 250, 150, 3)
    disrupted = run_scenario(demand, 300, 100, 250, 150, 3, **SCENARIO_PRESETS["Supply Disruption"])
    assert disrupted["stockout_days"] >= baseline["stockout_days"], \
        "reduced deliveries + delay should never produce FEWER stockouts than baseline"


def test_scenario_recovery_plan_reduces_or_matches_risk():
    demand = np.array([25.0] * 56)
    baseline = run_scenario(demand, 300, 100, 250, 150, 3)
    recovery = run_scenario(demand, 300, 100, 250, 150, 3, **SCENARIO_PRESETS["Recovery Plan"])
    assert recovery["stockout_days"] <= baseline["stockout_days"]


def test_scenario_reset_to_baseline_matches_zero_adjustment():
    demand = np.array([25.0] * 56)
    baseline = run_scenario(demand, 300, 100, 250, 150, 3)
    reset = run_scenario(demand, 300, 100, 250, 150, 3, **SCENARIO_PRESETS["Reset to Baseline"])
    assert baseline["ending_inventory"] == reset["ending_inventory"]
    assert baseline["stockout_days"] == reset["stockout_days"]


def test_all_presets_have_required_keys():
    for name, params in SCENARIO_PRESETS.items():
        assert set(params.keys()) == {"demand_adj_pct", "delivery_adj_pct", "delivery_delay_days"}, name
