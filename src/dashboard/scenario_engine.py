"""
Shared logic for both dashboards: risk categorization of inventory
site-days, and the Scenario Simulator's live what-if recompute.

The what-if engine reuses the exact same (s, S) reorder-point policy as
notebooks/05_inventory_simulation.ipynb and src/pipeline.py — same
formula, just parameterized so sliders can perturb demand, delivery
size, and lead time without retraining anything. This keeps the
dashboard's "what if" numbers consistent with the real simulation
rather than a simplified stand-in.
"""
import numpy as np
import pandas as pd

# Thresholds derived from the real inventory_simulation.parquet distribution
# (see the percentile check done before building this module): overcapacity
# at the ~98th percentile of utilization, low-stock near the 25th percentile.
OVERCAPACITY_THRESHOLD = 0.90
LOW_STOCK_THRESHOLD = 0.15


def categorize_risk(row) -> str:
    """One row of inventory_simulation.parquet -> a risk label."""
    if row["is_stockout"]:
        return "Stockout"
    if row["silo_utilization_pct"] >= OVERCAPACITY_THRESHOLD:
        return "Overcapacity"
    if row["silo_utilization_pct"] < LOW_STOCK_THRESHOLD:
        return "Low Stock"
    return "Normal"


def add_risk_category(inv_sim: pd.DataFrame) -> pd.DataFrame:
    df = inv_sim.copy()
    df["risk_category"] = df.apply(categorize_risk, axis=1)
    return df


def classify_urgency(row) -> str:
    """Reorder alert urgency, based on how far inventory position sits
    below the reorder point at the moment the order was placed."""
    gap = row["ROP"] - row["inventory_position"]
    if gap > row["ROP"] * 0.5 or row["inventory_position"] <= 0:
        return "Emergency Reorder"
    return "Place Reorder"


def run_scenario(
    forecast_demand: np.ndarray,
    silo_capacity: float,
    rop: float,
    order_up_to: float,
    opening_inventory: float,
    lead_time_days: int,
    demand_adj_pct: float = 0.0,
    delivery_adj_pct: float = 0.0,
    delivery_delay_days: int = 0,
) -> dict:
    """Re-run the (s, S) reorder-point simulation with perturbed
    parameters. Same loop structure as notebooks/05 and pipeline.py's
    simulate_site — see those for the unperturbed baseline version.

    Returns per-day series (for charting) plus summary KPIs.
    """
    adj_demand = forecast_demand * (1 + demand_adj_pct / 100.0)
    effective_lead_time = max(int(lead_time_days) + int(delivery_delay_days), 1)
    delivery_scale = 1 + delivery_adj_pct / 100.0

    opening = opening_inventory
    on_order = []
    rows = []
    stockout_days = 0
    overcapacity_days = 0

    for t in range(len(adj_demand)):
        delivered_today = sum(q for (arr, q) in on_order if arr == t)
        on_order = [(arr, q) for (arr, q) in on_order if arr != t]
        available = opening + delivered_today
        demand = max(adj_demand[t], 0)
        consumed = min(demand, available)
        is_stockout = demand > available + 1e-6
        stockout_days += int(is_stockout)

        raw_closing = available - consumed
        is_overcapacity = raw_closing >= silo_capacity * OVERCAPACITY_THRESHOLD
        overcapacity_days += int(is_overcapacity)
        closing = min(raw_closing, silo_capacity)

        in_transit = sum(q for (arr, q) in on_order)
        inv_position = closing + in_transit
        if inv_position <= rop:
            order_qty = max((order_up_to - inv_position), 0) * delivery_scale
            on_order.append((t + effective_lead_time, order_qty))

        rows.append(dict(day=t, opening=opening, demand=demand, consumed=consumed, closing=closing))
        opening = closing

    sim_df = pd.DataFrame(rows)
    return {
        "series": sim_df,
        "ending_inventory": float(sim_df["closing"].iloc[-1]),
        "minimum_inventory": float(sim_df["closing"].min()),
        "stockout_days": int(stockout_days),
        "overcapacity_days": int(overcapacity_days),
        "total_risk_days": int(stockout_days + overcapacity_days),
    }


SCENARIO_PRESETS = {
    "Demand Surge": dict(demand_adj_pct=20, delivery_adj_pct=0, delivery_delay_days=0),
    "Supply Disruption": dict(demand_adj_pct=0, delivery_adj_pct=-25, delivery_delay_days=3),
    "Recovery Plan": dict(demand_adj_pct=0, delivery_adj_pct=25, delivery_delay_days=0),
    "Reset to Baseline": dict(demand_adj_pct=0, delivery_adj_pct=0, delivery_delay_days=0),
}
