"""
Streamlit dashboard — deployment-target-specific alternative to the
Plotly Dash app (src/dashboard/app.py), which remains the project spec's
official Step 6 deliverable ("Develop Plotly Dash application..."). This
file exists in addition to that, not instead of it, to match this
project's Streamlit-based deployment convention.

Same underlying data, same two views (Overview, Site Detail) as the Dash
app — kept in parity deliberately so neither dashboard drifts out of
sync with the other's feature set.

Run with:  streamlit run src/dashboard/streamlit_app.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

REQUIRED_FILES = [
    "operations_clean.parquet", "model_results_by_site.parquet",
    "model_forecasts_holdout.parquet", "inventory_simulation.parquet",
    "inventory_summary_by_site.parquet", "reorder_alerts.parquet",
]

st.set_page_config(page_title="MIG Cement Demand Forecasting", layout="wide")

missing = [f for f in REQUIRED_FILES if not (DATA_DIR / f).exists()]
if missing:
    st.error(
        f"Missing required data files in `{DATA_DIR}`:\n\n"
        + "\n".join(f"- {f}" for f in missing)
        + "\n\nRun notebooks `01_data_ingestion` → `03_feature_engineering` → "
          "`04_model_development` → `05_inventory_simulation` first (in that order)."
    )
    st.stop()


@st.cache_data
def load_data():
    ops = pd.read_parquet(DATA_DIR / "operations_clean.parquet")
    ops["date"] = pd.to_datetime(ops["date"])

    model_results = pd.read_parquet(DATA_DIR / "model_results_by_site.parquet")

    forecasts_holdout = pd.read_parquet(DATA_DIR / "model_forecasts_holdout.parquet")
    forecasts_holdout["date"] = pd.to_datetime(forecasts_holdout["date"])

    inv_sim = pd.read_parquet(DATA_DIR / "inventory_simulation.parquet")
    inv_sim["date"] = pd.to_datetime(inv_sim["date"])

    inv_summary = pd.read_parquet(DATA_DIR / "inventory_summary_by_site.parquet")

    reorder_alerts = pd.read_parquet(DATA_DIR / "reorder_alerts.parquet")
    reorder_alerts["date"] = pd.to_datetime(reorder_alerts["date"])

    site_meta = ops.drop_duplicates("site_id").set_index("site_id")[["region", "behavior", "silo_capacity"]]
    return ops, model_results, forecasts_holdout, inv_sim, inv_summary, reorder_alerts, site_meta


ops, model_results, forecasts_holdout, inv_sim, inv_summary, reorder_alerts, site_meta = load_data()
site_ids = sorted(model_results["site_id"].unique())

MAPE_TARGET = 15.0
READINESS_TARGET = 0.98

st.title("MIG Cement Demand Forecasting — Operations Dashboard")
st.caption("Forecasts, inventory projections, and reorder alerts by site.")

tab_overview, tab_detail = st.tabs(["Overview (all sites)", "Site Detail (drill-down)"])

# ---------------------------------------------------------------
with tab_overview:
    mean_mape = model_results["rf_mape"].mean()
    n_meeting_readiness = (inv_summary["pour_readiness_post_warmup"] >= READINESS_TARGET).sum()
    mean_readiness = inv_summary["pour_readiness_post_warmup"].mean()
    total_alerts = len(reorder_alerts)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Forecast MAPE (RF)", f"{mean_mape:.1f}%", help=f"Target ≤ {MAPE_TARGET:.0f}%")
    c2.metric("Avg Pour Readiness", f"{mean_readiness*100:.1f}%",
              help=f"{n_meeting_readiness}/{len(inv_summary)} sites ≥ 98%")
    c3.metric("Avg Silo Utilization", f"{inv_summary['avg_silo_utilization'].mean()*100:.1f}%")
    c4.metric("Reorder Alerts (8-wk horizon)", f"{total_alerts}", help="across all sites")

    col_a, col_b = st.columns(2)

    with col_a:
        fig_mape = go.Figure()
        for behavior, grp in model_results.groupby("behavior"):
            fig_mape.add_trace(go.Bar(x=grp["site_id"], y=grp["rf_mape"], name=behavior))
        fig_mape.add_hline(y=MAPE_TARGET, line_dash="dash", line_color="red",
                            annotation_text=f"Target ≤{MAPE_TARGET:.0f}%")
        fig_mape.update_layout(title="Forecast MAPE by site (Random Forest)", barmode="group",
                                yaxis_title="MAPE (%)", height=380)
        st.plotly_chart(fig_mape, width='stretch')

    with col_b:
        fig_readiness = go.Figure()
        for behavior, grp in inv_summary.groupby("behavior"):
            fig_readiness.add_trace(go.Bar(x=grp["site_id"], y=grp["pour_readiness_post_warmup"] * 100, name=behavior))
        fig_readiness.add_hline(y=98, line_dash="dash", line_color="red", annotation_text="Target ≥98%")
        fig_readiness.update_layout(title="Pour readiness by site (post-warmup)", barmode="group",
                                     yaxis_title="Pour readiness (%)", height=380)
        st.plotly_chart(fig_readiness, width='stretch')

    st.subheader("Upcoming Reorder Alerts (all sites)")
    display_alerts = reorder_alerts.sort_values("date").assign(
        date=lambda d: d["date"].dt.strftime("%Y-%m-%d"),
        order_placed=lambda d: d["order_placed"].round(1),
    )[["date", "site_id", "behavior", "order_placed", "inventory_position"]]
    st.dataframe(display_alerts, width='stretch', height=350)

# ---------------------------------------------------------------
with tab_detail:
    site_id = st.selectbox("Select site:", site_ids)

    meta = site_meta.loc[site_id]
    result = model_results[model_results.site_id == site_id].iloc[0]
    summary = inv_summary[inv_summary.site_id == site_id].iloc[0]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Region / Behavior", f"{meta['region']} / {meta['behavior']}")
    c2.metric("Silo Capacity", f"{meta['silo_capacity']:.0f} t")
    c3.metric("Forecast MAPE (RF)", f"{result['rf_mape']:.1f}%")
    c4.metric("Reorder Point", f"{summary['ROP']:.0f} t")
    c5.metric("Pour Readiness", f"{summary['pour_readiness_post_warmup']*100:.1f}%")

    d_fc = forecasts_holdout[forecasts_holdout.site_id == site_id].sort_values("date")
    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=d_fc["date"], y=d_fc["actual"], name="Actual",
                                       line=dict(color="black", width=2)))
    fig_forecast.add_trace(go.Scatter(x=d_fc["date"], y=d_fc["rf_pred"], name="Random Forest forecast",
                                       line=dict(color="green")))
    fig_forecast.add_trace(go.Scatter(x=d_fc["date"], y=d_fc["sarimax_pred"], name="SARIMAX forecast",
                                       line=dict(color="orange", dash="dot")))
    fig_forecast.update_layout(title=f"Forecast vs. actual (holdout period) — {site_id}",
                                yaxis_title="Tonnes/day", height=380)
    st.plotly_chart(fig_forecast, width='stretch')

    d_inv = inv_sim[inv_sim.site_id == site_id].sort_values("date")
    fig_inv = go.Figure()
    fig_inv.add_trace(go.Scatter(x=d_inv["date"], y=d_inv["closing_inventory"],
                                  name="Projected closing inventory", line=dict(color="steelblue")))
    fig_inv.add_hline(y=summary["ROP"], line_dash="dash", line_color="orange", annotation_text="Reorder point")
    fig_inv.add_hline(y=summary["silo_capacity"], line_dash="dot", line_color="red", annotation_text="Silo capacity")
    for sd in d_inv[d_inv.is_stockout]["date"]:
        fig_inv.add_vline(x=sd, line_color="red", opacity=0.15)
    fig_inv.update_layout(title=f"Projected silo level, next 8 weeks — {site_id}",
                           yaxis_title="Tonnes", height=380)
    st.plotly_chart(fig_inv, width='stretch')

    st.subheader("Reorder Alerts — this site")
    site_alerts = reorder_alerts[reorder_alerts.site_id == site_id].sort_values("date")
    if site_alerts.empty:
        st.info("No reorder alerts projected for this site in the 8-week horizon.")
    else:
        display_site_alerts = site_alerts.assign(
            date=lambda x: x["date"].dt.strftime("%Y-%m-%d"),
            order_placed=lambda x: x["order_placed"].round(1),
        )[["date", "order_placed", "inventory_position", "closing_inventory"]]
        st.dataframe(display_site_alerts, width='stretch')
