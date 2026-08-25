"""
Step 6 — Dashboard Application
MIG Cement Demand Forecasting

Per the project spec: "Develop Plotly Dash application with interactive
visualizations for forecasts, inventory projections, and reorder alerts.
Enable site-level drill-down and aggregate views for operations
management."

Run with:  python app.py
Then open: http://127.0.0.1:8050

Reads pre-computed outputs from ../../data/processed/ (produced by
notebooks 01, 03, 04, 05 — run those first if any file is missing).
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

REQUIRED_FILES = [
    "operations_clean.parquet", "model_results_by_site.parquet",
    "model_forecasts_holdout.parquet", "inventory_simulation.parquet",
    "inventory_summary_by_site.parquet", "reorder_alerts.parquet",
]
missing = [f for f in REQUIRED_FILES if not (DATA_DIR / f).exists()]
if missing:
    print(f"ERROR: missing required data files in {DATA_DIR}:")
    for f in missing:
        print(f"  - {f}")
    print("\nRun notebooks 01_data_ingestion -> 03_feature_engineering -> "
          "04_model_development -> 05_inventory_simulation first (in that order).")
    sys.exit(1)

# ---------------------------------------------------------------
# Load data once at startup
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

site_ids = sorted(model_results["site_id"].unique())
site_meta = ops.drop_duplicates("site_id").set_index("site_id")[["region", "behavior", "silo_capacity"]]

MAPE_TARGET = 15.0
READINESS_TARGET = 0.98

# ---------------------------------------------------------------
app = Dash(__name__)
app.title = "MIG Cement Demand Forecasting"

KPI_STYLE = {
    "border": "1px solid #ddd", "borderRadius": "8px", "padding": "16px",
    "textAlign": "center", "flex": "1", "margin": "0 8px", "backgroundColor": "#fafafa",
}


def kpi_card(label, value, sub=""):
    return html.Div([
        html.Div(label, style={"fontSize": "13px", "color": "#666"}),
        html.Div(value, style={"fontSize": "28px", "fontWeight": "bold", "margin": "4px 0"}),
        html.Div(sub, style={"fontSize": "12px", "color": "#999"}),
    ], style=KPI_STYLE)


# ---------------------------------------------------------------
# Overview tab

def build_overview():
    mean_mape = model_results["rf_mape"].mean()
    n_meeting_readiness = (inv_summary["pour_readiness_post_warmup"] >= READINESS_TARGET).sum()
    mean_readiness = inv_summary["pour_readiness_post_warmup"].mean()
    total_alerts = len(reorder_alerts)

    kpi_row = html.Div([
        kpi_card("Avg Forecast MAPE (RF)", f"{mean_mape:.1f}%", f"target ≤ {MAPE_TARGET:.0f}%"),
        kpi_card("Avg Pour Readiness", f"{mean_readiness*100:.1f}%",
                  f"{n_meeting_readiness}/{len(inv_summary)} sites ≥ 98%"),
        kpi_card("Avg Silo Utilization", f"{inv_summary['avg_silo_utilization'].mean()*100:.1f}%"),
        kpi_card("Reorder Alerts (8-wk horizon)", f"{total_alerts}", "across all sites"),
    ], style={"display": "flex", "marginBottom": "24px"})

    fig_mape = go.Figure()
    for behavior, grp in model_results.groupby("behavior"):
        fig_mape.add_trace(go.Bar(x=grp["site_id"], y=grp["rf_mape"], name=behavior))
    fig_mape.add_hline(y=MAPE_TARGET, line_dash="dash", line_color="red",
                        annotation_text=f"Target ≤{MAPE_TARGET:.0f}%")
    fig_mape.update_layout(title="Forecast MAPE by site (Random Forest)", barmode="group",
                            yaxis_title="MAPE (%)", height=380)

    fig_readiness = go.Figure()
    for behavior, grp in inv_summary.groupby("behavior"):
        fig_readiness.add_trace(go.Bar(x=grp["site_id"], y=grp["pour_readiness_post_warmup"] * 100, name=behavior))
    fig_readiness.add_hline(y=98, line_dash="dash", line_color="red", annotation_text="Target ≥98%")
    fig_readiness.update_layout(title="Pour readiness by site (post-warmup)", barmode="group",
                                 yaxis_title="Pour readiness (%)", height=380)

    alert_table = dash_table.DataTable(
        data=reorder_alerts.sort_values("date").assign(
            date=lambda d: d["date"].dt.strftime("%Y-%m-%d"),
            order_placed=lambda d: d["order_placed"].round(1),
        )[["date", "site_id", "behavior", "order_placed", "inventory_position"]].to_dict("records"),
        columns=[
            {"name": "Date", "id": "date"}, {"name": "Site", "id": "site_id"},
            {"name": "Behavior", "id": "behavior"}, {"name": "Order Qty (t)", "id": "order_placed"},
            {"name": "Inv. Position (t)", "id": "inventory_position"},
        ],
        page_size=10, sort_action="native", filter_action="native",
        style_table={"overflowX": "auto"}, style_cell={"padding": "6px", "fontSize": "13px"},
    )

    return html.Div([
        kpi_row,
        html.Div([
            html.Div(dcc.Graph(figure=fig_mape), style={"flex": "1"}),
            html.Div(dcc.Graph(figure=fig_readiness), style={"flex": "1"}),
        ], style={"display": "flex", "gap": "16px"}),
        html.H4("Upcoming Reorder Alerts (all sites)", style={"marginTop": "24px"}),
        alert_table,
    ])


# ---------------------------------------------------------------
# Site Detail tab

def build_site_detail_layout():
    return html.Div([
        html.Div([
            html.Label("Select site:"),
            dcc.Dropdown(id="site-dropdown", options=[{"label": s, "value": s} for s in site_ids],
                         value=site_ids[0], clearable=False, style={"width": "300px"}),
        ], style={"marginBottom": "16px"}),
        html.Div(id="site-info-panel", style={"marginBottom": "16px"}),
        dcc.Graph(id="forecast-chart"),
        dcc.Graph(id="inventory-chart"),
        html.H4("Reorder Alerts — this site"),
        html.Div(id="site-alert-table"),
    ])


@app.callback(Output("site-info-panel", "children"), Input("site-dropdown", "value"))
def update_site_info(site_id):
    meta = site_meta.loc[site_id]
    result = model_results[model_results.site_id == site_id].iloc[0]
    summary = inv_summary[inv_summary.site_id == site_id].iloc[0]
    return html.Div([
        kpi_card("Region / Behavior", f"{meta['region']} / {meta['behavior']}"),
        kpi_card("Silo Capacity", f"{meta['silo_capacity']:.0f} t"),
        kpi_card("Forecast MAPE (RF)", f"{result['rf_mape']:.1f}%"),
        kpi_card("Reorder Point", f"{summary['ROP']:.0f} t"),
        kpi_card("Pour Readiness", f"{summary['pour_readiness_post_warmup']*100:.1f}%"),
    ], style={"display": "flex"})


@app.callback(Output("forecast-chart", "figure"), Input("site-dropdown", "value"))
def update_forecast_chart(site_id):
    d = forecasts_holdout[forecasts_holdout.site_id == site_id].sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["date"], y=d["actual"], name="Actual", line=dict(color="black", width=2)))
    fig.add_trace(go.Scatter(x=d["date"], y=d["rf_pred"], name="Random Forest forecast", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=d["date"], y=d["sarimax_pred"], name="SARIMAX forecast",
                              line=dict(color="orange", dash="dot")))
    fig.update_layout(title=f"Forecast vs. actual (holdout period) — {site_id}",
                       yaxis_title="Tonnes/day", height=380)
    return fig


@app.callback(Output("inventory-chart", "figure"), Input("site-dropdown", "value"))
def update_inventory_chart(site_id):
    d = inv_sim[inv_sim.site_id == site_id].sort_values("date")
    summary = inv_summary[inv_summary.site_id == site_id].iloc[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["date"], y=d["closing_inventory"], name="Projected closing inventory",
                              line=dict(color="steelblue")))
    fig.add_hline(y=summary["ROP"], line_dash="dash", line_color="orange", annotation_text="Reorder point")
    fig.add_hline(y=summary["silo_capacity"], line_dash="dot", line_color="red", annotation_text="Silo capacity")
    stockout_dates = d[d.is_stockout]["date"]
    for sd in stockout_dates:
        fig.add_vline(x=sd, line_color="red", opacity=0.15)
    fig.update_layout(title=f"Projected silo level, next 8 weeks — {site_id}",
                       yaxis_title="Tonnes", height=380)
    return fig


@app.callback(Output("site-alert-table", "children"), Input("site-dropdown", "value"))
def update_site_alert_table(site_id):
    d = reorder_alerts[reorder_alerts.site_id == site_id].sort_values("date")
    if d.empty:
        return html.Div("No reorder alerts projected for this site in the 8-week horizon.")
    return dash_table.DataTable(
        data=d.assign(date=lambda x: x["date"].dt.strftime("%Y-%m-%d"),
                      order_placed=lambda x: x["order_placed"].round(1))
              [["date", "order_placed", "inventory_position", "closing_inventory"]].to_dict("records"),
        columns=[{"name": c, "id": c} for c in ["date", "order_placed", "inventory_position", "closing_inventory"]],
        page_size=10, style_table={"overflowX": "auto"}, style_cell={"padding": "6px", "fontSize": "13px"},
    )


# ---------------------------------------------------------------
app.layout = html.Div([
    html.H2("MIG Cement Demand Forecasting — Operations Dashboard"),
    html.P("Forecasts, inventory projections, and reorder alerts by site.",
           style={"color": "#666", "marginTop": "-8px"}),
    dcc.Tabs([
        dcc.Tab(label="Overview (all sites)", children=[build_overview()]),
        dcc.Tab(label="Site Detail (drill-down)", children=[build_site_detail_layout()]),
    ]),
], style={"maxWidth": "1200px", "margin": "0 auto", "padding": "24px", "fontFamily": "Arial, sans-serif"})


if __name__ == "__main__":
    import os
    debug_mode = os.environ.get("DASH_DEBUG", "true").lower() == "true"
    # host="0.0.0.0" (not 127.0.0.1) so the app is reachable from outside
    # a Docker container, not just from inside it — same fix already
    # applied to the API's uvicorn command in Dockerfile.api.
    app.run(debug=debug_mode, host="0.0.0.0", port=8050)
