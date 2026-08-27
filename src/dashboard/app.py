"""
Step 6 — Dashboard Application (Dash)
MIG Cement Demand Forecasting — "Control Tower" redesign

Sidebar navigation across 7 sections reflecting the real operational
decision flow: monitor -> predict -> assess exposure -> act -> simulate.
Design tokens in theme.py; simulation/risk logic in scenario_engine.py —
both shared with streamlit_app.py so neither dashboard drifts out of
visual or numerical sync with the other.

Run with:  python app.py
Then open: http://127.0.0.1:8050
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output, State, ctx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from theme import COLORS, RISK_COLORS, FONTS, GOOGLE_FONTS_URL, plotly_layout_defaults
from scenario_engine import add_risk_category, classify_urgency, run_scenario, SCENARIO_PRESETS

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
    print("\nRun notebooks 01 -> 03 -> 04 -> 05 first (in that order).")
    sys.exit(1)

# ---------------------------------------------------------------
# Load + precompute
ops = pd.read_parquet(DATA_DIR / "operations_clean.parquet")
ops["date"] = pd.to_datetime(ops["date"])

model_results = pd.read_parquet(DATA_DIR / "model_results_by_site.parquet")
forecasts_holdout = pd.read_parquet(DATA_DIR / "model_forecasts_holdout.parquet")
forecasts_holdout["date"] = pd.to_datetime(forecasts_holdout["date"])

inv_sim = pd.read_parquet(DATA_DIR / "inventory_simulation.parquet")
inv_sim["date"] = pd.to_datetime(inv_sim["date"])
inv_sim = add_risk_category(inv_sim)

inv_summary = pd.read_parquet(DATA_DIR / "inventory_summary_by_site.parquet")

reorder_alerts = pd.read_parquet(DATA_DIR / "reorder_alerts.parquet")
reorder_alerts["date"] = pd.to_datetime(reorder_alerts["date"])
# NOTE: reorder_alerts.parquet already carries ROP/silo_capacity/behavior
# from notebook 05's own merge — re-merging inv_summary here would create
# ROP_x/ROP_y suffix collisions instead of a clean ROP column (caught via
# a startup crash when building this dashboard: KeyError: 'ROP').
reorder_alerts["urgency"] = reorder_alerts.apply(classify_urgency, axis=1)

site_meta = ops.drop_duplicates("site_id").set_index("site_id")[["region", "behavior", "silo_capacity"]]
site_ids = sorted(model_results["site_id"].unique())

network_daily = forecasts_holdout.groupby("date")[["actual", "rf_pred"]].sum().reset_index()

MAPE_TARGET = 15.0
READINESS_TARGET = 0.98

# ---------------------------------------------------------------
NAV_ITEMS = [
    ("overview", "Executive Overview"),
    ("forecast", "Demand Forecast"),
    ("inventory", "Inventory Control"),
    ("risk", "Risk Monitor"),
    ("reorder", "Reorder Recommendations"),
    ("drilldown", "Site Drilldown"),
    ("scenario", "Scenario Simulator"),
]
PAGE_TITLES = {
    "overview": ("Executive Command View", "Operational intelligence at a glance"),
    "forecast": ("Demand Signal", "Predicted vs. actual cement demand by site"),
    "inventory": ("Silo Levels", "Projected inventory against reorder points and capacity"),
    "risk": ("Risk Exposure", "Stockout and overcapacity exposure across the network"),
    "reorder": ("Daily Action Queue", "Site-level supply recommendations"),
    "drilldown": ("Site Detail", "Full operational picture for one site"),
    "scenario": ("What-If Intelligence", "Stress-test demand, delivery, and timing assumptions"),
}

app = Dash(__name__, external_stylesheets=[GOOGLE_FONTS_URL])
app.title = "MIG Cement Intelligence"


def kpi_card(label, value, sub="", accent=None):
    style = {"borderTopColor": accent} if accent else {}
    return html.Div([
        html.Div(label, className="kpi-label"),
        html.Div(value, className="kpi-value"),
        html.Div(sub, className="kpi-sub"),
    ], className="kpi-card", style=style)


def panel(eyebrow, title, children):
    return html.Div([
        html.Div(eyebrow, className="panel-eyebrow"),
        html.Div(title, className="panel-title"),
        children,
    ], className="panel")


def site_dropdown(dropdown_id):
    return dcc.Dropdown(
        id=dropdown_id, options=[{"label": s, "value": s} for s in site_ids],
        value=site_ids[0], clearable=False, className="site-selector",
        style={"width": "220px"},
    )


# ---------------------------------------------------------------
def build_sidebar():
    nav_items = [
        html.Div(label, id={"type": "nav-item", "index": key}, className="nav-item" + (" active" if key == "overview" else ""))
        for key, label in NAV_ITEMS
    ]
    return html.Div([
        html.Div([
            html.Div([
                html.Div("M", className="brand-mark"),
                html.Div([
                    html.Div("MIG", className="brand-name"),
                    html.Div("Cement Intelligence", className="brand-sub"),
                ]),
            ], className="brand"),
            html.Div("Control Tower", className="nav-eyebrow"),
            *nav_items,
        ]),
        html.Div([
            html.Div(className="status-dot"),
            html.Span("Forecast engine online"),
        ], className="status-pill"),
    ], className="sidebar")


def topbar(page_key):
    eyebrow, title = PAGE_TITLES[page_key]
    return html.Div([
        html.Div([
            html.Div("MIG CEMENT OPERATIONS", className="topbar-eyebrow"),
            html.Div(title, className="topbar-title"),
        ]),
        html.Div("LIVE FORECAST", className="live-badge"),
    ], className="topbar")


# ---------------------------------------------------------------
def overview_page():
    total_stockout = int(inv_sim["is_stockout"].sum())
    total_overcap = int((inv_sim["risk_category"] == "Overcapacity").sum())
    mean_mape = model_results["rf_mape"].mean()

    risk_counts = inv_sim["risk_category"].value_counts()
    total_days = len(inv_sim)
    donut = go.Figure(data=[go.Pie(
        labels=risk_counts.index, values=risk_counts.values, hole=0.65,
        marker=dict(colors=[RISK_COLORS[c] for c in risk_counts.index], line=dict(color=COLORS["surface"], width=2)),
        textinfo="percent", textfont=dict(color=COLORS["text"], family=FONTS["body"]),
    )])
    donut.update_layout(**plotly_layout_defaults(), showlegend=False, height=320,
                         annotations=[dict(text=f"{total_days:,}<br><span style='font-size:11px'>SITE-DAYS</span>",
                                            x=0.5, y=0.5, font=dict(size=20, color=COLORS["text"], family=FONTS["display"]),
                                            showarrow=False)])

    legend_rows = [
        html.Div([
            html.Div(className="legend-dot", style={"background": RISK_COLORS[cat]}),
            html.Div([
                html.Div(cat, className="legend-label"),
                html.Div(f"{count:,} site-days", className="legend-sub"),
            ]),
            html.Div(f"{count/total_days*100:.1f}%", className="legend-pct", style={"color": RISK_COLORS[cat]}),
        ], className="legend-row")
        for cat, count in risk_counts.items()
    ]

    fig_network = go.Figure()
    fig_network.add_trace(go.Scatter(x=network_daily["date"], y=network_daily["actual"], name="Actual",
                                      line=dict(color=COLORS["text_muted"], width=1.5)))
    fig_network.add_trace(go.Scatter(x=network_daily["date"], y=network_daily["rf_pred"], name="Forecast",
                                      line=dict(color=COLORS["accent_blue"], width=2.5)))
    fig_network.update_layout(**plotly_layout_defaults(), height=280, yaxis_title="Tonnes/day")

    top_alerts = reorder_alerts.sort_values("date").head(8)
    alert_table = dash_table.DataTable(
        data=top_alerts.assign(
            date=lambda d: d["date"].dt.strftime("%Y-%m-%d"), order_placed=lambda d: d["order_placed"].round(1),
        )[["date", "site_id", "urgency", "order_placed", "inventory_position"]].to_dict("records"),
        columns=[{"name": c, "id": c} for c in ["date", "site_id", "urgency", "order_placed", "inventory_position"]],
        style_header={"backgroundColor": COLORS["surface_alt"], "color": COLORS["text_muted"], "fontFamily": FONTS["mono"], "fontSize": "11px", "textTransform": "uppercase", "border": "none"},
        style_cell={"backgroundColor": COLORS["surface"], "color": COLORS["text"], "fontFamily": FONTS["mono"], "fontSize": "12.5px", "border": "none", "borderBottom": f"1px solid {COLORS['border']}", "padding": "8px 12px"},
        style_data_conditional=[
            {"if": {"filter_query": '{urgency} = "Emergency Reorder"'}, "borderLeft": f"3px solid {COLORS['danger']}"},
            {"if": {"filter_query": '{urgency} = "Place Reorder"'}, "borderLeft": f"3px solid {COLORS['warning']}"},
        ],
        style_as_list_view=True,
    )

    return html.Div([
        html.Div([
            kpi_card("Sites Monitored", "30", "Active cement operations", COLORS["accent_blue"]),
            kpi_card("Best Model", "Random Forest", f"{mean_mape:.2f}% MAPE", COLORS["accent_amber"]),
            kpi_card("Forecast Horizon", "56 days", "Forward planning window", COLORS["overcapacity"]),
            kpi_card("Stockout Alerts", str(total_stockout), "Critical site-days", COLORS["danger"]),
            kpi_card("Capacity Alerts", str(total_overcap), "Overcapacity site-days", COLORS["warning"]),
        ], className="kpi-row"),
        panel("DEMAND SIGNAL", "Forecast versus actual demand (network-wide)", dcc.Graph(figure=fig_network, config={"displayModeBar": False})),
        html.Div([
            html.Div(panel("RISK EXPOSURE", "Inventory status distribution", dcc.Graph(figure=donut, config={"displayModeBar": False})), style={"flex": "1"}),
            html.Div(panel("STATUS BREAKDOWN", "Exposure by category", html.Div(legend_rows)), style={"flex": "1"}),
        ], style={"display": "flex", "gap": "20px"}),
        panel("DAILY ACTION QUEUE", "Nearest reorder recommendations", alert_table),
    ])


def demand_forecast_page():
    return html.Div([
        html.Div(id="forecast-kpi-row", className="kpi-row"),
        panel("SITE DEMAND SIGNAL", "Actual versus predicted demand", dcc.Graph(id="forecast-chart", config={"displayModeBar": False})),
    ])


def inventory_control_page():
    return html.Div([
        html.Div(id="inventory-kpi-row", className="kpi-row"),
        panel("SILO PROJECTION", "Closing inventory vs. reorder point and capacity", dcc.Graph(id="inventory-chart", config={"displayModeBar": False})),
    ])


def risk_monitor_page():
    site_risk = inv_sim.groupby("site_id").agg(
        stockout_days=("is_stockout", "sum"),
        overcapacity_days=("risk_category", lambda s: (s == "Overcapacity").sum()),
    ).reset_index()
    site_risk = site_risk.merge(inv_summary[["site_id", "pour_readiness_post_warmup", "behavior"]], on="site_id")
    site_risk = site_risk.sort_values("stockout_days", ascending=False)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=site_risk["site_id"], y=site_risk["stockout_days"], name="Stockout days", marker_color=COLORS["danger"]))
    fig.add_trace(go.Bar(x=site_risk["site_id"], y=site_risk["overcapacity_days"], name="Overcapacity days", marker_color=COLORS["overcapacity"]))
    fig.update_layout(**plotly_layout_defaults(), barmode="stack", height=380, yaxis_title="Days (8-week horizon)")

    table = dash_table.DataTable(
        data=site_risk.assign(pour_readiness_post_warmup=lambda d: (d["pour_readiness_post_warmup"] * 100).round(1)).to_dict("records"),
        columns=[
            {"name": "Site", "id": "site_id"}, {"name": "Behavior", "id": "behavior"},
            {"name": "Stockout Days", "id": "stockout_days"}, {"name": "Overcapacity Days", "id": "overcapacity_days"},
            {"name": "Pour Readiness %", "id": "pour_readiness_post_warmup"},
        ],
        style_header={"backgroundColor": COLORS["surface_alt"], "color": COLORS["text_muted"], "fontFamily": FONTS["mono"], "fontSize": "11px", "textTransform": "uppercase", "border": "none"},
        style_cell={"backgroundColor": COLORS["surface"], "color": COLORS["text"], "fontFamily": FONTS["mono"], "fontSize": "12.5px", "border": "none", "borderBottom": f"1px solid {COLORS['border']}", "padding": "8px 12px"},
        style_as_list_view=True, page_size=10, sort_action="native",
    )

    return html.Div([
        panel("NETWORK RISK RANKING", "Sites ranked by combined risk days", dcc.Graph(figure=fig, config={"displayModeBar": False})),
        panel("SITE RISK DETAIL", "Full breakdown, sortable", table),
    ])


def reorder_recommendations_page():
    display = reorder_alerts.sort_values("date").assign(
        date=lambda d: d["date"].dt.strftime("%Y-%m-%d"), order_placed=lambda d: d["order_placed"].round(1),
        inventory_position=lambda d: d["inventory_position"].round(1),
    )[["date", "site_id", "urgency", "inventory_position", "ROP", "order_placed"]]

    table = dash_table.DataTable(
        data=display.to_dict("records"),
        columns=[
            {"name": "Date", "id": "date"}, {"name": "Site", "id": "site_id"}, {"name": "Action", "id": "urgency"},
            {"name": "Inventory", "id": "inventory_position"}, {"name": "Reorder Point", "id": "ROP"},
            {"name": "Order Quantity", "id": "order_placed"},
        ],
        style_header={"backgroundColor": COLORS["surface_alt"], "color": COLORS["text_muted"], "fontFamily": FONTS["mono"], "fontSize": "11px", "textTransform": "uppercase", "border": "none"},
        style_cell={"backgroundColor": COLORS["surface"], "color": COLORS["text"], "fontFamily": FONTS["mono"], "fontSize": "12.5px", "border": "none", "borderBottom": f"1px solid {COLORS['border']}", "padding": "10px 12px"},
        style_data_conditional=[
            {"if": {"filter_query": '{urgency} = "Emergency Reorder"'}, "borderLeft": f"3px solid {COLORS['danger']}", "color": COLORS["danger"]},
            {"if": {"filter_query": '{urgency} = "Place Reorder"'}, "borderLeft": f"3px solid {COLORS['warning']}"},
        ],
        style_cell_conditional=[{"if": {"column_id": "urgency"}, "fontWeight": "600"}],
        style_as_list_view=True, page_size=15, sort_action="native", filter_action="native",
    )
    n_emergency = (reorder_alerts["urgency"] == "Emergency Reorder").sum()
    return html.Div([
        html.Div([
            kpi_card("Total Alerts", str(len(reorder_alerts)), "Across 8-week horizon", COLORS["accent_blue"]),
            kpi_card("Emergency Reorders", str(n_emergency), "Immediate action needed", COLORS["danger"]),
            kpi_card("Standard Reorders", str(len(reorder_alerts) - n_emergency), "Scheduled ordering", COLORS["warning"]),
        ], className="kpi-row"),
        panel("DAILY ACTION QUEUE", "Site-level supply recommendations", table),
    ])


def site_drilldown_page():
    return html.Div(id="drilldown-content")


def scenario_simulator_page():
    slider_marks_style = {"color": COLORS["text_muted"]}
    return html.Div([
        html.Div([
            panel("QUICK SCENARIOS", "Apply a predefined operating stress test", html.Div([
                html.Button([html.Div(name, className="preset-btn-title"), html.Div(_preset_sub(name), className="preset-btn-sub")],
                            id={"type": "preset-btn", "index": name}, className="preset-btn", n_clicks=0)
                for name in SCENARIO_PRESETS
            ], style={"display": "flex", "gap": "12px"})),
        ]),
        html.Div([
            html.Div(panel("DEMAND ASSUMPTION", "Forecast demand adjustment", html.Div([
                html.Div(id="demand-slider-label", className="slider-value"),
                dcc.Slider(id="demand-slider", min=-30, max=30, step=5, value=0,
                           marks={i: {"label": f"{i:+d}%", "style": slider_marks_style} for i in range(-30, 31, 15)}),
            ])), style={"flex": "1"}),
            html.Div(panel("SUPPLY ASSUMPTION", "Planned delivery adjustment", html.Div([
                html.Div(id="delivery-slider-label", className="slider-value"),
                dcc.Slider(id="delivery-slider", min=-50, max=50, step=5, value=0,
                           marks={i: {"label": f"{i:+d}%", "style": slider_marks_style} for i in range(-50, 51, 25)}),
            ])), style={"flex": "1"}),
            html.Div(panel("TIMING ASSUMPTION", "Delivery delay (days)", html.Div([
                html.Div(id="delay-slider-label", className="slider-value"),
                dcc.Slider(id="delay-slider", min=0, max=7, step=1, value=0,
                           marks={i: {"label": str(i), "style": slider_marks_style} for i in [0, 1, 3, 5, 7]}),
            ])), style={"flex": "1"}),
        ], style={"display": "flex", "gap": "16px", "margin": "18px 0"}),
        html.Div(id="scenario-kpi-row", className="kpi-row"),
        panel("PROJECTED INVENTORY", "Scenario vs. baseline silo level", dcc.Graph(id="scenario-chart", config={"displayModeBar": False})),
    ])


def _preset_sub(name):
    p = SCENARIO_PRESETS[name]
    parts = []
    if p["demand_adj_pct"]:
        parts.append(f"{p['demand_adj_pct']:+d}% demand")
    if p["delivery_adj_pct"]:
        parts.append(f"{p['delivery_adj_pct']:+d}% deliveries")
    if p["delivery_delay_days"]:
        parts.append(f"{p['delivery_delay_days']}-day delay")
    return ", ".join(parts) if parts else "Return every control to zero"


# ---------------------------------------------------------------
PAGE_BUILDERS = {
    "overview": overview_page, "forecast": demand_forecast_page, "inventory": inventory_control_page,
    "risk": risk_monitor_page, "reorder": reorder_recommendations_page, "drilldown": site_drilldown_page,
    "scenario": scenario_simulator_page,
}

app.layout = html.Div([
    build_sidebar(),
    html.Div([
        html.Div(id="topbar-container"),
        html.Div([
            html.Label("SELECT SITE", style={"fontFamily": FONTS["mono"], "fontSize": "10px", "color": COLORS["text_muted"], "display": "block", "marginBottom": "4px"}),
            site_dropdown("global-site-dropdown"),
        ], id="site-selector-row", style={"marginBottom": "18px"}),
        html.Div([
            html.Div(PAGE_BUILDERS[key](), id=f"page-{key}", style={"display": "block" if key == "overview" else "none"})
            for key, _ in NAV_ITEMS
        ]),
    ], className="main-content"),
    dcc.Store(id="active-page", data="overview"),
], style={"backgroundColor": COLORS["bg"], "minHeight": "100vh"})


# ---------------------------------------------------------------
# Navigation callback
@app.callback(
    Output("active-page", "data"),
    [Input({"type": "nav-item", "index": key}, "n_clicks") for key, _ in NAV_ITEMS],
    prevent_initial_call=True,
)
def handle_nav_click(*_):
    triggered = ctx.triggered_id
    if triggered and isinstance(triggered, dict):
        return triggered["index"]
    return "overview"


@app.callback(
    [Output(f"page-{key}", "style") for key, _ in NAV_ITEMS]
    + [Output({"type": "nav-item", "index": key}, "className") for key, _ in NAV_ITEMS]
    + [Output("topbar-container", "children"), Output("site-selector-row", "style")],
    Input("active-page", "data"),
)
def render_active_page(active):
    styles = [{"display": "block" if key == active else "none"} for key, _ in NAV_ITEMS]
    classnames = ["nav-item active" if key == active else "nav-item" for key, _ in NAV_ITEMS]
    selector_style = {"marginBottom": "18px"} if active in ("forecast", "inventory", "drilldown", "scenario") else {"display": "none"}
    return styles + classnames + [topbar(active), selector_style]


# Demand Forecast page
@app.callback(
    [Output("forecast-chart", "figure"), Output("forecast-kpi-row", "children")],
    Input("global-site-dropdown", "value"),
)
def update_forecast_page(site_id):
    d = forecasts_holdout[forecasts_holdout.site_id == site_id].sort_values("date")
    result = model_results[model_results.site_id == site_id].iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["date"], y=d["actual"], name="Actual Demand", line=dict(color=COLORS["text_muted"], width=1.5)))
    fig.add_trace(go.Scatter(x=d["date"], y=d["rf_pred"], name="Forecast Demand", line=dict(color=COLORS["accent_blue"], width=2.5)))
    fig.update_layout(**plotly_layout_defaults(), height=380, yaxis_title="Tonnes")

    kpis = [
        kpi_card("Site MAPE", f"{result['rf_mape']:.2f}%", "Non-zero demand days", COLORS["accent_blue"]),
        kpi_card("Site RMSE", f"{result['rf_rmse']:.2f} t", "Forecast error in tonnes", COLORS["overcapacity"]),
        kpi_card("Average Forecast", f"{d['rf_pred'].mean():.2f} t", "Average tonnes per day", COLORS["accent_amber"]),
        kpi_card("Peak Forecast", f"{d['rf_pred'].max():.2f} t", "Highest predicted demand", COLORS["warning"]),
    ]
    return fig, kpis


# Inventory Control page
@app.callback(
    [Output("inventory-chart", "figure"), Output("inventory-kpi-row", "children")],
    Input("global-site-dropdown", "value"),
)
def update_inventory_page(site_id):
    d = inv_sim[inv_sim.site_id == site_id].sort_values("date")
    summary = inv_summary[inv_summary.site_id == site_id].iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["date"], y=d["closing_inventory"], name="Closing Inventory",
                              line=dict(color=COLORS["accent_blue"], width=2), fill="tozeroy",
                              fillcolor="rgba(79,168,224,0.08)"))
    fig.add_hline(y=summary["ROP"], line_dash="dash", line_color=COLORS["warning"], annotation_text="Reorder point")
    fig.add_hline(y=summary["silo_capacity"], line_dash="dot", line_color=COLORS["danger"], annotation_text="Silo capacity")
    for sd in d[d.is_stockout]["date"]:
        fig.add_vline(x=sd, line_color=COLORS["danger"], opacity=0.15)
    fig.update_layout(**plotly_layout_defaults(), height=380, yaxis_title="Tonnes")

    kpis = [
        kpi_card("Reorder Point", f"{summary['ROP']:.0f} t", "Trigger threshold", COLORS["warning"]),
        kpi_card("Order-Up-To Level", f"{summary['order_up_to_S']:.0f} t", "Capacity-capped target", COLORS["accent_blue"]),
        kpi_card("Avg Utilization", f"{summary['avg_silo_utilization']*100:.1f}%", "Of silo capacity", COLORS["accent_amber"]),
        kpi_card("Pour Readiness", f"{summary['pour_readiness_post_warmup']*100:.1f}%", "Post warm-up", COLORS["success"] if summary["pour_readiness_post_warmup"] >= READINESS_TARGET else COLORS["danger"]),
    ]
    return fig, kpis


# Site Drilldown page — consolidated view
@app.callback(Output("drilldown-content", "children"), Input("global-site-dropdown", "value"))
def update_drilldown_page(site_id):
    meta = site_meta.loc[site_id]
    result = model_results[model_results.site_id == site_id].iloc[0]
    summary = inv_summary[inv_summary.site_id == site_id].iloc[0]
    d_fc = forecasts_holdout[forecasts_holdout.site_id == site_id].sort_values("date")
    d_inv = inv_sim[inv_sim.site_id == site_id].sort_values("date")
    site_alerts = reorder_alerts[reorder_alerts.site_id == site_id].sort_values("date")

    kpis = [
        kpi_card("Region / Behavior", f"{meta['region']} / {meta['behavior']}", "", COLORS["accent_blue"]),
        kpi_card("Silo Capacity", f"{meta['silo_capacity']:.0f} t", "", COLORS["overcapacity"]),
        kpi_card("Forecast MAPE", f"{result['rf_mape']:.2f}%", "", COLORS["accent_amber"]),
        kpi_card("Reorder Alerts", str(len(site_alerts)), "8-week horizon", COLORS["warning"]),
    ]

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(x=d_fc["date"], y=d_fc["actual"], name="Actual", line=dict(color=COLORS["text_muted"], width=1.5)))
    fig_fc.add_trace(go.Scatter(x=d_fc["date"], y=d_fc["rf_pred"], name="Forecast", line=dict(color=COLORS["accent_blue"], width=2.5)))
    fig_fc.update_layout(**plotly_layout_defaults(), height=300, yaxis_title="Tonnes")

    fig_inv = go.Figure()
    fig_inv.add_trace(go.Scatter(x=d_inv["date"], y=d_inv["closing_inventory"], name="Inventory", line=dict(color=COLORS["accent_blue"], width=2)))
    fig_inv.add_hline(y=summary["ROP"], line_dash="dash", line_color=COLORS["warning"])
    fig_inv.add_hline(y=summary["silo_capacity"], line_dash="dot", line_color=COLORS["danger"])
    fig_inv.update_layout(**plotly_layout_defaults(), height=300, yaxis_title="Tonnes")

    return html.Div([
        html.Div(kpis, className="kpi-row"),
        html.Div([
            html.Div(panel("DEMAND", "Forecast vs. actual", dcc.Graph(figure=fig_fc, config={"displayModeBar": False})), style={"flex": "1"}),
            html.Div(panel("INVENTORY", "Projected silo level", dcc.Graph(figure=fig_inv, config={"displayModeBar": False})), style={"flex": "1"}),
        ], style={"display": "flex", "gap": "16px"}),
    ])


# Scenario Simulator
@app.callback(
    [Output("demand-slider", "value"), Output("delivery-slider", "value"), Output("delay-slider", "value")],
    [Input({"type": "preset-btn", "index": name}, "n_clicks") for name in SCENARIO_PRESETS],
    prevent_initial_call=True,
)
def apply_preset(*_):
    triggered = ctx.triggered_id
    if triggered and isinstance(triggered, dict):
        p = SCENARIO_PRESETS[triggered["index"]]
        return p["demand_adj_pct"], p["delivery_adj_pct"], p["delivery_delay_days"]
    return 0, 0, 0


@app.callback(
    [Output("demand-slider-label", "children"), Output("delivery-slider-label", "children"), Output("delay-slider-label", "children"),
     Output("scenario-kpi-row", "children"), Output("scenario-chart", "figure")],
    [Input("demand-slider", "value"), Input("delivery-slider", "value"), Input("delay-slider", "value"),
     Input("global-site-dropdown", "value")],
)
def update_scenario(demand_adj, delivery_adj, delay, site_id):
    summary = inv_summary[inv_summary.site_id == site_id].iloc[0]
    d_inv = inv_sim[inv_sim.site_id == site_id].sort_values("date")
    forecast_arr = d_inv["forecast_demand"].values
    opening_start = d_inv["opening_inventory"].iloc[0]

    baseline = run_scenario(forecast_arr, summary["silo_capacity"], summary["ROP"], summary["order_up_to_S"],
                             opening_start, 3)  # lead_time_days=3, matches project default
    result = run_scenario(forecast_arr, summary["silo_capacity"], summary["ROP"], summary["order_up_to_S"],
                           opening_start, 3, demand_adj, delivery_adj, delay)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=baseline["series"]["closing"], name="Baseline", line=dict(color=COLORS["text_muted"], width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(y=result["series"]["closing"], name="Scenario", line=dict(color=COLORS["accent_amber"], width=2.5)))
    fig.add_hline(y=summary["silo_capacity"], line_dash="dot", line_color=COLORS["danger"], annotation_text="Capacity")
    fig.update_layout(**plotly_layout_defaults(), height=340, yaxis_title="Tonnes", xaxis_title="Day")

    kpis = [
        kpi_card("Ending Inventory", f"{result['ending_inventory']:.2f} t",
                 f"{result['ending_inventory']-baseline['ending_inventory']:+.2f} t vs baseline", COLORS["accent_blue"]),
        kpi_card("Minimum Inventory", f"{result['minimum_inventory']:.2f} t",
                 f"{result['minimum_inventory']-baseline['minimum_inventory']:+.2f} t vs baseline", COLORS["warning"]),
        kpi_card("Stockout Days", str(result["stockout_days"]),
                 f"{result['stockout_days']-baseline['stockout_days']:+d} vs baseline", COLORS["danger"]),
        kpi_card("Total Risk Days", str(result["total_risk_days"]),
                 f"{result['total_risk_days']-baseline['total_risk_days']:+d} vs baseline", COLORS["overcapacity"]),
    ]
    return f"Current demand adjustment: {demand_adj:+d}%", f"Current delivery adjustment: {delivery_adj:+d}%", \
        f"Current delivery delay: {delay} days", kpis, fig


if __name__ == "__main__":
    import os
    debug_mode = os.environ.get("DASH_DEBUG", "true").lower() == "true"
    app.run(debug=debug_mode, host="0.0.0.0", port=8050)
