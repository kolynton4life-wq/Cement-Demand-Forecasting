"""
Streamlit dashboard — "Control Tower" redesign, kept in feature and
visual parity with app.py (Dash). Design tokens in theme.py, simulation/
risk logic in scenario_engine.py — shared by both apps.

Uses st.sidebar.radio (not st.navigation) for page switching: proven
testable with Streamlit's AppTest framework, which is how every page and
interaction in this file is verified before being called done.

Run with:  streamlit run src/dashboard/streamlit_app.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from theme import COLORS, RISK_COLORS, FONTS, plotly_layout_defaults
from scenario_engine import add_risk_category, classify_urgency, run_scenario, SCENARIO_PRESETS

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "processed"

REQUIRED_FILES = [
    "operations_clean.parquet", "model_results_by_site.parquet",
    "model_forecasts_holdout.parquet", "inventory_simulation.parquet",
    "inventory_summary_by_site.parquet", "reorder_alerts.parquet",
]

st.set_page_config(page_title="MIG Cement Intelligence", layout="wide", initial_sidebar_state="expanded")

missing = [f for f in REQUIRED_FILES if not (DATA_DIR / f).exists()]
if missing:
    st.error(
        f"Missing required data files in `{DATA_DIR}`:\n\n" + "\n".join(f"- {f}" for f in missing)
        + "\n\nRun notebooks `01` -> `03` -> `04` -> `05` first."
    )
    st.stop()

# ---------------------------------------------------------------
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{ font-family: {FONTS['body']}; }}
h1, h2, h3 {{ font-family: {FONTS['display']} !important; }}

/* Sidebar nav styling — hide radio circles, style like nav list */
[data-testid="stSidebar"] div[role="radiogroup"] label {{
    background: transparent; border-left: 3px solid transparent;
    border-radius: 8px; padding: 8px 10px; margin-bottom: 2px; width: 100%;
}}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{ background: {COLORS['surface_alt']}; }}
[data-testid="stSidebar"] div[role="radiogroup"] input:checked + div {{ color: {COLORS['text']} !important; font-weight: 600; }}

.kpi-card {{
    background: {COLORS['surface']}; border: 1px solid {COLORS['border']};
    border-top: 3px solid {COLORS['accent_blue']}; border-radius: 10px;
    padding: 14px 16px; margin-bottom: 8px;
}}
.kpi-label {{ font-family: {FONTS['mono']}; font-size: 10px; letter-spacing: 0.06em;
    color: {COLORS['text_muted']}; text-transform: uppercase; margin-bottom: 6px; }}
.kpi-value {{ font-family: {FONTS['display']}; font-size: 24px; font-weight: 600; color: {COLORS['text']}; }}
.kpi-sub {{ font-family: {FONTS['body']}; font-size: 11px; color: {COLORS['text_muted']}; margin-top: 3px; }}

.panel-eyebrow {{ font-family: {FONTS['mono']}; font-size: 10.5px; letter-spacing: 0.07em;
    color: {COLORS['accent_blue']}; text-transform: uppercase; margin-bottom: 2px; }}
.panel-title {{ font-family: {FONTS['display']}; font-size: 16px; font-weight: 600; color: {COLORS['text']}; margin-bottom: 10px; }}

.legend-row {{ display: flex; align-items: center; gap: 10px; padding: 9px 0; border-bottom: 1px solid {COLORS['border']}; }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}
.legend-label {{ font-family: {FONTS['body']}; font-size: 13px; color: {COLORS['text']}; font-weight: 500; }}
.legend-sub {{ font-family: {FONTS['body']}; font-size: 11px; color: {COLORS['text_muted']}; }}
.legend-pct {{ font-family: {FONTS['display']}; font-size: 15px; font-weight: 600; margin-left: auto; }}

.live-badge {{ font-family: {FONTS['mono']}; font-size: 11px; color: {COLORS['success']};
    border: 1px solid {COLORS['success']}; border-radius: 20px; padding: 4px 14px; display: inline-block; }}
.topbar-eyebrow {{ font-family: {FONTS['mono']}; font-size: 11px; letter-spacing: 0.08em;
    color: {COLORS['accent_blue']}; text-transform: uppercase; margin-bottom: 2px; }}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
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
    # reorder_alerts.parquet already has ROP/silo_capacity/behavior from
    # notebook 05's own merge — do NOT re-merge inv_summary here (that
    # caused a ROP_x/ROP_y column collision in the Dash app; fixed there,
    # avoided here from the start).
    reorder_alerts["urgency"] = reorder_alerts.apply(classify_urgency, axis=1)
    site_meta = ops.drop_duplicates("site_id").set_index("site_id")[["region", "behavior", "silo_capacity"]]
    network_daily = forecasts_holdout.groupby("date")[["actual", "rf_pred"]].sum().reset_index()
    return ops, model_results, forecasts_holdout, inv_sim, inv_summary, reorder_alerts, site_meta, network_daily


ops, model_results, forecasts_holdout, inv_sim, inv_summary, reorder_alerts, site_meta, network_daily = load_data()
site_ids = sorted(model_results["site_id"].unique())
READINESS_TARGET = 0.98

# ---------------------------------------------------------------
def kpi_card(label, value, sub="", accent=None):
    border = f"border-top-color:{accent};" if accent else ""
    st.markdown(f"""<div class="kpi-card" style="{border}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


def panel_header(eyebrow, title):
    st.markdown(f'<div class="panel-eyebrow">{eyebrow}</div><div class="panel-title">{title}</div>', unsafe_allow_html=True)


def topbar(eyebrow_label, title):
    c1, c2 = st.columns([5, 1])
    with c1:
        st.markdown(f'<div class="topbar-eyebrow">MIG CEMENT OPERATIONS</div>', unsafe_allow_html=True)
        st.markdown(f"## {title}")
    with c2:
        st.markdown('<div style="text-align:right; padding-top:18px;"><span class="live-badge">LIVE FORECAST</span></div>', unsafe_allow_html=True)


def site_picker(key="site_select"):
    return st.selectbox("Select site", site_ids, key=key)


# ---------------------------------------------------------------
def page_overview():
    topbar("EXECUTIVE COMMAND VIEW", "Operational intelligence at a glance")

    total_stockout = int(inv_sim["is_stockout"].sum())
    total_overcap = int((inv_sim["risk_category"] == "Overcapacity").sum())
    mean_mape = model_results["rf_mape"].mean()

    cols = st.columns(5)
    with cols[0]: kpi_card("Sites Monitored", "30", "Active cement operations", COLORS["accent_blue"])
    with cols[1]: kpi_card("Best Model", "Random Forest", f"{mean_mape:.2f}% MAPE", COLORS["accent_amber"])
    with cols[2]: kpi_card("Forecast Horizon", "56 days", "Forward planning window", COLORS["overcapacity"])
    with cols[3]: kpi_card("Stockout Alerts", str(total_stockout), "Critical site-days", COLORS["danger"])
    with cols[4]: kpi_card("Capacity Alerts", str(total_overcap), "Overcapacity site-days", COLORS["warning"])

    panel_header("DEMAND SIGNAL", "Forecast versus actual demand (network-wide)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=network_daily["date"], y=network_daily["actual"], name="Actual", line=dict(color=COLORS["text_muted"], width=1.5)))
    fig.add_trace(go.Scatter(x=network_daily["date"], y=network_daily["rf_pred"], name="Forecast", line=dict(color=COLORS["accent_blue"], width=2.5)))
    fig.update_layout(**plotly_layout_defaults(), height=280)
    st.plotly_chart(fig, width="stretch")

    col_a, col_b = st.columns(2)
    with col_a:
        panel_header("RISK EXPOSURE", "Inventory status distribution")
        risk_counts = inv_sim["risk_category"].value_counts()
        donut = go.Figure(data=[go.Pie(
            labels=risk_counts.index, values=risk_counts.values, hole=0.65,
            marker=dict(colors=[RISK_COLORS[c] for c in risk_counts.index], line=dict(color=COLORS["surface"], width=2)),
            textinfo="percent",
        )])
        donut.update_layout(**plotly_layout_defaults(), showlegend=False, height=300)
        st.plotly_chart(donut, width="stretch")
    with col_b:
        panel_header("STATUS BREAKDOWN", "Exposure by category")
        total_days = len(inv_sim)
        for cat, count in risk_counts.items():
            st.markdown(f"""<div class="legend-row">
                <div class="legend-dot" style="background:{RISK_COLORS[cat]}"></div>
                <div><div class="legend-label">{cat}</div><div class="legend-sub">{count:,} site-days</div></div>
                <div class="legend-pct" style="color:{RISK_COLORS[cat]}">{count/total_days*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)

    panel_header("DAILY ACTION QUEUE", "Nearest reorder recommendations")
    top_alerts = reorder_alerts.sort_values("date").head(8).assign(
        date=lambda d: d["date"].dt.strftime("%Y-%m-%d"), order_placed=lambda d: d["order_placed"].round(1),
    )[["date", "site_id", "urgency", "order_placed", "inventory_position"]]
    st.dataframe(top_alerts, width="stretch", hide_index=True)


def page_forecast():
    topbar("DEMAND SIGNAL", "Predicted vs. actual cement demand by site")
    site_id = site_picker("forecast_site")
    d = forecasts_holdout[forecasts_holdout.site_id == site_id].sort_values("date")
    result = model_results[model_results.site_id == site_id].iloc[0]

    cols = st.columns(4)
    with cols[0]: kpi_card("Site MAPE", f"{result['rf_mape']:.2f}%", "Non-zero demand days", COLORS["accent_blue"])
    with cols[1]: kpi_card("Site RMSE", f"{result['rf_rmse']:.2f} t", "Forecast error", COLORS["overcapacity"])
    with cols[2]: kpi_card("Average Forecast", f"{d['rf_pred'].mean():.2f} t", "Tonnes/day", COLORS["accent_amber"])
    with cols[3]: kpi_card("Peak Forecast", f"{d['rf_pred'].max():.2f} t", "Highest predicted demand", COLORS["warning"])

    panel_header("SITE DEMAND SIGNAL", "Actual versus predicted demand")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["date"], y=d["actual"], name="Actual Demand", line=dict(color=COLORS["text_muted"], width=1.5)))
    fig.add_trace(go.Scatter(x=d["date"], y=d["rf_pred"], name="Forecast Demand", line=dict(color=COLORS["accent_blue"], width=2.5)))
    fig.update_layout(**plotly_layout_defaults(), height=400)
    st.plotly_chart(fig, width="stretch")


def page_inventory():
    topbar("SILO LEVELS", "Projected inventory against reorder points and capacity")
    site_id = site_picker("inventory_site")
    d = inv_sim[inv_sim.site_id == site_id].sort_values("date")
    summary = inv_summary[inv_summary.site_id == site_id].iloc[0]

    cols = st.columns(4)
    with cols[0]: kpi_card("Reorder Point", f"{summary['ROP']:.0f} t", "Trigger threshold", COLORS["warning"])
    with cols[1]: kpi_card("Order-Up-To Level", f"{summary['order_up_to_S']:.0f} t", "Capacity-capped target", COLORS["accent_blue"])
    with cols[2]: kpi_card("Avg Utilization", f"{summary['avg_silo_utilization']*100:.1f}%", "Of silo capacity", COLORS["accent_amber"])
    readiness_color = COLORS["success"] if summary["pour_readiness_post_warmup"] >= READINESS_TARGET else COLORS["danger"]
    with cols[3]: kpi_card("Pour Readiness", f"{summary['pour_readiness_post_warmup']*100:.1f}%", "Post warm-up", readiness_color)

    panel_header("SILO PROJECTION", "Closing inventory vs. reorder point and capacity")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["date"], y=d["closing_inventory"], name="Closing Inventory",
                              line=dict(color=COLORS["accent_blue"], width=2), fill="tozeroy", fillcolor="rgba(79,168,224,0.08)"))
    fig.add_hline(y=summary["ROP"], line_dash="dash", line_color=COLORS["warning"], annotation_text="Reorder point")
    fig.add_hline(y=summary["silo_capacity"], line_dash="dot", line_color=COLORS["danger"], annotation_text="Silo capacity")
    for sd in d[d.is_stockout]["date"]:
        fig.add_vline(x=sd, line_color=COLORS["danger"], opacity=0.15)
    fig.update_layout(**plotly_layout_defaults(), height=400)
    st.plotly_chart(fig, width="stretch")


def page_risk():
    topbar("RISK EXPOSURE", "Stockout and overcapacity exposure across the network")
    site_risk = inv_sim.groupby("site_id").agg(
        stockout_days=("is_stockout", "sum"),
        overcapacity_days=("risk_category", lambda s: (s == "Overcapacity").sum()),
    ).reset_index()
    site_risk = site_risk.merge(inv_summary[["site_id", "pour_readiness_post_warmup", "behavior"]], on="site_id")
    site_risk = site_risk.sort_values("stockout_days", ascending=False)

    panel_header("NETWORK RISK RANKING", "Sites ranked by combined risk days")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=site_risk["site_id"], y=site_risk["stockout_days"], name="Stockout days", marker_color=COLORS["danger"]))
    fig.add_trace(go.Bar(x=site_risk["site_id"], y=site_risk["overcapacity_days"], name="Overcapacity days", marker_color=COLORS["overcapacity"]))
    fig.update_layout(**plotly_layout_defaults(), barmode="stack", height=380)
    st.plotly_chart(fig, width="stretch")

    panel_header("SITE RISK DETAIL", "Full breakdown, sortable")
    display = site_risk.assign(pour_readiness_post_warmup=lambda d: (d["pour_readiness_post_warmup"] * 100).round(1))
    st.dataframe(display, width="stretch", hide_index=True)


def page_reorder():
    topbar("DAILY ACTION QUEUE", "Site-level supply recommendations")
    n_emergency = (reorder_alerts["urgency"] == "Emergency Reorder").sum()

    cols = st.columns(3)
    with cols[0]: kpi_card("Total Alerts", str(len(reorder_alerts)), "Across 8-week horizon", COLORS["accent_blue"])
    with cols[1]: kpi_card("Emergency Reorders", str(n_emergency), "Immediate action needed", COLORS["danger"])
    with cols[2]: kpi_card("Standard Reorders", str(len(reorder_alerts) - n_emergency), "Scheduled ordering", COLORS["warning"])

    panel_header("DAILY ACTION QUEUE", "Site-level supply recommendations")
    display = reorder_alerts.sort_values("date").assign(
        date=lambda d: d["date"].dt.strftime("%Y-%m-%d"), order_placed=lambda d: d["order_placed"].round(1),
        inventory_position=lambda d: d["inventory_position"].round(1),
    )[["date", "site_id", "urgency", "inventory_position", "ROP", "order_placed"]]
    st.dataframe(display, width="stretch", hide_index=True, height=500)


def page_drilldown():
    topbar("SITE DETAIL", "Full operational picture for one site")
    site_id = site_picker("drilldown_site")
    meta = site_meta.loc[site_id]
    result = model_results[model_results.site_id == site_id].iloc[0]
    summary = inv_summary[inv_summary.site_id == site_id].iloc[0]
    d_fc = forecasts_holdout[forecasts_holdout.site_id == site_id].sort_values("date")
    d_inv = inv_sim[inv_sim.site_id == site_id].sort_values("date")
    site_alerts = reorder_alerts[reorder_alerts.site_id == site_id]

    cols = st.columns(4)
    with cols[0]: kpi_card("Region / Behavior", f"{meta['region']} / {meta['behavior']}", "", COLORS["accent_blue"])
    with cols[1]: kpi_card("Silo Capacity", f"{meta['silo_capacity']:.0f} t", "", COLORS["overcapacity"])
    with cols[2]: kpi_card("Forecast MAPE", f"{result['rf_mape']:.2f}%", "", COLORS["accent_amber"])
    with cols[3]: kpi_card("Reorder Alerts", str(len(site_alerts)), "8-week horizon", COLORS["warning"])

    col_a, col_b = st.columns(2)
    with col_a:
        panel_header("DEMAND", "Forecast vs. actual")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d_fc["date"], y=d_fc["actual"], name="Actual", line=dict(color=COLORS["text_muted"], width=1.5)))
        fig.add_trace(go.Scatter(x=d_fc["date"], y=d_fc["rf_pred"], name="Forecast", line=dict(color=COLORS["accent_blue"], width=2.5)))
        fig.update_layout(**plotly_layout_defaults(), height=300)
        st.plotly_chart(fig, width="stretch")
    with col_b:
        panel_header("INVENTORY", "Projected silo level")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=d_inv["date"], y=d_inv["closing_inventory"], name="Inventory", line=dict(color=COLORS["accent_blue"], width=2)))
        fig2.add_hline(y=summary["ROP"], line_dash="dash", line_color=COLORS["warning"])
        fig2.add_hline(y=summary["silo_capacity"], line_dash="dot", line_color=COLORS["danger"])
        fig2.update_layout(**plotly_layout_defaults(), height=300)
        st.plotly_chart(fig2, width="stretch")


def _apply_preset(name):
    p = SCENARIO_PRESETS[name]
    st.session_state["demand_adj"] = p["demand_adj_pct"]
    st.session_state["delivery_adj"] = p["delivery_adj_pct"]
    st.session_state["delay_days"] = p["delivery_delay_days"]


def page_scenario():
    topbar("WHAT-IF INTELLIGENCE", "Stress-test demand, delivery, and timing assumptions")
    site_id = site_picker("scenario_site")

    panel_header("QUICK SCENARIOS", "Apply a predefined operating stress test")
    preset_cols = st.columns(len(SCENARIO_PRESETS))
    for col, name in zip(preset_cols, SCENARIO_PRESETS):
        with col:
            st.button(name, key=f"preset_{name}", on_click=_apply_preset, args=(name,), width="stretch")

    st.session_state.setdefault("demand_adj", 0)
    st.session_state.setdefault("delivery_adj", 0)
    st.session_state.setdefault("delay_days", 0)

    c1, c2, c3 = st.columns(3)
    with c1:
        panel_header("DEMAND ASSUMPTION", "Forecast demand adjustment")
        demand_adj = st.slider("Demand %", -30, 30, step=5, key="demand_adj", label_visibility="collapsed")
    with c2:
        panel_header("SUPPLY ASSUMPTION", "Planned delivery adjustment")
        delivery_adj = st.slider("Delivery %", -50, 50, step=5, key="delivery_adj", label_visibility="collapsed")
    with c3:
        panel_header("TIMING ASSUMPTION", "Delivery delay (days)")
        delay_days = st.slider("Delay", 0, 7, step=1, key="delay_days", label_visibility="collapsed")

    summary = inv_summary[inv_summary.site_id == site_id].iloc[0]
    d_inv = inv_sim[inv_sim.site_id == site_id].sort_values("date")
    forecast_arr = d_inv["forecast_demand"].values
    opening_start = d_inv["opening_inventory"].iloc[0]

    baseline = run_scenario(forecast_arr, summary["silo_capacity"], summary["ROP"], summary["order_up_to_S"], opening_start, 3)
    result = run_scenario(forecast_arr, summary["silo_capacity"], summary["ROP"], summary["order_up_to_S"],
                           opening_start, 3, demand_adj, delivery_adj, delay_days)

    cols = st.columns(4)
    with cols[0]:
        kpi_card("Ending Inventory", f"{result['ending_inventory']:.2f} t",
                 f"{result['ending_inventory']-baseline['ending_inventory']:+.2f} t vs baseline", COLORS["accent_blue"])
    with cols[1]:
        kpi_card("Minimum Inventory", f"{result['minimum_inventory']:.2f} t",
                 f"{result['minimum_inventory']-baseline['minimum_inventory']:+.2f} t vs baseline", COLORS["warning"])
    with cols[2]:
        kpi_card("Stockout Days", str(result["stockout_days"]),
                 f"{result['stockout_days']-baseline['stockout_days']:+d} vs baseline", COLORS["danger"])
    with cols[3]:
        kpi_card("Total Risk Days", str(result["total_risk_days"]),
                 f"{result['total_risk_days']-baseline['total_risk_days']:+d} vs baseline", COLORS["overcapacity"])

    panel_header("PROJECTED INVENTORY", "Scenario vs. baseline silo level")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=baseline["series"]["closing"], name="Baseline", line=dict(color=COLORS["text_muted"], width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(y=result["series"]["closing"], name="Scenario", line=dict(color=COLORS["accent_amber"], width=2.5)))
    fig.add_hline(y=summary["silo_capacity"], line_dash="dot", line_color=COLORS["danger"], annotation_text="Capacity")
    fig.update_layout(**plotly_layout_defaults(), height=360)
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------
PAGES = {
    "Executive Overview": page_overview, "Demand Forecast": page_forecast, "Inventory Control": page_inventory,
    "Risk Monitor": page_risk, "Reorder Recommendations": page_reorder, "Site Drilldown": page_drilldown,
    "Scenario Simulator": page_scenario,
}

with st.sidebar:
    st.markdown('<div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">'
                f'<div style="width:34px;height:34px;border-radius:8px;background:{COLORS["accent_amber"]};'
                'display:flex;align-items:center;justify-content:center;font-family:Space Grotesk;'
                f'font-weight:700;color:{COLORS["bg"]};">M</div>'
                f'<div><div style="font-family:Space Grotesk;font-weight:600;color:{COLORS["text"]};font-size:15px;">MIG</div>'
                f'<div style="font-family:Inter;color:{COLORS["text_muted"]};font-size:11px;">Cement Intelligence</div></div></div>',
                unsafe_allow_html=True)
    st.markdown('<div style="font-family:IBM Plex Mono;font-size:10px;letter-spacing:0.08em;'
                f'color:{COLORS["text_muted"]};text-transform:uppercase;margin-bottom:8px;">Control Tower</div>',
                unsafe_allow_html=True)
    selected = st.radio("Navigation", list(PAGES.keys()), label_visibility="collapsed")
    st.markdown(f'<div style="margin-top:30px;display:flex;align-items:center;gap:8px;'
                f'font-family:Inter;font-size:12px;color:{COLORS["text_muted"]};">'
                f'<div style="width:8px;height:8px;border-radius:50%;background:{COLORS["success"]};'
                f'box-shadow:0 0 6px {COLORS["success"]};"></div>Forecast engine online</div>',
                unsafe_allow_html=True)

PAGES[selected]()
