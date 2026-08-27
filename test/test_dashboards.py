import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "dashboard"))

DASHBOARD_DATA_FILES = [
    "operations_clean.parquet", "model_results_by_site.parquet",
    "model_forecasts_holdout.parquet", "inventory_simulation.parquet",
    "inventory_summary_by_site.parquet", "reorder_alerts.parquet",
]
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
data_available = all((DATA_DIR / f).exists() for f in DASHBOARD_DATA_FILES)

pytestmark = pytest.mark.skipif(
    not data_available,
    reason="Dashboard data not found — run notebooks 01->03->04->05 first."
)


# --- Dash: import-time smoke test (callbacks register without error) ---
def test_dash_app_imports_and_layout_builds():
    import importlib
    if "app" in sys.modules:
        del sys.modules["app"]
    app_module = importlib.import_module("app")
    assert app_module.app.layout is not None
    assert len(app_module.site_ids) == 30


def test_dash_reorder_alerts_has_no_column_suffix_collision():
    """Regression test for the ROP_x/ROP_y bug caught while building this
    dashboard — reorder_alerts.parquet already has ROP from notebook 05's
    own merge; re-merging it produces suffixed columns instead."""
    import importlib
    if "app" in sys.modules:
        del sys.modules["app"]
    app_module = importlib.import_module("app")
    assert "ROP" in app_module.reorder_alerts.columns
    assert "ROP_x" not in app_module.reorder_alerts.columns
    assert "ROP_y" not in app_module.reorder_alerts.columns


# --- Streamlit: page load + interaction tests ---
@pytest.fixture
def streamlit_apptest():
    from streamlit.testing.v1 import AppTest
    path = str(Path(__file__).resolve().parent.parent / "src" / "dashboard" / "streamlit_app.py")
    return lambda: AppTest.from_file(path, default_timeout=60)


ALL_PAGES = [
    "Executive Overview", "Demand Forecast", "Inventory Control",
    "Risk Monitor", "Reorder Recommendations", "Site Drilldown", "Scenario Simulator",
]


@pytest.mark.parametrize("page", ALL_PAGES)
def test_streamlit_every_page_loads_without_exception(streamlit_apptest, page):
    at = streamlit_apptest()
    at.run()
    at.radio[0].set_value(page).run()
    assert at.exception == []


def test_streamlit_scenario_preset_sets_correct_session_state(streamlit_apptest):
    at = streamlit_apptest()
    at.run()
    at.radio[0].set_value("Scenario Simulator").run()
    at.button(key="preset_Supply Disruption").click().run()
    assert at.session_state["demand_adj"] == 0
    assert at.session_state["delivery_adj"] == -25
    assert at.session_state["delay_days"] == 3


def test_streamlit_scenario_slider_updates_session_state(streamlit_apptest):
    at = streamlit_apptest()
    at.run()
    at.radio[0].set_value("Scenario Simulator").run()
    at.slider(key="demand_adj").set_value(20).run()
    assert at.session_state["demand_adj"] == 20
    assert at.exception == []


def test_streamlit_site_switch_updates_selection(streamlit_apptest):
    at = streamlit_apptest()
    at.run()
    at.radio[0].set_value("Demand Forecast").run()
    at.selectbox(key="forecast_site").set_value("SITE_020").run()
    assert at.exception == []
    assert at.selectbox(key="forecast_site").value == "SITE_020"
