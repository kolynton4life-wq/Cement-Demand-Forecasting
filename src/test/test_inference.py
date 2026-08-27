import time

import pytest
from model_registry import list_available_models
from inference import forecast_site

pytestmark = pytest.mark.skipif(
    len(list_available_models()) == 0,
    reason="No trained models found — run `python src/pipeline.py` first."
)


def test_forecast_site_returns_expected_shape():
    site_id = list_available_models()[0]
    result = forecast_site(site_id, horizon=14)
    assert result["site_id"] == site_id
    assert len(result["dates"]) == 14
    assert len(result["forecast_tonnes"]) == 14


def test_forecast_values_are_non_negative():
    site_id = list_available_models()[0]
    result = forecast_site(site_id, horizon=14)
    assert all(v >= 0 for v in result["forecast_tonnes"])


def test_forecast_does_not_retrain_model():
    """Inference must be fast — if this were silently retraining a fresh
    model instead of loading the saved one, it would take several
    seconds per site (see pipeline.py timing). This is the core
    production-readiness property of Step 7's inference layer."""
    site_id = list_available_models()[0]
    t0 = time.time()
    forecast_site(site_id, horizon=14)
    elapsed = time.time() - t0
    assert elapsed < 5.0, f"Inference took {elapsed:.1f}s — too slow, may be retraining instead of loading"


def test_inventory_params_present_and_sane():
    site_id = list_available_models()[0]
    result = forecast_site(site_id, horizon=14)
    params = result["inventory_params"]
    assert params["reorder_point"] > 0
    assert params["order_up_to_level"] <= params["silo_capacity"] + 1e-6, \
        "order_up_to_level must never exceed physical silo capacity"


def test_unknown_site_raises_clear_error():
    with pytest.raises(FileNotFoundError):
        forecast_site("SITE_DOES_NOT_EXIST", horizon=7)
