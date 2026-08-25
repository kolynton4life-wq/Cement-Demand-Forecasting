import pandas as pd
from monitoring import check_site_performance, safe_mape, MAPE_THRESHOLD


def test_safe_mape_basic():
    actual = pd.Series([10, 20, 30])
    pred = pd.Series([10, 20, 30])
    assert safe_mape(actual.values, pred.values) == 0.0


def test_safe_mape_excludes_zero_actuals():
    actual = pd.Series([0, 20, 30])
    pred = pd.Series([5, 20, 30])
    # only the non-zero-actual rows should count
    result = safe_mape(actual.values, pred.values)
    assert result == 0.0  # rows 2,3 are perfect; row 1 (actual=0) excluded, not blown up to inf


def test_degraded_performance_triggers_retrain():
    actual = pd.Series([30, 32, 28, 35, 31])
    bad_pred = pd.Series([5, 60, 10, 55, 8])
    result = check_site_performance("SITE_TEST", actual, bad_pred)
    assert result["degraded"] is True
    assert result["action"] == "TRIGGER_RETRAIN"
    assert result["live_mape"] > MAPE_THRESHOLD


def test_good_performance_does_not_trigger():
    actual = pd.Series([30, 32, 28, 35, 31])
    good_pred = pd.Series([31, 31, 29, 34, 30])
    result = check_site_performance("SITE_TEST", actual, good_pred)
    assert result["degraded"] is False
    assert result["action"] == "OK"
    assert result["live_mape"] <= MAPE_THRESHOLD


def test_boundary_exactly_at_threshold_does_not_trigger():
    """Degradation should require mape > threshold, not >=, so a site
    sitting exactly on target isn't flagged."""
    actual = pd.Series([100.0])
    pred = pd.Series([100.0 * (1 + MAPE_THRESHOLD / 100)])  # exactly at threshold
    result = check_site_performance("SITE_TEST", actual, pred)
    assert result["degraded"] is False
