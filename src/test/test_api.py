import pytest
from fastapi.testclient import TestClient
from api import app
from model_registry import list_available_models

client = TestClient(app)

pytestmark = pytest.mark.skipif(
    len(list_available_models()) == 0,
    reason="No trained models found — run `python src/pipeline.py` first."
)


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["available_sites"] > 0


def test_sites_endpoint():
    r = client.get("/sites")
    assert r.status_code == 200
    assert isinstance(r.json()["sites"], list)


def test_forecast_endpoint_valid_site():
    site_id = list_available_models()[0]
    r = client.get(f"/forecast/{site_id}?horizon=7")
    assert r.status_code == 200
    body = r.json()
    assert len(body["dates"]) == 7
    assert len(body["forecast_tonnes"]) == 7


def test_forecast_endpoint_unknown_site_returns_404():
    r = client.get("/forecast/SITE_DOES_NOT_EXIST")
    assert r.status_code == 404


def test_forecast_endpoint_invalid_horizon_returns_400():
    site_id = list_available_models()[0]
    r = client.get(f"/forecast/{site_id}?horizon=9999")
    assert r.status_code == 400


def test_monitoring_check_endpoint():
    r = client.post("/monitoring/check", json={
        "site_id": "SITE_TEST", "actual": [30, 32, 28], "predicted": [5, 60, 10],
    })
    assert r.status_code == 200
    assert r.json()["action"] == "TRIGGER_RETRAIN"


def test_monitoring_check_endpoint_mismatched_lengths_returns_400():
    r = client.post("/monitoring/check", json={
        "site_id": "SITE_TEST", "actual": [30, 32, 28], "predicted": [5, 60],
    })
    assert r.status_code == 400
