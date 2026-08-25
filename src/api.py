"""
Step 7 — Prediction API (FastAPI).

Thin HTTP layer over inference.py, for the dashboard or other services
to request forecasts without needing Python/model access directly.
Matches the Dockerfile.api container in the deployment layout.

Run with:  uvicorn src.api:app --reload --port 8000
Docs at:   http://127.0.0.1:8000/docs
"""
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference import forecast_site                 # noqa: E402
from model_registry import list_available_models    # noqa: E402
from monitoring import check_site_performance, MAPE_THRESHOLD  # noqa: E402

app = FastAPI(
    title="MIG Cement Demand Forecasting API",
    description="Serves per-site cement demand forecasts and inventory reorder parameters.",
    version="1.0.0",
)


class ForecastResponse(BaseModel):
    site_id: str
    dates: list
    forecast_tonnes: list
    model_trained_at: str
    model_metrics: dict
    inventory_params: dict


class HealthResponse(BaseModel):
    status: str
    available_sites: int


class MonitoringCheckRequest(BaseModel):
    site_id: str
    actual: list
    predicted: list


@app.get("/health", response_model=HealthResponse)
def health():
    sites = list_available_models()
    return HealthResponse(status="ok", available_sites=len(sites))


@app.get("/sites")
def sites():
    return {"sites": list_available_models()}


@app.get("/forecast/{site_id}", response_model=ForecastResponse)
def forecast(site_id: str, horizon: int = 56):
    if site_id not in list_available_models():
        raise HTTPException(status_code=404, detail=f"No trained model for site '{site_id}'. "
                                                      f"Available: {list_available_models()}")
    if horizon < 1 or horizon > 90:
        raise HTTPException(status_code=400, detail="horizon must be between 1 and 90 days")
    result = forecast_site(site_id, horizon=horizon)
    return result


@app.post("/monitoring/check")
def monitoring_check(req: MonitoringCheckRequest):
    if len(req.actual) != len(req.predicted):
        raise HTTPException(status_code=400, detail="actual and predicted must be the same length")
    import pandas as pd
    result = check_site_performance(req.site_id, pd.Series(req.actual), pd.Series(req.predicted))
    return result
