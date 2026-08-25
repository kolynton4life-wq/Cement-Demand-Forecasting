"""
Step 7 — Inference: load a saved model and produce a forecast without
retraining. This is the actual production prediction interface — used
by api.py and by the dashboard for live requests.

Contrast with notebooks 04/05, which retrain a fresh Random Forest every
run (fine for development/backtesting, not for production latency).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "data"))
from extract import get_full_dataset          # noqa: E402
from preprocessing import run_preprocessing, NON_LAG_FEATURES  # noqa: E402
from model_registry import load_model, list_available_models  # noqa: E402


def forecast_site(site_id: str, horizon: int = 56) -> dict:
    """Recursive multi-step forecast for one site using its saved model.
    Same recursive-forecast logic as notebooks 04/05 (predict a day,
    feed the prediction back in as the next lag), but with NO model
    fitting — the model is loaded, not trained, here."""
    model, metadata = load_model(site_id)
    feature_list = metadata["feature_list"]

    raw = get_full_dataset()
    df = run_preprocessing(raw)
    df["date"] = pd.to_datetime(df["date"])
    site_df = df[df.site_id == site_id].sort_values("date").reset_index(drop=True)

    train = site_df.dropna(subset=feature_list + ["consumed_tonnes"])
    last_date = site_df["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

    # Same-calendar-period-prior-year proxy for future exogenous inputs
    # as notebook 05 — see that notebook's Section 2 for the full caveat
    # about this being a stand-in for MIG's real pour schedule/weather feed.
    proxy_dates = future_dates - pd.DateOffset(years=1)
    proxy = site_df.set_index("date").reindex(proxy_dates)
    proxy[NON_LAG_FEATURES] = proxy[NON_LAG_FEATURES].fillna(site_df[NON_LAG_FEATURES].mean())

    history = train["consumed_tonnes"].tolist()
    preds = []
    for i in range(horizon):
        prow = proxy.iloc[i]
        feat = {c: prow[c] for c in NON_LAG_FEATURES}
        feat["consumed_lag_1d"] = history[-1]
        feat["consumed_lag_7d"] = history[-7]
        feat["consumed_lag_14d"] = history[-14]
        feat["consumed_lag_28d"] = history[-28]
        feat["consumed_roll_mean_7d"] = np.mean(history[-7:])
        feat["consumed_roll_std_7d"] = np.std(history[-7:])
        feat["consumed_roll_mean_14d"] = np.mean(history[-14:])
        feat["consumed_roll_std_14d"] = np.std(history[-14:])
        feat["consumed_roll_mean_28d"] = np.mean(history[-28:])
        feat["consumed_roll_std_28d"] = np.std(history[-28:])
        X = pd.DataFrame([feat])[feature_list]
        p = max(float(model.predict(X)[0]), 0.0)
        preds.append(p)
        history.append(p)

    return {
        "site_id": site_id,
        "dates": [d.strftime("%Y-%m-%d") for d in future_dates],
        "forecast_tonnes": preds,
        "model_trained_at": metadata["trained_at"],
        "model_metrics": {k: v for k, v in metadata["metrics"].items() if k != "per_window"},
        "inventory_params": metadata["inventory_params"],
    }


def forecast_all_sites(horizon: int = 56) -> dict:
    sites = list_available_models()
    return {s: forecast_site(s, horizon) for s in sites}


if __name__ == "__main__":
    import json
    site = sys.argv[1] if len(sys.argv) > 1 else list_available_models()[0]
    result = forecast_site(site, horizon=14)
    print(json.dumps({k: v for k, v in result.items() if k != "dates"}, indent=2, default=str))
    print(f"First 5 forecast days: {list(zip(result['dates'][:5], result['forecast_tonnes'][:5]))}")
