"""
Model registry — save/load trained per-site Random Forest models plus the
metadata needed to reproduce feature engineering at inference time
(without retraining), and the Step 5 reorder-point parameters.

Directory layout under models/:
    models/
      <site_id>/
        rf_model.joblib       - trained RandomForestRegressor
        metadata.json         - training date, feature list, metrics, ROP/S
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def save_model(site_id: str, model, feature_list: list, metrics: dict, inventory_params: dict = None):
    site_dir = MODELS_DIR / site_id
    site_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, site_dir / "rf_model.joblib")

    metadata = {
        "site_id": site_id,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_list": feature_list,
        "metrics": metrics,
        "inventory_params": inventory_params or {},
    }
    with open(site_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return site_dir


def load_model(site_id: str):
    site_dir = MODELS_DIR / site_id
    model_path = site_dir / "rf_model.joblib"
    meta_path = site_dir / "metadata.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No saved model for site '{site_id}' at {model_path}. "
            f"Run `python src/pipeline.py` first to train and save models."
        )
    model = joblib.load(model_path)
    with open(meta_path) as f:
        metadata = json.load(f)
    return model, metadata


def list_available_models() -> list:
    if not MODELS_DIR.exists():
        return []
    return sorted([d.name for d in MODELS_DIR.iterdir() if d.is_dir() and (d / "rf_model.joblib").exists()])
