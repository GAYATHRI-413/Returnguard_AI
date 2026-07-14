"""
backend/app/services/explainability_service.py

Provides GLOBAL feature importance (as opposed to prediction_service's
per-prediction SHAP explanation), used by GET /feature_importance to power
the dashboard's importance bar chart.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from backend.app.core.logging_config import get_logger
from backend.app.services.prediction_service import get_engine
from ml.explainability.shap_explainer import humanize_feature

logger = get_logger(__name__)


def get_global_feature_importance() -> Dict[str, Any]:
    engine = get_engine()
    if not engine.loaded:
        raise RuntimeError("Model not loaded. Train models first.")

    model = engine.model
    feature_names = engine.feature_names

    importances: np.ndarray
    if hasattr(model, "feature_importances_"):
        # RandomForest / XGBoost expose this directly (fast path, no SHAP needed)
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        # Logistic Regression: absolute coefficient magnitude as importance proxy
        importances = np.abs(np.asarray(model.coef_).ravel())
    else:
        # Deep learning / fallback: use mean absolute SHAP value over a small sample
        from ml.preprocessing.preprocessor import run_preprocessing

        splits = run_preprocessing()
        sample = splits["X_test"][:50]
        shap_vals = []
        for row in sample:
            explanation = engine.explainer.explain_instance(row, feature_names)
            per_row = {f["feature"]: abs(f["shap_value"]) for f in explanation["top_features"]}
            shap_vals.append(per_row)
        agg: Dict[str, float] = {}
        for row_dict in shap_vals:
            for k, v in row_dict.items():
                agg[k] = agg.get(k, 0.0) + v
        importances = np.array([agg.get(f, 0.0) for f in feature_names])

    # Normalize to sum to 1 for interpretability
    total = importances.sum() or 1.0
    normalized = importances / total

    ranked_idx = np.argsort(-normalized)
    features: List[Dict[str, Any]] = [
        {
            "feature": feature_names[i],
            "business_label": humanize_feature(feature_names[i]),
            "importance": round(float(normalized[i]), 4),
        }
        for i in ranked_idx
    ]

    return {"model_used": engine.model_name, "features": features}
