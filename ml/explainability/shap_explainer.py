"""
ml/explainability/shap_explainer.py

Wraps SHAP so a single prediction can be explained with:
  - Top-K contributing features (name + SHAP value + direction)
  - A human-readable business explanation string

Supports tree models (RandomForest/XGBoost) via TreeExplainer and the
Logistic Regression / DNN via KernelExplainer (fallback, sampled background).

Usage:
    explainer = FraudExplainer()
    explainer.fit_or_load(model, model_name, X_background)
    result = explainer.explain_instance(x_row, feature_names)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.app.core.config import settings, PROJECT_ROOT  # noqa: E402
from backend.app.core.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)

# Maps raw feature names -> plain-English business phrasing used in explanations
FEATURE_BUSINESS_LABELS: Dict[str, str] = {
    "return_percentage": "High personal return rate",
    "avg_days_to_return": "Speed of return (fast returns are riskier)",
    "seller_defect_rate": "Seller's product defect rate",
    "seller_reliability_score": "Seller reliability score",
    "customer_risk_score": "Overall customer risk score",
    "delivery_risk_score": "Delivery risk score",
    "is_high_value_order": "High value order",
    "is_luxury_product": "Luxury product",
    "return_without_tags": "Return without original tags",
    "packaging_damaged_claim": "Packaging damage claim",
    "used_multiple_addresses": "Multiple shipping addresses used",
    "used_multiple_payment_methods": "Multiple payment methods used",
    "account_age_days": "Customer account age",
    "customer_lifetime_value": "Customer lifetime value",
    "return_reason": "Stated return reason",
    "customer_segment": "Customer loyalty segment",
    "product_price": "Product price",
    "order_value": "Order value",
    "seller_rating": "Seller rating",
}


def humanize_feature(feature_name: str) -> str:
    return FEATURE_BUSINESS_LABELS.get(feature_name, feature_name.replace("_", " ").title())


class FraudExplainer:
    """Thin wrapper around SHAP explainers with graceful model-type dispatch."""

    def __init__(self) -> None:
        self.explainer = None
        self.model_type: str = ""

    def fit_or_load(self, model: Any, model_name: str, X_background: np.ndarray) -> None:
        import shap

        self.model_type = model_name
        bg_size = settings.explainability_config.get("shap_background_samples", 100)
        background = X_background[: min(bg_size, len(X_background))]

        if model_name in ("random_forest", "xgboost"):
            self.explainer = shap.TreeExplainer(model)
        elif model_name == "deep_learning":
            self.explainer = shap.Explainer(model, background)
        else:  # logistic_regression / generic fallback
            self.explainer = shap.Explainer(model.predict_proba, background)

        explainer_path = PROJECT_ROOT / settings.paths["shap_explainer"]
        explainer_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            joblib.dump({"model_type": model_name}, explainer_path)  # explainer itself isn't always picklable
        except Exception:
            pass

    def explain_instance(self, x_row: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """
        Returns:
            {
              "top_features": [{"feature": str, "shap_value": float, "direction": "increases"/"decreases"}, ...],
              "business_explanation": str
            }
        """
        top_k = settings.explainability_config.get("top_k_features", 5)
        x_row = x_row.reshape(1, -1)

        try:
            shap_values = self.explainer(x_row)
            values = np.array(shap_values.values)
            if values.ndim == 3:  # (samples, features, classes) -> take fraud class (index 1)
                values = values[:, :, 1]
            values = values[0]
        except Exception as exc:
            logger.warning(f"SHAP explanation failed, falling back to zero contributions: {exc}")
            values = np.zeros(len(feature_names))

        order = np.argsort(-np.abs(values))[:top_k]
        top_features = []
        for idx in order:
            fname = feature_names[idx]
            val = float(values[idx])
            top_features.append({
                "feature": fname,
                "business_label": humanize_feature(fname),
                "shap_value": round(val, 4),
                "direction": "increases_risk" if val > 0 else "decreases_risk",
            })

        business_explanation = self._build_business_explanation(top_features)
        return {"top_features": top_features, "business_explanation": business_explanation}

    @staticmethod
    def _build_business_explanation(top_features: List[Dict[str, Any]]) -> str:
        increasing = [f["business_label"] for f in top_features if f["direction"] == "increases_risk"]
        decreasing = [f["business_label"] for f in top_features if f["direction"] == "decreases_risk"]

        parts = []
        if increasing:
            parts.append("Risk factors identified: " + ", ".join(increasing) + ".")
        if decreasing:
            parts.append("Mitigating factors: " + ", ".join(decreasing) + ".")
        if not parts:
            parts.append("No strong risk signals were identified for this return.")
        return " ".join(parts)


def load_best_model_and_explainer():
    """
    Convenience loader used by the prediction_service: loads the best model
    (per model_comparison.py output) plus a fitted explainer over training data.
    """
    metadata_path = PROJECT_ROOT / settings.paths["best_model_metadata"]
    with open(metadata_path) as f:
        metadata = json.load(f)

    model_path_key = metadata["model_path_key"]
    model_path = PROJECT_ROOT / settings.paths[model_path_key]
    model_name = metadata["best_model"]

    if model_name == "deep_learning":
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path)
    else:
        model = joblib.load(model_path)

    from ml.preprocessing.preprocessor import run_preprocessing

    splits = run_preprocessing()
    explainer = FraudExplainer()
    explainer.fit_or_load(model, model_name, splits["X_train"])

    return model, model_name, explainer, splits["feature_names"]
