"""
backend/app/services/prediction_service.py

The core orchestrator:
    raw request -> feature engineering -> preprocessing (encode+scale)
    -> model.predict_proba -> SHAP explanation -> business rules
    -> structured result (+ optional DB persistence)

Loads the best model ONCE at import time (singleton pattern) instead of
per-request, which is what makes this production-viable instead of a
notebook-style script.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from backend.app.core.config import settings, PROJECT_ROOT
from backend.app.core.logging_config import get_logger
from backend.app.services.business_rules import evaluate_business_rules
from backend.app.services.feature_engineering import engineer_features, features_to_dataframe

logger = get_logger(__name__)


class PredictionEngine:
    """Singleton-style holder for the loaded model + explainer + preprocessing artifacts."""

    _instance = None

    def __init__(self) -> None:
        self.model = None
        self.model_name: str = ""
        self.explainer = None
        self.feature_names: List[str] = []
        self.label_encoders: Dict[str, Any] = {}
        self.scaler = None
        self.loaded = False
        self._load()

    @classmethod
    def instance(cls) -> "PredictionEngine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load(self) -> None:
        try:
            from ml.explainability.shap_explainer import load_best_model_and_explainer

            self.model, self.model_name, self.explainer, self.feature_names = (
                load_best_model_and_explainer()
            )
            self.label_encoders = joblib.load(PROJECT_ROOT / settings.paths["label_encoders"])
            self.scaler = joblib.load(PROJECT_ROOT / settings.paths["scaler"])
            self.loaded = True
            logger.info(f"PredictionEngine loaded. Best model = {self.model_name}")
        except FileNotFoundError as exc:
            logger.warning(
                f"No trained model artifacts found yet ({exc}). "
                f"Run `python -m ml.training.model_comparison` first. "
                f"API will report model_loaded=False until then."
            )
            self.loaded = False
        except Exception as exc:  # pragma: no cover
            logger.error(f"Failed to load PredictionEngine: {exc}")
            self.loaded = False

    def reload(self) -> None:
        """Called after /retrain to hot-swap in the newly trained best model."""
        self._load()

    def _prepare_input_matrix(self, features: Dict[str, Any]) -> np.ndarray:
        feat_cfg = settings.feature_config
        df = features_to_dataframe(features)

        for col in feat_cfg["categorical"]:
            le = self.label_encoders[col]
            known = set(le.classes_)
            val = str(df.at[0, col])
            if val not in known:
                val = le.classes_[0]
            df[col] = le.transform([val])

        for col in feat_cfg["binary_flags"]:
            df[col] = df[col].astype(int)

        df[feat_cfg["numerical"]] = self.scaler.transform(df[feat_cfg["numerical"]])

        ordered = df[self.feature_names]
        return ordered.values[0]

    def predict_proba(self, x_row: np.ndarray) -> float:
        x_batch = x_row.reshape(1, -1)
        if self.model_name == "deep_learning":
            proba = float(self.model.predict(x_batch, verbose=0).ravel()[0])
        else:
            proba = float(self.model.predict_proba(x_batch)[0, 1])
        return proba

    def predict_single(self, request) -> Dict[str, Any]:
        if not self.loaded:
            raise RuntimeError(
                "No trained model available. Train models first via "
                "`python -m ml.training.model_comparison`."
            )

        features = engineer_features(request)
        x_row = self._prepare_input_matrix(features)

        fraud_probability = self.predict_proba(x_row)
        business = evaluate_business_rules(fraud_probability)
        explanation = self.explainer.explain_instance(x_row, self.feature_names)

        result = {
            "request_id": str(uuid.uuid4()),
            "customer_id": request.customer_id,
            "seller_id": request.seller_id,
            "fraud_probability": round(fraud_probability, 4),
            "risk_score": business["risk_score"],
            "risk_level": business["risk_level"],
            "recommended_action": business["recommended_action"],
            "model_used": self.model_name,
            "top_features": explanation["top_features"],
            "business_explanation": explanation["business_explanation"],
        }
        return result


def get_engine() -> PredictionEngine:
    return PredictionEngine.instance()
