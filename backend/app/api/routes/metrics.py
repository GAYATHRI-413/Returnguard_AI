"""
backend/app/api/routes/metrics.py

GET /model_metrics        -> comparison table of all trained models
GET /feature_importance   -> global SHAP/feature-importance ranking
GET /prediction_history    -> paginated past predictions
GET /model_information     -> summary metadata about the deployed model
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.app.core.config import settings, PROJECT_ROOT
from backend.app.db import crud
from backend.app.db.database import get_db
from backend.app.services.explainability_service import get_global_feature_importance
from backend.app.services.prediction_service import get_engine
from backend.app.api.schemas.responses import (
    FeatureImportanceItem,
    FeatureImportanceResponse,
    ModelInformationResponse,
    ModelMetricEntry,
    ModelMetricsResponse,
    PredictionHistoryItem,
    PredictionHistoryResponse,
)

router = APIRouter(tags=["Metrics & History"])


@router.get("/model_metrics", response_model=ModelMetricsResponse, summary="Model comparison metrics")
def model_metrics(db: Session = Depends(get_db)):
    rows = crud.get_latest_model_metrics(db)
    if not rows:
        raise HTTPException(status_code=404, detail="No model metrics found. Train models first.")

    best = next((r.model_name for r in rows if r.is_best_model), rows[0].model_name)
    return ModelMetricsResponse(
        best_model=best,
        models=[
            ModelMetricEntry(
                model_name=r.model_name,
                accuracy=r.accuracy,
                precision=r.precision,
                recall=r.recall,
                f1_score=r.f1_score,
                roc_auc=r.roc_auc,
                is_best_model=r.is_best_model,
                trained_at=r.trained_at,
            )
            for r in rows
        ],
    )


@router.get("/feature_importance", response_model=FeatureImportanceResponse, summary="Global feature importance")
def feature_importance():
    try:
        data = get_global_feature_importance()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return FeatureImportanceResponse(
        model_used=data["model_used"],
        features=[FeatureImportanceItem(**f) for f in data["features"]],
    )


@router.get("/prediction_history", response_model=PredictionHistoryResponse, summary="Recent prediction history")
def prediction_history(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    rows = crud.get_prediction_history(db, limit=limit, offset=offset)
    items = [
        PredictionHistoryItem(
            id=r.id,
            request_id=r.request_id,
            customer_code=r.customer_code,
            seller_code=r.seller_code,
            fraud_probability=r.fraud_probability,
            risk_score=r.risk_score,
            risk_level=r.risk_level,
            recommended_action=r.recommended_action,
            model_used=r.model_used,
            business_explanation=r.business_explanation,
            created_at=r.created_at,
        )
        for r in rows
    ]
    return PredictionHistoryResponse(total=len(items), items=items)


@router.get("/model_information", response_model=ModelInformationResponse, summary="Deployed model metadata")
def model_information():
    engine = get_engine()
    if not engine.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Train models first.")

    metadata_path = PROJECT_ROOT / settings.paths["best_model_metadata"]
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    dataset_path = PROJECT_ROOT / settings.paths["raw_dataset"]
    dataset_size = None
    fraud_rate = None
    if dataset_path.exists():
        import pandas as pd

        df = pd.read_csv(dataset_path)
        dataset_size = len(df)
        fraud_rate = round(float(df["is_fraud"].mean()), 4)

    return ModelInformationResponse(
        app_name=settings.app_name,
        version=settings.app_version,
        best_model=engine.model_name,
        all_models_trained=list(metadata.get("all_results", {}).keys()),
        feature_count=len(engine.feature_names),
        dataset_size=dataset_size,
        fraud_rate=fraud_rate,
        metrics=metadata.get("all_results", {}),
    )
