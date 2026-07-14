"""
backend/app/api/routes/predict.py

POST /predict          -> single return-fraud prediction
POST /batch_predict     -> batch prediction (up to 500 requests)
"""
from __future__ import annotations

import datetime as dt

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.app.core.logging_config import get_logger
from backend.app.db import crud
from backend.app.db.database import get_db
from backend.app.services.prediction_service import get_engine
from backend.app.api.schemas.prediction import BatchPredictionRequest, ReturnPredictionRequest
from backend.app.api.schemas.responses import (
    BatchPredictionResponse,
    PredictionResponse,
    TopFeature,
)

router = APIRouter(tags=["Prediction"])
logger = get_logger(__name__)


def _run_and_persist(request: ReturnPredictionRequest, db: Session) -> PredictionResponse:
    engine = get_engine()
    if not engine.loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not trained yet. Run `python -m ml.training.model_comparison` first.",
        )

    result = engine.predict_single(request)

    crud.create_prediction_record(
        db,
        request_id=result["request_id"],
        customer_code=result["customer_id"],
        seller_code=result["seller_id"],
        fraud_probability=result["fraud_probability"],
        risk_score=result["risk_score"],
        risk_level=result["risk_level"],
        recommended_action=result["recommended_action"],
        model_used=result["model_used"],
        top_features=result["top_features"],
        business_explanation=result["business_explanation"],
        input_payload=request.model_dump(),
    )

    if result["risk_level"] == "HIGH":
        crud.create_fraud_log(
            db,
            prediction_id=None,
            customer_code=result["customer_id"],
            risk_level=result["risk_level"],
            reason_summary=result["business_explanation"],
        )

    crud.create_audit_log(db, actor="api", action="PREDICT", detail=result["request_id"])

    return PredictionResponse(
        request_id=result["request_id"],
        customer_id=result["customer_id"],
        seller_id=result["seller_id"],
        fraud_probability=result["fraud_probability"],
        risk_score=result["risk_score"],
        risk_level=result["risk_level"],
        recommended_action=result["recommended_action"],
        model_used=result["model_used"],
        top_features=[TopFeature(**f) for f in result["top_features"]],
        business_explanation=result["business_explanation"],
        created_at=dt.datetime.now(dt.timezone.utc),
    )


@router.post("/predict", response_model=PredictionResponse, summary="Predict return fraud risk")
def predict(request: ReturnPredictionRequest, db: Session = Depends(get_db)):
    return _run_and_persist(request, db)


@router.post("/batch_predict", response_model=BatchPredictionResponse, summary="Batch predict return fraud risk")
def batch_predict(payload: BatchPredictionRequest, db: Session = Depends(get_db)):
    results = [_run_and_persist(req, db) for req in payload.requests]

    high = sum(1 for r in results if r.risk_level == "HIGH")
    medium = sum(1 for r in results if r.risk_level == "MEDIUM")
    low = sum(1 for r in results if r.risk_level == "LOW")

    return BatchPredictionResponse(
        results=results,
        total_processed=len(results),
        high_risk_count=high,
        medium_risk_count=medium,
        low_risk_count=low,
    )
