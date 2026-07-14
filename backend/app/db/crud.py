"""
backend/app/db/crud.py

Reusable CRUD helper functions over the ORM models. Keeps route handlers
thin and free of raw SQLAlchemy query logic.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from sqlalchemy import desc
from sqlalchemy.orm import Session

from backend.app.db import models


# ------------------------------------------------------------------------
# Prediction History
# ------------------------------------------------------------------------
def create_prediction_record(
    db: Session,
    *,
    request_id: str,
    customer_code: Optional[str],
    seller_code: Optional[str],
    fraud_probability: float,
    risk_score: float,
    risk_level: str,
    recommended_action: str,
    model_used: str,
    top_features: List[Dict[str, Any]],
    business_explanation: str,
    input_payload: Dict[str, Any],
) -> models.PredictionHistory:
    record = models.PredictionHistory(
        request_id=request_id,
        customer_code=customer_code,
        seller_code=seller_code,
        fraud_probability=fraud_probability,
        risk_score=risk_score,
        risk_level=risk_level,
        recommended_action=recommended_action,
        model_used=model_used,
        top_features_json=json.dumps(top_features),
        business_explanation=business_explanation,
        input_payload_json=json.dumps(input_payload, default=str),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_prediction_history(
    db: Session, limit: int = 50, offset: int = 0
) -> List[models.PredictionHistory]:
    return (
        db.query(models.PredictionHistory)
        .order_by(desc(models.PredictionHistory.created_at))
        .offset(offset)
        .limit(limit)
        .all()
    )


def get_prediction_by_request_id(db: Session, request_id: str) -> Optional[models.PredictionHistory]:
    return (
        db.query(models.PredictionHistory)
        .filter(models.PredictionHistory.request_id == request_id)
        .first()
    )


# ------------------------------------------------------------------------
# Model Metrics
# ------------------------------------------------------------------------
def save_model_metrics(
    db: Session,
    *,
    model_name: str,
    accuracy: float,
    precision: float,
    recall: float,
    f1_score: float,
    roc_auc: float,
    is_best_model: bool = False,
) -> models.ModelMetrics:
    record = models.ModelMetrics(
        model_name=model_name,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        roc_auc=roc_auc,
        is_best_model=is_best_model,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_latest_model_metrics(db: Session) -> List[models.ModelMetrics]:
    return db.query(models.ModelMetrics).order_by(desc(models.ModelMetrics.trained_at)).all()


def seed_model_metrics_if_empty(db: Session) -> None:
    """
    On a fresh database (e.g. a new Render Postgres instance), the
    model_metrics table starts empty because metrics are normally written
    only by ml/training/model_comparison.py running against that exact DB.
    Since the models are already pre-trained and shipped with the repo,
    seed the table once from ml/models/best_model_metadata.json instead of
    requiring a live retrain.
    """
    import json
    from backend.app.core.config import PROJECT_ROOT, settings

    if db.query(models.ModelMetrics).first() is not None:
        return  # already seeded

    metadata_path = PROJECT_ROOT / settings.paths["best_model_metadata"]
    if not metadata_path.exists():
        return

    with open(metadata_path) as f:
        metadata = json.load(f)

    best_model = metadata["best_model"]
    for model_name, m in metadata["all_results"].items():
        save_model_metrics(
            db,
            model_name=model_name,
            accuracy=m["accuracy"],
            precision=m["precision"],
            recall=m["recall"],
            f1_score=m["f1_score"],
            roc_auc=m["roc_auc"],
            is_best_model=(model_name == best_model),
        )


# ------------------------------------------------------------------------
# Fraud Logs
# ------------------------------------------------------------------------
def create_fraud_log(
    db: Session,
    *,
    prediction_id: Optional[int],
    customer_code: Optional[str],
    risk_level: str,
    reason_summary: str,
) -> models.FraudLog:
    record = models.FraudLog(
        prediction_id=prediction_id,
        customer_code=customer_code,
        risk_level=risk_level,
        reason_summary=reason_summary,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


# ------------------------------------------------------------------------
# Audit Logs
# ------------------------------------------------------------------------
def create_audit_log(db: Session, *, actor: str, action: str, detail: str = "") -> models.AuditLog:
    record = models.AuditLog(actor=actor, action=action, detail=detail)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_audit_logs(db: Session, limit: int = 100) -> List[models.AuditLog]:
    return db.query(models.AuditLog).order_by(desc(models.AuditLog.created_at)).limit(limit).all()
