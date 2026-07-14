"""
backend/app/api/schemas/responses.py

Pydantic response models returned by the API. Structured, documented, and
consistent across single/batch prediction, metrics, and history endpoints.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TopFeature(BaseModel):
    feature: str
    business_label: str
    shap_value: float
    direction: str  # "increases_risk" | "decreases_risk"


class PredictionResponse(BaseModel):
    request_id: str
    customer_id: Optional[str] = None
    seller_id: Optional[str] = None

    fraud_probability: float = Field(..., ge=0, le=1)
    risk_score: float = Field(..., ge=0, le=100)
    risk_level: str  # LOW | MEDIUM | HIGH
    recommended_action: str  # AUTO_APPROVE | MANUAL_REVIEW | INVESTIGATION_REQUIRED

    model_used: str
    top_features: List[TopFeature]
    business_explanation: str

    created_at: dt.datetime


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total_processed: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    model_loaded: bool
    best_model: Optional[str] = None
    database_connected: bool


class ModelMetricEntry(BaseModel):
    model_name: str
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    roc_auc: Optional[float]
    is_best_model: bool
    trained_at: dt.datetime


class ModelMetricsResponse(BaseModel):
    best_model: str
    models: List[ModelMetricEntry]


class FeatureImportanceItem(BaseModel):
    feature: str
    business_label: str
    importance: float


class FeatureImportanceResponse(BaseModel):
    model_used: str
    features: List[FeatureImportanceItem]


class PredictionHistoryItem(BaseModel):
    id: int
    request_id: str
    customer_code: Optional[str]
    seller_code: Optional[str]
    fraud_probability: float
    risk_score: float
    risk_level: str
    recommended_action: str
    model_used: str
    business_explanation: Optional[str]
    created_at: dt.datetime


class PredictionHistoryResponse(BaseModel):
    total: int
    items: List[PredictionHistoryItem]


class ModelInformationResponse(BaseModel):
    app_name: str
    version: str
    best_model: str
    all_models_trained: List[str]
    feature_count: int
    dataset_size: Optional[int]
    fraud_rate: Optional[float]
    metrics: Dict[str, Any]


class RetrainResponse(BaseModel):
    status: str
    message: str
    best_model: Optional[str] = None
    comparison: Optional[Dict[str, Any]] = None
