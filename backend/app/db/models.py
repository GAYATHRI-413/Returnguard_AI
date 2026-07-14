"""
backend/app/db/models.py

SQLAlchemy ORM models: Customer, Seller, PredictionHistory, ModelMetrics,
FraudLog, AuditLog.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
)

from backend.app.db.database import Base


class Customer(Base):
    __tablename__ = "customers"

    id = Column(Integer, primary_key=True, index=True)
    customer_code = Column(String(64), unique=True, index=True, nullable=False)
    name = Column(String(128), nullable=True)
    age = Column(Integer, nullable=True)
    account_age_days = Column(Integer, nullable=True)
    total_orders = Column(Integer, default=0)
    total_returns = Column(Integer, default=0)
    customer_lifetime_value = Column(Float, default=0.0)
    customer_segment = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)


class Seller(Base):
    __tablename__ = "sellers"

    id = Column(Integer, primary_key=True, index=True)
    seller_code = Column(String(64), unique=True, index=True, nullable=False)
    name = Column(String(128), nullable=True)
    rating = Column(Float, default=0.0)
    defect_rate = Column(Float, default=0.0)
    reliability_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=dt.datetime.utcnow)


class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String(64), index=True, nullable=False)
    customer_code = Column(String(64), index=True, nullable=True)
    seller_code = Column(String(64), index=True, nullable=True)

    fraud_probability = Column(Float, nullable=False)
    risk_score = Column(Float, nullable=False)
    risk_level = Column(String(32), nullable=False)      # LOW / MEDIUM / HIGH
    recommended_action = Column(String(64), nullable=False)
    model_used = Column(String(64), nullable=False)

    top_features_json = Column(Text, nullable=True)       # JSON-encoded SHAP top features
    business_explanation = Column(Text, nullable=True)
    input_payload_json = Column(Text, nullable=True)      # full request payload, JSON-encoded

    created_at = Column(DateTime, default=dt.datetime.utcnow, index=True)


class ModelMetrics(Base):
    __tablename__ = "model_metrics"

    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(64), nullable=False)
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    is_best_model = Column(Boolean, default=False)
    trained_at = Column(DateTime, default=dt.datetime.utcnow)


class FraudLog(Base):
    __tablename__ = "fraud_logs"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, nullable=True, index=True)
    customer_code = Column(String(64), index=True, nullable=True)
    risk_level = Column(String(32), nullable=False)
    reason_summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    actor = Column(String(64), nullable=True)     # e.g. "system", "admin"
    action = Column(String(128), nullable=False)  # e.g. "PREDICT", "RETRAIN", "LOGIN"
    detail = Column(Text, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow, index=True)
