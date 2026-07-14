"""
tests/test_api.py

Basic API test suite using FastAPI's TestClient. Assumes models have
already been trained (`python -m ml.training.model_comparison`) before
running these tests.

Run:
    pytest tests/ -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.app.main import app

SAMPLE_PAYLOAD = {
    "customer_id": "CUST100045",
    "seller_id": "SELL1021",
    "order_id": "ORD500123",
    "customer_age": 29,
    "account_age_days": 120,
    "total_orders": 14,
    "total_returns": 6,
    "customer_segment": "Regular",
    "days_since_last_purchase": 12,
    "product_category": "Electronics",
    "product_price": 349.99,
    "order_value": 379.99,
    "product_weight_kg": 1.2,
    "seller_rating": 4.2,
    "seller_defect_rate": 0.03,
    "delivery_days": 3,
    "distance_km": 45.0,
    "shipping_method": "Express",
    "return_reason": "Changed Mind",
    "avg_days_to_return": 1.5,
    "payment_method": "Cash on Delivery",
    "device_type": "Mobile App",
    "num_payment_methods_used": 3,
    "used_multiple_addresses": True,
    "used_multiple_payment_methods": True,
    "return_without_tags": True,
    "packaging_damaged_claim": False,
    "is_holiday_purchase": False,
}


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


def test_root(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "message" in resp.json()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "model_loaded" in body


def test_predict(client):
    resp = client.post("/api/v1/predict", json=SAMPLE_PAYLOAD)
    assert resp.status_code in (200, 503)  # 503 if models not yet trained
    if resp.status_code == 200:
        body = resp.json()
        assert 0.0 <= body["fraud_probability"] <= 1.0
        assert 0.0 <= body["risk_score"] <= 100.0
        assert body["risk_level"] in ("LOW", "MEDIUM", "HIGH")
        assert body["recommended_action"] in ("AUTO_APPROVE", "MANUAL_REVIEW", "INVESTIGATION_REQUIRED")
        assert len(body["top_features"]) > 0


def test_predict_validation_error(client):
    bad_payload = dict(SAMPLE_PAYLOAD)
    bad_payload["total_returns"] = 999999  # exceeds total_orders -> should fail validation
    resp = client.post("/api/v1/predict", json=bad_payload)
    assert resp.status_code == 422


def test_prediction_history(client):
    resp = client.get("/api/v1/prediction_history")
    assert resp.status_code == 200
    assert "items" in resp.json()


def test_model_information(client):
    resp = client.get("/api/v1/model_information")
    assert resp.status_code in (200, 503)
