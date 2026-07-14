"""
backend/app/api/schemas/prediction.py

Pydantic request models. These validate every field coming in from the
Streamlit frontend / API clients before anything touches the ML pipeline.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ReturnPredictionRequest(BaseModel):
    """A single return-fraud prediction request."""

    # Identifiers
    customer_id: Optional[str] = Field(default=None, description="Customer identifier, e.g. CUST100045")
    seller_id: Optional[str] = Field(default=None, description="Seller identifier, e.g. SELL1021")
    order_id: Optional[str] = Field(default=None, description="Order identifier")

    # Customer features
    customer_age: int = Field(..., ge=13, le=100)
    account_age_days: int = Field(..., ge=0, le=20000)
    total_orders: int = Field(..., ge=1, le=5000)
    total_returns: int = Field(..., ge=0, le=5000)
    customer_segment: str = Field(..., description="New | Regular | Loyal | VIP")
    days_since_last_purchase: int = Field(..., ge=0, le=5000)

    # Product features
    product_category: str
    product_price: float = Field(..., gt=0)
    order_value: float = Field(..., gt=0)
    product_weight_kg: float = Field(..., gt=0)

    # Seller features
    seller_rating: float = Field(..., ge=1.0, le=5.0)
    seller_defect_rate: float = Field(..., ge=0.0, le=1.0)

    # Delivery features
    delivery_days: int = Field(..., ge=0, le=60)
    distance_km: float = Field(..., ge=0)
    shipping_method: str = Field(..., description="Standard | Express | Same Day")

    # Return-specific / behaviour features
    return_reason: str
    avg_days_to_return: float = Field(..., ge=0, le=365)
    payment_method: str
    device_type: str
    num_payment_methods_used: int = Field(..., ge=1, le=20)
    used_multiple_addresses: bool = False
    used_multiple_payment_methods: bool = False
    return_without_tags: bool = False
    packaging_damaged_claim: bool = False
    is_holiday_purchase: bool = False

    @field_validator("total_returns")
    @classmethod
    def returns_not_exceed_orders(cls, v: int, info) -> int:
        total_orders = info.data.get("total_orders")
        if total_orders is not None and v > total_orders:
            raise ValueError("total_returns cannot exceed total_orders")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )


class BatchPredictionRequest(BaseModel):
    requests: List[ReturnPredictionRequest] = Field(..., min_length=1, max_length=500)


class RetrainRequest(BaseModel):
    """Optional overrides for a retraining run; empty body triggers defaults."""
    regenerate_dataset: bool = Field(default=False, description="If true, regenerate synthetic dataset first")
    num_records: Optional[int] = Field(default=None, ge=1000, le=200000)
