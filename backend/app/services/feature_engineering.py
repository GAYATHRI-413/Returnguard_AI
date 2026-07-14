"""
backend/app/services/feature_engineering.py

Turns a single ReturnPredictionRequest into the exact same engineered
feature set used during training (ml/data/generate_dataset.py's
_engineer_intermediate_features), so the live model sees data shaped
identically to what it was trained on.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from backend.app.api.schemas.prediction import ReturnPredictionRequest


def _price_category(price: float) -> str:
    if price <= 30:
        return "Budget"
    if price <= 100:
        return "Mid-Range"
    if price <= 300:
        return "Premium"
    return "Luxury"


def engineer_features(request: ReturnPredictionRequest) -> Dict[str, Any]:
    """
    Reconstructs every engineered feature from a raw request payload using
    the same formulas as the offline dataset generator.
    """
    total_orders = max(request.total_orders, 1)
    return_percentage = round((request.total_returns / total_orders) * 100, 2)

    is_luxury_product = int(request.product_price > 250)
    # NOTE: "is_high_value_order" in training was relative to the dataset's
    # 85th percentile order value (~ historically around $250-300). We use a
    # fixed business threshold at inference time since there's no live batch
    # distribution to compute a percentile against.
    is_high_value_order = int(request.order_value > 250)

    seller_reliability_score = round(
        float(np.clip((request.seller_rating / 5.0) * (1 - request.seller_defect_rate) * 100, 0, 100)), 2
    )

    # Normalize delivery risk using the same rough scale as training
    # (21 days max delivery, ~1200km max distance observed in training data)
    delivery_risk_score = round(
        float(np.clip((request.delivery_days / 21.0) * 60 + (min(request.distance_km, 1200) / 1200) * 40, 0, 100)),
        2,
    )

    customer_lifetime_value = round(request.total_orders * request.order_value, 2)
    avg_order_value = round(customer_lifetime_value / total_orders, 2)

    is_seasonal_return = int(request.is_holiday_purchase and request.avg_days_to_return < 20)

    normalized_return_pct = min(return_percentage / 100, 1.0)
    fast_return_factor = float(np.clip(1 - (request.avg_days_to_return / 30), 0, 1))
    new_account_factor = float(np.clip(1 - (request.account_age_days / 730), 0, 1))
    customer_risk_score = round(
        float(np.clip(
            (normalized_return_pct * 45) + (fast_return_factor * 35) + (new_account_factor * 20), 0, 100
        )), 2,
    )

    features: Dict[str, Any] = {
        # numerical
        "order_value": request.order_value,
        "product_price": request.product_price,
        "customer_age": request.customer_age,
        "account_age_days": request.account_age_days,
        "total_orders": request.total_orders,
        "total_returns": request.total_returns,
        "return_percentage": return_percentage,
        "avg_days_to_return": request.avg_days_to_return,
        "customer_lifetime_value": customer_lifetime_value,
        "avg_order_value": avg_order_value,
        "seller_rating": request.seller_rating,
        "seller_defect_rate": request.seller_defect_rate,
        "seller_reliability_score": seller_reliability_score,
        "delivery_days": request.delivery_days,
        "delivery_risk_score": delivery_risk_score,
        "customer_risk_score": customer_risk_score,
        "days_since_last_purchase": request.days_since_last_purchase,
        "num_payment_methods_used": request.num_payment_methods_used,
        "distance_km": request.distance_km,
        "product_weight_kg": request.product_weight_kg,
        # categorical
        "product_category": request.product_category,
        "price_category": _price_category(request.product_price),
        "return_reason": request.return_reason,
        "payment_method": request.payment_method,
        "shipping_method": request.shipping_method,
        "customer_segment": request.customer_segment,
        "device_type": request.device_type,
        # binary flags
        "is_luxury_product": is_luxury_product,
        "is_holiday_purchase": int(request.is_holiday_purchase),
        "is_seasonal_return": is_seasonal_return,
        "is_high_value_order": is_high_value_order,
        "used_multiple_addresses": int(request.used_multiple_addresses),
        "used_multiple_payment_methods": int(request.used_multiple_payment_methods),
        "return_without_tags": int(request.return_without_tags),
        "packaging_damaged_claim": int(request.packaging_damaged_claim),
    }
    return features


def features_to_dataframe(features: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([features])
