"""
ml/data/generate_dataset.py

Generates a realistic e-commerce RETURN FRAUD dataset (default: 12,000 records).

Design goals:
  - Includes Customer / Product / Seller / Delivery / Behaviour / Historical /
    Return features, plus the target `is_fraud`.
  - Fraud labels are NOT random. They come from a weighted business-rule
    scoring function that mimics how a real fraud team would reason, with
    injected noise so the problem isn't trivially separable.
  - Includes deliberate "hard" edge cases, e.g. a customer with a high
    return rate but who bought from a seller with a high defect rate --
    that combination should trend GENUINE, not fraud, and the dataset
    encodes that.

Run:
    python -m ml.data.generate_dataset
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.app.core.config import settings, PROJECT_ROOT  # noqa: E402
from backend.app.core.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)

RNG_SEED = settings.dataset_config.get("random_seed", 42)
NUM_RECORDS = settings.dataset_config.get("num_records", 12000)

PRODUCT_CATEGORIES = [
    "Electronics", "Fashion", "Footwear", "Home & Kitchen", "Beauty",
    "Sports", "Toys", "Books", "Jewelry", "Mobile Accessories",
]
RETURN_REASONS = [
    "Defective / Not Working", "Wrong Item Delivered", "Size Issue",
    "Not as Described", "Changed Mind", "Better Price Found",
    "Damaged in Transit", "Late Delivery", "Missing Parts", "Quality Issue",
]
PAYMENT_METHODS = ["Credit Card", "Debit Card", "UPI", "Net Banking", "Cash on Delivery", "Wallet"]
SHIPPING_METHODS = ["Standard", "Express", "Same Day"]
CUSTOMER_SEGMENTS = ["New", "Regular", "Loyal", "VIP"]
DEVICE_TYPES = ["Mobile App", "Desktop Web", "Mobile Web"]


def _rng() -> np.random.Generator:
    return np.random.default_rng(RNG_SEED)


def _generate_base_frame(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Generate raw, independently-sampled columns before rule-based labeling."""

    customer_age = rng.integers(18, 70, size=n)
    account_age_days = rng.integers(1, 2500, size=n)

    total_orders = np.clip(rng.negative_binomial(6, 0.35, size=n), 1, 400)
    # returns bounded by orders; base return propensity
    return_propensity = rng.beta(1.5, 6, size=n)  # skewed low
    total_returns = np.minimum(
        (total_orders * return_propensity).astype(int) + rng.integers(0, 2, size=n),
        total_orders,
    )

    product_category = rng.choice(PRODUCT_CATEGORIES, size=n)
    product_price = np.round(rng.gamma(shape=2.2, scale=45, size=n) + 5, 2)
    order_value = np.round(product_price * rng.uniform(0.9, 1.4, size=n), 2)

    seller_rating = np.round(np.clip(rng.normal(4.1, 0.6, size=n), 1.0, 5.0), 2)
    seller_defect_rate = np.round(np.clip(rng.beta(2, 20, size=n), 0.0, 0.6), 4)

    delivery_days = np.clip(rng.poisson(4, size=n) + 1, 1, 21)
    distance_km = np.round(rng.gamma(2.0, 60, size=n), 1)
    product_weight_kg = np.round(np.clip(rng.gamma(1.5, 1.2, size=n), 0.05, 30), 2)

    return_reason = rng.choice(RETURN_REASONS, size=n)
    payment_method = rng.choice(PAYMENT_METHODS, size=n)
    shipping_method = rng.choice(SHIPPING_METHODS, size=n)
    customer_segment = rng.choice(CUSTOMER_SEGMENTS, size=n, p=[0.30, 0.40, 0.20, 0.10])
    device_type = rng.choice(DEVICE_TYPES, size=n)

    avg_days_to_return = np.clip(rng.exponential(6, size=n), 0, 60).round(1)
    days_since_last_purchase = rng.integers(0, 400, size=n)

    num_payment_methods_used = np.clip(rng.poisson(1.4, size=n) + 1, 1, 6)
    used_multiple_addresses = rng.random(n) < 0.12
    used_multiple_payment_methods = num_payment_methods_used > 2
    return_without_tags = rng.random(n) < 0.10
    packaging_damaged_claim = rng.random(n) < 0.15
    is_holiday_purchase = rng.random(n) < 0.18

    df = pd.DataFrame({
        "customer_age": customer_age,
        "account_age_days": account_age_days,
        "total_orders": total_orders,
        "total_returns": total_returns,
        "product_category": product_category,
        "product_price": product_price,
        "order_value": order_value,
        "seller_rating": seller_rating,
        "seller_defect_rate": seller_defect_rate,
        "delivery_days": delivery_days,
        "distance_km": distance_km,
        "product_weight_kg": product_weight_kg,
        "return_reason": return_reason,
        "payment_method": payment_method,
        "shipping_method": shipping_method,
        "customer_segment": customer_segment,
        "device_type": device_type,
        "avg_days_to_return": avg_days_to_return,
        "days_since_last_purchase": days_since_last_purchase,
        "num_payment_methods_used": num_payment_methods_used,
        "used_multiple_addresses": used_multiple_addresses,
        "used_multiple_payment_methods": used_multiple_payment_methods,
        "return_without_tags": return_without_tags,
        "packaging_damaged_claim": packaging_damaged_claim,
        "is_holiday_purchase": is_holiday_purchase,
    })
    return df


def _engineer_intermediate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features needed both for realistic labeling AND as model inputs later."""
    df = df.copy()

    df["return_percentage"] = np.round(
        (df["total_returns"] / df["total_orders"].clip(lower=1)) * 100, 2
    )
    df["is_luxury_product"] = (df["product_price"] > 250).astype(int)
    df["is_high_value_order"] = (df["order_value"] > df["order_value"].quantile(0.85)).astype(int)

    price_bins = [-1, 30, 100, 300, np.inf]
    price_labels = ["Budget", "Mid-Range", "Premium", "Luxury"]
    df["price_category"] = pd.cut(df["product_price"], bins=price_bins, labels=price_labels).astype(str)

    df["seller_reliability_score"] = np.round(
        np.clip((df["seller_rating"] / 5.0) * (1 - df["seller_defect_rate"]) * 100, 0, 100), 2
    )

    df["delivery_risk_score"] = np.round(
        np.clip((df["delivery_days"] / 21.0) * 60 + (df["distance_km"] / df["distance_km"].max()) * 40, 0, 100),
        2,
    )

    df["customer_lifetime_value"] = np.round(
        df["total_orders"] * (df["order_value"] / df["total_orders"].clip(lower=1)) * 1.0, 2
    )
    df["avg_order_value"] = np.round(df["customer_lifetime_value"] / df["total_orders"].clip(lower=1), 2)

    # Seasonal return flag: purchases near "holiday season" more likely to be seasonal returns
    df["is_seasonal_return"] = ((df["is_holiday_purchase"]) & (df["avg_days_to_return"] < 20)).astype(int)

    # Customer risk score: combines return % with speed of return and account age (inverse)
    normalized_return_pct = np.clip(df["return_percentage"] / 100, 0, 1)
    fast_return_factor = np.clip(1 - (df["avg_days_to_return"] / 30), 0, 1)
    new_account_factor = np.clip(1 - (df["account_age_days"] / 730), 0, 1)
    df["customer_risk_score"] = np.round(
        np.clip(
            (normalized_return_pct * 45) + (fast_return_factor * 35) + (new_account_factor * 20),
            0, 100,
        ), 2,
    )

    return df


def _compute_fraud_score(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """
    Weighted business-rule fraud score (0-1 continuous), later thresholded
    with added noise into the binary `is_fraud` label.

    Key edge case handled explicitly:
      "Customer has many returns but seller defect rate is high" -> the
      seller_defect_rate term SUBTRACTS risk, so a high personal return rate
      paired with a high seller defect rate nets out to a LOWER fraud score
      than the same return rate with a low-defect seller. This teaches the
      model that not all high-return customers are fraudulent.
    """
    score = np.zeros(len(df))

    # High return percentage increases risk...
    score += np.clip(df["return_percentage"] / 100, 0, 1) * 0.22

    # ...but is heavily discounted when the seller's own defect rate explains it.
    seller_explains_returns = np.clip(df["seller_defect_rate"] / df["seller_defect_rate"].max(), 0, 1)
    score -= seller_explains_returns * 0.15

    # Very fast returns (return within a couple of days) are suspicious.
    score += np.clip(1 - (df["avg_days_to_return"] / 15), 0, 1) * 0.15

    # New accounts with high-value orders and quick returns are riskier.
    new_account_factor = np.clip(1 - (df["account_age_days"] / 400), 0, 1)
    score += new_account_factor * (df["is_high_value_order"]) * 0.12

    # Returning without tags / claiming packaging damage repeatedly is a fraud signal
    score += df["return_without_tags"].astype(int) * 0.10
    score += df["packaging_damaged_claim"].astype(int) * 0.05

    # Multiple addresses / payment methods (identity cycling) is a strong signal
    score += df["used_multiple_addresses"].astype(int) * 0.10
    score += df["used_multiple_payment_methods"].astype(int) * 0.06

    # Luxury / high value items with "changed mind" reason are more often abused
    changed_mind = (df["return_reason"] == "Changed Mind").astype(int)
    score += changed_mind * df["is_luxury_product"] * 0.08

    # High seller rating + defect-explained return reasons -> genuine, reduce risk
    genuine_reason = df["return_reason"].isin(
        ["Defective / Not Working", "Damaged in Transit", "Missing Parts", "Wrong Item Delivered"]
    ).astype(int)
    score -= genuine_reason * seller_explains_returns * 0.10

    # VIP / Loyal long-standing customers are statistically less likely to commit fraud
    loyal_customer = df["customer_segment"].isin(["Loyal", "VIP"]).astype(int)
    score -= loyal_customer * 0.08

    # Cash on delivery + new account combo (harder to trace) is riskier
    cod_new_account = ((df["payment_method"] == "Cash on Delivery") & (new_account_factor > 0.6)).astype(int)
    score += cod_new_account * 0.05

    # Random noise so the classes aren't perfectly separable (realistic messiness)
    score += rng.normal(0, 0.06, size=len(df))

    return np.clip(score, 0, 1)


def generate_dataset(n: int = NUM_RECORDS, seed: int = RNG_SEED) -> pd.DataFrame:
    logger.info(f"Generating synthetic dataset with n={n}, seed={seed}")
    rng = np.random.default_rng(seed)

    df = _generate_base_frame(n, rng)
    df = _engineer_intermediate_features(df)

    fraud_score = _compute_fraud_score(df, rng)
    df["fraud_risk_raw_score"] = np.round(fraud_score, 4)

    # Threshold tuned so overall fraud rate lands close to configured fraud_rate
    target_rate = settings.dataset_config.get("fraud_rate", 0.18)
    threshold = np.quantile(fraud_score, 1 - target_rate)
    df["is_fraud"] = (fraud_score >= threshold).astype(int)

    # IDs
    df.insert(0, "customer_id", [f"CUST{100000 + i}" for i in range(n)])
    df.insert(1, "seller_id", [f"SELL{rng.integers(1000, 1500)}" for _ in range(n)])
    df.insert(2, "order_id", [f"ORD{500000 + i}" for i in range(n)])

    logger.info(f"Dataset generated. Fraud rate = {df['is_fraud'].mean():.4f}")
    return df


def main() -> None:
    df = generate_dataset()
    out_path = PROJECT_ROOT / settings.paths["raw_dataset"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved dataset -> {out_path} ({len(df)} rows, {len(df.columns)} columns)")
    print(df["is_fraud"].value_counts(normalize=True))
    print(df.head(3).T)


if __name__ == "__main__":
    main()
