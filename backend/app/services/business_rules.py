"""
backend/app/services/business_rules.py

Converts a raw fraud probability (0-1) into:
  - risk_score (0-100)
  - risk_level (LOW / MEDIUM / HIGH)
  - recommended_action (AUTO_APPROVE / MANUAL_REVIEW / INVESTIGATION_REQUIRED)

Thresholds are configurable via config.yaml -> risk_scoring, so business
teams can retune sensitivity without touching code.
"""
from __future__ import annotations

from typing import Dict

from backend.app.core.config import settings


def probability_to_risk_score(fraud_probability: float) -> float:
    """Simple linear scaling of probability [0,1] -> risk score [0,100]."""
    scale_min = settings.risk_scoring_config.get("scale_min", 0)
    scale_max = settings.risk_scoring_config.get("scale_max", 100)
    score = scale_min + fraud_probability * (scale_max - scale_min)
    return round(min(max(score, scale_min), scale_max), 2)


def classify_risk(risk_score: float) -> Dict[str, str]:
    thresholds = settings.risk_scoring_config.get("thresholds", {})
    actions = settings.risk_scoring_config.get("actions", {})

    low_max = thresholds.get("low_risk_max", 35)
    medium_max = thresholds.get("medium_risk_max", 70)

    if risk_score <= low_max:
        return {"risk_level": "LOW", "recommended_action": actions.get("low", "AUTO_APPROVE")}
    if risk_score <= medium_max:
        return {"risk_level": "MEDIUM", "recommended_action": actions.get("medium", "MANUAL_REVIEW")}
    return {"risk_level": "HIGH", "recommended_action": actions.get("high", "INVESTIGATION_REQUIRED")}


def evaluate_business_rules(fraud_probability: float) -> Dict[str, object]:
    """Single entry point: probability -> full business decision payload."""
    risk_score = probability_to_risk_score(fraud_probability)
    classification = classify_risk(risk_score)
    return {
        "risk_score": risk_score,
        "risk_level": classification["risk_level"],
        "recommended_action": classification["recommended_action"],
    }
