"""
backend/app/api/routes/health.py

GET /            -> basic root info
GET /health      -> liveness/readiness probe (model loaded? DB reachable?)
"""
from __future__ import annotations

from fastapi import APIRouter

from backend.app.core.config import settings
from backend.app.db.database import engine
from backend.app.services.prediction_service import get_engine
from backend.app.api.schemas.responses import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/", summary="Root")
def read_root():
    return {
        "message": f"Welcome to {settings.app_name} API",
        "version": settings.app_version,
        "docs": "/docs",
    }


@router.get("/health", response_model=HealthResponse, summary="Health check")
def health_check():
    prediction_engine = get_engine()

    db_ok = True
    try:
        with engine.connect() as conn:
            conn.execute(__import__("sqlalchemy").text("SELECT 1"))
    except Exception:
        db_ok = False

    return HealthResponse(
        status="ok" if prediction_engine.loaded and db_ok else "degraded",
        app_name=settings.app_name,
        version=settings.app_version,
        model_loaded=prediction_engine.loaded,
        best_model=prediction_engine.model_name or None,
        database_connected=db_ok,
    )
