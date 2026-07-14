"""
backend/app/main.py

FastAPI application entrypoint.

Run locally:
    uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000

Docs available at:
    /docs   (Swagger UI)
    /redoc  (ReDoc)
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.core.logging_config import configure_logging, get_logger
from backend.app.db.database import init_db
from backend.app.api.routes import health, predict, metrics, retrain

configure_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    logger.info(f"Starting {settings.app_name} v{settings.app_version} ({settings.ENVIRONMENT})")
    init_db()

    from backend.app.db.database import SessionLocal
    from backend.app.db import crud

    with SessionLocal() as db:
        crud.seed_model_metrics_if_empty(db)

    from backend.app.services.prediction_service import get_engine

    engine = get_engine()
    if engine.loaded:
        logger.info(f"Prediction engine ready. Serving model: {engine.model_name}")
    else:
        logger.warning(
            "Prediction engine NOT ready -- no trained model found. "
            "Run `python -m ml.training.model_comparison` then restart, "
            "or call POST /retrain."
        )
    yield
    # --- shutdown ---
    logger.info("Shutting down ReturnGuard AI API.")


app = FastAPI(
    title=settings.app_name,
    description=(
        "AI Powered Return Fraud Detection & Risk Intelligence System. "
        "Predicts fraud probability, computes a 0-100 risk score, explains "
        "the prediction with SHAP, and recommends a business action."
    ),
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers -- all mounted under the configured API prefix (default /api/v1),
# except the bare root ("/") and "/health" which stay unprefixed for
# platform health-check conventions (Render, k8s, etc).
app.include_router(health.router)
app.include_router(predict.router, prefix=settings.api_prefix)
app.include_router(metrics.router, prefix=settings.api_prefix)
app.include_router(retrain.router, prefix=settings.api_prefix)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.app.main:app",
        host=settings.API_HOST,
        port=settings.effective_port,
        reload=settings.DEBUG,
    )
