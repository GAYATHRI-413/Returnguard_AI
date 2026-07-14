"""
backend/app/api/routes/retrain.py

POST /retrain -> triggers a full retraining run (optionally regenerating the
                 synthetic dataset first), then hot-reloads the prediction
                 engine with the new best model.

NOTE: Runs synchronously for simplicity/transparency in this project. In a
real production system this would be pushed to a background job queue
(Celery / RQ / a training orchestrator) so the HTTP request doesn't block
for minutes; the docstring in main.py's README documents this tradeoff.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.core.logging_config import get_logger
from backend.app.services.prediction_service import get_engine
from backend.app.api.schemas.prediction import RetrainRequest
from backend.app.api.schemas.responses import RetrainResponse

router = APIRouter(tags=["Retrain"])
logger = get_logger(__name__)


@router.post("/retrain", response_model=RetrainResponse, summary="Retrain all models")
def retrain(payload: RetrainRequest):
    try:
        if payload.regenerate_dataset:
            from ml.data.generate_dataset import generate_dataset, main as regen_main
            import backend.app.core.config as config_module

            if payload.num_records:
                # Temporarily patch the record count for this run only.
                config_module.settings.yaml_config["dataset"]["num_records"] = payload.num_records
            regen_main()
            logger.info("Dataset regenerated as part of /retrain.")

        from ml.training.model_comparison import train_all_models, build_comparison_table, select_best_model, persist_best_model_metadata

        results = train_all_models()
        table = build_comparison_table(results)
        best_model = select_best_model(results)
        persist_best_model_metadata(best_model, results)

        engine = get_engine()
        engine.reload()

        return RetrainResponse(
            status="success",
            message=f"Retraining complete. Best model: {best_model}",
            best_model=best_model,
            comparison=table.to_dict(orient="index"),
        )
    except Exception as exc:
        logger.error(f"Retraining failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {exc}")
