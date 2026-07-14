"""
ml/training/model_comparison.py

Orchestrates training of all 4 models, builds a comparison table, and
determines + persists the best model (by ROC-AUC, tie-broken by F1) so the
backend's prediction_service.py knows which artifact to load.

Run:  python -m ml.training.model_comparison
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.app.core.config import settings, PROJECT_ROOT  # noqa: E402
from backend.app.core.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)

MODEL_KEY_TO_PATH_KEY = {
    "logistic_regression": "logistic_regression_model",
    "random_forest": "random_forest_model",
    "xgboost": "xgboost_model",
    "deep_learning": "deep_learning_model",
}


def train_all_models() -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}

    from ml.training import train_logistic_regression, train_random_forest, train_deep_learning

    logger.info("=== Training Logistic Regression ===")
    results["logistic_regression"] = train_logistic_regression.train()

    logger.info("=== Training Random Forest ===")
    results["random_forest"] = train_random_forest.train()

    try:
        logger.info("=== Training XGBoost ===")
        from ml.training import train_xgboost
        results["xgboost"] = train_xgboost.train()
    except ImportError:
        logger.warning("Skipping XGBoost: package not installed in this environment.")

    try:
        logger.info("=== Training Deep Neural Network ===")
        results["deep_learning"] = train_deep_learning.train()
    except ImportError:
        logger.warning("Skipping Deep Learning: tensorflow not installed in this environment.")

    return results


def build_comparison_table(results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(results).T
    df = df[["accuracy", "precision", "recall", "f1_score", "roc_auc"]]
    df = df.sort_values("roc_auc", ascending=False)
    return df


def select_best_model(results: Dict[str, Dict[str, float]]) -> str:
    best_model = max(results.items(), key=lambda kv: (kv[1]["roc_auc"], kv[1]["f1_score"]))[0]
    return best_model


def persist_best_model_metadata(best_model: str, results: Dict[str, Dict[str, float]]) -> None:
    metadata = {
        "best_model": best_model,
        "model_path_key": MODEL_KEY_TO_PATH_KEY[best_model],
        "metrics": results[best_model],
        "all_results": results,
    }
    out_path = PROJECT_ROOT / settings.paths["best_model_metadata"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Best model: {best_model} | Saved metadata -> {out_path}")


def main() -> None:
    results = train_all_models()
    table = build_comparison_table(results)

    print("\n" + "=" * 70)
    print("MODEL COMPARISON TABLE")
    print("=" * 70)
    print(table.to_string())
    print("=" * 70)

    best_model = select_best_model(results)
    print(f"\nBest performing model: {best_model.upper()}")

    persist_best_model_metadata(best_model, results)

    # Push metrics into the DB too, for the /model_metrics API + Streamlit dashboard
    try:
        from backend.app.db.database import SessionLocal, init_db
        from backend.app.db import crud

        init_db()
        db = SessionLocal()
        for model_name, metrics in results.items():
            crud.save_model_metrics(
                db,
                model_name=model_name,
                accuracy=metrics["accuracy"],
                precision=metrics["precision"],
                recall=metrics["recall"],
                f1_score=metrics["f1_score"],
                roc_auc=metrics["roc_auc"],
                is_best_model=(model_name == best_model),
            )
        db.close()
        logger.info("Model metrics persisted to database.")
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Could not persist metrics to DB: {exc}")


if __name__ == "__main__":
    main()
