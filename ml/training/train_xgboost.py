"""
ml/training/train_xgboost.py

Trains an XGBoost classifier. If the xgboost package is unavailable in the
environment, this script fails gracefully with a clear message rather than
crashing the whole pipeline (model_comparison.py checks for this).

Run:  python -m ml.training.train_xgboost
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.app.core.config import settings, PROJECT_ROOT  # noqa: E402
from backend.app.core.logging_config import get_logger  # noqa: E402
from ml.preprocessing.preprocessor import run_preprocessing  # noqa: E402
from ml.training.train_utils import (  # noqa: E402
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    save_metrics_json,
)

logger = get_logger(__name__)
MODEL_NAME = "xgboost"


def train() -> dict:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        logger.error("xgboost is not installed. `pip install xgboost` to enable this model.")
        raise

    splits = run_preprocessing()
    cfg = settings.training_config["xgboost"]

    model = XGBClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        eval_metric=cfg["eval_metric"],
        random_state=settings.dataset_config.get("random_seed", 42),
        n_jobs=-1,
    )
    logger.info("Training XGBoost...")
    model.fit(
        splits["X_train"], splits["y_train"],
        eval_set=[(splits["X_val"], splits["y_val"])],
        verbose=False,
    )

    y_pred = model.predict(splits["X_test"])
    y_proba = model.predict_proba(splits["X_test"])[:, 1]

    metrics = compute_classification_metrics(splits["y_test"], y_pred, y_proba)
    plot_confusion_matrix(splits["y_test"], y_pred, MODEL_NAME)
    plot_roc_curve(splits["y_test"], y_proba, MODEL_NAME)
    plot_precision_recall_curve(splits["y_test"], y_proba, MODEL_NAME)
    save_metrics_json(MODEL_NAME, metrics)

    model_path = PROJECT_ROOT / settings.paths["xgboost_model"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Saved {MODEL_NAME} -> {model_path} | metrics={metrics}")
    return metrics


if __name__ == "__main__":
    print(train())
