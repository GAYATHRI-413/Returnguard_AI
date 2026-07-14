"""
ml/training/train_random_forest.py

Trains a Random Forest classifier.
Run:  python -m ml.training.train_random_forest
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier

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
MODEL_NAME = "random_forest"


def train() -> dict:
    splits = run_preprocessing()
    cfg = settings.training_config["random_forest"]

    model = RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        min_samples_split=cfg["min_samples_split"],
        min_samples_leaf=cfg["min_samples_leaf"],
        class_weight=cfg["class_weight"],
        n_jobs=cfg.get("n_jobs", -1),
        random_state=settings.dataset_config.get("random_seed", 42),
    )
    logger.info("Training Random Forest...")
    model.fit(splits["X_train"], splits["y_train"])

    y_pred = model.predict(splits["X_test"])
    y_proba = model.predict_proba(splits["X_test"])[:, 1]

    metrics = compute_classification_metrics(splits["y_test"], y_pred, y_proba)
    plot_confusion_matrix(splits["y_test"], y_pred, MODEL_NAME)
    plot_roc_curve(splits["y_test"], y_proba, MODEL_NAME)
    plot_precision_recall_curve(splits["y_test"], y_proba, MODEL_NAME)
    save_metrics_json(MODEL_NAME, metrics)

    model_path = PROJECT_ROOT / settings.paths["random_forest_model"]
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    logger.info(f"Saved {MODEL_NAME} -> {model_path} | metrics={metrics}")
    return metrics


if __name__ == "__main__":
    print(train())
