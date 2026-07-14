"""
ml/training/train_utils.py

Shared utilities used by every model-training script:
  - compute_classification_metrics(): accuracy/precision/recall/f1/roc_auc
  - plot_confusion_matrix / plot_roc_curve / plot_precision_recall_curve
  - save_metrics_json(): persist metrics for the model comparison step
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.app.core.config import settings, PROJECT_ROOT  # noqa: E402
from backend.app.core.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)

PLOTS_DIR = PROJECT_ROOT / settings.paths["training_plots_dir"]
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
    }


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Path:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Genuine", "Fraud"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Genuine", "Fraud"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    out_path = PLOTS_DIR / f"confusion_matrix_{model_name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, model_name: str) -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {model_name}")
    ax.legend(loc="lower right")
    out_path = PLOTS_DIR / f"roc_curve_{model_name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_precision_recall_curve(y_true: np.ndarray, y_proba: np.ndarray, model_name: str) -> Path:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve - {model_name}")
    out_path = PLOTS_DIR / f"precision_recall_{model_name}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def save_metrics_json(model_name: str, metrics: Dict[str, float]) -> None:
    report_path = PROJECT_ROOT / settings.paths["model_comparison_report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if report_path.exists():
        with open(report_path) as f:
            data = json.load(f)
    data[model_name] = metrics
    with open(report_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved metrics for {model_name}: {metrics}")
