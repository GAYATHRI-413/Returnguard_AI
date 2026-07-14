"""
ml/training/train_deep_learning.py

Trains a Deep Neural Network (TensorFlow/Keras) for return fraud detection.

Architecture:
    Input Layer
      -> Dense(128, relu) -> BatchNorm -> Dropout(0.3)
      -> Dense(64, relu)  -> Dropout(0.3)
      -> Dense(32, relu)
      -> Dense(1, sigmoid)   [Output: fraud probability]

Callbacks:
    - EarlyStopping (on val_loss)
    - ModelCheckpoint (saves best val_loss weights)
    - ReduceLROnPlateau (learning rate scheduler)

Plots saved to ml/models/plots/:
    - training_accuracy_curve.png
    - training_loss_curve.png
    - confusion_matrix_deep_learning.png
    - roc_curve_deep_learning.png
    - precision_recall_deep_learning.png

Run:  python -m ml.training.train_deep_learning
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.app.core.config import settings, PROJECT_ROOT  # noqa: E402
from backend.app.core.logging_config import get_logger  # noqa: E402
from ml.preprocessing.preprocessor import run_preprocessing  # noqa: E402
from ml.training.train_utils import (  # noqa: E402
    PLOTS_DIR,
    compute_classification_metrics,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
    save_metrics_json,
)

logger = get_logger(__name__)
MODEL_NAME = "deep_learning"


def build_model(input_dim: int, cfg: dict):
    """Build the DNN per the architecture described in the module docstring."""
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers

    l2 = regularizers.l2(cfg.get("l2_regularization", 0.001))
    hidden_layers = cfg.get("hidden_layers", [128, 64, 32])
    dropout_rate = cfg.get("dropout_rate", 0.3)

    inputs = tf.keras.Input(shape=(input_dim,), name="input_features")

    x = layers.Dense(hidden_layers[0], activation="relu", kernel_regularizer=l2)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(hidden_layers[1], activation="relu", kernel_regularizer=l2)(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(hidden_layers[2], activation="relu", kernel_regularizer=l2)(x)

    outputs = layers.Dense(1, activation="sigmoid", name="fraud_probability")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="ReturnGuard_DNN")

    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.get("learning_rate", 0.001))
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def plot_training_curves(history) -> None:
    hist = history.history

    # Accuracy curve
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hist["accuracy"], label="Train Accuracy")
    ax.plot(hist["val_accuracy"], label="Validation Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title("Training vs Validation Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "training_accuracy_curve.png", dpi=120)
    plt.close(fig)

    # Loss curve
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hist["loss"], label="Train Loss")
    ax.plot(hist["val_loss"], label="Validation Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "training_loss_curve.png", dpi=120)
    plt.close(fig)


def train() -> dict:
    import tensorflow as tf

    splits = run_preprocessing()
    cfg = settings.training_config["deep_learning"]

    tf.random.set_seed(settings.dataset_config.get("random_seed", 42))

    model = build_model(input_dim=splits["X_train"].shape[1], cfg=cfg)
    model.summary(print_fn=lambda line: logger.info(line))

    checkpoint_path = PROJECT_ROOT / settings.paths["deep_learning_model"]
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.get("early_stopping_patience", 10),
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=cfg.get("lr_scheduler_factor", 0.5),
            patience=cfg.get("lr_scheduler_patience", 5),
            min_lr=1e-6,
        ),
    ]

    logger.info("Training Deep Neural Network...")
    history = model.fit(
        splits["X_train"], splits["y_train"],
        validation_data=(splits["X_val"], splits["y_val"]),
        epochs=cfg.get("epochs", 100),
        batch_size=cfg.get("batch_size", 64),
        callbacks=callbacks,
        verbose=2,
    )

    plot_training_curves(history)

    y_proba = model.predict(splits["X_test"]).ravel()
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = compute_classification_metrics(splits["y_test"], y_pred, y_proba)
    plot_confusion_matrix(splits["y_test"], y_pred, MODEL_NAME)
    plot_roc_curve(splits["y_test"], y_proba, MODEL_NAME)
    plot_precision_recall_curve(splits["y_test"], y_proba, MODEL_NAME)
    save_metrics_json(MODEL_NAME, metrics)

    # ModelCheckpoint already saved the best weights to checkpoint_path,
    # but we save again explicitly in case training finished on last epoch.
    model.save(checkpoint_path)
    logger.info(f"Saved {MODEL_NAME} -> {checkpoint_path} | metrics={metrics}")
    return metrics


if __name__ == "__main__":
    print(train())
