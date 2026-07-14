"""
ml/preprocessing/preprocessor.py

Turns the raw generated dataset into model-ready train/val/test splits:
  - Label-encodes categorical columns (saved for reuse at inference time)
  - Scales numerical columns with StandardScaler (saved for reuse)
  - Persists the final feature column order (critical: inference must match)
  - Splits into train / validation / test

Run:
    python -m ml.preprocessing.preprocessor
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[2]))

from backend.app.core.config import settings, PROJECT_ROOT  # noqa: E402
from backend.app.core.logging_config import get_logger  # noqa: E402

logger = get_logger(__name__)


def load_raw_dataset() -> pd.DataFrame:
    path = PROJECT_ROOT / settings.paths["raw_dataset"]
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. Run `python -m ml.data.generate_dataset` first."
        )
    return pd.read_csv(path)


def build_feature_matrix(df: pd.DataFrame, fit: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Encode categoricals + assemble the final feature matrix X (all numeric).

    If fit=True: fits new LabelEncoders/StandardScaler and saves them.
    If fit=False: loads previously saved encoders/scaler (for inference-time
    consistency) and applies them, gracefully handling unseen categories.
    """
    feat_cfg = settings.feature_config
    numerical_cols = feat_cfg["numerical"]
    categorical_cols = feat_cfg["categorical"]
    binary_cols = feat_cfg["binary_flags"]

    df = df.copy()

    encoders_path = PROJECT_ROOT / settings.paths["label_encoders"]
    scaler_path = PROJECT_ROOT / settings.paths["scaler"]

    # ---- Categorical encoding ----
    if fit:
        encoders: Dict[str, LabelEncoder] = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        encoders_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(encoders, encoders_path)
    else:
        encoders = joblib.load(encoders_path)
        for col in categorical_cols:
            le = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(lambda v: v if v in known else le.classes_[0])
            df[col] = le.transform(df[col])

    # ---- Binary flags -> int ----
    for col in binary_cols:
        df[col] = df[col].astype(int)

    # ---- Numerical scaling ----
    if fit:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        df[numerical_cols] = scaler.transform(df[numerical_cols])

    feature_columns = numerical_cols + categorical_cols + binary_cols
    X = df[feature_columns]

    if fit:
        cols_path = PROJECT_ROOT / settings.paths["feature_columns"]
        cols_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cols_path, "w") as f:
            json.dump(feature_columns, f, indent=2)

    return X, {"numerical": numerical_cols, "categorical": categorical_cols, "binary": binary_cols}


def run_preprocessing() -> Dict[str, np.ndarray]:
    df = load_raw_dataset()
    target_col = settings.feature_config["target"]

    X, _ = build_feature_matrix(df, fit=True)
    y = df[target_col].values

    test_size = settings.dataset_config.get("train_test_split", 0.20)
    val_size = settings.dataset_config.get("validation_split", 0.15)
    seed = settings.dataset_config.get("random_seed", 42)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size, random_state=seed, stratify=y_train_full
    )

    logger.info(
        f"Preprocessing complete. Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}, "
        f"Features={X.shape[1]}"
    )

    processed_path = PROJECT_ROOT / settings.paths["processed_dataset"]
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    full_processed = X.copy()
    full_processed[target_col] = y
    full_processed.to_csv(processed_path, index=False)

    return {
        "X_train": X_train.values, "y_train": y_train,
        "X_val": X_val.values, "y_val": y_val,
        "X_test": X_test.values, "y_test": y_test,
        "feature_names": list(X.columns),
    }


if __name__ == "__main__":
    splits = run_preprocessing()
    for k, v in splits.items():
        if hasattr(v, "shape"):
            print(k, v.shape)
