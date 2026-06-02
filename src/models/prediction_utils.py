from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features.target_features import VALENCE_LABELS


def load_bundle(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run src/models/train_models.py first.")
    return joblib.load(path)


def model_frame(bundle: dict[str, object], rows: pd.DataFrame) -> pd.DataFrame:
    columns = list(bundle["feature_columns"])
    return rows.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")


def predict_regression(bundle: dict[str, object], rows: pd.DataFrame) -> np.ndarray:
    X = model_frame(bundle, rows)
    model = bundle.get("model")
    if model is None:
        return np.zeros(len(X), dtype=float)
    predictions = np.asarray(model.predict(X), dtype=float)
    if bundle.get("clip_predictions"):
        prediction_min = float(bundle.get("prediction_min", -1.0))
        prediction_max = float(bundle.get("prediction_max", 1.0))
        predictions = np.clip(predictions, prediction_min, prediction_max)
    return predictions


def predict_class_probabilities(bundle: dict[str, object], rows: pd.DataFrame) -> pd.DataFrame:
    X = model_frame(bundle, rows)
    constant_class = bundle.get("constant_class")
    model = bundle.get("model")
    probabilities = pd.DataFrame(0.0, index=rows.index, columns=VALENCE_LABELS)

    if model is None or constant_class:
        label = str(constant_class or "neutral")
        if label not in probabilities.columns:
            label = "neutral"
        probabilities[label] = 1.0
        return probabilities

    raw = model.predict_proba(X)
    for index, class_label in enumerate(model.classes_):
        class_text = str(class_label)
        if class_text in probabilities.columns:
            probabilities[class_text] = raw[:, index]
    row_sums = probabilities.sum(axis=1).replace(0, 1)
    return probabilities.div(row_sums, axis=0)
