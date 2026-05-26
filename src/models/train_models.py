from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))

from biobeat_paths import MODELS_DIR, TRAINING_TABLE_CSV, ensure_project_dirs, repo_path  # noqa: E402


TARGET_COLUMNS = ["preference_binary", "arousal_binary", "valence_binary"]
BIOMETRIC_COLUMNS = [
    "hr_mean",
    "hr_max",
    "hr_change_from_baseline",
    "hr_slope",
    "hr_recovery",
    "eda_mean",
    "eda_change_from_baseline",
    "eda_peak_count",
    "eda_max_amplitude",
    "eda_slope",
    "eda_recovery",
]
AUDIO_PREFIXES = (
    "duration_sec",
    "tempo",
    "rms_energy_",
    "zero_crossing_rate_",
    "spectral_centroid_",
    "spectral_bandwidth_",
    "spectral_rolloff_",
    "chroma_mean_",
    "mfcc_mean_",
    "mfcc_std_",
)


def feature_columns(table: pd.DataFrame) -> list[str]:
    audio_columns = [
        column
        for column in table.columns
        if any(column == prefix or column.startswith(prefix) for prefix in AUDIO_PREFIXES)
    ]
    biometric_columns = [column for column in BIOMETRIC_COLUMNS if column in table.columns]
    return audio_columns + biometric_columns


def split_data(X: pd.DataFrame, y: pd.Series, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    value_counts = y.value_counts()
    if len(value_counts) < 2:
        return X, X, y, y, "single_class_resubstitution"
    if len(y) < 12 or value_counts.min() < 2:
        return X, X, y, y, "small_sample_resubstitution"

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=seed,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test, "holdout"


def probability_from_positive_class(model: object, X: pd.DataFrame) -> np.ndarray:
    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 in classes:
        return probabilities[:, classes.index(1)]
    return np.zeros(len(X), dtype=float)


def evaluate_model(model: object, X_test: pd.DataFrame, y_test: pd.Series) -> str:
    predictions = model.predict(X_test)
    lines = [
        f"accuracy: {accuracy_score(y_test, predictions):.3f}",
        "confusion_matrix:",
        str(confusion_matrix(y_test, predictions, labels=[0, 1])),
        "classification_report:",
        classification_report(y_test, predictions, labels=[0, 1], zero_division=0),
    ]
    return "\n".join(lines)


def feature_importance_text(model: RandomForestClassifier, columns: list[str], top_n: int = 15) -> str:
    importances = pd.DataFrame(
        {
            "feature": columns,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    return importances.head(top_n).to_string(index=False)


def train_for_target(table: pd.DataFrame, target: str, seed: int) -> str:
    columns = feature_columns(table)
    if not columns:
        raise ValueError("No audio or biometric feature columns found in the training table.")

    frame = table[columns + [target]].copy()
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    frame = frame.dropna(subset=[target])
    X = frame[columns].apply(pd.to_numeric, errors="coerce")
    y = frame[target].astype(int)

    if y.empty:
        return f"{target}: skipped because no labeled rows were available."

    X_train, X_test, y_train, y_test, split_note = split_data(X, y, seed)
    positive_rate = float(y.mean())

    logistic = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    forest = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=seed,
                    class_weight="balanced",
                    min_samples_leaf=2,
                ),
            ),
        ]
    )

    if y.nunique() < 2:
        bundle = {
            "target": target,
            "feature_columns": columns,
            "model_type": "constant",
            "model": None,
            "constant_probability": positive_rate,
            "split_note": split_note,
        }
        joblib.dump(bundle, MODELS_DIR / f"{target}_random_forest.joblib")
        return (
            f"# {target}\n\n"
            f"Only one class was present. Saved a constant probability bundle: {positive_rate:.3f}.\n"
        )

    logistic.fit(X_train, y_train)
    forest.fit(X_train, y_train)

    log_text = evaluate_model(logistic, X_test, y_test)
    forest_text = evaluate_model(forest, X_test, y_test)
    forest_model = forest.named_steps["model"]
    importance_text = feature_importance_text(forest_model, columns)

    for model_name, model in [("logistic_regression", logistic), ("random_forest", forest)]:
        bundle = {
            "target": target,
            "feature_columns": columns,
            "model_type": model_name,
            "model": model,
            "constant_probability": None,
            "split_note": split_note,
        }
        joblib.dump(bundle, MODELS_DIR / f"{target}_{model_name}.joblib")

    return (
        f"# {target}\n\n"
        f"rows: {len(y)}\n"
        f"positive_rate: {positive_rate:.3f}\n"
        f"split: {split_note}\n\n"
        "## Logistic regression\n\n"
        f"{log_text}\n\n"
        "## Random forest\n\n"
        f"{forest_text}\n\n"
        "## Random forest feature importance\n\n"
        f"{importance_text}\n"
    )


def train_models(training_path: Path, *, seed: int = 42) -> dict[str, str]:
    ensure_project_dirs()
    table = pd.read_csv(training_path)
    metrics: dict[str, str] = {}
    for target in TARGET_COLUMNS:
        if target not in table.columns:
            metrics[target] = f"{target}: skipped because the column is missing."
            continue
        metrics[target] = train_for_target(table, target, seed)
        metric_path = MODELS_DIR / "metrics" / f"{target}_metrics.md"
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        metric_path.write_text(metrics[target], encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BioBeat baseline binary classifiers.")
    parser.add_argument("--training-table", default=str(TRAINING_TABLE_CSV))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics = train_models(repo_path(args.training_table), seed=args.seed)
    for target, text in metrics.items():
        first_line = text.splitlines()[0] if text else target
        print(first_line)
    print(f"Wrote model bundles and metrics to {MODELS_DIR}")


if __name__ == "__main__":
    main()
