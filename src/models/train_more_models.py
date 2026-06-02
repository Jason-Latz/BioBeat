from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))

from biobeat_paths import MODELS_DIR, TRAINING_TABLE_CSV, ensure_project_dirs, repo_path  # noqa: E402
from features.target_features import BIOMETRIC_COLUMNS, VALENCE_LABELS, add_model_targets  # noqa: E402


AUDIO_PREFIXES = (
    "duration_sec",
    "tempo",
    "beat_",
    "rhythm_",
    "onset_rate",
    "rms_energy_",
    "loudness_",
    "quiet_fraction",
    "onset_strength_",
    "zero_crossing_rate_",
    "spectral_centroid_",
    "spectral_bandwidth_",
    "spectral_rolloff_",
    "spectral_contrast_",
    "mode_",
    "key_pitch_class",
    "major_key_correlation",
    "minor_key_correlation",
    "chroma_mean_",
    "mfcc_mean_",
    "mfcc_std_",
)


def audio_feature_columns(table: pd.DataFrame) -> list[str]:
    return [
        column
        for column in table.columns
        if any(column == prefix or column.startswith(prefix) for prefix in AUDIO_PREFIXES)
    ]


def biometric_feature_columns(table: pd.DataFrame) -> list[str]:
    return [column for column in BIOMETRIC_COLUMNS if column in table.columns]


def split_classification(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    value_counts = y.value_counts()
    if len(value_counts) < 2:
        return X, X, y, y, "single_class_resubstitution"
    if len(y) < 15 or value_counts.min() < 2:
        return X, X, y, y, "small_sample_resubstitution"
    return (*train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y), "holdout")


def split_regression(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    if len(y) < 12:
        return X, X, y, y, "small_sample_resubstitution"
    return (*train_test_split(X, y, test_size=0.25, random_state=seed), "holdout")


def regression_report(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    prediction_min: float,
    prediction_max: float,
) -> str:
    predictions = np.clip(model.predict(X_test), prediction_min, prediction_max)
    lines = [
        f"mae: {mean_absolute_error(y_test, predictions):.3f}",
        f"r2: {r2_score(y_test, predictions):.3f}" if len(y_test) > 1 else "r2: nan",
    ]
    return "\n".join(lines)


def classification_report_text(model: object, X_test: pd.DataFrame, y_test: pd.Series) -> str:
    predictions = model.predict(X_test)
    lines = [
        f"accuracy: {accuracy_score(y_test, predictions):.3f}",
        "confusion_matrix:",
        str(confusion_matrix(y_test, predictions, labels=VALENCE_LABELS)),
        "classification_report:",
        classification_report(y_test, predictions, labels=VALENCE_LABELS, zero_division=0),
    ]
    return "\n".join(lines)


def forest_importance_text(model: object, columns: list[str], top_n: int = 15) -> str:
    if not hasattr(model, "feature_importances_"):
        return "No feature importance available."
    importances = pd.DataFrame(
        {"feature": columns, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    return importances.head(top_n).to_string(index=False)


def fit_regression_models(
    table: pd.DataFrame,
    *,
    target: str,
    feature_columns: list[str],
    seed: int,
    model_prefix: str,
    clip_predictions: bool,
    prediction_min: float,
    prediction_max: float,
) -> str:
    frame = table[feature_columns + [target]].copy()
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    frame = frame.dropna(subset=[target])
    X = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    observed_columns = [column for column in feature_columns if X[column].notna().any()]
    if not observed_columns:
        return f"# {model_prefix}\n\nSkipped because no observed feature values were available for labeled rows.\n"
    X = X[observed_columns]
    y = frame[target].astype(float)
    has_features = X.notna().any(axis=1)
    X = X[has_features]
    y = y[has_features]
    if y.empty:
        return f"# {model_prefix}\n\nSkipped because no labeled rows had observed feature values.\n"

    X_train, X_test, y_train, y_test, split_note = split_regression(X, y, seed)
    n_neighbors = max(1, min(7, len(X_train)))

    regression_models: list[tuple[str, Pipeline]] = [
        (
            "ridge_regression",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=1.0)),
                ]
            ),
        ),
        (
            "random_forest_regressor",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=400,
                            random_state=seed,
                            min_samples_leaf=2,
                        ),
                    ),
                ]
            ),
        ),
        (
            "extra_trees_regressor",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        ExtraTreesRegressor(
                            n_estimators=500,
                            random_state=seed,
                            min_samples_leaf=2,
                            max_features="sqrt",
                        ),
                    ),
                ]
            ),
        ),
        (
            "gradient_boosting_regressor",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        GradientBoostingRegressor(
                            n_estimators=150,
                            learning_rate=0.04,
                            max_depth=2,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        ),
        (
            "svr_rbf",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", SVR(kernel="rbf", C=1.0, epsilon=0.08, gamma="scale")),
                ]
            ),
        ),
        (
            "knn_regressor",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsRegressor(n_neighbors=n_neighbors, weights="distance")),
                ]
            ),
        ),
        (
            "mlp_regressor",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        MLPRegressor(
                            hidden_layer_sizes=(16,),
                            activation="relu",
                            alpha=0.01,
                            learning_rate_init=0.001,
                            max_iter=2000,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        ),
    ]

    report_sections: list[str] = []
    fitted_models: dict[str, Pipeline] = {}
    for model_name, model in regression_models:
        model.fit(X_train, y_train)
        fitted_models[model_name] = model
        bundle = {
            "target": target,
            "feature_columns": observed_columns,
            "model_type": model_name,
            "task": "regression",
            "model": model,
            "clip_predictions": clip_predictions,
            "prediction_min": prediction_min,
            "prediction_max": prediction_max,
            "split_note": split_note,
        }
        joblib.dump(bundle, MODELS_DIR / f"{model_prefix}_{model_name}.joblib")
        pretty_name = model_name.replace("_", " ").title()
        report_sections.append(
            f"## {pretty_name}\n\n"
            f"{regression_report(model, X_test, y_test, prediction_min=prediction_min, prediction_max=prediction_max)}\n"
        )

    forest_model = fitted_models["random_forest_regressor"].named_steps["model"]
    extra_trees_model = fitted_models["extra_trees_regressor"].named_steps["model"]
    return (
        f"# {model_prefix}\n\n"
        f"rows: {len(y)}\n"
        f"target_mean: {float(y.mean()):.3f}\n"
        f"split: {split_note}\n\n"
        + "\n".join(report_sections)
        + "\n## Random forest feature importance\n\n"
        f"{forest_importance_text(forest_model, observed_columns)}\n\n"
        "## Extra trees feature importance\n\n"
        f"{forest_importance_text(extra_trees_model, observed_columns)}\n"
    )


def fit_valence_classifier(
    table: pd.DataFrame,
    *,
    feature_columns: list[str],
    seed: int,
    model_prefix: str,
) -> str:
    frame = table[feature_columns + ["valence_label"]].copy()
    frame = frame.dropna(subset=["valence_label"])
    X = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    observed_columns = [column for column in feature_columns if X[column].notna().any()]
    if not observed_columns:
        return f"# {model_prefix}\n\nSkipped because no observed feature values were available for labeled rows.\n"
    X = X[observed_columns]
    y = frame["valence_label"].astype(str)
    has_features = X.notna().any(axis=1)
    X = X[has_features]
    y = y[has_features]
    if y.empty:
        return f"# {model_prefix}\n\nSkipped because no valence-labeled rows had observed feature values.\n"

    X_train, X_test, y_train, y_test, split_note = split_classification(X, y, seed)
    class_counts = y.value_counts().to_dict()

    if y.nunique() < 2:
        bundle = {
            "target": "valence_label",
            "feature_columns": observed_columns,
            "model_type": "constant",
            "task": "classification",
            "model": None,
            "classes": VALENCE_LABELS,
            "constant_class": str(y.iloc[0]),
            "split_note": split_note,
        }
        joblib.dump(bundle, MODELS_DIR / f"{model_prefix}_random_forest_classifier.joblib")
        return f"# {model_prefix}\n\nOnly one class was present: {y.iloc[0]}.\n"

    n_neighbors = max(1, min(7, len(X_train)))

    classification_models: list[tuple[str, Pipeline]] = [
        (
            "logistic_regression",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(max_iter=1000, class_weight="balanced"),
                    ),
                ]
            ),
        ),
        (
            "random_forest_classifier",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=400,
                            random_state=seed,
                            class_weight="balanced",
                            min_samples_leaf=2,
                        ),
                    ),
                ]
            ),
        ),
        (
            "extra_trees_classifier",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        ExtraTreesClassifier(
                            n_estimators=500,
                            random_state=seed,
                            class_weight="balanced",
                            min_samples_leaf=2,
                            max_features="sqrt",
                        ),
                    ),
                ]
            ),
        ),
        (
            "gradient_boosting_classifier",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "model",
                        GradientBoostingClassifier(
                            n_estimators=120,
                            learning_rate=0.04,
                            max_depth=2,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        ),
        (
            "svc_rbf",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")),
                ]
            ),
        ),
        (
            "knn_classifier",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")),
                ]
            ),
        ),
        (
            "mlp_classifier",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        MLPClassifier(
                            hidden_layer_sizes=(16,),
                            activation="relu",
                            alpha=0.01,
                            learning_rate_init=0.001,
                            max_iter=2000,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
        ),
    ]

    report_sections: list[str] = []
    fitted_models: dict[str, Pipeline] = {}
    for model_name, model in classification_models:
        model.fit(X_train, y_train)
        fitted_models[model_name] = model
        bundle = {
            "target": "valence_label",
            "feature_columns": observed_columns,
            "model_type": model_name,
            "task": "classification",
            "model": model,
            "classes": VALENCE_LABELS,
            "constant_class": None,
            "split_note": split_note,
        }
        joblib.dump(bundle, MODELS_DIR / f"{model_prefix}_{model_name}.joblib")
        pretty_name = model_name.replace("_", " ").title()
        report_sections.append(
            f"## {pretty_name}\n\n"
            f"{classification_report_text(model, X_test, y_test)}\n"
        )

    forest_model = fitted_models["random_forest_classifier"].named_steps["model"]
    extra_trees_model = fitted_models["extra_trees_classifier"].named_steps["model"]
    return (
        f"# {model_prefix}\n\n"
        f"rows: {len(y)}\n"
        f"class_counts: {class_counts}\n"
        f"split: {split_note}\n\n"
        + "\n".join(report_sections)
        + "\n## Random forest feature importance\n\n"
        f"{forest_importance_text(forest_model, observed_columns)}\n\n"
        "## Extra trees feature importance\n\n"
        f"{forest_importance_text(extra_trees_model, observed_columns)}\n"
    )


def train_models(training_path: Path, *, seed: int = 42) -> dict[str, str]:
    ensure_project_dirs()
    table = add_model_targets(pd.read_csv(training_path))
    audio_columns = audio_feature_columns(table)
    biometric_columns = biometric_feature_columns(table)
    if not audio_columns:
        raise ValueError("No audio feature columns found in the training table.")

    metrics: dict[str, str] = {}
    metrics["arousal_score_audio"] = fit_regression_models(
        table,
        target="arousal_score",
        feature_columns=audio_columns,
        seed=seed,
        model_prefix="arousal_score_audio",
        clip_predictions=True,
        prediction_min=0.0,
        prediction_max=1.0,
    )
    metrics["valence_label_audio"] = fit_valence_classifier(
        table,
        feature_columns=audio_columns,
        seed=seed,
        model_prefix="valence_label_audio",
    )
    metrics["valence_ordinal_audio"] = fit_regression_models(
        table,
        target="valence_ordinal",
        feature_columns=audio_columns,
        seed=seed,
        model_prefix="valence_ordinal_audio",
        clip_predictions=True,
        prediction_min=-1.0,
        prediction_max=1.0,
    )

    if biometric_columns:
        audio_bio_columns = audio_columns + biometric_columns
        metrics["valence_label_audio_bio_experiment"] = fit_valence_classifier(
            table,
            feature_columns=audio_bio_columns,
            seed=seed,
            model_prefix="valence_label_audio_bio_experiment",
        )
        metrics["valence_ordinal_audio_bio_experiment"] = fit_regression_models(
            table,
            target="valence_ordinal",
            feature_columns=audio_bio_columns,
            seed=seed,
            model_prefix="valence_ordinal_audio_bio_experiment",
            clip_predictions=True,
            prediction_min=-1.0,
            prediction_max=1.0,
        )

    for target, text in metrics.items():
        metric_path = MODELS_DIR / "metrics" / f"{target}_metrics.md"
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        metric_path.write_text(text, encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BioBeat arousal and valence models.")
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
