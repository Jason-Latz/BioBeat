from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

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


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, rem = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {rem:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {rem:.1f}s"


def audio_feature_columns(table: pd.DataFrame) -> list[str]:
    return [
        column
        for column in table.columns
        if any(column == prefix or column.startswith(prefix) for prefix in AUDIO_PREFIXES)
    ]


def biometric_feature_columns(table: pd.DataFrame) -> list[str]:
    return [column for column in BIOMETRIC_COLUMNS if column in table.columns]


def clean_numeric_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    label: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Force every model input to float64 before sklearn sees it.

    This prevents errors like:
    TypeError: ufunc 'isnan' not supported for the input types
    """
    log(f"{label}: coercing {len(feature_columns)} candidate features to numeric float64")
    raw = frame[feature_columns]
    object_columns = [column for column in raw.columns if raw[column].dtype == "object"]
    if object_columns:
        preview = ", ".join(object_columns[:8])
        more = "..." if len(object_columns) > 8 else ""
        log(f"{label}: object-like feature columns before coercion: {preview}{more}")

    X = raw.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).astype(np.float64)

    observed_columns = [column for column in X.columns if X[column].notna().any()]
    X = X[observed_columns]

    total_values = X.shape[0] * X.shape[1]
    missing_values = int(X.isna().sum().sum()) if total_values else 0
    missing_pct = 100.0 * missing_values / total_values if total_values else 0.0

    log(f"{label}: kept {len(observed_columns)} observed feature columns")
    log(f"{label}: X shape after coercion = {X.shape[0]} rows x {X.shape[1]} columns")
    log(f"{label}: missing values = {missing_values}/{total_values} ({missing_pct:.1f}%)")
    log(f"{label}: dtype check = {sorted({str(dtype) for dtype in X.dtypes})}")
    return X, observed_columns


def split_regression(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, str]:
    if len(y) < 12:
        return X, X, y, y, "small_sample_resubstitution"
    return (*train_test_split(X, y, test_size=0.25, random_state=seed), "holdout")


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


def regression_cv(y_train: pd.Series, seed: int) -> KFold:
    n_splits = 3 if len(y_train) >= 18 else 2
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def classification_cv(y_train: pd.Series, seed: int) -> StratifiedKFold:
    min_class = int(y_train.value_counts().min())
    n_splits = min(5, min_class)
    if n_splits < 2:
        n_splits = 2
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def gradient_boosting_grid(preset: str) -> dict[str, list[Any]]:
    """Small-data-friendly grids.

    quick is intentionally tiny. It is meant to answer "can tuning beat the
    baseline?" without another 30 minute field trip.
    """
    if preset == "quick":
        return {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.03, 0.1],
            "model__max_depth": [1, 2],
            "model__min_samples_leaf": [2, 4],
            "model__subsample": [0.8, 1.0],
        }
    if preset == "medium":
        return {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.01, 0.03, 0.1],
            "model__max_depth": [1, 2, 3],
            "model__min_samples_leaf": [1, 2, 4],
            "model__subsample": [0.8, 1.0],
        }
    if preset == "full":
        return {
            "model__n_estimators": [50, 100, 200, 400],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_depth": [1, 2, 3],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__subsample": [0.6, 0.8, 1.0],
        }
    raise ValueError(f"Unknown preset {preset!r}; expected quick, medium, or full.")


def grid_size(param_grid: dict[str, list[Any]], cv_splits: int) -> tuple[int, int]:
    candidates = len(list(ParameterGrid(param_grid)))
    return candidates, candidates * cv_splits


def top_grid_results(search: GridSearchCV, score_name: str, n: int = 12) -> str:
    results = pd.DataFrame(search.cv_results_)
    rank_col = f"rank_test_{score_name}"
    columns = [
        rank_col,
        f"mean_test_{score_name}",
        f"std_test_{score_name}",
        f"mean_train_{score_name}",
        f"std_train_{score_name}",
    ]
    param_columns = [column for column in results.columns if column.startswith("param_")]
    display_columns = [column for column in columns if column in results.columns] + param_columns
    return results.sort_values(rank_col).head(n)[display_columns].to_string(index=False)


def make_regressor(seed: int, **model_kwargs: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(random_state=seed, **model_kwargs)),
        ]
    )


def make_classifier(seed: int, **model_kwargs: Any) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(random_state=seed, **model_kwargs)),
        ]
    )


def tune_arousal_gradient_boosting(
    table: pd.DataFrame,
    *,
    seed: int,
    n_jobs: int,
    search_verbose: int,
    preset: str,
) -> str:
    label = "arousal/GradientBoostingRegressor"
    target = "arousal_score"
    model_prefix = "arousal_score_audio_gradient_boosting_selected"
    tuned_prefix = "arousal_score_audio_gradient_boosting_tuned_candidate"
    feature_columns = audio_feature_columns(table)

    log("=" * 88)
    log(f"Starting {label}")
    log(f"{label}: target={target}")
    log(f"{label}: detected {len(feature_columns)} audio feature columns")
    if not feature_columns:
        return f"# {model_prefix}\n\nSkipped because no audio feature columns were available.\n"

    frame = table[feature_columns + [target]].copy()
    frame[target] = pd.to_numeric(frame[target], errors="coerce")
    before_drop = len(frame)
    frame = frame.dropna(subset=[target])
    log(f"{label}: labeled rows = {len(frame)} / {before_drop}")

    X, observed_columns = clean_numeric_features(frame, feature_columns, label=label)
    if not observed_columns:
        return f"# {model_prefix}\n\nSkipped because no observed feature values were available.\n"

    y = frame[target].astype(float)
    has_features = X.notna().any(axis=1)
    X = X.loc[has_features]
    y = y.loc[has_features]
    log(f"{label}: final usable rows={len(y)}, features={len(observed_columns)}")
    log(
        f"{label}: target mean={float(y.mean()):.3f}, std={float(y.std(ddof=0)):.3f}, "
        f"min={float(y.min()):.3f}, max={float(y.max()):.3f}"
    )
    if y.empty:
        return f"# {model_prefix}\n\nSkipped because no labeled rows had observed feature values.\n"

    X_train, X_test, y_train, y_test, split_note = split_regression(X, y, seed)
    log(f"{label}: split={split_note}; train={len(y_train)}, test={len(y_test)}")

    # Baseline = sklearn defaults, matching the plain model run as closely as possible.
    baseline = make_regressor(seed)
    start = time.perf_counter()
    log(f"{label}: fitting baseline default GradientBoostingRegressor")
    baseline.fit(X_train, y_train)
    baseline_elapsed = time.perf_counter() - start
    baseline_predictions = np.clip(baseline.predict(X_test), 0.0, 1.0)
    baseline_mae = mean_absolute_error(y_test, baseline_predictions)
    baseline_r2 = r2_score(y_test, baseline_predictions) if len(y_test) > 1 else float("nan")
    log(f"{label}: baseline holdout MAE={baseline_mae:.4f}, R2={baseline_r2:.4f} ({format_seconds(baseline_elapsed)})")

    pipeline = make_regressor(seed)
    param_grid = gradient_boosting_grid(preset)
    cv = regression_cv(y_train, seed)
    cv_splits = cv.get_n_splits(X_train, y_train)
    candidates, fits = grid_size(param_grid, cv_splits)
    log(f"{label}: grid preset={preset}; candidates={candidates}; CV fits={fits}; n_jobs={n_jobs}")

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring={"mae": "neg_mean_absolute_error", "r2": "r2"},
        refit="mae",
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=True,
        error_score="raise",
        verbose=search_verbose,
        pre_dispatch="2*n_jobs",
    )
    start = time.perf_counter()
    log(f"{label}: starting GridSearchCV.fit()")
    search.fit(X_train, y_train)
    search_elapsed = time.perf_counter() - start
    log(f"{label}: finished GridSearchCV.fit() in {format_seconds(search_elapsed)}")
    log(f"{label}: tuned best CV MAE={-float(search.best_score_):.4f}")
    log(f"{label}: tuned best params={search.best_params_}")

    tuned = search.best_estimator_
    tuned_predictions = np.clip(tuned.predict(X_test), 0.0, 1.0)
    tuned_mae = mean_absolute_error(y_test, tuned_predictions)
    tuned_r2 = r2_score(y_test, tuned_predictions) if len(y_test) > 1 else float("nan")
    log(f"{label}: tuned holdout MAE={tuned_mae:.4f}, R2={tuned_r2:.4f}")

    selected_name = "tuned" if tuned_mae < baseline_mae else "baseline"
    selected_model = tuned if selected_name == "tuned" else baseline
    selected_mae = tuned_mae if selected_name == "tuned" else baseline_mae
    selected_r2 = tuned_r2 if selected_name == "tuned" else baseline_r2
    log(f"{label}: SELECTED {selected_name.upper()} model based on lower holdout MAE")

    selected_path = MODELS_DIR / f"{model_prefix}.joblib"
    tuned_path = MODELS_DIR / f"{tuned_prefix}.joblib"
    common_bundle = {
        "target": target,
        "feature_columns": observed_columns,
        "task": "regression",
        "clip_predictions": True,
        "prediction_min": 0.0,
        "prediction_max": 1.0,
        "split_note": split_note,
        "selection_metric": "holdout_mae",
    }
    joblib.dump(
        {
            **common_bundle,
            "model_type": f"gradient_boosting_regressor_{selected_name}_selected",
            "model": selected_model,
            "selected_model": selected_name,
            "holdout_mae": float(selected_mae),
            "holdout_r2": float(selected_r2),
            "baseline_holdout_mae": float(baseline_mae),
            "tuned_holdout_mae": float(tuned_mae),
            "best_params": search.best_params_ if selected_name == "tuned" else None,
        },
        selected_path,
    )
    joblib.dump(
        {
            **common_bundle,
            "model_type": "gradient_boosting_regressor_tuned_candidate",
            "model": tuned,
            "best_params": search.best_params_,
            "best_cv_neg_mae": float(search.best_score_),
            "holdout_mae": float(tuned_mae),
            "holdout_r2": float(tuned_r2),
        },
        tuned_path,
    )
    log(f"{label}: wrote selected model bundle to {selected_path}")
    log(f"{label}: wrote tuned candidate bundle to {tuned_path}")

    return (
        f"# {model_prefix}\n\n"
        f"rows: {len(y)}\n"
        f"features: {len(observed_columns)}\n"
        f"split: {split_note}\n"
        f"cv: {cv}\n"
        f"grid_preset: {preset}\n"
        f"grid_candidates: {len(search.cv_results_['params'])}\n"
        f"search_fit_time: {format_seconds(search_elapsed)}\n\n"
        "## Selected model\n\n"
        f"selected_model: {selected_name}\n"
        f"selection_rule: lower holdout MAE\n"
        f"selected_holdout_mae: {selected_mae:.3f}\n"
        f"selected_holdout_r2: {selected_r2:.3f}\n\n"
        "## Baseline default GradientBoostingRegressor\n\n"
        f"mae: {baseline_mae:.3f}\n"
        f"r2: {baseline_r2:.3f}\n\n"
        "## Tuned GradientBoostingRegressor candidate\n\n"
        f"best_params:\n```python\n{search.best_params_}\n```\n"
        f"best_cv_mae: {-float(search.best_score_):.3f}\n"
        f"holdout_mae: {tuned_mae:.3f}\n"
        f"holdout_r2: {tuned_r2:.3f}\n\n"
        "## Top grid results\n\n"
        f"```text\n{top_grid_results(search, 'mae')}\n```\n"
    )


def tune_valence_gradient_boosting(
    table: pd.DataFrame,
    *,
    seed: int,
    n_jobs: int,
    search_verbose: int,
    preset: str,
) -> str:
    label = "valence/GradientBoostingClassifier"
    target = "valence_label"
    model_prefix = "valence_label_audio_gradient_boosting_selected"
    tuned_prefix = "valence_label_audio_gradient_boosting_tuned_candidate"
    feature_columns = audio_feature_columns(table)

    log("=" * 88)
    log(f"Starting {label}")
    log(f"{label}: target={target}")
    log(f"{label}: detected {len(feature_columns)} audio feature columns")
    if not feature_columns:
        return f"# {model_prefix}\n\nSkipped because no audio feature columns were available.\n"

    frame = table[feature_columns + [target]].copy()
    before_drop = len(frame)
    frame = frame.dropna(subset=[target])
    log(f"{label}: labeled rows = {len(frame)} / {before_drop}")

    X, observed_columns = clean_numeric_features(frame, feature_columns, label=label)
    if not observed_columns:
        return f"# {model_prefix}\n\nSkipped because no observed feature values were available.\n"

    y = frame[target].astype(str)
    invalid_target_mask = y.isin(["nan", "None", ""])
    if invalid_target_mask.any():
        log(f"{label}: dropping {int(invalid_target_mask.sum())} invalid labels after string conversion")
    y = y.mask(invalid_target_mask).dropna()
    X = X.loc[y.index]

    has_features = X.notna().any(axis=1)
    X = X.loc[has_features]
    y = y.loc[has_features]
    log(f"{label}: final usable rows={len(y)}, features={len(observed_columns)}")
    log(f"{label}: class counts={y.value_counts().to_dict()}")
    if y.empty:
        return f"# {model_prefix}\n\nSkipped because no valence-labeled rows had observed feature values.\n"
    if y.nunique() < 2:
        return f"# {model_prefix}\n\nSkipped because only one class was present: {y.iloc[0]}.\n"

    X_train, X_test, y_train, y_test, split_note = split_classification(X, y, seed)
    log(f"{label}: split={split_note}; train={len(y_train)}, test={len(y_test)}")
    log(f"{label}: train class counts={y_train.value_counts().to_dict()}")
    log(f"{label}: test class counts={y_test.value_counts().to_dict()}")

    baseline = make_classifier(seed)
    start = time.perf_counter()
    log(f"{label}: fitting baseline default GradientBoostingClassifier")
    baseline.fit(X_train, y_train)
    baseline_elapsed = time.perf_counter() - start
    baseline_predictions = baseline.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_predictions)
    baseline_balanced_accuracy = balanced_accuracy_score(y_test, baseline_predictions)
    baseline_macro_f1 = f1_score(y_test, baseline_predictions, average="macro")
    log(
        f"{label}: baseline holdout accuracy={baseline_accuracy:.4f}, "
        f"balanced_accuracy={baseline_balanced_accuracy:.4f}, macro_f1={baseline_macro_f1:.4f} "
        f"({format_seconds(baseline_elapsed)})"
    )

    pipeline = make_classifier(seed)
    param_grid = gradient_boosting_grid(preset)
    cv = classification_cv(y_train, seed)
    cv_splits = cv.get_n_splits(X_train, y_train)
    candidates, fits = grid_size(param_grid, cv_splits)
    log(f"{label}: grid preset={preset}; candidates={candidates}; CV fits={fits}; n_jobs={n_jobs}")

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring={
            "macro_f1": "f1_macro",
            "accuracy": "accuracy",
            "balanced_accuracy": "balanced_accuracy",
        },
        refit="macro_f1",
        cv=cv,
        n_jobs=n_jobs,
        return_train_score=True,
        error_score="raise",
        verbose=search_verbose,
        pre_dispatch="2*n_jobs",
    )
    start = time.perf_counter()
    log(f"{label}: starting GridSearchCV.fit()")
    search.fit(X_train, y_train)
    search_elapsed = time.perf_counter() - start
    log(f"{label}: finished GridSearchCV.fit() in {format_seconds(search_elapsed)}")
    log(f"{label}: tuned best CV macro F1={float(search.best_score_):.4f}")
    log(f"{label}: tuned best params={search.best_params_}")

    tuned = search.best_estimator_
    tuned_predictions = tuned.predict(X_test)
    tuned_accuracy = accuracy_score(y_test, tuned_predictions)
    tuned_balanced_accuracy = balanced_accuracy_score(y_test, tuned_predictions)
    tuned_macro_f1 = f1_score(y_test, tuned_predictions, average="macro")
    log(
        f"{label}: tuned holdout accuracy={tuned_accuracy:.4f}, "
        f"balanced_accuracy={tuned_balanced_accuracy:.4f}, macro_f1={tuned_macro_f1:.4f}"
    )

    # Use macro F1 because the class counts are not perfectly balanced and we care
    # about all three valence classes, not just the majority neutral class.
    selected_name = "tuned" if tuned_macro_f1 > baseline_macro_f1 else "baseline"
    selected_model = tuned if selected_name == "tuned" else baseline
    selected_predictions = tuned_predictions if selected_name == "tuned" else baseline_predictions
    selected_accuracy = tuned_accuracy if selected_name == "tuned" else baseline_accuracy
    selected_balanced_accuracy = tuned_balanced_accuracy if selected_name == "tuned" else baseline_balanced_accuracy
    selected_macro_f1 = tuned_macro_f1 if selected_name == "tuned" else baseline_macro_f1
    log(f"{label}: SELECTED {selected_name.upper()} model based on higher holdout macro F1")

    selected_path = MODELS_DIR / f"{model_prefix}.joblib"
    tuned_path = MODELS_DIR / f"{tuned_prefix}.joblib"
    common_bundle = {
        "target": target,
        "feature_columns": observed_columns,
        "task": "classification",
        "classes": VALENCE_LABELS,
        "constant_class": None,
        "split_note": split_note,
        "selection_metric": "holdout_macro_f1",
    }
    joblib.dump(
        {
            **common_bundle,
            "model_type": f"gradient_boosting_classifier_{selected_name}_selected",
            "model": selected_model,
            "selected_model": selected_name,
            "holdout_accuracy": float(selected_accuracy),
            "holdout_balanced_accuracy": float(selected_balanced_accuracy),
            "holdout_macro_f1": float(selected_macro_f1),
            "baseline_holdout_macro_f1": float(baseline_macro_f1),
            "tuned_holdout_macro_f1": float(tuned_macro_f1),
            "best_params": search.best_params_ if selected_name == "tuned" else None,
        },
        selected_path,
    )
    joblib.dump(
        {
            **common_bundle,
            "model_type": "gradient_boosting_classifier_tuned_candidate",
            "model": tuned,
            "best_params": search.best_params_,
            "best_cv_macro_f1": float(search.best_score_),
            "holdout_accuracy": float(tuned_accuracy),
            "holdout_balanced_accuracy": float(tuned_balanced_accuracy),
            "holdout_macro_f1": float(tuned_macro_f1),
        },
        tuned_path,
    )
    log(f"{label}: wrote selected model bundle to {selected_path}")
    log(f"{label}: wrote tuned candidate bundle to {tuned_path}")

    labels = [label_name for label_name in VALENCE_LABELS if label_name in sorted(y.unique())]
    return (
        f"# {model_prefix}\n\n"
        f"rows: {len(y)}\n"
        f"features: {len(observed_columns)}\n"
        f"class_counts: {y.value_counts().to_dict()}\n"
        f"split: {split_note}\n"
        f"cv: {cv}\n"
        f"grid_preset: {preset}\n"
        f"grid_candidates: {len(search.cv_results_['params'])}\n"
        f"search_fit_time: {format_seconds(search_elapsed)}\n\n"
        "## Selected model\n\n"
        f"selected_model: {selected_name}\n"
        f"selection_rule: higher holdout macro F1\n"
        f"selected_accuracy: {selected_accuracy:.3f}\n"
        f"selected_balanced_accuracy: {selected_balanced_accuracy:.3f}\n"
        f"selected_macro_f1: {selected_macro_f1:.3f}\n"
        "confusion_matrix:\n"
        f"{confusion_matrix(y_test, selected_predictions, labels=labels)}\n"
        "classification_report:\n"
        f"{classification_report(y_test, selected_predictions, labels=labels, zero_division=0)}\n\n"
        "## Baseline default GradientBoostingClassifier\n\n"
        f"accuracy: {baseline_accuracy:.3f}\n"
        f"balanced_accuracy: {baseline_balanced_accuracy:.3f}\n"
        f"macro_f1: {baseline_macro_f1:.3f}\n\n"
        "## Tuned GradientBoostingClassifier candidate\n\n"
        f"best_params:\n```python\n{search.best_params_}\n```\n"
        f"best_cv_macro_f1: {float(search.best_score_):.3f}\n"
        f"holdout_accuracy: {tuned_accuracy:.3f}\n"
        f"holdout_balanced_accuracy: {tuned_balanced_accuracy:.3f}\n"
        f"holdout_macro_f1: {tuned_macro_f1:.3f}\n\n"
        "## Top grid results\n\n"
        f"```text\n{top_grid_results(search, 'macro_f1')}\n```\n"
    )


def tune_models(
    training_path: Path,
    *,
    seed: int,
    n_jobs: int,
    search_verbose: int,
    preset: str,
) -> dict[str, str]:
    ensure_project_dirs()
    log("Starting BioBeat safe fast hyperparameter search")
    log(f"training table path: {training_path}")
    log(f"models dir: {MODELS_DIR}")
    log(f"seed: {seed}")
    log(f"n_jobs: {n_jobs}")
    log(f"GridSearchCV verbose: {search_verbose}")
    log(f"grid preset: {preset}")

    load_start = time.perf_counter()
    raw_table = pd.read_csv(training_path)
    log(f"loaded raw training table: {raw_table.shape[0]} rows x {raw_table.shape[1]} columns")
    table = add_model_targets(raw_table)
    log(f"after add_model_targets: {table.shape[0]} rows x {table.shape[1]} columns")
    log(f"audio feature columns detected: {len(audio_feature_columns(table))}")
    log(f"biometric feature columns detected: {len(biometric_feature_columns(table))}")
    log(f"data loading + target creation took {format_seconds(time.perf_counter() - load_start)}")

    total_start = time.perf_counter()
    reports = {
        "arousal_score_audio_gradient_boosting_selected": tune_arousal_gradient_boosting(
            table,
            seed=seed,
            n_jobs=n_jobs,
            search_verbose=search_verbose,
            preset=preset,
        ),
        "valence_label_audio_gradient_boosting_selected": tune_valence_gradient_boosting(
            table,
            seed=seed,
            n_jobs=n_jobs,
            search_verbose=search_verbose,
            preset=preset,
        ),
    }

    log("=" * 88)
    log("Writing metrics reports")
    for target, text in reports.items():
        metric_path = MODELS_DIR / "metrics" / f"{target}_metrics.md"
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        metric_path.write_text(text, encoding="utf-8")
        log(f"wrote metrics report: {metric_path}")

    log(f"all tuning finished in {format_seconds(time.perf_counter() - total_start)}")
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fast, safe BioBeat hyperparameter search. Tunes the current best "
            "GradientBoosting models, compares baseline vs tuned on holdout, and "
            "saves whichever actually wins."
        )
    )
    parser.add_argument("--training-table", default=str(TRAINING_TABLE_CSV))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for GridSearchCV. Use -1 for all cores, 1 for easiest debugging.",
    )
    parser.add_argument(
        "--search-verbose",
        type=int,
        default=1,
        help="GridSearchCV verbosity. Use 0 for quieter output or 2 for per-fit progress.",
    )
    parser.add_argument(
        "--preset",
        choices=["quick", "medium", "full"],
        default="quick",
        help="Grid size. quick is much faster; medium is broader; full is the old large search.",
    )
    args = parser.parse_args()

    reports = tune_models(
        repo_path(args.training_table),
        seed=args.seed,
        n_jobs=args.n_jobs,
        search_verbose=args.search_verbose,
        preset=args.preset,
    )
    log("Summary of generated reports:")
    for target, text in reports.items():
        first_line = text.splitlines()[0] if text else target
        log(f"{target}: {first_line}")
    log(f"Wrote selected model bundles and metrics to {MODELS_DIR}")


if __name__ == "__main__":
    main()
