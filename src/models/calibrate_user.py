from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from biobeat_paths import MODELS_DIR, PROCESSED_DIR, TRAINING_TABLE_CSV, ensure_project_dirs, repo_path  # noqa: E402
from features.target_features import BIOMETRIC_COLUMNS, VALENCE_LABELS, add_model_targets  # noqa: E402
from models.prediction_utils import load_bundle, predict_class_probabilities, predict_regression  # noqa: E402


DEFAULT_OUTPUT = PROCESSED_DIR / "user_calibration_profiles.csv"
AROUSAL_MODEL = MODELS_DIR / "arousal_score_audio_random_forest_regressor.joblib"
VALENCE_MODEL = MODELS_DIR / "valence_label_audio_random_forest_classifier.joblib"


def calibration_rows(
    table: pd.DataFrame,
    *,
    user_id: str,
    session_id: str | None,
    max_clips: int,
) -> pd.DataFrame:
    rows = table[table["user_id"].astype(str) == str(user_id)].copy()
    if session_id:
        rows = rows[rows["session_id"].astype(str) == str(session_id)]
    if "trial_index" in rows.columns:
        rows["trial_index"] = pd.to_numeric(rows["trial_index"], errors="coerce")
        rows = rows.sort_values(["session_id", "trial_index"])
    return rows.head(max_clips)


def valence_prior_shift(observed: pd.Series, predicted: pd.DataFrame) -> dict[str, float]:
    observed_counts = observed.value_counts()
    observed_prior = {
        label: (float(observed_counts.get(label, 0)) + 1.0) / (float(len(observed)) + len(VALENCE_LABELS))
        for label in VALENCE_LABELS
    }
    predicted_prior = {
        label: float(predicted[label].mean()) if label in predicted.columns and not predicted.empty else 1.0 / 3.0
        for label in VALENCE_LABELS
    }
    return {
        label: observed_prior[label] / max(predicted_prior[label], 1e-6)
        for label in VALENCE_LABELS
    }


def build_profile(
    table: pd.DataFrame,
    *,
    user_id: str,
    session_id: str | None,
    max_clips: int,
) -> pd.DataFrame:
    rows = calibration_rows(table, user_id=user_id, session_id=session_id, max_clips=max_clips)
    rows = rows.dropna(subset=["arousal_score", "valence_label"])
    if rows.empty:
        raise ValueError("No calibration rows with arousal_score and valence_label were found.")

    arousal_bundle = load_bundle(AROUSAL_MODEL)
    valence_bundle = load_bundle(VALENCE_MODEL)
    predicted_arousal = np.clip(predict_regression(arousal_bundle, rows), 0.0, 1.0)
    observed_arousal = pd.to_numeric(rows["arousal_score"], errors="coerce").to_numpy(dtype=float)
    arousal_offset = float(np.nanmean(observed_arousal - predicted_arousal))

    predicted_valence = predict_class_probabilities(valence_bundle, rows)
    shifts = valence_prior_shift(rows["valence_label"].astype(str), predicted_valence)

    profile: dict[str, object] = {
        "user_id": user_id,
        "session_id": session_id or "",
        "calibration_clip_count": len(rows),
        "arousal_offset": round(arousal_offset, 5),
    }
    for label in VALENCE_LABELS:
        profile[f"valence_shift_{label}"] = round(float(shifts[label]), 5)

    for column in BIOMETRIC_COLUMNS:
        if column in rows.columns:
            numeric = pd.to_numeric(rows[column], errors="coerce")
            profile[f"{column}_median"] = round(float(numeric.median()), 5) if numeric.notna().any() else ""

    return pd.DataFrame([profile])


def upsert_profile(output_path: Path, profile: pd.DataFrame) -> pd.DataFrame:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        existing = pd.read_csv(output_path)
    else:
        existing = pd.DataFrame()

    if not existing.empty:
        mask = existing["user_id"].astype(str) == str(profile["user_id"].iloc[0])
        existing = existing[~mask]
    result = pd.concat([existing, profile], ignore_index=True)
    result.to_csv(output_path, index=False)
    return result


def calibrate_user(
    training_path: Path,
    output_path: Path,
    *,
    user_id: str,
    session_id: str | None = None,
    max_clips: int = 5,
) -> pd.DataFrame:
    ensure_project_dirs()
    table = add_model_targets(pd.read_csv(training_path))
    profile = build_profile(table, user_id=user_id, session_id=session_id, max_clips=max_clips)
    return upsert_profile(output_path, profile)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a BioBeat user calibration profile from a short sensor pass.")
    parser.add_argument("--training-table", default=str(TRAINING_TABLE_CSV))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--session-id")
    parser.add_argument("--max-clips", type=int, default=5)
    args = parser.parse_args()

    result = calibrate_user(
        repo_path(args.training_table),
        repo_path(args.output),
        user_id=args.user_id,
        session_id=args.session_id,
        max_clips=args.max_clips,
    )
    print(f"Wrote {len(result)} calibration profile row(s) to {repo_path(args.output)}")


if __name__ == "__main__":
    main()

