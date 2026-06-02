from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from biobeat_paths import AUDIO_FEATURES_CSV, CLIPS_CSV, MODELS_DIR, PROCESSED_DIR, RECOMMENDATIONS_CSV, ensure_project_dirs, repo_path  # noqa: E402
from features.target_features import VALENCE_LABELS  # noqa: E402
from models.prediction_utils import load_bundle, predict_class_probabilities, predict_regression  # noqa: E402


AROUSAL_MODEL = MODELS_DIR / "arousal_score_audio_random_forest_regressor.joblib"
VALENCE_MODEL = MODELS_DIR / "valence_label_audio_random_forest_classifier.joblib"
CALIBRATION_PROFILES_CSV = PROCESSED_DIR / "user_calibration_profiles.csv"

MOOD_TARGETS = {
    "calm": ("neutral", 0.20),
    "relaxed": ("positive", 0.20),
    "happy": ("positive", 0.45),
    "excited": ("positive", 0.85),
    "hype": ("positive", 0.85),
    "sad": ("negative", 0.20),
    "tense": ("negative", 0.80),
    "angry": ("negative", 0.90),
    "neutral": ("neutral", 0.35),
    "alert": ("neutral", 0.75),
}
DEFAULT_ALL_MOODS = ["relaxed", "excited", "sad", "tense", "neutral"]


def candidate_table(clips_path: Path, audio_features_path: Path, *, user_id: str) -> pd.DataFrame:
    clips = pd.read_csv(clips_path)
    audio = pd.read_csv(audio_features_path)
    table = clips.merge(audio, on="clip_id", how="left")
    table["user_id"] = user_id
    numeric_columns = table.select_dtypes(include=["number"]).columns
    table[numeric_columns] = table[numeric_columns].fillna(table[numeric_columns].median(numeric_only=True))
    table[numeric_columns] = table[numeric_columns].fillna(0)
    return table


def load_user_profile(path: Path, user_id: str) -> pd.Series | None:
    if not path.exists():
        return None
    profiles = pd.read_csv(path)
    if profiles.empty or "user_id" not in profiles.columns:
        return None
    rows = profiles[profiles["user_id"].astype(str) == str(user_id)]
    return None if rows.empty else rows.iloc[-1]


def apply_calibration(
    arousal: np.ndarray,
    valence_probs: pd.DataFrame,
    profile: pd.Series | None,
) -> tuple[np.ndarray, pd.DataFrame]:
    if profile is None:
        return arousal, valence_probs

    offset = pd.to_numeric(pd.Series([profile.get("arousal_offset", 0.0)]), errors="coerce").fillna(0.0).iloc[0]
    calibrated_arousal = np.clip(arousal + float(offset), 0.0, 1.0)

    calibrated_probs = valence_probs.copy()
    for label in VALENCE_LABELS:
        shift = pd.to_numeric(
            pd.Series([profile.get(f"valence_shift_{label}", 1.0)]),
            errors="coerce",
        ).fillna(1.0).iloc[0]
        calibrated_probs[label] = calibrated_probs[label] * float(shift)
    row_sums = calibrated_probs.sum(axis=1).replace(0, 1)
    calibrated_probs = calibrated_probs.div(row_sums, axis=0)
    return calibrated_arousal, calibrated_probs


def inferred_mood(valence_label: str, arousal: float) -> str:
    if valence_label == "positive":
        return "excited" if arousal >= 0.65 else "happy" if arousal >= 0.40 else "relaxed"
    if valence_label == "negative":
        return "tense" if arousal >= 0.60 else "sad"
    return "alert" if arousal >= 0.65 else "calm" if arousal <= 0.35 else "neutral"


def score_for_mood(
    table: pd.DataFrame,
    *,
    target_mood: str,
    arousal: np.ndarray,
    valence_probs: pd.DataFrame,
) -> pd.DataFrame:
    if target_mood not in MOOD_TARGETS:
        raise ValueError(f"Unknown target mood: {target_mood}")
    target_valence, target_arousal = MOOD_TARGETS[target_mood]
    arousal_fit = 1.0 - np.abs(arousal - target_arousal)
    valence_fit = valence_probs[target_valence].to_numpy(dtype=float)

    scored = table.copy()
    scored["target_mood"] = target_mood
    scored["predicted_arousal"] = arousal
    for label in VALENCE_LABELS:
        scored[f"valence_prob_{label}"] = valence_probs[label].to_numpy(dtype=float)
    scored["predicted_valence"] = valence_probs.idxmax(axis=1).to_numpy()
    scored["predicted_mood"] = [
        inferred_mood(label, value)
        for label, value in zip(scored["predicted_valence"], scored["predicted_arousal"], strict=False)
    ]
    scored["score"] = 0.60 * valence_fit + 0.40 * arousal_fit
    scored = scored.sort_values("score", ascending=False).reset_index(drop=True)
    scored["rank"] = scored.index + 1
    return scored[
        [
            "user_id",
            "target_mood",
            "rank",
            "clip_id",
            "track_name",
            "artist",
            "score",
            "predicted_mood",
            "predicted_arousal",
            "predicted_valence",
            "valence_prob_negative",
            "valence_prob_neutral",
            "valence_prob_positive",
        ]
    ]


def recommend(
    clips_path: Path,
    audio_features_path: Path,
    output_path: Path,
    *,
    user_id: str,
    target_mood: str,
    calibration_profiles_path: Path = CALIBRATION_PROFILES_CSV,
) -> pd.DataFrame:
    ensure_project_dirs()
    table = candidate_table(clips_path, audio_features_path, user_id=user_id)
    arousal_bundle = load_bundle(AROUSAL_MODEL)
    valence_bundle = load_bundle(VALENCE_MODEL)

    arousal = np.clip(predict_regression(arousal_bundle, table), 0.0, 1.0)
    valence_probs = predict_class_probabilities(valence_bundle, table)
    profile = load_user_profile(calibration_profiles_path, user_id)
    arousal, valence_probs = apply_calibration(arousal, valence_probs, profile)

    moods = DEFAULT_ALL_MOODS if target_mood == "all" else [target_mood]
    frames = [
        score_for_mood(table, target_mood=mood, arousal=arousal, valence_probs=valence_probs)
        for mood in moods
    ]
    result = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank BioBeat recommendations for a desired mood.")
    parser.add_argument("--clips", default=str(CLIPS_CSV))
    parser.add_argument("--audio-features", default=str(AUDIO_FEATURES_CSV))
    parser.add_argument("--output", default=str(RECOMMENDATIONS_CSV))
    parser.add_argument("--target-mood", "--target-mode", choices=sorted(MOOD_TARGETS) + ["all"], default="calm")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--session-id", help="Accepted for older commands; calibration profiles are selected by user ID.")
    parser.add_argument("--biometric-features", help="Accepted for older commands; recommendations use audio features plus calibration profiles.")
    parser.add_argument("--calibration-profiles", default=str(CALIBRATION_PROFILES_CSV))
    args = parser.parse_args()

    result = recommend(
        repo_path(args.clips),
        repo_path(args.audio_features),
        repo_path(args.output),
        user_id=args.user_id,
        target_mood=args.target_mood,
        calibration_profiles_path=repo_path(args.calibration_profiles),
    )
    print(f"Wrote {len(result)} recommendations to {repo_path(args.output)}")


if __name__ == "__main__":
    main()
