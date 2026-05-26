from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from biobeat_paths import (  # noqa: E402
    AUDIO_FEATURES_CSV,
    CLIPS_CSV,
    FAKE_BIOMETRICS_CSV,
    MODELS_DIR,
    RECOMMENDATIONS_CSV,
    ensure_project_dirs,
    repo_path,
)


TARGET_MODEL_FILES = {
    "preference_binary": "preference_binary_random_forest.joblib",
    "arousal_binary": "arousal_binary_random_forest.joblib",
    "valence_binary": "valence_binary_random_forest.joblib",
}


def load_bundle(target: str) -> dict[str, object]:
    path = MODELS_DIR / TARGET_MODEL_FILES[target]
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run src/models/train_models.py first.")
    return joblib.load(path)


def predict_positive_probability(bundle: dict[str, object], rows: pd.DataFrame) -> np.ndarray:
    columns = list(bundle["feature_columns"])
    X = rows.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
    constant = bundle.get("constant_probability")
    model = bundle.get("model")
    if model is None or constant is not None:
        return np.full(len(X), float(constant or 0.0))

    probabilities = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(X), dtype=float)
    return probabilities[:, classes.index(1)]


def candidate_table(
    clips_path: Path,
    audio_features_path: Path,
    biometric_features_path: Path,
    *,
    user_id: str,
    session_id: str | None,
) -> pd.DataFrame:
    clips = pd.read_csv(clips_path)
    audio = pd.read_csv(audio_features_path)
    biometrics = pd.read_csv(biometric_features_path)

    user_biometrics = biometrics[biometrics["user_id"].astype(str) == str(user_id)].copy()
    if user_biometrics.empty:
        user_biometrics = biometrics.copy()
        user_biometrics["user_id"] = user_id

    if session_id:
        session_rows = user_biometrics[user_biometrics["session_id"].astype(str) == str(session_id)].copy()
        if not session_rows.empty:
            user_biometrics = session_rows
    elif "session_id" in user_biometrics.columns:
        selected_session = sorted(user_biometrics["session_id"].astype(str).unique())[-1]
        user_biometrics = user_biometrics[user_biometrics["session_id"].astype(str) == selected_session]

    table = clips.merge(audio, on="clip_id", how="left").merge(
        user_biometrics,
        on="clip_id",
        how="left",
        suffixes=("", "_biometric"),
    )
    table["user_id"] = user_id
    if "session_id" not in table.columns or table["session_id"].isna().all():
        table["session_id"] = session_id or "recommendation_session"

    numeric_columns = table.select_dtypes(include=["number"]).columns
    table[numeric_columns] = table[numeric_columns].fillna(table[numeric_columns].median(numeric_only=True))
    table[numeric_columns] = table[numeric_columns].fillna(0)
    return table


def score_recommendations(table: pd.DataFrame, target_mode: str) -> pd.DataFrame:
    preference_bundle = load_bundle("preference_binary")
    arousal_bundle = load_bundle("arousal_binary")
    valence_bundle = load_bundle("valence_binary")

    scored = table.copy()
    scored["preference_prob"] = predict_positive_probability(preference_bundle, scored)
    scored["arousal_prob"] = predict_positive_probability(arousal_bundle, scored)
    scored["valence_prob"] = predict_positive_probability(valence_bundle, scored)
    scored["calm_prob"] = 1 - scored["arousal_prob"]

    if target_mode == "calm":
        scored["score"] = (
            0.45 * scored["preference_prob"]
            + 0.40 * scored["calm_prob"]
            + 0.15 * scored["valence_prob"]
        )
    elif target_mode == "hype":
        scored["score"] = (
            0.45 * scored["preference_prob"]
            + 0.40 * scored["arousal_prob"]
            + 0.15 * scored["valence_prob"]
        )
    else:
        raise ValueError("target_mode must be calm or hype")

    scored = scored.sort_values("score", ascending=False).reset_index(drop=True)
    scored["rank"] = scored.index + 1
    scored["target_mode"] = target_mode
    columns = [
        "user_id",
        "target_mode",
        "rank",
        "clip_id",
        "track_name",
        "artist",
        "score",
        "preference_prob",
        "arousal_prob",
        "valence_prob",
    ]
    return scored[columns]


def recommend(
    clips_path: Path,
    audio_features_path: Path,
    biometric_features_path: Path,
    output_path: Path,
    *,
    user_id: str,
    target_mode: str,
    session_id: str | None = None,
) -> pd.DataFrame:
    ensure_project_dirs()
    table = candidate_table(
        clips_path,
        audio_features_path,
        biometric_features_path,
        user_id=user_id,
        session_id=session_id,
    )
    modes = ["calm", "hype"] if target_mode == "all" else [target_mode]
    frames = [score_recommendations(table, mode) for mode in modes]
    result = pd.concat(frames, ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank BioBeat recommendations for calm or hype mode.")
    parser.add_argument("--clips", default=str(CLIPS_CSV))
    parser.add_argument("--audio-features", default=str(AUDIO_FEATURES_CSV))
    parser.add_argument("--biometric-features", default=str(FAKE_BIOMETRICS_CSV))
    parser.add_argument("--output", default=str(RECOMMENDATIONS_CSV))
    parser.add_argument("--target-mode", choices=["calm", "hype", "all"], default="calm")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--session-id")
    args = parser.parse_args()

    result = recommend(
        repo_path(args.clips),
        repo_path(args.audio_features),
        repo_path(args.biometric_features),
        repo_path(args.output),
        user_id=args.user_id,
        target_mode=args.target_mode,
        session_id=args.session_id,
    )
    print(f"Wrote {len(result)} recommendations to {repo_path(args.output)}")


if __name__ == "__main__":
    main()
