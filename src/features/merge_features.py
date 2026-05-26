from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from biobeat_paths import (  # noqa: E402
    AUDIO_FEATURES_CSV,
    CLIPS_CSV,
    FAKE_BIOMETRICS_CSV,
    LABELS_DIR,
    PROCESSED_DIR,
    TRAINING_TABLE_CSV,
    ensure_project_dirs,
    repo_path,
)


LABEL_COLUMNS = [
    "user_id",
    "session_id",
    "clip_id",
    "trial_index",
    "rest_start_time",
    "rest_end_time",
    "clip_start_time",
    "clip_end_time",
    "preference",
    "arousal",
    "valence",
    "mood",
    "familiarity",
    "notes",
]


def read_labels(labels_dir: Path) -> pd.DataFrame:
    paths = sorted(glob.glob(str(labels_dir / "*_labels.csv")))
    if not paths:
        raise FileNotFoundError(
            f"No label files found in {labels_dir}. Run the Streamlit runner or "
            "src/features/generate_demo_labels.py first."
        )
    frames = [pd.read_csv(path) for path in paths]
    labels = pd.concat(frames, ignore_index=True)
    required = {
        "user_id",
        "session_id",
        "clip_id",
        "trial_index",
        "clip_start_time",
        "clip_end_time",
        "preference",
        "arousal",
        "valence",
        "mood",
        "familiarity",
    }
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Label files are missing required columns: {', '.join(sorted(missing))}")
    for column in LABEL_COLUMNS:
        if column not in labels.columns:
            labels[column] = ""
    return labels


def add_binary_labels(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    for column in ("preference", "arousal", "valence"):
        table[column] = pd.to_numeric(table[column], errors="coerce")
        table[f"{column}_binary"] = (table[column] >= 4).astype(int)
    return table


def merge_features(
    clips_path: Path,
    labels_dir: Path,
    audio_features_path: Path,
    biometric_features_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    ensure_project_dirs()
    clips = pd.read_csv(clips_path)
    labels = read_labels(labels_dir)

    if not audio_features_path.exists():
        raise FileNotFoundError(
            f"{audio_features_path} does not exist. Run src/features/extract_audio_features.py first."
        )
    if not biometric_features_path.exists():
        raise FileNotFoundError(
            f"{biometric_features_path} does not exist. Run src/features/generate_fake_biometrics.py first."
        )

    audio = pd.read_csv(audio_features_path)
    biometrics = pd.read_csv(biometric_features_path)

    table = (
        labels.merge(
            clips[
                [
                    "clip_id",
                    "track_name",
                    "artist",
                    "album",
                    "genre",
                    "preview_url",
                    "track_id",
                    "intended_arousal",
                    "intended_valence",
                    "intended_mood",
                ]
            ],
            on="clip_id",
            how="left",
        )
        .merge(audio, on="clip_id", how="left")
        .merge(biometrics, on=["user_id", "session_id", "clip_id"], how="left")
    )
    table = add_binary_labels(table)

    first_columns = [
        "user_id",
        "session_id",
        "clip_id",
        "trial_index",
        "track_name",
        "artist",
        "genre",
        "intended_arousal",
        "intended_valence",
        "intended_mood",
    ]
    remaining = [column for column in table.columns if column not in first_columns]
    table = table[first_columns + remaining]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge labels, metadata, audio, and biometric features.")
    parser.add_argument("--clips", default=str(CLIPS_CSV))
    parser.add_argument("--labels-dir", default=str(LABELS_DIR))
    parser.add_argument("--audio-features", default=str(AUDIO_FEATURES_CSV))
    parser.add_argument("--biometric-features", default=str(FAKE_BIOMETRICS_CSV))
    parser.add_argument("--use-real-biometrics", action="store_true")
    parser.add_argument("--output", default=str(TRAINING_TABLE_CSV))
    args = parser.parse_args()

    biometric_path = repo_path(args.biometric_features)
    if args.use_real_biometrics:
        real_path = PROCESSED_DIR / "biometric_features_real.csv"
        if not real_path.exists():
            raise FileNotFoundError(f"{real_path} does not exist.")
        biometric_path = real_path

    table = merge_features(
        repo_path(args.clips),
        repo_path(args.labels_dir),
        repo_path(args.audio_features),
        biometric_path,
        repo_path(args.output),
    )
    print(f"Wrote {len(table)} training rows to {repo_path(args.output)}")


if __name__ == "__main__":
    main()
