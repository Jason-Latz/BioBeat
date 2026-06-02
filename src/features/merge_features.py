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
from features.target_features import add_model_targets  # noqa: E402


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
EDA_QUALITY_MANIFEST_CSV = PROCESSED_DIR / "real_eda_quality_manifest.csv"
GOOD_REAL_DATA_CSV = PROCESSED_DIR / "good_real_data.csv"
EDA_COLUMNS = [
    "eda_mean",
    "eda_change_from_baseline",
    "eda_peak_count",
    "eda_max_amplitude",
    "eda_slope",
    "eda_recovery",
]
CLIP_METADATA_COLUMNS = [
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


def read_label_frames(paths: list[Path]) -> pd.DataFrame:
    if not paths:
        raise FileNotFoundError(
            "No label files found. Run the Streamlit runner or "
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


def read_labels(labels_dir: Path) -> pd.DataFrame:
    paths = [Path(path) for path in sorted(glob.glob(str(labels_dir / "*_labels.csv")))]
    return read_label_frames(paths)


def read_labels_file(labels_file: Path) -> pd.DataFrame:
    return read_label_frames([labels_file])


def merge_features(
    clips_path: Path,
    labels_dir: Path,
    audio_features_path: Path,
    biometric_features_path: Path,
    output_path: Path,
    labels_file: Path | None = None,
) -> pd.DataFrame:
    ensure_project_dirs()
    clips = pd.read_csv(clips_path)
    labels = read_labels_file(labels_file) if labels_file is not None else read_labels(labels_dir)
    labels = labels.drop(columns=CLIP_METADATA_COLUMNS, errors="ignore")

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
            clips[["clip_id", *CLIP_METADATA_COLUMNS]],
            on="clip_id",
            how="left",
        )
        .merge(audio, on="clip_id", how="left")
        .merge(biometrics, on=["user_id", "session_id", "clip_id"], how="left")
    )
    if EDA_QUALITY_MANIFEST_CSV.exists():
        manifest = pd.read_csv(EDA_QUALITY_MANIFEST_CSV)
        manifest_columns = [
            "session_id",
            "trial_index",
            "clip_id",
            "use_eda_for_primary_biometric_training",
            "collection_quality",
            "exclusion_reason",
        ]
        available_columns = [column for column in manifest_columns if column in manifest.columns]
        table = table.merge(
            manifest[available_columns],
            on=[column for column in ["session_id", "trial_index", "clip_id"] if column in available_columns],
            how="left",
        )
        if "use_eda_for_primary_biometric_training_y" in table.columns:
            table["use_eda_for_primary_biometric_training"] = table[
                "use_eda_for_primary_biometric_training_y"
            ].combine_first(table.get("use_eda_for_primary_biometric_training_x"))
            table = table.drop(
                columns=[
                    "use_eda_for_primary_biometric_training_x",
                    "use_eda_for_primary_biometric_training_y",
                ],
                errors="ignore",
            )
        use_eda = table.get("use_eda_for_primary_biometric_training", "yes")
        if not isinstance(use_eda, pd.Series):
            use_eda = pd.Series("yes", index=table.index)
        # Missing manifest rows are common for fresh calibration sessions. Only mask
        # EDA when the quality manifest explicitly marks the trial as not usable.
        excluded = use_eda.notna() & use_eda.astype(str).str.lower().isin({"no", "false", "0"})
        for column in EDA_COLUMNS:
            if column in table.columns:
                table.loc[excluded, column] = pd.NA
    table = add_model_targets(table)

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
    parser.add_argument(
        "--labels-file",
        help="Optional single label/export CSV. Use data/processed/good_real_data.csv for curated real training data.",
    )
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
        labels_file=repo_path(args.labels_file) if args.labels_file else None,
    )
    print(f"Wrote {len(table)} training rows to {repo_path(args.output)}")


if __name__ == "__main__":
    main()
