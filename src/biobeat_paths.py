from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
RAW_DIR = DATA_DIR / "raw"
LABELS_DIR = RAW_DIR / "labels"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

CLIPS_CSV = DATA_DIR / "clips.csv"
DESIRED_SONGS_CSV = INPUT_DIR / "desired_songs.csv"
AUDIO_FEATURES_CSV = PROCESSED_DIR / "audio_features.csv"
FAKE_BIOMETRICS_CSV = PROCESSED_DIR / "biometric_features_fake.csv"
TRAINING_TABLE_CSV = PROCESSED_DIR / "training_table.csv"
RECOMMENDATIONS_CSV = PROCESSED_DIR / "recommendations.csv"


def ensure_project_dirs() -> None:
    for directory in (
        INPUT_DIR,
        LABELS_DIR,
        RAW_DIR / "sensor",
        PROCESSED_DIR,
        MODELS_DIR,
        MODELS_DIR / "metrics",
    ):
        directory.mkdir(parents=True, exist_ok=True)


def repo_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
