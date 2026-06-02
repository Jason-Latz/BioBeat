from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))

from biobeat_paths import AUDIO_FEATURES_CSV, CLIPS_CSV, ensure_project_dirs, repo_path  # noqa: E402


BASE_FEATURE_COLUMNS = [
    "clip_id",
    "duration_sec",
    "tempo",
    "beat_count",
    "beat_rate",
    "beat_interval_mean",
    "beat_interval_std",
    "rhythm_regularity",
    "onset_rate",
    "rms_energy_mean",
    "rms_energy_std",
    "rms_energy_range",
    "rms_energy_slope",
    "loudness_db_mean",
    "loudness_db_std",
    "loudness_db_range",
    "loudness_db_p10",
    "loudness_db_p90",
    "quiet_fraction",
    "zero_crossing_rate_mean",
    "zero_crossing_rate_std",
    "spectral_centroid_mean",
    "spectral_centroid_std",
    "spectral_centroid_slope",
    "spectral_bandwidth_mean",
    "spectral_bandwidth_std",
    "spectral_rolloff_mean",
    "spectral_rolloff_std",
    "spectral_contrast_mean",
    "spectral_contrast_std",
    "onset_strength_mean",
    "onset_strength_std",
    "mode_major",
    "mode_confidence",
    "key_pitch_class",
    "major_key_correlation",
    "minor_key_correlation",
]
FEATURE_COLUMNS = (
    BASE_FEATURE_COLUMNS
    + [f"chroma_mean_{index}" for index in range(1, 13)]
    + [f"mfcc_mean_{index}" for index in range(1, 14)]
    + [f"mfcc_std_{index}" for index in range(1, 14)]
)


def suffix_from_url(preview_url: str) -> str:
    path = urlparse(preview_url).path
    suffix = Path(path).suffix
    return suffix if suffix else ".m4a"


def download_preview_to_temp(preview_url: str) -> tempfile.NamedTemporaryFile:
    response = requests.get(preview_url, timeout=30)
    response.raise_for_status()
    temp_file = tempfile.NamedTemporaryFile(suffix=suffix_from_url(preview_url))
    temp_file.write(response.content)
    temp_file.flush()
    return temp_file


def feature_slope(values: np.ndarray) -> float:
    flat = np.ravel(values).astype(float)
    if flat.size < 2:
        return 0.0
    x = np.arange(flat.size, dtype=float)
    return float(np.polyfit(x, flat, 1)[0])


def safe_mean(values: np.ndarray) -> float:
    flat = np.ravel(values).astype(float)
    return float(np.mean(flat)) if flat.size else 0.0


def safe_std(values: np.ndarray) -> float:
    flat = np.ravel(values).astype(float)
    return float(np.std(flat)) if flat.size else 0.0


def estimate_mode_features(chroma: np.ndarray) -> dict[str, float]:
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    chroma_mean = np.mean(chroma, axis=1).astype(float)

    if not np.any(chroma_mean):
        return {
            "mode_major": 0.0,
            "mode_confidence": 0.0,
            "key_pitch_class": 0.0,
            "major_key_correlation": 0.0,
            "minor_key_correlation": 0.0,
        }

    chroma_centered = chroma_mean - np.mean(chroma_mean)
    if np.linalg.norm(chroma_centered) == 0:
        return {
            "mode_major": 0.0,
            "mode_confidence": 0.0,
            "key_pitch_class": 0.0,
            "major_key_correlation": 0.0,
            "minor_key_correlation": 0.0,
        }
    major_centered = major_profile - np.mean(major_profile)
    minor_centered = minor_profile - np.mean(minor_profile)

    major_scores = [
        float(np.corrcoef(chroma_centered, np.roll(major_centered, pitch_class))[0, 1])
        for pitch_class in range(12)
    ]
    minor_scores = [
        float(np.corrcoef(chroma_centered, np.roll(minor_centered, pitch_class))[0, 1])
        for pitch_class in range(12)
    ]
    best_major = max(major_scores)
    best_minor = max(minor_scores)
    mode_is_major = best_major >= best_minor
    best_score = best_major if mode_is_major else best_minor
    runner_up = best_minor if mode_is_major else best_major
    key_pitch_class = major_scores.index(best_major) if mode_is_major else minor_scores.index(best_minor)

    return {
        "mode_major": float(mode_is_major),
        "mode_confidence": float(max(0.0, best_score - runner_up)),
        "key_pitch_class": float(key_pitch_class),
        "major_key_correlation": float(best_major),
        "minor_key_correlation": float(best_minor),
    }


def row_has_complete_features(row: dict[str, object]) -> bool:
    for column in FEATURE_COLUMNS:
        value = row.get(column)
        if value is None or value == "" or pd.isna(value):
            return False
    return True


def extract_features_from_preview(preview_url: str) -> dict[str, float]:
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError(
            "librosa is not installed. Run `pip install -r requirements.txt` first."
        ) from exc

    with download_preview_to_temp(preview_url) as audio_file:
        try:
            y, sr = librosa.load(audio_file.name, sr=None, mono=True)
        except Exception as exc:
            raise RuntimeError(
                "Could not load the preview audio. If this is an .m4a preview, install ffmpeg "
                "with `brew install ffmpeg` and rerun the script."
            ) from exc

    if y.size == 0:
        raise RuntimeError("Downloaded preview contained no audio samples.")

    duration_sec = float(librosa.get_duration(y=y, sr=sr))
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    beat_intervals = np.diff(beat_times)
    rms = librosa.feature.rms(y=y)
    rms_db = librosa.amplitude_to_db(rms, ref=1.0)
    zcr = librosa.feature.zero_crossing_rate(y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
    onset_events = librosa.onset.onset_detect(y=y, sr=sr, onset_envelope=onset_strength)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mode_features = estimate_mode_features(chroma)
    beat_interval_mean = safe_mean(beat_intervals)
    beat_interval_std = safe_std(beat_intervals)
    rhythm_regularity = 0.0
    if beat_interval_mean > 0:
        rhythm_regularity = float(1.0 / (1.0 + beat_interval_std / beat_interval_mean))

    features = {
        "duration_sec": duration_sec,
        "tempo": float(np.ravel(tempo)[0]),
        "beat_count": float(len(beat_frames)),
        "beat_rate": float(len(beat_frames) / duration_sec) if duration_sec else 0.0,
        "beat_interval_mean": beat_interval_mean,
        "beat_interval_std": beat_interval_std,
        "rhythm_regularity": rhythm_regularity,
        "onset_rate": float(len(onset_events) / duration_sec) if duration_sec else 0.0,
        "rms_energy_mean": float(np.mean(rms)),
        "rms_energy_std": float(np.std(rms)),
        "rms_energy_range": float(np.max(rms) - np.min(rms)),
        "rms_energy_slope": feature_slope(rms),
        "loudness_db_mean": float(np.mean(rms_db)),
        "loudness_db_std": float(np.std(rms_db)),
        "loudness_db_range": float(np.max(rms_db) - np.min(rms_db)),
        "loudness_db_p10": float(np.percentile(rms_db, 10)),
        "loudness_db_p90": float(np.percentile(rms_db, 90)),
        "quiet_fraction": float(np.mean(rms_db < -45.0)),
        "zero_crossing_rate_mean": float(np.mean(zcr)),
        "zero_crossing_rate_std": float(np.std(zcr)),
        "spectral_centroid_mean": float(np.mean(centroid)),
        "spectral_centroid_std": float(np.std(centroid)),
        "spectral_centroid_slope": feature_slope(centroid),
        "spectral_bandwidth_mean": float(np.mean(bandwidth)),
        "spectral_bandwidth_std": float(np.std(bandwidth)),
        "spectral_rolloff_mean": float(np.mean(rolloff)),
        "spectral_rolloff_std": float(np.std(rolloff)),
        "spectral_contrast_mean": float(np.mean(contrast)),
        "spectral_contrast_std": float(np.std(contrast)),
        "onset_strength_mean": float(np.mean(onset_strength)),
        "onset_strength_std": float(np.std(onset_strength)),
        **mode_features,
    }

    for index, value in enumerate(np.mean(chroma, axis=1), start=1):
        features[f"chroma_mean_{index}"] = float(value)

    for index, value in enumerate(np.mean(mfcc, axis=1), start=1):
        features[f"mfcc_mean_{index}"] = float(value)

    for index, value in enumerate(np.std(mfcc, axis=1), start=1):
        features[f"mfcc_std_{index}"] = float(value)

    return features


def load_existing_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=FEATURE_COLUMNS)
    return pd.read_csv(path).reindex(columns=FEATURE_COLUMNS)


def write_feature_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def extract_all_features(
    clips_path: Path,
    output_path: Path,
    *,
    force: bool = False,
    limit: int | None = None,
) -> pd.DataFrame:
    ensure_project_dirs()
    clips = pd.read_csv(clips_path)
    existing = load_existing_features(output_path)
    existing_by_clip: dict[str, dict[str, object]] = {}
    if not force and not existing.empty:
        existing_by_clip = {
            str(row["clip_id"]): row
            for row in existing.to_dict("records")
            if pd.notna(row.get("clip_id"))
        }
    processed_any = False

    if limit is not None:
        clips = clips.head(limit)

    for _, clip in clips.iterrows():
        clip_id = str(clip["clip_id"])
        existing_row = existing_by_clip.get(clip_id)
        if existing_row is not None and row_has_complete_features(existing_row):
            print(f"Skipping cached audio features for {clip_id}")
            continue

        print(f"Extracting audio features for {clip_id}: {clip['track_name']} - {clip['artist']}")
        try:
            features = extract_features_from_preview(str(clip["preview_url"]))
        except Exception as exc:
            print(f"Failed to extract {clip_id}: {exc}", file=sys.stderr)
            continue

        existing_by_clip[clip_id] = {"clip_id": clip_id, **features}
        processed_any = True
        write_feature_rows(output_path, list(existing_by_clip.values()))

    result = pd.DataFrame(list(existing_by_clip.values()))
    if not result.empty and (force or processed_any or not output_path.exists()):
        result = result.reindex(columns=FEATURE_COLUMNS)
        result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract librosa features from iTunes preview URLs.")
    parser.add_argument("--clips", default=str(CLIPS_CSV))
    parser.add_argument("--output", default=str(AUDIO_FEATURES_CSV))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    result = extract_all_features(
        repo_path(args.clips),
        repo_path(args.output),
        force=args.force,
        limit=args.limit,
    )
    print(f"Wrote {len(result)} audio feature rows to {repo_path(args.output)}")


if __name__ == "__main__":
    main()
