from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from biobeat_paths import CLIPS_CSV, PROCESSED_DIR, RAW_DIR, ensure_project_dirs, repo_path  # noqa: E402
from sensors.audit_eda_quality import classify, read_trial_values  # noqa: E402


REAL_SESSIONS = (
    "self_20260531_155844",
    "self_20260531_220954",
    "self_20260601_112658",
    "self_20260601_210739",
)
GOOD_REAL_DATA_CSV = PROCESSED_DIR / "good_real_data.csv"
EDA_QUALITY_MANIFEST_CSV = PROCESSED_DIR / "real_eda_quality_manifest.csv"

GOOD_REAL_DATA_FIELDS = [
    "user_id",
    "session_id",
    "clip_id",
    "trial_index",
    "track_name",
    "artist",
    "genre",
    "rest_start_time",
    "rest_end_time",
    "clip_start_time",
    "clip_end_time",
    "valence",
    "mood",
    "familiarity",
    "rating_usable",
    "data_scope",
    "eda_quality",
    "use_eda_for_primary_biometric_training",
    "eda_exclusion_reason",
]
EDA_QUALITY_MANIFEST_FIELDS = [
    "session_id",
    "trial_index",
    "clip_id",
    "has_rating",
    "serial_pattern_quality",
    "collection_quality",
    "use_eda_for_primary_biometric_training",
    "exclusion_reason",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def normalize_mood(value: str) -> str:
    mood = value.strip().lower()
    return {"sad": "negative", "happy": "positive"}.get(mood, mood)


def normalized_valence(label: dict[str, str]) -> str:
    mood = normalize_mood(label.get("mood", ""))
    if mood in {"negative", "neutral", "positive"}:
        return mood
    value = str(label.get("valence", "")).strip().lower()
    if value in {"negative", "neutral", "positive"}:
        return value
    numeric_map = {"1": "negative", "2": "neutral", "3": "positive"}
    return numeric_map.get(value, "neutral")


def collection_quality(session_id: str, trial_index: int, has_rating: bool) -> tuple[str, str]:
    if not has_rating:
        return "excluded_unrated_capture", "No saved human rating exists for this sensor capture."

    if session_id == "self_20260601_112658":
        return (
            "approved_clean_reverse_verified",
            "",
        )

    if session_id == "self_20260531_155844":
        return (
            "excluded_wrong_sensor_fit",
            "Sensor readings are structurally plausible, but the sensor was worn incorrectly.",
        )

    if session_id == "self_20260531_220954":
        if 1 <= trial_index <= 16:
            return (
                "excluded_questionable_sensor_fit",
                "Sensor placement was still being corrected during this block.",
            )
        if 17 <= trial_index <= 31 or 36 <= trial_index <= 48:
            return (
                "excluded_disconnected_lead_artifact",
                "Disconnected lead produced a repetitive artificial oscillation.",
            )
        if 32 <= trial_index <= 35:
            return (
                "excluded_reconnection_transient",
                "Brief reconnection window is too transient to trust as primary biometric data.",
            )

    return "excluded_unverified_capture", "Capture has not been verified for primary biometric training."


def use_primary_eda(quality: str) -> bool:
    return quality.startswith("approved_")


def export_good_real_data(
    clips_path: Path,
    output_path: Path,
    manifest_path: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    clips = {row["clip_id"]: row for row in read_rows(clips_path)}
    curated_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []

    for session_id in REAL_SESSIONS:
        labels_path = RAW_DIR / "labels" / f"{session_id}_labels.csv"
        sensor_path = RAW_DIR / "sensor" / f"{session_id}_sensor.csv"
        labels = read_rows(labels_path)
        labels_by_trial = {int(row["trial_index"]): row for row in labels}
        sensor_values = read_trial_values(sensor_path)

        for trial_index, values in sorted(sensor_values.items()):
            label = labels_by_trial.get(trial_index)
            serial_status, _, _ = classify(values)
            quality, reason = collection_quality(session_id, trial_index, label is not None)
            use_eda = use_primary_eda(quality)
            manifest_rows.append(
                {
                    "session_id": session_id,
                    "trial_index": trial_index,
                    "clip_id": label["clip_id"] if label else "",
                    "has_rating": "yes" if label else "no",
                    "serial_pattern_quality": serial_status,
                    "collection_quality": quality,
                    "use_eda_for_primary_biometric_training": "yes" if use_eda else "no",
                    "exclusion_reason": reason,
                }
            )

        for trial_index, label in sorted(labels_by_trial.items()):
            clip = clips[label["clip_id"]]
            quality, reason = collection_quality(session_id, trial_index, True)
            use_eda = use_primary_eda(quality)
            curated_rows.append(
                {
                    **label,
                    "valence": normalized_valence(label),
                    "mood": normalize_mood(label.get("mood", "")),
                    "track_name": clip.get("track_name", ""),
                    "artist": clip.get("artist", ""),
                    "genre": clip.get("genre", ""),
                    "rating_usable": "yes",
                    "data_scope": "human_rating_and_verified_eda" if use_eda else "human_rating_only",
                    "eda_quality": quality,
                    "use_eda_for_primary_biometric_training": "yes" if use_eda else "no",
                    "eda_exclusion_reason": reason,
                }
            )

    write_rows(output_path, curated_rows, GOOD_REAL_DATA_FIELDS)
    write_rows(manifest_path, manifest_rows, EDA_QUALITY_MANIFEST_FIELDS)
    return curated_rows, manifest_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Export usable BioBeat real ratings and an EDA quality manifest.")
    parser.add_argument("--clips", default=str(CLIPS_CSV))
    parser.add_argument("--output", default=str(GOOD_REAL_DATA_CSV))
    parser.add_argument("--manifest", default=str(EDA_QUALITY_MANIFEST_CSV))
    args = parser.parse_args()

    ensure_project_dirs()
    curated_rows, manifest_rows = export_good_real_data(
        repo_path(args.clips),
        repo_path(args.output),
        repo_path(args.manifest),
    )
    print(f"Wrote {len(curated_rows)} usable rating rows to {repo_path(args.output)}")
    print(f"Wrote {len(manifest_rows)} EDA audit rows to {repo_path(args.manifest)}")


if __name__ == "__main__":
    main()
