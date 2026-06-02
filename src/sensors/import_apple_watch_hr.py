from __future__ import annotations

import argparse
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from biobeat_paths import RAW_DIR, repo_path  # noqa: E402
from sensors.serial_readers import append_sensor_rows  # noqa: E402


SENSOR_COLUMNS = [
    "timestamp",
    "user_id",
    "session_id",
    "clip_id",
    "trial_index",
    "phase",
    "hr",
    "eda",
    "sensor_elapsed_ms",
    "source",
    "raw_line",
]
TIME_ALIASES = {
    "timestamp",
    "time",
    "date",
    "datetime",
    "start",
    "startdate",
    "starttime",
    "startdatetime",
    "creationdate",
}
HR_ALIASES = {
    "hr",
    "bpm",
    "value",
    "heartrate",
    "heart",
    "heartbpm",
    "heart_rate",
    "heart rate",
    "heart_rate_bpm",
    "heart rate bpm",
    "heart rate count min",
}
APPLE_HEALTH_HR_TYPE = "HKQuantityTypeIdentifierHeartRate"


def normalize_column(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def find_column(columns: list[str], aliases: set[str], *, description: str) -> str:
    normalized_aliases = {normalize_column(alias) for alias in aliases}
    for column in columns:
        if normalize_column(column) in normalized_aliases:
            return column
    raise ValueError(
        f"Could not find a {description} column. Available columns: {', '.join(columns)}"
    )


def parse_timestamps(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, errors="coerce")
    try:
        if parsed.dt.tz is not None:
            parsed = parsed.dt.tz_localize(None)
    except AttributeError:
        parsed = parsed.apply(lambda value: value.tz_localize(None) if getattr(value, "tzinfo", None) else value)
    return parsed


def read_apple_health_xml(source: object) -> pd.DataFrame:
    tree = ET.parse(source)
    rows: list[dict[str, object]] = []
    for record in tree.iterfind(".//Record"):
        if record.attrib.get("type") != APPLE_HEALTH_HR_TYPE:
            continue
        rows.append(
            {
                "timestamp": record.attrib.get("startDate") or record.attrib.get("creationDate"),
                "value": record.attrib.get("value"),
                "unit": record.attrib.get("unit", ""),
                "sourceName": record.attrib.get("sourceName", ""),
            }
        )
    if not rows:
        raise ValueError("No Apple Health heart-rate records were found in the XML file.")
    return pd.DataFrame(rows)


def read_health_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".xml":
        return read_apple_health_xml(path)
    return pd.read_csv(path)


def default_sensor_path(labels_path: Path) -> Path:
    name = labels_path.name
    if name.endswith("_labels.csv"):
        return RAW_DIR / "sensor" / name.replace("_labels.csv", "_sensor.csv")
    return RAW_DIR / "sensor" / f"{labels_path.stem}_sensor.csv"


def build_hr_rows_from_frames(
    *,
    health: pd.DataFrame,
    labels: pd.DataFrame,
    source: str,
    time_offset_seconds: float,
) -> list[dict[str, object]]:
    if health.empty:
        return []

    time_column = find_column(list(health.columns), TIME_ALIASES, description="timestamp")
    hr_column = find_column(list(health.columns), HR_ALIASES, description="heart-rate")

    health = health.copy()
    health["_timestamp"] = parse_timestamps(health[time_column])
    if time_offset_seconds:
        health["_timestamp"] = health["_timestamp"] + pd.to_timedelta(time_offset_seconds, unit="s")
    health["_hr"] = pd.to_numeric(health[hr_column], errors="coerce")
    health = health.dropna(subset=["_timestamp", "_hr"])

    rows: list[dict[str, object]] = []
    for _, label in labels.iterrows():
        windows = [
            ("rest", label.get("rest_start_time"), label.get("rest_end_time")),
            ("listen", label.get("clip_start_time"), label.get("clip_end_time")),
        ]
        for phase, start_value, end_value in windows:
            start = pd.to_datetime(start_value, errors="coerce")
            end = pd.to_datetime(end_value, errors="coerce")
            if pd.isna(start) or pd.isna(end):
                continue

            window_rows = health[(health["_timestamp"] >= start) & (health["_timestamp"] <= end)]
            for _, sample in window_rows.iterrows():
                timestamp = sample["_timestamp"].to_pydatetime().isoformat(timespec="milliseconds")
                rows.append(
                    {
                        "timestamp": timestamp,
                        "user_id": label["user_id"],
                        "session_id": label["session_id"],
                        "clip_id": label["clip_id"],
                        "trial_index": int(label["trial_index"]),
                        "phase": phase,
                        "hr": round(float(sample["_hr"]), 3),
                        "eda": "",
                        "sensor_elapsed_ms": "",
                        "source": source,
                        "raw_line": f"{time_column}={sample[time_column]},{hr_column}={sample[hr_column]}",
                    }
                )
    return rows


def build_hr_rows(
    *,
    health_csv: Path,
    labels_file: Path,
    source: str,
    time_offset_seconds: float,
) -> list[dict[str, object]]:
    return build_hr_rows_from_frames(
        health=read_health_file(health_csv),
        labels=pd.read_csv(labels_file),
        source=source,
        time_offset_seconds=time_offset_seconds,
    )


def write_imported_rows(path: Path, rows: list[dict[str, object]], *, source: str, replace_source: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if replace_source and path.exists():
        existing = pd.read_csv(path)
        if "source" in existing.columns:
            existing = existing[existing["source"].astype(str) != source]
        existing = existing.reindex(columns=SENSOR_COLUMNS)
        existing.to_csv(path, index=False)
    append_sensor_rows(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import Apple Watch or Apple Health heart-rate CSV/XML rows into a BioBeat raw sensor file."
    )
    parser.add_argument("--health-csv", required=True, help="CSV or Apple Health XML exported from Apple Watch/Apple Health.")
    parser.add_argument("--labels-file", required=True, help="BioBeat labels CSV for the matching session.")
    parser.add_argument("--output", help="Raw sensor CSV to update. Defaults to the matching session sensor file.")
    parser.add_argument("--source", default="apple_watch_csv")
    parser.add_argument(
        "--time-offset-seconds",
        type=float,
        default=0.0,
        help="Optional timestamp shift if the Health file clock does not line up with the dashboard clock.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append rows without replacing previously imported Apple Watch rows from the same source.",
    )
    args = parser.parse_args()

    labels_file = repo_path(args.labels_file)
    output = repo_path(args.output) if args.output else default_sensor_path(labels_file)
    rows = build_hr_rows(
        health_csv=repo_path(args.health_csv),
        labels_file=labels_file,
        source=args.source,
        time_offset_seconds=args.time_offset_seconds,
    )
    write_imported_rows(output, rows, source=args.source, replace_source=not args.append)
    print(f"Wrote {len(rows)} Apple Watch HR rows to {output}")


if __name__ == "__main__":
    main()
