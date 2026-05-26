from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from biobeat_paths import PROCESSED_DIR, RAW_DIR, ensure_project_dirs, repo_path  # noqa: E402


OUTPUT_COLUMNS = [
    "user_id",
    "session_id",
    "clip_id",
    "hr_mean",
    "hr_max",
    "hr_change_from_baseline",
    "hr_slope",
    "hr_recovery",
    "eda_mean",
    "eda_change_from_baseline",
    "eda_peak_count",
    "eda_max_amplitude",
    "eda_slope",
    "eda_recovery",
]


def read_raw_sensor_files(sensor_dir: Path) -> pd.DataFrame:
    paths = sorted(glob.glob(str(sensor_dir / "*_sensor.csv")))
    if not paths:
        raise FileNotFoundError(f"No raw sensor files found in {sensor_dir}")
    frames = [pd.read_csv(path) for path in paths]
    data = pd.concat(frames, ignore_index=True)
    required = {"user_id", "session_id", "clip_id", "trial_index", "phase", "hr", "eda"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Raw sensor files are missing columns: {', '.join(sorted(missing))}")
    data["hr"] = pd.to_numeric(data["hr"], errors="coerce")
    data["eda"] = pd.to_numeric(data["eda"], errors="coerce")
    return data


def slope(values: pd.Series) -> float:
    values = values.dropna().astype(float)
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    return float(np.polyfit(x, values.to_numpy(), 1)[0])


def peak_count(values: pd.Series, baseline: float) -> int:
    values = values.dropna().astype(float).to_numpy()
    if len(values) < 3:
        return 0
    threshold = baseline + max(float(np.std(values)) * 0.5, 0.02)
    peaks = 0
    for index in range(1, len(values) - 1):
        if values[index] > threshold and values[index] >= values[index - 1] and values[index] >= values[index + 1]:
            peaks += 1
    return peaks


def summarize_group(group: pd.DataFrame) -> dict[str, object]:
    rest = group[group["phase"].astype(str) == "rest"]
    listen = group[group["phase"].astype(str) == "listen"]
    if listen.empty:
        listen = group

    hr_rest_mean = float(rest["hr"].mean()) if rest["hr"].notna().any() else float(listen["hr"].mean())
    hr_mean = float(listen["hr"].mean()) if listen["hr"].notna().any() else np.nan
    hr_max = float(listen["hr"].max()) if listen["hr"].notna().any() else np.nan

    eda_rest_mean = float(rest["eda"].mean()) if rest["eda"].notna().any() else float(listen["eda"].mean())
    eda_mean = float(listen["eda"].mean()) if listen["eda"].notna().any() else np.nan
    eda_max = float(listen["eda"].max()) if listen["eda"].notna().any() else np.nan

    hr_change = hr_mean - hr_rest_mean if not np.isnan(hr_mean) and not np.isnan(hr_rest_mean) else np.nan
    eda_change = eda_mean - eda_rest_mean if not np.isnan(eda_mean) and not np.isnan(eda_rest_mean) else np.nan

    hr_values = listen["hr"].dropna().astype(float)
    eda_values = listen["eda"].dropna().astype(float)

    return {
        "user_id": group["user_id"].iloc[0],
        "session_id": group["session_id"].iloc[0],
        "clip_id": group["clip_id"].iloc[0],
        "hr_mean": round(hr_mean, 3) if not np.isnan(hr_mean) else "",
        "hr_max": round(hr_max, 3) if not np.isnan(hr_max) else "",
        "hr_change_from_baseline": round(hr_change, 3) if not np.isnan(hr_change) else "",
        "hr_slope": round(slope(hr_values), 5),
        "hr_recovery": round(float(hr_values.iloc[-1] - hr_values.max()), 3) if len(hr_values) else "",
        "eda_mean": round(eda_mean, 5) if not np.isnan(eda_mean) else "",
        "eda_change_from_baseline": round(eda_change, 5) if not np.isnan(eda_change) else "",
        "eda_peak_count": peak_count(eda_values, eda_rest_mean) if not np.isnan(eda_rest_mean) else 0,
        "eda_max_amplitude": round(max(0.0, eda_max - eda_rest_mean), 5) if not np.isnan(eda_max) and not np.isnan(eda_rest_mean) else "",
        "eda_slope": round(slope(eda_values), 6),
        "eda_recovery": round(float(eda_values.iloc[-1] - eda_values.max()), 5) if len(eda_values) else "",
    }


def extract_biometric_features(sensor_dir: Path, output_path: Path) -> pd.DataFrame:
    ensure_project_dirs()
    data = read_raw_sensor_files(sensor_dir)
    rows = [
        summarize_group(group)
        for _, group in data.groupby(["user_id", "session_id", "clip_id"], sort=False)
    ]
    result = pd.DataFrame(rows).reindex(columns=OUTPUT_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract real biometric features from raw dashboard sensor CSVs.")
    parser.add_argument("--sensor-dir", default=str(RAW_DIR / "sensor"))
    parser.add_argument("--output", default=str(PROCESSED_DIR / "biometric_features_real.csv"))
    args = parser.parse_args()

    result = extract_biometric_features(repo_path(args.sensor_dir), repo_path(args.output))
    print(f"Wrote {len(result)} real biometric feature rows to {repo_path(args.output)}")


if __name__ == "__main__":
    main()
