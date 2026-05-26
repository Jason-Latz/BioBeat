from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from biobeat_paths import CLIPS_CSV, FAKE_BIOMETRICS_CSV, LABELS_DIR, ensure_project_dirs, repo_path  # noqa: E402


BIOMETRIC_COLUMNS = [
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

AROUSAL_PARAMS = {
    "low": {
        "hr_change": (2.0, 1.4, 0.0, 5.0),
        "eda_peaks": (1.0, 0, 3),
        "eda_amp": (0.06, 0.035, 0.01, 0.16),
    },
    "medium": {
        "hr_change": (5.5, 2.0, 2.0, 10.0),
        "eda_peaks": (2.6, 1, 5),
        "eda_amp": (0.15, 0.055, 0.04, 0.30),
    },
    "high": {
        "hr_change": (10.0, 3.0, 5.0, 17.0),
        "eda_peaks": (5.0, 3, 8),
        "eda_amp": (0.30, 0.09, 0.12, 0.55),
    },
}


def read_label_rows(labels_dir: Path) -> pd.DataFrame:
    paths = sorted(glob.glob(str(labels_dir / "*_labels.csv")))
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        return pd.DataFrame(columns=["user_id", "session_id", "clip_id"])
    labels = pd.concat(frames, ignore_index=True)
    return labels[["user_id", "session_id", "clip_id"]].drop_duplicates()


def baseline_rows(clips: pd.DataFrame, user_id: str, session_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": user_id,
            "session_id": session_id,
            "clip_id": clips["clip_id"],
        }
    )


def clipped_normal(rng: np.random.Generator, mean: float, sd: float, low: float, high: float) -> float:
    return float(np.clip(rng.normal(mean, sd), low, high))


def generate_row(
    rng: np.random.Generator,
    user_id: str,
    session_id: str,
    clip_id: str,
    intended_arousal: str,
) -> dict[str, object]:
    params = AROUSAL_PARAMS.get(str(intended_arousal).lower(), AROUSAL_PARAMS["medium"])
    hr_change = clipped_normal(rng, *params["hr_change"])
    user_baseline_hr = clipped_normal(
        rng,
        68 + (sum(ord(ch) for ch in user_id) % 9) - 4,
        2.5,
        56,
        82,
    )
    hr_mean = user_baseline_hr + hr_change * rng.uniform(0.45, 0.75)
    hr_max = hr_mean + max(1.0, hr_change * rng.uniform(0.45, 0.90)) + rng.normal(0, 1.2)

    eda_baseline = clipped_normal(rng, 1.1 + (len(session_id) % 5) * 0.08, 0.18, 0.45, 2.4)
    peak_lambda, peak_low, peak_high = params["eda_peaks"]
    eda_peak_count = int(np.clip(rng.poisson(peak_lambda), peak_low, peak_high))
    eda_amp = clipped_normal(rng, *params["eda_amp"])
    eda_change = max(0.0, eda_amp * rng.uniform(0.7, 1.4) + eda_peak_count * rng.uniform(0.015, 0.035))
    eda_mean = eda_baseline + eda_change * rng.uniform(0.35, 0.70)

    return {
        "user_id": user_id,
        "session_id": session_id,
        "clip_id": clip_id,
        "hr_mean": round(float(hr_mean), 3),
        "hr_max": round(float(max(hr_max, hr_mean)), 3),
        "hr_change_from_baseline": round(float(hr_change), 3),
        "hr_slope": round(float(hr_change / 30 + rng.normal(0, 0.025)), 4),
        "hr_recovery": round(float(-hr_change * rng.uniform(0.25, 0.65)), 3),
        "eda_mean": round(float(eda_mean), 4),
        "eda_change_from_baseline": round(float(eda_change), 4),
        "eda_peak_count": eda_peak_count,
        "eda_max_amplitude": round(float(eda_amp), 4),
        "eda_slope": round(float(eda_change / 30 + rng.normal(0, 0.003)), 5),
        "eda_recovery": round(float(-eda_change * rng.uniform(0.20, 0.55)), 4),
    }


def generate_fake_biometrics(
    clips_path: Path,
    output_path: Path,
    *,
    labels_dir: Path = LABELS_DIR,
    user_id: str = "demo_user",
    session_id: str = "demo_session",
    seed: int = 42,
) -> pd.DataFrame:
    ensure_project_dirs()
    clips = pd.read_csv(clips_path)
    label_rows = read_label_rows(labels_dir)
    if label_rows.empty:
        label_rows = baseline_rows(clips, user_id, session_id)

    base = label_rows.merge(
        clips[["clip_id", "intended_arousal"]],
        on="clip_id",
        how="left",
    )
    rng = np.random.default_rng(seed)
    rows = [
        generate_row(
            rng,
            str(row.user_id),
            str(row.session_id),
            str(row.clip_id),
            str(row.intended_arousal),
        )
        for row in base.itertuples(index=False)
    ]
    result = pd.DataFrame(rows).reindex(columns=BIOMETRIC_COLUMNS)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plausible fake HR/EDA features.")
    parser.add_argument("--clips", default=str(CLIPS_CSV))
    parser.add_argument("--output", default=str(FAKE_BIOMETRICS_CSV))
    parser.add_argument("--labels-dir", default=str(LABELS_DIR))
    parser.add_argument("--user-id", default="demo_user")
    parser.add_argument("--session-id", default="demo_session")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = generate_fake_biometrics(
        repo_path(args.clips),
        repo_path(args.output),
        labels_dir=repo_path(args.labels_dir),
        user_id=args.user_id,
        session_id=args.session_id,
        seed=args.seed,
    )
    print(f"Wrote {len(result)} fake biometric rows to {repo_path(args.output)}")


if __name__ == "__main__":
    main()
