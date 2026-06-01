from __future__ import annotations

import argparse
import csv
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from biobeat_paths import CLIPS_CSV, LABELS_DIR, ensure_project_dirs, repo_path  # noqa: E402


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

AROUSAL_BASE = {"low": 2, "medium": 3, "high": 5}
VALENCE_BASE = {"negative": 2, "neutral": 3, "positive": 4}
MOOD_BY_LABEL = {
    ("high", "positive"): ["excited", "happy"],
    ("medium", "positive"): ["happy", "relaxed", "neutral"],
    ("low", "positive"): ["relaxed", "neutral"],
    ("low", "negative"): ["sad", "relaxed"],
    ("medium", "negative"): ["tense", "annoyed", "sad"],
    ("high", "negative"): ["tense", "annoyed"],
}


def clamp_rating(value: float) -> int:
    return int(max(1, min(5, round(value))))


def demo_rating(
    rng: random.Random,
    base: int,
    *,
    user_bias: float = 0.0,
    noise: float = 0.8,
) -> int:
    return clamp_rating(base + user_bias + rng.gauss(0, noise))


def demo_valence_label(value: int) -> str:
    if value <= 2:
        return "negative"
    if value >= 4:
        return "positive"
    return "neutral"


def generate_rows(clips: pd.DataFrame, user_id: str, session_id: str, seed: int) -> list[dict[str, object]]:
    rng = random.Random(f"{seed}:{user_id}:{session_id}")
    order = clips.sample(frac=1, random_state=seed + len(user_id) + len(session_id)).reset_index(drop=True)
    start = datetime.now().replace(microsecond=0) - timedelta(days=1)
    user_preference_bias = rng.uniform(-0.4, 0.5)

    rows: list[dict[str, object]] = []
    for index, clip in order.iterrows():
        rest_start = start + timedelta(minutes=index * 2)
        rest_end = rest_start + timedelta(seconds=30, milliseconds=rng.randint(0, 300))
        trial_start = rest_end + timedelta(seconds=2)
        trial_end = trial_start + timedelta(seconds=30, milliseconds=rng.randint(0, 700))
        intended_arousal = str(clip.get("intended_arousal", "medium")).lower()
        intended_valence = str(clip.get("intended_valence", "neutral")).lower()
        intended_mood = str(clip.get("intended_mood", "neutral")).lower()

        arousal = demo_rating(rng, AROUSAL_BASE.get(intended_arousal, 3), noise=0.7)
        valence_rating = demo_rating(rng, VALENCE_BASE.get(intended_valence, 3), noise=0.8)
        valence = demo_valence_label(valence_rating)
        familiarity = rng.randint(1, 5)
        preference_base = 3 + (0.45 if intended_valence == "positive" else -0.25) + (0.15 * (familiarity - 3))
        preference = demo_rating(rng, preference_base, user_bias=user_preference_bias, noise=0.9)
        mood_options = MOOD_BY_LABEL.get((intended_arousal, intended_valence), [intended_mood, "neutral"])

        rows.append(
            {
                "user_id": user_id,
                "session_id": session_id,
                "clip_id": clip["clip_id"],
                "trial_index": index + 1,
                "rest_start_time": rest_start.isoformat(timespec="milliseconds"),
                "rest_end_time": rest_end.isoformat(timespec="milliseconds"),
                "clip_start_time": trial_start.isoformat(timespec="milliseconds"),
                "clip_end_time": trial_end.isoformat(timespec="milliseconds"),
                "preference": preference,
                "arousal": arousal,
                "valence": valence,
                "mood": rng.choice(mood_options),
                "familiarity": familiarity,
                "notes": "",
            }
        )
    return rows


def write_labels(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plausible demo labels for local model testing.")
    parser.add_argument("--clips", default=str(CLIPS_CSV))
    parser.add_argument("--participants", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    ensure_project_dirs()
    clips = pd.read_csv(repo_path(args.clips))
    if clips.empty:
        raise SystemExit("No clips found. Run src/itunes/build_clips_csv.py first.")

    total = 0
    for participant_index in range(1, args.participants + 1):
        user_id = "demo_user" if participant_index == 1 else f"demo_user_{participant_index:02d}"
        session_id = "demo_session" if participant_index == 1 else f"demo_session_{participant_index:02d}"
        rows = generate_rows(clips, user_id, session_id, args.seed + participant_index)
        write_labels(LABELS_DIR / f"{session_id}_labels.csv", rows)
        total += len(rows)
        print(f"Wrote {len(rows)} labels for {user_id}/{session_id}")

    print(f"Wrote {total} demo label rows to {LABELS_DIR}")


if __name__ == "__main__":
    main()
