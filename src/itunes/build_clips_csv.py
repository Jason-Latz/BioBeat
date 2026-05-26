from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from biobeat_paths import CLIPS_CSV, DESIRED_SONGS_CSV, INPUT_DIR, ensure_project_dirs, repo_path  # noqa: E402
from search_itunes import search_itunes, write_candidates  # noqa: E402


CANDIDATES_CSV = INPUT_DIR / "itunes_candidates.csv"
OVERRIDES_CSV = INPUT_DIR / "clip_overrides.csv"
OVERRIDES_TEMPLATE_CSV = INPUT_DIR / "clip_overrides_template.csv"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_overrides(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    rows = read_csv_rows(path)
    overrides: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row.get("song_query", "").strip(), row.get("artist_query", "").strip())
        if any(row.get(field, "").strip() for field in ("selected_track_id", "manual_preview_url")):
            overrides[key] = row
    return overrides


def manual_override_to_clip(row: dict[str, str]) -> dict[str, Any]:
    return {
        "track_name": row.get("manual_track_name", "").strip(),
        "artist": row.get("manual_artist", "").strip(),
        "album": row.get("manual_album", "").strip(),
        "genre": row.get("manual_genre", "").strip(),
        "preview_url": row.get("manual_preview_url", "").strip(),
        "track_id": row.get("selected_track_id", "").strip(),
        "selection_method": "manual_url_override",
        "match_score": "",
    }


def choose_candidate(
    desired_row: dict[str, str],
    candidates: list[dict[str, Any]],
    override: dict[str, str] | None,
    *,
    interactive: bool,
) -> dict[str, Any] | None:
    if override:
        manual_url = override.get("manual_preview_url", "").strip()
        if manual_url:
            return manual_override_to_clip(override)

        selected_track_id = override.get("selected_track_id", "").strip()
        if selected_track_id:
            for candidate in candidates:
                if str(candidate.get("track_id", "")) == selected_track_id:
                    candidate = dict(candidate)
                    candidate["selection_method"] = "track_id_override"
                    return candidate
            print(
                f"Override track_id={selected_track_id} was not found for "
                f"{desired_row['song_query']} - {desired_row['artist_query']}."
            )

    if not candidates:
        return None

    if interactive:
        print(f"\n{desired_row['song_query']} - {desired_row['artist_query']}")
        for index, candidate in enumerate(candidates, start=1):
            print(
                f"{index}. {candidate['track_name']} - {candidate['artist']} "
                f"({candidate['album']}) score={candidate['match_score']} "
                f"track_id={candidate['track_id']}"
            )
        answer = input("Select candidate number, blank for best score, or s to skip: ").strip()
        if answer.lower() == "s":
            return None
        if answer:
            selected_index = int(answer) - 1
            selected = dict(candidates[selected_index])
            selected["selection_method"] = "interactive"
            return selected

    selected = dict(candidates[0])
    selected["selection_method"] = "best_score"
    return selected


def build_clips(
    input_path: Path,
    output_path: Path,
    *,
    interactive: bool = False,
    limit: int = 5,
) -> list[dict[str, Any]]:
    ensure_project_dirs()
    desired_rows = read_csv_rows(input_path)
    overrides = load_overrides(OVERRIDES_CSV)

    all_candidates: list[dict[str, Any]] = []
    override_template_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for desired_index, desired_row in enumerate(desired_rows, start=1):
        song_query = desired_row.get("song_query", "").strip()
        artist_query = desired_row.get("artist_query", "").strip()
        if not song_query:
            continue

        candidates = search_itunes(song_query, artist_query, limit=limit)
        for candidate in candidates:
            all_candidates.append(
                {
                    **desired_row,
                    **candidate,
                }
            )

        override = overrides.get((song_query, artist_query))
        selected = choose_candidate(
            desired_row,
            candidates,
            override,
            interactive=interactive,
        )
        if not selected:
            print(f"Skipping {song_query} - {artist_query}; no selected preview URL.")
            continue

        clip_id = f"clip_{len(selected_rows) + 1:03d}"
        selected_rows.append(
            {
                "clip_id": clip_id,
                "track_name": selected.get("track_name", ""),
                "artist": selected.get("artist", ""),
                "album": selected.get("album", ""),
                "genre": selected.get("genre", ""),
                "preview_url": selected.get("preview_url", ""),
                "track_id": selected.get("track_id", ""),
                "intended_arousal": desired_row.get("intended_arousal", ""),
                "intended_valence": desired_row.get("intended_valence", ""),
                "intended_mood": desired_row.get("intended_mood", ""),
                "song_query": song_query,
                "artist_query": artist_query,
                "selection_method": selected.get("selection_method", ""),
                "match_score": selected.get("match_score", ""),
            }
        )

        best = candidates[0] if candidates else {}
        override_template_rows.append(
            {
                "song_query": song_query,
                "artist_query": artist_query,
                "selected_track_id": selected.get("track_id", ""),
                "manual_track_name": "",
                "manual_artist": "",
                "manual_album": "",
                "manual_genre": "",
                "manual_preview_url": "",
                "current_selection": f"{selected.get('track_name', '')} - {selected.get('artist', '')}",
                "best_scored_track_id": best.get("track_id", ""),
                "best_scored_candidate": f"{best.get('track_name', '')} - {best.get('artist', '')}",
            }
        )
        print(
            f"{desired_index:02d}. selected {clip_id}: "
            f"{selected.get('track_name', '')} - {selected.get('artist', '')}"
        )

    clip_fields = [
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
        "song_query",
        "artist_query",
        "selection_method",
        "match_score",
    ]
    write_csv(output_path, selected_rows, clip_fields)

    candidate_fields = [
        "song_query",
        "artist_query",
        "intended_arousal",
        "intended_valence",
        "intended_mood",
        "candidate_rank",
        "match_score",
        "track_name",
        "artist",
        "album",
        "genre",
        "preview_url",
        "track_id",
        "collection_id",
        "release_date",
        "track_time_millis",
        "track_explicitness",
        "itunes_url",
    ]
    write_candidates(CANDIDATES_CSV, all_candidates)
    write_csv(
        OVERRIDES_TEMPLATE_CSV,
        override_template_rows,
        [
            "song_query",
            "artist_query",
            "selected_track_id",
            "manual_track_name",
            "manual_artist",
            "manual_album",
            "manual_genre",
            "manual_preview_url",
            "current_selection",
            "best_scored_track_id",
            "best_scored_candidate",
        ],
    )

    return selected_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/clips.csv from iTunes previews.")
    parser.add_argument("--input", default=str(DESIRED_SONGS_CSV))
    parser.add_argument("--output", default=str(CLIPS_CSV))
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    rows = build_clips(
        repo_path(args.input),
        repo_path(args.output),
        interactive=args.interactive,
        limit=args.limit,
    )
    print(f"Wrote {len(rows)} selected clips to {repo_path(args.output)}")
    print(f"Wrote iTunes candidates to {CANDIDATES_CSV}")
    print(f"Edit {OVERRIDES_TEMPLATE_CSV.name}, save as {OVERRIDES_CSV.name}, and rerun to override selections.")


if __name__ == "__main__":
    main()
