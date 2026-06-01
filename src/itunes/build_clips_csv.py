from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))

from biobeat_paths import CLIPS_CSV, DESIRED_SONGS_CSV, INPUT_DIR, ensure_project_dirs, repo_path  # noqa: E402
from search_itunes import normalize_text, search_itunes, write_candidates  # noqa: E402


CANDIDATES_CSV = INPUT_DIR / "itunes_candidates.csv"
OVERRIDES_CSV = INPUT_DIR / "clip_overrides.csv"
OVERRIDES_TEMPLATE_CSV = INPUT_DIR / "clip_overrides_template.csv"
SELECTION_PENALTY_TERMS = {
    "a cappella": 20,
    "acappella": 20,
    "instrumental": 20,
    "karaoke": 20,
    "lullaby": 12,
    "mixed": 8,
    "remix": 8,
    "tribute": 20,
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
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


def desired_key(row: dict[str, str]) -> tuple[str, str]:
    return (row.get("song_query", "").strip(), row.get("artist_query", "").strip())


def load_candidate_cache(path: Path) -> dict[tuple[str, str], list[dict[str, Any]]]:
    if not path.exists():
        return {}

    cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in read_csv_rows(path):
        key = desired_key(row)
        if not key[0]:
            continue
        cache.setdefault(key, []).append(row)

    def sort_key(row: dict[str, Any]) -> tuple[float, int]:
        try:
            match_score = float(row.get("match_score", 0))
        except (TypeError, ValueError):
            match_score = 0.0
        try:
            candidate_rank = int(row.get("candidate_rank", 9999))
        except (TypeError, ValueError):
            candidate_rank = 9999
        return (-match_score, candidate_rank)

    for rows in cache.values():
        rows.sort(key=sort_key)
    return cache


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


def candidate_match_score(candidate: dict[str, Any]) -> float:
    try:
        return float(candidate.get("match_score", 0))
    except (TypeError, ValueError):
        return 0.0


def candidate_rank(candidate: dict[str, Any]) -> int:
    try:
        return int(candidate.get("candidate_rank", 9999))
    except (TypeError, ValueError):
        return 9999


def selection_score(candidate: dict[str, Any], desired_row: dict[str, str]) -> float:
    score = candidate_match_score(candidate)
    desired_artist = normalize_text(desired_row.get("artist_query"))
    candidate_artist = normalize_text(candidate.get("artist"))
    candidate_title = normalize_text(candidate.get("track_name"))

    if desired_artist and (candidate_artist == desired_artist or desired_artist in candidate_artist):
        score += 15

    for term, penalty in SELECTION_PENALTY_TERMS.items():
        if term in candidate_title:
            score -= penalty

    return score


def rank_candidates(
    candidates: list[dict[str, Any]], desired_row: dict[str, str]
) -> list[dict[str, Any]]:
    return sorted(
        candidates,
        key=lambda candidate: (
            -selection_score(candidate, desired_row),
            -candidate_match_score(candidate),
            candidate_rank(candidate),
        ),
    )


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

    candidates = rank_candidates(candidates, desired_row)
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
    skip_search_errors: bool = False,
    use_candidate_cache: bool = True,
    request_delay: float = 3.2,
    candidates_path: Path = CANDIDATES_CSV,
    overrides_path: Path = OVERRIDES_CSV,
    overrides_template_path: Path = OVERRIDES_TEMPLATE_CSV,
    clip_id_start: int = 1,
) -> list[dict[str, Any]]:
    ensure_project_dirs()
    desired_rows = read_csv_rows(input_path)
    overrides = load_overrides(overrides_path)
    candidate_cache = load_candidate_cache(candidates_path) if use_candidate_cache else {}

    override_template_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []

    for desired_index, desired_row in enumerate(desired_rows, start=1):
        song_query = desired_row.get("song_query", "").strip()
        artist_query = desired_row.get("artist_query", "").strip()
        if not song_query:
            continue

        cache_key = desired_key(desired_row)
        candidates = candidate_cache.get(cache_key)
        if candidates:
            candidates = [dict(candidate) for candidate in candidates]
        else:
            try:
                candidates = search_itunes(
                    song_query,
                    artist_query,
                    limit=limit,
                    retry_delay=request_delay,
                )
            except requests.RequestException as exc:
                if not skip_search_errors:
                    raise
                print(
                    f"Skipping {song_query} - {artist_query}; "
                    f"iTunes search failed: {exc}"
                )
                if request_delay > 0:
                    time.sleep(request_delay)
                continue
            if request_delay > 0:
                time.sleep(request_delay)

            candidate_cache[cache_key] = [
                {
                    **desired_row,
                    **candidate,
                }
                for candidate in candidates
            ]
            write_candidates(
                candidates_path,
                [
                    candidate
                    for cache_rows in candidate_cache.values()
                    for candidate in cache_rows
                ],
            )

        for candidate in candidates:
            candidate.update(desired_row)

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

        clip_id = f"clip_{clip_id_start + len(selected_rows):03d}"
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
    write_csv(
        overrides_template_path,
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
    parser.add_argument(
        "--no-candidate-cache",
        action="store_true",
        help="Ignore cached candidates and query iTunes for every desired song.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=3.2,
        help="Seconds to wait between uncached iTunes searches.",
    )
    parser.add_argument(
        "--skip-search-errors",
        action="store_true",
        help="Continue building clips when individual iTunes searches fail.",
    )
    parser.add_argument(
        "--candidates",
        default=str(CANDIDATES_CSV),
        help="Candidate cache CSV. Successful uncached searches are written immediately.",
    )
    parser.add_argument(
        "--overrides",
        default=str(OVERRIDES_CSV),
        help="Optional clip override CSV.",
    )
    parser.add_argument(
        "--override-template",
        default=str(OVERRIDES_TEMPLATE_CSV),
        help="Output path for the generated override template CSV.",
    )
    parser.add_argument(
        "--clip-id-start",
        type=int,
        default=1,
        help="Numeric ID for the first generated clip.",
    )
    args = parser.parse_args()

    rows = build_clips(
        repo_path(args.input),
        repo_path(args.output),
        interactive=args.interactive,
        limit=args.limit,
        skip_search_errors=args.skip_search_errors,
        use_candidate_cache=not args.no_candidate_cache,
        request_delay=args.request_delay,
        candidates_path=repo_path(args.candidates),
        overrides_path=repo_path(args.overrides),
        overrides_template_path=repo_path(args.override_template),
        clip_id_start=args.clip_id_start,
    )
    print(f"Wrote {len(rows)} selected clips to {repo_path(args.output)}")
    print(f"Wrote iTunes candidates to {repo_path(args.candidates)}")
    print(
        f"Edit {repo_path(args.override_template)}, save as {repo_path(args.overrides)}, "
        "and rerun to override selections."
    )


if __name__ == "__main__":
    main()
