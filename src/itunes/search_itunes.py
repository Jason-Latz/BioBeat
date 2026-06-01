from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))

from biobeat_paths import ensure_project_dirs  # noqa: E402


ITUNES_SEARCH_URL = "https://itunes.apple.com/search"
SEARCH_STOP_WORDS = {"a", "an", "and", "but", "for", "of", "or", "the", "to"}


def normalize_text(value: str | None) -> str:
    if not value:
        return ""
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def remove_search_stop_words(value: str) -> str:
    tokens = re.findall(r"[a-zA-Z0-9']+", value)
    return " ".join(token for token in tokens if token.lower() not in SEARCH_STOP_WORDS)


def simplify_search_term(value: str) -> str:
    value = value.replace("'", "").replace("’", "")
    tokens = re.findall(r"[a-zA-Z0-9]+", value)
    return " ".join(tokens)


def score_candidate(
    candidate: dict[str, Any], song_query: str, artist_query: str
) -> float:
    track_name = normalize_text(candidate.get("trackName"))
    artist_name = normalize_text(candidate.get("artistName"))
    song = normalize_text(song_query)
    artist = normalize_text(artist_query)

    score = 0.0
    if track_name == song:
        score += 60
    elif song and song in track_name:
        score += 35
    else:
        song_tokens = set(song.split())
        track_tokens = set(track_name.split())
        if song_tokens:
            score += 25 * (len(song_tokens & track_tokens) / len(song_tokens))

    if artist_name == artist:
        score += 30
    elif artist and artist in artist_name:
        score += 18
    else:
        artist_tokens = set(artist.split())
        candidate_tokens = set(artist_name.split())
        if artist_tokens:
            score += 15 * (len(artist_tokens & candidate_tokens) / len(artist_tokens))

    if candidate.get("previewUrl"):
        score += 10
    if candidate.get("trackExplicitness") == "notExplicit":
        score += 2

    return round(score, 3)


def search_itunes(
    song_query: str,
    artist_query: str = "",
    *,
    country: str = "US",
    limit: int = 5,
    timeout: int = 30,
    retry_delay: float = 0.0,
) -> list[dict[str, Any]]:
    term = " ".join(part for part in [song_query, artist_query] if part).strip()
    params = {
        "term": term,
        "media": "music",
        "entity": "song",
        "country": country,
        "limit": limit,
    }
    search_terms = [term]
    if artist_query.strip():
        first_artist_token = artist_query.split()[0]
        first_song_token = song_query.split()[0]
        simplified_song_query = simplify_search_term(song_query)
        compact_song_query = remove_search_stop_words(song_query)
        search_terms.extend(
            [
                song_query,
                f"{song_query} {first_artist_token}",
                simplified_song_query,
                f"{simplified_song_query} {first_artist_token}",
                compact_song_query,
                f"{compact_song_query} {first_artist_token}",
                f"{first_song_token} {first_artist_token}",
            ]
        )

    response = None
    last_error: requests.HTTPError | None = None
    for search_term in dict.fromkeys(search_terms):
        response = requests.get(
            ITUNES_SEARCH_URL,
            params={**params, "term": search_term},
            timeout=timeout,
        )
        try:
            response.raise_for_status()
            break
        except requests.HTTPError as exc:
            last_error = exc
            if response.status_code != 403:
                raise
            if retry_delay > 0:
                time.sleep(retry_delay)
    else:
        if last_error:
            raise last_error

    if response is None:
        return []
    payload = response.json()
    results = payload.get("results", [])

    candidates: list[dict[str, Any]] = []
    for rank, item in enumerate(results, start=1):
        if not item.get("previewUrl"):
            continue
        candidate = {
            "candidate_rank": rank,
            "match_score": score_candidate(item, song_query, artist_query),
            "track_name": item.get("trackName", ""),
            "artist": item.get("artistName", ""),
            "album": item.get("collectionName", ""),
            "genre": item.get("primaryGenreName", ""),
            "preview_url": item.get("previewUrl", ""),
            "track_id": item.get("trackId", ""),
            "collection_id": item.get("collectionId", ""),
            "release_date": item.get("releaseDate", ""),
            "track_time_millis": item.get("trackTimeMillis", ""),
            "track_explicitness": item.get("trackExplicitness", ""),
            "itunes_url": item.get("trackViewUrl", ""),
        }
        candidates.append(candidate)

    return sorted(candidates, key=lambda row: (-row["match_score"], row["candidate_rank"]))


def write_candidates(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search iTunes preview candidates.")
    parser.add_argument("song_query")
    parser.add_argument("artist_query", nargs="?", default="")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    ensure_project_dirs()
    candidates = search_itunes(args.song_query, args.artist_query, limit=args.limit)
    if not candidates:
        print("No preview candidates found.")
        raise SystemExit(1)

    for idx, row in enumerate(candidates, start=1):
        print(
            f"{idx}. {row['track_name']} - {row['artist']} "
            f"({row['album']}) score={row['match_score']} track_id={row['track_id']}"
        )
        print(f"   {row['preview_url']}")


if __name__ == "__main__":
    main()
