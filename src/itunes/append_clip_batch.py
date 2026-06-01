from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from biobeat_paths import CLIPS_CSV, repo_path  # noqa: E402


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def duplicate_values(rows: list[dict[str, str]], field: str) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in rows:
        value = row.get(field, "").strip()
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def append_batch(master_path: Path, batch_path: Path) -> list[dict[str, str]]:
    master_rows = read_rows(master_path)
    batch_rows = read_rows(batch_path)
    rows = master_rows + batch_rows

    if not batch_rows:
        raise ValueError(f"No clips found in {batch_path}.")

    for field in ("clip_id", "track_id"):
        duplicates = duplicate_values(rows, field)
        if duplicates:
            raise ValueError(f"Duplicate {field} values: {', '.join(duplicates)}")

    fieldnames = list(master_rows[0]) if master_rows else list(batch_rows[0])
    if not fieldnames:
        raise ValueError("Clip CSV header is missing.")

    unknown_fields = sorted({field for row in batch_rows for field in row} - set(fieldnames))
    if unknown_fields:
        raise ValueError(f"Batch has unexpected fields: {', '.join(unknown_fields)}")

    temp_path = master_path.with_suffix(f"{master_path.suffix}.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(master_path)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Append a validated continuation clip batch to the master registry.")
    parser.add_argument("--master", default=str(CLIPS_CSV))
    parser.add_argument("--batch", required=True)
    args = parser.parse_args()

    rows = append_batch(repo_path(args.master), repo_path(args.batch))
    print(f"Wrote {len(rows)} master clips to {repo_path(args.master)}")


if __name__ == "__main__":
    main()
