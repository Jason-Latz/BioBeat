from __future__ import annotations

import os

from biobeat_paths import repo_path
from dashboard import app as dashboard_app


def main() -> None:
    clips_path = os.environ.get("BIOBEAT_CLIPS_CSV", "").strip()
    if not clips_path:
        raise RuntimeError("Set BIOBEAT_CLIPS_CSV to the collection batch CSV.")

    dashboard_app.CLIPS_CSV = repo_path(clips_path)
    dashboard_app.main()


if __name__ == "__main__":
    main()
