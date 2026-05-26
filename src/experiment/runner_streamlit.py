from __future__ import annotations

import csv
import random
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from biobeat_paths import CLIPS_CSV, LABELS_DIR, ensure_project_dirs  # noqa: E402


LABEL_COLUMNS = [
    "user_id",
    "session_id",
    "clip_id",
    "trial_index",
    "clip_start_time",
    "clip_end_time",
    "preference",
    "arousal",
    "valence",
    "mood",
    "familiarity",
    "notes",
]

MOOD_OPTIONS = [
    "happy",
    "sad",
    "relaxed",
    "tense",
    "excited",
    "annoyed",
    "neutral",
    "other",
]


def iso_now() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


@st.cache_data
def load_clips(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist. Run src/itunes/build_clips_csv.py first.")
    clips = pd.read_csv(path)
    required = {"clip_id", "track_name", "artist", "preview_url"}
    missing = required - set(clips.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    return clips


def label_path(session_id: str) -> Path:
    safe_session = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in session_id)
    return LABELS_DIR / f"{safe_session}_labels.csv"


def append_label(row: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def initialize_session(clips: pd.DataFrame, user_id: str, session_id: str) -> None:
    order = clips["clip_id"].tolist()
    random.Random(session_id).shuffle(order)
    st.session_state.started = True
    st.session_state.user_id = user_id
    st.session_state.session_id = session_id
    st.session_state.order = order
    st.session_state.trial_position = 0
    st.session_state.clip_start_time = None
    st.session_state.trial_active = False


def current_clip(clips: pd.DataFrame) -> pd.Series:
    clip_id = st.session_state.order[st.session_state.trial_position]
    return clips.loc[clips["clip_id"] == clip_id].iloc[0]


def advance_trial() -> None:
    st.session_state.trial_position += 1
    st.session_state.clip_start_time = None
    st.session_state.trial_active = False


def main() -> None:
    ensure_project_dirs()
    st.set_page_config(page_title="BioBeat Experiment", page_icon="BB", layout="centered")
    st.title("BioBeat Experiment")

    try:
        clips = load_clips(CLIPS_CSV)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.header("Session")
        default_session = datetime.now().strftime("s%Y%m%d_%H%M%S")
        user_id = st.text_input("User ID", value=st.session_state.get("user_id", ""))
        session_id = st.text_input("Session ID", value=st.session_state.get("session_id", default_session))
        start_disabled = not user_id.strip() or not session_id.strip()
        if st.button("Start", disabled=start_disabled, width="stretch"):
            initialize_session(clips, user_id.strip(), session_id.strip())

        if st.session_state.get("started"):
            st.caption(f"Saving to `{label_path(st.session_state.session_id).relative_to(Path.cwd())}`")

    if not st.session_state.get("started"):
        st.info("Enter a user ID and session ID to begin.")
        st.stop()

    total = len(st.session_state.order)
    position = st.session_state.trial_position
    if position >= total:
        st.success("Session complete.")
        st.write(f"Saved labels to `{label_path(st.session_state.session_id)}`")
        if st.button("Start another randomized pass"):
            initialize_session(clips, st.session_state.user_id, st.session_state.session_id)
        st.stop()

    clip = current_clip(clips)
    st.subheader(f"Clip {position + 1} of {total}")
    st.write(f"**{clip['track_name']}**")
    st.write(str(clip["artist"]))

    if st.button("Start clip", disabled=st.session_state.trial_active, width="stretch"):
        st.session_state.clip_start_time = iso_now()
        st.session_state.trial_active = True

    if st.session_state.trial_active:
        st.audio(str(clip["preview_url"]))
        st.caption(f"Clip started at {st.session_state.clip_start_time}")
    else:
        st.audio(str(clip["preview_url"]))

    with st.form("ratings_form", clear_on_submit=True):
        preference = st.slider("Preference", 1, 5, 3, help="1 = dislike, 5 = like")
        arousal = st.slider("Arousal", 1, 5, 3, help="1 = calm, 5 = energized")
        valence = st.slider("Valence", 1, 5, 3, help="1 = negative, 5 = positive")
        mood = st.selectbox("Mood", MOOD_OPTIONS)
        other_mood = ""
        if mood == "other":
            other_mood = st.text_input("Mood detail")
        familiarity = st.slider("Familiarity", 1, 5, 3, help="1 = unfamiliar, 5 = very familiar")
        notes = st.text_area("Notes", height=80)
        submitted = st.form_submit_button("Save rating")

    if submitted:
        clip_start = st.session_state.clip_start_time or iso_now()
        row = {
            "user_id": st.session_state.user_id,
            "session_id": st.session_state.session_id,
            "clip_id": clip["clip_id"],
            "trial_index": position + 1,
            "clip_start_time": clip_start,
            "clip_end_time": iso_now(),
            "preference": preference,
            "arousal": arousal,
            "valence": valence,
            "mood": other_mood.strip() if mood == "other" and other_mood.strip() else mood,
            "familiarity": familiarity,
            "notes": notes.strip(),
        }
        append_label(row, label_path(st.session_state.session_id))
        advance_trial()
        st.rerun()


if __name__ == "__main__":
    main()
