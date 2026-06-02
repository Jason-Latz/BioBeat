from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from biobeat_paths import AUDIO_FEATURES_CSV, CLIPS_CSV, PROCESSED_DIR  # noqa: E402
from models import recommend as recommender  # noqa: E402


CALIBRATION_PROFILES_CSV = PROCESSED_DIR / "user_calibration_profiles.csv"
DEFAULT_PLAYLIST_SIZE = 10


MOOD_LABELS = {
    "relaxed": "Relaxed",
    "happy": "Happy",
    "excited": "Excited",
    "hype": "Hype",
    "calm": "Calm",
    "neutral": "Neutral",
    "alert": "Alert",
    "sad": "Sad",
    "tense": "Tense",
    "angry": "Angry",
}


def full_width_dataframe(data: pd.DataFrame, **kwargs: object) -> None:
    if "width" in inspect.signature(st.dataframe).parameters:
        st.dataframe(data, width="stretch", **kwargs)
    else:
        st.dataframe(data, use_container_width=True, **kwargs)


@st.cache_data
def load_profiles(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    profiles = pd.read_csv(path)
    if "user_id" not in profiles.columns:
        return pd.DataFrame()
    profiles = profiles.dropna(subset=["user_id"]).copy()
    profiles["user_id"] = profiles["user_id"].astype(str)
    return profiles.sort_values("user_id")


@st.cache_data
def load_clips(path: Path) -> pd.DataFrame:
    clips = pd.read_csv(path)
    keep_columns = [
        column
        for column in ("clip_id", "track_name", "artist", "album", "genre", "preview_url")
        if column in clips.columns
    ]
    return clips[keep_columns].copy()


@st.cache_data
def ranked_playlist(user_id: str, target_mood: str) -> pd.DataFrame:
    ranked = recommender.rank_recommendations(
        CLIPS_CSV,
        AUDIO_FEATURES_CSV,
        user_id=user_id,
        target_mood=target_mood,
        calibration_profiles_path=CALIBRATION_PROFILES_CSV,
    )
    clips = load_clips(CLIPS_CSV)
    extra_columns = [column for column in ("clip_id", "album", "genre", "preview_url") if column in clips.columns]
    display = ranked.merge(clips[extra_columns], on="clip_id", how="left")
    display["score"] = pd.to_numeric(display["score"], errors="coerce").round(3)
    display["predicted_arousal"] = pd.to_numeric(display["predicted_arousal"], errors="coerce").round(3)
    return display


def profile_summary(profiles: pd.DataFrame, user_id: str) -> pd.Series:
    rows = profiles[profiles["user_id"].astype(str) == str(user_id)]
    return rows.iloc[-1] if not rows.empty else pd.Series(dtype=object)


def render_playlist(rows: pd.DataFrame) -> None:
    table_columns = [
        column
        for column in ("rank", "track_name", "artist", "genre", "score", "predicted_mood", "predicted_arousal")
        if column in rows.columns
    ]
    full_width_dataframe(rows[table_columns], hide_index=True)

    for _, row in rows.iterrows():
        title = str(row.get("track_name", "Unknown track"))
        artist = str(row.get("artist", "Unknown artist"))
        detail_parts = [
            str(row.get("genre"))
            for column in ("genre",)
            if column in row and pd.notna(row.get(column)) and str(row.get(column)).strip()
        ]
        detail_parts.append(f"score {float(row.get('score', 0.0)):.3f}")
        with st.container(border=True):
            st.markdown(f"**{int(row['rank'])}. {title}**")
            st.caption(f"{artist} | {' | '.join(detail_parts)}")
            preview_url = row.get("preview_url", "")
            if isinstance(preview_url, str) and preview_url.strip():
                st.audio(preview_url)


def main() -> None:
    st.set_page_config(page_title="BioBeat Playlist", page_icon="music", layout="wide")
    st.title("BioBeat Playlist")

    profiles = load_profiles(CALIBRATION_PROFILES_CSV)
    if profiles.empty:
        st.error("No calibrated users found. Run the calibration step before opening this app.")
        st.code("PYTHONPATH=src .venv/bin/python src/models/calibrate_user.py --user-id USER_ID --max-clips 5")
        return

    user_ids = profiles["user_id"].drop_duplicates().tolist()
    mood_keys = [key for key in MOOD_LABELS if key in recommender.MOOD_TARGETS]

    user_col, mood_col, size_col = st.columns([1.2, 1.2, 0.8])
    with user_col:
        user_id = st.selectbox("User", user_ids)
    with mood_col:
        target_mood = st.selectbox(
            "Desired mood",
            mood_keys,
            format_func=lambda key: MOOD_LABELS.get(key, key.title()),
        )
    with size_col:
        playlist_size = st.number_input(
            "Songs",
            min_value=3,
            max_value=50,
            value=DEFAULT_PLAYLIST_SIZE,
            step=1,
        )

    profile = profile_summary(profiles, user_id)
    calibration_count = profile.get("calibration_clip_count", "")
    session_id = profile.get("session_id", "")
    st.caption(f"Calibrated profile: {user_id} | session {session_id or 'not specified'} | {calibration_count} clips")

    try:
        playlist = ranked_playlist(user_id, target_mood).head(int(playlist_size))
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except ValueError as exc:
        st.error(str(exc))
        return

    st.subheader(f"{MOOD_LABELS.get(target_mood, target_mood.title())} Playlist")
    render_playlist(playlist)

    csv = playlist.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download playlist CSV",
        csv,
        file_name=f"biobeat_{user_id}_{target_mood}_playlist.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
