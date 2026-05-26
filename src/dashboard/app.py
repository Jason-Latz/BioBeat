from __future__ import annotations

import csv
import glob
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from biobeat_paths import (  # noqa: E402
    CLIPS_CSV,
    FAKE_BIOMETRICS_CSV,
    LABELS_DIR,
    RECOMMENDATIONS_CSV,
    ensure_project_dirs,
)


REST_SECONDS = 30
LISTEN_SECONDS = 30
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


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_labels(labels_dir: Path) -> pd.DataFrame:
    paths = sorted(glob.glob(str(labels_dir / "*_labels.csv")))
    frames = [pd.read_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=LABEL_COLUMNS)


@st.cache_data
def load_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clips = read_csv_if_exists(CLIPS_CSV)
    labels = load_labels(LABELS_DIR)
    biometrics = read_csv_if_exists(FAKE_BIOMETRICS_CSV)
    recommendations = read_csv_if_exists(RECOMMENDATIONS_CSV)
    return clips, labels, biometrics, recommendations


def label_path(session_id: str) -> Path:
    safe_session = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in session_id)
    return LABELS_DIR / f"{safe_session}_labels.csv"


def write_label_row(row: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

    rows = [
        existing
        for existing in rows
        if not (
            str(existing.get("user_id")) == str(row["user_id"])
            and str(existing.get("session_id")) == str(row["session_id"])
            and str(existing.get("clip_id")) == str(row["clip_id"])
        )
    ]
    rows.append(row)
    rows.sort(key=lambda item: int(item.get("trial_index", 0)))

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def session_label_rows(labels: pd.DataFrame, user_id: str, session_id: str) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame(columns=LABEL_COLUMNS)
    rows = labels[
        (labels["user_id"].astype(str) == str(user_id))
        & (labels["session_id"].astype(str) == str(session_id))
    ].copy()
    if not rows.empty:
        rows["trial_index"] = pd.to_numeric(rows["trial_index"], errors="coerce")
        rows = rows.sort_values("trial_index")
    return rows


def initialize_collection(clips: pd.DataFrame, user_id: str, session_id: str) -> None:
    order = clips["clip_id"].astype(str).tolist()
    random.Random(session_id).shuffle(order)
    st.session_state.collection_started = True
    st.session_state.user_id = user_id
    st.session_state.session_id = session_id
    st.session_state.order = order
    st.session_state.trial_position = 0
    reset_trial_state(stage="rest")


def reset_trial_state(*, stage: str) -> None:
    st.session_state.stage = stage
    st.session_state.rest_started_at = None
    st.session_state.listen_started_at = None
    st.session_state.clip_start_time = None
    st.session_state.clip_end_time = None


def current_clip(clips: pd.DataFrame) -> pd.Series:
    clip_id = st.session_state.order[st.session_state.trial_position]
    return clips[clips["clip_id"].astype(str) == clip_id].iloc[0]


def move_trial(delta: int) -> None:
    next_position = st.session_state.trial_position + delta
    st.session_state.trial_position = max(0, min(next_position, len(st.session_state.order) - 1))
    reset_trial_state(stage="rest")


def existing_label(labels: pd.DataFrame, user_id: str, session_id: str, clip_id: str) -> pd.Series | None:
    rows = session_label_rows(labels, user_id, session_id)
    rows = rows[rows["clip_id"].astype(str) == str(clip_id)]
    return None if rows.empty else rows.iloc[0]


def biometric_row(
    biometrics: pd.DataFrame,
    user_id: str,
    session_id: str,
    clip_id: str,
) -> pd.Series | None:
    if biometrics.empty:
        return None
    rows = biometrics[
        (biometrics["user_id"].astype(str) == str(user_id))
        & (biometrics["session_id"].astype(str) == str(session_id))
        & (biometrics["clip_id"].astype(str) == str(clip_id))
    ]
    if rows.empty:
        rows = biometrics[biometrics["clip_id"].astype(str) == str(clip_id)]
    return None if rows.empty else rows.iloc[0]


def prediction_row(recommendations: pd.DataFrame, user_id: str, clip_id: str) -> pd.Series | None:
    if recommendations.empty:
        return None
    rows = recommendations[
        (recommendations["user_id"].astype(str) == str(user_id))
        & (recommendations["clip_id"].astype(str) == str(clip_id))
    ]
    return None if rows.empty else rows.iloc[0]


def next_recommendation(
    recommendations: pd.DataFrame,
    user_id: str,
    target_mode: str,
    current_clip_id: str,
) -> pd.Series | None:
    if recommendations.empty:
        return None
    rows = recommendations[
        (recommendations["user_id"].astype(str) == str(user_id))
        & (recommendations["target_mode"].astype(str) == target_mode)
        & (recommendations["clip_id"].astype(str) != str(current_clip_id))
    ].sort_values("rank")
    return None if rows.empty else rows.iloc[0]


def arousal_state(row: pd.Series | None, prediction: pd.Series | None) -> tuple[str, float | None]:
    if prediction is not None and "arousal_prob" in prediction:
        probability = float(prediction["arousal_prob"])
        return ("energized" if probability >= 0.5 else "calm", probability)

    if row is None:
        return "unknown", None
    hr_change = float(row.get("hr_change_from_baseline", 0))
    eda_peaks = float(row.get("eda_peak_count", 0))
    proxy = min(1.0, max(0.0, (hr_change / 15 * 0.65) + (eda_peaks / 8 * 0.35)))
    return ("energized" if proxy >= 0.5 else "calm", proxy)


def trial_traces(row: pd.Series | None, clip_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(abs(hash(clip_id)) % (2**32))
    seconds = np.arange(0, 31)
    if row is None:
        empty = pd.DataFrame({"second": seconds, "value": np.nan}).set_index("second")
        return empty, empty

    hr_mean = float(row.get("hr_mean", 70))
    hr_change = float(row.get("hr_change_from_baseline", 0))
    eda_mean = float(row.get("eda_mean", 1.0))
    eda_change = float(row.get("eda_change_from_baseline", 0))
    ramp = np.sin(np.linspace(0, np.pi, len(seconds)))

    hr = hr_mean - hr_change * 0.35 + ramp * hr_change * 0.75 + rng.normal(0, 0.7, len(seconds))
    eda = eda_mean - eda_change * 0.30 + ramp * eda_change * 0.85 + rng.normal(0, 0.012, len(seconds))

    hr_frame = pd.DataFrame({"second": seconds, "HR bpm": hr}).set_index("second")
    eda_frame = pd.DataFrame({"second": seconds, "EDA conductance": eda}).set_index("second")
    return hr_frame, eda_frame


def countdown_panel(label: str, state_key: str, seconds: int, done_stage: str) -> None:
    started_at = st.session_state.get(state_key)
    if started_at is None:
        if st.button(f"Start {seconds}s {label}", width="stretch"):
            st.session_state[state_key] = time.time()
            st.rerun()
        return

    target_ms = int((float(started_at) + seconds) * 1000)
    timer_id = f"{label}-{state_key}".replace("_", "-")
    st.iframe(
        f"""
        <div style="font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
                    border: 1px solid #e5e7eb; border-radius: 8px; padding: 14px 16px;
                    background: #f8fafc;">
          <div style="font-size: 13px; color: #475569; margin-bottom: 8px;">
            {seconds}-second {label} timer
          </div>
          <div id="{timer_id}" style="font-size: 36px; font-weight: 700; color: #0f172a;">
            --s
          </div>
          <div id="{timer_id}-note" style="font-size: 13px; color: #475569; margin-top: 8px;">
            Stay still and breathe normally. This timer updates locally, so the page should not flicker.
          </div>
        </div>
        <script>
          const target{timer_id.replace("-", "")} = {target_ms};
          const valueEl{timer_id.replace("-", "")} = document.getElementById("{timer_id}");
          const noteEl{timer_id.replace("-", "")} = document.getElementById("{timer_id}-note");
          function tick{timer_id.replace("-", "")}() {{
            const remaining = Math.max(0, Math.ceil((target{timer_id.replace("-", "")} - Date.now()) / 1000));
            valueEl{timer_id.replace("-", "")}.textContent = remaining + "s";
            if (remaining === 0) {{
              noteEl{timer_id.replace("-", "")}.textContent = "Timer complete. Click Continue when ready.";
              valueEl{timer_id.replace("-", "")}.style.color = "#166534";
            }} else {{
              window.setTimeout(tick{timer_id.replace("-", "")}, 250);
            }}
          }}
          tick{timer_id.replace("-", "")}();
        </script>
        """,
        width="stretch",
        height=142,
    )

    if st.button(f"Continue after {label}", width="stretch"):
        elapsed = time.time() - float(started_at)
        if elapsed < seconds:
            st.warning(f"Wait {int(np.ceil(seconds - elapsed))} more seconds before continuing.")
        else:
            st.session_state.stage = done_stage
            st.rerun()


def render_navigation(total: int) -> None:
    prev_col, center_col, next_col = st.columns([1, 2, 1])
    with prev_col:
        if st.button("← Previous", disabled=st.session_state.trial_position == 0, width="stretch"):
            move_trial(-1)
            st.rerun()
    with center_col:
        st.progress(
            (st.session_state.trial_position + 1) / total,
            text=f"Song {st.session_state.trial_position + 1} of {total}",
        )
    with next_col:
        if st.button("Next →", disabled=st.session_state.trial_position >= total - 1, width="stretch"):
            move_trial(1)
            st.rerun()


def render_collection_flow(clip: pd.Series, labels: pd.DataFrame) -> None:
    user_id = st.session_state.user_id
    session_id = st.session_state.session_id
    clip_id = str(clip["clip_id"])
    saved = existing_label(labels, user_id, session_id, clip_id)

    if saved is not None:
        st.success("This song already has a saved rating. You can overwrite it below or move to another song.")

    st.subheader(f"{clip['track_name']} - {clip['artist']}")
    st.caption("Flow: rest for 30 seconds, play the 30-second preview, then rate how you felt.")

    stage = st.session_state.get("stage", "rest")
    if stage == "rest":
        st.info("Rest quietly before the song so HR and EDA can settle toward baseline.")
        countdown_panel("rest", "rest_started_at", REST_SECONDS, "listen")
        return

    if stage == "listen":
        st.audio(str(clip["preview_url"]))
        st.caption("Press play on the audio, then start the 30-second listen timer.")
        if st.session_state.clip_start_time is None and st.session_state.listen_started_at is None:
            if st.button("Start song timer", width="stretch"):
                st.session_state.clip_start_time = iso_now()
                st.session_state.listen_started_at = time.time()
                st.rerun()
            return
        countdown_panel("song", "listen_started_at", LISTEN_SECONDS, "rate")
        if st.session_state.stage == "rate" and st.session_state.clip_end_time is None:
            st.session_state.clip_end_time = iso_now()
        return

    default_mood = str(saved["mood"]) if saved is not None and str(saved["mood"]) in MOOD_OPTIONS else "neutral"
    with st.form("rating_form"):
        st.write("Rate the song you just heard.")
        preference = st.slider(
            "Preference",
            1,
            5,
            int(saved["preference"]) if saved is not None else 3,
            help="1 = dislike, 5 = like",
        )
        arousal = st.slider(
            "Arousal",
            1,
            5,
            int(saved["arousal"]) if saved is not None else 3,
            help="1 = calm, 5 = energized",
        )
        valence = st.slider(
            "Valence",
            1,
            5,
            int(saved["valence"]) if saved is not None else 3,
            help="1 = negative, 5 = positive",
        )
        mood = st.selectbox("Mood", MOOD_OPTIONS, index=MOOD_OPTIONS.index(default_mood))
        other_mood = st.text_input("Mood detail") if mood == "other" else ""
        familiarity = st.slider(
            "Familiarity",
            1,
            5,
            int(saved["familiarity"]) if saved is not None else 3,
            help="1 = unfamiliar, 5 = very familiar",
        )
        notes = st.text_area("Notes", value="" if saved is None or pd.isna(saved.get("notes", "")) else str(saved["notes"]))
        submitted = st.form_submit_button("Save rating")

    if submitted:
        row = {
            "user_id": user_id,
            "session_id": session_id,
            "clip_id": clip_id,
            "trial_index": st.session_state.trial_position + 1,
            "clip_start_time": st.session_state.clip_start_time or iso_now(),
            "clip_end_time": st.session_state.clip_end_time or iso_now(),
            "preference": preference,
            "arousal": arousal,
            "valence": valence,
            "mood": other_mood.strip() if mood == "other" and other_mood.strip() else mood,
            "familiarity": familiarity,
            "notes": notes.strip(),
        }
        write_label_row(row, label_path(session_id))
        st.cache_data.clear()
        if st.session_state.trial_position < len(st.session_state.order) - 1:
            move_trial(1)
        else:
            st.session_state.stage = "complete"
        st.rerun()


def render_biometrics_and_prediction(
    biometrics: pd.DataFrame,
    recommendations: pd.DataFrame,
    user_id: str,
    session_id: str,
    clip_id: str,
    target_mode: str,
) -> None:
    bio = biometric_row(biometrics, user_id, session_id, clip_id)
    pred = prediction_row(recommendations, user_id, clip_id)
    state, arousal_prob = arousal_state(bio, pred)
    next_song = next_recommendation(recommendations, user_id, target_mode, clip_id)

    prediction_col, rec_col = st.columns(2)
    with prediction_col:
        st.subheader("Model Signal")
        st.metric("Predicted arousal", state)
        if arousal_prob is not None:
            st.progress(float(arousal_prob), text=f"{arousal_prob:.0%} energized")
    with rec_col:
        st.subheader("Next Recommendation")
        if next_song is None:
            st.info("Run `python src/models/recommend.py --target-mode all --user-id <id> --session-id <id>` after collecting labels.")
        else:
            st.write(f"**{next_song['track_name']}**")
            st.write(str(next_song["artist"]))
            st.metric("Score", f"{float(next_song['score']):.2f}")

    st.subheader("Biometric Preview")
    if bio is None:
        st.info("No biometric row found yet. After real sensors are ready, this section should use the real export.")
        return

    metrics = st.columns(6)
    metrics[0].metric("HR mean", f"{float(bio['hr_mean']):.1f} bpm")
    metrics[1].metric("HR max", f"{float(bio['hr_max']):.1f} bpm")
    metrics[2].metric("HR change", f"{float(bio['hr_change_from_baseline']):+.1f} bpm")
    metrics[3].metric("EDA mean", f"{float(bio['eda_mean']):.3f}")
    metrics[4].metric("EDA peaks", int(bio["eda_peak_count"]))
    metrics[5].metric("EDA amplitude", f"{float(bio['eda_max_amplitude']):.3f}")

    hr_frame, eda_frame = trial_traces(bio, clip_id)
    hr_col, eda_col = st.columns(2)
    with hr_col:
        st.caption("Heart rate response")
        st.line_chart(hr_frame, height=240)
    with eda_col:
        st.caption("EDA/GSR response")
        st.line_chart(eda_frame, height=240)


def main() -> None:
    ensure_project_dirs()
    st.set_page_config(page_title="BioBeat Self-Training", page_icon="BB", layout="wide")
    st.title("BioBeat Self-Training")

    clips, labels, biometrics, recommendations = load_dashboard_data()
    if clips.empty:
        st.error("No clips found. Run `python src/itunes/build_clips_csv.py` first.")
        st.stop()

    with st.sidebar:
        st.header("Session")
        default_session = datetime.now().strftime("self_%Y%m%d_%H%M%S")
        user_id = st.text_input("User ID", value=st.session_state.get("user_id", ""))
        session_id = st.text_input("Session ID", value=st.session_state.get("session_id", default_session))
        target_mode = st.segmented_control("Recommendation mode", ["calm", "hype"], default="calm")
        if st.button("Begin / restart session", disabled=not user_id.strip() or not session_id.strip(), width="stretch"):
            initialize_collection(clips, user_id.strip(), session_id.strip())
            st.rerun()

        if st.session_state.get("collection_started"):
            path = label_path(st.session_state.session_id)
            st.caption(f"Saving to `{path.relative_to(Path.cwd())}`")

    if not st.session_state.get("collection_started"):
        st.info("Enter a user ID and session ID, then begin. The app will guide you through rest, listening, and rating.")
        st.stop()

    total = len(st.session_state.order)
    render_navigation(total)

    if st.session_state.get("stage") == "complete":
        st.success("Session complete. Labels were saved after each song.")
        st.stop()

    clip = current_clip(clips)
    session_rows = session_label_rows(labels, st.session_state.user_id, st.session_state.session_id)
    completed = session_rows["clip_id"].nunique() if not session_rows.empty else 0
    st.caption(f"Saved ratings: {completed} of {total}")

    collection_col, signal_col = st.columns([1.05, 0.95])
    with collection_col:
        render_collection_flow(clip, labels)
    with signal_col:
        render_biometrics_and_prediction(
            biometrics,
            recommendations,
            st.session_state.user_id,
            st.session_state.session_id,
            str(clip["clip_id"]),
            str(target_mode),
        )


if __name__ == "__main__":
    main()
