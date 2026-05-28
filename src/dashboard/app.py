from __future__ import annotations

import csv
import glob
import inspect
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from biobeat_paths import CLIPS_CSV, LABELS_DIR, RAW_DIR, ensure_project_dirs  # noqa: E402
from sensors.import_apple_watch_hr import build_hr_rows_from_frames, write_imported_rows  # noqa: E402
from sensors.serial_readers import (  # noqa: E402
    MockSensorReader,
    SerialSensorReader,
    append_sensor_rows,
    list_serial_ports,
    sample_to_row,
)


REST_SECONDS = 20
LISTEN_SECONDS = 30
APPLE_WATCH_SOURCE = "apple_watch_csv"
EMOTION_LABELS = {
    1: "sad",
    2: "neutral",
    3: "happy",
}
FAMILIARITY_LABELS = {
    1: "never heard it before",
    2: "maybe recognize it",
    3: "have heard it a few times",
    4: "know it well",
    5: "very familiar / personally meaningful / know what is coming",
}
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
SENSOR_COLUMNS = [
    "timestamp",
    "user_id",
    "session_id",
    "clip_id",
    "trial_index",
    "phase",
    "hr",
    "eda",
    "sensor_elapsed_ms",
    "source",
    "raw_line",
]

def iso_now() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def full_width_button(label: str, **kwargs: object) -> bool:
    if "width" in inspect.signature(st.button).parameters:
        return st.button(label, width="stretch", **kwargs)
    return st.button(label, use_container_width=True, **kwargs)


def safe_file_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)


def label_path(session_id: str) -> Path:
    return LABELS_DIR / f"{safe_file_id(session_id)}_labels.csv"


def sensor_path(session_id: str) -> Path:
    return RAW_DIR / "sensor" / f"{safe_file_id(session_id)}_sensor.csv"


def read_csv_if_exists(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path)


@st.cache_data
def load_clips(path: Path) -> pd.DataFrame:
    clips = read_csv_if_exists(path)
    required = {"clip_id", "track_name", "artist", "preview_url"}
    missing = required - set(clips.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
    return clips


@st.cache_data
def load_labels(labels_dir: Path) -> pd.DataFrame:
    paths = sorted(glob.glob(str(labels_dir / "*_labels.csv")))
    frames = [pd.read_csv(path) for path in paths]
    if not frames:
        return pd.DataFrame(columns=LABEL_COLUMNS)
    labels = pd.concat(frames, ignore_index=True)
    for column in LABEL_COLUMNS:
        if column not in labels.columns:
            labels[column] = ""
    return labels[LABEL_COLUMNS]


def load_sensor_rows(session_id: str) -> pd.DataFrame:
    rows = read_csv_if_exists(sensor_path(session_id), SENSOR_COLUMNS)
    for column in SENSOR_COLUMNS:
        if column not in rows.columns:
            rows[column] = ""
    return rows[SENSOR_COLUMNS]


def summarize_hr_rows(rows: list[dict[str, object]], labels: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    hr_rows = pd.DataFrame(rows)
    for _, label in labels.iterrows():
        trial_index = int(label["trial_index"])
        clip_id = str(label["clip_id"])
        trial_rows = pd.DataFrame()
        if not hr_rows.empty:
            trial_rows = hr_rows[
                (hr_rows["clip_id"].astype(str) == clip_id)
                & (pd.to_numeric(hr_rows["trial_index"], errors="coerce") == trial_index)
            ]
        summary_rows.append(
            {
                "trial_index": trial_index,
                "clip_id": clip_id,
                "rest_hr_samples": int((trial_rows["phase"] == "rest").sum()) if not trial_rows.empty else 0,
                "listen_hr_samples": int((trial_rows["phase"] == "listen").sum()) if not trial_rows.empty else 0,
            }
        )
    return pd.DataFrame(summary_rows)


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


def saved_field(saved: pd.Series | None, field: str, default: object = "") -> object:
    if saved is None or field not in saved or pd.isna(saved.get(field)):
        return default
    value = saved.get(field)
    if isinstance(value, str) and not value.strip():
        return default
    return value


def saved_int(saved: pd.Series | None, field: str, default: int) -> int:
    value = saved_field(saved, field, default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


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


def existing_label(labels: pd.DataFrame, user_id: str, session_id: str, clip_id: str) -> pd.Series | None:
    rows = session_label_rows(labels, user_id, session_id)
    rows = rows[rows["clip_id"].astype(str) == str(clip_id)]
    return None if rows.empty else rows.iloc[0]


def initialize_collection(
    clips: pd.DataFrame,
    *,
    user_id: str,
    session_id: str,
    sensor_source: str,
    serial_port: str,
    baud_rate: int,
) -> None:
    order = clips["clip_id"].astype(str).tolist()
    random.Random(session_id).shuffle(order)
    st.session_state.collection_started = True
    st.session_state.user_id = user_id
    st.session_state.session_id = session_id
    st.session_state.order = order
    st.session_state.trial_position = 0
    st.session_state.sensor_source = sensor_source
    st.session_state.serial_port = serial_port
    st.session_state.baud_rate = baud_rate
    reset_trial_state(stage="rest")


def reset_trial_state(*, stage: str) -> None:
    st.session_state.stage = stage
    st.session_state.rating_step = "emotion"
    st.session_state.rest_start_time = None
    st.session_state.rest_end_time = None
    st.session_state.clip_start_time = None
    st.session_state.clip_end_time = None
    st.session_state.rest_sample_count = 0
    st.session_state.listen_sample_count = 0
    st.session_state.rating_error = ""


def current_clip(clips: pd.DataFrame) -> pd.Series:
    clip_id = st.session_state.order[st.session_state.trial_position]
    return clips[clips["clip_id"].astype(str) == clip_id].iloc[0]


def move_trial(delta: int, *, stage: str = "rest") -> None:
    next_position = st.session_state.trial_position + delta
    st.session_state.trial_position = max(0, min(next_position, len(st.session_state.order) - 1))
    reset_trial_state(stage=stage)


def rating_digit_key(clip_id: str, field: str) -> str:
    return f"rating_digit_{clip_id}_{field}"


def parse_digit(value: object, *, allowed: set[int]) -> int | None:
    text = str(value).strip()
    if not text.isdigit():
        return None
    number = int(text)
    return number if number in allowed else None


def saved_emotion_value(saved: pd.Series | None) -> str:
    if saved is None:
        return ""
    mood = str(saved_field(saved, "mood", "")).strip().lower()
    for number, label in EMOTION_LABELS.items():
        if mood == label:
            return str(number)

    value = saved_int(saved, "valence", 2)
    if value in EMOTION_LABELS:
        return str(value)
    if value <= 2:
        return "1"
    if value >= 4:
        return "3"
    return "2"


def saved_familiarity_value(saved: pd.Series | None) -> str:
    if saved is None:
        return ""
    value = saved_int(saved, "familiarity", 3)
    return str(max(1, min(value, 5)))


def rating_inputs_are_valid(clip_id: str) -> tuple[int | None, int | None]:
    emotion = parse_digit(
        st.session_state.get(rating_digit_key(clip_id, "emotion"), ""),
        allowed=set(EMOTION_LABELS),
    )
    familiarity = parse_digit(
        st.session_state.get(rating_digit_key(clip_id, "familiarity"), ""),
        allowed=set(FAMILIARITY_LABELS),
    )
    return emotion, familiarity


def choose_digit_button(label: str, value: int, *, key: str) -> bool:
    return full_width_button(f"{value}. {label}", key=key)


def save_current_rating(clip: pd.Series, labels: pd.DataFrame) -> None:
    user_id = st.session_state.user_id
    session_id = st.session_state.session_id
    clip_id = str(clip["clip_id"])
    trial_index = int(st.session_state.trial_position + 1)
    saved = existing_label(labels, user_id, session_id, clip_id)
    emotion, familiarity = rating_inputs_are_valid(clip_id)
    if emotion is None or familiarity is None:
        raise ValueError("Enter emotion as 1, 2, or 3 and familiarity as 1 through 5.")
    mood = EMOTION_LABELS[emotion]
    row = {
        "user_id": user_id,
        "session_id": session_id,
        "clip_id": clip_id,
        "trial_index": trial_index,
        "rest_start_time": st.session_state.rest_start_time or saved_field(saved, "rest_start_time", ""),
        "rest_end_time": st.session_state.rest_end_time or saved_field(saved, "rest_end_time", ""),
        "clip_start_time": st.session_state.clip_start_time or saved_field(saved, "clip_start_time", iso_now()),
        "clip_end_time": st.session_state.clip_end_time or saved_field(saved, "clip_end_time", iso_now()),
        "preference": "",
        "arousal": "",
        "valence": emotion,
        "mood": mood,
        "familiarity": familiarity,
        "notes": "",
    }
    write_label_row(row, label_path(session_id))
    st.cache_data.clear()


def save_rating_and_advance(clip: pd.Series, labels: pd.DataFrame, total: int) -> None:
    clip_id = str(clip["clip_id"])
    emotion, familiarity = rating_inputs_are_valid(clip_id)
    if emotion is None or familiarity is None:
        st.session_state.rating_error = "Emotion must be 1-3 and familiarity must be 1-5."
        return

    save_current_rating(clip, labels)
    st.session_state.rating_error = ""
    if st.session_state.trial_position >= total - 1:
        st.session_state.stage = "complete"
    else:
        move_trial(1)


def set_rating_digit_and_advance(
    *,
    clip: pd.Series,
    labels: pd.DataFrame,
    total: int,
    field: str,
    value: int,
) -> None:
    clip_id = str(clip["clip_id"])
    st.session_state[rating_digit_key(clip_id, field)] = str(value)
    st.session_state.rating_error = ""
    if field == "emotion":
        st.session_state.rating_step = "familiarity"
    else:
        save_rating_and_advance(clip, labels, total)


def open_sensor_reader() -> MockSensorReader | SerialSensorReader:
    source = st.session_state.get("sensor_source", "Mock sensor")
    if source == "Raspberry Pi Seeed GSR serial":
        port = st.session_state.get("serial_port", "")
        if not port:
            raise RuntimeError("Select the Raspberry Pi serial port before recording.")
        return SerialSensorReader(
            port=port,
            baud_rate=int(st.session_state.get("baud_rate", 115200)),
            source="raspberry_pi_seeed_gsr",
            single_value_metric="eda",
        )
    return MockSensorReader()


def capture_sensor_phase(
    *,
    phase: str,
    duration_seconds: int,
    clip_id: str,
    trial_index: int,
) -> tuple[str, str, int]:
    reader = open_sensor_reader()
    rows: list[dict[str, object]] = []
    start_time = iso_now()
    start_monotonic = time.time()

    progress = st.progress(0.0, text=f"Recording {phase} sensor data")
    latest = st.empty()
    hr_chart = st.empty()
    eda_chart = st.empty()
    sample_frame = pd.DataFrame(columns=["elapsed_sec", "hr", "eda", "sensor_elapsed_ms"])

    try:
        while True:
            elapsed = time.time() - start_monotonic
            if elapsed >= duration_seconds:
                break

            sample = reader.read_sample(timeout=0.2)
            if sample is not None:
                row = sample_to_row(
                    sample,
                    user_id=st.session_state.user_id,
                    session_id=st.session_state.session_id,
                    clip_id=clip_id,
                    trial_index=trial_index,
                    phase=phase,
                )
                rows.append(row)
                sample_frame.loc[len(sample_frame)] = {
                    "elapsed_sec": round(elapsed, 2),
                    "hr": sample.hr,
                    "eda": sample.eda,
                    "sensor_elapsed_ms": sample.sensor_elapsed_ms,
                }

                hr_text = "--" if sample.hr is None else f"{sample.hr:.1f} bpm"
                eda_text = "--" if sample.eda is None else f"{sample.eda:.3f}"
                if sample.hr is None and sample.sensor_elapsed_ms is not None:
                    latest.metric("Latest GSR sample", f"EDA {eda_text}", f"{sample.sensor_elapsed_ms:.0f} ms")
                else:
                    latest.metric("Latest sample", f"HR {hr_text} / EDA {eda_text}")
                if sample_frame["hr"].notna().any():
                    hr_chart.line_chart(sample_frame[["elapsed_sec", "hr"]].dropna().set_index("elapsed_sec"), height=180)
                if sample_frame["eda"].notna().any():
                    eda_chart.line_chart(sample_frame[["elapsed_sec", "eda"]].dropna().set_index("elapsed_sec"), height=180)

            progress.progress(min(elapsed / duration_seconds, 1.0), text=f"{phase.title()} recording: {max(0, int(duration_seconds - elapsed))}s remaining")

        end_time = iso_now()
        append_sensor_rows(sensor_path(st.session_state.session_id), rows)
        progress.progress(1.0, text=f"{phase.title()} recording complete")
        return start_time, end_time, len(rows)
    finally:
        reader.close()


def render_navigation(total: int, clip: pd.Series, labels: pd.DataFrame) -> None:
    is_last = st.session_state.trial_position >= total - 1
    stage = st.session_state.get("stage", "rest")
    prev_col, center_col, next_col = st.columns([1, 2, 1])
    with prev_col:
        if full_width_button("← Previous", disabled=st.session_state.trial_position == 0):
            move_trial(-1, stage="rate")
            st.rerun()
    with center_col:
        st.progress(
            (st.session_state.trial_position + 1) / total,
            text=f"Song {st.session_state.trial_position + 1} of {total}",
        )
    with next_col:
        label = "Finish" if is_last else "Next →"
        if full_width_button(label, disabled=stage != "rate"):
            if stage == "rate":
                try:
                    save_current_rating(clip, labels)
                except ValueError as exc:
                    st.session_state.rating_error = str(exc)
                    st.rerun()
            if is_last:
                st.session_state.stage = "complete"
            else:
                move_trial(1)
            st.rerun()


def render_rating_flashcard(clip: pd.Series, labels: pd.DataFrame, saved: pd.Series | None) -> None:
    clip_id = str(clip["clip_id"])
    total = len(st.session_state.order)
    emotion_key = rating_digit_key(clip_id, "emotion")
    familiarity_key = rating_digit_key(clip_id, "familiarity")
    if emotion_key not in st.session_state:
        st.session_state[emotion_key] = saved_emotion_value(saved)
    if familiarity_key not in st.session_state:
        st.session_state[familiarity_key] = saved_familiarity_value(saved)

    step = st.session_state.get("rating_step", "emotion")
    if step not in {"emotion", "familiarity"}:
        step = "emotion"
        st.session_state.rating_step = step

    if step == "emotion":
        current = st.session_state.get(emotion_key, "")
        st.subheader("Emotion?")
        st.markdown("### Click one answer")
        st.caption("1 = sad | 2 = neutral | 3 = happy")
        if current:
            st.caption(f"Current saved answer: {current}")
        cols = st.columns(3)
        for index, (value, label) in enumerate(EMOTION_LABELS.items()):
            with cols[index]:
                if choose_digit_button(label, value, key=f"emotion_{clip_id}_{value}"):
                    set_rating_digit_and_advance(
                        clip=clip,
                        labels=labels,
                        total=total,
                        field="emotion",
                        value=value,
                    )
                    st.rerun()
        return

    current = st.session_state.get(familiarity_key, "")
    emotion = st.session_state.get(emotion_key, "")
    st.subheader("Familiarity?")
    st.markdown("### Click one answer")
    emotion_number = parse_digit(emotion, allowed=set(EMOTION_LABELS))
    if emotion_number is not None:
        st.caption(f"Emotion: {emotion_number} ({EMOTION_LABELS[emotion_number]})")
    if current:
        st.caption(f"Current saved answer: {current}")

    for value, label in FAMILIARITY_LABELS.items():
        if choose_digit_button(label, value, key=f"familiarity_{clip_id}_{value}"):
            set_rating_digit_and_advance(
                clip=clip,
                labels=labels,
                total=total,
                field="familiarity",
                value=value,
            )
            st.rerun()


def render_collection_flow(clip: pd.Series, labels: pd.DataFrame) -> None:
    user_id = st.session_state.user_id
    session_id = st.session_state.session_id
    clip_id = str(clip["clip_id"])
    trial_index = int(st.session_state.trial_position + 1)
    saved = existing_label(labels, user_id, session_id, clip_id)

    if saved is not None:
        st.success("This song already has a saved rating. You can overwrite it or move to another song.")

    st.subheader(f"{clip['track_name']} - {clip['artist']}")
    st.caption("Training flow: record baseline rest, record the music response, then enter two quick ratings.")

    stage = st.session_state.get("stage", "rest")
    if stage == "rest":
        st.info("Rest period started. Sit quietly until the song begins automatically.")
        try:
            start_time, end_time, sample_count = capture_sensor_phase(
                phase="rest",
                duration_seconds=REST_SECONDS,
                clip_id=clip_id,
                trial_index=trial_index,
            )
        except Exception as exc:
            st.error(str(exc))
            return
        st.session_state.rest_start_time = start_time
        st.session_state.rest_end_time = end_time
        st.session_state.rest_sample_count = sample_count
        st.session_state.stage = "recording"
        st.rerun()
        return

    if stage == "listen":
        st.session_state.stage = "recording"
        st.rerun()
        return

    if stage == "recording":
        st.audio(str(clip["preview_url"]), autoplay=True)
        st.info("Song recording started. If your browser blocks autoplay, press play immediately.")
        try:
            start_time, end_time, sample_count = capture_sensor_phase(
                phase="listen",
                duration_seconds=LISTEN_SECONDS,
                clip_id=clip_id,
                trial_index=trial_index,
            )
        except Exception as exc:
            st.session_state.stage = "listen"
            st.error(str(exc))
            return
        st.session_state.clip_start_time = start_time
        st.session_state.clip_end_time = end_time
        st.session_state.listen_sample_count = sample_count
        st.session_state.stage = "rate"
        st.session_state.rating_step = "emotion"
        st.rerun()

    if stage == "rate":
        render_rating_flashcard(clip, labels, saved)
        if st.session_state.get("rating_error"):
            st.error(st.session_state.rating_error)
        return

    if full_width_button("Reset this trial"):
        reset_trial_state(stage="rest")
        st.rerun()

    return


def render_sensor_panel(clip_id: str, trial_index: int) -> None:
    st.subheader("Sensor Recording")
    st.caption("Raw Seeed GSR/EDA samples are saved during collection. Apple Watch HR CSV rows can be imported into the same file after the run.")
    sensor_file = sensor_path(st.session_state.session_id)
    st.code(str(sensor_file.relative_to(Path.cwd())), language="text")

    rows = load_sensor_rows(st.session_state.session_id)
    rows = rows[
        (rows["clip_id"].astype(str) == str(clip_id))
        & (pd.to_numeric(rows["trial_index"], errors="coerce") == trial_index)
    ].copy()
    if rows.empty:
        st.info("No sensor samples saved for this song yet.")
        return

    rows["sample_index"] = range(1, len(rows) + 1)
    rest_count = int((rows["phase"] == "rest").sum())
    listen_count = int((rows["phase"] == "listen").sum())
    metric_cols = st.columns(4)
    metric_cols[0].metric("Rest samples", rest_count)
    metric_cols[1].metric("Listen samples", listen_count)
    if rows["hr"].notna().any():
        metric_cols[2].metric("Latest HR", f"{float(rows['hr'].dropna().iloc[-1]):.1f} bpm")
    if rows["eda"].notna().any():
        metric_cols[3].metric("Latest EDA", f"{float(rows['eda'].dropna().iloc[-1]):.3f}")

    hr_rows = rows[["sample_index", "phase", "hr"]].dropna()
    eda_rows = rows[["sample_index", "phase", "eda"]].dropna()
    hr_col, eda_col = st.columns(2)
    with hr_col:
        st.caption("Heart rate")
        if hr_rows.empty:
            st.info("No HR values parsed.")
        else:
            st.line_chart(hr_rows.set_index("sample_index")[["hr"]], height=240)
    with eda_col:
        st.caption("EDA/GSR")
        if eda_rows.empty:
            st.info("No EDA values parsed.")
        else:
            st.line_chart(eda_rows.set_index("sample_index")[["eda"]], height=240)


def render_sensor_setup() -> tuple[str, str, int]:
    sensor_source = st.radio("Sensor source", ["Mock sensor", "Raspberry Pi Seeed GSR serial"], horizontal=True)
    serial_port = ""
    baud_rate = 115200

    if sensor_source == "Raspberry Pi Seeed GSR serial":
        ports = list_serial_ports()
        if ports:
            serial_port = st.selectbox("Raspberry Pi serial port", ports)
        else:
            st.warning("No serial ports found. Connect the Raspberry Pi serial output, then reload.")
            serial_port = st.text_input("Manual serial port", value="/dev/cu.usbserial")
        baud_rate = int(st.number_input("Baud rate", min_value=1200, max_value=2000000, value=115200, step=9600))
        st.caption("Accepted Seeed GSR formats: `GSR:1.42`, `EDA:1.42`, `1.42`, or JSON like `{\"gsr\":1.42}`. Apple Watch HR is imported from CSV later.")
    else:
        st.caption("Mock sensor generates plausible HR and EDA values so the training flow can be tested without hardware.")

    return sensor_source, serial_port, baud_rate


def render_apple_watch_start_instructions() -> None:
    st.info(
        "Before you begin: start an Apple Watch Workout and keep it running for the full BioBeat session. "
        "After the last song, export heart-rate data as a CSV and upload it on the completion screen."
    )


def render_apple_watch_import(labels: pd.DataFrame) -> None:
    st.subheader("Apple Watch Heart Rate Upload")
    st.caption(
        "Upload the heart-rate CSV after finishing the session. BioBeat syncs it by matching Apple Watch timestamps "
        "to each recorded rest and song window."
    )

    session_labels = session_label_rows(labels, st.session_state.user_id, st.session_state.session_id)
    if session_labels.empty:
        st.warning("No saved ratings/windows were found for this session yet.")
        return

    uploaded = st.file_uploader("Upload Apple Watch / Apple Health heart-rate CSV", type=["csv"])
    offset = st.number_input(
        "Time offset seconds",
        value=0.0,
        step=1.0,
        help="Use this only if the CSV timestamps look shifted from the dashboard timestamps.",
    )
    replace_prior = st.checkbox("Replace previous Apple Watch import for this session", value=True)

    if uploaded is None:
        st.info("Upload the exported HR CSV here when the session is done.")
        return

    try:
        health = pd.read_csv(uploaded)
        rows = build_hr_rows_from_frames(
            health=health,
            labels=session_labels,
            source=APPLE_WATCH_SOURCE,
            time_offset_seconds=float(offset),
        )
    except Exception as exc:
        st.error(str(exc))
        return

    coverage = summarize_hr_rows(rows, session_labels)
    st.dataframe(coverage, hide_index=True)

    missing = coverage[(coverage["rest_hr_samples"] == 0) | (coverage["listen_hr_samples"] == 0)]
    if missing.empty:
        st.success(f"Ready to import {len(rows)} HR samples. Every song has rest and listen HR coverage.")
    else:
        st.warning(
            f"Ready to import {len(rows)} HR samples, but {len(missing)} song(s) have a rest or listen window with no HR samples. "
            "If that seems wrong, adjust the time offset and check the table again."
        )

    if full_width_button("Import Apple Watch HR", disabled=len(rows) == 0):
        write_imported_rows(
            sensor_path(st.session_state.session_id),
            rows,
            source=APPLE_WATCH_SOURCE,
            replace_source=replace_prior,
        )
        st.cache_data.clear()
        st.success(f"Imported {len(rows)} Apple Watch HR rows into {sensor_path(st.session_state.session_id).relative_to(Path.cwd())}.")


def main() -> None:
    ensure_project_dirs()
    st.set_page_config(page_title="BioBeat Training Collector", page_icon="BB", layout="wide")
    st.title("BioBeat Training Collector")

    try:
        clips = load_clips(CLIPS_CSV)
    except Exception as exc:
        st.error(str(exc))
        st.stop()
    labels = load_labels(LABELS_DIR)

    with st.sidebar:
        st.header("Session")
        default_session = datetime.now().strftime("self_%Y%m%d_%H%M%S")
        user_id = st.text_input("User ID", value=st.session_state.get("user_id", ""))
        session_id = st.text_input("Session ID", value=st.session_state.get("session_id", default_session))
        sensor_source, serial_port, baud_rate = render_sensor_setup()
        st.divider()
        st.caption("Apple Watch: start a Workout before clicking Begin, then upload the HR CSV after the session.")
        if full_width_button("Begin / restart session", disabled=not user_id.strip() or not session_id.strip()):
            initialize_collection(
                clips,
                user_id=user_id.strip(),
                session_id=session_id.strip(),
                sensor_source=sensor_source,
                serial_port=serial_port,
                baud_rate=baud_rate,
            )
            st.rerun()

        if st.session_state.get("collection_started"):
            st.caption(f"Labels: `{label_path(st.session_state.session_id).relative_to(Path.cwd())}`")
            st.caption(f"Sensors: `{sensor_path(st.session_state.session_id).relative_to(Path.cwd())}`")

    if not st.session_state.get("collection_started"):
        render_apple_watch_start_instructions()
        st.info("Enter a user ID/session ID, choose mock or Raspberry Pi Seeed GSR serial, then begin.")
        st.stop()

    total = len(st.session_state.order)
    if st.session_state.get("stage") == "complete":
        st.success("Session complete. Labels and raw GSR samples were saved incrementally.")
        render_apple_watch_import(labels)
        st.stop()

    clip = current_clip(clips)
    trial_index = int(st.session_state.trial_position + 1)
    render_navigation(total, clip, labels)

    session_rows = session_label_rows(labels, st.session_state.user_id, st.session_state.session_id)
    completed = session_rows["clip_id"].nunique() if not session_rows.empty else 0
    st.caption(f"Saved ratings: {completed} of {total}")

    collection_col, sensor_col = st.columns([1.0, 1.0])
    with collection_col:
        render_collection_flow(clip, labels)
    with sensor_col:
        render_sensor_panel(str(clip["clip_id"]), trial_index)


if __name__ == "__main__":
    main()
