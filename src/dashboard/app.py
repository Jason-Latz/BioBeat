from __future__ import annotations

import glob
import sys
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


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_labels(labels_dir: Path) -> pd.DataFrame:
    paths = sorted(glob.glob(str(labels_dir / "*_labels.csv")))
    frames = [pd.read_csv(path) for path in paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data
def load_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    clips = read_csv_if_exists(CLIPS_CSV)
    labels = load_labels(LABELS_DIR)
    biometrics = read_csv_if_exists(FAKE_BIOMETRICS_CSV)
    recommendations = read_csv_if_exists(RECOMMENDATIONS_CSV)
    return clips, labels, biometrics, recommendations


def selected_label_rows(labels: pd.DataFrame, user_id: str, session_id: str) -> pd.DataFrame:
    if labels.empty:
        return labels
    rows = labels[
        (labels["user_id"].astype(str) == str(user_id))
        & (labels["session_id"].astype(str) == str(session_id))
    ].copy()
    if "trial_index" in rows.columns:
        rows = rows.sort_values("trial_index")
    return rows


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


def trial_trace(row: pd.Series | None, clip_id: str) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash(clip_id)) % (2**32))
    seconds = np.arange(0, 31)
    if row is None:
        return pd.DataFrame({"second": seconds, "HR": np.nan, "EDA": np.nan}).set_index("second")

    hr_mean = float(row.get("hr_mean", 70))
    hr_change = float(row.get("hr_change_from_baseline", 0))
    eda_mean = float(row.get("eda_mean", 1.0))
    eda_change = float(row.get("eda_change_from_baseline", 0))

    ramp = np.sin(np.linspace(0, np.pi, len(seconds)))
    hr = hr_mean - hr_change * 0.35 + ramp * hr_change * 0.75 + rng.normal(0, 0.7, len(seconds))
    eda = eda_mean - eda_change * 0.30 + ramp * eda_change * 0.85 + rng.normal(0, 0.012, len(seconds))
    return pd.DataFrame({"second": seconds, "HR": hr, "EDA": eda}).set_index("second")


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


def main() -> None:
    ensure_project_dirs()
    st.set_page_config(page_title="BioBeat Dashboard", page_icon="BB", layout="wide")
    st.title("BioBeat Dashboard")

    clips, labels, biometrics, recommendations = load_dashboard_data()
    if clips.empty:
        st.error("No clips found. Run `python src/itunes/build_clips_csv.py` first.")
        st.stop()

    with st.sidebar:
        mode = st.radio("Mode", ["Replay/demo", "Live"], horizontal=True)
        if mode == "Live":
            st.warning("Live sensor mode is reserved for the sensor-team stream.")

        user_options = sorted(labels["user_id"].astype(str).unique()) if not labels.empty else ["demo_user"]
        user_id = st.selectbox("Participant", user_options)

        session_options = (
            sorted(labels[labels["user_id"].astype(str) == user_id]["session_id"].astype(str).unique())
            if not labels.empty
            else ["demo_session"]
        )
        session_id = st.selectbox("Session", session_options)
        target_mode = st.segmented_control("Recommendation", ["calm", "hype"], default="calm")

    session_labels = selected_label_rows(labels, user_id, session_id)
    if session_labels.empty:
        clip_choices = clips["clip_id"].astype(str).tolist()
        selected_clip_id = st.sidebar.selectbox("Clip", clip_choices)
        label = None
    else:
        trial_labels = [
            f"{int(row.trial_index)} - {row.clip_id}" for row in session_labels.itertuples(index=False)
        ]
        selected_trial = st.sidebar.selectbox("Trial", trial_labels)
        selected_clip_id = selected_trial.split(" - ", 1)[1]
        label = session_labels[session_labels["clip_id"].astype(str) == selected_clip_id].iloc[0]

    clip = clips[clips["clip_id"].astype(str) == selected_clip_id].iloc[0]
    bio = biometric_row(biometrics, user_id, session_id, selected_clip_id)
    pred = prediction_row(recommendations, user_id, selected_clip_id)
    state, arousal_prob = arousal_state(bio, pred)
    next_song = next_recommendation(recommendations, user_id, str(target_mode), selected_clip_id)

    top_left, top_mid, top_right = st.columns([1.3, 1.1, 1])
    with top_left:
        st.caption("Current song")
        st.subheader(str(clip["track_name"]))
        st.write(str(clip["artist"]))
        st.audio(str(clip["preview_url"]))

    with top_mid:
        st.caption("Participant")
        st.metric("User", user_id)
        st.metric("Session", session_id)
        if label is not None:
            st.metric("Preference", int(label["preference"]))
            st.metric("Arousal rating", int(label["arousal"]))
            st.metric("Valence rating", int(label["valence"]))

    with top_right:
        st.caption("Prediction")
        st.metric("Predicted arousal", state)
        if arousal_prob is not None:
            st.progress(float(arousal_prob), text=f"{arousal_prob:.0%} energized")
        if next_song is not None:
            st.caption("Recommended next")
            st.write(f"**{next_song['track_name']}**")
            st.write(str(next_song["artist"]))
            st.metric("Score", f"{float(next_song['score']):.2f}")

    chart_col, metric_col = st.columns([1.5, 1])
    with chart_col:
        st.subheader("Trial Trace")
        st.line_chart(trial_trace(bio, selected_clip_id), height=320)

    with metric_col:
        st.subheader("Biometrics")
        if bio is None:
            st.info("No biometric row found for this trial.")
        else:
            st.metric("HR mean", f"{float(bio['hr_mean']):.1f} bpm")
            st.metric("HR max", f"{float(bio['hr_max']):.1f} bpm")
            st.metric("HR change", f"{float(bio['hr_change_from_baseline']):+.1f} bpm")
            st.metric("EDA mean", f"{float(bio['eda_mean']):.3f}")
            st.metric("EDA peaks", int(bio["eda_peak_count"]))
            st.metric("EDA max amplitude", f"{float(bio['eda_max_amplitude']):.3f}")

    if label is not None:
        st.subheader("Self Report")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "preference": label["preference"],
                        "arousal": label["arousal"],
                        "valence": label["valence"],
                        "mood": label["mood"],
                        "familiarity": label["familiarity"],
                        "notes": label.get("notes", ""),
                    }
                ]
            ),
            hide_index=True,
            width="stretch",
        )


if __name__ == "__main__":
    main()
