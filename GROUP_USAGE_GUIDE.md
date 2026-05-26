# BioBeat Group Usage Guide

This guide explains how the group should use the BioBeat repo for the class prototype. The current system is designed to work before the real HR and EDA/GSR sensors are ready.

## What Each Teammate Should Own

Use these ownership lanes to avoid overwriting each other.

| Role | Main files | Responsibility |
| --- | --- | --- |
| Music/data person | `data/input/desired_songs.csv`, `data/clips.csv` | Add songs, run the iTunes clip builder, check that preview URLs are correct. |
| Experiment operator | `src/dashboard/app.py`, `data/raw/labels/` | Run participant sessions and make sure labels save correctly. |
| Sensor team | `data/raw/sensor/`, `data/processed/biometric_features_real.csv` | Collect HR/EDA data and export real biometric features with the agreed schema. |
| ML person | `src/features/`, `src/models/` | Extract audio features, merge tables, train/evaluate models, generate recommendations. |
| Dashboard/demo person | `src/dashboard/app.py`, `data/processed/recommendations.csv` | Run the replay dashboard and prepare the final demo flow. |

One person should coordinate GitHub merges/pushes so generated CSV files do not get accidentally overwritten.

## First-Time Setup

From a fresh clone:

```bash
git clone https://github.com/Jason-Latz/BioBeat.git
cd BioBeat
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If audio extraction fails on iTunes `.m4a` previews, install `ffmpeg`:

```bash
brew install ffmpeg
```

For future terminal sessions, only reactivate the environment:

```bash
cd BioBeat
source .venv/bin/activate
```

## Normal Workflow

Run the project in this order.

### 1. Add Or Edit Desired Songs

Edit:

```text
data/input/desired_songs.csv
```

Each row should look like:

```csv
song_query,artist_query,intended_arousal,intended_valence,intended_mood
Good Days,SZA,medium,positive,relaxed
```

Use these label values consistently:

- `intended_arousal`: `low`, `medium`, `high`
- `intended_valence`: `negative`, `neutral`, `positive`
- `intended_mood`: `happy`, `sad`, `relaxed`, `tense`, `excited`, `annoyed`, `neutral`, or another short mood label

### 2. Build iTunes Preview Clips

```bash
python src/itunes/build_clips_csv.py
```

This writes:

```text
data/clips.csv
data/input/itunes_candidates.csv
data/input/clip_overrides_template.csv
```

Check `data/clips.csv` to make sure the selected songs are right. The script does not blindly trust the first iTunes result; it scores candidates and saves all candidates.

If a selected song is wrong:

1. Open `data/input/itunes_candidates.csv`.
2. Find the correct `track_id`.
3. Copy `data/input/clip_overrides_template.csv` to `data/input/clip_overrides.csv`.
4. Put the correct `track_id` in `selected_track_id`.
5. Rerun:

```bash
python src/itunes/build_clips_csv.py
```

### 3. Run A Self-Training Session

```bash
streamlit run src/dashboard/app.py
```

Open the Streamlit URL, usually:

```text
http://localhost:8501
```

For each participant:

1. Enter a unique `user_id`, for example `jason`, `p01`, or `participant_03`.
2. Enter a unique `session_id`, for example `s01` or `p03_s01`.
3. Click `Begin / restart session`.
4. Use `Previous` and `Next` to move between songs if needed.
5. For each song, complete the 30-second rest period before listening.
6. Play the 30-second preview, start the song timer, then rate the song.

The dashboard saves after every rating, so losing the browser should not lose the whole session.

Labels are written to:

```text
data/raw/labels/{session_id}_labels.csv
```

Do not reuse the same `session_id` for different participants.

### 4. Extract Audio Features

```bash
python src/features/extract_audio_features.py
```

This downloads each preview to a temporary file, extracts features, deletes the temporary audio, and writes:

```text
data/processed/audio_features.csv
```

Audio files should not be committed. Only the extracted CSV should be committed.

### 5. Generate Fake Biometrics Until Sensors Are Ready

```bash
python src/features/generate_fake_biometrics.py
```

This writes:

```text
data/processed/biometric_features_fake.csv
```

The fake data is only for testing the end-to-end pipeline. It should be replaced by real sensor features for the final sensor-backed analysis.

### 6. Merge The Training Table

With fake biometrics:

```bash
python src/features/merge_features.py
```

With real biometrics after the sensor team has exported them:

```bash
python src/features/merge_features.py --use-real-biometrics
```

This writes:

```text
data/processed/training_table.csv
```

The table includes metadata, labels, audio features, biometric features, and binary labels:

- `preference_binary`
- `arousal_binary`
- `valence_binary`

### 7. Train Baseline Models

```bash
python src/models/train_models.py
```

This trains separate models for:

- preference
- arousal
- valence

Metrics are written to:

```text
models/metrics/
```

Model bundles are written locally as `.joblib` files under `models/`, but those are intentionally ignored by Git. Recreate them by rerunning the training script.

### 8. Generate Recommendations

For one participant/session:

```bash
python src/models/recommend.py --target-mode all --user-id demo_user --session-id demo_session
```

For a real participant, replace the IDs:

```bash
python src/models/recommend.py --target-mode all --user-id p03 --session-id p03_s01
```

This writes:

```text
data/processed/recommendations.csv
```

Recommendation modes:

- `calm`: prefers high predicted preference, low arousal, positive valence
- `hype`: prefers high predicted preference, high arousal, positive valence

### 9. Run The Dashboard

```bash
streamlit run src/dashboard/app.py
```

Open the Streamlit URL, usually:

```text
http://localhost:8501
```

Use the dashboard to collect training labels and inspect model signals. It shows:

- current participant
- current song
- audio player
- previous/next song navigation
- 30-second rest timer before each song
- self-report labels
- fake or replayed HR values
- fake or replayed EDA values
- predicted arousal state
- recommended next song
- separate HR and EDA charts for the current trial

## Quick Full Demo Reset

Use this if you just want to prove the full pipeline runs without collecting new labels:

```bash
source .venv/bin/activate
python src/itunes/build_clips_csv.py
python src/features/extract_audio_features.py
python src/features/generate_demo_labels.py --participants 8
python src/features/generate_fake_biometrics.py
python src/features/merge_features.py
python src/models/train_models.py
python src/models/recommend.py --target-mode all --user-id demo_user --session-id demo_session
streamlit run src/dashboard/app.py
```

## Sensor Team Contract

The sensor team should eventually produce:

```text
data/processed/biometric_features_real.csv
```

It must use this exact schema:

```csv
user_id,session_id,clip_id,hr_mean,hr_max,hr_change_from_baseline,hr_slope,hr_recovery,eda_mean,eda_change_from_baseline,eda_peak_count,eda_max_amplitude,eda_slope,eda_recovery
```

The most important join keys are:

```text
user_id
session_id
clip_id
```

The real sensor file should have one row per participant/session/clip. Once that file exists, the rest of the pipeline can switch from fake to real biometrics with:

```bash
python src/features/merge_features.py --use-real-biometrics
python src/models/train_models.py
```

## What To Commit

Commit:

- source code under `src/`
- docs like `README.md` and this guide
- song request CSVs
- selected clip metadata
- extracted feature CSVs
- label CSVs if they are class/demo data and not private
- model metrics
- recommendations CSVs for demo runs

Do not commit:

- `.venv/`
- `.env`
- downloaded audio files
- `.m4a`, `.mp3`, `.wav`
- model `.joblib` or `.pkl` files
- private participant data unless the group has permission

## Troubleshooting

If `requests`, `streamlit`, or `pandas` is missing:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

If Streamlit says the port is already in use:

```bash
streamlit run src/dashboard/app.py --server.port 8502
```

If audio extraction fails on `.m4a`:

```bash
brew install ffmpeg
python src/features/extract_audio_features.py --force
```

If the model script fails because there are not enough labels, collect more participant sessions or run:

```bash
python src/features/generate_demo_labels.py --participants 8
```

If the dashboard has no recommendation, rerun:

```bash
python src/models/train_models.py
python src/models/recommend.py --target-mode all --user-id demo_user --session-id demo_session
```

## Final Demo Story

The strongest final presentation flow is:

1. Show the iTunes clips list.
2. Run or replay a participant session.
3. Show self-report labels.
4. Show HR/EDA features, fake now or real later.
5. Show arousal prediction.
6. Show calm and hype recommendations.
7. Explain that real sensor features can replace fake features without changing the rest of the pipeline.

The realistic claim is:

BioBeat uses short-term physiological responses, especially HR and EDA changes from baseline, plus audio features and self-reported labels, to personalize mood-based music recommendations.
