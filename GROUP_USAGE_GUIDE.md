# BioBeat Usage Guide

This is the practical guide for using BioBeat right now.

There are basically two jobs:

1. Jason runs the music/rating/training pipeline.
2. The sensor team gets Seeed GSR/EDA values streaming from the Raspberry Pi Pico H over serial while Jason imports Apple Watch HRV from CSV after the run.

Everything else is just files and commands.

## What BioBeat Does Right Now

BioBeat is currently a training data collector.

For each song it:

1. Records 20 seconds of rest/baseline sensor data.
2. Plays a 30-second iTunes preview.
3. Records GSR/EDA during the song and leaves room for Apple Watch HRV to be imported later.
4. Asks for quick emotion and familiarity ratings.
5. Saves the labels and raw sensor samples.

After that, Python scripts turn the saved data into features and train simple models.

## Setup

Run this once:

```bash
git clone https://github.com/Jason-Latz/BioBeat.git
cd BioBeat
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For later sessions:

```bash
cd BioBeat
source .venv/bin/activate
```

If audio extraction fails later:

```bash
brew install ffmpeg
```

## Choose Songs

Edit:

```text
data/input/desired_songs.csv
```

Each row is:

```csv
song_query,artist_query,intended_arousal,intended_valence,intended_mood
Feather,Nujabes,medium,positive,relaxed
```

Use these rough labels:

- `intended_arousal`: `low`, `medium`, `high`
- `intended_valence`: `negative`, `neutral`, `positive`
- `intended_mood`: `happy`, `sad`, `relaxed`, `tense`, `excited`, `annoyed`, `neutral`

Then rebuild the iTunes preview list:

```bash
python src/itunes/build_clips_csv.py
```

This creates:

```text
data/clips.csv
```

Check that file to make sure iTunes picked the right songs. If it picked the wrong version, look at:

```text
data/input/itunes_candidates.csv
```

## Run A Training Session

Start the app:

```bash
streamlit run src/dashboard/app.py
```

Open:

```text
http://localhost:8501
```

Then:

1. Start an Apple Watch Workout and keep it running for the full BioBeat session.
2. Enter a `user_id`, like `jason`.
3. Enter a `session_id`, like `jason_s01`.
4. Choose `Mock sensor` if hardware is not ready.
5. Choose `Raspberry Pi Seeed GSR serial` if the sensor team has the Pico serial stream connected.
6. Click `Begin session`.
7. For each song:
   - the app automatically records 20 seconds of rest
   - the app automatically starts the song preview and records 30 seconds of song GSR response
   - click one emotion button and one familiarity button; after that, the app saves and moves to the next rest period
8. Stop the Apple Watch Workout after the last song, export heart-rate data as CSV, and upload it on the completion screen.

The app saves as you go.

Labels go here:

```text
data/raw/labels/{session_id}_labels.csv
```

Raw sensor samples go here:

```text
data/raw/sensor/{session_id}_sensor.csv
```

## What The Ratings Mean

You rate each song after hearing it.

`emotion`: how the song made you feel.

- `negative` = sad, angry, tense, uneasy, or otherwise negative; it does not mean dislike
- `neutral` = emotionally mixed or not clearly positive/negative
- `positive` = pleasant, happy, peaceful, excited, or otherwise positive

Older collected files may contain numeric `valence` values `1`, `2`, and `3` beside a string `mood` column. The modeling pipeline uses the string `mood` label when it is `negative`, `neutral`, or `positive`, and writes normalized training output as string valence labels.

`familiarity`: how familiar the song felt.

- `1` = never heard it before
- `2` = maybe recognize it
- `3` = have heard it a few times
- `4` = know it well
- `5` = very familiar / personally meaningful / know what is coming

## Do You Need To Rate The Songs Yourself?

Yes, if you want a model personalized to you.

The model needs examples of:

- what your body did during each song
- how you rated that song afterward

Without your ratings, the app can still use demo/fake data, but it will not learn your preferences or your mood responses.

Minimum useful amount:

- one full pass through all songs is enough to test the pipeline
- several sessions are better
- more participants are better if the final model is supposed to generalize beyond you

For a class demo, one or two full self-runs is a reasonable starting point.

## Sensor Team Instructions

The Raspberry Pi Pico should print one Seeed GSR/EDA sample per line over serial.

Preferred format:

```text
GSR:1.42
```

Also accepted:

```text
EDA:1.42
1.42
1234,1.42
{"gsr":1.42}
```

Use:

- EDA/GSR in one consistent unit
- a steady sample rate if possible
- if sending `milliseconds,gsr`, the first value is stored as `sensor_elapsed_ms`, not HR

The dashboard records these lines into:

```text
data/raw/sensor/{session_id}_sensor.csv
```

## Apple Watch HR Import

Apple Watch heart rate is not treated as a live serial sensor. After a collection session, export heart-rate samples from Apple Health as XML or from a Health export app as CSV, then upload the file on the dashboard completion screen. The app previews how many HR samples matched each rest/listen window before importing.

You can also import from the command line:

```bash
python src/sensors/import_apple_watch_hr.py --health-csv path/to/apple_watch_hr.csv --labels-file data/raw/labels/jason_s01_labels.csv
```

The importer finds a timestamp column and an HR/BPM/value column, matches samples to each rest/listen window, and writes HR-only rows into:

```text
data/raw/sensor/{session_id}_sensor.csv
```

If the Health file clock is offset from the dashboard clock, add `--time-offset-seconds`.

## Train The Model

After collecting a session, run:

```bash
python src/features/extract_audio_features.py
python src/features/extract_biometric_features.py
python src/features/merge_features.py --use-real-biometrics
python src/models/train_models.py
```

This creates:

```text
data/processed/audio_features.csv
data/processed/biometric_features_real.csv
data/processed/training_table.csv
models/metrics/
```

To generate recommendation rankings:

```bash
python src/models/calibrate_user.py --user-id jason --session-id jason_s01 --max-clips 5
python src/models/recommend.py --target-mood all --user-id jason
```

Replace `jason` and `jason_s01` with the IDs you used in the dashboard.

## If Sensors Are Not Ready

Use mock data:

```bash
streamlit run src/dashboard/app.py
```

Choose `Mock sensor`.

Or regenerate the demo dataset:

```bash
python src/features/generate_demo_labels.py --participants 8
python src/features/generate_fake_biometrics.py
python src/features/merge_features.py
python src/models/train_models.py
```

Mock/fake data is only for testing the pipeline. Real results need real sensor recordings.

## What Not To Commit

Do not commit:

- `.venv/`
- downloaded audio files
- `.m4a`, `.mp3`, `.wav`
- `.joblib` or `.pkl` model files
- private raw sensor recordings unless everyone agrees

Commit source code, docs, song lists, selected clip metadata, and non-private demo CSVs.

## Final Demo Story

The clean story is:

1. BioBeat plays short music previews.
2. It records rest/song GSR and imports Apple Watch HR.
3. It asks for self-report ratings.
4. It extracts audio and biometric features.
5. It trains simple models, especially for arousal.
6. It uses those predictions to rank calm or hype recommendations.

The honest claim:

BioBeat does not magically know music taste from physiology. It uses short-term HR/EDA changes, audio features, and self-reported labels to build a simple personalized mood-based recommender.
