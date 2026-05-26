# BioBeat Usage Guide

This is the practical guide for using BioBeat right now.

There are basically two jobs:

1. Jason runs the music/rating/training pipeline.
2. The sensor team gets Arduino HR/EDA values streaming over USB.

Everything else is just files and commands.

## What BioBeat Does Right Now

BioBeat is currently a training data collector.

For each song it:

1. Records 30 seconds of rest/baseline sensor data.
2. Plays a 30-second iTunes preview.
3. Records HR/EDA during the song.
4. Asks you to rate how you felt.
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

1. Enter a `user_id`, like `jason`.
2. Enter a `session_id`, like `jason_s01`.
3. Choose `Mock sensor` if hardware is not ready.
4. Choose `Arduino USB` if the sensor team has the Arduino plugged in.
5. Click `Begin / restart session`.
6. For each song:
   - record 30 seconds of rest
   - click `Start song + record sensors`
   - the app starts the song preview and records 30 seconds of song response
   - rate the song

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

`preference`: how much you liked it.

- `1` = disliked it
- `5` = liked it a lot

`arousal`: how activated or energized you felt.

- `1` = calm/sleepy
- `5` = energized/hyped

`valence`: how positive or negative the feeling was.

- `1` = negative/unpleasant/sad
- `5` = positive/pleasant/happy

Valence is not the same as arousal. A song can be:

- low arousal, positive valence: calm and pleasant
- low arousal, negative valence: sad or heavy
- high arousal, positive valence: exciting
- high arousal, negative valence: tense or stressful

For the project, arousal is the main target because HR and EDA are most directly related to activation. Valence is secondary and harder to predict, but it helps separate “calm and pleasant” from “calm and sad.”

`mood` is different from valence. Mood is just an optional human-readable tag, like `sad`, `relaxed`, or `excited`. Valence is the numeric training label. If you are unsure, trust the valence slider and pick the closest mood tag.

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

The Arduino should print one HR/EDA sample per line over USB serial.

Preferred format:

```text
HR:72,EDA:1.42
```

Also accepted:

```text
72,1.42
{"hr":72,"eda":1.42}
```

Use:

- HR in beats per minute
- EDA/GSR in one consistent unit
- a steady sample rate if possible

The dashboard records these lines into:

```text
data/raw/sensor/{session_id}_sensor.csv
```

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
python src/models/recommend.py --target-mode all --user-id jason --session-id jason_s01
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
2. It records rest and song HR/EDA.
3. It asks for self-report ratings.
4. It extracts audio and biometric features.
5. It trains simple models, especially for arousal.
6. It uses those predictions to rank calm or hype recommendations.

The honest claim:

BioBeat does not magically know music taste from physiology. It uses short-term HR/EDA changes, audio features, and self-reported labels to build a simple personalized mood-based recommender.
