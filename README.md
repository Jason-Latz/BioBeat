# BioBeat

BioBeat is a music recommendation prototype that combines short music previews, self-reported mood labels, audio features, and biometric features to train mood-aware recommendation models.

The current implementation has a training collector for self-runs. It can record mock HR/EDA data for testing, or read Arduino-over-USB serial data when the hardware is ready. It also extracts audio features with `librosa`, turns raw sensor streams into biometric features, and trains baseline models.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `.m4a` audio loading fails during feature extraction, install `ffmpeg`:

```bash
brew install ffmpeg
```

## Main Workflow

```bash
python src/itunes/build_clips_csv.py
streamlit run src/dashboard/app.py
python src/features/extract_audio_features.py
python src/features/extract_biometric_features.py
python src/features/merge_features.py --use-real-biometrics
python src/models/train_models.py
python src/models/recommend.py --target-mode all --user-id demo_user
```

The dashboard is the training collection UI. It guides a participant through a 30-second rest recording, a 30-second song recording, and a rating form for each song. Raw sensor samples are saved to `data/raw/sensor/{session_id}_sensor.csv`.

Arduino serial lines can be any of these formats:

```text
HR:72,EDA:1.42
72,1.42
{"hr":72,"eda":1.42}
```

For a fully local demo without collecting labels by hand, generate synthetic labels first:

```bash
python src/features/generate_demo_labels.py
```

For group workflow, role ownership, and demo-day instructions, read `GROUP_USAGE_GUIDE.md`.

## Data Contract

Starter song requests live in `data/input/desired_songs.csv`. The iTunes builder writes:

- `data/input/itunes_candidates.csv`: all candidate search results
- `data/input/clip_overrides_template.csv`: editable selection template
- `data/clips.csv`: final selected preview clips

Experiment labels are written incrementally to `data/raw/labels/{session_id}_labels.csv`. Raw sensor samples are written to `data/raw/sensor/{session_id}_sensor.csv`.

The sensor feature extractor writes `data/processed/biometric_features_real.csv` using this schema:

```csv
user_id,session_id,clip_id,hr_mean,hr_max,hr_change_from_baseline,hr_slope,hr_recovery,eda_mean,eda_change_from_baseline,eda_peak_count,eda_max_amplitude,eda_slope,eda_recovery
```

As long as the schema matches, the merger, model training, recommender, and dashboard can reuse the same pipeline.

## Project Structure

```text
data/
  input/
  raw/
    labels/
    sensor/
  processed/
src/
  itunes/
  experiment/
  features/
  models/
  dashboard/
notebooks/
```
