# BioBeat

BioBeat is a music recommendation prototype that combines short music previews, self-reported mood labels, audio features, and biometric features to train mood-aware recommendation models.

The current implementation has a training collector for self-runs. It can record mock HR/EDA data for testing, read EDA/GSR values from a Seeed GSR sensor connected through a Raspberry Pi Pico serial stream during collection, and import Apple Watch/Apple Health heart-rate CSV rows after a session. It also extracts audio features with `librosa`, turns raw sensor streams into biometric features, and trains baseline models.

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
python src/sensors/import_apple_watch_hr.py --health-csv path/to/apple_watch_hr.csv --labels-file data/raw/labels/jason_s01_labels.csv
python src/features/extract_audio_features.py
python src/features/extract_biometric_features.py
python src/features/merge_features.py --use-real-biometrics
python src/models/train_models.py
python src/models/recommend.py --target-mode all --user-id demo_user
```

The dashboard is the training collection UI. It first reminds the participant to start an Apple Watch Workout, then runs a streamlined loop: 20-second rest recording, automatic 30-second song playback/recording, then two large rating-button prompts for emotion and familiarity. Raw sensor samples are saved to `data/raw/sensor/{session_id}_sensor.csv`.

The Raspberry Pi Pico stream for the Seeed GSR sensor can print any of these formats:

```text
GSR:1.42
EDA:1.42
1.42
{"gsr":1.42}
```

Apple Watch HR is imported at the end of the dashboard session from a CSV exported from Apple Health or a Health export app. The upload panel matches HR timestamps to the rest/listen windows in the labels file, previews per-song HR sample coverage, then appends HR-only rows to the same raw sensor CSV. If the Seeed stream includes milliseconds since sensor start, BioBeat stores that as `sensor_elapsed_ms`; Apple Watch sync still uses the dashboard's wall-clock timestamps.

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

Experiment labels are written incrementally to `data/raw/labels/{session_id}_labels.csv`. Raw sensor samples are written to `data/raw/sensor/{session_id}_sensor.csv`. It is fine for individual rows to contain only HR or only EDA; the feature extractor summarizes whichever values are present.

For the audited real-session export, run:

```bash
PYTHONPATH=src .venv/bin/python src/features/export_good_real_data.py
```

This writes `data/processed/good_real_data.csv` with usable human ratings and `data/processed/real_eda_quality_manifest.csv` with the per-trial EDA audit. The current curated ratings export deliberately marks EDA as excluded from primary biometric training because the recorded sessions contain known sensor-fit and disconnected-lead issues.

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
