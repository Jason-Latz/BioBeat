# BioBeat

BioBeat is a music recommendation prototype that combines short music previews, self-reported mood labels, audio features, and biometric features to recommend songs for calm or hype listening modes.

The current implementation works without physical sensors. It uses iTunes 30-second previews, extracts audio features with `librosa`, generates plausible fake HR/EDA features, trains baseline models, and runs a Streamlit experiment runner plus replay dashboard.

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
python src/features/generate_fake_biometrics.py
python src/features/merge_features.py
python src/models/train_models.py
python src/models/recommend.py --target-mode all --user-id demo_user
```

The dashboard is the preferred self-training collection UI. It guides a participant through a 30-second rest period, a 30-second music preview, and a rating form for each song.

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

Experiment labels are written incrementally to `data/raw/labels/{session_id}_labels.csv`.

The sensor team can later replace `data/processed/biometric_features_fake.csv` with `data/processed/biometric_features_real.csv` using the same schema:

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
