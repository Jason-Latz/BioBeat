# BioBeat Project Handoff

Last updated: June 1, 2026

This document is the authoritative handoff for future BioBeat chats. Read it before changing the collector, starting another collection session, interpreting existing data, regenerating song files, training a model, or cleaning up the repository.

The short version is:

- The collector and song catalog are working.
- The repository contains all raw captures, checkpoints, labels, recovery queues, and quality-audit tooling collected so far.
- There are `109` usable real human ratings.
- There is currently **no GSR/EDA block approved for primary biometric training** because the first real run used the sensor with an incorrect fit and the second real run contained a disconnected-lead artifact.
- There are `41` unrated deep-cut songs and `49` already-rated deep-cut songs in the existing recovery queues.
- Jason's next collection request is broader: preserve the historical data, then rerecord the full `240`-clip catalog in explicit descending clip order so deeper-cut and previously uncollected songs are recorded first.
- Do not resume the currently running continuation collector blindly. Validate the repaired hardware first.

## 1. Current State

### Repository

Repository root:

```text
/Users/jason/Downloads/CS Classes/Projects/BioBeat
```

Git remote:

```text
https://github.com/Jason-Latz/BioBeat.git
```

The first completed implementation and data baseline was pushed to `origin/main` on May 31, 2026. That baseline ended at:

```text
9d74aef docs: record collection safeguards and rating terms
```

The open Streamlit session later produced one additional saved rating and one additional sensor capture during hardware troubleshooting. Those stable June 1, 2026 rows, refreshed exports, and corrected recovery queues were checkpointed in:

```text
745fbb6 data: sync continuation hardware-check delta
```

The repository intentionally tracks the raw sensor CSVs and archived checkpoints from the real collection attempts, even though active raw sensor files normally match a `.gitignore` rule. They were added explicitly because Jason asked to push all project data gathered so far.

Local-only files that should remain untracked:

```text
.venv/
src/**/__pycache__/
models/*.joblib
```

### Collector Server

The historical continuation launcher was stopped during clean-reverse preparation. The intended collector URL remains:

```text
http://127.0.0.1:8503/
```

The process was:

```text
.venv/bin/streamlit run src/dashboard/batch_app.py --server.port 8503 --server.address 127.0.0.1
```

For the pre-block hardware check, launch it with:

```text
BIOBEAT_CLIPS_CSV=data/collection_batches/clean_reverse_hardware_check_1.csv
BIOBEAT_CLIP_ORDER=csv
PYTHONPATH=src
```

This produces a one-song ordered hardware-check session only. Do not resume the historical continuation page.

Use port `8503` consistently. Temporary ports such as `8504` were used during earlier validation and caused confusion. Do not direct Jason back to them.

### Hardware

The Seeed GSR/EDA sensor is connected through a Raspberry Pi Pico serial stream. During the successful dry-run setup, the USB serial port was:

```text
/dev/cu.usbmodem101
```

Do not assume the port is unchanged after reconnecting hardware. Recheck the serial-device dropdown manually before another run.

Jason discovered two physical issues after collection:

1. During the first real run, the sensors were worn incorrectly.
2. During the second real run, one lead became unplugged.

The second failure did **not** produce a flat line. It produced a large, repetitive artificial oscillation. A visibly moving chart is therefore not enough to prove that the sensor is connected correctly.

### Apple Watch Heart Rate

Apple Watch HR has not been imported into either real session yet.

Verified active real sensor files:

| Session | Raw sensor rows | Trials captured | EDA rows | HR rows |
| --- | ---: | ---: | ---: | ---: |
| `self_20260531_155844` | `149212` | `61` | `149212` | `0` |
| `self_20260531_220954` | `122322` | `50` | `122322` | `0` |

The raw files currently contain Raspberry Pi Seeed GSR/EDA rows only:

```text
source=raspberry_pi_seeed_gsr
```

Do not claim that HR is present until the Apple Watch CSV import has actually been completed and verified.

## 2. What Was Built

### Guided Real-Data Collector

The real collector is:

```text
src/dashboard/app.py
```

Use this for Raspberry Pi GSR/EDA collection. Do not use `src/experiment/runner_streamlit.py` for real sensor collection; that file is the older ratings-only collector.

Current guided workflow:

1. Start an Apple Watch Workout manually.
2. Enter a user ID and session ID.
3. Select `Raspberry Pi Seeed GSR serial`.
4. Select the real USB serial port, typically `/dev/cu.usbmodem101`.
5. Click `Begin session`.
6. The app automatically records a `20`-second rest period.
7. The app automatically starts the iTunes preview and records a `30`-second listen period.
8. The app asks for emotion and familiarity ratings.
9. The familiarity click saves the label and advances to the next song.
10. At session completion, upload the Apple Watch HR CSV.

Human emotion buttons now use:

```text
1 = negative
2 = neutral
3 = positive
```

Familiarity buttons use:

```text
1 = never heard it before
2 = maybe recognize it
3 = have heard it a few times
4 = know it well
5 = very familiar / personally meaningful / know what is coming
```

### Collector Safety Changes

The collector was hardened after several Streamlit regressions:

- Raw sensor rows flush to disk incrementally every second during a timed phase.
- Rows flush again in `finally` when leaving a timed phase.
- Chart rendering is throttled so Streamlit does not fall behind the serial stream.
- The EDA monitor remains available but is placed one full viewport below the primary controls.
- There is no show/hide EDA widget during active capture because widget reruns reset timers.
- Historical sensor summaries are inside a collapsed expander.
- Session setup is locked while collection is active.
- Previous navigation is disabled during timed phases.
- `Start a new session` appears only after capture reaches a rating or completion state.
- The Raspberry Pi Seeed reader requires `sensor_elapsed_ms`, which filters bare startup lines such as `74` or `36`.

The batch-specific launcher is:

```text
src/dashboard/batch_app.py
```

It reads:

```text
BIOBEAT_CLIPS_CSV
```

and launches the same hardened dashboard using only the selected batch CSV.

### Serial Reader Behavior

The relevant code is:

```text
src/sensors/serial_readers.py
```

The Seeed stream can contain:

```text
milliseconds_since_start,eda
```

The first value must be stored as:

```text
sensor_elapsed_ms
```

It must never be interpreted as heart rate.

The dashboard enables:

```python
require_sensor_elapsed_ms=True
```

for real Raspberry Pi Seeed collection because the serial stream can emit a single bare numeric startup line when the reader opens.

### iTunes Song Tooling

Relevant files:

```text
src/itunes/search_itunes.py
src/itunes/build_clips_csv.py
src/itunes/append_clip_batch.py
```

The iTunes API became rate-limited during song expansion. The tooling was updated to:

- Cache successful candidate lookups.
- Resume only missing songs.
- Persist each successful uncached lookup immediately.
- Retry narrower or simplified search terms after a combined song-and-artist `403`.
- Support conservative request pacing.
- Penalize likely wrong versions such as instrumental, karaoke, tribute, and remix tracks.
- Write isolated candidate caches and override templates for continuation batches.
- Append validated continuation clips to the master registry with duplicate-ID checks.

For deeper-cut continuation batches, use a request delay around:

```text
6.5 seconds
```

Do not make rapid repeated batch calls against the iTunes API. Apple returned `429` rate limits during earlier work.

## 3. Song Catalog History

### Original Expanded Catalog

Jason requested `123` new songs across broad genres, including:

- jazz
- classical
- rap
- EDM
- pop
- 80s
- metal
- Latin
- additional styles

The original list grew to:

```text
150 clips
```

### Familiarity Concern

During the first real session, Jason noticed that too many tracks were familiar. At the clean pause boundary:

```text
60 rated songs
```

Familiarity distribution:

| Familiarity | Count |
| --- | ---: |
| `1` | `22` |
| `2` | `3` |
| `3` | `5` |
| `4` | `24` |
| `5` | `6` |

That meant:

```text
30 / 60 ratings had familiarity 4-5
25 / 60 ratings had familiarity 1-2
```

The first `60` songs remain useful. They were not discarded. Familiarity is a recorded variable and should be used explicitly in later analysis.

### Deeper-Cut Continuation

A separate `90`-song continuation queue was built to avoid wasting more collection time on broadly recognizable tracks:

```text
data/collection_batches/continuation_deep_cuts_90.csv
```

The continuation batch uses clip IDs:

```text
clip_151 ... clip_240
```

All `90 / 90` direct preview URLs returned HTTP `206` during preflight validation.

The master registry now contains:

```text
240 clips
```

in:

```text
data/clips.csv
```

The continuation generation process rejected or replaced several catalog fallbacks before collection. Examples included instrumental or wrong-version matches for Earl Sweatshirt, Madvillain, Run The Jewels, Joey Bada$$, Freddie Gibbs, JPEGMAFIA, and Portishead requests. The final checked continuation CSV contains no flagged instrumental, karaoke, tribute, lullaby, remix, or mixed-version titles.

## 4. Real Collection Attempts

There are two real human-rating sessions.

### Session A: Original Mixed-Familiarity Run

Session:

```text
self_20260531_155844
```

User ID:

```text
Official_run_1
```

Active files:

```text
data/raw/labels/self_20260531_155844_labels.csv
data/raw/sensor/self_20260531_155844_sensor.csv
```

Checkpoint:

```text
data/archive/official_run_1_checkpoint_20260531_170217/
```

Facts:

| Metric | Value |
| --- | ---: |
| Saved ratings | `60` |
| Sensor trials | `61` |
| Unrated captured trial | `61` |
| Raw sensor rows | `149212` |
| HR rows | `0` |
| EDA rows | `149212` |
| Typical rows per trial | about `2445` |

The session was stopped safely after the rating screen appeared. Trial `61` was captured but not rated. It is intentionally ignored by the human-rating export.

Signal characteristics are structurally plausible:

```text
median adjacent-sample difference: roughly 0.0016 to 0.0024
fraction of adjacent jumps >= 0.02: generally near 0.000 to 0.019
```

However, Jason later reported that the sensor was worn incorrectly. Therefore:

- Human ratings remain usable.
- Familiarity remains usable.
- The raw EDA remains preserved for audit and exploratory work.
- This EDA must not be treated as clean primary biometric training data.

### Session B: Deep-Cut Continuation With Hardware Failure

Session:

```text
self_20260531_220954
```

Active files:

```text
data/raw/labels/self_20260531_220954_labels.csv
data/raw/sensor/self_20260531_220954_sensor.csv
```

Checkpoint:

```text
data/archive/continuation_hardware_issue_checkpoint_20260531_231446/
```

Facts at the stable June 1, 2026 handoff:

| Metric | Value |
| --- | ---: |
| Saved ratings | `49` |
| Sensor trials | `50` |
| Unrated captured trial | `50` |
| Raw sensor rows | `122322` |
| HR rows | `0` |
| EDA rows | `122322` |
| Typical rows per trial | about `2446` |

Jason reported that sensor fit was corrected after roughly the first ten songs, then later discovered that one wire had become unplugged.

The first pushed checkpoint for this session contained `48` ratings and `49` sensor captures. The open Streamlit continuation session was later used again during hardware troubleshooting:

| Trial | Clip | Saved rating | Numeric stream status | Interpretation |
| ---: | --- | --- | --- | --- |
| `49` | `clip_202` | `neutral`, familiarity `1` | `questionable_noise_or_transient` | The previously captured trial was rated later. Preserve the rating, but do not treat the EDA as verified. |
| `50` | `clip_209` | none | `questionable_noise_or_transient` | A later hardware-check attempt captured a full rest/listen block but no rating. Treat it as unverified and recollect it in the unrated queue. |

The raw files stopped changing after trial `50` ended at approximately `2026-06-01T00:05:04`. A future chat must still check modification times before assuming the files remain unchanged.

The audit found:

| Trials | Classification | Reason |
| --- | --- | --- |
| `1-16` | `excluded_questionable_sensor_fit` | Sensor placement was still being corrected. |
| `17-31` | `excluded_disconnected_lead_artifact` | Large repetitive artificial oscillation. |
| `32-35` | `excluded_reconnection_transient` | Brief reconnection window; too transient to trust. |
| `36-48` | `excluded_disconnected_lead_artifact` | Large repetitive artificial oscillation returned. |
| `49` | `excluded_unverified_capture` | Rating exists, but the capture has not been verified for primary biometric training. |
| `50` | `excluded_unrated_capture` | Captured but not rated. |

The disconnected-lead artifact is obvious numerically:

| Trials | Median adjacent-sample difference | Median fraction of jumps >= `0.02` |
| --- | ---: | ---: |
| `1-10` | `0.0045` | `0.012` |
| `11-16` | `0.0069` | `0.045` |
| `17-31` | `0.0427` | `0.725` |
| `32-35` | `0.0045` | `0.036` |
| `36-48` | `0.0436` | `0.723` |
| `49` | `0.0064` | `0.049` |
| `50` | `0.0065` | `0.112` |

The failure produced fake variance, not a flat line. This is a critical lesson for future runs.

### Historical Dry Runs

Earlier dry-run files were archived under:

```text
data/archive/pre_real_take_20260531_154526/
```

Important dry-run events:

- `test88` exposed malformed startup values such as bare `74` and `36` lines when serial readers opened.
- The reader was changed to require `sensor_elapsed_ms` for the Seeed dashboard stream.
- `test89` then passed at approximately `49 Hz` with no malformed EDA values, missing elapsed times, timing reversals, or startup outliers.
- Jason manually confirmed that timers continued uninterrupted, audio played correctly, and the EDA monitor stayed below the fold.

Keep these archived files. They document why the startup-line guard exists.

## 5. Data Inventory

### Master Clip Registry

```text
data/clips.csv
```

Contains:

```text
240 clips
```

Important fields:

```text
clip_id
track_name
artist
album
genre
preview_url
track_id
intended_arousal
intended_valence
intended_mood
song_query
artist_query
selection_method
match_score
```

### Active Real Labels

```text
data/raw/labels/self_20260531_155844_labels.csv
data/raw/labels/self_20260531_220954_labels.csv
```

Combined real ratings:

```text
109
```

Current real human-label distribution:

| Mood | Count |
| --- | ---: |
| `negative` | `38` |
| `neutral` | `53` |
| `positive` | `18` |

Combined familiarity distribution:

| Familiarity | Count |
| --- | ---: |
| `1` | `68` |
| `2` | `3` |
| `3` | `6` |
| `4` | `26` |
| `5` | `6` |

### Active Real Sensor CSVs

```text
data/raw/sensor/self_20260531_155844_sensor.csv
data/raw/sensor/self_20260531_220954_sensor.csv
```

Sensor schema:

```text
timestamp
user_id
session_id
clip_id
trial_index
phase
hr
eda
sensor_elapsed_ms
source
raw_line
```

These raw files are intentionally preserved in full. They contain useful audit evidence even where the biometric signal must be excluded from primary training.

### Audited Human-Rating Export

```text
data/processed/good_real_data.csv
```

This is the best single CSV for a future chat to inspect first.

It contains:

```text
109 rows
```

Critical semantic warning:

> `good_real_data.csv` means the human ratings are usable. It does **not** mean the current GSR/EDA readings are approved for primary biometric training.

Every row explicitly includes:

```text
rating_usable=yes
data_scope=human_rating_only
use_eda_for_primary_biometric_training=no
```

It also records:

```text
eda_quality
eda_exclusion_reason
```

### Per-Trial EDA Quality Manifest

```text
data/processed/real_eda_quality_manifest.csv
```

Contains:

```text
111 sensor-capture rows
```

Why `111`, not `109`:

- `109` rated trials
- `2` additional captured but unrated trials

Manifest collection-quality counts:

| Collection quality | Rows |
| --- | ---: |
| `excluded_wrong_sensor_fit` | `60` |
| `excluded_questionable_sensor_fit` | `16` |
| `excluded_disconnected_lead_artifact` | `28` |
| `excluded_reconnection_transient` | `4` |
| `excluded_unverified_capture` | `1` |
| `excluded_unrated_capture` | `2` |

Manifest serial-pattern counts:

| Serial pattern | Rows |
| --- | ---: |
| `plausible` | `61` |
| `questionable_noise_or_transient` | `22` |
| `invalid_lead_artifact` | `28` |

The distinction matters:

- `serial_pattern_quality` describes what can be inferred from the numeric stream.
- `collection_quality` incorporates Jason's physical-hardware observations.
- A numerically plausible signal can still be excluded because the sensor was worn incorrectly.

### Recovery Queues

Unrated deep-cut songs:

```text
data/collection_batches/continuation_unrated_41.csv
```

Contains:

```text
41 clips
```

These are the highest-value next songs because Jason has not rated them during the deep-cut continuation session. `40` have never been played in that session. `clip_209` was played during unverified trial `50`, but no rating was saved, so it remains in this queue for a clean recollection and rating.

Already-rated deep cuts to recollect later:

```text
data/collection_batches/continuation_recollect_49.csv
```

Contains:

```text
49 clips
```

These ratings remain useful, but their EDA should be recollected later if time permits.

Full validated deeper-cut batch:

```text
data/collection_batches/continuation_deep_cuts_90.csv
```

Contains:

```text
90 clips
```

The unrated and recollect queues are disjoint and their union is exactly the full deeper-cut batch:

```text
41 + 49 = 90
```

### Archives

Archives are intentionally non-destructive:

```text
data/archive/pre_real_take_20260531_154526/
data/archive/official_run_1_checkpoint_20260531_170217/
data/archive/continuation_hardware_issue_checkpoint_20260531_231446/
data/archive/pre_clean_reverse_20260601_002708/
data/archive/clean_reverse_hardware_checks_20260601_112050/
```

Do not delete these without explicit approval.

## 6. Label Terminology

Human emotion labels were migrated from:

```text
sad
neutral
happy
```

to:

```text
negative
neutral
positive
```

The new collector writes only the new terms.

Compatibility behavior:

- `src/dashboard/app.py` can still read old `sad` and `happy` values.
- `src/features/merge_features.py` still recognizes legacy `happy` as a positive alias.
- CSV `mood` columns in the repository were migrated to the new terms.

Do not bulk-rewrite `intended_mood` values in the song catalog merely because they include values such as `sad`, `happy`, `relaxed`, `tense`, or `excited`. Those are richer catalog descriptors, not the three-button human-rating field.

## 7. Quality-Audit Tooling

### EDA Audit Script

```text
src/sensors/audit_eda_quality.py
```

Run:

```bash
PYTHONPATH=src .venv/bin/python src/sensors/audit_eda_quality.py path/to/session_sensor.csv
```

The script computes:

```text
median_abs_diff
jump_fraction_ge_0.02
```

Classification thresholds:

```text
invalid_lead_artifact:
  median_abs_diff >= 0.015
  OR jump_fraction_ge_0.02 >= 0.35

questionable_noise_or_transient:
  median_abs_diff >= 0.0035
  OR jump_fraction_ge_0.02 >= 0.02

plausible:
  neither condition above
```

This classifier catches the disconnected-lead oscillation observed during the second real run.

### Curated Export Script

```text
src/features/export_good_real_data.py
```

Run:

```bash
PYTHONPATH=src .venv/bin/python src/features/export_good_real_data.py
```

It regenerates:

```text
data/processed/good_real_data.csv
data/processed/real_eda_quality_manifest.csv
```

The exporter deliberately excludes all current EDA from primary biometric training while retaining every usable human rating.

## 8. Model-Training Implications

The baseline training pipeline currently targets:

```text
preference_binary
arousal_binary
valence_binary
```

in:

```text
src/models/train_models.py
```

Important limitations of the current real data:

- The guided self-collector stores real `valence`, `mood`, and `familiarity`.
- The guided self-collector currently leaves `preference` and `arousal` blank.
- Apple Watch HR has not yet been imported.
- No current real GSR/EDA block is approved for primary biometric training.

Therefore:

- Use `good_real_data.csv` for human-rating analysis, familiarity analysis, and planning recollection.
- Do not merge the current raw EDA into a primary biometric model as if it were clean.
- Do not claim a clean real biometric model exists yet.
- The tracked `data/processed/training_table.csv` is primarily legacy/demo pipeline output, not a validated clean-real-data training table.

Familiarity is a confounder:

- Familiar songs can change EDA because of recognition, anticipation, memory, and personal preference.
- Familiarity should remain visible in analysis.
- Consider stratifying or controlling by familiarity when clean biometric data is eventually collected.

## 9. Next Safe Collection Plan

### Updated Direction: Clean Reverse Rerecord

Jason requested a fresh rerecord of the complete catalog while keeping all historical data. The new run should start at the back of the catalog and work forward:

```text
clip_240
clip_239
...
clip_001
```

This intentionally prioritizes the newer deeper-cut material before the older broadly recognizable songs.

Do not overwrite or delete any existing raw CSV. Every new run must use a fresh session ID so it writes separate files.

The dashboard normally shuffles clips deterministically from the session ID. The batch launcher now supports an opt-in ordered mode:

```text
BIOBEAT_CLIP_ORDER=csv
```

Use that setting for every clean reverse launch. Shuffle behavior remains the default for unrelated runs.

Do not record all `240` clips as one uninterrupted session. A single full run would take several hours and would make another unnoticed hardware failure expensive. The generated descending `40`-clip blocks are:

| Block | Descending clip range | Batch CSV |
| --- | --- | --- |
| `01` | `clip_240` through `clip_201` | `data/collection_batches/clean_reverse_b01_240_201.csv` |
| `02` | `clip_200` through `clip_161` | `data/collection_batches/clean_reverse_b02_200_161.csv` |
| `03` | `clip_160` through `clip_121` | `data/collection_batches/clean_reverse_b03_160_121.csv` |
| `04` | `clip_120` through `clip_081` | `data/collection_batches/clean_reverse_b04_120_081.csv` |
| `05` | `clip_080` through `clip_041` | `data/collection_batches/clean_reverse_b05_080_041.csv` |
| `06` | `clip_040` through `clip_001` | `data/collection_batches/clean_reverse_b06_040_001.csv` |

Run numerical EDA QC after the one-song hardware check and again between every block.

### Step 1: Fix Hardware Physically

Jason should finish repairing and fitting the GSR/EDA hardware before any long collection.

### Step 2: Do One Hardware Dry Run

Do **not** resume the old continuation session.

The isolated one-song queue is:

```text
data/collection_batches/clean_reverse_hardware_check_1.csv
```

It uses `clip_001` so the dry run does not expose a priority deep cut.

Launch the dry-run server:

```bash
BIOBEAT_CLIPS_CSV=data/collection_batches/clean_reverse_hardware_check_1.csv \
BIOBEAT_CLIP_ORDER=csv \
PYTHONPATH=src \
.venv/bin/streamlit run src/dashboard/batch_app.py \
  --server.port 8503 \
  --server.address 127.0.0.1
```

Use a fresh session:

```text
user_id=hardware_check
session_id=hardware_check_reverse_20260601_1
```

Record exactly one complete song, then stop on the rating screen without rating it.

Do not click into the next song.

Run the QC script against the new sensor CSV:

```bash
PYTHONPATH=src .venv/bin/python src/sensors/audit_eda_quality.py data/raw/sensor/hardware_check_reverse_20260601_1_sensor.csv
```

The result should be inspected before Jason invests more time. A moving chart is not sufficient evidence.

### Latest Hardware Check

Jason repeated the one-song hardware check on June 1, 2026. The dashboard retained its auto-generated session ID, so the actual sensor file is:

```text
data/raw/sensor/self_20260601_111921_sensor.csv
```

The file contains one complete `clip_001` capture:

| Metric | Value |
| --- | ---: |
| Total rows | `2439` |
| Rest rows | `982` |
| Listen rows | `1457` |
| Approximate sample rate | `49 Hz` |
| Missing EDA values | `0` |
| Missing elapsed-time values | `0` |
| Non-positive elapsed-time steps | `0` |
| Median adjacent EDA change | `0.0048` |
| Fraction of adjacent EDA changes >= `0.02` | `0.034` |

The conservative audit script labels this `questionable_noise_or_transient`, but the stream does **not** resemble the disconnected-lead artifact. The known unplugged-lead block was roughly `0.04 / 70%+` on the same two change metrics. An earlier accepted final dry run was roughly `0.0025 / 3.3%`.

Interpretation:

- The latest check passes the numeric screen for the previously observed disconnected-lead failure.
- A CSV cannot prove physical sensor placement.
- Jason manually confirmed that both leads were firmly attached and the sensors were worn in the corrected position before block `01` was launched.

Both one-song clean-reverse checks are preserved in:

```text
data/archive/clean_reverse_hardware_checks_20260601_112050/
```

### Active Clean-Reverse Block

Block `01` was launched on `http://127.0.0.1:8503/` after the numeric check and Jason's physical confirmation.

```text
BIOBEAT_CLIPS_CSV=data/collection_batches/clean_reverse_b01_240_201.csv
BIOBEAT_CLIP_ORDER=csv
```

Verified order:

```text
clip_240 Roads - Portishead
...
clip_201 So Hot You're Hurting My Feelings - Caroline Polachek
```

Do not relaunch block `02` until block `01` has ended and its saved EDA CSV has been audited.

### Step 3: Launch The First Ordered Reverse Block

After the one-song dry run passes, stop the dry-run server deliberately and launch block `01`.

The block `01` launch command is:

```bash
BIOBEAT_CLIPS_CSV=data/collection_batches/clean_reverse_b01_240_201.csv \
BIOBEAT_CLIP_ORDER=csv \
PYTHONPATH=src \
.venv/bin/streamlit run src/dashboard/batch_app.py \
  --server.port 8503 \
  --server.address 127.0.0.1
```

Every later block launch must also set `BIOBEAT_CLIP_ORDER=csv`. Do not reuse the historical continuation launcher or rely on the session ID shuffle.

### Step 4: Collect Reverse Blocks

Start with block `01`, then move downward one block at a time. Use a new session ID for every block, for example:

```text
clean_reverse_b01_20260601
clean_reverse_b02_20260601
```

Start a fresh Apple Watch Workout for each block so HR imports can be matched cleanly. After each block, stop and audit the saved EDA CSV before continuing.

### Step 5: Import Apple Watch HR

After each completed clean block, import or upload the corresponding Apple Watch HR CSV and verify per-song coverage before training.

### Existing Recovery Queues

The earlier recovery queues remain useful historical artifacts:

```text
data/collection_batches/continuation_unrated_41.csv
data/collection_batches/continuation_recollect_49.csv
```

The new reverse full-catalog rerecord supersedes them as the primary next collection plan.

## 10. Operational Rules For Future Chats

These rules came from actual failures and should be treated as requirements.

### During Active Capture

- Do not edit Streamlit collector source while Jason is collecting.
- Do not rewrite `data/clips.csv` during an active collection take.
- Do not change the active batch during a take.
- Do not touch the serial port while a timed phase is running.
- Do not use browser automation against Jason's active collector tab.
- Do not click widgets during timed rest/listen phases.
- Do not start a temporary Streamlit validation server that might redirect Jason's tab.
- Do not assume visible EDA variance means the hardware is valid.

### Before A Long Run

- Confirm the intended URL is `http://127.0.0.1:8503/`.
- Confirm the correct isolated batch CSV.
- Confirm the real USB port manually.
- Start an Apple Watch Workout.
- Run a one-song hardware dry run.
- Audit the resulting CSV numerically.
- Ask Jason to perform the final short manual UI check.

### Data Handling

- Archive old self-collection files non-destructively.
- Keep ratings even if biometric quality fails.
- Separate human-rating usability from biometric usability.
- Exclude questionable GSR/EDA from primary training unless explicitly modeled as noisy data.
- Preserve raw evidence and quality manifests.
- Normalize CSV line endings and run `git diff --check` after regeneration.

### iTunes API

- Use cached candidates.
- Resume only missing lookups.
- Persist successes immediately.
- Pace deeper-cut requests around `6.5` seconds apart.
- Avoid repeated rapid preflights against Apple.

## 11. Useful Commands

### Check Git State

```bash
git status -sb
git log --oneline -12
```

### Check Collector Listener

```bash
lsof -nP -iTCP:8503 -sTCP:LISTEN
```

### Compile Relevant Python Files

```bash
PYTHONPATH=src .venv/bin/python -m py_compile \
  src/dashboard/app.py \
  src/dashboard/batch_app.py \
  src/sensors/serial_readers.py \
  src/sensors/audit_eda_quality.py \
  src/features/export_good_real_data.py
```

### Regenerate Audited Exports

```bash
PYTHONPATH=src .venv/bin/python src/features/export_good_real_data.py
```

### Audit Existing Real EDA

```bash
PYTHONPATH=src .venv/bin/python \
  src/sensors/audit_eda_quality.py \
  data/raw/sensor/self_20260531_155844_sensor.csv

PYTHONPATH=src .venv/bin/python \
  src/sensors/audit_eda_quality.py \
  data/raw/sensor/self_20260531_220954_sensor.csv
```

### Check CSV Formatting

```bash
git diff --check
```

## 12. Commit History For This Work

The work was split into scoped commits:

```text
7835669 feat: harden guided sensor collection
4daf057 feat: improve cached iTunes clip generation
b68eb99 data: add expanded clip catalog and deep-cut queue
dfad7d1 feat: standardize valence rating terminology
f75ef48 feat: add EDA quality audit and curated rating export
120f9c5 data: migrate saved ratings to valence terms
d31d5fe data: add raw sensor captures and checkpoints
9d74aef docs: record collection safeguards and rating terms
745fbb6 data: sync continuation hardware-check delta
f1d9782 feat: support ordered batch collection
dc614c7 data: add clean reverse rerecord queues and baseline archive
```

## 13. Final Interpretation

The work is not wasted.

What is valid now:

- `109` real human ratings
- familiarity labels for all `109` rated songs
- a `240`-clip master registry
- a validated `90`-song deeper-cut batch
- `41` unrated deeper-cut songs ready for the next clean session
- complete raw evidence for both real attempts
- per-trial EDA quality labels
- reproducible quality-audit and export tooling

What is not valid yet:

- a clean real GSR/EDA training set
- imported Apple Watch HR for the real sessions
- a clean real multimodal biometric model

The correct next move is not to discard the ratings. Preserve the historical files, add ordered-run support, validate the repaired hardware with one song, and rerecord the full catalog in descending `40`-song blocks with EDA QC between blocks.
