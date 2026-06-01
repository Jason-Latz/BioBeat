# Clean Reverse Block 02 Remote Runbook

Block `02` is an `80`-song ordered collection run:

```text
clip_200
clip_199
...
clip_121
```

Use a distinct participant ID for the other collector and a fresh session ID. Do not reuse Jason's identity or block `01` session.

## 1. Prepare The Computer

Pull the latest `main` branch. From the repository root:

```bash
git pull --ff-only origin main
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The virtual-environment setup is needed only once on that computer.

## 2. Check The Hardware

Before the `80`-song run:

1. Attach both Seeed GSR/EDA leads firmly.
2. Wear the sensors in the corrected position.
3. Connect the Raspberry Pi Pico USB serial cable.
4. Start a fresh Apple Watch Workout if Apple Watch HR will be imported.
5. Launch the one-song hardware-check queue:

```bash
BIOBEAT_CLIPS_CSV=data/collection_batches/clean_reverse_hardware_check_1.csv \
BIOBEAT_CLIP_ORDER=csv \
PYTHONPATH=src \
.venv/bin/streamlit run src/dashboard/batch_app.py \
  --server.port 8503 \
  --server.address 127.0.0.1
```

Open `http://127.0.0.1:8503/`, select the real `/dev/cu.usbmodem*` port, complete one song, and stop the server. Audit the newly created sensor CSV:

```bash
PYTHONPATH=src .venv/bin/python \
  src/sensors/audit_eda_quality.py \
  data/raw/sensor/<new-session-id>_sensor.csv
```

Do not begin block `02` until the hardware-check CSV has been reviewed. A visibly moving chart is not enough.

## 3. Launch Block 02

After the hardware check passes:

```bash
./scripts/run_clean_reverse_block02_remote.sh
```

The launcher validates that the queue contains exactly `80` preview URLs in descending `clip_200` through `clip_121` order and refuses to launch if port `8503` is already occupied.

Open `http://127.0.0.1:8503/`. Before clicking **Begin session**:

1. Enter a distinct participant ID for the person doing block `02`.
2. Confirm the fresh session ID.
3. Select Raspberry Pi Seeed GSR serial.
4. Confirm the real `/dev/cu.usbmodem*` port.
5. Confirm the Apple Watch Workout is running if HR will be imported.

Complete all `80` songs in one take. Do not click other widgets during timed rest or listen phases. After the final rating, stop and audit the new EDA CSV before collecting another block.
