import csv
import json
import re
import time

import serial
from serial.tools import list_ports

BAUD = 115200

ports = [
    p.device for p in list_ports.comports()
    if "bluetooth" not in p.device.lower()
    and "debug-console" not in p.device.lower()
]

if not ports:
    raise SystemExit("No USB serial ports found.")

port = sorted(ports, key=lambda p: (not p.startswith("/dev/cu.usb"), p))[0]
print(f"Reading from {port} at {BAUD} baud")
print("Press Ctrl+C to stop.\n")

def parse_line(line):
    raw = line.strip()
    if not raw:
        return None

    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            eda = payload.get("eda") or payload.get("gsr")
            elapsed = payload.get("elapsed_ms") or payload.get("ms") or payload.get("millis")
            return elapsed, eda
    except json.JSONDecodeError:
        pass

    pairs = dict(re.findall(r"([A-Za-z_]+)\s*[:=]\s*(-?\d+(?:\.\d+)?)", raw))
    if pairs:
        eda = pairs.get("EDA") or pairs.get("eda") or pairs.get("GSR") or pairs.get("gsr")
        elapsed = pairs.get("ms") or pairs.get("millis") or pairs.get("elapsed_ms")
        return elapsed, eda

    values = next(csv.reader([raw]))
    if len(values) >= 2:
        return values[0], values[1]
    if len(values) == 1:
        return None, values[0]

    return None

with serial.Serial(port, BAUD, timeout=1) as ser:
    time.sleep(2)
    ser.reset_input_buffer()

    while True:
        raw_bytes = ser.readline()
        if not raw_bytes:
            continue

        raw = raw_bytes.decode("utf-8", errors="replace").strip()
        parsed = parse_line(raw)

        if parsed is None:
            print(f"raw={raw}")
            continue

        elapsed_ms, eda = parsed
        print(f"raw={raw} | elapsed_ms={elapsed_ms} | eda={eda}")