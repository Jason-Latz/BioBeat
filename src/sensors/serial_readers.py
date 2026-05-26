from __future__ import annotations

import csv
import json
import math
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal


SingleValueMetric = Literal["hr", "eda"]


@dataclass
class SensorSample:
    timestamp: str
    hr: float | None
    eda: float | None
    source: str
    raw_line: str = ""


def iso_now() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def list_serial_ports() -> list[str]:
    try:
        from serial.tools import list_ports
    except ImportError:
        return []
    devices = [port.device for port in list_ports.comports()]
    preferred = [
        device
        for device in devices
        if "bluetooth" not in device.lower() and "debug-console" not in device.lower()
    ]
    visible_devices = preferred or devices
    return sorted(visible_devices, key=lambda device: (not device.startswith("/dev/cu.usb"), device))


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(str(value).strip())
    except ValueError:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _sample_from_mapping(values: dict[str, object], raw_line: str, source: str) -> SensorSample | None:
    normalized = {str(key).strip().lower(): value for key, value in values.items()}
    hr = None
    eda = None

    for key in ("hr", "heart_rate", "heartrate", "bpm", "pulse"):
        hr = _float_or_none(normalized.get(key))
        if hr is not None:
            break

    for key in ("eda", "gsr", "conductance", "skin_conductance", "microsiemens", "us"):
        eda = _float_or_none(normalized.get(key))
        if eda is not None:
            break

    if hr is None and eda is None:
        return None
    return SensorSample(timestamp=iso_now(), hr=hr, eda=eda, source=source, raw_line=raw_line)


def parse_sensor_line(
    line: str,
    *,
    source: str = "serial",
    single_value_metric: SingleValueMetric | None = None,
) -> SensorSample | None:
    raw_line = line.strip()
    if not raw_line:
        return None

    try:
        payload = json.loads(raw_line)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        return _sample_from_mapping(payload, raw_line, source)

    key_value_pairs = re.findall(r"([A-Za-z_]+)\s*[:=]\s*(-?\d+(?:\.\d+)?)", raw_line)
    if key_value_pairs:
        return _sample_from_mapping(dict(key_value_pairs), raw_line, source)

    csv_values = next(csv.reader([raw_line]))
    if len(csv_values) >= 2:
        hr = _float_or_none(csv_values[0])
        eda = _float_or_none(csv_values[1])
        if hr is not None or eda is not None:
            return SensorSample(timestamp=iso_now(), hr=hr, eda=eda, source=source, raw_line=raw_line)

    if len(csv_values) == 1 and single_value_metric is not None:
        value = _float_or_none(csv_values[0])
        if value is not None:
            return SensorSample(
                timestamp=iso_now(),
                hr=value if single_value_metric == "hr" else None,
                eda=value if single_value_metric == "eda" else None,
                source=source,
                raw_line=raw_line,
            )

    return None


class MockSensorReader:
    def __init__(self) -> None:
        self.started_at = time.time()
        self.rng = random.Random(42)

    def read_sample(self, timeout: float = 0.2) -> SensorSample:
        time.sleep(min(timeout, 0.15))
        elapsed = time.time() - self.started_at
        hr = 72 + math.sin(elapsed / 5) * 4 + self.rng.gauss(0, 0.7)
        eda = 1.25 + math.sin(elapsed / 7) * 0.12 + self.rng.gauss(0, 0.015)
        return SensorSample(timestamp=iso_now(), hr=round(hr, 3), eda=round(eda, 4), source="mock")

    def close(self) -> None:
        return None


class SerialSensorReader:
    def __init__(
        self,
        port: str,
        baud_rate: int = 115200,
        *,
        source: str = "serial",
        single_value_metric: SingleValueMetric | None = None,
    ) -> None:
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("pyserial is not installed. Run `pip install -r requirements.txt`.") from exc

        self.source = source
        self.single_value_metric = single_value_metric
        self.serial = serial.Serial(port=port, baudrate=baud_rate, timeout=0.2)
        time.sleep(2)
        self.serial.reset_input_buffer()

    def read_sample(self, timeout: float = 0.5) -> SensorSample | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            raw = self.serial.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace")
            sample = parse_sensor_line(
                line,
                source=self.source,
                single_value_metric=self.single_value_metric,
            )
            if sample is not None:
                return sample
        return None

    def close(self) -> None:
        if self.serial.is_open:
            self.serial.close()


def sample_to_row(
    sample: SensorSample,
    *,
    user_id: str,
    session_id: str,
    clip_id: str,
    trial_index: int,
    phase: str,
) -> dict[str, object]:
    return {
        "timestamp": sample.timestamp,
        "user_id": user_id,
        "session_id": session_id,
        "clip_id": clip_id,
        "trial_index": trial_index,
        "phase": phase,
        "hr": sample.hr,
        "eda": sample.eda,
        "source": sample.source,
        "raw_line": sample.raw_line,
    }


def append_sensor_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fieldnames = [
        "timestamp",
        "user_id",
        "session_id",
        "clip_id",
        "trial_index",
        "phase",
        "hr",
        "eda",
        "source",
        "raw_line",
    ]
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)
