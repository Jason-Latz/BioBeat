from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median


def read_trial_values(path: Path) -> dict[int, list[float]]:
    trials: dict[int, list[float]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            trials[int(row["trial_index"])].append(float(row["eda"]))
    return dict(trials)


def classify(values: list[float]) -> tuple[str, float, float]:
    differences = [abs(current - previous) for previous, current in zip(values, values[1:])]
    median_difference = median(differences) if differences else 0.0
    large_jump_fraction = (
        sum(difference >= 0.02 for difference in differences) / len(differences)
        if differences
        else 0.0
    )

    if median_difference >= 0.015 or large_jump_fraction >= 0.35:
        status = "invalid_lead_artifact"
    elif median_difference >= 0.0035 or large_jump_fraction >= 0.02:
        status = "questionable_noise_or_transient"
    else:
        status = "plausible"
    return status, median_difference, large_jump_fraction


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit BioBeat GSR/EDA trials for disconnected-lead artifacts.")
    parser.add_argument("sensor_csv")
    args = parser.parse_args()

    trials = read_trial_values(Path(args.sensor_csv))
    print("trial\tstatus\trows\tmedian_abs_diff\tjump_fraction_ge_0.02")
    for trial_index, values in sorted(trials.items()):
        status, median_difference, large_jump_fraction = classify(values)
        print(
            f"{trial_index}\t{status}\t{len(values)}\t"
            f"{median_difference:.6f}\t{large_jump_fraction:.3f}"
        )


if __name__ == "__main__":
    main()
