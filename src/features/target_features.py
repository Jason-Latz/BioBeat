from __future__ import annotations

import numpy as np
import pandas as pd


VALENCE_LABELS = ["negative", "neutral", "positive"]
VALENCE_TO_ORDINAL = {"negative": -1, "neutral": 0, "positive": 1}
ORDINAL_TO_VALENCE = {-1: "negative", 0: "neutral", 1: "positive"}

BIOMETRIC_COLUMNS = [
    "hr_mean",
    "hr_max",
    "hr_change_from_baseline",
    "hr_slope",
    "hr_recovery",
    "eda_mean",
    "eda_change_from_baseline",
    "eda_peak_count",
    "eda_max_amplitude",
    "eda_slope",
    "eda_recovery",
]

AROUSAL_SOURCE_COLUMNS = [
    "hr_change_from_baseline",
    "hr_slope",
    "eda_change_from_baseline",
    "eda_peak_count",
    "eda_max_amplitude",
    "eda_slope",
]


def canonical_valence(value: object) -> str | None:
    text = str(value).strip().lower()
    if not text or text == "nan":
        return None
    if text in VALENCE_TO_ORDINAL:
        return text
    try:
        number = float(text)
    except ValueError:
        return None
    if number <= 2:
        return "negative"
    if number >= 4:
        return "positive"
    return "neutral"


def valence_ordinal(value: object) -> float:
    label = canonical_valence(value)
    if label is None:
        return np.nan
    return float(VALENCE_TO_ORDINAL[label])


def ordinal_to_valence(value: float) -> str:
    if pd.isna(value):
        return "neutral"
    rounded = int(np.clip(round(float(value)), -1, 1))
    return ORDINAL_TO_VALENCE[rounded]


def robust_minmax(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    low = float(numeric.quantile(0.05))
    high = float(numeric.quantile(0.95))
    if np.isclose(low, high):
        high = float(numeric.max())
        low = float(numeric.min())
    if np.isclose(low, high):
        return pd.Series(0.5, index=values.index, dtype=float).where(numeric.notna(), np.nan)
    return ((numeric - low) / (high - low)).clip(0.0, 1.0)


def weighted_available(parts: list[tuple[float, pd.Series]]) -> pd.Series:
    total = pd.Series(0.0, index=parts[0][1].index, dtype=float)
    weights = pd.Series(0.0, index=parts[0][1].index, dtype=float)
    for weight, values in parts:
        valid = values.notna()
        total = total + values.fillna(0.0) * weight
        weights = weights + valid.astype(float) * weight
    return total.div(weights.replace(0, np.nan))


def biometric_arousal_score(table: pd.DataFrame) -> pd.Series:
    normalized: dict[str, pd.Series] = {}
    for column in AROUSAL_SOURCE_COLUMNS:
        if column in table.columns:
            normalized[column] = robust_minmax(table[column])
        else:
            normalized[column] = pd.Series(np.nan, index=table.index, dtype=float)

    eda_activation = weighted_available(
        [
            (0.45, normalized["eda_change_from_baseline"]),
            (0.35, normalized["eda_max_amplitude"]),
            (0.20, normalized["eda_peak_count"]),
        ]
    )
    hr_activation = weighted_available(
        [
            (0.70, normalized["hr_change_from_baseline"]),
            (0.30, normalized["hr_slope"]),
        ]
    )
    dynamics = weighted_available(
        [
            (0.65, normalized["eda_slope"]),
            (0.35, normalized["hr_slope"]),
        ]
    )

    score = weighted_available(
        [
            (0.60, eda_activation),
            (0.30, hr_activation),
            (0.10, dynamics),
        ]
    )
    score = score.where(score.notna(), np.nan)
    return score.clip(0.0, 1.0)


def add_model_targets(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    if "valence" in table.columns:
        table["valence_label"] = table["valence"].map(canonical_valence)
        table["valence_ordinal"] = table["valence"].map(valence_ordinal)
    else:
        table["valence_label"] = np.nan
        table["valence_ordinal"] = np.nan

    table["arousal_score"] = biometric_arousal_score(table)

    if "preference" in table.columns:
        table["preference"] = pd.to_numeric(table["preference"], errors="coerce")
        table["preference_binary"] = (table["preference"] >= 4).astype(int)
    if "arousal" in table.columns:
        table["arousal"] = pd.to_numeric(table["arousal"], errors="coerce")
        table["arousal_binary"] = (table["arousal"] >= 4).astype(int)
    table["valence_binary"] = (table["valence_label"] == "positive").astype(int)
    return table
