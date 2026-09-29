"""I/O utilities for reading experiment CSVs."""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

REQUIRED_COLUMNS: List[str] = [
    "sample_id",
    "actual_label",
    "predicted_label",
    "confidence",
    "inference_time_ms",
]


def read_results_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Input CSV is empty")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # enforce types
    df = df.copy()
    df["actual_label"] = df["actual_label"].astype(int)
    df["predicted_label"] = df["predicted_label"].astype(int)
    df["confidence"] = df["confidence"].astype(float)
    df["inference_time_ms"] = df["inference_time_ms"].astype(float)

    return df
