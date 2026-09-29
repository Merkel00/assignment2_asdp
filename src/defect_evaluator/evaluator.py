"""Evaluation logic for defect detection results."""
from __future__ import annotations

from typing import Dict

import pandas as pd


def _safe_div(n: float, d: float) -> float:
    return float(n / d) if d != 0 else 0.0


def evaluate_results(df: pd.DataFrame) -> Dict[str, float]:
    """Compute confusion matrix and metrics from results DataFrame.

    Expected columns: actual_label, predicted_label, inference_time_ms
    """
    total = len(df)

    tp = int(((df["actual_label"] == 1) & (df["predicted_label"] == 1)).sum())
    tn = int(((df["actual_label"] == 0) & (df["predicted_label"] == 0)).sum())
    fp = int(((df["actual_label"] == 0) & (df["predicted_label"] == 1)).sum())
    fn = int(((df["actual_label"] == 1) & (df["predicted_label"] == 0)).sum())

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = _safe_div(tp + tn, total)

    return {
        "samples": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "accuracy": accuracy,
    }
