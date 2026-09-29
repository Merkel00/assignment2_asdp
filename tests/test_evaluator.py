import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from defect_evaluator.evaluator import evaluate_results
import pandas as pd


def test_confusion_and_metrics():
    df = pd.DataFrame([
        {"actual_label": 1, "predicted_label": 1, "inference_time_ms": 10},
        {"actual_label": 1, "predicted_label": 0, "inference_time_ms": 10},
        {"actual_label": 0, "predicted_label": 1, "inference_time_ms": 10},
        {"actual_label": 0, "predicted_label": 0, "inference_time_ms": 10},
    ])

    metrics = evaluate_results(df)
    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1

    # precision = tp / (tp+fp) = 1/2
    assert pytest.approx(metrics["precision"], rel=1e-6) == 0.5
    assert pytest.approx(metrics["recall"], rel=1e-6) == 0.5
    # f1 = 2 * p * r / (p + r) = 0.5
    assert pytest.approx(metrics["f1_score"], rel=1e-6) == 0.5
    assert pytest.approx(metrics["accuracy"], rel=1e-6) == 0.5
