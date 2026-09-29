import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from defect_evaluator.quality_gate import check_quality_gate


def test_quality_gate_pass():
    metrics = {"precision": 0.9, "recall": 0.85, "f1_score": 0.87}
    fps = 30.0
    res = check_quality_gate(metrics, fps)
    assert res["pass"] is True


def test_quality_gate_fail():
    metrics = {"precision": 0.7, "recall": 0.79, "f1_score": 0.74}
    fps = 20.0
    res = check_quality_gate(metrics, fps)
    assert res["pass"] is False
    assert "precision" in res["failed"]
    assert "recall" in res["failed"] or "fps" in res["failed"]
