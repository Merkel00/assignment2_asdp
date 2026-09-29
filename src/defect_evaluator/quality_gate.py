"""Quality gate evaluation for experiment results."""
from __future__ import annotations

from typing import Dict, List

DEFAULT_THRESHOLDS = {
    "precision": 0.80,
    "recall": 0.80,
    "f1_score": 0.80,
    "fps": 25.0,
}


def check_quality_gate(metrics: Dict[str, float], fps: float, thresholds: Dict[str, float] | None = None) -> Dict[str, object]:
    thresholds = thresholds or DEFAULT_THRESHOLDS
    failed: List[str] = []

    if metrics.get("precision", 0) < thresholds["precision"]:
        failed.append("precision")
    if metrics.get("recall", 0) < thresholds["recall"]:
        failed.append("recall")
    if metrics.get("f1_score", 0) < thresholds["f1_score"]:
        failed.append("f1_score")
    if fps < thresholds["fps"]:
        failed.append("fps")

    return {"pass": len(failed) == 0, "failed": failed, "thresholds": thresholds}
