"""Visualization utilities: simple metrics bar chart."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt


def plot_metrics(metrics: Dict[str, float], out_path: Path) -> None:
    labels = ["Precision", "Recall", "F1-score", "Accuracy"]
    values = [
        metrics.get("precision", 0.0),
        metrics.get("recall", 0.0),
        metrics.get("f1_score", 0.0),
        metrics.get("accuracy", 0.0),
    ]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["#2b8cbe", "#2ca25f", "#6a51a3", "#de2d26"])  # simple palette
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Evaluation Metrics")
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
