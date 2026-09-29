#!/usr/bin/env python3
"""Entry point for the Industrial Defect Detection Evaluation Tool."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.defect_evaluator.io_utils import read_results_csv
from src.defect_evaluator.evaluator import evaluate_results
from src.defect_evaluator.quality_gate import check_quality_gate
from src.defect_evaluator.visualization import plot_metrics


def print_report(results: dict, avg_time_ms: float, fps: float, gate_status: dict) -> None:
    print("INDUSTRIAL DEFECT DETECTION EVALUATION")
    print("--------------------------------------")
    print(f"Samples: {results['samples']}")
    print()
    print("Confusion Matrix")
    print(f"TP: {results['tp']}")
    print(f"TN: {results['tn']}")
    print(f"FP: {results['fp']}")
    print(f"FN: {results['fn']}")
    print()
    print("Performance Metrics")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-score: {results['f1_score']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print()
    print("Runtime Performance")
    print(f"Average inference time: {avg_time_ms:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print()
    print("Quality Gate")
    status = "PASS" if gate_status["pass"] else "FAIL"
    print(f"Status: {status}")
    if not gate_status["pass"]:
        print("Failed metrics:")
        for m in gate_status["failed"]:
            print(f" - {m}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate industrial defect detection experiment results from CSV"
    )
    parser.add_argument("--input", "-i", required=True, help="Path to results CSV")
    parser.add_argument("--output", "-o", default="output", help="Output folder")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        df = read_results_csv(input_path)
    except Exception as e:
        print(f"Error reading input: {e}")
        return 2

    metrics = evaluate_results(df)

    avg_time_ms = df["inference_time_ms"].mean() if not df.empty else 0.0
    fps = 1000.0 / avg_time_ms if avg_time_ms > 0 else 0.0

    gate = check_quality_gate(metrics, fps=fps)

    # Save reports
    csv_out = out_dir / "evaluation_report.csv"
    json_out = out_dir / "evaluation_report.json"
    chart_out = out_dir / "metrics_chart.png"

    # flatten metrics for CSV/JSON
    report = {
        "samples": metrics["samples"],
        "tp": metrics["tp"],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"],
        "accuracy": metrics["accuracy"],
        "average_inference_time_ms": avg_time_ms,
        "fps": fps,
        "quality_gate_pass": gate["pass"],
        "quality_gate_failed_metrics": gate["failed"],
    }

    # save csv
    try:
        import pandas as pd

        pd.DataFrame([report]).to_csv(csv_out, index=False)
    except Exception:
        pass

    with json_out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    plot_metrics(metrics, chart_out)

    print_report(metrics, avg_time_ms, fps, gate)
    print(f"\nReports written to: {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
