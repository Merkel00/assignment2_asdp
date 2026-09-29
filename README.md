# Industrial Defect Detection Evaluation Tool

Small Python CLI tool to evaluate defect-detection experiment results for industrial quality control research. This repository contains a compact, well-tested evaluation module suitable for university assignments and reproducible experiments.

See also: research context uses YOLOv8 (visual detection), LSTM (telemetry), late fusion, and explainable AI (Grad-CAM / EigenCAM). This tool focuses on experiment evaluation only.

Features
- Compute confusion matrix (TP/TN/FP/FN)
- Precision, recall, F1-score, accuracy
- Average inference time and FPS
- Quality gate checks
- Output CSV/JSON and a simple metrics bar chart (PNG)
- Automated tests with pytest and CI via GitHub Actions

Installation

1. Create virtual environment (Python 3.10+ recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Running

```bash
python main.py --input data/sample_results.csv
```

This writes outputs into `output/` and prints a clean report to the terminal.

Project structure

See the repository tree for file locations. Source code lives in `src/defect_evaluator`.

Technology justification
- Python: standard for ML research and scientific tooling.
- pandas: convenient CSV processing and aggregations.
- matplotlib: simple scientific plotting for publication-ready charts.
- pytest: lightweight test runner used in CI.
- GitHub Actions: simple CI to run tests automatically on push/PR.

Limitations and notes
- This tool evaluates experiment outputs only — it is not a full production defect-detection system.
- The wider research system includes YOLOv8 + LSTM + late fusion; this project uses results CSVs produced by such experiments.

License

MIT. See LICENSE file.
