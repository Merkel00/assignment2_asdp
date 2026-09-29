# Suggested GitHub Issues

## 1. Add mAP50 support

**Description**
Add computation and reporting of mean Average Precision at IoU=0.5 (mAP50) for visual detection experiments.

**Motivation**
mAP50 is a standard object detection metric commonly reported in vision research and will make the evaluation output more comparable to YOLOv8 baselines.

**Acceptance criteria**
- Implement mAP50 computation for per-sample predictions (or provide a wrapper invoking pycocotools when available).
- Include tests and an example in the README.

## 2. Add comparison of multiple model experiments

**Description**
Allow the tool to ingest multiple CSV experiment result files and generate a comparative report and plots.

**Motivation**
Researchers commonly run hyperparameter sweeps and multiple model variants; side-by-side comparison helps select the best model.

**Acceptance criteria**
- CLI accepts multiple input files or a directory of CSVs.
- Output contains aggregated table comparing precision/recall/F1/FPS per experiment and a chart.
- Tests for aggregation logic.
