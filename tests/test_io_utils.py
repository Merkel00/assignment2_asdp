import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from defect_evaluator.io_utils import read_results_csv
import pandas as pd


def test_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    df = pd.DataFrame([{"a": 1, "b": 2}])
    df.to_csv(p, index=False)
    with pytest.raises(ValueError):
        read_results_csv(p)


def test_empty_file(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    with pytest.raises(Exception):
        read_results_csv(p)
