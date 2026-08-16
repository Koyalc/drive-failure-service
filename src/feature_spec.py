"""SMART attribute selection and feature-naming, shared by training (polars/xgboost)
and serving (predictor.py) without pulling either side's heavy deps into the other.

The serving image (requirements.txt) intentionally excludes polars/xgboost/mlflow to
stay small -- this module must stay free of those imports so predictor.py can use it.
"""

SMART_ATTRIBUTES = [5, 9, 187, 188, 194, 197, 198, 241, 242]

BASE_COLUMNS = ["date", "serial_number", "model", "capacity_bytes", "failure"]

SMART_COLUMNS = [f"smart_{n}_raw" for n in SMART_ATTRIBUTES] + [
    f"smart_{n}_normalized" for n in SMART_ATTRIBUTES
]


def delta7d_column(n: int) -> str:
    return f"smart_{n}_raw_delta7d"
