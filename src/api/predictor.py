import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.api.schemas import DriveRecord
from src.feature_spec import SMART_ATTRIBUTES, delta7d_column


class Predictor:
    def __init__(self, model_path: str, feature_config_path: str):
        config = json.loads(Path(feature_config_path).read_text())
        self.model_version: str = config["model_version"]
        self.feature_names: list[str] = config["features"]
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_name = self.session.get_inputs()[0].name

    def _feature_values(self, r: DriveRecord) -> dict[str, float]:
        """Same delta definition as src.pipeline.features.add_delta_features: this
        week's raw reading minus the reading from ~7 days ago, per SMART attribute."""
        values = {"capacity_bytes": r.capacity_bytes}
        for n in SMART_ATTRIBUTES:
            raw = getattr(r, f"smart_{n}_raw")
            values[f"smart_{n}_raw"] = raw
            values[f"smart_{n}_normalized"] = getattr(r, f"smart_{n}_normalized")
            values[delta7d_column(n)] = raw - getattr(r, f"smart_{n}_raw_7d_ago")
        return values

    def predict(self, records: list[DriveRecord]) -> list[float]:
        rows = [
            [self._feature_values(r)[name] for name in self.feature_names] for r in records
        ]
        batch = np.array(rows, dtype=np.float32)
        # convert_xgboost's classifier output is [P(no-failure), P(failure)] per row.
        (probabilities,) = self.session.run(["probabilities"], {self._input_name: batch})
        return probabilities[:, 1].tolist()
