import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.api.schemas import DriveRecord


class Predictor:
    def __init__(self, model_path: str, feature_config_path: str):
        config = json.loads(Path(feature_config_path).read_text())
        self.model_version: str = config["model_version"]
        self.feature_names: list[str] = config["features"]
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self._input_name = self.session.get_inputs()[0].name

    def predict(self, records: list[DriveRecord]) -> list[float]:
        rows = [[getattr(r, name) for name in self.feature_names] for r in records]
        batch = np.array(rows, dtype=np.float32)
        (probabilities,) = self.session.run(None, {self._input_name: batch})
        return probabilities.reshape(-1).tolist()
