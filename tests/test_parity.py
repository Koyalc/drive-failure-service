"""Phase 1.6 parity check: XGBoost predictions must match the exported ONNX model.

Skipped until `python -m src.pipeline.train` has produced a real trained
model -- the dummy ONNX model used by the other tests has no XGBoost
counterpart to compare against.
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

XGB_MODEL_PATH = "artifacts/xgb_model.json"
ONNX_MODEL_PATH = "artifacts/model.onnx"
FEATURE_CONFIG_PATH = "artifacts/feature_config.json"

pytestmark = pytest.mark.skipif(
    not Path(XGB_MODEL_PATH).exists(),
    reason="no trained XGBoost model yet -- run `python -m src.pipeline.train` first",
)


def test_onnx_matches_xgboost_within_tolerance():
    xgb = pytest.importorskip("xgboost")
    config = json.loads(Path(FEATURE_CONFIG_PATH).read_text())
    n_features = len(config["features"])

    rng = np.random.default_rng(0)
    X = rng.normal(size=(1000, n_features)).astype(np.float32)

    booster = xgb.XGBClassifier()
    booster.load_model(XGB_MODEL_PATH)
    xgb_probs = booster.predict_proba(X)[:, 1]

    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    (onnx_probs,) = session.run(None, {input_name: X})

    np.testing.assert_allclose(xgb_probs, onnx_probs.reshape(-1), atol=1e-4)
