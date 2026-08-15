"""Builds a placeholder ONNX model (random logistic weights) matching feature_config.json.

Mirrors onnxmltools' convert_xgboost output interface -- a "label" tensor and a
"probabilities" tensor shaped [N, 2] (P(no-failure), P(failure)) -- rather than
a single [N, 1] tensor, so the dev fixture exercises the same contract
Predictor and the real ONNX export use. Lets the API, tests, and Docker build
be exercised end-to-end before Phase 1 training produces a real model. Replace
artifacts/model.onnx via `make train` once you have data.
"""
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


def build_dummy_model(feature_config_path: str, output_path: str, seed: int = 0) -> None:
    config = json.loads(Path(feature_config_path).read_text())
    n_features = len(config["features"])

    rng = np.random.default_rng(seed)
    weights = rng.normal(scale=0.05, size=(n_features, 1)).astype(np.float32)
    bias = np.zeros((1,), dtype=np.float32)

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, n_features])
    probabilities_tensor = helper.make_tensor_value_info(
        "probabilities", TensorProto.FLOAT, [None, 2]
    )
    label_tensor = helper.make_tensor_value_info("label", TensorProto.INT64, [None])

    weight_init = helper.make_tensor("weight", TensorProto.FLOAT, weights.shape, weights.flatten())
    bias_init = helper.make_tensor("bias", TensorProto.FLOAT, bias.shape, bias.flatten())
    one_init = helper.make_tensor("one", TensorProto.FLOAT, [1], [1.0])

    nodes = [
        helper.make_node("MatMul", ["input", "weight"], ["logits_raw"]),
        helper.make_node("Add", ["logits_raw", "bias"], ["logits"]),
        helper.make_node("Sigmoid", ["logits"], ["p1"]),
        helper.make_node("Sub", ["one", "p1"], ["p0"]),
        helper.make_node("Concat", ["p0", "p1"], ["probabilities"], axis=1),
        helper.make_node("ArgMax", ["probabilities"], ["label"], axis=1),
    ]

    graph = helper.make_graph(
        nodes,
        "dummy-drive-failure-model",
        [input_tensor],
        [probabilities_tensor, label_tensor],
        initializer=[weight_init, bias_init, one_init],
    )
    model = helper.make_model(graph, producer_name="drive-failure-service-dev")
    model.opset_import[0].version = 17
    onnx.checker.check_model(model)
    onnx.save(model, output_path)


if __name__ == "__main__":
    build_dummy_model("artifacts/feature_config.json", "artifacts/model.onnx")
