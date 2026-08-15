"""Builds a placeholder ONNX model (random logistic weights) matching feature_config.json.

Lets the API, tests, and Docker build be exercised end-to-end before Phase 1
training produces a real model. Replace artifacts/model.onnx via `make train`
once you have data.
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
    output_tensor = helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, 1])

    weight_init = helper.make_tensor("weight", TensorProto.FLOAT, weights.shape, weights.flatten())
    bias_init = helper.make_tensor("bias", TensorProto.FLOAT, bias.shape, bias.flatten())

    matmul_node = helper.make_node("MatMul", ["input", "weight"], ["logits_raw"])
    add_node = helper.make_node("Add", ["logits_raw", "bias"], ["logits"])
    sigmoid_node = helper.make_node("Sigmoid", ["logits"], ["probabilities"])

    graph = helper.make_graph(
        [matmul_node, add_node, sigmoid_node],
        "dummy-drive-failure-model",
        [input_tensor],
        [output_tensor],
        initializer=[weight_init, bias_init],
    )
    model = helper.make_model(graph, producer_name="drive-failure-service-dev")
    model.opset_import[0].version = 17
    onnx.checker.check_model(model)
    onnx.save(model, output_path)


if __name__ == "__main__":
    build_dummy_model("artifacts/feature_config.json", "artifacts/model.onnx")
