import os

import pytest
from fastapi.testclient import TestClient

from src.pipeline.dummy_model import build_dummy_model

VALID_RECORD = {
    "serial_number": "Z1TEST0001",
    "capacity_bytes": 4_000_787_030_016.0,
    "smart_5_raw": 0.0,
    "smart_5_normalized": 100.0,
    "smart_5_raw_7d_ago": 0.0,
    "smart_9_raw": 12000.0,
    "smart_9_normalized": 90.0,
    "smart_9_raw_7d_ago": 11832.0,
    "smart_187_raw": 0.0,
    "smart_187_normalized": 100.0,
    "smart_187_raw_7d_ago": 0.0,
    "smart_188_raw": 0.0,
    "smart_188_normalized": 100.0,
    "smart_188_raw_7d_ago": 0.0,
    "smart_194_raw": 30.0,
    "smart_194_normalized": 65.0,
    "smart_194_raw_7d_ago": 29.0,
    "smart_197_raw": 0.0,
    "smart_197_normalized": 100.0,
    "smart_197_raw_7d_ago": 0.0,
    "smart_198_raw": 0.0,
    "smart_198_normalized": 100.0,
    "smart_198_raw_7d_ago": 0.0,
    "smart_241_raw": 500000.0,
    "smart_241_normalized": 100.0,
    "smart_241_raw_7d_ago": 498000.0,
    "smart_242_raw": 500000.0,
    "smart_242_normalized": 100.0,
    "smart_242_raw_7d_ago": 498000.0,
}


@pytest.fixture(scope="session")
def feature_config_path():
    return "artifacts/feature_config.json"


@pytest.fixture(scope="session")
def dummy_model_path(tmp_path_factory, feature_config_path):
    out = tmp_path_factory.mktemp("model") / "model.onnx"
    build_dummy_model(feature_config_path, str(out))
    return str(out)


@pytest.fixture()
def client(dummy_model_path, feature_config_path):
    os.environ["MODEL_PATH"] = dummy_model_path
    os.environ["FEATURE_CONFIG_PATH"] = feature_config_path

    from src.api.main import app

    with TestClient(app) as test_client:
        yield test_client

    os.environ.pop("MODEL_PATH", None)
    os.environ.pop("FEATURE_CONFIG_PATH", None)
