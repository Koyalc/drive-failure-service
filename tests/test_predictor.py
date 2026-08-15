from src.api.predictor import Predictor
from src.api.schemas import DriveRecord
from tests.conftest import VALID_RECORD


def test_predictor_returns_one_probability_per_record(dummy_model_path, feature_config_path):
    predictor = Predictor(dummy_model_path, feature_config_path)
    records = [DriveRecord(**VALID_RECORD), DriveRecord(**{**VALID_RECORD, "serial_number": "Z2"})]

    probabilities = predictor.predict(records)

    assert len(probabilities) == 2
    assert all(0.0 <= p <= 1.0 for p in probabilities)
