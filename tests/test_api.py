from tests.conftest import VALID_RECORD


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_probabilities(client):
    r = client.post("/predict", json={"records": [VALID_RECORD]})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["predictions"][0]["failure_probability"] <= 1.0
    assert body["predictions"][0]["serial_number"] == VALID_RECORD["serial_number"]


def test_predict_batch(client):
    records = [{**VALID_RECORD, "serial_number": f"Z{i}"} for i in range(10)]
    r = client.post("/predict", json={"records": records})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 10


def test_rejects_missing_smart_field(client):
    bad = {k: v for k, v in VALID_RECORD.items() if k != "smart_5_raw"}
    r = client.post("/predict", json={"records": [bad]})
    assert r.status_code == 422


def test_rejects_unknown_field(client):
    bad = {**VALID_RECORD, "extra_field": 1}
    r = client.post("/predict", json={"records": [bad]})
    assert r.status_code == 422


def test_rejects_empty_batch(client):
    r = client.post("/predict", json={"records": []})
    assert r.status_code == 422


def test_rejects_oversized_batch(client):
    records = [{**VALID_RECORD, "serial_number": f"Z{i}"} for i in range(501)]
    r = client.post("/predict", json={"records": records})
    assert r.status_code == 422


def test_metrics_endpoint(client):
    client.post("/predict", json={"records": [VALID_RECORD]})
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"predictions_total" in r.content
