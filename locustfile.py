import random

from locust import HttpUser, between, task

SMART_ATTRIBUTES = [5, 9, 187, 188, 194, 197, 198, 241, 242]

FEATURE_KEYS = ["capacity_bytes"] + [
    f"smart_{n}_{suffix}"
    for n in SMART_ATTRIBUTES
    for suffix in ("raw", "normalized", "raw_7d_ago")
]


def random_record():
    record = {"serial_number": f"Z{random.randint(0, 1_000_000)}"}
    record.update({k: random.uniform(0, 100_000) for k in FEATURE_KEYS})
    return record


class PredictUser(HttpUser):
    wait_time = between(0.1, 0.5)

    @task
    def predict_batch(self):
        batch = [random_record() for _ in range(random.randint(1, 50))]
        self.client.post("/predict", json={"records": batch})
