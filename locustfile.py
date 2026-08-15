import random

from locust import HttpUser, between, task

FEATURE_KEYS = [
    "capacity_bytes",
    "smart_5_raw", "smart_5_normalized",
    "smart_9_raw", "smart_9_normalized",
    "smart_187_raw", "smart_187_normalized",
    "smart_188_raw", "smart_188_normalized",
    "smart_194_raw", "smart_194_normalized",
    "smart_197_raw", "smart_197_normalized",
    "smart_198_raw", "smart_198_normalized",
    "smart_241_raw", "smart_241_normalized",
    "smart_242_raw", "smart_242_normalized",
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
