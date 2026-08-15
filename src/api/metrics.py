from prometheus_client import Counter, Histogram

PREDICTIONS_TOTAL = Counter(
    "predictions_total", "Total number of drive predictions served", ["model_version"]
)

PREDICTION_LATENCY_SECONDS = Histogram(
    "prediction_latency_seconds",
    "Inference latency per /predict request",
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

PREDICTION_PROBABILITY = Histogram(
    "prediction_probability",
    "Distribution of predicted failure probabilities",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)
