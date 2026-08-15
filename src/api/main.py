import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response

from src.api.metrics import (
    PREDICTION_LATENCY_SECONDS,
    PREDICTION_PROBABILITY,
    PREDICTIONS_TOTAL,
)
from src.api.predictor import Predictor
from src.api.schemas import (
    DrivePrediction,
    HealthResponse,
    PredictRequest,
    PredictResponse,
)

MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/model.onnx")
FEATURE_CONFIG_PATH = os.environ.get("FEATURE_CONFIG_PATH", "artifacts/feature_config.json")


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = Predictor(MODEL_PATH, FEATURE_CONFIG_PATH)
    yield
    app.state.predictor = None


app = FastAPI(lifespan=lifespan, title="Drive Failure Prediction")


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", model_version=app.state.predictor.model_version)


@app.get("/metrics")
def metrics():
    from prometheus_client import generate_latest

    return Response(generate_latest(), media_type="text/plain; version=0.0.4")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    predictor: Predictor = app.state.predictor

    start = time.perf_counter()
    probabilities = predictor.predict(req.records)
    PREDICTION_LATENCY_SECONDS.observe(time.perf_counter() - start)

    PREDICTIONS_TOTAL.labels(model_version=predictor.model_version).inc(len(req.records))
    for p in probabilities:
        PREDICTION_PROBABILITY.observe(p)

    predictions = [
        DrivePrediction(serial_number=r.serial_number, failure_probability=p)
        for r, p in zip(req.records, probabilities)
    ]
    return PredictResponse(model_version=predictor.model_version, predictions=predictions)
