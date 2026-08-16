from pydantic import BaseModel, ConfigDict, Field

MAX_BATCH_SIZE = 500


class DriveRecord(BaseModel):
    """smart_N_raw_7d_ago is that attribute's raw reading from ~7 days before this
    snapshot -- the caller (a nightly batch job with access to prior SMART history)
    supplies it so the server can derive the same trend features training used,
    without the server needing its own historical store. See predictor.py."""

    model_config = ConfigDict(extra="forbid")

    serial_number: str
    capacity_bytes: float
    smart_5_raw: float
    smart_5_normalized: float
    smart_5_raw_7d_ago: float
    smart_9_raw: float
    smart_9_normalized: float
    smart_9_raw_7d_ago: float
    smart_187_raw: float
    smart_187_normalized: float
    smart_187_raw_7d_ago: float
    smart_188_raw: float
    smart_188_normalized: float
    smart_188_raw_7d_ago: float
    smart_194_raw: float
    smart_194_normalized: float
    smart_194_raw_7d_ago: float
    smart_197_raw: float
    smart_197_normalized: float
    smart_197_raw_7d_ago: float
    smart_198_raw: float
    smart_198_normalized: float
    smart_198_raw_7d_ago: float
    smart_241_raw: float
    smart_241_normalized: float
    smart_241_raw_7d_ago: float
    smart_242_raw: float
    smart_242_normalized: float
    smart_242_raw_7d_ago: float


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    records: list[DriveRecord] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class DrivePrediction(BaseModel):
    serial_number: str
    failure_probability: float


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    predictions: list[DrivePrediction]


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_version: str
