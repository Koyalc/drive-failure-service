from pydantic import BaseModel, ConfigDict, Field

MAX_BATCH_SIZE = 500


class DriveRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    serial_number: str
    capacity_bytes: float
    smart_5_raw: float
    smart_5_normalized: float
    smart_9_raw: float
    smart_9_normalized: float
    smart_187_raw: float
    smart_187_normalized: float
    smart_188_raw: float
    smart_188_normalized: float
    smart_194_raw: float
    smart_194_normalized: float
    smart_197_raw: float
    smart_197_normalized: float
    smart_198_raw: float
    smart_198_normalized: float
    smart_241_raw: float
    smart_241_normalized: float
    smart_242_raw: float
    smart_242_normalized: float


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
