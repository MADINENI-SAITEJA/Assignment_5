from __future__ import annotations

import logging
import os
from time import perf_counter
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field, model_validator
from starlette.exceptions import HTTPException

logger = logging.getLogger(__name__)

bundle = joblib.load(os.environ.get("ETA_MODEL", "data/eta_v1.joblib"))
model = bundle["model"]
feature_columns = bundle["feature_columns"]
model_version = bundle["version"]

registry = CollectorRegistry()
requests_total = Counter(
    "eta_requests_total",
    "Number of ETA HTTP requests",
    ["path", "status"],
    registry=registry,
)
request_latency = Histogram(
    "eta_request_latency_seconds",
    "Time to handle an ETA HTTP request",
    registry=registry,
)

app = FastAPI(title="SwiftParcel ETA API")


class Delivery(BaseModel):
    distance_km: float = Field(ge=0.1, le=500)
    n_stops: int = Field(ge=1, le=60)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    traffic_index: float = Field(ge=0, le=1)
    rain: int = Field(ge=0, le=1)
    weight_kg: float = Field(gt=0, le=1000)

    @model_validator(mode="after")
    def check_feasibility(self) -> Delivery:
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError("infeasible delivery: distance exceeds 150 km and stops exceed 20")
        return self


class PredictionRequest(BaseModel):
    instances: list[Delivery] = Field(min_length=1, max_length=500)


@app.middleware("http")
async def track_request(request: Request, call_next):
    start = perf_counter()
    request_id = request.headers.get("X-Request-ID")
    if request_id is None:
        request_id = str(uuid4())
    request.state.request_id = request_id

    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Request %s failed", request_id)
        response = JSONResponse(
            {"error": "internal_error", "request_id": request_id},
            status_code=500,
        )

    response.headers["X-Request-ID"] = request_id
    path = request.url.path
    if path != "/metrics":
        requests_total.labels(path=path, status=str(response.status_code)).inc()
        request_latency.observe(perf_counter() - start)
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    details = []
    for error in exc.errors():
        location = error["loc"]
        if location and location[0] == "body":
            location = location[1:]
        field = ".".join(str(part) for part in location) or "body"
        details.append({"field": field, "msg": error["msg"]})

    return JSONResponse(
        {
            "error": "validation_failed",
            "details": details,
            "request_id": request.state.request_id,
        },
        status_code=422,
    )


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    return JSONResponse(
        {"detail": exc.detail, "request_id": request.state.request_id},
        status_code=exc.status_code,
        headers=exc.headers,
    )


@app.post("/v1/models/eta:predict")
def predict(payload: PredictionRequest, request: Request):
    rows = [delivery.model_dump() for delivery in payload.instances]
    frame = pd.DataFrame(rows, columns=feature_columns)
    predictions = [round(float(value), 1) for value in model.predict(frame)]
    return {
        "predictions": predictions,
        "model_version": model_version,
        "request_id": request.state.request_id,
    }


@app.get("/healthz")
def healthz(request: Request):
    return {
        "status": "ok",
        "model_version": model_version,
        "request_id": request.state.request_id,
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(registry), headers={"Content-Type": CONTENT_TYPE_LATEST})
