from hashlib import sha256
from math import isfinite
from pathlib import Path
from time import sleep

import joblib
import pandas as pd
import ray
from fastapi import FastAPI, Header
from pydantic import BaseModel, Field
from ray import serve

http_app = FastAPI(title="SwiftParcel ETA Canary")
PREDICT_PATH = "/v1/models/eta:predict"


class DeliveryInput(BaseModel):
    distance_km: float
    n_stops: int
    hour: int
    weekday: int
    traffic_index: float
    rain: int
    weight_kg: float


class CanaryRequest(BaseModel):
    customer_id: str = Field(min_length=1)
    instances: list[DeliveryInput] = Field(min_length=1, max_length=500)


def in_canary(customer_id, weight):
    digest = sha256(customer_id.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big")
    return bucket < int(weight * (1 << 64))


@serve.deployment(ray_actor_options={"num_cpus": 0.25})
class EtaModel:
    def __init__(self, bundle_path, extra_work_ms=0.0):
        self.bundle = joblib.load(bundle_path)
        self.extra_work_ms = extra_work_ms

    def predict(self, rows):
        if self.extra_work_ms:
            sleep(self.extra_work_ms / 1000)

        frame = pd.DataFrame(rows, columns=self.bundle["feature_columns"])
        values = self.bundle["model"].predict(frame)
        return {
            "predictions": [round(float(value), 1) for value in values],
            "model_version": self.bundle["version"],
        }


@serve.deployment(ray_actor_options={"num_cpus": 0.25})
@serve.ingress(http_app)
class EtaRouter:
    def __init__(self, prod, candidate, weight):
        self.prod = prod
        self.candidate = candidate
        self.weight = weight

    @http_app.post(PREDICT_PATH)
    async def predict(
        self,
        payload: CanaryRequest,
        x_beta: str | None = Header(default=None, alias="X-Beta"),
    ):
        use_candidate = x_beta == "1" or in_canary(payload.customer_id, self.weight)
        model = self.candidate if use_candidate else self.prod
        rows = [delivery.model_dump() for delivery in payload.instances]
        result = await model.predict.remote(rows)
        return {**result, "arm": "v2" if use_candidate else "v1"}


def build(weight=0.1, extra_work_ms=0.0):
    if not 0 <= weight <= 1:
        raise ValueError("weight must be between 0 and 1.")
    if not isfinite(extra_work_ms) or extra_work_ms < 0:
        raise ValueError("extra_work_ms must be finite and nonnegative.")

    data_dir = Path("data").resolve()
    prod = EtaModel.options(name="prod", num_replicas=2).bind(
        str(data_dir / "eta_v1.joblib")
    )
    candidate = EtaModel.options(
        name="candidate",
        max_ongoing_requests=4,
        autoscaling_config={
            "min_replicas": 1,
            "max_replicas": 3,
            "target_ongoing_requests": 2,
            "upscale_delay_s": 5,
            "downscale_delay_s": 30,
        },
    ).bind(str(data_dir / "eta_v2.joblib"), extra_work_ms)

    return EtaRouter.bind(prod, candidate, weight)


if __name__ == "__main__":
    ray.init(num_cpus=2, include_dashboard=False)
    try:
        serve.start(http_options={"host": "127.0.0.1", "port": 8001})
        serve.run(build(), name="swiftparcel-canary", route_prefix="/")
        print("Canary ready on http://127.0.0.1:8001", flush=True)
        while True:
            sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        serve.shutdown()
        ray.shutdown()
