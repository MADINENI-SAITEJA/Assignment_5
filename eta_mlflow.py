import os
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

MODEL_NAME = "eta-model"


def predict_bundle(bundle, model_input):
    features = model_input.loc[:, bundle["feature_columns"]]
    eta = bundle["model"].predict(features).round(1)
    return pd.DataFrame(
        {"eta_minutes": eta, "model_version": bundle["version"]},
        index=model_input.index,
    )


class EtaPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.bundle = joblib.load(context.artifacts["bundle"])

    def predict(self, context, model_input, params=None):
        return predict_bundle(self.bundle, model_input)


def register():
    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    mlflow.set_experiment("swiftparcel-eta")

    data_dir = Path("data").resolve()
    fresh = pd.read_csv(data_dir / "eta_fresh.csv", nrows=5)
    packages = ("mlflow", "cloudpickle", "joblib", "numpy", "pandas", "scikit-learn", "scipy")
    requirements = [f"{package}=={version(package)}" for package in packages]

    versions = {}
    with mlflow.start_run(run_name="register-eta-models"):
        for alias, bundle_version in (("prod", "v1"), ("candidate", "v2")):
            bundle_path = data_dir / f"eta_{bundle_version}.joblib"
            bundle = joblib.load(bundle_path)
            example = fresh.loc[:, bundle["feature_columns"]]

            logged = mlflow.pyfunc.log_model(
                name=f"eta-{bundle_version}",
                python_model=EtaPyfunc(),
                artifacts={"bundle": str(bundle_path)},
                code_paths=[str(Path(__file__).resolve())],
                signature=infer_signature(example, predict_bundle(bundle, example)),
                input_example=example,
                pip_requirements=requirements,
            )
            registered = mlflow.register_model(logged.model_uri, MODEL_NAME)
            versions[alias] = registered.version

    client = MlflowClient()
    for alias, registry_version in versions.items():
        client.set_registered_model_alias(MODEL_NAME, alias, registry_version)
        print(f"{MODEL_NAME} | registry version {registry_version} | alias {alias}")

    return versions


def decide(prod, cand, min_rel_gain=0.03, max_p95_ms=10.0, max_slice_regression=0.05):
    prod_mae = prod["mae"]
    cand_mae = cand["mae"]
    if prod_mae == 0:
        mae_ok = cand_mae == 0 and min_rel_gain <= 0
    else:
        mae_ok = cand_mae <= prod_mae * (1 - min_rel_gain)

    criteria = {
        "mae": mae_ok,
        "p95_ms": cand["p95_ms"] <= max_p95_ms,
        "mae_rush": cand["mae_rush"] <= prod["mae_rush"] * (1 + max_slice_regression),
        "mae_rain": cand["mae_rain"] <= prod["mae_rain"] * (1 + max_slice_regression),
    }
    return {"criteria": criteria, "promote": all(criteria.values())}


def promotion_gate(**thresholds):
    from eta_data import FEATURES, make_data

    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)

    fresh = make_data(2000, seed=99)
    features = fresh.loc[:, FEATURES]
    rush = fresh["hour"].between(7, 9) | fresh["hour"].between(16, 19)
    rain = fresh["rain"].eq(1)

    metrics = {}
    for alias in ("prod", "candidate"):
        model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{alias}")
        predictions = model.predict(features)["eta_minutes"]
        errors = (predictions - fresh["eta_minutes"]).abs()

        latencies_ms = []
        for index in range(200):
            row = features.iloc[index:index + 1]
            start = perf_counter()
            model.predict(row)
            latencies_ms.append((perf_counter() - start) * 1000)

        metrics[alias] = {
            "mae": float(errors.mean()),
            "mae_rush": float(errors[rush].mean()),
            "mae_rain": float(errors[rain].mean()),
            "p95_ms": float(np.percentile(latencies_ms, 95)),
        }

    return {**metrics, **decide(metrics["prod"], metrics["candidate"], **thresholds)}

if __name__ == "__main__":
    register()
