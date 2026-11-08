# DA3408 Module 4 Assignment 5 — SwiftParcel ETA

This walkthrough reproduces the ten-terminal run for Parts A–D using the Python files and `requirements.txt` in the repository root.

## File inventory

### Code and saved results

| File | Purpose |
| --- | --- |
| `eta_app.py` | FastAPI prediction, health, and metrics endpoints for Part A. |
| `eta_data.py` | Generates the datasets and trains the v1 and v2 model bundles. |
| `load_test.py` | Benchmarks the API at different concurrency levels and batch sizes. |
| `requirements.txt` | Python dependencies for running the assignment. |
| `test_eta_app.py` | Tests the Part A API, validation, and metrics. |
| `eta_mlflow.py` | Registers both models and runs the promotion gate for Part B. |
| `test_gate.py` | Tests the promotion gate's decision rules. |
| `eta_ray.py` | Serves v1 and v2 with sticky canary routing for Part C. |
| `verify_ray.py` | Checks the traffic split, sticky routing, redeploy behavior, and header override. |
| `smoke_eta.py` | Runs the release smoke checks for Part D. |
| `results/loadtest_results.json` | Saved one-worker load-test baseline. |
| `results/gate_output.txt` | Saved results for the default and stricter promotion gates. |
| `results/shadow_output.txt` | Saved comparison of the two live MLflow servers over 200 request pairs. |
| `results/verify_output.txt` | Saved output from the five Ray Serve verification checks. |

### Submission documents

| File | Purpose |
| --- | --- |
| `README.md` | Setup instructions and the ten-terminal walkthrough. |
| `Report.pdf` | Assignment report with benchmark results, analysis, and written answers. |
| `AI_disclosure.md` | Disclosure of AI assistance used for the assignment. |

## Benchmark environment

I ran the assignment on an Ubuntu VM with 6 vCPUs, 11 GiB of RAM, and Python 3.14.4. All runs used the CPU. Latency and throughput can vary with hardware and system load.

## Before you start

Use Bash on Ubuntu with Git, curl, and Python 3.14.4 with `venv` available. The local ports `8000`, `8001`, `5101`, and `5102` must be available. If you already cloned the repo, enter its root and start at the Python commands below.

~~~bash
git clone https://github.com/MADINENI-SAITEJA/Assignment_5.git
cd Assignment_5

python3.14 --version
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt 'mlflow==3.18.0' 'ray[serve]==2.60.0'
mkdir -p rerun
~~~

The install command uses the MLflow and Ray versions from the recorded run. For each additional terminal, change into the same `Assignment_5` directory and run `source .venv/bin/activate` before copying its commands. Leave foreground servers running until the instructions say to stop them. Start Terminal 1 after setup. It generates the data and both model bundles needed by the remaining steps.

The commands save fresh measurements in `rerun/` so you can compare them with the submitted evidence in `results/`.

## Part A — FastAPI and load testing

### Terminal 1 — Generate data and start the API

~~~bash
python -m pip check && python eta_data.py

python - <<'PY'
import pandas as pd
from eta_data import FEATURES

fresh = pd.read_csv("data/eta_fresh.csv")
sample = fresh.loc[:, FEATURES].head(200)
sample.to_csv("data/eta_sample.csv", index=False)
print(f"Saved {len(sample)} rows with {len(sample.columns)} features")
PY

ls -lh data
wc -l data/eta_train.csv data/eta_fresh.csv data/eta_sample.csv
~~~

The recorded fresh-data MAE was 11.95 minutes for v1 and 6.10 minutes for v2. The load-test sample contains the first 200 fresh rows and their seven input features.

~~~bash
ETA_MODEL=data/eta_v1.joblib python -m uvicorn eta_app:app \
  --host 127.0.0.1 --port 8000 --workers 1
~~~

Leave this server running for the HTTP checks in Terminal 2.

### Terminal 2 — Check HTTP behavior and run API tests

~~~bash
curl -sS -i http://127.0.0.1:8000/healthz
echo

curl -sS -i http://127.0.0.1:8000/v1/models/eta:predict \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: a1-demo' \
  -d '{"instances":[{"distance_km":15,"n_stops":4,"hour":10,"weekday":2,"traffic_index":0.35,"rain":0,"weight_kg":6}]}'
echo

curl -sS -i http://127.0.0.1:8000/v1/models/eta:predict \
  -H 'Content-Type: application/json' \
  -d '{"instances":[{"distance_km":151,"n_stops":21,"hour":10,"weekday":2,"traffic_index":0.35,"rain":0,"weight_kg":6}]}'
echo

curl -fsS http://127.0.0.1:8000/metrics |
  grep -E '^eta_requests_total|^eta_request_latency_seconds_(count|sum)|^eta_request_latency_seconds_bucket.*\+Inf'
~~~

The recorded checks returned HTTP 200 for health and the valid prediction. The valid request echoed `a1-demo` and returned 50.1 minutes. The infeasible delivery returned HTTP 422. The metrics included request counters and a latency histogram.

~~~bash
ETA_MODEL=data/eta_v1.joblib python -m pytest -v test_eta_app.py
~~~

The recorded test run passed all 15 tests. The submitted app enables Swagger UI at `/docs`. Stop Terminal 1 with `Ctrl+C` before using port 8000 for the benchmarks.

### Terminal 3 — Switch between benchmark server configurations

Start v1 with one worker and run the one-worker tests in Terminal 4.

~~~bash
ETA_MODEL=data/eta_v1.joblib python -m uvicorn eta_app:app \
  --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
~~~

When the one-worker tests finish, stop this server with `Ctrl+C`. Start two workers on the same port and run the two-worker tests in Terminal 4.

~~~bash
ETA_MODEL=data/eta_v1.joblib python -m uvicorn eta_app:app \
  --host 127.0.0.1 --port 8000 --workers 2 --no-access-log
~~~

After the two-worker tests finish, stop the server with `Ctrl+C` and start v1 with one worker and access logging. Leave it running for Terminal 10. Swagger UI is available at `http://127.0.0.1:8000/docs`.

~~~bash
ETA_MODEL=data/eta_v1.joblib python -m uvicorn eta_app:app \
  --host 127.0.0.1 --port 8000 --workers 1
~~~

### Terminal 4 — Warm up and measure

With **one worker** running in Terminal 3, warm up the API, sweep concurrency with one row per request, and run the 100-row batch test with one client.

~~~bash
predict_url=http://127.0.0.1:8000/v1/models/eta:predict

python load_test.py --url "$predict_url" \
  --data data/eta_sample.csv --concurrency 4 --duration 2 --batch 1

python load_test.py --url "$predict_url" \
  --data data/eta_sample.csv --concurrency 1 2 4 8 16 \
  --duration 4 --batch 1 --out rerun/loadtest_results.json

python load_test.py --url "$predict_url" \
  --data data/eta_sample.csv --concurrency 1 \
  --duration 4 --batch 100 --out rerun/loadtest_batch100.json
~~~

Once Terminal 3 is running **two workers**, repeat the warm-up and batch-1 sweep.

~~~bash
predict_url=http://127.0.0.1:8000/v1/models/eta:predict

python load_test.py --url "$predict_url" \
  --data data/eta_sample.csv --concurrency 4 --duration 2 --batch 1

python load_test.py --url "$predict_url" \
  --data data/eta_sample.csv --concurrency 1 2 4 8 16 \
  --duration 4 --batch 1 --out rerun/loadtest_workers2.json
~~~

The sweeps used four seconds per concurrency level and reported zero errors. The committed `results/loadtest_results.json` contains the one-worker baseline.

## Part B — MLflow registration, gate, and shadow run

### Terminal 5 — Register and evaluate the models

Use the local SQLite tracking store with MLflow telemetry disabled. Register once when setting up a fresh store.

~~~bash
export MLFLOW_TRACKING_URI="sqlite:///$PWD/eta_registry.db"
export MLFLOW_DISABLE_TELEMETRY=true
python -c 'import mlflow; print("MLflow", mlflow.__version__)'
python eta_mlflow.py
~~~

Verify that each alias resolves to the expected model and returns one prediction per input.

~~~bash
python - <<'PY'
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

client = MlflowClient()
sample = pd.read_csv("data/eta_sample.csv", nrows=3)

for alias, expected_version in [("prod", "v1"), ("candidate", "v2")]:
    registered = client.get_model_version_by_alias("eta-model", alias)
    model = mlflow.pyfunc.load_model(f"models:/eta-model@{alias}")
    predictions = model.predict(sample)

    assert list(predictions.columns) == ["eta_minutes", "model_version"]
    assert len(predictions) == len(sample)
    assert predictions["model_version"].eq(expected_version).all()

    print(f"\n{alias} -> registry version {registered.version} -> bundle {expected_version}")
    print(predictions.to_string(index=False))
PY
~~~

The original registration assigned registry version 1 to `prod` (v1) and registry version 2 to `candidate` (v2).

~~~bash
python -m pytest -v test_gate.py

set -o pipefail
python - <<'PY' | tee rerun/gate_output.txt
import json

from eta_mlflow import promotion_gate

print("Default thresholds")
print(json.dumps(promotion_gate(), indent=2))

print("\nStricter threshold: minimum overall MAE gain = 60%")
print(json.dumps(promotion_gate(min_rel_gain=0.60), indent=2))
PY
~~~

All 14 gate tests passed. The default policy approved v2. Requiring a 60% improvement in overall MAE rejected it. The original output is `results/gate_output.txt`.

### Terminal 6 — Serve the production alias

~~~bash
export MLFLOW_TRACKING_URI="sqlite:///$PWD/eta_registry.db"
export MLFLOW_DISABLE_TELEMETRY=true

mlflow models serve -m 'models:/eta-model@prod' \
  --host 127.0.0.1 --port 5101 --workers 1 --env-manager local
~~~

### Terminal 7 — Serve the candidate alias

~~~bash
export MLFLOW_TRACKING_URI="sqlite:///$PWD/eta_registry.db"
export MLFLOW_DISABLE_TELEMETRY=true

mlflow models serve -m 'models:/eta-model@candidate' \
  --host 127.0.0.1 --port 5102 --workers 1 --env-manager local
~~~

### Back in Terminal 5 — Compare the two live servers

Keep both MLflow servers running. Check that both respond, then send the same 200 fresh rows to each server, one row per request. Alternate which server goes first in each pair.

~~~bash
curl -fsS -o /dev/null -w 'prod health HTTP %{http_code}\n' \
  http://127.0.0.1:5101/ping

curl -fsS -o /dev/null -w 'candidate health HTTP %{http_code}\n' \
  http://127.0.0.1:5102/ping
~~~

~~~bash
set -o pipefail
python - <<'PY' | tee rerun/shadow_output.txt
from time import perf_counter

import numpy as np
import pandas as pd
import requests

from eta_data import FEATURES

rows = pd.read_csv("data/eta_fresh.csv", usecols=FEATURES, nrows=200)
if len(rows) != 200:
    raise ValueError("Expected 200 fresh rows.")
records = rows.loc[:, FEATURES].to_dict(orient="records")

servers = {"prod": (5101, "v1"), "candidate": (5102, "v2")}
predictions = {alias: [] for alias in servers}
latencies = {alias: [] for alias in servers}

with requests.Session() as session:
    for index, row in enumerate(records):
        order = ("prod", "candidate") if index % 2 == 0 else ("candidate", "prod")
        for alias in order:
            port, expected_version = servers[alias]
            start = perf_counter()
            response = session.post(
                f"http://127.0.0.1:{port}/invocations",
                json={"dataframe_records": [row]},
                timeout=10,
            )
            elapsed_ms = (perf_counter() - start) * 1000

            if not response.ok:
                raise RuntimeError(f"{alias} HTTP {response.status_code}: {response.text}")

            (prediction,) = response.json()["predictions"]
            if prediction["model_version"] != expected_version:
                raise RuntimeError(f"{alias} returned an unexpected version: {prediction}")

            predictions[alias].append(float(prediction["eta_minutes"]))
            latencies[alias].append(elapsed_ms)

differences = np.round(
    np.abs(np.asarray(predictions["prod"]) - np.asarray(predictions["candidate"])), 1
)
over_ten = int(np.count_nonzero(differences > 10.0))

print(f"Paired requests: {len(differences)}")
print(f"Mean absolute prediction difference: {differences.mean():.4f} min")
print(f"Difference > 10 min: {over_ten}/{len(differences)} ({100 * over_ten / len(differences):.1f}%)")
for alias, (port, model_version) in servers.items():
    p95 = np.percentile(latencies[alias], 95)
    print(f"{alias} ({model_version}, port {port}) HTTP p95: {p95:.3f} ms")
PY
~~~

The original output in `results/shadow_output.txt` recorded 200 pairs, a mean absolute prediction difference of 9.4845 minutes, and 70 pairs (35.0%) differing by more than 10 minutes. The observed HTTP p95 was 2.896 ms for prod and 3.009 ms for candidate. Both MLflow servers can be stopped with `Ctrl+C` after the comparison.

## Part C — Ray Serve canary

### Terminal 8 — Start the manual canary

~~~bash
export RAY_USAGE_STATS_ENABLED=0
python -c 'import ray; print("Ray", ray.__version__)'
python -m py_compile eta_ray.py && python eta_ray.py
~~~

Wait for `Canary ready on http://127.0.0.1:8001` before running the manual requests in Terminal 9.

### Terminal 9 — Check routing and run the verifier

~~~bash
ray_url=http://127.0.0.1:8001/v1/models/eta:predict
payload='{"customer_id":"demo-customer","instances":[{"distance_km":15,"n_stops":4,"hour":10,"weekday":2,"traffic_index":0.35,"rain":0,"weight_kg":6}]}'

curl -fsS "$ray_url" \
  -H 'Content-Type: application/json' \
  -d "$payload"
echo

curl -fsS "$ray_url" \
  -H 'Content-Type: application/json' \
  -H 'X-Beta: 1' \
  -d "$payload"
echo
~~~

The first response returned v1 and 50.1 minutes. The header override returned v2 and 51.2 minutes for the same request.

Stop Terminal 8 with `Ctrl+C` to free port 8001 before running the verifier. It starts a separate local Ray instance with two CPUs.

~~~bash
set -o pipefail
export RAY_USAGE_STATS_ENABLED=0
python -u verify_ray.py | tee rerun/verify_output.txt
~~~

The recorded verifier printed all five `PASS` lines. At a 20% weight, 192 of 1,000 customers went to v2. The repeat and redeploy checks each preserved all 200 assignments. Raising the weight to 50% kept all 192 original v2 customers on v2 and expanded that arm to 488 customers. All 50 override checks passed. The original output is `results/verify_output.txt`. The terminal also logged a Ray controller shutdown error after the five checks.

If Ray on Linux fails to start with a `PermissionError` for `gcs_server`, restore the execute bit in the active virtual environment and retry the Ray command.

~~~bash
ray_dir=$(python -c 'from pathlib import Path; import ray; print(Path(ray.__file__).parent)')
chmod u+x "$ray_dir/core/src/ray/gcs/gcs_server"
~~~

## Part D — Release smoke test

### Terminal 10 — Check success and version mismatch

Keep Terminal 3's final one-worker v1 server running on port 8000. Run both cases in the terminal and read the exit code immediately after each.

~~~bash
curl -fsS -w '\n' http://127.0.0.1:8000/healthz

python smoke_eta.py --url http://127.0.0.1:8000 \
  --expect-version v1 --max-p95-ms 80
echo "Exit code: $?"

python smoke_eta.py --url http://127.0.0.1:8000 \
  --expect-version v2 --max-p95-ms 80
echo "Exit code: $?"
~~~

The v1 run passed all six checks, measured a 2.076 ms HTTP p95 across 30 requests, and exited with code 0. The deliberate v2 expectation failed the health version check and exited with code 1.

## Output files

| Run | File written by the walkthrough |
| --- | --- |
| Part A one-worker baseline | `rerun/loadtest_results.json` |
| Part A batch-100 comparison | `rerun/loadtest_batch100.json` |
| Part A two-worker sweep | `rerun/loadtest_workers2.json` |
| Part B promotion gate | `rerun/gate_output.txt` |
| Part B shadow comparison | `rerun/shadow_output.txt` |
| Part C canary checks | `rerun/verify_output.txt` |

Part D prints both smoke-test results and exit codes in Terminal 10. The report contains the benchmark analysis, canary design answers, and proposed release abort criteria.
