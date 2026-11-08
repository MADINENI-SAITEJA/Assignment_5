import argparse
import math
import sys
from statistics import quantiles
from time import perf_counter

import requests

PREDICT_PATH = "/v1/models/eta:predict"
BASE = {
    "distance_km": 15,
    "n_stops": 4,
    "hour": 10,
    "weekday": 2,
    "traffic_index": 0.35,
    "rain": 0,
    "weight_kg": 6,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_object(response):
    body = response.json()
    require(isinstance(body, dict), "Expected a JSON object in the response")
    return body


def predict(session, url, rows, expected_version):
    start = perf_counter()
    response = session.post(url, json={"instances": rows}, timeout=5)
    elapsed_ms = (perf_counter() - start) * 1000
    require(response.status_code == 200, f"Prediction returned HTTP {response.status_code}")

    body = json_object(response)
    version = body.get("model_version")
    require(version == expected_version, f"Expected prediction model {expected_version}, got {version}")
    values = body.get("predictions")
    require(isinstance(values, list) and len(values) == len(rows), "Expected one prediction per row")
    require(
        all(type(value) in (int, float) and math.isfinite(value) for value in values),
        "Expected finite numeric predictions",
    )
    return values, elapsed_ms


def run_smoke(session, base_url, expected_version, max_p95_ms):
    health = session.get(f"{base_url}/healthz", timeout=5)
    require(health.status_code == 200, f"Health check returned HTTP {health.status_code}")
    body = json_object(health)
    require(body.get("status") == "ok", "Health check status must be ok")
    version = body.get("model_version")
    require(version == expected_version, f"Expected health model {expected_version}, got {version}")
    print(f"PASS 1: Health is ok and reports {version}")

    url = f"{base_url}{PREDICT_PATH}"
    rows = [BASE, {**BASE, "distance_km": 45}, {**BASE, "traffic_index": 0.90}]
    (baseline, longer, heavier), _ = predict(session, url, rows, expected_version)
    print(f"PASS 2: Valid batch returned three numeric predictions from {expected_version}")

    require(longer >= baseline, f"Longer distance shortened ETA: {baseline} -> {longer}")
    print(f"PASS 3: Longer distance raised or preserved ETA ({baseline} -> {longer} min)")

    require(heavier >= baseline, f"Heavier traffic shortened ETA: {baseline} -> {heavier}")
    print(f"PASS 4: Heavier traffic raised or preserved ETA ({baseline} -> {heavier} min)")

    invalid = session.post(
        url, json={"instances": [{**BASE, "distance_km": 0}]}, timeout=5
    )
    require(invalid.status_code == 422, f"Invalid input returned HTTP {invalid.status_code}, expected 422")
    require(json_object(invalid).get("error") == "validation_failed", "Expected validation_failed")
    print("PASS 5: Invalid distance returned HTTP 422")

    timings = [predict(session, url, [BASE], expected_version)[1] for _ in range(30)]
    p95 = quantiles(timings, n=100, method="inclusive")[94]
    require(p95 <= max_p95_ms, f"HTTP p95 {p95:.3f} ms exceeds {max_p95_ms:g} ms")
    print(f"PASS 6: HTTP p95 {p95:.3f} ms <= {max_p95_ms:g} ms over 30 requests")


def main():
    parser = argparse.ArgumentParser(description="Smoke-test the SwiftParcel FastAPI service.")
    parser.add_argument("--url", required=True, help="Server base URL or full prediction endpoint")
    parser.add_argument("--expect-version", required=True)
    parser.add_argument("--max-p95-ms", type=float, default=80.0)
    args = parser.parse_args()

    try:
        require(math.isfinite(args.max_p95_ms) and args.max_p95_ms > 0, "Latency limit must be positive")
        base_url = args.url.rstrip("/").removesuffix(PREDICT_PATH)
        with requests.Session() as session:
            run_smoke(session, base_url, args.expect_version, args.max_p95_ms)
    except (requests.RequestException, ValueError) as error:
        print(f"SMOKE FAIL: {error}", file=sys.stderr)
        return 1

    print("SMOKE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
