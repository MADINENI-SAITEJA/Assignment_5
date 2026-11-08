from uuid import UUID

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import eta_app

PREDICT = "/v1/models/eta:predict"
VALID = {
    "distance_km": 15,
    "n_stops": 4,
    "hour": 10,
    "weekday": 2,
    "traffic_index": 0.35,
    "rain": 0,
    "weight_kg": 6,
}


@pytest.fixture
def client():
    with TestClient(eta_app.app) as test_client:
        yield test_client


def test_valid_prediction_echoes_request_id(client):
    response = client.post(
        PREDICT, json={"instances": [VALID]}, headers={"X-Request-ID": "pytest-a1"}
    )
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "pytest-a1"
    assert response.json() == {
        "predictions": [50.1],
        "model_version": "v1",
        "request_id": "pytest-a1",
    }


def test_predictions_preserve_batch_order(client):
    rows = [{**VALID, "distance_km": 45}, VALID, {**VALID, "distance_km": 5}]
    frame = pd.DataFrame(rows, columns=eta_app.feature_columns)
    expected = [round(float(value), 1) for value in eta_app.model.predict(frame)]
    assert len(set(expected)) == len(rows)

    response = client.post(PREDICT, json={"instances": rows})
    assert response.status_code == 200
    assert response.json()["predictions"] == expected


@pytest.mark.parametrize(
    ("row", "field"),
    [
        ({key: value for key, value in VALID.items() if key != "distance_km"}, "distance_km"),
        ({**VALID, "n_stops": "many"}, "n_stops"),
        ({**VALID, "traffic_index": 1.1}, "traffic_index"),
    ],
    ids=["missing-feature", "wrong-type", "out-of-range"],
)
def test_invalid_feature(client, row, field):
    response = client.post(PREDICT, json={"instances": [row]})
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_failed"
    assert any(detail["field"].endswith(field) for detail in body["details"])


def test_infeasible_delivery(client):
    row = {**VALID, "distance_km": 151, "n_stops": 21}
    response = client.post(PREDICT, json={"instances": [row]})
    assert response.status_code == 422
    assert any("infeasible" in item["msg"] for item in response.json()["details"])


@pytest.mark.parametrize("size", [0, 501])
def test_batch_size_rejected(client, size):
    response = client.post(PREDICT, json={"instances": [VALID] * size})
    assert response.status_code == 422
    assert response.json()["error"] == "validation_failed"


def test_maximum_batch_accepted(client):
    response = client.post(PREDICT, json={"instances": [VALID] * 500})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 500


def test_prediction_generates_request_id(client):
    response = client.post(PREDICT, json={"instances": [VALID]})
    assert response.status_code == 200
    request_id = response.headers["X-Request-ID"]
    assert UUID(request_id).version == 4
    assert response.json()["request_id"] == request_id


@pytest.mark.parametrize("supplied_id", [None, "invalid-request"])
def test_validation_response_has_matching_request_id(client, supplied_id):
    headers = {} if supplied_id is None else {"X-Request-ID": supplied_id}
    response = client.post(PREDICT, json={"instances": []}, headers=headers)
    assert response.status_code == 422
    request_id = response.headers["X-Request-ID"]
    assert response.json()["request_id"] == request_id
    if supplied_id is None:
        assert UUID(request_id).version == 4
    else:
        assert request_id == supplied_id


def test_health(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_version": "v1",
        "request_id": response.headers["X-Request-ID"],
    }


def test_prometheus_metrics_exclude_scrapes(client):
    assert client.post(PREDICT, json={"instances": [VALID]}).status_code == 200
    assert client.post(PREDICT, json={"instances": []}).status_code == 422

    response = client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["Content-Type"].startswith("text/plain")
    text = response.text
    assert f'eta_requests_total{{path="{PREDICT}",status="200"}}' in text
    assert f'eta_requests_total{{path="{PREDICT}",status="422"}}' in text
    assert 'eta_request_latency_seconds_bucket{le="+Inf"}' in text
    assert "eta_request_latency_seconds_sum" in text
    assert "eta_request_latency_seconds_count" in text
    assert 'path="/metrics"' not in text
    assert client.get("/metrics").text == text


def test_inference_failure_returns_generic_500(client, monkeypatch):
    def broken_predict(_frame):
        raise RuntimeError("private model error")

    monkeypatch.setattr(eta_app.model, "predict", broken_predict)
    response = client.post(
        PREDICT, json={"instances": [VALID]}, headers={"X-Request-ID": "failed-inference"}
    )
    assert response.status_code == 500
    assert response.headers["X-Request-ID"] == "failed-inference"
    assert response.json() == {
        "error": "internal_error",
        "request_id": "failed-inference",
    }
