import pytest

from eta_mlflow import decide

PROD = {"mae": 10.0, "mae_rush": 20.0, "mae_rain": 8.0, "p95_ms": 4.0}
GOOD = {"mae": 9.0, "mae_rush": 19.0, "mae_rain": 7.0, "p95_ms": 5.0}


@pytest.mark.parametrize(
    ("metric", "limit"),
    [("mae", 9.7), ("p95_ms", 10.0), ("mae_rush", 21.0), ("mae_rain", 8.4)],
)
@pytest.mark.parametrize(
    ("offset", "passes"),
    [(-0.001, True), (0.0, True), (0.001, False)],
    ids=["just-inside", "at-limit", "just-outside"],
)
def test_gate_boundaries(metric, limit, offset, passes):
    candidate = {**GOOD, metric: limit + offset}
    result = decide(PROD, candidate)
    assert result["criteria"][metric] is passes
    assert result["promote"] is passes


def test_stricter_overall_gain():
    assert decide(PROD, GOOD)["promote"] is True
    assert decide(PROD, GOOD, min_rel_gain=0.60)["promote"] is False


def test_zero_baseline_mae():
    prod = {**PROD, "mae": 0.0}
    candidate = {**GOOD, "mae": 0.0}
    assert decide(prod, candidate)["promote"] is False
    assert decide(prod, candidate, min_rel_gain=0.0)["promote"] is True
