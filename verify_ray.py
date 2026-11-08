import ray
import requests
from ray import serve

from eta_ray import build

APP_NAME = "swiftparcel-canary-check"
HOST = "127.0.0.1"
PORT = 8001
URL = f"http://{HOST}:{PORT}/v1/models/eta:predict"
SAMPLE = {
    "distance_km": 15,
    "n_stops": 4,
    "hour": 10,
    "weekday": 2,
    "traffic_index": 0.35,
    "rain": 0,
    "weight_kg": 6,
}


def deploy(weight):
    serve.run(build(weight=weight), name=APP_NAME, route_prefix="/")


def request_arm(session, customer_id, headers=None):
    response = session.post(
        URL,
        json={"customer_id": customer_id, "instances": [SAMPLE]},
        headers=headers,
        timeout=15,
    )
    response.raise_for_status()
    result = response.json()
    arm = result["arm"]
    assert arm in ("v1", "v2"), f"Unexpected arm for {customer_id}: {arm}"
    assert result["model_version"] == arm, f"Model label mismatch for {customer_id}"
    return arm


def assignments(session, customer_ids):
    return {customer_id: request_arm(session, customer_id) for customer_id in customer_ids}


def main():
    ray.init(address="local", num_cpus=2, include_dashboard=False, log_to_driver=False)
    try:
        cpus = ray.cluster_resources().get("CPU", 0)
        assert cpus == 2, f"Expected a 2-CPU Ray cluster, got {cpus}"
        print(f"Ray CPU resources: {cpus:g}")
        serve.start(http_options={"host": HOST, "port": PORT})

        customers = [f"customer-{i}" for i in range(1000)]
        repeat_ids = customers[:200]

        deploy(weight=0.2)
        with requests.Session() as session:
            baseline = assignments(session, customers)
            initial_v2 = {customer_id for customer_id, arm in baseline.items() if arm == "v2"}
            share = len(initial_v2) / len(customers)
            assert 0.16 <= share <= 0.24, f"Candidate share out of range: {share:.1%}"
            print(f"PASS (a) Candidate share: {len(initial_v2)}/1000 ({share:.1%}) at weight 20%")

            expected = {customer_id: baseline[customer_id] for customer_id in repeat_ids}
            assert assignments(session, repeat_ids) == expected, "An arm changed on repeat"
            print("PASS (b) 200/200 repeat requests kept their original arm")

            serve.delete(APP_NAME)
            deploy(weight=0.2)
            assert assignments(session, repeat_ids) == expected, "An arm changed after redeploy"
            print("PASS (c) 200/200 customers kept their arm after an identical redeploy")

            deploy(weight=0.5)
            expanded = assignments(session, customers)
            expanded_v2 = {customer_id for customer_id, arm in expanded.items() if arm == "v2"}
            assert initial_v2 <= expanded_v2, "An original candidate customer moved back to prod"
            print(
                f"PASS (d) All {len(initial_v2)} original v2 customers stayed on v2; "
                f"{len(expanded_v2)}/1000 customers now go to v2"
            )

            override_ids = [
                customer_id for customer_id in customers if expanded[customer_id] == "v1"
            ][:50]
            assert len(override_ids) == 50, "Need 50 v1 customers for the override check"
            for customer_id in override_ids:
                assert request_arm(session, customer_id, {"X-Beta": "1"}) == "v2", customer_id
                assert request_arm(session, customer_id) == "v1", customer_id
            print("PASS (e) 50/50 v1 customers forced to v2, then returned to v1 without the header")
    finally:
        serve.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()
