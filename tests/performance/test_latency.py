import time
import pytest
from fastapi.testclient import TestClient


def test_single_prediction_latency_benchmark(api_client: TestClient, sample_api_payload):
    """
    Benchmarks API inference latency for single prediction requests.
    Enforces that average inference latency remains well below the 50ms production SLA.
    """
    # 1. Warm-up runs to ensure model weights, Python bytecode, and caches are hot
    for _ in range(5):
        warmup_resp = api_client.post("/predict", json=sample_api_payload)
        assert warmup_resp.status_code == 200

    # 2. Benchmark 30 sequential inference requests
    latencies_ms = []
    iterations = 30

    for _ in range(iterations):
        start_time = time.perf_counter()
        response = api_client.post("/predict", json=sample_api_payload)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        assert response.status_code == 200
        latencies_ms.append(elapsed_ms)

    mean_latency = sum(latencies_ms) / len(latencies_ms)
    p95_latency = sorted(latencies_ms)[int(0.95 * len(latencies_ms))]

    print(
        f"\n[Latency Benchmark] Mean: {mean_latency:.2f}ms | P95: {p95_latency:.2f}ms | Min: {min(latencies_ms):.2f}ms | Max: {max(latencies_ms):.2f}ms"
    )

    # Production SLA: Mean latency must be < 50ms
    assert (
        mean_latency < 50.0
    ), f"Mean inference latency ({mean_latency:.2f}ms) exceeded 50ms SLA limit"
