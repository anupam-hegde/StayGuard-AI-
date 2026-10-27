import pytest
from fastapi.testclient import TestClient


def test_get_root(api_client: TestClient):
    """
    Verifies root endpoint returns service info and status 200.
    """
    response = api_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert "docs_url" in data


def test_get_health(api_client: TestClient):
    """
    Verifies that GET /health returns HTTP 200 and indicates the model pipeline is loaded.
    """
    response = api_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "model_path" in data
    assert "version" in data
    assert "timestamp" in data


def test_post_predict_valid_payload(api_client: TestClient, sample_api_payload):
    """
    Verifies that POST /predict returns HTTP 200 with complete prediction schema on valid input.
    """
    response = api_client.post("/predict", json=sample_api_payload)
    assert response.status_code == 200
    data = response.json()

    assert "prediction" in data
    assert data["prediction"] in (0, 1)

    assert "cancellation_probability" in data
    assert 0.0 <= data["cancellation_probability"] <= 1.0

    assert "status_label" in data
    assert data["status_label"] in ("Not_Canceled", "Canceled")

    assert "risk_level" in data
    assert data["risk_level"] in ("Low", "Medium", "High")

    # Verify structured middleware headers
    assert "X-Request-ID" in response.headers
    assert "X-Process-Time" in response.headers


@pytest.mark.parametrize(
    "corrupted_field, invalid_value",
    [
        ("lead_time", -10.0),  # Violates ge=0
        ("type_of_meal_plan", "Invalid Meal"),  # Violates Literal options
        ("room_type_reserved", "Unknown Room"),  # Violates Literal options
        ("market_segment_type", "Illegal Segment"),  # Violates Literal options
        ("arrival_month", 13),  # Violates le=12
        ("arrival_date", 35),  # Violates le=31
        ("required_car_parking_space", 5),  # Violates Literal[0, 1]
    ],
)
def test_post_predict_invalid_fields_returns_422(
    api_client: TestClient, sample_api_payload, corrupted_field, invalid_value
):
    """
    Verifies that POST /predict returns HTTP 422 Unprocessable Entity when fields violate Pydantic schema rules.
    """
    payload = sample_api_payload.copy()
    payload[corrupted_field] = invalid_value
    response = api_client.post("/predict", json=payload)

    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_post_predict_missing_payload_returns_422(api_client: TestClient):
    """
    Verifies that POST /predict returns HTTP 422 when body is empty or malformed.
    """
    response = api_client.post("/predict", json={})
    assert response.status_code == 422


def test_post_predict_batch_valid_payload(api_client: TestClient, sample_api_payload):
    """
    Verifies that POST /predict-batch processes multiple records and returns batch predictions.
    """
    batch_payload = {
        "items": [
            sample_api_payload,
            {**sample_api_payload, "lead_time": 180.0, "avg_price_per_room": 220.0},
        ]
    }
    response = api_client.post("/predict-batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_records"] == 2
    assert len(data["predictions"]) == 2

    for pred_item in data["predictions"]:
        assert pred_item["prediction"] in (0, 1)
        assert 0.0 <= pred_item["cancellation_probability"] <= 1.0
        assert pred_item["status_label"] in ("Not_Canceled", "Canceled")
        assert pred_item["risk_level"] in ("Low", "Medium", "High")
