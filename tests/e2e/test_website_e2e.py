import pytest
import requests
from bs4 import BeautifulSoup

BASE_URL = "http://127.0.0.1:8080"


@pytest.fixture(scope="module")
def web_server_response():
    try:
        resp = requests.get(BASE_URL, timeout=3)
        return resp
    except Exception as e:
        pytest.fail(f"Web server at {BASE_URL} is unreachable: {e}")


def test_homepage_status_and_seo(web_server_response):
    assert web_server_response.status_code == 200
    soup = BeautifulSoup(web_server_response.text, "html.parser")
    assert "StayGuard AI" in soup.title.string
    meta_desc = soup.find("meta", attrs={"name": "description"})
    assert meta_desc is not None


def test_dom_elements_present(web_server_response):
    soup = BeautifulSoup(web_server_response.text, "html.parser")
    assert soup.find("form", id="predictionForm") is not None
    assert soup.find(id="lead_time_range") is not None
    assert soup.find(id="avg_price_range") is not None
    assert soup.find(id="gaugeRing") is not None
    assert soup.find(id="riskStatusPill") is not None
    assert soup.find(id="actionRecommendation") is not None


@pytest.mark.parametrize("asset_path", [
    "/static/style.css",
    "/static/images/resort.jpg",
    "/static/images/suite.jpg",
    "/static/images/concierge.jpg",
])
def test_static_assets_status(asset_path):
    r = requests.get(f"{BASE_URL}{asset_path}", timeout=3)
    assert r.status_code == 200
    assert len(r.content) > 1000


def test_api_predict_high_risk():
    payload = {
        "lead_time": 280,
        "avg_price_per_room": 85.0,
        "no_of_special_requests": 0,
        "market_segment_type": "Online",
        "repeated_guest": 0,
        "no_of_previous_cancellations": 2,
    }
    resp = requests.post(f"{BASE_URL}/api/predict", json=payload, timeout=3)
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction"] == 1
    assert data["risk_level"] == "High"
    assert data["cancellation_probability"] > 0.70
    assert "deposit" in data["action"].lower() or "risk" in data["action"].lower()


def test_api_predict_low_risk_corporate():
    payload = {
        "lead_time": 10,
        "avg_price_per_room": 130.0,
        "no_of_special_requests": 1,
        "market_segment_type": "Corporate",
        "repeated_guest": 1,
        "no_of_previous_cancellations": 0,
        "no_of_previous_bookings_not_canceled": 3,
    }
    resp = requests.post(f"{BASE_URL}/api/predict", json=payload, timeout=3)
    assert resp.status_code == 200
    data = resp.json()
    assert data["prediction"] == 0
    assert data["risk_level"] == "Low"
    assert data["cancellation_probability"] < 0.40


def test_ssr_form_post():
    form_data = {
        "no_of_adults": "2",
        "no_of_children": "0",
        "no_of_weekend_nights": "1",
        "no_of_week_nights": "2",
        "type_of_meal_plan": "Meal Plan 1",
        "required_car_parking_space": "0",
        "room_type_reserved": "Room_Type 1",
        "lead_time": "15",
        "arrival_year": "2018",
        "arrival_month": "10",
        "arrival_date": "15",
        "market_segment_type": "Online",
        "repeated_guest": "0",
        "no_of_previous_cancellations": "0",
        "no_of_previous_bookings_not_canceled": "0",
        "avg_price_per_room": "110.0",
        "no_of_special_requests": "1",
    }
    resp = requests.post(BASE_URL, data=form_data, timeout=5)
    assert resp.status_code == 200
    soup = BeautifulSoup(resp.text, "html.parser")
    pct_display = soup.find(id="gaugePct")
    assert pct_display is not None
    assert "%" in pct_display.text
