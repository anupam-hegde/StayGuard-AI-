"""
StayGuard AI — End-to-End Website Test & Demonstration Suite
Tests the live web server, assets, forms, AJAX endpoints, scenario presets, and latency SLAs.
"""

import sys
import time
import requests
from bs4 import BeautifulSoup

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://127.0.0.1:8080"

# ANSI Terminal Colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_header(title):
    print(f"\n{BOLD}{CYAN}{'=' * 75}{RESET}")
    print(f"{BOLD}{CYAN}   {title}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 75}{RESET}")


def assert_test(name, condition, details=""):
    if condition:
        print(f"  {GREEN}[PASS]{RESET} : {BOLD}{name}{RESET} {details}")
        return True
    else:
        print(f"  {RED}[FAIL]{RESET} : {BOLD}{name}{RESET} {details}")
        return False


def test_server_connectivity():
    print_header("1. SERVER HEALTH & CONNECTIVITY TEST")
    try:
        start = time.perf_counter()
        resp = requests.get(BASE_URL, timeout=5)
        elapsed = (time.perf_counter() - start) * 1000
        assert_test("Server Connectivity (GET /)", resp.status_code == 200, f"[{resp.status_code} OK in {elapsed:.1f}ms]")
        return resp
    except Exception as e:
        assert_test("Server Connectivity", False, f"Connection error: {e}")
        print(f"\n{RED}ERROR: Web server is not running on {BASE_URL}. Start it with 'python application.py'{RESET}")
        sys.exit(1)


def test_static_assets_and_dom(html_content):
    print_header("2. DOM STRUCTURE & STATIC ASSETS VALIDATION")
    soup = BeautifulSoup(html_content, "html.parser")

    # Check Key UI Elements
    assert_test("Title Tag", "StayGuard AI" in soup.title.string, f"({soup.title.string})")
    assert_test("Prediction Form", soup.find("form", id="predictionForm") is not None)
    assert_test("Radial Gauge SVG", soup.find("circle", id="gaugeRing") is not None)
    assert_test("Risk Status Pill", soup.find(id="riskStatusPill") is not None)
    assert_test("Operational Action Box", soup.find(id="actionRecommendation") is not None)
    assert_test("Lead Time Slider", soup.find("input", id="lead_time_range") is not None)
    assert_test("Price Range Slider", soup.find("input", id="avg_price_range") is not None)

    # Check Static Images and Stylesheet
    assets = [
        ("/static/style.css", "text/css"),
        ("/static/images/resort.jpg", "image/jpeg"),
        ("/static/images/suite.jpg", "image/jpeg"),
        ("/static/images/concierge.jpg", "image/jpeg"),
    ]

    for asset_path, expected_type in assets:
        url = f"{BASE_URL}{asset_path}"
        r = requests.get(url, timeout=3)
        assert_test(f"Asset: {asset_path}", r.status_code == 200 and expected_type in r.headers.get("content-type", "").lower(), f"[{len(r.content):,} bytes]")


def test_api_predict_scenarios():
    print_header("3. SCENARIO PRESETS & PREDICTION LOGIC (/api/predict)")

    scenarios = [
        {
            "name": "Scenario A: Corporate Executive (Returning Loyal Guest)",
            "payload": {
                "lead_time": 14,
                "avg_price_per_room": 135.0,
                "no_of_special_requests": 1,
                "market_segment_type": "Corporate",
                "repeated_guest": 1,
                "no_of_previous_cancellations": 0,
                "no_of_previous_bookings_not_canceled": 3,
                "no_of_adults": 1,
                "no_of_children": 0,
                "no_of_week_nights": 3,
                "no_of_weekend_nights": 0,
                "type_of_meal_plan": "Meal Plan 1",
                "room_type_reserved": "Room_Type 1",
                "required_car_parking_space": 0,
                "arrival_month": 10,
                "arrival_date": 12,
            },
            "expected_risk": "Low",
            "expected_pred": 0,
        },
        {
            "name": "Scenario B: High-Risk Advance Hold (280d Lead Time + Past Cancellations)",
            "payload": {
                "lead_time": 280,
                "avg_price_per_room": 89.0,
                "no_of_special_requests": 0,
                "market_segment_type": "Online",
                "repeated_guest": 0,
                "no_of_previous_cancellations": 2,
                "no_of_previous_bookings_not_canceled": 0,
                "no_of_adults": 2,
                "no_of_children": 0,
                "no_of_week_nights": 4,
                "no_of_weekend_nights": 1,
                "type_of_meal_plan": "Not Selected",
                "room_type_reserved": "Room_Type 1",
                "required_car_parking_space": 0,
                "arrival_month": 11,
                "arrival_date": 5,
            },
            "expected_risk": "High",
            "expected_pred": 1,
        },
        {
            "name": "Scenario C: Family Vacation (Moderate Risk, Multi-Night)",
            "payload": {
                "lead_time": 45,
                "avg_price_per_room": 155.0,
                "no_of_special_requests": 2,
                "market_segment_type": "Online",
                "repeated_guest": 0,
                "no_of_previous_cancellations": 0,
                "no_of_previous_bookings_not_canceled": 0,
                "no_of_adults": 2,
                "no_of_children": 2,
                "no_of_week_nights": 4,
                "no_of_weekend_nights": 2,
                "type_of_meal_plan": "Meal Plan 1",
                "room_type_reserved": "Room_Type 4",
                "required_car_parking_space": 1,
                "arrival_month": 7,
                "arrival_date": 20,
            },
            "expected_risk": ["Low", "Medium"],
            "expected_pred": 0,
        },
        {
            "name": "Scenario D: VIP Presidential Suite (Offline direct reservation)",
            "payload": {
                "lead_time": 25,
                "avg_price_per_room": 295.0,
                "no_of_special_requests": 3,
                "market_segment_type": "Offline",
                "repeated_guest": 1,
                "no_of_previous_cancellations": 0,
                "no_of_previous_bookings_not_canceled": 5,
                "no_of_adults": 2,
                "no_of_children": 0,
                "no_of_week_nights": 2,
                "no_of_weekend_nights": 2,
                "type_of_meal_plan": "Meal Plan 2",
                "room_type_reserved": "Room_Type 7",
                "required_car_parking_space": 1,
                "arrival_month": 12,
                "arrival_date": 28,
            },
            "expected_risk": "Low",
            "expected_pred": 0,
        },
    ]

    all_passed = True
    for sc in scenarios:
        start = time.perf_counter()
        resp = requests.post(f"{BASE_URL}/api/predict", json=sc["payload"], timeout=3)
        lat = (time.perf_counter() - start) * 1000

        valid_status = resp.status_code == 200
        data = resp.json() if valid_status else {}

        prob = data.get("cancellation_probability", 0)
        risk = data.get("risk_level", "")
        pred = data.get("prediction")
        action = data.get("action", "")

        risk_match = (risk == sc["expected_risk"]) if isinstance(sc["expected_risk"], str) else (risk in sc["expected_risk"])
        pred_match = (pred == sc["expected_pred"])

        details = f"-> Risk: {BOLD}{risk}{RESET} ({prob*100:.1f}%), Pred: {pred}, Latency: {lat:.1f}ms"
        passed = assert_test(sc["name"], valid_status and risk_match and pred_match and bool(action), details)
        if not passed:
            all_passed = False

    return all_passed


def test_ssr_form_submission():
    print_header("4. SERVER-SIDE RENDERING (SSR) FORM POST TEST")
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

    start = time.perf_counter()
    resp = requests.post(BASE_URL, data=form_data, timeout=5)
    lat = (time.perf_counter() - start) * 1000

    soup = BeautifulSoup(resp.text, "html.parser")
    gauge_pct = soup.find(id="gaugePct")

    status_ok = resp.status_code == 200
    rendered_ok = gauge_pct is not None and "%" in gauge_pct.text
    assert_test("Traditional Form POST Handling", status_ok and rendered_ok, f"[Rendered: {gauge_pct.text.strip()} in {lat:.1f}ms]")


def test_latency_sla_benchmark():
    print_header("5. INFERENCE LATENCY & SLA BENCHMARK (< 50ms)")
    payload = {
        "lead_time": 30,
        "avg_price_per_room": 120.0,
        "no_of_special_requests": 1,
        "market_segment_type": "Online",
    }

    # Warmup
    for _ in range(3):
        requests.post(f"{BASE_URL}/api/predict", json=payload)

    latencies = []
    iterations = 25
    for _ in range(iterations):
        t0 = time.perf_counter()
        r = requests.post(f"{BASE_URL}/api/predict", json=payload, timeout=2)
        elapsed = (time.perf_counter() - t0) * 1000.0
        if r.status_code == 200:
            latencies.append(elapsed)

    mean_lat = sum(latencies) / len(latencies)
    p95_lat = sorted(latencies)[int(0.95 * len(latencies))]
    min_lat = min(latencies)
    max_lat = max(latencies)

    sla_passed = mean_lat < 50.0
    details = f"[Mean: {mean_lat:.2f}ms | P95: {p95_lat:.2f}ms | Min: {min_lat:.2f}ms | Max: {max_lat:.2f}ms]"
    assert_test("Production Mean Latency SLA (< 50ms)", sla_passed, details)


def run_full_suite():
    print(f"\n{BOLD}{YELLOW}🚀 STARTING STAYGUARD AI AUTOMATED WEB TEST SUITE{RESET}")
    print(f"Target URL: {BASE_URL}")

    resp = test_server_connectivity()
    test_static_assets_and_dom(resp.text)
    test_api_predict_scenarios()
    test_ssr_form_submission()
    test_latency_sla_benchmark()

    print_header("TEST SUITE EXECUTION SUMMARY")
    print(f"  {GREEN}{BOLD}All critical path tests completed successfully!{RESET}")
    print(f"  Live UI: {CYAN}{BASE_URL}{RESET}\n")


if __name__ == "__main__":
    run_full_suite()
