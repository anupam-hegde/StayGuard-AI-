import pytest
import pandas as pd
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture(scope="session")
def sample_raw_dict():
    return {
        "Booking_ID": "INN00001",
        "no_of_adults": 2,
        "no_of_children": 0,
        "no_of_weekend_nights": 1,
        "no_of_week_nights": 2,
        "type_of_meal_plan": "Meal Plan 1",
        "required_car_parking_space": 0,
        "room_type_reserved": "Room_Type 1",
        "lead_time": 45.0,
        "arrival_year": 2018,
        "arrival_month": 10,
        "arrival_date": 15,
        "market_segment_type": "Online",
        "repeated_guest": 0,
        "no_of_previous_cancellations": 0,
        "no_of_previous_bookings_not_canceled": 0,
        "avg_price_per_room": 115.5,
        "no_of_special_requests": 1,
        "booking_status": "Not_Canceled",
    }


@pytest.fixture(scope="session")
def sample_raw_df(sample_raw_dict):
    return pd.DataFrame([sample_raw_dict])


@pytest.fixture(scope="session")
def sample_api_payload():
    return {
        "no_of_adults": 2,
        "no_of_children": 0,
        "no_of_weekend_nights": 1,
        "no_of_week_nights": 2,
        "type_of_meal_plan": "Meal Plan 1",
        "required_car_parking_space": 0,
        "room_type_reserved": "Room_Type 1",
        "lead_time": 45.0,
        "arrival_year": 2018,
        "arrival_month": 10,
        "arrival_date": 15,
        "market_segment_type": "Online",
        "repeated_guest": 0,
        "no_of_previous_cancellations": 0,
        "no_of_previous_bookings_not_canceled": 0,
        "avg_price_per_room": 115.5,
        "no_of_special_requests": 1,
    }


@pytest.fixture(scope="module")
def api_client():
    with TestClient(app) as client:
        yield client
