import pytest
import pandas as pd
from src.schemas.data_schema import validate_raw_data, validate_processed_data
from src.custom_exception import CustomException


def test_validate_raw_data_valid_input(sample_raw_df):
    """
    Verifies that a well-formed raw DataFrame passes schema validation cleanly.
    """
    validated = validate_raw_data(sample_raw_df)
    assert isinstance(validated, pd.DataFrame)
    assert len(validated) == len(sample_raw_df)


@pytest.mark.parametrize(
    "corrupted_field, invalid_value",
    [
        ("no_of_adults", -1),
        ("no_of_children", -2),
        ("no_of_weekend_nights", -1),
        ("no_of_week_nights", -3),
        ("lead_time", -10.0),
        ("avg_price_per_room", -50.0),
        ("no_of_previous_cancellations", -1),
        ("no_of_special_requests", -1),
    ],
)
def test_validate_raw_data_rejects_negative_bounds(sample_raw_dict, corrupted_field, invalid_value):
    """
    Verifies that Pandera rejects negative numerical values where >= 0 is required.
    """
    corrupted_data = sample_raw_dict.copy()
    corrupted_data[corrupted_field] = invalid_value
    df_corrupted = pd.DataFrame([corrupted_data])

    with pytest.raises(CustomException):
        validate_raw_data(df_corrupted)


@pytest.mark.parametrize(
    "corrupted_field, invalid_value",
    [
        ("type_of_meal_plan", "Invalid Meal Option"),
        ("room_type_reserved", "Penthouse Deluxe 99"),
        ("market_segment_type", "Dark Web"),
        ("booking_status", "Maybe_Canceled"),
        ("required_car_parking_space", 5),
        ("repeated_guest", 2),
    ],
)
def test_validate_raw_data_rejects_invalid_categories(sample_raw_dict, corrupted_field, invalid_value):
    """
    Verifies that Pandera rejects domain values outside allowed categorical enumerations.
    """
    corrupted_data = sample_raw_dict.copy()
    corrupted_data[corrupted_field] = invalid_value
    df_corrupted = pd.DataFrame([corrupted_data])

    with pytest.raises(CustomException):
        validate_raw_data(df_corrupted)


@pytest.mark.parametrize(
    "corrupted_field, invalid_value",
    [
        ("arrival_month", 0),
        ("arrival_month", 13),
        ("arrival_date", 0),
        ("arrival_date", 32),
        ("arrival_year", 1990),
    ],
)
def test_validate_raw_data_rejects_invalid_date_ranges(sample_raw_dict, corrupted_field, invalid_value):
    """
    Verifies that Pandera rejects invalid month, day, or year values.
    """
    corrupted_data = sample_raw_dict.copy()
    corrupted_data[corrupted_field] = invalid_value
    df_corrupted = pd.DataFrame([corrupted_data])

    with pytest.raises(CustomException):
        validate_raw_data(df_corrupted)


def test_validate_raw_data_rejects_missing_mandatory_columns(sample_raw_dict):
    """
    Verifies that Pandera rejects DataFrames missing required features.
    """
    corrupted_data = sample_raw_dict.copy()
    del corrupted_data["lead_time"]
    del corrupted_data["booking_status"]
    df_corrupted = pd.DataFrame([corrupted_data])

    with pytest.raises(CustomException):
        validate_raw_data(df_corrupted)


def test_validate_processed_data_valid():
    """
    Verifies that clean processed DataFrames pass processed contract validation.
    """
    clean_processed_df = pd.DataFrame({
        "lead_time": [30.0, 10.0],
        "avg_price_per_room": [100.0, 150.0],
        "booking_status": [0, 1],
    })
    validated = validate_processed_data(clean_processed_df)
    assert isinstance(validated, pd.DataFrame)


def test_validate_processed_data_rejects_nulls():
    """
    Verifies that processed contract validation rejects any NaN/null values.
    """
    null_df = pd.DataFrame({
        "lead_time": [30.0, None],
        "booking_status": [0, 1],
    })
    with pytest.raises(CustomException):
        validate_processed_data(null_df)


def test_validate_processed_data_rejects_invalid_target():
    """
    Verifies that processed contract validation rejects target values other than 0 or 1.
    """
    invalid_target_df = pd.DataFrame({
        "lead_time": [30.0, 10.0],
        "booking_status": [0, 99],
    })
    with pytest.raises(CustomException):
        validate_processed_data(invalid_target_df)
