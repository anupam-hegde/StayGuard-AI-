import sys
import pandas as pd
from src.logger import get_logger
from src.custom_exception import CustomException

try:
    import pandera.pandas as pa
    from pandera.pandas import Column, Check, DataFrameSchema
except ImportError:
    import pandera as pa
    from pandera import Column, Check, DataFrameSchema

logger = get_logger(__name__)

# ==============================================================================
# 1. RAW DATA CONTRACT & SCHEMA
# ==============================================================================

RAW_DATA_SCHEMA = DataFrameSchema(
    columns={
        "Booking_ID": Column(
            pa.String,
            nullable=False,
            required=False,
            description="Unique identifier for each reservation",
        ),
        "no_of_adults": Column(
            pa.Int,
            checks=[Check.greater_than_or_equal_to(0, error="no_of_adults must be >= 0")],
            nullable=False,
            required=True,
            coerce=True,
            description="Number of adults in the reservation",
        ),
        "no_of_children": Column(
            pa.Int,
            checks=[Check.greater_than_or_equal_to(0, error="no_of_children must be >= 0")],
            nullable=False,
            required=True,
            coerce=True,
            description="Number of children in the reservation",
        ),
        "no_of_weekend_nights": Column(
            pa.Int,
            checks=[Check.greater_than_or_equal_to(0, error="no_of_weekend_nights must be >= 0")],
            nullable=False,
            required=True,
            coerce=True,
            description="Number of weekend nights booked",
        ),
        "no_of_week_nights": Column(
            pa.Int,
            checks=[Check.greater_than_or_equal_to(0, error="no_of_week_nights must be >= 0")],
            nullable=False,
            required=True,
            coerce=True,
            description="Number of weekday nights booked",
        ),
        "type_of_meal_plan": Column(
            pa.String,
            checks=[
                Check.isin(
                    ["Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"],
                    error="Invalid type_of_meal_plan category",
                )
            ],
            nullable=False,
            required=True,
            coerce=True,
            description="Meal plan chosen by the customer",
        ),
        "required_car_parking_space": Column(
            pa.Int,
            checks=[
                Check.isin([0, 1], error="required_car_parking_space must be binary (0 or 1)")
            ],
            nullable=False,
            required=True,
            coerce=True,
            description="Binary flag for parking requirement",
        ),
        "room_type_reserved": Column(
            pa.String,
            checks=[
                Check.isin(
                    [
                        "Room_Type 1",
                        "Room_Type 2",
                        "Room_Type 3",
                        "Room_Type 4",
                        "Room_Type 5",
                        "Room_Type 6",
                        "Room_Type 7",
                    ],
                    error="Invalid room_type_reserved category",
                )
            ],
            nullable=False,
            required=True,
            coerce=True,
            description="Type of room reserved by customer",
        ),
        "lead_time": Column(
            pa.Float,
            checks=[Check.greater_than_or_equal_to(0.0, error="lead_time must be >= 0")],
            nullable=False,
            required=True,
            coerce=True,
            description="Days between booking date and arrival date",
        ),
        "arrival_year": Column(
            pa.Int,
            checks=[Check.greater_than_or_equal_to(2000, error="arrival_year must be >= 2000")],
            nullable=False,
            required=True,
            coerce=True,
            description="Year of arrival",
        ),
        "arrival_month": Column(
            pa.Int,
            checks=[Check.in_range(1, 12, error="arrival_month must be between 1 and 12")],
            nullable=False,
            required=True,
            coerce=True,
            description="Month of arrival",
        ),
        "arrival_date": Column(
            pa.Int,
            checks=[Check.in_range(1, 31, error="arrival_date must be between 1 and 31")],
            nullable=False,
            required=True,
            coerce=True,
            description="Day of the month of arrival",
        ),
        "market_segment_type": Column(
            pa.String,
            checks=[
                Check.isin(
                    ["Online", "Offline", "Corporate", "Complementary", "Aviation"],
                    error="Invalid market_segment_type category",
                )
            ],
            nullable=False,
            required=True,
            coerce=True,
            description="Market segment designation",
        ),
        "repeated_guest": Column(
            pa.Int,
            checks=[Check.isin([0, 1], error="repeated_guest must be binary (0 or 1)")],
            nullable=False,
            required=True,
            coerce=True,
            description="Binary flag indicating repeated guest",
        ),
        "no_of_previous_cancellations": Column(
            pa.Int,
            checks=[
                Check.greater_than_or_equal_to(0, error="no_of_previous_cancellations must be >= 0")
            ],
            nullable=False,
            required=True,
            coerce=True,
            description="Number of previous bookings canceled",
        ),
        "no_of_previous_bookings_not_canceled": Column(
            pa.Int,
            checks=[
                Check.greater_than_or_equal_to(
                    0, error="no_of_previous_bookings_not_canceled must be >= 0"
                )
            ],
            nullable=False,
            required=True,
            coerce=True,
            description="Number of previous bookings not canceled",
        ),
        "avg_price_per_room": Column(
            pa.Float,
            checks=[
                Check.greater_than_or_equal_to(0.0, error="avg_price_per_room must be >= 0.0")
            ],
            nullable=False,
            required=True,
            coerce=True,
            description="Average price per day for the reservation",
        ),
        "no_of_special_requests": Column(
            pa.Int,
            checks=[
                Check.greater_than_or_equal_to(0, error="no_of_special_requests must be >= 0")
            ],
            nullable=False,
            required=True,
            coerce=True,
            description="Total number of special requests made",
        ),
        "booking_status": Column(
            pa.String,
            checks=[
                Check.isin(
                    ["Not_Canceled", "Canceled", "Not Canceled"],
                    error="booking_status must be 'Not_Canceled' or 'Canceled'",
                )
            ],
            nullable=False,
            required=True,
            coerce=True,
            description="Target label: booking status of the reservation",
        ),
        "Unnamed: 0": Column(
            pa.Int,
            nullable=True,
            required=False,
            coerce=True,
            description="Optional CSV row index column",
        ),
    },
    strict=False,
    coerce=True,
)


# ==============================================================================
# 2. PROCESSED DATA CONTRACT & SCHEMA
# ==============================================================================

PROCESSED_DATA_SCHEMA = DataFrameSchema(
    columns={
        "booking_status": Column(
            pa.Int,
            checks=[
                Check.isin([0, 1], error="Target booking_status must be binary integer (0 or 1)")
            ],
            nullable=False,
            required=True,
            coerce=True,
            description="Target encoded label (0 = Not Canceled, 1 = Canceled)",
        )
    },
    checks=[
        Check(
            lambda df: not df.isnull().any().any(),
            error="Processed dataset must not contain any null values",
        ),
    ],
    strict=False,
    coerce=True,
)


# ==============================================================================
# 3. SCHEMA VALIDATION HELPER FUNCTIONS
# ==============================================================================

def validate_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates the raw input DataFrame against RAW_DATA_SCHEMA.
    Raises CustomException if validation fails.
    """
    try:
        logger.info("Validating raw data schema with Pandera")
        validated_df = RAW_DATA_SCHEMA.validate(df, lazy=True)
        logger.info("Raw data schema validation successful")
        return validated_df
    except Exception as e:
        logger.error(f"Raw data schema validation contract failed: {e}")
        raise CustomException(f"Raw data validation error: {e}", sys)


def validate_processed_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates preprocessed DataFrame against PROCESSED_DATA_SCHEMA.
    Raises CustomException if validation fails.
    """
    try:
        logger.info("Validating processed data schema with Pandera")
        validated_df = PROCESSED_DATA_SCHEMA.validate(df, lazy=True)
        logger.info("Processed data schema validation successful")
        return validated_df
    except Exception as e:
        logger.error(f"Processed data schema validation contract failed: {e}")
        raise CustomException(f"Processed data validation error: {e}", sys)
