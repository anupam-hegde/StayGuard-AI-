from typing import List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


class ReservationFeatures(BaseModel):
    """
    Input schema for hotel reservation cancellation prediction features.
    """
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )

    no_of_adults: int = Field(
        default=1,
        ge=0,
        le=10,
        description="Number of adults in the reservation",
        examples=[2],
    )
    no_of_children: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Number of children in the reservation",
        examples=[0],
    )
    no_of_weekend_nights: int = Field(
        default=0,
        ge=0,
        le=20,
        description="Number of weekend nights (Saturday/Sunday) booked",
        examples=[1],
    )
    no_of_week_nights: int = Field(
        default=1,
        ge=0,
        le=30,
        description="Number of weekday nights (Monday to Friday) booked",
        examples=[2],
    )
    type_of_meal_plan: Literal[
        "Meal Plan 1", "Meal Plan 2", "Meal Plan 3", "Not Selected"
    ] = Field(
        default="Meal Plan 1",
        description="Meal plan chosen by the customer",
        examples=["Meal Plan 1"],
    )
    required_car_parking_space: Literal[0, 1] = Field(
        default=0,
        description="Binary flag: 1 if parking space is requested, 0 otherwise",
        examples=[0],
    )
    room_type_reserved: Literal[
        "Room_Type 1",
        "Room_Type 2",
        "Room_Type 3",
        "Room_Type 4",
        "Room_Type 5",
        "Room_Type 6",
        "Room_Type 7",
    ] = Field(
        default="Room_Type 1",
        description="Type of room reserved by the guest",
        examples=["Room_Type 1"],
    )
    lead_time: float = Field(
        default=30.0,
        ge=0.0,
        le=1000.0,
        description="Number of days between booking and arrival",
        examples=[45.0],
    )
    arrival_year: int = Field(
        default=2018,
        ge=2015,
        le=2035,
        description="Year of arrival",
        examples=[2018],
    )
    arrival_month: int = Field(
        default=10,
        ge=1,
        le=12,
        description="Month of arrival (1-12)",
        examples=[10],
    )
    arrival_date: int = Field(
        default=15,
        ge=1,
        le=31,
        description="Day of arrival month (1-31)",
        examples=[15],
    )
    market_segment_type: Literal[
        "Online", "Offline", "Corporate", "Complementary", "Aviation"
    ] = Field(
        default="Online",
        description="Market segment designation",
        examples=["Online"],
    )
    repeated_guest: Literal[0, 1] = Field(
        default=0,
        description="Binary flag: 1 if customer is a returning guest, 0 otherwise",
        examples=[0],
    )
    no_of_previous_cancellations: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Number of previous bookings canceled by the customer",
        examples=[0],
    )
    no_of_previous_bookings_not_canceled: int = Field(
        default=0,
        ge=0,
        le=200,
        description="Number of previous bookings completed without cancellation",
        examples=[0],
    )
    avg_price_per_room: float = Field(
        default=100.0,
        ge=0.0,
        description="Average daily rate / price per room in currency units",
        examples=[115.5],
    )
    no_of_special_requests: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Total number of special requests made by the guest",
        examples=[1],
    )


class BatchReservationRequest(BaseModel):
    """
    Request model for batch inference containing a list of reservation feature objects.
    """
    items: List[ReservationFeatures] = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="List of reservation records for batch inference",
    )


class PredictionResponse(BaseModel):
    """
    Response model for single reservation cancellation prediction.
    """
    prediction: int = Field(
        ...,
        description="Predicted class: 1 (Canceled) or 0 (Not_Canceled)",
        examples=[0],
    )
    cancellation_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated probability that the booking will be canceled",
        examples=[0.1837],
    )
    status_label: str = Field(
        ...,
        description="Human-readable prediction label ('Canceled' or 'Not_Canceled')",
        examples=["Not_Canceled"],
    )
    risk_level: str = Field(
        ...,
        description="Categorized cancellation risk: 'Low', 'Medium', or 'High'",
        examples=["Low"],
    )


class BatchPredictionResponse(BaseModel):
    """
    Response model for batch prediction queries.
    """
    total_records: int = Field(
        ...,
        description="Total number of evaluated records",
        examples=[1],
    )
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of prediction results corresponding to input items",
    )


class HealthResponse(BaseModel):
    """
    Response model for health check and service metadata.
    """
    status: str = Field(
        default="healthy",
        description="Service health status ('healthy' or 'degraded')",
        examples=["healthy"],
    )
    model_loaded: bool = Field(
        ...,
        description="True if the ML model pipeline is loaded and ready for inference",
        examples=[True],
    )
    model_path: str = Field(
        ...,
        description="Filesystem path of the loaded model pipeline artifact",
        examples=["artifacts/models/lgbm_model.pkl"],
    )
    version: str = Field(
        default="1.0.0",
        description="API service version",
        examples=["1.0.0"],
    )
    timestamp: str = Field(
        ...,
        description="UTC timestamp of the health check query",
        examples=["2026-10-07T18:30:00Z"],
    )
