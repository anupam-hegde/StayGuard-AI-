import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.data_preprocessing import get_preprocessor, build_pipeline, DataProcessor
from config.paths_config import CONFIG_PATH


@pytest.fixture
def feature_columns():
    cat_cols = [
        "type_of_meal_plan",
        "required_car_parking_space",
        "room_type_reserved",
        "market_segment_type",
        "repeated_guest",
    ]
    num_cols = [
        "no_of_adults",
        "no_of_children",
        "no_of_weekend_nights",
        "no_of_week_nights",
        "lead_time",
        "arrival_year",
        "arrival_month",
        "arrival_date",
        "no_of_previous_cancellations",
        "no_of_previous_bookings_not_canceled",
        "avg_price_per_room",
        "no_of_special_requests",
    ]
    return cat_cols, num_cols


def test_get_preprocessor_structure(feature_columns):
    """
    Verifies that get_preprocessor returns a valid ColumnTransformer with numerical and categorical branches.
    """
    cat_cols, num_cols = feature_columns
    preprocessor = get_preprocessor(cat_cols, num_cols)

    assert isinstance(preprocessor, ColumnTransformer)
    transformer_names = [name for name, _, _ in preprocessor.transformers]
    assert "num" in transformer_names
    assert "cat" in transformer_names


def test_build_pipeline_structure(feature_columns):
    """
    Verifies that build_pipeline returns a unified Scikit-Learn Pipeline with preprocessor and classifier steps.
    """
    cat_cols, num_cols = feature_columns
    pipe = build_pipeline(cat_cols, num_cols)

    assert isinstance(pipe, Pipeline)
    assert "preprocessor" in pipe.named_steps
    assert "classifier" in pipe.named_steps


def test_preprocessor_fit_transform(feature_columns, sample_raw_df):
    """
    Tests fitting and transforming valid features through the preprocessor.
    """
    cat_cols, num_cols = feature_columns
    preprocessor = get_preprocessor(cat_cols, num_cols)

    X = sample_raw_df[num_cols + cat_cols]
    transformed = preprocessor.fit_transform(X)

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == len(sample_raw_df)
    assert not np.isnan(transformed).any()


def test_preprocessor_unseen_category_handling(feature_columns, sample_raw_dict):
    """
    Verifies that unseen/novel categorical values are ignored gracefully without throwing errors.
    """
    cat_cols, num_cols = feature_columns
    preprocessor = get_preprocessor(cat_cols, num_cols)

    # Fit on training record
    train_df = pd.DataFrame([sample_raw_dict])
    preprocessor.fit(train_df[num_cols + cat_cols])

    # Transform test record with novel/unseen category
    test_record = sample_raw_dict.copy()
    test_record["type_of_meal_plan"] = "Alien Meal Plan"
    test_record["market_segment_type"] = "Metaverse"
    test_df = pd.DataFrame([test_record])

    transformed = preprocessor.transform(test_df[num_cols + cat_cols])

    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()


def test_preprocessor_null_imputation(feature_columns, sample_raw_dict):
    """
    Verifies that null / NaN values in numerical and categorical features are properly imputed.
    """
    cat_cols, num_cols = feature_columns
    preprocessor = get_preprocessor(cat_cols, num_cols)

    # Fit on clean sample
    train_df = pd.DataFrame([sample_raw_dict, sample_raw_dict])
    preprocessor.fit(train_df[num_cols + cat_cols])

    # Inject NaNs into test record
    test_record = sample_raw_dict.copy()
    test_record["lead_time"] = np.nan
    test_record["avg_price_per_room"] = np.nan
    test_record["type_of_meal_plan"] = None
    test_df = pd.DataFrame([test_record])

    transformed = preprocessor.transform(test_df[num_cols + cat_cols])

    assert isinstance(transformed, np.ndarray)
    assert not np.isnan(transformed).any()


def test_data_processor_clean_dataset(sample_raw_dict):
    """
    Verifies DataProcessor clean_dataset drops metadata columns, removes duplicates, and encodes binary target.
    """
    processor = DataProcessor("dummy_train.csv", "dummy_test.csv", "dummy_dir", CONFIG_PATH)

    df_raw = pd.DataFrame([
        sample_raw_dict,
        sample_raw_dict,  # Duplicate row
        {**sample_raw_dict, "Booking_ID": "INN00002", "booking_status": "Canceled"},
    ])
    df_raw["Unnamed: 0"] = [0, 1, 2]

    cleaned = processor.clean_dataset(df_raw)

    assert "Unnamed: 0" not in cleaned.columns
    assert "Booking_ID" not in cleaned.columns
    assert len(cleaned) == 2  # Deduplicated from 3 to 2
    assert set(cleaned["booking_status"].unique()) == {0, 1}
