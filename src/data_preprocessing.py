import os
import sys
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import lightgbm as lgb

from src.logger import get_logger
from src.custom_exception import CustomException
from src.schemas.data_schema import validate_raw_data, validate_processed_data
from config.paths_config import *
from utils.common_functions import read_yaml, load_data

logger = get_logger(__name__)


def get_preprocessor(categorical_cols: list, numerical_cols: list) -> ColumnTransformer:
    """
    Builds a reusable Scikit-Learn ColumnTransformer for numerical and categorical features.
    """
    try:
        logger.info("Constructing feature transformation pipelines")

        # Numerical pipeline: Median imputation followed by standard scaling
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        # Categorical pipeline: Frequent value imputation followed by One-Hot Encoding
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, numerical_cols),
                ("cat", cat_pipeline, categorical_cols),
            ],
            remainder="drop",
        )

        logger.info("ColumnTransformer successfully constructed")
        return preprocessor

    except Exception as e:
        logger.error(f"Error while constructing ColumnTransformer: {e}")
        raise CustomException("Failed to construct preprocessor pipeline", sys)


def build_pipeline(
    categorical_cols: list, numerical_cols: list, classifier=None
) -> Pipeline:
    """
    Builds a cohesive end-to-end Scikit-Learn Pipeline combining preprocessing and estimator.
    """
    try:
        preprocessor = get_preprocessor(categorical_cols, numerical_cols)
        estimator = classifier or lgb.LGBMClassifier(random_state=42, verbose=-1)

        full_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ]
        )

        logger.info("Unified Scikit-Learn Pipeline constructed successfully")
        return full_pipeline

    except Exception as e:
        logger.error(f"Error while building unified pipeline: {e}")
        raise CustomException("Failed to build unified pipeline", sys)


class DataProcessor:
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir

        self.config = read_yaml(config_path)
        self.target_col = self.config["data_processing"].get("target_column", "booking_status")
        self.cat_cols = [
            col
            for col in self.config["data_processing"]["categorical_columns"]
            if col != self.target_col
        ]
        self.num_cols = self.config["data_processing"]["numerical_columns"]

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir, exist_ok=True)

    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans dataset by removing identifier columns, removing duplicate rows,
        and encoding the binary target variable.
        """
        try:
            logger.info("Cleaning dataset and encoding target label")

            # Drop identifiers and CSV index columns if present
            cols_to_drop = [
                col for col in ["Unnamed: 0", "Booking_ID"] if col in df.columns
            ]
            if cols_to_drop:
                logger.info(f"Dropping identifier columns: {cols_to_drop}")
                df = df.drop(columns=cols_to_drop)

            # Drop duplicate records
            df = df.drop_duplicates()

            # Encode target variable: 1 for Canceled, 0 for Not_Canceled
            if self.target_col in df.columns and df[self.target_col].dtype == object:
                logger.info(f"Encoding target column '{self.target_col}' (Canceled -> 1, Not_Canceled -> 0)")
                df[self.target_col] = (
                    df[self.target_col].astype(str).str.strip() == "Canceled"
                ).astype(int)

            return df

        except Exception as e:
            logger.error(f"Error during dataset cleaning: {e}")
            raise CustomException("Error while cleaning dataset", sys)

    def save_data(self, df: pd.DataFrame, file_path: str):
        try:
            logger.info(f"Saving cleaned dataset to {file_path}")
            df.to_csv(file_path, index=False)
            logger.info(f"Data saved successfully to {file_path}")

        except Exception as e:
            logger.error(f"Error during saving data: {e}")
            raise CustomException("Error while saving processed data", sys)

    def process(self):
        try:
            logger.info("Loading raw partitions for data processing")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            # 1. Enforce input schema validation
            logger.info("Enforcing raw schema contract validation on train & test sets")
            train_df = validate_raw_data(train_df)
            test_df = validate_raw_data(test_df)

            # 2. Clean datasets and encode target
            train_df = self.clean_dataset(train_df)
            test_df = self.clean_dataset(test_df)

            # 3. Enforce processed schema contract validation
            logger.info("Enforcing processed schema validation on cleaned datasets")
            train_df = validate_processed_data(train_df)
            test_df = validate_processed_data(test_df)

            # 4. Save cleaned partitions
            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)

            logger.info("Data processing stage completed successfully")
            return train_df, test_df

        except Exception as e:
            logger.error(f"Error in data preprocessing stage: {e}")
            raise CustomException("Data preprocessing pipeline failed", sys)


if __name__ == "__main__":
    processor = DataProcessor(
        TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH
    )
    processor.process()
