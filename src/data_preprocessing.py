import os
import sys
import pandas as pd
import numpy as np
from src.logger import get_logger
from src.custom_exception import CustomException
from src.schemas.data_schema import validate_raw_data, validate_processed_data
from config.paths_config import *
from utils.common_functions import read_yaml, load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

logger = get_logger(__name__)

class DataProcessor:

    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir

        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir, exist_ok=True)
        
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Starting Data Processing step")

            # Drop identifiers and CSV index columns if present
            cols_to_drop = [col for col in ['Unnamed: 0', 'Booking_ID'] if col in df.columns]
            if cols_to_drop:
                logger.info(f"Dropping columns: {cols_to_drop}")
                df = df.drop(columns=cols_to_drop)

            df = df.drop_duplicates()

            cat_cols = self.config["data_processing"]["categorical_columns"]
            num_cols = self.config["data_processing"]["numerical_columns"]

            logger.info("Applying Label Encoding to categorical columns")

            label_encoder = LabelEncoder()
            mappings = {}

            for col in cat_cols:
                if col in df.columns:
                    df[col] = label_encoder.fit_transform(df[col])
                    mappings[col] = {
                        label: code
                        for label, code in zip(
                            label_encoder.classes_,
                            label_encoder.transform(label_encoder.classes_)
                        )
                    }

            logger.info(f"Label Mappings applied: {mappings}")

            logger.info("Handling numerical skewness")

            skew_threshold = self.config["data_processing"]["skewness_threshold"]
            existing_num_cols = [col for col in num_cols if col in df.columns]
            skewness = df[existing_num_cols].apply(lambda x: x.skew())

            for column in skewness[skewness > skew_threshold].index:
                logger.info(f"Applying log1p transform on skewed column: {column}")
                df[column] = np.log1p(df[column])

            return df
        
        except Exception as e:
            logger.error(f"Error during preprocess step: {e}")
            raise CustomException("Error while preprocessing data", sys)
        
    def balance_data(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Handling Imbalanced Data using SMOTE")
            X = df.drop(columns='booking_status')
            y = df["booking_status"]

            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            balanced_df = pd.DataFrame(X_resampled, columns=X.columns)
            balanced_df["booking_status"] = y_resampled

            logger.info("Data balanced successfully with SMOTE")
            return balanced_df
        
        except Exception as e:
            logger.error(f"Error during balancing data step: {e}")
            raise CustomException("Error while balancing data", sys)
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Starting Feature selection step via RandomForest")

            X = df.drop(columns='booking_status')
            y = df["booking_status"]

            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)

            feature_importance = model.feature_importances_

            feature_importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': feature_importance
            })
            top_features_importance_df = feature_importance_df.sort_values(by="importance", ascending=False)

            num_features_to_select = self.config["data_processing"]["no_of_features"]
            top_features = top_features_importance_df["feature"].head(num_features_to_select).values

            logger.info(f"Top {num_features_to_select} features selected: {top_features.tolist()}")

            selected_df = df[top_features.tolist() + ["booking_status"]]

            logger.info("Feature selection completed successfully")
            return selected_df
        
        except Exception as e:
            logger.error(f"Error during feature selection step: {e}")
            raise CustomException("Error while selecting features", sys)
    
    def save_data(self, df: pd.DataFrame, file_path: str):
        try:
            logger.info(f"Saving data to {file_path}")
            df.to_csv(file_path, index=False)
            logger.info(f"Data saved successfully to {file_path}")

        except Exception as e:
            logger.error(f"Error during saving data: {e}")
            raise CustomException("Error while saving processed data", sys)

    def process(self):
        try:
            logger.info("Loading data from RAW directory")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            # 1. Enforce raw data schema contracts on input sets
            logger.info("Enforcing input schema contract validation on train and test datasets")
            train_df = validate_raw_data(train_df)
            test_df = validate_raw_data(test_df)

            # 2. Preprocess features
            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            # 3. Handle class imbalance
            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)

            # 4. Feature selection
            train_df = self.select_features(train_df)
            test_df = test_df[train_df.columns]

            # 5. Enforce processed data schema contracts
            logger.info("Enforcing processed schema contract validation on transformed datasets")
            train_df = validate_processed_data(train_df)
            test_df = validate_processed_data(test_df)

            # 6. Persist processed datasets
            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)

            logger.info("Data processing and validation pipeline completed successfully")
        except Exception as e:
            logger.error(f"Error in data preprocessing pipeline: {e}")
            raise CustomException("Error during data preprocessing pipeline", sys)
              
if __name__ == "__main__":
    processor = DataProcessor(TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH)
    processor.process()
