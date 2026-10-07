import os
import sys
import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.custom_exception import CustomException
from src.schemas.data_schema import validate_raw_data
from config.paths_config import *
from utils.common_functions import read_yaml

logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingestion"]
        self.bucket_name = self.config["bucket_name"]
        self.file_name = self.config["bucket_file_name"]
        self.train_test_ratio = self.config["train_ratio"]

        os.makedirs(RAW_DIR, exist_ok=True)

        logger.info(f"Data Ingestion initialized for bucket '{self.bucket_name}' and file '{self.file_name}'")

    def download_csv_from_gcp(self):
        try:
            logger.info(f"Downloading {self.file_name} from GCP bucket {self.bucket_name}")
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            blob = bucket.blob(self.file_name)

            blob.download_to_filename(RAW_FILE_PATH)

            logger.info(f"CSV file successfully downloaded to {RAW_FILE_PATH}")

        except Exception as e:
            logger.error(f"Error while downloading CSV file from GCP: {e}")
            raise CustomException("Failed to download CSV file from GCP", sys)
        
    def split_data(self):
        try:
            logger.info("Starting the validation and splitting process")
            data = pd.read_csv(RAW_FILE_PATH)
            
            # Enforce raw data schema contract validation
            data = validate_raw_data(data)

            train_data, test_data = train_test_split(
                data,
                test_size=1 - self.train_test_ratio,
                random_state=42
            )

            train_data.to_csv(TRAIN_FILE_PATH, index=False)
            test_data.to_csv(TEST_FILE_PATH, index=False)

            logger.info(f"Validated train data saved to {TRAIN_FILE_PATH}")
            logger.info(f"Validated test data saved to {TEST_FILE_PATH}")
        
        except Exception as e:
            logger.error(f"Error while validating or splitting data: {e}")
            raise CustomException("Failed to validate and split data into training and test sets", sys)
        
    def run(self):
        try:
            logger.info("Starting data ingestion workflow")

            self.download_csv_from_gcp()
            self.split_data()

            logger.info("Data ingestion completed successfully")
        
        except Exception as e:
            logger.error(f"Data ingestion pipeline execution error: {e}")
            raise CustomException("Data ingestion execution failed", sys)
        
        finally:
            logger.info("Data ingestion process ended")

if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()
