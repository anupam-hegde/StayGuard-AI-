import os
import sys
from src.logger import get_logger
from src.custom_exception import CustomException
from src.data_ingestion import DataIngestion
from src.data_preprocessing import DataProcessor
from src.model_training import ModelTraining
from utils.common_functions import read_yaml
from config.paths_config import *

logger = get_logger(__name__)


class TrainingPipeline:
    def __init__(self, config_path=CONFIG_PATH):
        self.config_path = config_path
        self.config = read_yaml(config_path)

    def run_pipeline(self):
        try:
            logger.info("==================================================")
            logger.info("   STARTING END-TO-END MLOPS TRAINING PIPELINE   ")
            logger.info("==================================================")

            # Stage 1: Data Ingestion (Checks/Downloads raw data & partitions)
            logger.info(">>> Stage 1: Data Ingestion Initialized <<<")
            data_ingestion = DataIngestion(self.config)
            if not os.path.exists(RAW_FILE_PATH):
                data_ingestion.download_csv_from_gcp()
            data_ingestion.split_data()
            logger.info(">>> Stage 1: Data Ingestion Completed Successfully <<<")

            # Stage 2: Data Preprocessing & Validation
            logger.info(">>> Stage 2: Data Preprocessing & Validation Initialized <<<")
            processor = DataProcessor(
                TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, self.config_path
            )
            processor.process()
            logger.info(">>> Stage 2: Data Preprocessing Completed Successfully <<<")

            # Stage 3: Unified Pipeline Training, Tuning & MLflow Tracking
            logger.info(
                ">>> Stage 3: Model Training & Unified Pipeline Serialization Initialized <<<"
            )
            trainer = ModelTraining(
                train_path=PROCESSED_TRAIN_DATA_PATH,
                test_path=PROCESSED_TEST_DATA_PATH,
                model_output_path=MODEL_OUTPUT_PATH,
                config_path=self.config_path,
            )
            pipeline, metrics = trainer.run()
            logger.info(">>> Stage 3: Model Training Completed Successfully <<<")

            logger.info("==================================================")
            logger.info("   TRAINING PIPELINE EXECUTED SUCCESSFULLY!      ")
            logger.info(f"   Final Metrics: {metrics}                       ")
            logger.info("==================================================")
            return pipeline, metrics

        except Exception as e:
            logger.error(f"Training Pipeline failed: {e}")
            raise CustomException("Training pipeline execution encountered an error", sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()