import os
import sys
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
import lightgbm as lgb
import mlflow
import mlflow.sklearn

from src.logger import get_logger
from src.custom_exception import CustomException
from src.data_preprocessing import build_pipeline
from config.paths_config import *
from config.model_params import LIGHTGBM_PIPELINE_PARAMS, RANDOM_SEARCH_PARAMS
from utils.common_functions import read_yaml, load_data

logger = get_logger(__name__)


class ModelTraining:
    def __init__(
        self,
        train_path=PROCESSED_TRAIN_DATA_PATH,
        test_path=PROCESSED_TEST_DATA_PATH,
        model_output_path=MODEL_OUTPUT_PATH,
        config_path=CONFIG_PATH,
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.config = read_yaml(config_path)
        self.target_col = self.config["data_processing"].get("target_column", "booking_status")
        self.cat_cols = [
            col
            for col in self.config["data_processing"]["categorical_columns"]
            if col != self.target_col
        ]
        self.num_cols = self.config["data_processing"]["numerical_columns"]

        self.param_distributions = LIGHTGBM_PIPELINE_PARAMS
        self.random_search_config = RANDOM_SEARCH_PARAMS

        # Decouple MLflow Tracking URI via environment variable with SQLite fallback
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        self.experiment_name = os.getenv(
            "MLFLOW_EXPERIMENT_NAME", "Hotel_Reservation_Cancellation"
        )

        logger.info(f"Initialized ModelTraining with MLflow Tracking URI: {self.tracking_uri}")

    def load_and_prepare_data(self):
        """
        Loads cleaned train and test partitions into feature matrices and target vectors.
        """
        try:
            logger.info(f"Loading training data from {self.train_path}")
            train_df = load_data(self.train_path)

            logger.info(f"Loading test data from {self.test_path}")
            test_df = load_data(self.test_path)

            X_train = train_df.drop(columns=[self.target_col])
            y_train = train_df[self.target_col]

            X_test = test_df.drop(columns=[self.target_col])
            y_test = test_df[self.target_col]

            logger.info(
                f"Data successfully prepared. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}"
            )
            return X_train, y_train, X_test, y_test

        except Exception as e:
            logger.error(f"Error while loading and preparing dataset: {e}")
            raise CustomException("Failed to prepare data for model training", sys)

    def train_pipeline(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Builds the unified pipeline and executes RandomizedSearchCV across cross-validation folds.
        """
        try:
            logger.info("Building unified Scikit-Learn pipeline (Preprocessor + LightGBM)")
            base_pipeline = build_pipeline(self.cat_cols, self.num_cols)

            logger.info("Initializing RandomizedSearchCV for hyperparameter tuning")
            random_search = RandomizedSearchCV(
                estimator=base_pipeline,
                param_distributions=self.param_distributions,
                n_iter=self.random_search_config.get("n_iter", 3),
                cv=self.random_search_config.get("cv", 3),
                n_jobs=self.random_search_config.get("n_jobs", -1),
                verbose=self.random_search_config.get("verbose", 2),
                random_state=self.random_search_config.get("random_state", 42),
                scoring=self.random_search_config.get("scoring", "f1"),
            )

            logger.info("Fitting hyperparameter search on training dataset")
            random_search.fit(X_train, y_train)

            best_pipeline = random_search.best_estimator_
            best_params = random_search.best_params_

            logger.info(f"Hyperparameter tuning completed. Best Parameters: {best_params}")
            return best_pipeline, best_params

        except Exception as e:
            logger.error(f"Error during pipeline training and tuning: {e}")
            raise CustomException("Failed to train unified pipeline", sys)

    def evaluate_pipeline(self, pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluates the fitted unified pipeline on held-out test data across comprehensive classification metrics.
        """
        try:
            logger.info("Evaluating unified pipeline on test set")

            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_prob)
            pr_auc = average_precision_score(y_test, y_prob)

            metrics = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "roc_auc": float(roc_auc),
                "pr_auc": float(pr_auc),
            }

            logger.info(
                f"Evaluation Metrics -> Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | "
                f"Recall: {recall:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}"
            )
            return metrics

        except Exception as e:
            logger.error(f"Error during pipeline evaluation: {e}")
            raise CustomException("Failed to evaluate pipeline", sys)

    def save_pipeline(self, pipeline):
        """
        Serializes the unified pipeline artifact into disk for direct serving deployment.
        """
        try:
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            logger.info(f"Saving fitted pipeline artifact to {self.model_output_path}")
            joblib.dump(pipeline, self.model_output_path)
            logger.info("Pipeline artifact successfully saved")

        except Exception as e:
            logger.error(f"Error while saving pipeline artifact: {e}")
            raise CustomException("Failed to save pipeline artifact", sys)

    def run(self):
        """
        Executes end-to-end model training, evaluation, MLflow tracking, and model persistence.
        """
        try:
            logger.info(f"Connecting to MLflow Tracking Server at: {self.tracking_uri}")
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)

            with mlflow.start_run() as run:
                run_id = run.info.run_id
                logger.info(f"Initiated MLflow Run ID: {run_id}")

                # 1. Log dataset artifacts to MLflow
                if os.path.exists(self.train_path):
                    mlflow.log_artifact(self.train_path, artifact_path="datasets")
                if os.path.exists(self.test_path):
                    mlflow.log_artifact(self.test_path, artifact_path="datasets")

                # 2. Load dataset partitions
                X_train, y_train, X_test, y_test = self.load_and_prepare_data()

                # 3. Train unified pipeline with hyperparameter search
                best_pipeline, best_params = self.train_pipeline(X_train, y_train)

                # 4. Evaluate pipeline on held-out test data
                metrics = self.evaluate_pipeline(best_pipeline, X_test, y_test)

                # 5. Persist the pipeline artifact locally
                self.save_pipeline(best_pipeline)

                # 6. Log parameters, metrics, and pipeline artifact to MLflow
                logger.info("Logging parameters, metrics, and unified pipeline to MLflow")
                mlflow.log_params(best_params)
                mlflow.log_metrics(metrics)
                mlflow.log_artifact(self.model_output_path, artifact_path="model_artifact")
                mlflow.sklearn.log_model(best_pipeline, artifact_path="unified_pipeline_model")

                logger.info(f"Model Training and MLflow tracking run ({run_id}) successfully completed")
                return best_pipeline, metrics

        except Exception as e:
            logger.error(f"Error in Model Training workflow: {e}")
            raise CustomException("Model training workflow failed", sys)


if __name__ == "__main__":
    trainer = ModelTraining()
    trainer.run()