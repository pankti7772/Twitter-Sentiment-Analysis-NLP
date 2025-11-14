# src/components/data_ingestion.py

import os
import sys
import logging
from dataclasses import dataclass
from typing import Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

# Optional project-specific imports (if you have these modules)
try:
    from src.exception import CustomException
except Exception:  # pragma: no cover
    class CustomException(Exception):
        def __init__(self, original_exception, sys_info=None):
            super().__init__(str(original_exception))
            self.original_exception = original_exception
            self.sys_info = sys_info

try:
    from src.logger import logging as project_logging
    logging = project_logging  # use project logger if available
except Exception:  # pragma: no cover
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


@dataclass
class DataIngestionConfig:
    raw_data_path: str = os.path.join("artifacts", "data_raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:

    def __init__(self, input_data_path: str = "/Users/pankti/Desktop/sentiment_analysis/notebook/data/twitter_training_cleaned.csv"):
        """
        Uses your actual dataset by default.
        """
        self.ingestion_config = DataIngestionConfig()
        self.input_data_path = input_data_path

    def initiate_data_ingestion(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[str, str]:

        logging.info("Started data ingestion")

        try:
            if not os.path.exists(self.input_data_path):
                raise FileNotFoundError(f"Input data file not found: {self.input_data_path}")

            # Load dataset
            df = pd.read_csv(self.input_data_path)
            logging.info("Loaded data from %s (rows=%d, cols=%d)", self.input_data_path, df.shape[0], df.shape[1])

            # Ensure artifacts folder exists
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info("Saved raw dataset to %s", self.ingestion_config.raw_data_path)

            # Stratified split if 'label' exists
            stratify_col = df['label'] if 'label' in df.columns and df['label'].nunique() > 1 else None

            train_set, test_set = train_test_split(
                df, test_size=test_size, random_state=random_state, stratify=stratify_col
            )

            # Save output splits
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False)

            logging.info("Train and test data saved in artifacts folder")
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.exception("Error in data ingestion")
            raise CustomException(e, sys)


# Run when called with: python -m src.components.data_ingestion
if __name__ == "__main__":
    di = DataIngestion(
        input_data_path="/Users/pankti/Desktop/sentiment_analysis/notebook/data/twitter_training_cleaned.csv"
    )
    train_path, test_path = di.initiate_data_ingestion()
    print("Train:", train_path)
    print("Test:", test_path)
