import os
import yaml
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Configure logging
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file"""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters loaded successfully from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('Parameters file not found at %s', params_path)
        return params
    except yaml.YAMLError as e:
        logger.error('Error parsing YAML file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def load_data(data_path: str) -> pd.DataFrame:
    """Load data from a CSV file"""
    try:
        df = pd.read_csv(data_path)
        logger.debug('Data loaded successfully from %s', data_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Error parsing CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data by handling missing values, duplicates and empty strings"""
    try:
        # Handle missing values
        df.dropna(inplace=True)

        # Handle duplicates
        df.drop_duplicates(inplace=True)

        # Remove rows with empty strings
        df = df[df['clean_comment'].str.strip() != ""]

        logger.debug('Data Processing completed: Missing values, duplicates and empty strings removed')
        return df
    
    except KeyError as e:
        logger.error('Missing column: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets, create a directory if it doesn't exist"""
    try: 
        raw_data_path = os.path.join(data_path, 'raw')

        # Create directory if it doesn't exist
        os.makedirs(raw_data_path, exist_ok=True)

        # Save train and test datasets
        train_df.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_df.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)

        logger.debug('Train and test datasets saved successfully to %s', raw_data_path)

    except Exception as e:
        logger.error('Unexpected error during data saving: %s', e)
        raise


def main():
    try:
        # Load parameters from params.yaml in the root directory
        params = load_params(params_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../params.yaml"))
        test_size = params['data_ingestion']['test_size']

        # Load data from the specified path
        df = load_data(data_path='https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv')

        final_df = preprocess_data(df)

        # Split data into training and testing sets
        train_df, test_df = train_test_split(final_df, test_size=test_size, random_state=42)

        # Save the train and test datasets
        save_data(train_df, test_df, data_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data"))

    except Exception as e:
        logger.error('Unexpected error during data ingestion: %s', e)
        raise


if __name__ == "__main__":
    main()