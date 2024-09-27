# data_loader.py
import aeon
from aeon.datasets import load_classification
from tsc_datasets import univariate_equal_length
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_datasets(dataset_list):
    """
    Load train and test datasets for a list of time series classification datasets.

    Parameters:
    dataset_list (list): List of dataset names to load.

    Returns:
    data_dict (dict): Dictionary with dataset names as keys and corresponding train/test data as values.
    """
    data_dict = {}
    for dataset_name in dataset_list:
        try:
            X_train, y_train = load_classification(dataset_name, split="train")
            X_test, y_test = load_classification(dataset_name, split="test")
            data_dict[dataset_name] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test
            }
            logging.info(f"Successfully loaded dataset: {dataset_name}")
        except Exception as e:
            logging.error(f"Failed to load dataset: {dataset_name}. Error: {str(e)}")
    
    return data_dict

if __name__ == "__main__":
    logging.info(f"Loading univariate datasets without missing values: {len(univariate_equal_length)} datasets.")
    data = load_datasets(univariate_equal_length)
    logging.info("Dataset loading completed.")
    # Further steps: Time series Representation, Interval-Selection, Interval Features
