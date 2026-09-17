import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads CSV data into a pandas DataFrame.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    if not os.path.exists(file_path):
        directory = os.path.dirname(file_path)
        bundled_dataset = os.path.join(
            directory, "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        )
        if os.path.exists(bundled_dataset):
            file_path = bundled_dataset
        else:
            raise FileNotFoundError(f"File not found: {file_path}")
    
    return pd.read_csv(file_path)