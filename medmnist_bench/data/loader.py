import os
from typing import Tuple
import numpy as np
import pandas as pd

def load_csv(dataset_name: str, directory_path: str, target_column: int = -1) -> Tuple[np.ndarray, np.ndarray]:
    """Load a CSV file (features + label in last column by default)."""
    file_name = dataset_name if dataset_name.endswith(".csv") else f"{dataset_name}.csv"
    file_path = os.path.join(directory_path, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV not found: {file_path}")
    df = pd.read_csv(file_path)
    X = df.iloc[:, :target_column].values
    y = df.iloc[:, target_column].values
    return X, y
