import os, json
from typing import Any, Dict
import numpy as np
import pandas as pd

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def save_npz(path: str, **arrays) -> None:
    ensure_dir(os.path.dirname(path))
    np.savez_compressed(path, **arrays)

def save_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)
