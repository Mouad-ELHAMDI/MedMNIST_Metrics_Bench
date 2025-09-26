# src/medmnist_bench/utils/io.py
import os
import json
import numpy as np
import pandas as pd
from typing import Any, Dict

def ensure_dir(path: str | None) -> str:
    """
    Ensure a directory exists. If path is None or empty, treat as CWD ('.') and do nothing.
    Returns a normalized directory path.
    """
    if not path:
        return "."
    path = os.path.normpath(path)
    os.makedirs(path, exist_ok=True)
    return path

def ensure_parent(path: str) -> str:
    """
    Ensure the parent directory of a *file path* exists.
    Works for 'file.json' (parent '.') and nested paths like 'runs/exp1/file.json'.
    Returns the normalized file path (unchanged filename).
    """
    parent = os.path.dirname(path) or "."
    ensure_dir(parent)
    return os.path.normpath(path)

def save_json(obj: Dict[str, Any], path: str) -> None:
    path = ensure_parent(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def save_npz(path: str, **arrays) -> None:
    path = ensure_parent(path)
    np.savez_compressed(path, **arrays)

def save_csv(df: pd.DataFrame, path: str) -> None:
    path = ensure_parent(path)
    df.to_csv(path, index=False)
