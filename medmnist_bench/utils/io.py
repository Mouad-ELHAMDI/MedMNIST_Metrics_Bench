# src/medmnist_bench/utils/io.py
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any, Dict

def ensure_dir(path: str | None) -> str:
    """Ensure a directory exists and return its normalized string path."""
    if not path:
        return "."

    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return os.path.normpath(str(directory))

def ensure_parent(path: str) -> str:
    """Ensure the parent directory of a file path exists and return the normalized path."""
    target = Path(path)
    parent = target.parent if target.parent != Path("") else Path(".")
    ensure_dir(str(parent))
    return os.path.normpath(str(target))

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
