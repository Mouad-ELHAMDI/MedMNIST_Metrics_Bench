import os, json
from typing import Any, Dict
import numpy as np
import pandas as pd

def ensure_dir(path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def save_json(obj: Dict[str, Any], path: str) -> None:
    p = ensure_dir(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def save_npz(path: str, **arrays) -> None:
    p = ensure_dir(path)
    np.savez_compressed(str(p), **arrays)

def save_csv(df: pd.DataFrame, path: str) -> None:
    p = ensure_dir(path)
    df.to_csv(str(p), index=False)
