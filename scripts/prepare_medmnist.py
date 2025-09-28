import os, argparse
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/MedMNIST-V2/MedMNIST")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    try:
        import medmnist
        from medmnist import INFO
    except Exception as e:
        raise SystemExit("Install medmnist to use this helper: pip install medmnist") from e

    # Example: export two datasets as CSV (features flattened).
    for key in ["pneumoniamnist", "retinamnist"]:
        info = INFO[key]
        DataClass = getattr(medmnist, info["python_class"])
        ds = DataClass(split="train", download=True)
        X = ds.imgs.reshape(len(ds), -1)
        y = ds.labels.reshape(-1)
        import pandas as pd
        df = pd.DataFrame(np.hstack([X, y[:, None]]))
        df.to_csv(os.path.join(args.out_dir, f"{key}.csv"), index=False)
        print(f"Wrote {key}.csv")

if __name__ == "__main__":
    main()
