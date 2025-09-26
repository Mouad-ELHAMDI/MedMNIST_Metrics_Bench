from typing import Dict, Any, List
import numpy as np
import pandas as pd
from ..models.classifiers import NestedCVClassifier

def apply_classifiers(embeddings_dict: Dict[str, Dict[str, Any]], classifier_list: List[str], grids: Dict[str, Any], outer_splits: int, inner_splits: int, seed: int):
    results = {clf: {} for clf in classifier_list}
    for dataset, metrics_data in embeddings_dict.items():
        for metric, data in metrics_data.items():
            y = data["y"]
            for method in ["tsne", "umap"]:
                for param, emb in data[method].items():
                    if emb is None:
                        continue
                    X = np.array(emb)
                    for clf in classifier_list:
                        cv = NestedCVClassifier(clf, outer_splits, inner_splits, seed=seed)
                        best_params, scores = cv.perform(X, y, grids[clf])
                        results[clf].setdefault(dataset, {}).setdefault(metric, {"tsne": {}, "umap": {}})
                        results[clf][dataset][metric][method][param] = {
                            "best_params": best_params,
                            "accuracy": scores,
                        }
    return results

def extract_max_accuracies(results_dict: Dict[str, Dict[str, Any]]):
    out = {}
    for clf in results_dict:
        out[clf] = {"tsne": {"accuracy": -np.inf}, "umap": {"accuracy": -np.inf}}
        for dataset, metrics in results_dict[clf].items():
            for metric, methods in metrics.items():
                for method in ["tsne", "umap"]:
                    for param, res in methods[method].items():
                        acc = float(np.mean(res["accuracy"])) if isinstance(res["accuracy"], (list, tuple)) else res["accuracy"]
                        if acc > out[clf][method].get("accuracy", -np.inf):
                            out[clf][method] = {
                                "accuracy": acc,
                                "metric": metric,
                                "dataset": dataset,
                                "parameter": param,
                            }
    return out

def flatten_results(results_dict: Dict[str, Dict[str, Any]]):
    rows = []
    for clf, datasets in results_dict.items():
        for dataset, metrics in datasets.items():
            for metric, methods in metrics.items():
                for method in ["tsne", "umap"]:
                    for param, res in methods[method].items():
                        rows.append({
                            "classifier": clf,
                            "dataset": dataset,
                            "metric": metric,
                            "method": method,
                            "parameter": param,
                            "best_params": res["best_params"],
                            "accuracy": res["accuracy"],
                            "accuracy_mean": float(np.mean(res["accuracy"])) if len(res["accuracy"])>0 else float("nan"),
                        })
    return rows
