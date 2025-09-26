import os
from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def _discrete_cmap(n):
    base = plt.cm.get_cmap("tab20", n)
    return ListedColormap(base.colors[:n])

def plot_embedding(embedding: np.ndarray, y: np.ndarray, title: str, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 7), dpi=200)
    labels = np.unique(y)
    cmap = _discrete_cmap(len(labels))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap=cmap, s=6, alpha=0.9, linewidths=0)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()

def visualize_embeddings(embeddings_dict: Dict[str, Dict[str, Any]], out_dir: str):
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    for dataset, metrics in embeddings_dict.items():
        for metric, data in metrics.items():
            y = data["y"]
            for p, emb in data["tsne"].items():
                plot_embedding(np.array(emb), y, f"{dataset} - t-SNE ({metric}) p={p}", os.path.join(plots_dir, f"{dataset}_tsne_{metric}_p{p}.png"))
            for n, emb in data["umap"].items():
                plot_embedding(np.array(emb), y, f"{dataset} - UMAP ({metric}) n={n}", os.path.join(plots_dir, f"{dataset}_umap_{metric}_n{n}.png"))
