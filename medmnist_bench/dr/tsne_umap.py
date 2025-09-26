from typing import Dict, Any, List
import numpy as np
import openTSNE
from umap import UMAP
from ..utils.seed import set_global_seed

def apply_tsne(X: np.ndarray, metric: str, perplexity: float, seed: int) -> np.ndarray:
    set_global_seed(seed)
    tsne = openTSNE.TSNE(perplexity=perplexity, metric=metric, random_state=seed)
    return tsne.fit(X)

def apply_umap(X: np.ndarray, metric: str, n_neighbors: int, n_components: int, init: str, seed: int) -> np.ndarray:
    set_global_seed(seed)
    um = UMAP(n_neighbors=n_neighbors, n_components=n_components, metric=metric, init=init, random_state=seed)
    return um.fit_transform(X)

def generate_embeddings(
    datasets: List[str],
    metrics: List[str],
    tsne_perplexity: float,
    umap_n_neighbors: int,
    data_loader_fn,
    data_dir: str,
    seed: int,
    umap_n_components: int = 2,
    umap_init: str = "pca",
    tsne_metric_override: str = None,
    umap_metric_override: str = None,
) -> Dict[str, Dict[str, Any]]:
    """Return nested dict: dataset -> metric -> {'tsne': {p: emb}, 'umap': {n: emb}, 'y': labels}"""
    out: Dict[str, Dict[str, Any]] = {}
    for dataset in datasets:
        X, y = data_loader_fn(dataset, data_dir)
        out[dataset] = {}
        for metric in metrics:
            metric_tsne = tsne_metric_override or metric
            metric_umap = umap_metric_override or metric
            tsne_emb = apply_tsne(X, metric_tsne, perplexity=tsne_perplexity, seed=seed)
            umap_emb = apply_umap(X, metric_umap, n_neighbors=umap_n_neighbors, n_components=umap_n_components, init=umap_init, seed=seed)
            out[dataset][metric] = {
                "tsne": {tsne_perplexity: np.array(tsne_emb)},
                "umap": {umap_n_neighbors: np.array(umap_emb)},
                "y": y,
            }
    return out
