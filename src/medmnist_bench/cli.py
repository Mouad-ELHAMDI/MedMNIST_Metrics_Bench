import argparse, os, yaml, numpy as np, pandas as pd
from .utils.seed import set_global_seed
from .utils.io import ensure_dir, save_npz, save_csv, save_json
from .utils.log import get_logger
from .data.loader import load_csv
from .dr.tsne_umap import generate_embeddings
from .eval.cv import apply_classifiers, extract_max_accuracies, flatten_results
from .viz.plotting import visualize_embeddings

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def cmd_embed(cfg, out_dir, logger):
    logger.info("Generating embeddings...")
    embeddings = generate_embeddings(
        datasets=cfg["datasets"],
        metrics=cfg["metrics"],
        tsne_perplexity=cfg["tsne"]["perplexity"],
        umap_n_neighbors=cfg["umap"]["n_neighbors"],
        data_loader_fn=load_csv,
        data_dir=cfg["data_dir"],
        seed=cfg["seed"],
        umap_n_components=cfg["umap"]["n_components"],
        umap_init=cfg["umap"]["init"],
        tsne_metric_override=cfg["tsne"]["metric_override"],
        umap_metric_override=cfg["umap"]["metric_override"],
    )
    # Save compact npz (one file)
    save_npz(os.path.join(out_dir, "embeddings.npz"), data=np.array([embeddings], dtype=object))
    save_json(embeddings_to_manifest(embeddings), os.path.join(out_dir, "embeddings_manifest.json"))
    visualize_embeddings(embeddings, out_dir)
    return embeddings

def embeddings_to_manifest(embeddings: dict) -> dict:
    man = {}
    for d, m in embeddings.items():
        man[d] = {}
        for metric, data in m.items():
            man[d][metric] = {
                "tsne_params": list(data["tsne"].keys()),
                "umap_params": list(data["umap"].keys()),
                "num_samples": len(data["y"]),
            }
    return man

def cmd_classify(cfg, out_dir, logger, embeddings=None):
    logger.info("Running nested-CV classifiers...")
    if embeddings is None:
        # load from npz
        npz = np.load(os.path.join(out_dir, "embeddings.npz"), allow_pickle=True)
        embeddings = npz["data"].item()
    results = apply_classifiers(
        embeddings, cfg["classifiers"], grids=cfg["grids"],
        outer_splits=cfg["cv"]["outer_splits"], inner_splits=cfg["cv"]["inner_splits"], seed=cfg["seed"]
    )
    rows = flatten_results(results)
    df = pd.DataFrame(rows)
    save_csv(df, os.path.join(out_dir, "all_classifier_results.csv"))
    max_acc = extract_max_accuracies(results)
    df2 = pd.DataFrame([{"classifier": c, "method": m, **v} for c, d in max_acc.items() for m, v in d.items()])
    save_csv(df2, os.path.join(out_dir, "max_accuracies.csv"))
    save_json(max_acc, os.path.join(out_dir, "max_accuracies.json"))
    return results

def cmd_plot(cfg, out_dir, logger):
    logger.info("Plots already generated during embedding. Nothing extra to plot for now.")

def cmd_run_all(cfg, out_dir, logger):
    embeddings = cmd_embed(cfg, out_dir, logger)
    _ = cmd_classify(cfg, out_dir, logger, embeddings=embeddings)

def main():
    p = argparse.ArgumentParser(description="MedMNIST DR + Nested-CV Benchmark")
    p.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    p.add_argument("--out_dir", type=str, default=None, help="Output directory (overrides config.out_dir)")

    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("embed")
    sub.add_parser("classify")
    sub.add_parser("plot")
    sub.add_parser("run-all")

    args = p.parse_args()
    cfg = load_config(args.config)
    if args.out_dir:
        cfg["out_dir"] = args.out_dir
    out_dir = cfg.get("out_dir", "runs/default")
    ensure_dir(out_dir)

    set_global_seed(cfg.get("seed", 42))
    logger = get_logger(log_dir=out_dir)
    # Save resolved config for provenance
    save_json(cfg, os.path.join(out_dir, "resolved_config.json"))

    if args.cmd == "embed":
        cmd_embed(cfg, out_dir, logger)
    elif args.cmd == "classify":
        cmd_classify(cfg, out_dir, logger)
    elif args.cmd == "plot":
        cmd_plot(cfg, out_dir, logger)
    elif args.cmd == "run-all":
        cmd_run_all(cfg, out_dir, logger)
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")
