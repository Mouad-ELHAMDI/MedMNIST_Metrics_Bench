# MedMNIST-Bench (DR + Nested-CV)

A clean, minimal, and reproducible benchmark for evaluating dimensionality reduction (t-SNE/UMAP)
coupled with classic classifiers (SVM/kNN/RF/XGBoost) on MedMNIST datasets.

## Quickstart

```bash
git clone https://github.com/your-org/medmnist-bench.git
cd medmnist-bench
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
# optional:
pip install -r requirements-optional.txt

# Run end-to-end with defaults
medbench run-all --config configs/default.yaml
```

## Repository layout
```text
medmnist-bench/
├── configs/                # YAML configs (datasets, metrics, params, paths)
│   └── default.yaml
├── scripts/                # Utility scripts (data prep, examples)
│   ├── prepare_medmnist.py
│   └── run_all.sh
├── src/medmnist_bench/     # Python package
│   ├── cli.py              # CLI entrypoints (embed, classify, plot, run-all)
│   ├── data/loader.py      # CSV loading / dataset helpers
│   ├── dr/tsne_umap.py     # DR methods
│   ├── eval/cv.py          # Nested-CV evaluation + aggregation
│   ├── models/classifiers.py
│   ├── viz/plotting.py
│   └── utils/{seed.py,io.py,log.py}
├── requirements.txt
├── requirements-optional.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

## Design goals
- **Minimal deps**: numpy, pandas, scikit-learn, umap-learn, openTSNE, matplotlib, PyYAML. (XGBoost optional.)
- **Reproducible**: global seed control + config versioning.
- **Modular**: clean separation of data loading, DR, models, evaluation, and plotting.
- **CLI-first**: single command to run the full pipeline.
- **Publication-ready plots** and CSV/JSON outputs.

## Example commands
```bash
# Just embeddings
medbench embed --config configs/default.yaml --out_dir runs/exp1

# Classify on existing embeddings (embeddings.npz inside out_dir)
medbench classify --config configs/default.yaml --out_dir runs/exp1

# Plots
medbench plot --config configs/default.yaml --out_dir runs/exp1

# End-to-end
medbench run-all --config configs/default.yaml --out_dir runs/exp1
```

## Reproducibility
- We fix seeds for numpy/scikit-learn/UMAP/openTSNE as much as their APIs allow.
- KFold uses deterministic `random_state`.
- We save all configs alongside outputs for provenance.
