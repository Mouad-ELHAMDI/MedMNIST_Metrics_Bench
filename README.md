# MedMNIST Metrics Bench

A clean, minimal, and reproducible benchmark for evaluating dimensionality reduction (t-SNE/UMAP) coupled with classic classifiers (SVM/kNN/RF/XGBoost) on MedMNIST datasets.

---

## Quick Start

```bash
git clone https://github.com/Mouad-ELHAMDI/MedMNIST_Metrics_Bench.git
cd MedMNIST_Metrics_Bench

# create & activate a virtual environment
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# install deps
pip install -r requirements.txt
pip install -e .   # install the package in development (editable) mode
```

### Prepare Data

First, download and prepare the MedMNIST datasets:

```bash
# install medmnist dependency
pip install medmnist

# prepare/download the datasets (as configured in the default config)
python scripts/prepare_medmnist.py
```

### Run Benchmark

```bash
# run end-to-end with defaults
medbench --config configs/default.yaml run-all

# or specify an explicit output directory
medbench --config configs/default.yaml --out_dir runs/exp1 run-all
```

---

## Project Structure

```text
./
├── configs/                     # YAML configs (datasets, metrics, params, paths)
│   └── default.yaml
├── scripts/                     # Utility scripts (data prep, examples)
│   ├── prepare_medmnist.py
│   └── run_all.sh
├── medmnist_bench/              # Python package
│   ├── cli.py                   # CLI entrypoints (embed, classify, plot, run-all)
│   ├── data/loader.py           # CSV loading / dataset helpers
│   ├── dr/tsne_umap.py          # DR methods
│   ├── eval/cv.py               # Nested-CV evaluation + aggregation
│   ├── models/classifiers.py
│   ├── viz/plotting.py
│   └── utils/{seed.py,io.py,log.py}
├── requirements.txt
├── requirements-optional.txt
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## Features

- *Minimal deps*: numpy, pandas, scikit-learn, umap-learn, openTSNE, matplotlib, PyYAML (XGBoost optional).
- *Reproducible*: global seed control + config versioning.
- *Modular*: clean separation of data loading, DR, models, evaluation, and plotting.
- *CLI-first*: single command to run the full pipeline.
- *Publication-ready* plots and CSV/JSON outputs.

---

## Usage Examples

### Individual Commands

```bash
# just embeddings
medbench --config configs/default.yaml --out_dir runs/exp1 embed

# classify on existing embeddings (expects embeddings.npz inside out_dir)
medbench --config configs/default.yaml --out_dir runs/exp1 classify

# generate plots
medbench --config configs/default.yaml --out_dir runs/exp1 plot

# end-to-end pipeline
medbench --config configs/default.yaml --out_dir runs/exp1 run-all
```

### Custom Configuration

Create a custom config file based on `configs/default.yaml`:

```bash
cp configs/default.yaml configs/my_experiment.yaml
# edit configs/my_experiment.yaml as needed

# run with custom config
medbench --config configs/my_experiment.yaml run-all
```

---

## Reproducibility

- We fix seeds for NumPy/scikit-learn/UMAP/openTSNE where supported by their APIs.
- `KFold` uses a deterministic `random_state`.
- We save all configs alongside outputs for provenance.

---

## Configuration

The `configs/default.yaml` file controls:
- **Datasets**: which MedMNIST datasets to process
- **DR methods**: t-SNE and UMAP parameters
- **Classifiers**: SVM, kNN, Random Forest, XGBoost hyperparameter grids
- **Evaluation**: cross-validation settings
- **Metrics**: distance/accuracy metrics to evaluate

---

## Output

Results are saved in the specified output directory:
- `embeddings.npz`: generated embeddings from DR methods
- `results.csv`: classification performance metrics
- `plots/`: visualization outputs
- `config.yaml`: copy of configuration used for the experiment
