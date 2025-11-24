# CancerTRANSFORM

Clinical Attention-Based Cancer Prediction Model

This workspace is a fork/adaptation of the Attention Transformer repository and re-focused for clinical cancer prediction. It demonstrates a lightweight clinical attention model that integrates 7 clinical data types and provides an explainability-ready training/inference example.

**Highlights**
- Uses an attention-style module over 7 clinical features.
- Includes a runnable `run.py` demonstrating training and inference with placeholder data.
- Provides setup scripts and VS Code recommendations for fast iteration.

**Model in brief**
- Input: 7 clinical features per patient (columns `f0` .. `f6`).
- Architecture: per-feature linear embedding -> single-head self-attention across the 7 features -> pooled feature vector -> MLP classifier (binary output).
- Explainability: the model returns per-feature attention weights which can be interpreted as feature importance for each prediction.

Why 7 features?
- This workspace example treats the clinical profile as a sequence of 7 numeric features (for example: age, tumor_size, stage_score, biomarker_A, biomarker_B, symptom_score, comorbidity_count). The example code (in `data/sample.csv`) and `data/loader.py` expect exactly 7 columns named `f0`..`f6` to keep the demo minimal and consistent. You can map your real clinical fields to these columns or modify the model to accept a different feature count.

Model files and key functions
- `models/clinical_attention.py`: `ClinicalAttentionModel` implementation.
- `data/loader.py`: `load_dataset(path)` returns `(X, y)` tensors; `create_dataloaders(X, y, batch_size)` returns PyTorch `DataLoader`.
- `run.py`: training and evaluation runner. Training saves checkpoints to `checkpoints/model_checkpoint.pt`.
- `web_scraper.py`: helper functions to find and download CSVs from a webpage (use responsibly).

How to run (recommended)

1. Create a virtual environment and install dependencies (macOS / zsh):

```bash
chmod +x scripts/setup_venv.sh
./scripts/setup_venv.sh
source .venv/bin/activate
```

2. Quick training test (uses synthetic data by default):

```bash
python run.py --mode train --epochs 10
```

3. Run evaluation / explainability (uses `data/sample.csv` as an example):

```bash
PYTHONPATH=. python notebooks/explainability.py --data data/sample.csv
```

Notes on datasets
- Expected CSV format (when passing `--data path/to.csv`): header with columns `f0,f1,...,f6,label` where `label` should be 0 or 1.
- If your dataset uses different column names, either rename them or update `data/loader.py` to map your columns to the expected features.

Reproducibility
- The example sets a fixed seed (`SEED = 42`) in `run.py` for deterministic behavior where possible.

Next improvements you might want
- Replace the synthetic demo loader with your clinical dataset loader (including missing-value handling and categorical encodings).
- Add unit tests and dataset validation scripts.
- Add experiment logging (e.g., `wandb`, `tensorboard`) and checkpoint versioning.

**Quick Start**

1. Create and activate a Python virtual environment:

```bash
./scripts/setup_venv.sh
source .venv/bin/activate
```

2. Run training with synthetic/demo data:

```bash
python run.py --mode train --epochs 10
```

3. Run inference with synthetic/demo data:

```bash
python run.py --mode eval
```

**Files**
- `run.py`: Example training/inference script with a small PyTorch attention model.
- `requirements.txt`: Python dependencies.
- `scripts/setup_venv.sh`: Creates `.venv` and installs requirements.
- `.vscode/extensions.json`: Recommended extensions for this workspace.
 - `data/sample.csv`: Small example CSV with 7 feature columns and `label` for quick testing.
 - `notebooks/explainability.py`: Small script that runs inference and plots attention weights per feature.

**Model & Explainability**
The `ClinicalAttentionModel` in `run.py` treats the 7 clinical features as a sequence and applies a simple self-attention mechanism to weight and aggregate feature contributions. This design makes it straightforward to extract attention weights per feature for model explainability (feature importance per prediction).

**Reproducibility**
- Set a fixed random seed in `run.py` for reproducible experiments.
- Replace placeholder/synthetic loader with your clinical dataset: expect a CSV with 7 feature columns and one `label` column.

**Citation / Source**
This workspace draws structure and inspiration from: `https://github.com/kalyanipande8/Attention-Transformer-For-Global-HIV-Detection`.

**Next steps**
- Replace synthetic loader with your clinical dataset loader.
- Add unit tests and evaluation notebooks.
- Add model checkpointing and experiment logging (e.g., `wandb` or `tensorboard`).

Quick try (macOS / zsh):

```bash
./scripts/setup_venv.sh
source .venv/bin/activate
python run.py --mode train --epochs 2
python notebooks/explainability.py --data data/sample.csv
```

If you want, I can now wire in your real dataset loader, add a Jupyter notebook for explainability visualizations, or prepare a `pyproject.toml`/`setup.cfg` for packaging.

**Datasets & Credits**
- `data/Coimbra_breast_cancer_dataset.csv`: Coimbra Breast Cancer Dataset (clinical measurements). This repository includes a local copy/mirror for convenience. Original dataset mirrors are available on UCI Machine Learning Repository and Kaggle — please consult the original dataset page for licensing and citation details when publishing results.
- `data/dataR2.csv`: a variant/mirror of the Coimbra dataset included here for experiments.

Please credit the original dataset providers when using these data for publications or public releases. If you plan to publish results derived from these datasets, cite the dataset's original source (see the dataset page on UCI or Kaggle for the correct citation and licensing information).