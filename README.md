# Hypoglycemia Prediction with PyTorch

Predicting hypoglycemia events from glucose time-series data. Supports multiple backends (ROCm, CUDA, MPS, CPU) and includes baseline models (Logistic Regression, XGBoost, LightGBM, MLP) and deep learning models (TCN, LSTM, Transformer).

## Features

- **Multi-backend support**: Automatic detection of ROCm, CUDA, MPS, or CPU
- **Baseline models**: Logistic Regression, XGBoost, LightGBM, Simple MLP
- **Deep learning models**: TCN, LSTM/GRU, Transformer
- **Comprehensive feature engineering**: Rolling statistics, time features, delta features, cumulative features
- **Multitask learning**: Optional glucose regression alongside classification
- **Event-aware metrics**: PR-AUC, ROC-AUC, operational metrics (FN per 10k hours, false alarms per day)
- **Training infrastructure**: Early stopping, checkpointing, learning rate scheduling

## Setup

### Prerequisites

- Python 3.9+
- pipenv

### Installation

1. Install dependencies using pipenv:

```bash
pipenv install
```

2. Activate the virtual environment:

```bash
pipenv shell
```

## Project Structure

```
aimd/
├── src/
│   ├── data/
│   │   ├── dataset.py          # PyTorch Dataset classes
│   │   ├── preprocessing.py   # Data loading and preprocessing
│   │   └── features.py         # Feature engineering
│   ├── models/
│   │   ├── baselines.py        # Baseline models
│   │   ├── tcn.py              # TCN model
│   │   ├── rnn.py              # LSTM/GRU models
│   │   └── transformer.py      # Transformer model
│   ├── training/
│   │   ├── trainer.py          # Main training loop
│   │   ├── metrics.py          # Evaluation metrics
│   │   ├── losses.py           # Loss functions
│   │   └── callbacks.py        # Training callbacks
│   └── utils/
│       └── device.py           # Device detection
├── configs/
│   ├── baseline_config.yaml
│   ├── tcn_config.yaml
│   ├── lstm_config.yaml
│   └── transformer_config.yaml
├── scripts/
│   ├── train_baselines.py
│   ├── train_deep.py
│   ├── evaluate.py
│   └── compare_models.py
├── data/
│   └── glucose_timeseries_5000_24h.csv
└── Pipfile
```

## Usage

### Training Baseline Models

Train all baseline models (Logistic Regression, XGBoost, LightGBM, MLP):

```bash
python scripts/train_baselines.py --config configs/baseline_config.yaml
```

### Training Deep Learning Models

Train a specific deep learning model:

```bash
# TCN
python scripts/train_deep.py --model tcn --config configs/tcn_config.yaml

# LSTM
python scripts/train_deep.py --model lstm --config configs/lstm_config.yaml

# Transformer
python scripts/train_deep.py --model transformer --config configs/transformer_config.yaml
```

### Configuration

Edit the YAML configuration files in `configs/` to customize:

- Data paths and window sizes
- Model hyperparameters
- Training parameters (batch size, learning rate, etc.)
- Feature engineering options
- Device selection (or leave as null for auto-detection)

### Device Selection

The pipeline automatically detects the best available backend:

1. ROCm (if available)
2. CUDA (if available)
3. MPS (Apple Silicon, if available)
4. CPU (fallback)

To override, set `device.override` in the config file to `"cuda"`, `"mps"`, or `"cpu"`.

## Data Format

The CSV file should contain the following columns:

- `id`: Patient ID
- `timestamp`: Timestamp
- `hour`: Hour of day (0-23)
- `glucose_mg_dL`: Glucose level
- `activity_event`: Activity/meal event (categorical)
- `carbs_g`: Carbohydrates consumed
- `insulin_units`: Insulin units
- `exercise_level`: Exercise level
- `stress_level`: Stress level
- `sleep_flag`: Sleep indicator (0/1)
- `hypo_next_3_hours`: Target variable (0/1)
- Other metadata columns

## Evaluation Metrics

The pipeline computes:

- **PR-AUC**: Precision-Recall AUC (primary metric for imbalanced data)
- **ROC-AUC**: Receiver Operating Characteristic AUC
- **F1 Score**: At optimal threshold
- **Brier Score**: Calibration metric
- **Operational Metrics**:
  - False negatives per 10k hours
  - False alarms per day

## Model Checkpoints

Trained models are saved in `checkpoints/{model_name}/`:

- `checkpoint_best.pth`: Best model based on validation PR-AUC
- `checkpoint_last.pth`: Last epoch model

## Notes

- The pipeline uses person-wise train/val/test splits to prevent data leakage
- Missing values are forward-filled with missing indicators
- Features are normalized (globally or per-person)
- Class imbalance is handled via class weights or focal loss
- Gradient clipping is applied to prevent exploding gradients


