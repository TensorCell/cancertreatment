# cancertreatment/neural_networks_models

- **TiDE** (Time-series Dense Encoder) model for predicting tumour cell survival outcomes of radiotherapy protocols on the EMT6/Ro cell line
- **NHiTS** (Neural Hierarchical Interpolation for Time Series Forecasting) model for the EMT6/Ro radiotherapy protocol dataset

## Dataset

- **200,000** simulation protocols from the EMT6/Ro radiotherapy model
- Each protocol: up to 20 radiation doses (zero-padded), followed by 10 days observation
- **Target**: average normalised tumour cell count at day 10
- See [`data/README.md`](data/README.md) for full schema description

## Project structure

```
cancertreatment/neural_networks_models/
├── data/
│   ├── data.csv               # Raw dataset (200,000 protocols × 21 rows each)
│   └── README.md              # Data schema and preprocessing notes
├── model_name/
│   ├── config.py              # CancerTiDEConfig dataclass
│   ├── datasets/
│   │   └── cancer_dataset.py  # Data loading, scaling, PyTorch Dataset + DataLoaders
│   ├── models/
│   │   └── model_name.py            # For example tide.py for TiDE model (ResidualBlock, Encoder, Decoder, TemporalDecoder)
│   ├── trainer/
│   │   └── trainer.py         # Trainer + EarlyStopping
│   ├── metrics/
│   │   └── forecasting_metrics.py  # MAE, RMSE, MSE, MAPE, sMAPE, R², MASE
│   ├── evaluation/
│   │   └── evaluate.py        # Test-set evaluation, plots, CSV outputs
│   └── inference/
│       └── predict.py         # load_model, predict, predict_tumour_cells
├── train.py                   # CLI entry-point
├── pyproject.toml
└── README.md
```

## Setup

```bash
# Install uv if not already installed
pip install uv

# Install dependencies
uv sync
```

## Training

```bash
# Full training run (100 epochs, early stopping patience 15)
uv run python train.py --data_path data/data.csv

# Quick smoke-test
uv run python train.py --data_path data/data.csv --max_epochs 2 --batch_size 512

# Larger model
uv run python train.py --data_path data/data.csv --hidden_size 512 --num_encoder_layers 4

# MAE loss instead of MSE
uv run python train.py --data_path data/data.csv --loss mae
```

## Outputs

After training, the following files are written to `outputs/tide/`:

| File | Description |
|---|---|
| `predictions.csv` | Predicted and actual tumour cell count for every test protocol |
| `metrics.csv` | MAE, RMSE, MSE, MAPE, sMAPE, R², MASE |
| `forecast_vs_actual.png` | Scatter plot: predicted vs. actual |
| `residuals.png` | Residual distribution (sorted + histogram) |
| `training_history.png` | Train/val loss curves |

Checkpoints are saved to `checkpoints/model_name/best_model.pt`.