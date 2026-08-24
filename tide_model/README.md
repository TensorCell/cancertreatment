# cancertreatment/tide_model

TiDE (Time-series Dense Encoder) model for predicting tumour cell survival outcomes of radiotherapy protocols on the EMT6/Ro cell line.

## Dataset

- **200,000** simulation protocols from the EMT6/Ro radiotherapy model
- Each protocol: up to 20 radiation doses (zero-padded), followed by 10 days observation
- **Target**: average normalised tumour cell count at day 10
- See [`data/README.md`](data/README.md) for full schema description

## Project structure

```
cancertreatment/tide_model/
├── data/
│   ├── data.csv               # Raw dataset (200,000 protocols × 21 rows each)
│   └── README.md              # Data schema and preprocessing notes
├── tide/
│   ├── config.py              # CancerTiDEConfig dataclass
│   ├── datasets/
│   │   └── cancer_dataset.py  # Data loading, scaling, PyTorch Dataset + DataLoaders
│   ├── models/
│   │   └── tide.py            # TiDE model (ResidualBlock, Encoder, Decoder, TemporalDecoder)
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

Checkpoints are saved to `checkpoints/tide/best_model.pt`.

## Model architecture

TiDE uses only dense residual MLP blocks (no attention, no recurrence):

```
Dose sequence (20 steps)   ──┐
                              ├──► FeatureProjection ──► Encoder ──► Decoder ──► TemporalDecoder ──► Predicted cells
[time, time_gap] covariates ──┘                                                       ▲
                                                                              (skip: dose → output)
```

Key design choices for this dataset:
- `horizon = 1` — single scalar output per protocol
- `num_future_covariates = 0` — no future inputs after dosing ends
- `past_target` = dose sequence (not the padded target column)
- MinMax scaling fit on training protocols only
- Split by protocol ID (not chronological — protocols are independent)

## Inference (single protocol)

```python
from tide.config import CancerTiDEConfig
from tide.inference.predict import load_model, predict_tumour_cells
from pathlib import Path
import numpy as np

cfg = CancerTiDEConfig(data_path=Path("data/data.csv"))
model = load_model(cfg, device="cpu")

# Example: zero-padded protocol with 8 actual doses
dose_seq   = np.array([0,0,0,0,0,0,0,0,0,0,0,0, 2.0, 1.5, 1.0, 2.5, 0.5, 1.25, 0.75, 0.5])
time_seq   = np.array([0]*12 + [1200, 3600, 7200, 14400, 21600, 36000, 50400, 64800], dtype=float)
gap_seq    = np.zeros(20, dtype=float)  # simplified

# Retrieve scale stats from a fitted dataset
from tide.datasets.cancer_dataset import build_dataloaders
train_loader, _, _ = build_dataloaders(cfg)
stats = train_loader.dataset.scale_stats

prediction = predict_tumour_cells(
    model=model,
    dose_sequence=dose_seq,
    time_sequence=time_seq,
    time_gap_sequence=gap_seq,
    dose_min=stats.dose_min,
    dose_max=stats.dose_max,
    time_min=stats.time_min,
    time_max=stats.time_max,
    time_gap_min=stats.time_gap_min,
    time_gap_max=stats.time_gap_max,
)
print(f"Predicted normalised tumour cells: {prediction:.6f}")
```
