# Surrogate models

Neural network surrogate models approximating the EMT6/Ro cancer treatment
simulation. This directory is self-contained: it holds everything needed to
train and evaluate the best-performing configuration.

## Best configuration

`configs/cnn_lstm_multi_loss/1.yml` — a `MultiHeadTaskRegressor` built from three
`CNNAttLSTM` subnetworks (`mode: cnn_lstm`), each configured by
`configs/single/lstm_2_1.yml` (32 hidden units, 3 layers). It is trained with
three L1 heads plus a margin ranking loss (weight 5), which teaches the model to
rank two treatment protocols against each other, not just regress each one.

## Setup

```bash
conda env create -f env_entropy.yml
conda activate cancer-nn
```

## Data

`data/data.csv` is not checked in. Generate it (downloads the raw dataset and
reshapes each protocol into a 21-step series):

```bash
mkdir -p data
python -m cancer_nn.preprocess_data
```

## Weights & Biases

Training logs to W&B. Fill in `configs/wandb_config.yml` with your project name
and entity, and export your key:

```bash
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
```

## Training

Run the best config directly:

```bash
python main.py \
    -csv_path data/data.csv \
    -config cnn_lstm_multi_loss/1.yml \
    -config_folder configs \
    -wandb_config wandb_config.yml \
    -saved_model_path pretrained_models \
    -max_epochs 200 \
    -seed 1
```

Or submit it to Slurm (A100, 24h) with `sbatch job.sh`. After training, the best
checkpoint by validation loss is reloaded and evaluated on the test split, and
the custom ranking/error metrics are logged to W&B.
