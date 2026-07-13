#!/bin/bash
#SBATCH --partition=a100
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --job-name=cancer-nn-best
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

mkdir -p logs pretrained_models

export WANDB_API_KEY=<YOUR_WANDB_API_KEY>

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cancer-nn

python main.py \
    -csv_path data/data.csv \
    -config cnn_lstm_multi_loss/1.yml \
    -config_folder configs \
    -wandb_config wandb_config.yml \
    -saved_model_path pretrained_models \
    -max_epochs 200 \
    -seed 1
