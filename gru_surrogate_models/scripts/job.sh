#!/bin/bash
#SBATCH --partition=a100
#SBATCH --mem=80G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --job-name=cancer-nn-best
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
# Slurm job: train one config on the cluster through train_configs.sh (manifest logs/slurm_manifest.csv).
# Usage: sbatch scripts/job.sh   (fill in WANDB_API_KEY below first)
# Env: CONFIG (cnn_lstm_multi_loss/1.yml), SEEDS ("1"), MAX_EPOCHS (200), CSV_PATH (data/data_400k.csv)

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}" || exit 1
export WANDB_API_KEY="<YOUR_WANDB_API_KEY>"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cancer-nn

scripts/train_configs.sh slurm "${CONFIG:-cnn_lstm_multi_loss/1.yml}"
