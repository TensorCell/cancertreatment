# cancertreatment/gru_surrogate_models

Surrogate models for the EMT6/Ro radiotherapy simulator, developed from the upstream
`surogate_models/` (3x CNN-attention-LSTM) into GRU and event-based recurrent models,
and retrained on a doubled 400k-protocol dataset.

- **Best single model**: `42_event_gru_addlog`, 4.29 MAE on the 200k test split, 4.18 MAE on 400k
- **Best average**: five 400k runs, **4.03 MAE / 95.87 % correctly ordered pairs / 0.957 % MAPE**
- **Reference points**: upstream `1.yml` reproduces at 4.52 to 4.59 MAE; the thesis' best model is 4.87

Full results: [`report/main.pdf`](report/main.pdf). What every config changes and why:
[`configs/README.md`](configs/README.md).

## Dataset

- **400 000** protocols: the original uniform 200k plus 200k Sobol-sampled protocols
  simulated with EMT6-Ro (tumour 1, 100 replicates each), shuffled by series id so the
  8:1:1 range split holds both sources equally
- Each protocol: up to 20 doses in the first 5 days, simulation to day 10
- **Target**: mean live tumour cells at day 10
- See [`data/README.md`](data/README.md) for where to get the files

## Project structure

```
cancertreatment/gru_surrogate_models/
├── cancer_nn/
│   ├── used_networks.py       # CNNAttLSTM (upstream), GRURegressor, CNNAttGRU,
│   │                          # EventRecurrentRegressor, ContinuousTimeTransformer, Readout
│   ├── torch_lightning_modules.py  # CancerNet: losses, optimiser/scheduler options, logging
│   ├── datasets.py            # Dataset, DatasetMargin (ranking pairs, hard-pair mining)
│   ├── metrics.py             # evaluate(): MAE, RMSE, MAPE, correct-%, MAE99 over a split
│   └── experiments_helpers.py # split, config loading, seeding
├── configs/
│   ├── cnn_lstm_multi_loss/   # 1.yml (upstream) and 19..45
│   ├── single/                # per-sub-network hyper-parameters
│   └── README.md              # what each config changes, parameter counts, results
├── scripts/                   # batch runners (run from this directory)
├── main.py                    # training entry point
├── eval_ensemble.py           # re-score checkpoints, prediction averages, bootstrap CIs
├── stack_by_regime.py         # regime-gated stacking of ensemble members
├── check_residual_structure.py
├── make_data_subsets.py       # learning-curve subsets by whole series
├── summarize_gru_seed_sweep.py
├── data/README.md
├── report/main.pdf
├── env_rtx_5070.yml, env_entropy.yml, pyproject.toml
└── README.md
```

`data/`, `pretrained_models/`, `wandb/` and `logs/` are created locally and gitignored.

## Setup

```bash
conda env create -f env_rtx_5070.yml     # env_entropy.yml on the cluster
conda activate cancer-nn-5070
export WANDB_API_KEY=<your key>          # or `wandb login` once
```

Fill in `configs/wandb_config.yml` with your W&B project and entity.

## Training

```bash
# One run of the best model on the 400k data
python main.py -config cnn_lstm_multi_loss/42_event_gru_addlog.yml -config_folder configs \
    -wandb_config wandb_config.yml -csv_path data/data_400k.csv -max_epochs 300 -seed 1

# The 400k batch from the report, then its comparison table
scripts/run_400k_baseline.sh
EVAL_ONLY=1 scripts/run_400k_baseline.sh

# Any list of configs and seeds
SEEDS="1 2" scripts/train_configs.sh my_batch 24_gru_xl 42_event_gru_addlog
```

`-csv_path data/data.csv` (or `CSV_PATH=data/data.csv` for the scripts) trains on the
original 200k set. On Slurm: `sbatch scripts/job.sh` from this directory.

| script | purpose |
|---|---|
| `scripts/train_configs.sh <manifest> <config>...` | train each (config, seed); env `SEEDS`, `MAX_EPOCHS`, `CSV_PATH`, `EXTRA_ARGS` |
| `scripts/run_400k_baseline.sh` | 42 x3 seeds + 24_gru_xl x2 on 400k, then the like-for-like table |
| `scripts/run_next_batch.sh` | 42 seeds 2 to 3, ranking-loss runs 39/40, final 200k table |
| `scripts/run_architecture_sweep.sh` | configs 41 to 45 by `CONFIGS=` / `SEEDS=` |
| `scripts/run_learning_curve.sh` | 24_gru_xl on 25 % / 50 % subsets |
| `scripts/run_seed_ensemble.sh` | more seeds of one config + averaged evaluation |
| `scripts/job.sh` | Slurm wrapper around `train_configs.sh` |

All runners share `scripts/_lib.sh`, keep a manifest in `logs/` keyed by (config, seed)
and skip pairs whose checkpoint already exists, so an interrupted batch resumes.

## Outputs

- `pretrained_models/<run_id>.ckpt`: best checkpoint by validation loss
- W&B: `val/*` per epoch and `test/*` after training (MAE, RMSE, MAPE, percent,
  max_error_99), all from `cancer_nn.metrics.evaluate`
- `logs/`: per-run logs, batch manifests, comparison tables, saved predictions (`.npz`)

## Changes from upstream `surogate_models/`

- New sub-networks and output heads in `used_networks.py`; upstream has only `CNNAttLSTM`
- `optimizer` / `weight_decay` / `scheduler` config keys; `-swa_lrs` and `-train_on_val`
- Hard-pair mining for the ranking loss (`hard_pair_*` keys)
- One scoring function, `metrics.evaluate`, used everywhere
- `cudnn.deterministic` off: the fused cuDNN RNN kernels are much faster
- Default training data `data/data_400k.csv`
