# Reinforcement learning for radiotherapy scheduling

This directory contains the environments, PPO trainers, saved checkpoints,
evaluation notebooks, result tables, figures, and report source used for the
radiotherapy scheduling study.

## Setup

Run commands from `ppo_models`. The EMT6-Ro simulator must be available in
the sibling directory `../EMT6-Ro`, with its Python package built under
`../EMT6-Ro/build/python`.

Install the Python dependencies and both local packages with:

```bash
python -m pip install -r requirements.txt
```

Alternatively, create the Python 3.9 Conda environment:

```bash
conda env create -f environment.yml
conda activate rl4radiotherapy
```

## Third-party code

The `TumorGrowthEnv/` package is derived from
[Agnieszka Ilnicka's TumorGrowthEnv](https://github.com/ailnicka/TumorGrowthEnv).
The original project provides an OpenAI Gym interface to the EMT6-Ro tumour
simulation. It was modified for this study.

EMT6-Ro is a separate dependency developed by Rafał Banaś. 
Its source is available in the
[EMT6-Ro repository](https://github.com/banasraf/EMT6-Ro/).

## Repository structure

- `TumorGrowthEnv/` contains the model-specific Gym environments.
- `train_*_PPO.py` trains each PPO variant from a new model.
- `baselines/` and `fig05_compare_four_models_at_steps.ipynb` contain the figure evaluations.
- `baselines/results/models/` contains checkpoints grouped by PPO model.
- `check_ppo_choices.py` summarizes complete protocols selected by PPO models.

## PPO training

| Trainer | Environment | Default training steps |
|---|---|---:|
| `train_paper_PPO.py` | Paper PPO environment | 25,000 |
| `train_normalized_PPO.py` | Shared environment, normalized profile | 50,000 |
| `train_fract_PPO.py` | Shared environment, fraction profile | 50,000 |
| `train_mesh_PPO.py` | Shared environment, mesh profile | 25,000 |
| `train_cnn_mesh_PPO.py` | CNN Mesh PPO environment | 50,000 |

Run one trainer at a time:

```bash
python train_paper_PPO.py
python train_normalized_PPO.py
python train_fract_PPO.py
python train_mesh_PPO.py
python train_cnn_mesh_PPO.py
```

By default, training starts from a new model, uses seed 0, ten parallel
simulations, 24-hour treatment availability, and no treatment delays.
Checkpoints are saved every 1,000 steps under
`baselines/results/models/<model>/` together with logs and `settings.json`.

The original `*-v0` Gym environments remain available for the training scripts
and older evaluations; they accept `appointment_delays`. The added `*-v1`
environments accept `treatment_delay` and use a nonnegative delay sampled from
the normal distribution with mean 34.4 minutes and standard deviation 46.3
minutes. The Figure 3, 5, and 6 evaluation notebooks use `*-v1`; Figure 6
enables the delay, while Figures 3 and 5 disable it.

## Evaluation and figures

| Figure | Source | Purpose |
|---|---|---|
| 2 | `baselines/fig02_baselines_10x10_24h.ipynb` | Compare 25k PPO checkpoints with reference protocols over 10 rounds |
| 3 | `baselines/fig03_baselines_with_mesh_30x10_24h.ipynb` | Compare final PPO checkpoints, including Mesh PPO, with reference protocols over 30 rounds |
| 4 | `baselines/fig04_compare_ppo_checkpoints.ipynb` | Plot Paper, Normalized, Fract, and Mesh PPO performance from 1k to 25k steps |
| 5 | `fig05_compare_four_models_at_steps.ipynb` | Plot Normalized, Fract, Mesh, and CNN Mesh PPO performance from 1k to 50k steps |
| 6 | `baselines/fig06_baselines_treatment_delay.ipynb` | Evaluate models and protocols with treatment delays |
| 7 | `baselines/fig07_baselines_10h.ipynb` | Evaluate models and protocols with a 10-hour daily treatment window |

Figures 6 and 7 compare their evaluation summaries with the normal 24-hour
results. Generate the final comparison plots with:

```bash
python baselines/fig06_plot_treatment_delay_comparison.py
python baselines/fig07_plot_10h_comparison.py
```

Evaluation outputs are written to `baselines/results/`. Filenames beginning
with `fig01` through `fig07` correspond to the figure numbers in the report.
In result filenames, `10x10` and `30x10` mean evaluation rounds × parallel
simulations.

Figure 5 can be redrawn from its complete saved checkpoint-evaluation CSV.
When reevaluation is needed, checkpoint directories are discovered recursively
under `baselines/results/models/<model_name>/`.

To calculate how often PPO selects each complete treatment protocol, run:

```bash
python check_ppo_choices.py
```

`check_ppo_choices.py` was used to analyse the PPO-selected action sequences
reported in the study. The results supporting the seven Fract PPO sequences
and four grouped protocols are stored in
`baselines/results/ppo_choice_frequencies.csv`.
