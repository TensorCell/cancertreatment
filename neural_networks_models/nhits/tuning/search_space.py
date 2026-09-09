"""Ray Tune search-space definition for N-HiTS hyperparameters in cancer treatment forecasting.

Usage
-----
>>> from nhits.tuning.search_space import default_search_space
>>> space = default_search_space()
>>> tuner = Tuner(tune_nhits, param_space=space, ...)
"""

from __future__ import annotations

from typing import Any


def default_search_space() -> dict[str, Any]:
    """Return the default Ray Tune search space for cancer treatment N-HiTS."""
    from ray import tune

    return {
        # ── Optimiser ─────────────────────────────────────────────────────
        "lr": tune.loguniform(1e-4, 1e-2),
        "weight_decay": tune.loguniform(1e-5, 1e-3),
        # ── Data loading ──────────────────────────────────────────────────
        "batch_size": tune.choice([256, 512, 1024]),
        # ── Model architecture ────────────────────────────────────────────
        "hidden_size": tune.choice([32, 64, 128]),
        "num_mlp_layers": tune.choice([2, 3]),
        "dropout": tune.uniform(0.0, 0.3),
        "activation": tune.choice(["relu", "gelu"]),
        # ── Objective & Ranking Loss ──────────────────────────────────────
        "loss": tune.choice(["mae", "mse", "huber"]),
        "margin_loss": tune.choice([True, False]),
        "margin_loss_w": tune.uniform(0.1, 5.0),
    }
