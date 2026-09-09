"""Ray Tune trainable function for N-HiTS.

Each Ray Tune *trial* calls ``tune_nhits(config)`` once.  The function:

1. Merges the trial's ``config`` dict into a base ``CancerNHiTSConfig``.
2. Builds data loaders, model, and trainer.
3. Runs the training loop, reporting ``val_loss`` to Tune after every epoch
   (enables ASHA to prune poorly-performing trials early).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch


def _get_project_root() -> Path:
    """Find absolute project root directory."""
    candidate = Path(__file__).resolve()
    for parent in candidate.parents:
        if (parent / "data" / "data.csv").exists() or (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def _resolve_device() -> str:
    """Pick the best available device string."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def tune_nhits(config: dict[str, Any]) -> None:
    """Ray Tune trainable: train one NHiTS trial and report metrics."""
    from ray import tune

    from nhits.config import CancerNHiTSConfig, NHiTSConfig
    from nhits.datasets.cancer_dataset import build_dataloaders
    from nhits.models.nhits import NHiTSModel
    from nhits.trainer.trainer import Trainer

    root = _get_project_root()
    data_path_raw = config.get("data_path", root / "data" / "data.csv")
    if data_path_raw:
        p = Path(data_path_raw)
        data_path = p.resolve() if p.is_absolute() else (root / p).resolve()
    else:
        data_path = (root / "data" / "data.csv").resolve()

    # Handle pooling_sizes and n_blocks_per_stack formats
    raw_pooling = config.get("pooling_sizes", config.get("pooling_kernel_sizes", [4, 2, 1]))
    if not isinstance(raw_pooling, list):
        raw_pooling = [4, 2, 1]

    raw_blocks = config.get("n_blocks_per_stack", [1, 1, 1])
    if isinstance(raw_blocks, int):
        raw_blocks = [raw_blocks] * len(raw_pooling)
    elif not isinstance(raw_blocks, list):
        raw_blocks = [1] * len(raw_pooling)

    num_layers = config.get("num_layers_per_block", config.get("num_mlp_layers", 2))

    trial_id = os.environ.get("TUNE_TRIAL_ID", "local")
    checkpoint_dir = root / "checkpoints" / "tune" / trial_id
    output_dir = root / "outputs" / "tune" / trial_id

    cfg = CancerNHiTSConfig(
        data_path=data_path,
        pooling_sizes=raw_pooling,
        n_blocks_per_stack=raw_blocks,
        hidden_size=config.get("hidden_size", 64),
        num_layers_per_block=num_layers,
        dropout=config.get("dropout", 0.1),
        activation=config.get("activation", "gelu"),
        lr=config.get("lr", 1e-3),
        weight_decay=config.get("weight_decay", 1e-4),
        batch_size=config.get("batch_size", 512),
        max_epochs=config.get("max_epochs", 50),
        patience=config.get("patience", 10),
        loss=config.get("loss", "mse"),
        margin_loss=config.get("margin_loss", False),
        margin_loss_w=config.get("margin_loss_w", 1.0),
        grad_clip=config.get("grad_clip", 1.0),
        checkpoint_dir=checkpoint_dir,
        output_dir=output_dir,
        model_name="nhits",
        seed=config.get("seed", 42),
        wandb_enabled=False,
    )

    device = _resolve_device()

    train_loader, val_loader, _ = build_dataloaders(cfg)
    model = NHiTSModel.from_config(cfg)

    def _on_epoch_end(epoch: int, train_loss: float, val_loss: float) -> None:
        tune.report({"val_loss": val_loss, "train_loss": train_loss, "epoch": epoch})

    trainer = Trainer(model, cfg, device=device, on_epoch_end=_on_epoch_end)
    trainer.fit(train_loader, val_loader)
