"""Entry-point for N-HiTS hyperparameter search with Ray Tune + W&B.

Usage
-----
# Quick smoke-test (4 trials, 5 epochs each)
uv run python nhits/tuning/run_tuning.py --num_samples 4 --max_epochs 5

# Full search (20 trials) with W&B logging
uv run python nhits/tuning/run_tuning.py --num_samples 20 --wandb

# Custom data, GPU
uv run python nhits/tuning/run_tuning.py \
    --data_path data/data.csv \
    --num_samples 30 --max_epochs 50 \
    --cpus_per_trial 2 --gpus_per_trial 0.5 \
    --wandb --wandb_project cancertreatment-nhits-forecasting
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


# ---------------------------------------------------------------------------
# Project Root & .env loader
# ---------------------------------------------------------------------------

def _get_project_root() -> Path:
    """Find absolute project root directory."""
    candidate = Path(__file__).resolve()
    for parent in candidate.parents:
        if (parent / "data" / "data.csv").exists() or (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def _load_dotenv(env_path: Path | None = None) -> None:
    """Parse a .env file and inject variables into ``os.environ``."""
    if env_path is None:
        candidate = Path(__file__).resolve()
        for _ in range(6):
            candidate = candidate.parent
            dotenv = candidate / ".env"
            if dotenv.exists():
                env_path = dotenv
                break
    if env_path is None or not env_path.exists():
        return
    with env_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for N-HiTS using Ray Tune + W&B on cancer treatment data."
    )
    # Search budget
    parser.add_argument("--num_samples", type=int, default=20,
                        help="Number of hyperparameter trials to run (default: 20).")
    parser.add_argument("--max_epochs", type=int, default=50,
                        help="Maximum epochs per trial (default: 50).")
    parser.add_argument("--grace_period", type=int, default=5,
                        help="ASHA minimum epochs before pruning a trial (default: 5).")

    # Resources
    parser.add_argument("--cpus_per_trial", type=float, default=1.0,
                        help="CPUs allocated per trial (default: 1.0).")
    parser.add_argument("--gpus_per_trial", type=float, default=0.0,
                        help="GPUs allocated per trial (default: 0.0).")

    # Data
    root = _get_project_root()
    default_data_path = root / "data" / "data.csv"
    default_data_str = str(default_data_path) if default_data_path.exists() else "data/data.csv"
    parser.add_argument("--data_path", type=str, default=default_data_str,
                        help="Path to EMT6/Ro simulation CSV file (default: data/data.csv).")

    # Re-train best
    parser.add_argument("--retrain_best", action="store_true",
                        help="Re-train best config for full max_epochs after search.")
    parser.add_argument("--retrain_epochs", type=int, default=100,
                        help="Epochs for best-config re-train (default: 100).")

    # Ray Tune storage
    parser.add_argument("--storage_path", type=str, default=None,
                        help="Ray Tune storage path (default: ./ray_results).")

    # W&B
    parser.add_argument("--wandb", action="store_true",
                        help="Enable W&B logging for all trials.")
    parser.add_argument("--wandb_project", type=str, default="cancertreatment-nhits-forecasting",
                        help="W&B project name (default: cancertreatment-nhits-forecasting).")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="W&B entity (username or team). Reads WANDB_ENTITY env var if unset.")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# W&B login helper
# ---------------------------------------------------------------------------

def _wandb_login() -> None:
    """Authenticate with W&B using WANDB_API_KEY or interactive login."""
    import wandb

    api_key = os.environ.get("WANDB_API_KEY", "")
    if api_key and api_key != "your_wandb_api_key_here":
        wandb.login(key=api_key, relogin=True)
    else:
        if not wandb.login(anonymous="never"):
            raise RuntimeError(
                "W&B login failed. Set WANDB_API_KEY in .env or run `wandb login`."
            )


# ---------------------------------------------------------------------------
# Best-config re-train
# ---------------------------------------------------------------------------

def _retrain_best(
    best_config: dict,
    max_epochs: int,
    data_path: str | None,
) -> None:
    """Re-train the winning configuration for the full number of epochs."""
    import torch

    from nhits.config import CancerNHiTSConfig, NHiTSConfig
    from nhits.datasets.cancer_dataset import build_dataloaders
    from nhits.models.nhits import NHiTSModel
    from nhits.trainer.trainer import Trainer

    root = _get_project_root()
    if data_path:
        p = Path(data_path)
        abs_data_path = p if p.is_absolute() else (root / p).resolve()
    else:
        abs_data_path = (root / "data" / "data.csv").resolve()

    raw_pooling = best_config.get("pooling_sizes", best_config.get("pooling_kernel_sizes", [4, 2, 1]))
    if not isinstance(raw_pooling, list):
        raw_pooling = [4, 2, 1]

    raw_blocks = best_config.get("n_blocks_per_stack", [1, 1, 1])
    if isinstance(raw_blocks, int):
        raw_blocks = [raw_blocks] * len(raw_pooling)
    elif not isinstance(raw_blocks, list):
        raw_blocks = [1] * len(raw_pooling)

    num_layers = best_config.get("num_layers_per_block", best_config.get("num_mlp_layers", 2))

    cfg = CancerNHiTSConfig(
        data_path=abs_data_path,
        pooling_sizes=raw_pooling,
        n_blocks_per_stack=raw_blocks,
        hidden_size=best_config.get("hidden_size", 64),
        num_layers_per_block=num_layers,
        dropout=best_config.get("dropout", 0.1),
        activation=best_config.get("activation", "gelu"),
        lr=best_config.get("lr", 1e-3),
        weight_decay=best_config.get("weight_decay", 1e-4),
        batch_size=best_config.get("batch_size", 512),
        max_epochs=max_epochs,
        patience=max(1, max_epochs // 5),
        loss=best_config.get("loss", "mse"),
        margin_loss=best_config.get("margin_loss", False),
        margin_loss_w=best_config.get("margin_loss_w", 1.0),
        grad_clip=best_config.get("grad_clip", 1.0),
        checkpoint_dir=root / "checkpoints" / "tune" / "best",
        output_dir=root / "outputs" / "tune" / "best",
        model_name="nhits",
        seed=best_config.get("seed", 42),
        wandb_enabled=False,
    )

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    train_loader, val_loader, _ = build_dataloaders(cfg)
    model = NHiTSModel.from_config(cfg)
    trainer = Trainer(model, cfg, device=device)
    trainer.fit(train_loader, val_loader)
    print(f"\n  Best model checkpoint: {cfg.checkpoint_dir / 'best_model.pt'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full hyperparameter search."""
    _load_dotenv()
    args = _parse_args()
    root = _get_project_root()

    if args.wandb:
        try:
            _wandb_login()
        except Exception as exc:
            print(f"[WARNING] W&B login failed: {exc}. Disabling W&B.")
            args.wandb = False

    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.optuna import OptunaSearch

    from nhits.tuning.search_space import default_search_space
    from nhits.tuning.trainable import tune_nhits

    param_space = default_search_space()
    param_space["max_epochs"] = args.max_epochs
    param_space["patience"] = max(args.grace_period, args.max_epochs // 5)

    if args.data_path:
        p = Path(args.data_path)
        abs_path = p if p.is_absolute() else (root / p).resolve()
        param_space["data_path"] = str(abs_path)

    scheduler = ASHAScheduler(
        max_t=args.max_epochs,
        grace_period=args.grace_period,
        reduction_factor=2,
    )

    search_alg = OptunaSearch()

    callbacks = []
    wandb_group = f"nhits-tune-{ray.util.get_node_ip_address()}"
    if args.wandb:
        from ray.air.integrations.wandb import WandbLoggerCallback

        entity = args.wandb_entity or os.environ.get("WANDB_ENTITY")
        wb_kwargs: dict = dict(
            project=args.wandb_project,
            group=wandb_group,
            tags=["nhits", "tune", "asha", "optuna", "cancer-treatment"],
            log_config=True,
        )
        if entity:
            wb_kwargs["entity"] = entity

        callbacks.append(WandbLoggerCallback(**wb_kwargs))
        print(f"\n  W&B project : {args.wandb_project}")
        print(f"  W&B group   : {wandb_group}")

    storage_path = (
        args.storage_path
        if args.storage_path
        else str((root / "ray_results").resolve())
    )

    print(f"\n{'='*60}")
    print(f"  N-HiTS Hyperparameter Search — Ray Tune (Cancer Treatment)")
    print(f"  Trials     : {args.num_samples}")
    print(f"  Max epochs : {args.max_epochs}")
    print(f"  Grace period: {args.grace_period}")
    print(f"  Data path  : {param_space.get('data_path')}")
    print(f"  Resources  : {args.cpus_per_trial} CPU, {args.gpus_per_trial} GPU per trial")
    print(f"{'='*60}\n")

    tuner = tune.Tuner(
        tune.with_resources(
            tune_nhits,
            resources={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
        ),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=args.num_samples,
            trial_name_creator=lambda trial: f"trial_{trial.trial_id}",
            trial_dirname_creator=lambda trial: f"trial_{trial.trial_id}",
        ),
        run_config=tune.RunConfig(
            name="nhits_tune",
            storage_path=storage_path,
            callbacks=callbacks,
            verbose=1,
        ),
    )

    results = tuner.fit()

    best_result = results.get_best_result(metric="val_loss", mode="min")

    print(f"\n{'='*60}")
    print("  Hyperparameter Search Complete")
    print(f"{'='*60}")
    print(f"\n  Best val_loss : {best_result.metrics.get('val_loss', 'N/A'):.5f}")
    print("\n  Best config:")
    for k, v in sorted(best_result.config.items()):
        if k in {"max_epochs", "patience", "data_path"}:
            continue
        print(f"    {k:<22} = {v}")

    if args.retrain_best:
        print(f"\n  Re-training best config for {args.retrain_epochs} epochs …")
        _retrain_best(
            best_config=best_result.config,
            max_epochs=args.retrain_epochs,
            data_path=param_space.get("data_path"),
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
