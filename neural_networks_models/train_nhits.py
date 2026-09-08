"""NHiTS training entry point for cancer treatment tumour cell survival forecasting.

Usage
-----
# Full training run with defaults
uv run python train_nhits.py --data_path data/data.csv

# Quick smoke-test (2 epochs)
uv run python train_nhits.py --data_path data/data.csv --max_epochs 2 --batch_size 512

# Adjust model size and pooling
uv run python train_nhits.py --data_path data/data.csv --hidden_size 128 --pooling_sizes 4 2 1

# Use MAE loss instead of MSE
uv run python train_nhits.py --data_path data/data.csv --loss mae

# Enable Margin Ranking Loss alongside the base regression loss
uv run python train_nhits.py --data_path data/data.csv --margin_loss --margin_loss_w 3
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch

from nhits.config import CancerNHiTSConfig
from nhits.datasets.cancer_dataset import build_dataloaders
from nhits.evaluation.evaluate import evaluate
from nhits.inference.predict import explain_forecast_components, explain_input_saliency, load_model
from nhits.models.nhits import NHiTSModel
from nhits.trainer.trainer import Trainer

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------

def _load_dotenv(env_path: Path | None = None) -> None:
    """Parse a .env file and inject variables into ``os.environ``."""
    if env_path is None:
        candidate = Path(__file__).resolve()
        for _ in range(5):
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
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> CancerNHiTSConfig:
    """Parse CLI arguments and return a populated ``CancerNHiTSConfig``."""
    parser = argparse.ArgumentParser(
        description=(
            "Train an NHiTS model to predict tumour cell survival "
            "from EMT6/Ro radiotherapy protocol simulations."
        )
    )

    # Data
    parser.add_argument(
        "--data_path", type=str, required=True,
        help="Path to data.csv (EMT6/Ro simulation output)"
    )
    parser.add_argument("--input_len", type=int, default=20,
                        help="Number of dose steps per protocol (default: 20)")
    parser.add_argument("--train_frac", type=float, default=0.70)
    parser.add_argument("--val_frac", type=float, default=0.15)

    # Model
    parser.add_argument("--pooling_sizes", type=int, nargs="+", default=[4, 2, 1],
                        help="Pooling downsampling sizes per stack (default: 4 2 1)")
    parser.add_argument("--n_blocks_per_stack", type=int, nargs="+", default=[1, 1, 1],
                        help="Number of blocks per stack (default: 1 1 1)")
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers_per_block", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="gelu",
                        choices=["relu", "gelu"], help="Activation function (default: gelu)")
    parser.add_argument("--no_layer_norm", action="store_true",
                        help="Disable LayerNorm in MLP blocks")
    parser.add_argument("--no_global_skip", action="store_true",
                        help="Disable global linear skip connection")

    # Training
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--loss", type=str, default="mse",
                        choices=["mae", "mse", "huber"])
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Margin Ranking Loss
    parser.add_argument(
        "--margin_loss", action="store_true",
        help="Enable Margin Ranking Loss (MRL) alongside the base regression loss"
    )
    parser.add_argument(
        "--margin_loss_w", type=float, default=1.0,
        help="Weight on MRL term in combined loss"
    )

    # Output
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)

    # Weights & Biases
    parser.add_argument("--wandb", action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="cancertreatment-nhits-forecasting",
                        help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default="j95-jaworska-na",
                        help="W&B entity")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Optional W&B run name")

    args = parser.parse_args()

    return CancerNHiTSConfig(
        data_path=Path(args.data_path),
        input_len=args.input_len,
        horizon=1,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        num_hist_covariates=2,
        num_future_covariates=0,
        pooling_sizes=list(args.pooling_sizes),
        n_blocks_per_stack=list(args.n_blocks_per_stack),
        hidden_size=args.hidden_size,
        num_layers_per_block=args.num_layers_per_block,
        dropout=args.dropout,
        activation=args.activation,
        use_layer_norm=not args.no_layer_norm,
        use_global_skip=not args.no_global_skip,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        loss=args.loss,
        grad_clip=args.grad_clip,
        margin_loss=args.margin_loss,
        margin_loss_w=args.margin_loss_w,
        checkpoint_dir=Path(args.checkpoint_dir),
        output_dir=Path(args.output_dir),
        seed=args.seed,
        wandb_enabled=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Full training and evaluation pipeline for NHiTS."""
    _load_dotenv()

    cfg = parse_args()
    set_seed(cfg.seed)

    if cfg.wandb_enabled and _WANDB_AVAILABLE:
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key and api_key != "your_wandb_api_key_here":
            _wandb.login(key=api_key, relogin=True)
        else:
            if not _wandb.login(anonymous="never"):
                raise RuntimeError("W&B login failed.")

    if torch.cuda.is_available():
        device_str = "cuda"
    elif torch.backends.mps.is_available():
        device_str = "mps"
    else:
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")

    print("\nBuilding dataloaders …")
    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    n_train = len(train_loader.dataset)   # type: ignore[arg-type]
    n_val = len(val_loader.dataset)       # type: ignore[arg-type]
    n_test = len(test_loader.dataset)     # type: ignore[arg-type]
    print(f"  Protocols — train: {n_train:,}  val: {n_val:,}  test: {n_test:,}")

    scale_stats = train_loader.dataset.scale_stats  # type: ignore[attr-defined]

    print("\nConstructing NHiTSModel …")
    model = NHiTSModel.from_config(cfg)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {n_params:,}")
    print(f"  Pooling sizes:        {cfg.pooling_sizes}  | Blocks per stack: {cfg.n_blocks_per_stack}")
    print(f"  Hidden size:          {cfg.hidden_size}  |  Dropout: {cfg.dropout}")
    print(f"  Input len:            {cfg.input_len} steps  |  Horizon: {cfg.horizon}")

    trainer = Trainer(model, cfg, device=device_str)
    history = trainer.fit(train_loader, val_loader)

    print("\nLoading best checkpoint …")
    model = load_model(cfg, device=device_str)

    evaluate(
        model=model,
        test_loader=test_loader,
        cfg=cfg,
        device=device,
        history=history,
        train_targets=train_loader.dataset.targets.cpu().numpy(),  # type: ignore[attr-defined]
    )

    if cfg.wandb_enabled and _WANDB_AVAILABLE and _wandb.run is not None:
        _wandb.finish()
        print("  W&B run finished.")

    print("\n" + "=" * 60)
    print("  Inference demo: predicting one test protocol (NHiTS)")
    print("=" * 60)

    test_ds = test_loader.dataset  # type: ignore[attr-defined]
    sample = test_ds[0]
    dose_scaled = sample["past_target"].numpy()
    dose_range = scale_stats.dose_max - scale_stats.dose_min
    dose_raw = dose_scaled * dose_range + scale_stats.dose_min

    print(f"\n  Sample dose sequence (Gy): {dose_raw.round(3).tolist()}")
    print(f"  Actual tumour cells (normalised): {sample['target'].item():.6f}")

    past_t = sample["past_target"].unsqueeze(0).to(device)
    hist_c = sample["hist_covs"].unsqueeze(0).to(device)
    fut_c = sample["future_covs"].unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pred = model(past_t, hist_c, fut_c)

    print(f"  Predicted tumour cells (normalised): {pred.item():.6f}")
    print(f"  Absolute error: {abs(pred.item() - sample['target'].item()):.6f}")

    components = explain_forecast_components(
        model=model,
        past_t=past_t,
        hist_c=hist_c,
        fut_c=fut_c,
        scale_mean=scale_stats.target_mean,
        scale_std=scale_stats.target_std,
    )

    print("Total forecast:       ", components["total_forecast"])
    if "linear_skip_baseline" in components:
        print("Linear skip baseline: ", components["linear_skip_baseline"])
    print("Deep adjustment:      ", components["deep_nonlinear_adjustment"])

    saliency = explain_input_saliency(
        model=model,
        past_t=past_t,
        hist_c=hist_c,
        fut_c=fut_c,
        target_horizon_step=0,
    )
    print("Look-back timesteps saliency scores (sums to 1.0):")
    print(saliency)


if __name__ == "__main__":
    main()
