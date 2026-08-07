"""Model evaluation on the test set with metrics and visualisations.

Functions
---------
collect_predictions
    Run the model over a DataLoader, collect predictions and targets.
evaluate
    Compute all metrics and produce plots + CSVs saved to ``cfg.output_dir``.

Outputs saved
-------------
  outputs/tide/
    ├── metrics.csv            — all scalar metrics
    ├── predictions.csv        — predicted vs. actual for every test protocol
    ├── forecast_vs_actual.png — scatter plot: predicted vs actual tumour cells
    ├── residuals.png          — residual distribution (histogram + sorted plot)
    └── training_history.png   — train/val loss curves (if history provided)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from tide.config import CancerTiDEConfig
from tide.metrics.forecasting_metrics import compute_all_metrics
from tide.models.tide import TiDEModel

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Collect predictions
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(
    model: TiDEModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Iterate over *loader* and collect all predictions and ground-truth values.

    Parameters
    ----------
    model:
        ``TiDEModel`` in eval mode.
    loader:
        DataLoader (typically the test loader).
    device:
        Inference device.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(preds, targets)`` both of shape ``(N,)`` — one value per protocol.
    """
    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    for batch in loader:
        past_target = batch["past_target"].to(device)
        hist_covs = batch["hist_covs"].to(device)
        future_covs = batch["future_covs"].to(device)
        target = batch["target"].cpu().numpy()  # (B, 1)

        pred = model(past_target, hist_covs, future_covs).cpu().numpy()  # (B, 1)
        all_preds.append(pred)
        all_targets.append(target)

    return np.concatenate(all_preds).ravel(), np.concatenate(all_targets).ravel()


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _plot_forecast_vs_actual(
    preds: np.ndarray,
    targets: np.ndarray,
    output_dir: Path,
) -> None:
    """Scatter plot of predicted vs. actual normalised tumour cell counts."""
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(targets, preds, alpha=0.15, s=4, color="#2d6a9f", rasterized=True)

    # Perfect-prediction diagonal
    lo = min(targets.min(), preds.min())
    hi = max(targets.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], color="#e87040", linewidth=1.5, linestyle="--",
            label="Perfect prediction")

    ax.set_title("TiDE — Predicted vs Actual Tumour Cells (test set)")
    ax.set_xlabel("Actual normalised tumour cells")
    ax.set_ylabel("Predicted normalised tumour cells")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = output_dir / "forecast_vs_actual.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_residuals(
    preds: np.ndarray,
    targets: np.ndarray,
    output_dir: Path,
) -> None:
    """Residual distribution: histogram + sorted residuals."""
    residuals = targets - preds

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Sorted residuals
    ax = axes[0]
    ax.plot(np.sort(residuals), linewidth=0.6, color="#7b5ea7")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Residuals (sorted, all test protocols)")
    ax.set_xlabel("Protocol rank")
    ax.set_ylabel("Residual (actual − predicted)")
    ax.grid(True, alpha=0.3)

    # Histogram
    ax = axes[1]
    ax.hist(residuals, bins=80, color="#7b5ea7", alpha=0.75, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Residual distribution")
    ax.set_xlabel("Residual (actual − predicted)")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = output_dir / "residuals.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_training_history(
    history: dict[str, list[float]],
    output_dir: Path,
) -> None:
    """Plot train/val loss curves."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, history["train_loss"], label="Train loss", color="#2d6a9f")
    ax.plot(epochs, history["val_loss"], label="Val loss", color="#e87040")
    ax.set_title("Training History — TiDE Cancer Treatment")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = output_dir / "training_history.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def _save_predictions_csv(
    preds: np.ndarray,
    targets: np.ndarray,
    output_dir: Path,
) -> None:
    """Save per-protocol predicted and actual values to a CSV file."""
    df = pd.DataFrame({
        "protocol_idx": np.arange(len(preds)),
        "actual": targets,
        "predicted": preds,
        "residual": targets - preds,
        "abs_error": np.abs(targets - preds),
    })
    path = output_dir / "predictions.csv"
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


def _save_metrics_csv(
    metrics: dict[str, float],
    output_dir: Path,
) -> None:
    """Save scalar metrics to a CSV file."""
    df = pd.DataFrame(list(metrics.items()), columns=["metric", "value"])
    path = output_dir / "metrics.csv"
    df.to_csv(path, index=False)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main evaluate function
# ---------------------------------------------------------------------------

def evaluate(
    model: TiDEModel,
    test_loader: DataLoader,
    cfg: CancerTiDEConfig,
    device: torch.device,
    history: dict[str, list[float]] | None = None,
    train_targets: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate the model on the test set and save all outputs.

    Parameters
    ----------
    model:
        Trained ``TiDEModel`` in eval mode.
    test_loader:
        Test DataLoader.
    cfg:
        ``CancerTiDEConfig`` instance (used for ``output_dir``).
    device:
        Inference device.
    history:
        Training history dict (optional; used to plot loss curves).
    train_targets:
        Raw training target values for MASE computation (optional).

    Returns
    -------
    dict[str, float]
        Computed metrics.
    """
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  Evaluation on test set")
    print("=" * 60)

    preds, targets = collect_predictions(model, test_loader, device)

    metrics = compute_all_metrics(targets, preds, y_train=train_targets, seasonality=1)

    # ── Print metrics table ────────────────────────────────────────────────
    print(f"\n  {'Metric':<10} {'Value':>12}")
    print(f"  {'-'*24}")
    for name, value in metrics.items():
        if name in ("mape", "smape", "pct_correct"):
            unit, display = "%", value
        #elif name == "pct_correct":
        #    unit, display = "%", value*100
        else:
            unit, display = "", value
        print(f"  {name.upper():<14} {display:>11.6f}{unit}")

    # ── Save CSVs ──────────────────────────────────────────────────────────
    print("\n  Saving outputs …")
    _save_predictions_csv(preds, targets, cfg.output_dir)
    _save_metrics_csv(metrics, cfg.output_dir)

    # ── Save plots ─────────────────────────────────────────────────────────
    _plot_forecast_vs_actual(preds, targets, cfg.output_dir)
    _plot_residuals(preds, targets, cfg.output_dir)

    if history is not None:
        _plot_training_history(history, cfg.output_dir)

    # ── W&B: log test metrics and plots ───────────────────────────────
    if _WANDB_AVAILABLE and _wandb.run is not None:
        # scalar metrics (prefixed with "test/")
        wandb_metrics = {f"test/{k}": v for k, v in metrics.items()}

        # evaluation plots as W&B Images
        plot_files = [
            "forecast_vs_actual.png",
            "residuals.png",
            "error_by_horizon.png",
            "training_history.png",
        ]
        wandb_images: dict[str, _wandb.Image] = {}
        for fname in plot_files:
            fpath = cfg.output_dir / fname
            if fpath.exists():
                key = fname.replace(".png", "").replace("_", " ").title()
                wandb_images[f"eval/{key}"] = _wandb.Image(
                    str(fpath), caption=key
                )

        _wandb.log({**wandb_metrics, **wandb_images})
        # also surface metrics in the run summary for easy comparison
        for k, v in metrics.items():
            _wandb.summary[f"test/{k}"] = v
        print("  W&B: test metrics and plots logged.")

    print(f"\n  All outputs saved to: {cfg.output_dir.resolve()}")
    return metrics
