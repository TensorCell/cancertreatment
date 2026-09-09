"""Standalone inference and explainability pipeline for the NHiTS best model.

Follows the NHITS_INFERENCE_GUIDE:
  https://github.com/JJDec/neural-networks-guide/blob/main/nhits/NHITS_INFERENCE_GUIDE.md

Steps
-----
1.  Prerequisites  : load best_model.pt + rebuild data loaders (same split/scaling)
2.  Input prep     : reconstruct past_target / hist_covs / future_covs tensors
3.  Inference      : full test-set forward pass, predictions CSV
4.  Metrics        : MAE, RMSE, R2, MAPE, SMAPE, MASE, pct_correct
5.  Scatter plot   : predicted vs actual
6.  Residuals      : sorted residuals + histogram
7.  Component dec. : linear-skip baseline vs deep-block nonlinear adjustment
8.  Saliency       : averaged Gradient x Input saliency over N test protocols
9.  Dose sweep     : predicted tumour count vs uniform dose level (0->1)
10. Qual. review   : top-5 best and worst predictions with dose profiles
11. JSON summary   : machine-readable inference_summary.json

Usage
-----
uv run python infer_nhits.py --data_path data/data.csv
uv run python infer_nhits.py --data_path data/data.csv --checkpoint checkpoints/nhits/best_model.pt --n_explain 200
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend safe on all platforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from nhits.config import CancerNHiTSConfig
from nhits.datasets.cancer_dataset import build_dataloaders
from nhits.inference.predict import (
    explain_forecast_components,
    explain_input_saliency,
    load_model,
)
from nhits.metrics.forecasting_metrics import compute_all_metrics
from nhits.models.nhits import NHiTSModel


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility (guide sec 1.2)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=(
            "NHiTS inference + explainability pipeline for cancer treatment "
            "tumour cell survival forecasting."
        )
    )
    p.add_argument("--data_path",   default="data/data.csv",
                   help="Path to data.csv (default: data/data.csv)")
    p.add_argument("--checkpoint",  default="checkpoints/nhits/best_model.pt",
                   help="Path to best_model.pt")
    p.add_argument("--output_dir",  default="outputs/nhits/inference",
                   help="Output directory for all artefacts")
    p.add_argument("--n_explain",   type=int, default=500,
                   help="Test samples used for saliency / component analysis")
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--batch_size",  type=int, default=512)
    p.add_argument("--hidden_size", type=int, default=64,
                   help="Hidden size for MLP blocks (default: 64)")
    p.add_argument("--num_layers_per_block", type=int, default=3,
                   help="Number of layers per block MLP (default: 3 for best model)")
    p.add_argument("--dropout", type=float, default=0.0179,
                   help="Dropout probability (default: 0.0179)")
    p.add_argument("--pooling_sizes", type=int, nargs="+", default=[4, 2, 1],
                   help="Pooling downsampling sizes per stack (default: 4 2 1)")
    p.add_argument("--n_blocks_per_stack", type=int, nargs="+", default=[1, 1, 1],
                   help="Number of blocks per stack (default: 1 1 1)")
    p.add_argument("--no_layer_norm", action="store_true",
                   help="Disable LayerNorm in MLP blocks")
    p.add_argument("--no_global_skip", action="store_true",
                   help="Disable global linear skip connection")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 2+3  Input data prep + full test-set inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_test_predictions(
    model: NHiTSModel,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Run full test-set inference; returns (preds, targets, per-sample dicts).

    Guide sec 2: Each sample already has correctly shaped tensors:
      past_target  (1, L)        scaled dose sequence
      hist_covs    (1, L, 2)    scaled [time, time_gap]
      future_covs  (1, H, 0)    empty -- no future inputs
    Guide sec 3: model.eval() + torch.no_grad() are both active.
    """
    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    samples: list[dict[str, Any]] = []

    for batch in test_loader:
        pt = batch["past_target"].to(device)
        hc = batch["hist_covs"].to(device)
        fc = batch["future_covs"].to(device)
        pred = model(pt, hc, fc).cpu()   # (B, 1)
        tgt  = batch["target"].cpu()     # (B, 1)
        all_preds.append(pred.numpy())
        all_targets.append(tgt.numpy())
        for i in range(pt.size(0)):
            samples.append({
                "past_target":  pt[i].unsqueeze(0).cpu(),
                "hist_covs":    hc[i].unsqueeze(0).cpu(),
                "future_covs":  fc[i].unsqueeze(0).cpu(),
                "target": float(tgt[i].item()),
                "pred":   float(pred[i].item()),
            })

    return (
        np.concatenate(all_preds).ravel(),
        np.concatenate(all_targets).ravel(),
        samples,
    )


# ---------------------------------------------------------------------------
# Step 4  Metrics
# ---------------------------------------------------------------------------

def compute_and_print_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
    train_targets: np.ndarray,
    output_dir: Path,
) -> dict[str, float]:
    """Compute all regression metrics; print and save to CSV."""
    metrics = compute_all_metrics(targets, preds, y_train=train_targets, seasonality=1)
    print("\n  Metric summary:")
    print(f"  {'-'*30}")
    for name, value in metrics.items():
        suffix = "%" if name in ("mape", "smape", "pct_correct") else ""
        print(f"  {name.upper():<16} {value:>12.6f}{suffix}")
    df = pd.DataFrame(list(metrics.items()), columns=["metric", "value"])
    p = output_dir / "inference_metrics.csv"
    df.to_csv(p, index=False)
    print(f"  Metrics saved -> {p}")
    return metrics


# ---------------------------------------------------------------------------
# Steps 5+6  Scatter plot and residuals
# ---------------------------------------------------------------------------

def plot_forecast_vs_actual(preds: np.ndarray, targets: np.ndarray, output_dir: Path) -> None:
    """Scatter: predicted vs actual normalised tumour cell counts."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(targets, preds, alpha=0.10, s=3, color="#2d6a9f", rasterized=True)
    lo = min(targets.min(), preds.min())
    hi = max(targets.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], color="#e87040", lw=1.8, ls="--", label="Perfect prediction")
    ax.set_title("NHiTS - Predicted vs Actual Tumour Cells (test set)")
    ax.set_xlabel("Actual normalised tumour cells")
    ax.set_ylabel("Predicted normalised tumour cells")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = output_dir / "inference_forecast_vs_actual.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Plot saved -> {p}")


def plot_residuals(preds: np.ndarray, targets: np.ndarray, output_dir: Path) -> None:
    """Sorted residuals + histogram."""
    residuals = targets - preds
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(np.sort(residuals), lw=0.6, color="#7b5ea7")
    ax.axhline(0, color="black", lw=0.9, ls="--")
    ax.set_title("Residuals (sorted, all test protocols)")
    ax.set_xlabel("Protocol rank")
    ax.set_ylabel("actual - predicted")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(residuals, bins=80, color="#7b5ea7", alpha=0.78, edgecolor="white")
    ax.axvline(0, color="black", lw=1.0)
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    p = output_dir / "inference_residuals.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Plot saved -> {p}")


# ---------------------------------------------------------------------------
# Step 7  Component decomposition (guide sec 4 -- explainability)
# ---------------------------------------------------------------------------

def analyse_components(
    model: NHiTSModel,
    samples: list[dict[str, Any]],
    n_samples: int,
    scale_mean: float,
    scale_std: float,
    output_dir: Path,
) -> None:
    """Decompose each forecast into linear-skip baseline + deep-block adjustment.

    Per the guide: the global linear skip connection isolates the trend/baseline
    component; the hierarchical MLP blocks capture the nonlinear residual.
    Results are de-normalised using the training target mean/std.
    """
    rng = np.random.default_rng(42)
    chosen = rng.choice(len(samples), size=min(n_samples, len(samples)), replace=False)
    totals: list[float] = []
    skips:  list[float] = []
    deeps:  list[float] = []

    for idx in chosen:
        s = samples[idx]
        comp = explain_forecast_components(
            model=model,
            past_t=s["past_target"],
            hist_c=s["hist_covs"],
            fut_c=s["future_covs"],
            scale_mean=scale_mean,
            scale_std=scale_std,
        )
        totals.append(float(np.asarray(comp["total_forecast"]).item()))
        deeps.append(float(np.asarray(comp["deep_nonlinear_adjustment"]).item()))
        skip_val = comp.get("linear_skip_baseline")
        skips.append(float(np.asarray(skip_val).item()) if skip_val is not None else float("nan"))

    ta = np.array(totals)
    sa = np.array(skips)
    da = np.array(deeps)

    df_comp = pd.DataFrame({
        "sample_idx": chosen,
        "total_forecast": ta,
        "linear_skip_baseline": sa,
        "deep_nonlinear_adjustment": da,
    })
    csv_p = output_dir / "component_decomposition.csv"
    df_comp.to_csv(csv_p, index=False)
    print(f"  Component CSV -> {csv_p}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, arr, label, color in zip(
        axes,
        [ta, sa, da],
        ["Total Forecast", "Linear Skip Baseline", "Deep Nonlinear Adjustment"],
        ["#2d6a9f", "#e87040", "#3ba272"],
    ):
        valid = arr[~np.isnan(arr)]
        ax.hist(valid, bins=40, color=color, alpha=0.80, edgecolor="white")
        if len(valid):
            ax.axvline(float(valid.mean()), color="black", lw=1.2, ls="--",
                       label=f"Mean={valid.mean():.3f}")
        ax.set_title(label)
        ax.set_xlabel("Value (de-normalised)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("NHiTS Forecast Component Decomposition", fontsize=13, fontweight="bold")
    fig.tight_layout()
    pp = output_dir / "component_decomposition.png"
    fig.savefig(pp, dpi=150)
    plt.close(fig)
    print(f"  Component plot -> {pp}")

    print(f"  Component summary (n={len(chosen):,}):")
    print(f"    Total forecast        mean={np.nanmean(ta):.4f}  std={np.nanstd(ta):.4f}")
    if not np.all(np.isnan(sa)):
        print(f"    Linear skip baseline  mean={np.nanmean(sa):.4f}  std={np.nanstd(sa):.4f}")
    print(f"    Deep nonlinear adjust mean={np.nanmean(da):.4f}  std={np.nanstd(da):.4f}")


# ---------------------------------------------------------------------------
# Step 8  Gradient x Input saliency (guide sec 5)
# ---------------------------------------------------------------------------

def analyse_saliency(
    model: NHiTSModel,
    samples: list[dict[str, Any]],
    n_samples: int,
    output_dir: Path,
) -> np.ndarray:
    """Averaged Gradient x Input saliency across n_samples test protocols.

    Per the guide: the saliency score for each look-back timestep quantifies
    how much that dose step drives the predicted tumour cell count.
    Scores are normalised per sample so they sum to 1.
    """
    rng = np.random.default_rng(42)
    chosen = rng.choice(len(samples), size=min(n_samples, len(samples)), replace=False)
    saliency_list: list[np.ndarray] = []

    for idx in chosen:
        s = samples[idx]
        sal = explain_input_saliency(
            model=model,
            past_t=s["past_target"],
            hist_c=s["hist_covs"],
            fut_c=s["future_covs"],
            target_horizon_step=0,
        )
        saliency_list.append(sal)

    saliency_matrix = np.stack(saliency_list, axis=0)   # (n, L)
    mean_sal = saliency_matrix.mean(axis=0)              # (L,)
    std_sal  = saliency_matrix.std(axis=0)               # (L,)
    L = mean_sal.shape[0]

    df_sal = pd.DataFrame({
        "timestep":       np.arange(L),
        "mean_saliency":  mean_sal,
        "std_saliency":   std_sal,
    })
    csv_p = output_dir / "saliency.csv"
    df_sal.to_csv(csv_p, index=False)
    print(f"  Saliency CSV -> {csv_p}")

    x = np.arange(L)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, mean_sal, color="#2d6a9f", alpha=0.85, label="Mean saliency")
    ax.fill_between(x, mean_sal - std_sal, mean_sal + std_sal,
                    color="#2d6a9f", alpha=0.25, label="+-1 std")
    ax.set_title(
        f"NHiTS - Gradient x Input Saliency  (n={len(chosen):,} samples)",
        fontsize=11,
    )
    ax.set_xlabel("Look-back dose-step index")
    ax.set_ylabel("Normalised saliency score")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    pp = output_dir / "saliency.png"
    fig.savefig(pp, dpi=150)
    plt.close(fig)
    print(f"  Saliency plot -> {pp}")

    top_k = np.argsort(mean_sal)[::-1][:3]
    print(
        f"  Top-3 most salient dose steps: {top_k.tolist()} "
        f"(scores: {mean_sal[top_k].round(4).tolist()})"
    )
    return mean_sal


# ---------------------------------------------------------------------------
# Step 9  Dose-sweep sensitivity
# ---------------------------------------------------------------------------

def dose_sweep_sensitivity(
    model: NHiTSModel,
    reference_sample: dict[str, Any],
    output_dir: Path,
    n_points: int = 50,
) -> None:
    """Predicted tumour count as uniform dose is swept 0->1 (scaled).

    Holds the reference sample time/gap covariate structure constant;
    only the dose values are replaced -- reveals the learned dose-response curve.
    """
    model.eval()
    device = next(model.parameters()).device
    hc = reference_sample["hist_covs"].to(device)
    fc = reference_sample["future_covs"].to(device)
    L  = reference_sample["past_target"].shape[1]

    levels = np.linspace(0.0, 1.0, n_points)
    preds_sweep: list[float] = []
    with torch.no_grad():
        for d in levels:
            dose = torch.full((1, L), float(d), dtype=torch.float32, device=device)
            preds_sweep.append(float(model(dose, hc, fc).squeeze().item()))

    arr = np.array(preds_sweep)

    df_sw = pd.DataFrame({"dose_level_scaled": levels, "predicted_tumour_scaled": arr})
    csv_p = output_dir / "dose_sweep.csv"
    df_sw.to_csv(csv_p, index=False)
    print(f"  Dose-sweep CSV -> {csv_p}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(levels, arr, color="#e87040", lw=2.2, marker="o",
            markersize=4, markerfacecolor="white", markeredgewidth=1.5)
    ax.fill_between(levels, arr, alpha=0.15, color="#e87040")
    ax.set_title("NHiTS - Tumour Survival vs Uniform Dose (dose-sweep sensitivity)")
    ax.set_xlabel("Uniform dose level (normalised: 0=no dose, 1=max dose)")
    ax.set_ylabel("Predicted tumour cells (normalised)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    pp = output_dir / "dose_sweep.png"
    fig.savefig(pp, dpi=150)
    plt.close(fig)
    print(f"  Dose-sweep plot -> {pp}")


# ---------------------------------------------------------------------------
# Step 10  Qualitative review: top-5 best and worst
# ---------------------------------------------------------------------------

def qualitative_review(samples: list[dict[str, Any]], output_dir: Path) -> None:
    """Identify and plot dose profiles for the top-5 best and worst predictions."""
    abs_errors = np.array([abs(s["pred"] - s["target"]) for s in samples])
    best_idx  = np.argsort(abs_errors)[:5]
    worst_idx = np.argsort(abs_errors)[-5:][::-1]

    rows: list[dict[str, Any]] = []
    for rank, idx in enumerate(best_idx, 1):
        rows.append({
            "rank": rank, "group": "best", "sample_idx": int(idx),
            "actual":    float(samples[idx]["target"]),
            "predicted": float(samples[idx]["pred"]),
            "abs_error": float(abs_errors[idx]),
        })
    for rank, idx in enumerate(worst_idx, 1):
        rows.append({
            "rank": rank, "group": "worst", "sample_idx": int(idx),
            "actual":    float(samples[idx]["target"]),
            "predicted": float(samples[idx]["pred"]),
            "abs_error": float(abs_errors[idx]),
        })

    df = pd.DataFrame(rows)
    csv_p = output_dir / "qualitative_review.csv"
    df.to_csv(csv_p, index=False)
    print(f"  Qualitative CSV -> {csv_p}")

    fig, axes = plt.subplots(2, 5, figsize=(18, 6))
    for col, idx in enumerate(best_idx):
        s    = samples[idx]
        dose = s["past_target"].squeeze().numpy()
        ax   = axes[0, col]
        ax.bar(range(len(dose)), dose, color="#2d6a9f", alpha=0.85)
        ax.set_title(
            f"Best #{col+1}\nAct={s['target']:.3f} Pred={s['pred']:.3f}",
            fontsize=7,
        )
        ax.tick_params(labelsize=6)

    for col, idx in enumerate(worst_idx):
        s    = samples[idx]
        dose = s["past_target"].squeeze().numpy()
        ax   = axes[1, col]
        ax.bar(range(len(dose)), dose, color="#e87040", alpha=0.85)
        ax.set_title(
            f"Worst #{col+1}\nAct={s['target']:.3f} Pred={s['pred']:.3f}",
            fontsize=7,
        )
        ax.tick_params(labelsize=6)

    axes[0, 0].set_ylabel("Norm. Dose", fontsize=8)
    axes[1, 0].set_ylabel("Norm. Dose", fontsize=8)
    fig.suptitle(
        "NHiTS - Dose Profiles: Top-5 Best vs Top-5 Worst Predictions",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    pp = output_dir / "qualitative_review.png"
    fig.savefig(pp, dpi=150)
    plt.close(fig)
    print(f"  Qualitative plot -> {pp}")


# ---------------------------------------------------------------------------
# Step 11  JSON summary
# ---------------------------------------------------------------------------

def write_summary(
    metrics: dict[str, float],
    mean_saliency: np.ndarray,
    checkpoint_path: str,
    n_test: int,
    n_explain: int,
    output_dir: Path,
) -> None:
    """Write machine-readable JSON summary of the inference run."""
    top_k = int(np.argmax(mean_saliency))
    summary = {
        "checkpoint":        str(checkpoint_path),
        "n_test_samples":    n_test,
        "n_saliency_samples": n_explain,
        "metrics":           metrics,
        "saliency": {
            "mean_per_timestep":        mean_saliency.tolist(),
            "most_influential_timestep": top_k,
            "most_influential_score":    float(mean_saliency[top_k]),
        },
        "output_files": [
            "inference_predictions.csv",
            "inference_metrics.csv",
            "inference_forecast_vs_actual.png",
            "inference_residuals.png",
            "component_decomposition.csv",
            "component_decomposition.png",
            "saliency.csv",
            "saliency.png",
            "dose_sweep.csv",
            "dose_sweep.png",
            "qualitative_review.csv",
            "qualitative_review.png",
            "inference_summary.json",
        ],
    }
    jp = output_dir / "inference_summary.json"
    jp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"  Summary JSON -> {jp}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """End-to-end NHiTS inference and explainability pipeline."""
    args = parse_args()
    set_seed(args.seed)

    # Guide sec 1.2: device selection
    if   torch.cuda.is_available():          device_str = "cuda"
    elif torch.backends.mps.is_available():  device_str = "mps"
    else:                                    device_str = "cpu"
    device = torch.device(device_str)

    print("\n" + "=" * 60)
    print("  NHiTS Inference and Explainability Pipeline")
    print("=" * 60)
    print(f"  Device:     {device}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Data:       {args.data_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output dir: {output_dir.resolve()}")

    # Guide sec 1.1: config matches trained checkpoint
    cfg = CancerNHiTSConfig(
        data_path=Path(args.data_path),
        input_len=20,
        horizon=1,
        train_frac=0.70,
        val_frac=0.15,
        num_hist_covariates=2,
        num_future_covariates=0,
        pooling_sizes=list(args.pooling_sizes),
        n_blocks_per_stack=list(args.n_blocks_per_stack),
        hidden_size=args.hidden_size,
        num_layers_per_block=args.num_layers_per_block,
        dropout=args.dropout,
        use_layer_norm=not args.no_layer_norm,
        use_global_skip=not args.no_global_skip,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    # Guide sec 1.1: scaling parameters from training data only
    print("\n[1/11] Building data loaders (same split + scaling as training) ...")
    train_loader, _, test_loader = build_dataloaders(cfg)
    train_ds    = train_loader.dataset
    test_ds     = test_loader.dataset
    scale_stats = train_ds.scale_stats
    train_targets_np = train_ds.targets.cpu().numpy()
    print(f"  Test protocols : {len(test_ds):,}")
    print(
        f"  Scale stats    : target mean={scale_stats.target_mean:.4f} "
        f" std={scale_stats.target_std:.4f}"
    )

    # Guide sec 1.1 + 3.1: load checkpoint, model.eval() already called inside
    print(f"\n[2/11] Loading model from {args.checkpoint} ...")
    model = load_model(cfg, checkpoint_path=Path(args.checkpoint), device=device_str)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters : {n_params:,}")
    print(f"  use_global_skip      : {model.use_global_skip}")
    print(f"  NHiTS blocks         : {len(model.blocks)}")

    # Guide sec 2 + 3: input preparation + inference under torch.no_grad()
    print("\n[3/11] Test-set inference (input prep + forward pass) ...")
    preds, targets, samples = collect_test_predictions(model, test_loader, device)
    print(f"  Collected {len(preds):,} predictions")
    df_pred = pd.DataFrame({
        "protocol_idx": np.arange(len(preds)),
        "actual":       targets,
        "predicted":    preds,
        "residual":     targets - preds,
        "abs_error":    np.abs(targets - preds),
    })
    pred_csv = output_dir / "inference_predictions.csv"
    df_pred.to_csv(pred_csv, index=False)
    print(f"  Predictions CSV -> {pred_csv}")

    print("\n[4/11] Computing evaluation metrics ...")
    metrics = compute_and_print_metrics(preds, targets, train_targets_np, output_dir)

    print("\n[5/11] Scatter plot: predicted vs actual ...")
    plot_forecast_vs_actual(preds, targets, output_dir)

    print("\n[6/11] Residuals plot ...")
    plot_residuals(preds, targets, output_dir)

    n_exp = min(args.n_explain, len(samples))
    print(f"\n[7/11] Component decomposition (n={n_exp:,}) ...")
    analyse_components(
        model=model,
        samples=samples,
        n_samples=args.n_explain,
        scale_mean=scale_stats.target_mean,
        scale_std=scale_stats.target_std,
        output_dir=output_dir,
    )

    print(f"\n[8/11] Gradient x Input saliency (n={n_exp:,}) ...")
    mean_saliency = analyse_saliency(
        model=model,
        samples=samples,
        n_samples=args.n_explain,
        output_dir=output_dir,
    )

    print("\n[9/11] Dose-sweep sensitivity analysis ...")
    dose_sweep_sensitivity(
        model=model,
        reference_sample=samples[0],
        output_dir=output_dir,
    )

    print("\n[10/11] Qualitative review (top-5 best + worst predictions) ...")
    qualitative_review(samples=samples, output_dir=output_dir)

    print("\n[11/11] Writing JSON summary ...")
    write_summary(
        metrics=metrics,
        mean_saliency=mean_saliency,
        checkpoint_path=args.checkpoint,
        n_test=len(preds),
        n_explain=n_exp,
        output_dir=output_dir,
    )

    print("\n" + "=" * 60)
    print(f"  All outputs saved to: {output_dir.resolve()}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
