"""Inference and explainability utilities for the NHiTS cancer treatment forecasting model.

Functions
---------
load_model
    Load an ``NHiTSModel`` from a saved checkpoint.
predict
    Run a single forward pass with ``torch.no_grad()``.
predict_tumour_cells
    Convenience wrapper: given a dose schedule and time features,
    return the predicted normalised tumour cell count.
explain_forecast_components
    Decompose an NHiTS forecast into stack/block contributions and global skip connection.
explain_input_saliency
    Compute gradient-based attribution for look-back timesteps.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from nhits.config import CancerNHiTSConfig
from nhits.models.nhits import NHiTSModel


def load_model(
    cfg: CancerNHiTSConfig,
    checkpoint_path: Path | str | None = None,
    device: str | torch.device = "cpu",
) -> NHiTSModel:
    """Load an NHiTSModel from a checkpoint file."""
    if checkpoint_path is None:
        checkpoint_path = cfg.checkpoint_dir / "best_model.pt"

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt["model_state_dict"]

    # Detect architecture hyperparameters from state_dict to avoid mismatch
    linear_layer_count = sum(
        1 for k, v in state_dict.items()
        if k.startswith("blocks.0.mlp.") and v.ndim == 2
    )
    if linear_layer_count > 0 and linear_layer_count != cfg.num_layers_per_block:
        cfg.num_layers_per_block = linear_layer_count

    if "blocks.0.mlp.0.weight" in state_dict:
        detected_hidden = state_dict["blocks.0.mlp.0.weight"].shape[0]
        if detected_hidden != cfg.hidden_size:
            cfg.hidden_size = detected_hidden

    has_skip = any(k.startswith("skip.") for k in state_dict)
    if has_skip != cfg.use_global_skip:
        cfg.use_global_skip = has_skip

    model = NHiTSModel.from_config(cfg)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict(
    model: NHiTSModel,
    past_target: Tensor,   # (B, L)
    hist_covs: Tensor,     # (B, L, C_hist)
    future_covs: Tensor,   # (B, H, C_fut)
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Run a forward pass and return predictions as a numpy array."""
    device = torch.device(device)
    pred = model(
        past_target.to(device),
        hist_covs.to(device),
        future_covs.to(device),
    )
    return pred.cpu().numpy()


def predict_tumour_cells(
    model: NHiTSModel,
    dose_sequence: np.ndarray,       # (L,) raw dose values
    time_sequence: np.ndarray,        # (L,) raw time values (seconds)
    time_gap_sequence: np.ndarray,    # (L,) raw time_gap values (seconds)
    dose_min: float,
    dose_max: float,
    time_min: float,
    time_max: float,
    time_gap_min: float,
    time_gap_max: float,
    device: str | torch.device = "cpu",
    eps: float = 1e-8,
) -> float:
    """Predict the normalised tumour cell count for a single protocol."""
    scaled_dose = (dose_sequence - dose_min) / (dose_max - dose_min + eps)
    scaled_time = (time_sequence - time_min) / (time_max - time_min + eps)
    scaled_gap = (time_gap_sequence - time_gap_min) / (time_gap_max - time_gap_min + eps)

    past_t = torch.tensor(scaled_dose, dtype=torch.float32).unsqueeze(0)  # (1, L)

    cov_np = np.stack([scaled_time, scaled_gap], axis=-1)                 # (L, 2)
    hist_c = torch.tensor(cov_np, dtype=torch.float32).unsqueeze(0)       # (1, L, 2)

    horizon = model.horizon
    fut_c = torch.zeros(1, horizon, 0, dtype=torch.float32)               # (1, 1, 0)

    pred_scaled = predict(model, past_t, hist_c, fut_c, device)           # (1, 1)
    return float(pred_scaled[0, 0])


@torch.no_grad()
def explain_forecast_components(
    model: NHiTSModel,
    past_t: torch.Tensor,   # (1, L) — scaled dose sequence
    hist_c: torch.Tensor,   # (1, L, C_hist) — scaled covariates
    fut_c: torch.Tensor,    # (1, H, C_fut) — future covariates
    scale_mean: float,
    scale_std: float,
) -> dict[str, np.ndarray]:
    """Decompose an NHiTS forecast into linear skip baseline and stack/block contributions."""
    model.eval()
    device = next(model.parameters()).device
    past_t = past_t.to(device)
    hist_c = hist_c.to(device)
    fut_c = fut_c.to(device)

    # 1. Linear skip (if enabled)
    skip_raw: np.ndarray | None = None
    if model.use_global_skip:
        skip_scaled = model.skip(past_t).cpu().numpy().squeeze()
        skip_raw = skip_scaled * scale_std + scale_mean

    # 2. Block forecasts
    block_forecasts = model.get_block_forecasts(past_t, hist_c, fut_c)
    full_scaled = model(past_t, hist_c, fut_c).cpu().numpy().squeeze()
    total_raw = full_scaled * scale_std + scale_mean

    # Deep adjustment is the sum of block forecasts (excluding global skip if present)
    if model.use_global_skip:
        deep_blocks_scaled = sum(bf.cpu().numpy().squeeze() for bf in block_forecasts[:-1])
    else:
        deep_blocks_scaled = sum(bf.cpu().numpy().squeeze() for bf in block_forecasts)

    deep_raw = deep_blocks_scaled * scale_std

    results: dict[str, np.ndarray] = {
        "total_forecast": total_raw,
        "deep_nonlinear_adjustment": deep_raw,
    }
    if skip_raw is not None:
        results["linear_skip_baseline"] = skip_raw

    return results


def explain_input_saliency(
    model: NHiTSModel,
    past_t: torch.Tensor,   # (1, L)
    hist_c: torch.Tensor,   # (1, L, C_hist)
    fut_c: torch.Tensor,    # (1, H, C_fut)
    target_horizon_step: int = 0,
) -> np.ndarray:
    """Compute gradient-based attribution (Gradient × Input saliency) for a target horizon step."""
    model.eval()
    device = next(model.parameters()).device

    past_t_grad = past_t.clone().detach().to(device).requires_grad_(True)
    hist_c = hist_c.to(device)
    fut_c = fut_c.to(device)

    with torch.enable_grad():
        preds = model(past_t_grad, hist_c, fut_c)
        target_val = preds[0, target_horizon_step]
        target_val.backward()

    if past_t_grad.grad is None:
        raise RuntimeError("Gradient calculation failed: past_t_grad.grad is None.")

    raw_saliency = (past_t_grad.grad.abs() * past_t_grad.abs()).detach().cpu().numpy().squeeze(0)
    saliency = raw_saliency / (raw_saliency.sum() + 1e-8)
    return saliency
