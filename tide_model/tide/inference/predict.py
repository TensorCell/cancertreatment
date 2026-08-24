"""Inference utilities for the TiDE cancer treatment forecasting model.

Functions
---------
load_model
    Load a ``TiDEModel`` from a saved checkpoint.
predict
    Run a single forward pass with ``torch.no_grad()``.
predict_tumour_cells
    Convenience wrapper: given a dose schedule and time features,
    return the predicted normalised tumour cell count.
explain_forecast_components
    Decompose a forecast into its linear skip baseline and the deep
    nonlinear adjustment added by the encoder/decoder stacks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from tide.config import CancerTiDEConfig
from tide.models.tide import TiDEModel


def load_model(
    cfg: CancerTiDEConfig,
    checkpoint_path: Path | str | None = None,
    device: str | torch.device = "cpu",
) -> TiDEModel:
    """Load a TiDEModel from a checkpoint file.

    Parameters
    ----------
    cfg:
        Configuration used to construct the model architecture.
    checkpoint_path:
        Path to the ``.pt`` file.  Defaults to
        ``cfg.checkpoint_dir / "best_model.pt"``.
    device:
        Target device for inference.

    Returns
    -------
    TiDEModel
        Model in eval mode on *device*.
    """
    if checkpoint_path is None:
        checkpoint_path = cfg.checkpoint_dir / "best_model.pt"

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = TiDEModel.from_config(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict(
    model: TiDEModel,
    past_target: Tensor,   # (B, L)
    hist_covs: Tensor,     # (B, L, C_hist)
    future_covs: Tensor,   # (B, H, C_fut)
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Run a forward pass and return predictions as a numpy array.

    Parameters
    ----------
    model:
        Loaded ``TiDEModel`` in eval mode.
    past_target:
        Scaled dose sequence tensor, shape ``(B, L)``.
    hist_covs:
        Historical covariates, shape ``(B, L, C_hist)``.
    future_covs:
        Future covariates for the horizon, shape ``(B, H, C_fut)``.
        For this dataset this is an empty tensor (``C_fut == 0``).
    device:
        Device to run inference on.

    Returns
    -------
    np.ndarray
        Predictions in normalised space, shape ``(B, H)``.
    """
    device = torch.device(device)
    pred = model(
        past_target.to(device),
        hist_covs.to(device),
        future_covs.to(device),
    )
    return pred.cpu().numpy()


def predict_tumour_cells(
    model: TiDEModel,
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
    """Predict the normalised tumour cell count for a single protocol.

    This convenience function handles MinMax scaling so the caller can
    work in the original units (Gy for dose, seconds for time).

    Parameters
    ----------
    model:
        Loaded ``TiDEModel`` in eval mode.
    dose_sequence:
        Raw dose values for each of the 20 schedule steps, shape ``(L,)``.
        Zero-pad shorter protocols at the beginning.
    time_sequence:
        Absolute time of each dose in seconds, shape ``(L,)``.
    time_gap_sequence:
        Time gap since the previous dose for each step, shape ``(L,)``.
    dose_min, dose_max:
        MinMax statistics fitted on the training set (from
        ``ScaleStats.dose_min / dose_max``).
    time_min, time_max:
        MinMax statistics for the ``time`` feature.
    time_gap_min, time_gap_max:
        MinMax statistics for the ``time_gap`` feature.
    device:
        Device to run inference on.
    eps:
        Small constant to avoid division by zero for constant features.

    Returns
    -------
    float
        Predicted normalised average tumour cell count.
    """
    # ── MinMax scale inputs ────────────────────────────────────────────────
    scaled_dose = (dose_sequence - dose_min) / (dose_max - dose_min + eps)
    scaled_time = (time_sequence - time_min) / (time_max - time_min + eps)
    scaled_gap = (time_gap_sequence - time_gap_min) / (time_gap_max - time_gap_min + eps)

    # ── Build tensors with batch dimension = 1 ─────────────────────────────
    past_t = torch.tensor(scaled_dose, dtype=torch.float32).unsqueeze(0)  # (1, L)

    cov_np = np.stack([scaled_time, scaled_gap], axis=-1)                 # (L, 2)
    hist_c = torch.tensor(cov_np, dtype=torch.float32).unsqueeze(0)       # (1, L, 2)

    # Horizon=1, C_fut=0 → empty future covariates tensor
    horizon = model.horizon
    fut_c = torch.zeros(1, horizon, 0, dtype=torch.float32)               # (1, 1, 0)

    # ── Inference ──────────────────────────────────────────────────────────
    pred_scaled = predict(model, past_t, hist_c, fut_c, device)           # (1, 1)
    return float(pred_scaled[0, 0])


@torch.no_grad()
def explain_forecast_components(
    model: TiDEModel,
    past_t: torch.Tensor,   # (1, L) — scaled dose sequence
    hist_c: torch.Tensor,   # (1, L, C_hist) — scaled covariates
    fut_c: torch.Tensor,    # (1, H, C_fut)  — future covariates (empty here)
    scale_mean: float,
    scale_std: float,
) -> dict[str, np.ndarray]:
    """Decompose a TiDE forecast into linear skip baseline and deep nonlinear adjustment.

    TiDE's forward pass is architecturally additive (see ``TiDEModel.forward``)::

        output = temporal_decoder(decoder(encoder(feature_proj(inputs))))
                 + skip(past_target)
                         ↑
                 nn.Linear(input_len, horizon) — one learned weight per dose step

    This function separates those two additive terms and inverse-scales them
    back to the original tumour-cell-count units, making it possible to
    quantify how much of the prediction each pathway contributes.

    Parameters
    ----------
    model:
        ``TiDEModel`` in eval mode (e.g. loaded via ``load_model``).
    past_t:
        Scaled dose sequence, shape ``(1, L)``.  Same tensor you pass to
        ``model.forward``.
    hist_c:
        Scaled historical covariates, shape ``(1, L, C_hist)``.
    fut_c:
        Future covariates tensor, shape ``(1, H, C_fut)``.  For this
        dataset this is an empty tensor with ``C_fut == 0``.
    scale_mean:
        Mean used when normalising the target.  Add this back to recover
        absolute tumour cell counts from the normalised skip baseline.
    scale_std:
        Standard deviation used when normalising the target.  The deep
        adjustment is mean-zero by nature so it is scaled by ``scale_std``
        only (no mean shift).

    Returns
    -------
    dict[str, np.ndarray]
        Three arrays, each of shape ``(H,)`` in original (unscaled) units:

        ``"total_forecast"``
            Full model output inverse-scaled to tumour cell count units.
            Equals ``linear_skip_baseline + deep_nonlinear_adjustment``.

        ``"linear_skip_baseline"``
            Contribution of the global skip connection — a **learned linear
            combination** of the 20 scaled dose values projected directly to
            the horizon.  This is equivalent to a weighted dose-history
            average.  Captures the broad, dose-magnitude-driven trend in
            tumour survival.

        ``"deep_nonlinear_adjustment"``
            Everything the encoder/decoder stacks add on top of the linear
            baseline.  This term encodes nonlinear interactions between dose
            timing (``time``, ``time_gap`` covariates), dose sequencing, and
            their joint effect on cell-kill.  A large adjustment relative to
            the baseline indicates that protocol *shape* matters more than
            total dose.

    Notes
    -----
    Inverse-scaling convention
        The target was normalised as ``y_scaled = (y_raw - mean) / std``.
        Inverse scaling reverses this:

        - ``total_forecast``  : ``y_scaled * std + mean``
        - ``linear_skip``     : ``skip_scaled * std + mean``  (has a mean offset)
        - ``deep_adjustment`` : ``deep_scaled * std``          (zero-mean deviation)

        This ensures ``total = linear_skip_baseline + deep_nonlinear_adjustment``
        holds exactly in the returned arrays.

    Example
    -------
    ::

        from tide.inference.predict import load_model, explain_forecast_components

        model = load_model(cfg, device="cpu")
        components = explain_forecast_components(
            model, past_t, hist_c, fut_c,
            scale_mean=scale_stats.target_mean,
            scale_std=scale_stats.target_std,
        )
        print("Total forecast:       ", components["total_forecast"])
        print("Linear skip baseline: ", components["linear_skip_baseline"])
        print("Deep adjustment:      ", components["deep_nonlinear_adjustment"])
    """
    model.eval()
    device = next(model.parameters()).device
    past_t = past_t.to(device)
    hist_c = hist_c.to(device)
    fut_c = fut_c.to(device)

    # ── Linear skip: nn.Linear(input_len, horizon) applied to doses only ──
    # shape: (1, H) → squeeze → (H,)
    skip_scaled: np.ndarray = model.skip(past_t).cpu().numpy().squeeze(0)

    # ── Full forward pass ──────────────────────────────────────────────────
    full_scaled: np.ndarray = model(past_t, hist_c, fut_c).cpu().numpy().squeeze(0)

    # ── Deep residual = total − skip (both still in normalised space) ──────
    deep_scaled: np.ndarray = full_scaled - skip_scaled

    # ── Inverse-scale to original tumour-cell-count units ──────────────────
    # skip retains the mean offset; deep is a zero-centred deviation
    skip_raw  = skip_scaled * scale_std + scale_mean   # (H,)
    deep_raw  = deep_scaled * scale_std                 # (H,) — no mean shift
    total_raw = full_scaled * scale_std + scale_mean    # (H,)

    return {
        "total_forecast":            total_raw,
        "linear_skip_baseline":      skip_raw,
        "deep_nonlinear_adjustment": deep_raw,
    }


def explain_input_saliency(
    model: TiDEModel,
    past_t: torch.Tensor,   # (1, L)
    hist_c: torch.Tensor,   # (1, L, C_hist)
    fut_c: torch.Tensor,    # (1, H, C_fut)
    target_horizon_step: int = 0,
) -> np.ndarray:
    """Compute gradient-based attribution (Gradient × Input saliency) for a target horizon step.

    This function measures the sensitivity of a specific forecast horizon step
    with respect to each look-back dose timestep.  Attribution is computed as:

        saliency_i = |∂(pred) / ∂(past_target_i)| × |past_target_i|

    The resulting vector is normalised so that all elements sum to 1.

    Parameters
    ----------
    model:
        Loaded ``TiDEModel`` instance.
    past_t:
        Scaled dose sequence tensor, shape ``(1, L)``.
    hist_c:
        Scaled historical covariates tensor, shape ``(1, L, C_hist)``.
    fut_c:
        Future covariates tensor, shape ``(1, H, C_fut)``.  For this dataset,
        ``C_fut == 0``.
    target_horizon_step:
        Index of the horizon step to compute saliency for (default: 0).

    Returns
    -------
    np.ndarray
        Normalised saliency scores of shape ``(L,)`` where entries sum to 1.0.

    Raises
    ------
    RuntimeError
        If gradient computation fails or yields ``None``.
    """
    model.eval()
    device = next(model.parameters()).device

    # Clone tensor and enable gradients for past_t
    past_t_grad = past_t.clone().detach().to(device).requires_grad_(True)
    hist_c = hist_c.to(device)
    fut_c = fut_c.to(device)

    # Forward pass with autograd enabled
    with torch.enable_grad():
        preds = model(past_t_grad, hist_c, fut_c)
        target_val = preds[0, target_horizon_step]

        # Compute gradients wrt past_t
        target_val.backward()

    if past_t_grad.grad is None:
        raise RuntimeError("Gradient calculation failed: past_t_grad.grad is None.")

    # Gradient × Input saliency: |grad| * |input|
    raw_saliency = (past_t_grad.grad.abs() * past_t_grad.abs()).detach().cpu().numpy().squeeze(0)

    # Normalise saliency vector so elements sum to 1
    saliency = raw_saliency / (raw_saliency.sum() + 1e-8)
    return saliency
