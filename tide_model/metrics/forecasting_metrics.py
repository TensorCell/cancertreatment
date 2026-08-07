"""Forecasting evaluation metrics.

All functions operate on plain numpy arrays and return scalar floats.
Use ``compute_all_metrics`` to get a dict of every metric at once.

For the cancer treatment dataset the target is a single normalised scalar
(average tumour cell count), so MASE uses seasonality=1 (no seasonality).

Pair-ranking metrics (MRL, % correct)
--------------------------------------
Let P(y) denote a randomly permuted copy of y for a fixed permutation.
Both ``margin_loss`` and ``percent_correct`` receive

    (y_true_1=y,  y_true_2=P(y),  y_pred_1=ŷ,  y_pred_2=P(ŷ))

so they measure whether the model preserves the ranking of the original
target pairs.
"""

from __future__ import annotations

import numpy as np
import torch


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%).

    Parameters
    ----------
    eps:
        Small constant added to the denominator to avoid division by zero.
    """
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Symmetric Mean Absolute Percentage Error (%).

    sMAPE is bounded in [0, 200] and avoids the asymmetry of MAPE.
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + eps
    return float(np.mean(numerator / denominator) * 100)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination R²."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-8
    return float(1.0 - ss_res / ss_tot)


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1,
) -> float:
    """Mean Absolute Scaled Error.

    Scales MAE by the in-sample naive seasonal forecast error, making
    the metric unit-free and comparable across series.

    Parameters
    ----------
    y_train:
        Training target values used to compute the scale (naive forecast).
    seasonality:
        Seasonal lag for the naive baseline.  Default 1 (no seasonality)
        since cancer protocols are independent samples, not a time series.
    """
    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = np.mean(naive_errors) + 1e-8
    return float(mae(y_true, y_pred) / scale)



def margin_loss(
    y_true_1: np.ndarray,
    y_true_2: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
) -> float:
    """Margin Ranking Loss (MRL).

    Based on https://github.com/AWarno/CancerDLOptimization/blob/main/cancer_nn/metrics.py

    Measures whether the model predicts the correct ordering of pairs.
    Reference: https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html

    Parameters
    ----------
    y_true_1:
        True labels for sample 1.
    y_true_2:
        True labels for sample 2.
    y_pred_1:
        Predicted values for sample 1.
    y_pred_2:
        Predicted values for sample 2.

    Returns
    -------
    float
        MRL value (non-negative; lower is better).
    """
    t_pred_1 = torch.as_tensor(y_pred_1.ravel(), dtype=torch.float32)
    t_pred_2 = torch.as_tensor(y_pred_2.ravel(), dtype=torch.float32)
    t_true_1 = torch.as_tensor(y_true_1.ravel(), dtype=torch.float32)
    t_true_2 = torch.as_tensor(y_true_2.ravel(), dtype=torch.float32)
    order = (t_true_1 - t_true_2).sign()
    loss_fn = torch.nn.MarginRankingLoss()
    return loss_fn(t_pred_1, t_pred_2, order).item()


def percent_correct(
    y_true_1: np.ndarray,
    y_true_2: np.ndarray,
    y_pred_1: np.ndarray,
    y_pred_2: np.ndarray,
) -> float:
    """Percentage of correctly ordered pairs.

    Based on https://github.com/AWarno/CancerDLOptimization/blob/main/cancer_nn/metrics.py

    A pair is *correctly ordered* when the sign of (ŷᵢ − P(ŷ)ᵢ) matches
    the sign of (yᵢ − P(y)ᵢ), i.e.

        (yᵢ ≤ P(y)ᵢ ∧ ŷᵢ ≤ P(ŷ)ᵢ) ∨ (yᵢ ≥ P(y)ᵢ ∧ ŷᵢ ≥ P(ŷ)ᵢ)

    Parameters
    ----------
    y_true_1:
        True labels for sample 1.
    y_true_2:
        True labels for sample 2.
    y_pred_1:
        Predicted values for sample 1.
    y_pred_2:
        Predicted values for sample 2.

    Returns
    -------
    float
        Fraction in [0, 1]; higher is better.
    """
    t_pred_1 = torch.as_tensor(y_pred_1.ravel(), dtype=torch.float32)
    t_pred_2 = torch.as_tensor(y_pred_2.ravel(), dtype=torch.float32)
    t_true_1 = torch.as_tensor(y_true_1.ravel(), dtype=torch.float32)
    t_true_2 = torch.as_tensor(y_true_2.ravel(), dtype=torch.float32)
    order_true = (t_true_1 - t_true_2).sign()
    order_pred = (t_pred_1 - t_pred_2).sign()
    return float(((order_true == order_pred).sum() / t_true_1.size(0)).item()*100)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray | None = None,
    seasonality: int = 1,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Compute all forecasting metrics and return as a dict.

    Parameters
    ----------
    y_true:
        Ground-truth values, shape ``(N,)`` or ``(N, H)``.
    y_pred:
        Predicted values, same shape as *y_true*.
    y_train:
        Training targets required for MASE; if ``None`` MASE is omitted.
    seasonality:
        Seasonal period passed to ``mase``.  Defaults to 1.
    rng:
        Optional NumPy random generator used to produce the fixed
        permutation P(·) for the pair-ranking metrics.  When ``None``
        a default generator with seed 0 is used so results are
        reproducible across runs.

    Returns
    -------
    dict[str, float]
        Keys: ``mae``, ``rmse``, ``mse``, ``mape``, ``smape``, ``r2``,
        ``mrl``, ``pct_correct``, and optionally ``mase``.
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    metrics: dict[str, float] = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mse": mse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

    if y_train is not None:
        metrics["mase"] = mase(y_true, y_pred, y_train.ravel(), seasonality)

    # Pair-ranking metrics — build one fixed random permutation P(·)
    _rng = rng if rng is not None else np.random.default_rng(0)
    perm = _rng.permutation(len(y_true))
    y_true_perm = y_true[perm]
    y_pred_perm = y_pred[perm]

    metrics["mrl"] = margin_loss(y_true, y_true_perm, y_pred, y_pred_perm)
    metrics["pct_correct"] = percent_correct(y_true, y_true_perm, y_pred, y_pred_perm)

    return metrics
