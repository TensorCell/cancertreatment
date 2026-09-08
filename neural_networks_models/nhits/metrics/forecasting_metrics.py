"""Forecasting evaluation metrics for NHiTS.

Re-exports all metrics functions from tide.metrics.forecasting_metrics for NHiTS.
"""

from tide.metrics.forecasting_metrics import (
    compute_all_metrics,
    mae,
    mape,
    margin_loss,
    mase,
    mse,
    percent_correct,
    r2_score,
    rmse,
    smape,
)

__all__ = [
    "compute_all_metrics",
    "mae",
    "mape",
    "margin_loss",
    "mase",
    "mse",
    "percent_correct",
    "r2_score",
    "rmse",
    "smape",
]
