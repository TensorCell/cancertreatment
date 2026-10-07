"""Evaluation metrics; ``evaluate`` is the one scoring function used everywhere."""

import numpy as np
import torch
from sklearn.utils import shuffle


def evaluate(y_true: np.array, y_pred: np.array) -> dict:
    """Score a whole split in cells. ``percent`` pairs every series with one other series
    of the split by a fixed permutation (random_state=0)."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    y_true_1, y_pred_1 = shuffle(y_true, y_pred, random_state=0)
    return {
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "percent": float(percent_correct(y_true, y_true_1, y_pred, y_pred_1)),
        "max_error_99": float(max_error_99(y_true, y_pred)),
    }


def percent_correct(
    y_true_1: np.array, y_true_2: np.array, y_pred_1: np.array, y_pred_2: np.array
) -> float:
    """
    Percentage of correctly ordered pairs i.e.
    where order of true labels is in line with
    order of predicted labels:

    Args:
    ---------------
        y_true_1 (np.array): true labels for sample 1
        y_true_2 (np.array): predictions for sample 1
        y_pred_1 (np.array): true labels for sample 2
        y_pred_2 (np.array): predictions for sample 2

    Returns:
    ---------------
        float: Percentage of correctly ordered pairs
        as fraction,  value
    """
    y_pred_1 = torch.Tensor(y_pred_1.flatten())
    y_pred_2 = torch.Tensor(y_pred_2.flatten())
    y_true_1 = torch.Tensor(y_true_1.flatten())
    y_true_2 = torch.Tensor(y_true_2.flatten())
    order_1 = (y_true_1 - y_true_2).sign()
    order_2 = (y_pred_1 - y_pred_2).sign()
    percent = (order_1 == order_2).sum() / y_true_1.size()[0]
    return percent.item()


def mean_absolute_percentage_error(y_true: np.array, y_pred: np.array) -> float:
    """MAPE metric

    Args:
    ---------------
        y_true (np.array): true labels
        y_pred (np.array): predicted labels

    Returns:
    ---------------
        float: MAPE value
    """
    return np.mean(np.abs((y_true - y_pred) / np.abs(y_true)))


def max_error_99(y_true: np.array, y_pred: np.array) -> float:
    """99 percentile MAE error

    Args:
    ---------------
        y_true (np.array): true labels
        y_pred (np.array): predicted labels

    Returns:
    ---------------
        float: 99 percentile MAE error value
    """
    return np.max(np.sort(np.abs(y_true - y_pred))[: int(y_true.shape[0] * 0.99)])
