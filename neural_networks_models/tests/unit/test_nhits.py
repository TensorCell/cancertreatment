"""Unit test suite for NHiTS cancer treatment forecasting model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from nhits.config import CancerNHiTSConfig, NHiTSConfig
from nhits.evaluation.evaluate import collect_predictions
from nhits.inference.predict import load_model, predict, predict_tumour_cells
from nhits.metrics.forecasting_metrics import compute_all_metrics
from nhits.models.nhits import NHiTSBlock, NHiTSModel
from nhits.trainer.trainer import EarlyStopping, Trainer


def test_config_validation() -> None:
    """Test NHiTSConfig default values and post-init validation."""
    cfg = CancerNHiTSConfig()
    assert cfg.input_len == 20
    assert cfg.horizon == 1
    assert cfg.pooling_sizes == [4, 2, 1]
    assert cfg.n_blocks_per_stack == [1, 1, 1]
    assert cfg.test_frac == pytest.approx(0.15)

    with pytest.raises(ValueError, match="train_frac \\+ val_frac must be < 1.0"):
        CancerNHiTSConfig(train_frac=0.8, val_frac=0.3)

    with pytest.raises(ValueError, match="Unknown loss"):
        CancerNHiTSConfig(loss="invalid_loss")

    with pytest.raises(ValueError, match="Unknown activation"):
        CancerNHiTSConfig(activation="invalid_activation")

    with pytest.raises(ValueError, match="same length"):
        CancerNHiTSConfig(pooling_sizes=[4, 2], n_blocks_per_stack=[1])


@pytest.mark.parametrize("act", ["relu", "gelu"])
def test_nhits_block_forward(act: str) -> None:
    """Test single NHiTSBlock forward pass shape with different activations."""
    block = NHiTSBlock(
        input_len=20,
        horizon=1,
        in_channels=3,
        pooling_size=4,
        hidden_size=64,
        num_layers=2,
        activation=act,
    )

    B = 4
    x = torch.randn(B, 20, 3)
    backcast, forecast = block(x)

    assert backcast.shape == (B, 20, 3)
    assert forecast.shape == (B, 1)


@pytest.mark.parametrize("act", ["relu", "gelu"])
def test_nhits_model_forward_and_gradient(act: str) -> None:
    """Test full NHiTSModel forward pass and gradient flow with different activations."""
    model = NHiTSModel(
        input_len=20,
        horizon=1,
        num_hist_covariates=2,
        pooling_sizes=[4, 2, 1],
        n_blocks_per_stack=[1, 1, 1],
        hidden_size=64,
        num_layers_per_block=2,
        activation=act,
    )

    B = 4
    past_target = torch.randn(B, 20)
    hist_covs = torch.randn(B, 20, 2)
    future_covs = torch.zeros(B, 1, 0)
    target = torch.randn(B, 1)

    pred = model(past_target, hist_covs, future_covs)
    assert pred.shape == (B, 1)

    loss = torch.nn.functional.l1_loss(pred, target)
    loss.backward()

    for p in model.parameters():
        assert p.grad is not None


def test_metrics_computation() -> None:
    """Test forecasting metrics calculation."""
    y_true = np.array([0.1, 0.2, 0.3, 0.4])
    y_pred = np.array([0.11, 0.19, 0.31, 0.39])
    y_train = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45] * 5)

    metrics = compute_all_metrics(y_true, y_pred, y_train=y_train, seasonality=1)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "mape" in metrics
    assert "smape" in metrics
    assert "r2" in metrics
    assert "mase" in metrics
    assert metrics["mae"] == pytest.approx(0.01)


def test_early_stopping() -> None:
    """Test early stopping logic."""
    es = EarlyStopping(patience=2)
    assert not es.step(1.0)
    assert not es.step(0.9)
    assert not es.step(0.95)
    assert es.step(0.96)
    assert es.best_loss == pytest.approx(0.9)
