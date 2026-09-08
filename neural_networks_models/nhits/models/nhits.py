"""NHiTS: Neural Hierarchical Interpolation for Time Series Forecasting in PyTorch.

Architecture (Challu et al., AAAI 2023 — "N-BEATSx / N-HiTS"):
  1. Multi-rate Input Subsampling  — MaxPool1d temporal downsampling across stacks
  2. Dense MLP Block Stack         — Stacked FC layers with GELU, Dropout & LayerNorm
  3. Backcast & Forecast Synthesis  — Linear projections & interpolation
  4. Doubly Residual Connections   — Backcast residual subtraction + forecast accumulation
  5. (Optional) Global Linear Skip — Direct linear path from past_target to forecast

Designed for single-step/multi-step multivariate forecasting with static/historical covariates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from nhits.config import CancerNHiTSConfig


# ---------------------------------------------------------------------------
# NHiTS Block
# ---------------------------------------------------------------------------

class NHiTSBlock(nn.Module):
    """Single NHiTS block with multi-rate pooling, MLP, and backcast/forecast synthesis.

    Parameters
    ----------
    input_len:
        Look-back sequence length L (e.g. 20).
    horizon:
        Forecast horizon H (e.g. 1).
    in_channels:
        Number of input channels per timestep (e.g. 3 = 1 target + 2 hist_covs).
    pooling_size:
        Downsampling kernel size k for temporal pooling.
    hidden_size:
        Hidden layer dimension for block MLP.
    num_layers:
        Number of dense layers in block MLP.
    dropout:
        Dropout probability.
    use_layer_norm:
        Whether to apply LayerNorm after dense layers.
    """

    def __init__(
        self,
        input_len: int,
        horizon: int,
        in_channels: int,
        pooling_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.input_len = input_len
        self.horizon = horizon
        self.in_channels = in_channels
        self.pooling_size = pooling_size
        self.activation = activation

        # Temporal pooling (downsampling)
        if pooling_size > 1:
            self.pooling = nn.MaxPool1d(
                kernel_size=pooling_size,
                stride=pooling_size,
                ceil_mode=True,
            )
        else:
            self.pooling = nn.Identity()

        # Calculate pooled temporal length
        pooled_len = max(1, (input_len + pooling_size - 1) // pooling_size) if pooling_size > 1 else input_len
        self.pooled_len = pooled_len

        flat_in_dim = in_channels * pooled_len

        # Activation layer factory
        act_lower = activation.lower()
        if act_lower == "relu":
            act_fn = nn.ReLU()
        elif act_lower == "gelu":
            act_fn = nn.GELU()
        else:
            raise ValueError(f"Unknown activation '{activation}'. Choose relu | gelu.")

        # MLP stack
        mlp_layers: list[nn.Module] = []
        curr_dim = flat_in_dim
        for _ in range(num_layers):
            mlp_layers.append(nn.Linear(curr_dim, hidden_size))
            if use_layer_norm:
                mlp_layers.append(nn.LayerNorm(hidden_size))
            mlp_layers.append(act_fn)
            if dropout > 0.0:
                mlp_layers.append(nn.Dropout(dropout))
            curr_dim = hidden_size

        self.mlp = nn.Sequential(*mlp_layers)

        # Backcast projection: maps hidden state -> (in_channels * pooled_len)
        self.backcast_proj = nn.Linear(hidden_size, flat_in_dim)

        # Forecast projection: maps hidden state -> horizon H
        self.forecast_proj = nn.Linear(hidden_size, horizon)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass for a single NHiTS block.

        Parameters
        ----------
        x:
            Input sequence tensor of shape ``(B, L, C_in)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            - ``backcast``: Shape ``(B, L, C_in)`` (reconstructed input for residual update)
            - ``forecast``: Shape ``(B, H)`` (block forecast contribution)
        """
        B, L, C = x.shape

        # 1. Temporal pooling: (B, L, C) -> (B, C, L) -> pool -> (B, C, L_pooled)
        x_perm = x.permute(0, 2, 1)
        x_pooled = self.pooling(x_perm)
        L_pooled = x_pooled.size(-1)

        # 2. Flatten for MLP: (B, C * L_pooled)
        x_flat = x_pooled.reshape(B, -1)

        # 3. MLP feature extraction
        h = self.mlp(x_flat)

        # 4. Backcast synthesis
        theta_b = self.backcast_proj(h)                     # (B, C * L_pooled)
        backcast_pooled = theta_b.view(B, C, L_pooled)      # (B, C, L_pooled)

        # Interpolate backcast to full sequence length L if downsampled
        if L_pooled != L:
            backcast_perm = F.interpolate(
                backcast_pooled, size=L, mode="linear", align_corners=False
            )
        else:
            backcast_perm = backcast_pooled

        backcast = backcast_perm.permute(0, 2, 1)            # (B, L, C)

        # 5. Forecast synthesis
        forecast = self.forecast_proj(h)                     # (B, H)

        return backcast, forecast


# ---------------------------------------------------------------------------
# NHiTS Model
# ---------------------------------------------------------------------------

class NHiTSModel(nn.Module):
    """NHiTS: Neural Hierarchical Interpolation model for time-series forecasting.

    Full pipeline:
      Inputs [past_target + hist_covs] -> Stack of NHiTSBlocks -> Sum(Forecasts) + Skip

    Parameters
    ----------
    input_len:
        Look-back window length (dose steps). Default 20.
    horizon:
        Forecast horizon. Default 1.
    num_hist_covariates:
        Number of historical covariate channels (2: time, time_gap).
    pooling_sizes:
        List of pooling sizes for each stack (e.g. [4, 2, 1]).
    n_blocks_per_stack:
        List of block counts per stack (e.g. [1, 1, 1]).
    hidden_size:
        Width of dense layers inside NHiTS blocks.
    num_layers_per_block:
        Number of dense layers per block MLP.
    dropout:
        Dropout probability.
    use_layer_norm:
        Whether to use LayerNorm in blocks.
    use_global_skip:
        Whether to include a linear skip connection from past_target to forecast.
    """

    def __init__(
        self,
        input_len: int = 20,
        horizon: int = 1,
        num_hist_covariates: int = 2,
        pooling_sizes: list[int] | None = None,
        n_blocks_per_stack: list[int] | None = None,
        hidden_size: int = 64,
        num_layers_per_block: int = 2,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
        use_global_skip: bool = True,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.input_len = input_len
        self.horizon = horizon
        self.num_hist_covariates = num_hist_covariates
        self.activation = activation

        pooling_sizes = pooling_sizes or [4, 2, 1]
        n_blocks_per_stack = n_blocks_per_stack or [1, 1, 1]

        if len(pooling_sizes) != len(n_blocks_per_stack):
            raise ValueError(
                "pooling_sizes and n_blocks_per_stack must have equal length."
            )

        # Input channels = 1 (past target/dose) + num_hist_covariates
        self.in_channels = 1 + num_hist_covariates

        # Build stack of blocks
        self.blocks = nn.ModuleList()
        for pool_size, n_blocks in zip(pooling_sizes, n_blocks_per_stack, strict=True):
            for _ in range(n_blocks):
                self.blocks.append(
                    NHiTSBlock(
                        input_len=input_len,
                        horizon=horizon,
                        in_channels=self.in_channels,
                        pooling_size=pool_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers_per_block,
                        dropout=dropout,
                        use_layer_norm=use_layer_norm,
                        activation=activation,
                    )
                )

        # Global linear skip connection (past_target -> forecast)
        self.use_global_skip = use_global_skip
        if use_global_skip:
            self.skip = nn.Linear(input_len, horizon, bias=True)

    def forward(
        self,
        past_target: Tensor,   # (B, L)
        hist_covs: Tensor,     # (B, L, C_hist)
        future_covs: Tensor,   # (B, H, C_fut) — empty/unused for H=1
    ) -> Tensor:
        """Run the full NHiTS forward pass.

        Parameters
        ----------
        past_target:
            Scaled dose sequence, shape ``(B, L)``.
        hist_covs:
            Historical covariates [time, time_gap], shape ``(B, L, C_hist)``.
        future_covs:
            Future covariates, shape ``(B, H, C_fut)`` (empty tensor here).

        Returns
        -------
        Tensor
            Forecast of shape ``(B, H)`` — predicted normalised tumour cells.
        """
        B, L = past_target.shape

        # Concatenate past_target and hist_covs to create input tensor: (B, L, 1+C_hist)
        x = torch.cat([past_target.unsqueeze(-1), hist_covs], dim=-1)  # (B, L, C_in)

        # Doubly residual loop
        residuals = x
        forecast_acc = torch.zeros((B, self.horizon), device=past_target.device, dtype=past_target.dtype)

        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = residuals - backcast
            forecast_acc = forecast_acc + block_forecast

        # Global linear skip connection
        if self.use_global_skip:
            skip_out = self.skip(past_target)
            forecast_acc = forecast_acc + skip_out

        return forecast_acc

    def get_block_forecasts(
        self,
        past_target: Tensor,
        hist_covs: Tensor,
        future_covs: Tensor,
    ) -> list[Tensor]:
        """Return individual block forecasts for explainability and visualization.

        Returns
        -------
        list[Tensor]
            List of forecast tensors, each of shape ``(B, H)``.
            If global skip is enabled, the last item is the global skip forecast.
        """
        x = torch.cat([past_target.unsqueeze(-1), hist_covs], dim=-1)
        residuals = x
        block_forecasts: list[Tensor] = []

        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = residuals - backcast
            block_forecasts.append(block_forecast)

        if self.use_global_skip:
            skip_out = self.skip(past_target)
            block_forecasts.append(skip_out)

        return block_forecasts

    @classmethod
    def from_config(cls, cfg: CancerNHiTSConfig) -> NHiTSModel:
        """Construct an NHiTSModel from a ``CancerNHiTSConfig`` instance."""
        from nhits.config import NHiTSConfig
        return cls(
            input_len=cfg.input_len,
            horizon=cfg.horizon,
            num_hist_covariates=cfg.num_hist_covariates,
            pooling_sizes=cfg.pooling_sizes,
            n_blocks_per_stack=cfg.n_blocks_per_stack,
            hidden_size=cfg.hidden_size,
            num_layers_per_block=cfg.num_layers_per_block,
            dropout=cfg.dropout,
            use_layer_norm=cfg.use_layer_norm,
            use_global_skip=cfg.use_global_skip,
            activation=cfg.activation,
        )
