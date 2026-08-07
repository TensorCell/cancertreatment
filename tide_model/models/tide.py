"""TiDE: Time-series Dense Encoder model implementation in PyTorch.

Architecture (Das et al., 2023 — "Long-term Forecasting with TiDE"):
  1. Feature Projection  — project concatenated inputs to hidden_size
  2. Encoder             — stack of ResidualBlocks → latent vector
  3. Decoder             — stack of ResidualBlocks → per-step hidden states
  4. Temporal Decoder    — per-step dense layer → scalar forecast
  5. (Optional) skip connection from raw past_target → forecast

All components use dense residual MLP blocks.
No attention, no recurrence, no convolutions.

Cancer-treatment usage notes
-----------------------------
  - horizon=1 (single scalar outcome)
  - num_future_covariates=0 (no known future inputs)
  - The TemporalDecoder receives zero-width future_covs, making it a
    pure linear layer over temporal_decoder_hidden.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Dense residual block: Linear → GELU → Dropout → Linear + skip.

    If ``input_dim != output_dim`` a 1×1 linear projection is inserted on
    the skip path so that dimensions always match.

    Parameters
    ----------
    input_dim:
        Input feature dimension.
    output_dim:
        Output feature dimension.
    hidden_dim:
        Width of the inner expansion layer.  Defaults to ``output_dim``.
    dropout:
        Dropout probability (applied after the first activation).
    use_layer_norm:
        If ``True`` LayerNorm is applied to the residual output.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Skip projection (only when dimensions differ)
        self.skip = (
            nn.Linear(input_dim, output_dim, bias=False)
            if input_dim != output_dim
            else nn.Identity()
        )

        self.norm = nn.LayerNorm(output_dim) if use_layer_norm else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape ``(..., input_dim)``.

        Returns
        -------
        Tensor
            Output tensor of shape ``(..., output_dim)``.
        """
        residual = self.skip(x)
        out = self.fc2(self.drop(self.act(self.fc1(x))))
        return self.norm(out + residual)


# ---------------------------------------------------------------------------
# Feature Projection
# ---------------------------------------------------------------------------

class FeatureProjection(nn.Module):
    """Project concatenated model inputs into a single hidden-size vector.

    Inputs concatenated here (along the feature dimension):
      - past target values   : (B, L)   → flattened to (B, L)
      - historical covariates: (B, L, C_hist) → (B, L * C_hist)
      - future covariates    : (B, H, C_fut)  → (B, H * C_fut)

    The concatenated vector is then passed through a single ResidualBlock
    that maps it to ``hidden_size``.

    Parameters
    ----------
    input_len:
        Look-back window length L.
    horizon:
        Forecast horizon H.
    num_hist_covariates:
        Number of historical covariate features.
    num_future_covariates:
        Number of future covariate features.
    hidden_size:
        Output dimension (= encoder input dimension).
    dropout:
        Dropout probability.
    use_layer_norm:
        Whether to apply LayerNorm inside the residual block.
    """

    def __init__(
        self,
        input_len: int,
        horizon: int,
        num_hist_covariates: int,
        num_future_covariates: int,
        hidden_size: int,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_len = input_len
        self.horizon = horizon
        self.num_hist_covariates = num_hist_covariates
        self.num_future_covariates = num_future_covariates

        flat_dim = (
            input_len                              # past target
            + input_len * num_hist_covariates      # historical covariates
            + horizon * num_future_covariates      # future covariates (may be 0)
        )

        self.proj = ResidualBlock(
            input_dim=flat_dim,
            output_dim=hidden_size,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

    def forward(
        self,
        past_target: Tensor,   # (B, L)
        hist_covs: Tensor,     # (B, L, C_hist)
        future_covs: Tensor,   # (B, H, C_fut)  — may have C_fut == 0
    ) -> Tensor:
        """Concatenate all inputs and project to hidden_size.

        Returns
        -------
        Tensor
            Shape ``(B, hidden_size)``.
        """
        B = past_target.size(0)

        flat_parts: list[Tensor] = [
            past_target,                   # (B, L)
            hist_covs.reshape(B, -1),      # (B, L*C_hist)
        ]
        # Only concatenate future covs when the feature dimension is non-zero
        if future_covs.shape[-1] > 0:
            flat_parts.append(future_covs.reshape(B, -1))  # (B, H*C_fut)

        x = torch.cat(flat_parts, dim=-1)  # (B, flat_dim)
        return self.proj(x)                # (B, hidden_size)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class TiDEEncoder(nn.Module):
    """Stack of ``num_layers`` ResidualBlocks operating on the latent vector.

    Parameters
    ----------
    hidden_size:
        Input and output dimension of each residual block.
    num_layers:
        Number of residual blocks to stack.
    dropout:
        Dropout probability.
    use_layer_norm:
        Whether to apply LayerNorm inside each residual block.
    """

    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                ResidualBlock(
                    input_dim=hidden_size,
                    output_dim=hidden_size,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x:
            Shape ``(B, hidden_size)``.

        Returns
        -------
        Tensor
            Shape ``(B, hidden_size)`` — the latent representation.
        """
        return self.blocks(x)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class TiDEDecoder(nn.Module):
    """Stack of ResidualBlocks that maps the latent vector to per-step states.

    The decoder expands the latent vector from ``hidden_size`` to
    ``horizon * temporal_hidden``, which is then reshaped to
    ``(B, H, temporal_hidden)`` for the temporal decoder.

    Parameters
    ----------
    hidden_size:
        Input dimension (= encoder output dimension).
    horizon:
        Forecast horizon.
    temporal_hidden:
        Per-step hidden dimension fed to the temporal decoder.
    num_layers:
        Number of residual blocks.
    dropout:
        Dropout probability.
    use_layer_norm:
        Whether to apply LayerNorm.
    """

    def __init__(
        self,
        hidden_size: int,
        horizon: int,
        temporal_hidden: int,
        num_layers: int,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.temporal_hidden = temporal_hidden
        output_dim = horizon * temporal_hidden

        layers: list[nn.Module] = []
        in_dim = hidden_size
        for i in range(num_layers):
            out_dim = output_dim if i == num_layers - 1 else hidden_size
            layers.append(
                ResidualBlock(
                    input_dim=in_dim,
                    output_dim=out_dim,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                )
            )
            in_dim = out_dim

        self.blocks = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        z:
            Latent tensor of shape ``(B, hidden_size)``.

        Returns
        -------
        Tensor
            Shape ``(B, horizon, temporal_hidden)``.
        """
        out = self.blocks(z)                       # (B, H * temporal_hidden)
        return out.view(out.size(0), self.horizon, self.temporal_hidden)


# ---------------------------------------------------------------------------
# Temporal Decoder
# ---------------------------------------------------------------------------

class TemporalDecoder(nn.Module):
    """Per-step MLP: maps each horizon step's hidden state → scalar forecast.

    A single shared linear layer is applied independently to each of the H
    horizon positions.

    Parameters
    ----------
    temporal_hidden:
        Input dimension for each step (output of ``TiDEDecoder``).
    num_future_covariates:
        Future covariates are concatenated to the step's hidden vector
        before the final projection.  May be 0.
    """

    def __init__(
        self,
        temporal_hidden: int,
        num_future_covariates: int,
    ) -> None:
        super().__init__()
        in_dim = temporal_hidden + num_future_covariates
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, dec_out: Tensor, future_covs: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        dec_out:
            Decoder output ``(B, H, temporal_hidden)``.
        future_covs:
            Future covariates ``(B, H, C_fut)``.  May have ``C_fut == 0``.

        Returns
        -------
        Tensor
            Forecast of shape ``(B, H)``.
        """
        if future_covs.shape[-1] > 0:
            x = torch.cat([dec_out, future_covs], dim=-1)  # (B, H, temporal_hidden+C_fut)
        else:
            x = dec_out                                     # (B, H, temporal_hidden)
        return self.fc(x).squeeze(-1)                       # (B, H)


# ---------------------------------------------------------------------------
# TiDE Model
# ---------------------------------------------------------------------------

class TiDEModel(nn.Module):
    """TiDE: Time-series Dense Encoder for tumour cell survival forecasting.

    Full pipeline:
      FeatureProjection → Encoder → Decoder → TemporalDecoder

    A learnable linear skip connection from the raw look-back window
    directly to the forecast is also added (as in the paper).

    Parameters
    ----------
    input_len:
        Look-back window length (dose steps).
    horizon:
        Forecast horizon.  1 for this dataset.
    num_hist_covariates:
        Number of historical covariate channels (2: time, time_gap).
    num_future_covariates:
        Number of future covariate channels (0 for this dataset).
    hidden_size:
        Width of every dense layer.
    num_encoder_layers:
        Depth of the encoder stack.
    num_decoder_layers:
        Depth of the decoder stack.
    temporal_decoder_hidden:
        Per-step hidden size in the temporal decoder.
    dropout:
        Dropout probability.
    use_layer_norm:
        Whether to use LayerNorm in residual blocks.
    """

    def __init__(
        self,
        input_len: int,
        horizon: int,
        num_hist_covariates: int,
        num_future_covariates: int,
        hidden_size: int = 256,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 2,
        temporal_decoder_hidden: int = 64,
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_len = input_len
        self.horizon = horizon
        self.num_future_covariates = num_future_covariates

        # ── Feature projection ────────────────────────────────────────────
        self.feature_proj = FeatureProjection(
            input_len=input_len,
            horizon=horizon,
            num_hist_covariates=num_hist_covariates,
            num_future_covariates=num_future_covariates,
            hidden_size=hidden_size,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

        # ── Encoder ───────────────────────────────────────────────────────
        self.encoder = TiDEEncoder(
            hidden_size=hidden_size,
            num_layers=num_encoder_layers,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

        # ── Decoder ───────────────────────────────────────────────────────
        self.decoder = TiDEDecoder(
            hidden_size=hidden_size,
            horizon=horizon,
            temporal_hidden=temporal_decoder_hidden,
            num_layers=num_decoder_layers,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )

        # ── Temporal decoder ──────────────────────────────────────────────
        self.temporal_decoder = TemporalDecoder(
            temporal_hidden=temporal_decoder_hidden,
            num_future_covariates=num_future_covariates,
        )

        # ── Global skip connection (past_target → forecast) ───────────────
        # Maps the look-back window directly to the horizon (linear)
        self.skip = nn.Linear(input_len, horizon, bias=True)

    def forward(
        self,
        past_target: Tensor,   # (B, L)
        hist_covs: Tensor,     # (B, L, C_hist)
        future_covs: Tensor,   # (B, H, C_fut)  — may have C_fut == 0
    ) -> Tensor:
        """Run the full TiDE forward pass.

        Parameters
        ----------
        past_target:
            Scaled dose sequence, shape ``(B, L)``.
        hist_covs:
            Historical covariates [time, time_gap], shape ``(B, L, C_hist)``.
        future_covs:
            Future known covariates, shape ``(B, H, C_fut)``.
            For this dataset ``C_fut == 0`` so this is an empty tensor.

        Returns
        -------
        Tensor
            Forecast of shape ``(B, H)`` — predicted normalised tumour cells.
        """
        # 1. Project all inputs to hidden_size
        projected = self.feature_proj(past_target, hist_covs, future_covs)  # (B, hidden_size)

        # 2. Encode into a latent vector
        latent = self.encoder(projected)            # (B, hidden_size)

        # 3. Decode into per-step hidden states
        dec_out = self.decoder(latent)              # (B, H, temporal_hidden)

        # 4. Temporal decoder → scalar per step
        forecast = self.temporal_decoder(dec_out, future_covs)  # (B, H)

        # 5. Add global skip (linear mapping from past doses to horizon)
        skip_out = self.skip(past_target)           # (B, H)

        return forecast + skip_out

    @classmethod
    def from_config(cls, cfg: "CancerTiDEConfig") -> "TiDEModel":  # noqa: F821
        """Construct a TiDEModel from a ``CancerTiDEConfig`` instance."""
        from tide.config import CancerTiDEConfig  # local import to avoid circular deps
        return cls(
            input_len=cfg.input_len,
            horizon=cfg.horizon,
            num_hist_covariates=cfg.num_hist_covariates,
            num_future_covariates=cfg.num_future_covariates,
            hidden_size=cfg.hidden_size,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            temporal_decoder_hidden=cfg.temporal_decoder_hidden,
            dropout=cfg.dropout,
            use_layer_norm=cfg.use_layer_norm,
        )
