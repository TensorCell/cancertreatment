"""Configuration dataclass for the TiDE cancer treatment forecasting model.

All hyperparameters live here so that experiments are reproducible and nothing
is hardcoded inside model or trainer modules.

Dataset context
---------------
Each sample is one radiotherapy protocol (200,000 total):
  - 20-step dose schedule (zero-padded)
  - 2 historical covariates: time (absolute, seconds) + time_gap (seconds)
  - 1 target: normalised average tumour cell count after 10 days
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CancerTiDEConfig:
    """Centralised hyperparameter container for cancer treatment TiDE.

    Attributes
    ----------
    # ── Data ──────────────────────────────────────────────────────────────
    data_path:
        Path to the EMT6/Ro simulation CSV file.
    input_len:
        Number of dose steps in each protocol (always 20 after padding).
    horizon:
        Forecast horizon.  Always 1 — we predict a single scalar outcome
        (tumour cell count at day 10).
    train_frac:
        Fraction of protocols used for training (split by protocol ID).
    val_frac:
        Fraction of protocols used for validation.
    # ── Covariates ────────────────────────────────────────────────────────
    num_hist_covariates:
        Historical covariates fed alongside past doses.
        Default: 2 (``time``, ``time_gap``).
    num_future_covariates:
        Future covariates for the horizon step.
        Always 0 — no inputs are known after dosing ends.
    # ── Model ─────────────────────────────────────────────────────────────
    hidden_size:
        Width of every dense layer inside residual blocks.
    num_encoder_layers:
        Number of residual blocks in the encoder.
    num_decoder_layers:
        Number of residual blocks in the decoder.
    temporal_decoder_hidden:
        Hidden size of the per-step temporal decoder MLP.
    dropout:
        Dropout probability applied inside residual blocks.
    use_layer_norm:
        Whether to apply LayerNorm after each residual connection.
    # ── Training ──────────────────────────────────────────────────────────
    lr:
        Initial learning rate for AdamW.
    weight_decay:
        L2 regularisation coefficient.
    batch_size:
        Mini-batch size (large batches recommended: 200k samples).
    max_epochs:
        Maximum number of training epochs.
    patience:
        Early-stopping patience (epochs without val-loss improvement).
    grad_clip:
        Maximum gradient norm; ``None`` disables gradient clipping.
    loss:
        Loss function — ``"mae"`` | ``"mse"`` | ``"huber"``.
    margin_loss:
        Enable Margin Ranking Loss (MRL) as an additive term on top of the
        base regression loss.  When ``True``, random in-batch pairs are
        generated and the model is trained to rank them correctly (higher
        true target → higher predicted value).  MRL is active in **all**
        phases: training, validation, and testing.  The combined loss
        drives early stopping, checkpointing, and LR scheduling.
    margin_loss_w:
        Scalar weight applied to the MRL term in the combined loss::

            total_loss = regression_loss + margin_loss_w × mrl_loss

        Set to ``0`` to include the pair architecture without letting
        the ranking term influence optimisation or model selection.
        Only used during training and validation; the test phase reports
        the unweighted MRL as a metric.
    # ── Output ────────────────────────────────────────────────────────────
    checkpoint_dir:
        Directory where ``best_model.pt`` is saved.
    output_dir:
        Directory where evaluation plots and CSVs are saved.
    model_name:
        Sub-directory name appended to ``checkpoint_dir`` / ``output_dir``.
    seed:
        Global random seed for reproducibility.
    # ── Weights & Biases ──────────────────────────────────────────────────
    wandb_enabled:
        Whether to log this run to Weights & Biases.
    wandb_project:
        W&B project name (e.g. ``"tide-forecasting"``).
    wandb_entity:
        W&B entity (username or team, e.g. ``"j95-jaworska-na"``).
    wandb_run_name:
        Optional human-readable name for this W&B run.  If ``None`` W&B
        generates a random name automatically.
    """

    # ── Data ──────────────────────────────────────────────────────────────
    data_path: Path | None = None
    input_len: int = 20          # 20 dose steps (padded)
    horizon: int = 1             # single tumour-cell count at day 10
    train_frac: float = 0.70
    val_frac: float = 0.15       # test_frac = 1 - train_frac - val_frac

    # ── Covariates ────────────────────────────────────────────────────────
    num_hist_covariates: int = 2  # time, time_gap
    num_future_covariates: int = 0  # no future inputs available

    # ── Model ─────────────────────────────────────────────────────────────
    hidden_size: int = 64 # 256
    num_encoder_layers: int = 3
    num_decoder_layers: int = 2
    temporal_decoder_hidden: int = 64
    dropout: float = 0.1
    use_layer_norm: bool = True

    # ── Training ──────────────────────────────────────────────────────────
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 512
    max_epochs: int = 100
    patience: int = 15
    grad_clip: float | None = 1.0
    loss: str = "mse"            # "mae" | "mse" | "huber"
    margin_loss: bool = False      # enable Margin Ranking Loss
    margin_loss_w: float = 1.0     # weight on MRL term (0 = no ranking influence)

    # ── Output ────────────────────────────────────────────────────────────
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    model_name: str = "tide"
    seed: int = 42

    # ── Weights & Biases ──────────────────────────────────────────────────
    wandb_enabled: bool = False
    wandb_project: str = "tide-forecasting"
    wandb_entity: str = "j95-jaworska-na"
    wandb_run_name: str | None = None

    def __post_init__(self) -> None:
        """Validate config values after initialisation."""
        if self.train_frac + self.val_frac >= 1.0:
            raise ValueError("train_frac + val_frac must be < 1.0")
        if self.loss not in {"mae", "mse", "huber"}:
            raise ValueError(f"Unknown loss '{self.loss}'. Choose mae | mse | huber.")
        if self.horizon != 1:
            raise ValueError(
                "This dataset has a single-step outcome; horizon must be 1."
            )
        if self.margin_loss_w < 0:
            raise ValueError("margin_loss_w must be >= 0.")
        self.checkpoint_dir = Path(self.checkpoint_dir) / self.model_name
        self.output_dir = Path(self.output_dir) / self.model_name

    @property
    def test_frac(self) -> float:
        """Derived fraction of protocols reserved for the test set."""
        return 1.0 - self.train_frac - self.val_frac
