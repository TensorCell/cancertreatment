"""Training loop, early stopping, and checkpointing for NHiTS.

Classes
-------
EarlyStopping
    Monitors validation loss and raises a flag after *patience* epochs
    without improvement.

Trainer
    Orchestrates the full training loop:
      - AdamW optimiser
      - ReduceLROnPlateau scheduler
      - Validation every epoch
      - Checkpointing the best model
      - Early stopping

Loss composition
----------------
When ``cfg.margin_loss`` is ``True``, each forward pass additionally
computes ``nn.MarginRankingLoss`` over **random pairs** sampled from the
current mini-batch:

    combined_loss = regression_loss + margin_loss_w × mrl_loss
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from nhits.config import CancerNHiTSConfig

try:
    import wandb as _wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Loss factory & pair sampler
# ---------------------------------------------------------------------------

def _build_loss(name: str) -> nn.Module:
    """Return the requested regression loss module."""
    if name == "mae":
        return nn.L1Loss()
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        return nn.HuberLoss()
    raise ValueError(f"Unknown loss '{name}'.")


def _random_pairs(n: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """Sample *n* random pairs from indices ``[0, n)``."""
    idx_a = torch.arange(n, device=device)
    offsets = torch.randint(1, n, (n,), device=device)
    idx_b = (idx_a + offsets) % n
    return idx_a, idx_b


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when validation loss stops improving."""

    def __init__(self, patience: int = 15, min_delta: float = 1e-7) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best_loss: float = float("inf")
        self._counter: int = 0
        self.should_stop: bool = False

    def step(self, val_loss: float) -> bool:
        """Update state with the latest validation loss."""
        if val_loss < self._best_loss - self.min_delta:
            self._best_loss = val_loss
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                self.should_stop = True
        return self.should_stop

    @property
    def best_loss(self) -> float:
        """Best validation loss seen so far."""
        return self._best_loss


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """Full training loop for NHiTS cancer treatment forecasting.

    Parameters
    ----------
    model:
        The ``NHiTSModel`` instance to train.
    cfg:
        ``CancerNHiTSConfig`` containing all hyperparameters.
    device:
        PyTorch device string (e.g. ``"cuda"`` or ``"cpu"``).
    on_epoch_end:
        Optional callback called at the end of every epoch with
        ``(epoch, train_loss, val_loss)``.
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: CancerNHiTSConfig,
        device: str | torch.device = "cpu",
        on_epoch_end: Callable[[int, float, float], None] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = torch.device(device)
        self.on_epoch_end = on_epoch_end

        # ── Loss functions ───────────────────────────────────────────────
        self.criterion = _build_loss(cfg.loss)
        self.mrl_criterion: nn.MarginRankingLoss | None = (
            nn.MarginRankingLoss(margin=0.0) if cfg.margin_loss else None
        )

        # ── Optimiser & scheduler ────────────────────────────────────────
        self.optimizer = AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=cfg.patience // 2,
        )
        self.early_stopping = EarlyStopping(patience=cfg.patience)

        # ── Checkpoint path ──────────────────────────────────────────────
        cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._ckpt_path = cfg.checkpoint_dir / "best_model.pt"

        # ── Weights & Biases initialisation ─────────────────────────────
        self._wandb_run = None
        if cfg.wandb_enabled and _WANDB_AVAILABLE:
            self._wandb_run = _wandb.init(
                project=cfg.wandb_project,
                entity=cfg.wandb_entity,
                name=cfg.wandb_run_name,
                config={
                    "input_len": cfg.input_len,
                    "horizon": cfg.horizon,
                    "train_frac": cfg.train_frac,
                    "val_frac": cfg.val_frac,
                    "pooling_sizes": cfg.pooling_sizes,
                    "n_blocks_per_stack": cfg.n_blocks_per_stack,
                    "hidden_size": cfg.hidden_size,
                    "num_layers_per_block": cfg.num_layers_per_block,
                    "dropout": cfg.dropout,
                    "use_layer_norm": cfg.use_layer_norm,
                    "use_global_skip": cfg.use_global_skip,
                    "lr": cfg.lr,
                    "weight_decay": cfg.weight_decay,
                    "batch_size": cfg.batch_size,
                    "max_epochs": cfg.max_epochs,
                    "patience": cfg.patience,
                    "loss": cfg.loss,
                    "grad_clip": cfg.grad_clip,
                    "seed": cfg.seed,
                    "margin_loss": cfg.margin_loss,
                    "margin_loss_w": cfg.margin_loss_w,
                },
                tags=["nhits", "forecasting", "cancer-treatment-optimization"],
                reinit=True,
            )
            _wandb.watch(model, log="gradients", log_freq=10)
            _wandb.define_metric("epoch")
            _wandb.define_metric("train/loss", step_metric="epoch")
            _wandb.define_metric("val/loss", step_metric="epoch")
            _wandb.define_metric("train/lr", step_metric="epoch")
            _wandb.define_metric("epoch_time_s", step_metric="epoch")
            _wandb.define_metric("train/mrl_loss", step_metric="epoch")
            _wandb.define_metric("val/mrl_loss", step_metric="epoch")
        elif cfg.wandb_enabled and not _WANDB_AVAILABLE:
            print("  [WARNING] wandb_enabled=True but 'wandb' package is not installed.")

    def _batch_to_device(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Move all tensors in a batch dict to the target device."""
        return {k: v.to(self.device) for k, v in batch.items()}

    def _compute_mrl(
        self, pred: Tensor, target: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Compute Margin Ranking Loss over random in-batch pairs."""
        p = pred.squeeze()    # (B,)
        t = target.squeeze()  # (B,)
        B = p.size(0)

        if B < 2:
            zero = p.sum() * 0.0
            return zero, torch.ones(1, device=self.device).squeeze()

        idx_a, idx_b = _random_pairs(B, self.device)
        p_a, p_b = p[idx_a], p[idx_b]
        t_a, t_b = t[idx_a], t[idx_b]

        target_sign = (t_a - t_b).sign()
        mrl = self.mrl_criterion(p_a, p_b, target_sign)  # type: ignore[misc]

        non_tie = target_sign != 0
        if non_tie.any():
            pred_sign = (p_a - p_b).sign()
            correct_frac = (pred_sign[non_tie] == target_sign[non_tie]).float().mean()
        else:
            correct_frac = torch.ones(1, device=self.device).squeeze()

        return mrl, correct_frac

    def _train_epoch(self, loader: DataLoader) -> tuple[float, float | None]:
        """Run one training epoch."""
        self.model.train()
        total_combined = 0.0
        total_mrl = 0.0
        n_batches = 0

        for batch in loader:
            batch = self._batch_to_device(batch)

            self.optimizer.zero_grad()
            pred = self.model(
                batch["past_target"],
                batch["hist_covs"],
                batch["future_covs"],
            )
            reg_loss: Tensor = self.criterion(pred, batch["target"])
            combined_loss = reg_loss

            mrl_loss_val: float | None = None
            correct_frac_val: float | None = None

            if self.mrl_criterion is not None:
                mrl, correct_frac = self._compute_mrl(pred, batch["target"])
                mrl_weighted = self.cfg.margin_loss_w * mrl
                combined_loss = reg_loss + mrl_weighted
                mrl_loss_val = mrl.item()
                correct_frac_val = correct_frac.item()
                total_mrl += mrl_loss_val

            combined_loss.backward()

            if self.cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )

            self.optimizer.step()
            total_combined += combined_loss.item()
            n_batches += 1

            if self._wandb_run is not None and mrl_loss_val is not None:
                _wandb.log({
                    "train/batch_mrl_loss": mrl_loss_val,
                    "train/batch_correct_rank_pct": correct_frac_val,
                })

        avg_combined = total_combined / max(n_batches, 1)
        avg_mrl = (total_mrl / max(n_batches, 1)) if self.mrl_criterion is not None else None
        return avg_combined, avg_mrl

    @torch.no_grad()
    def _val_epoch(
        self, loader: DataLoader
    ) -> tuple[float, float | None]:
        """Run one validation epoch."""
        self.model.eval()
        total_combined = 0.0
        total_mrl = 0.0
        n_batches = 0

        for batch in loader:
            batch = self._batch_to_device(batch)
            pred = self.model(
                batch["past_target"],
                batch["hist_covs"],
                batch["future_covs"],
            )
            reg_loss = self.criterion(pred, batch["target"])
            combined_loss = reg_loss

            if self.mrl_criterion is not None:
                mrl, _ = self._compute_mrl(pred, batch["target"])
                combined_loss = reg_loss + self.cfg.margin_loss_w * mrl
                total_mrl += mrl.item()

            total_combined += combined_loss.item()
            n_batches += 1

        avg_combined = total_combined / max(n_batches, 1)
        avg_mrl = (total_mrl / max(n_batches, 1)) if self.mrl_criterion is not None else None
        return avg_combined, avg_mrl

    @torch.no_grad()
    def _test_epoch(
        self, loader: DataLoader
    ) -> tuple[float, float | None, float | None]:
        """Evaluate on a test set."""
        self.model.eval()
        total_reg = 0.0
        total_mrl = 0.0
        total_correct = 0.0
        n_batches = 0

        for batch in loader:
            batch = self._batch_to_device(batch)
            pred = self.model(
                batch["past_target"],
                batch["hist_covs"],
                batch["future_covs"],
            )
            total_reg += self.criterion(pred, batch["target"]).item()

            if self.mrl_criterion is not None:
                mrl, correct_frac = self._compute_mrl(pred, batch["target"])
                total_mrl += mrl.item()
                total_correct += correct_frac.item()

            n_batches += 1

        avg_reg = total_reg / max(n_batches, 1)
        if self.mrl_criterion is not None:
            avg_mrl: float | None = total_mrl / max(n_batches, 1)
            avg_correct: float | None = total_correct / max(n_batches, 1)
        else:
            avg_mrl = None
            avg_correct = None

        return avg_reg, avg_mrl, avg_correct

    def _save_checkpoint(self, epoch: int, val_loss: float) -> None:
        """Save the full model state to ``best_model.pt``."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "val_loss": val_loss,
            },
            self._ckpt_path,
        )

    def fit(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> dict[str, list[float]]:
        """Train the model for up to ``cfg.max_epochs`` epochs."""
        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        if self.cfg.margin_loss:
            history["train_mrl"] = []
            history["val_mrl"] = []

        best_val = float("inf")

        mrl_flag = f" [+ MarginRankingLoss w={self.cfg.margin_loss_w:.1f}]" \
            if self.cfg.margin_loss else ""

        print(f"\n{'='*60}")
        print(f"  NHiTS training — device: {self.device}")
        print(f"  Max epochs: {self.cfg.max_epochs}  |  Patience: {self.cfg.patience}")
        print(f"  Loss: {self.cfg.loss}{mrl_flag}")
        print(f"{'='*60}\n")

        for epoch in range(1, self.cfg.max_epochs + 1):
            t0 = time.perf_counter()
            train_loss, train_mrl = self._train_epoch(train_loader)
            val_loss, val_mrl = self._val_epoch(val_loader)
            elapsed = time.perf_counter() - t0

            self.scheduler.step(val_loss)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            if self.cfg.margin_loss:
                history["train_mrl"].append(train_mrl or 0.0)
                history["val_mrl"].append(val_mrl or 0.0)

            if val_loss < best_val:
                best_val = val_loss
                self._save_checkpoint(epoch, val_loss)
                ckpt_marker = " ✓"
            else:
                ckpt_marker = ""

            lr_now = self.optimizer.param_groups[0]["lr"]

            mrl_str = ""
            if self.cfg.margin_loss and val_mrl is not None:
                mrl_str = f" | mrl {val_mrl:.4f}"
            print(
                f"Epoch {epoch:4d}/{self.cfg.max_epochs} | "
                f"train {train_loss:.6f} | val {val_loss:.6f}"
                f"{mrl_str} | "
                f"lr {lr_now:.2e} | {elapsed:.1f}s{ckpt_marker}"
            )

            if self._wandb_run is not None:
                log_dict: dict[str, float] = {
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "train/lr": lr_now,
                    "epoch_time_s": elapsed,
                }
                if self.cfg.margin_loss:
                    log_dict["train/mrl_loss"] = train_mrl or 0.0
                    log_dict["val/mrl_loss"] = val_mrl or 0.0
                _wandb.log(log_dict)

            if self.on_epoch_end is not None:
                self.on_epoch_end(epoch, train_loss, val_loss)

            if self.early_stopping.step(val_loss):
                print(f"\nEarly stopping at epoch {epoch}.")
                break

        print(f"\nBest val loss: {best_val:.6f}")
        print(f"Checkpoint: {self._ckpt_path}\n")

        if self._wandb_run is not None:
            _wandb.summary["best_val_loss"] = best_val
            artifact = _wandb.Artifact(
                name=f"nhits-checkpoint-{self._wandb_run.id}",
                type="model",
                description="Best NHiTS checkpoint saved during training",
            )
            artifact.add_file(str(self._ckpt_path))
            self._wandb_run.log_artifact(artifact)

        return history

    def evaluate(self, test_loader: DataLoader) -> dict[str, float]:
        """Evaluate the model on a test set."""
        reg_loss, mrl_loss, correct_pct = self._test_epoch(test_loader)
        results: dict[str, float] = {"test_reg_loss": reg_loss}

        if mrl_loss is not None:
            results["test_mrl_loss"] = mrl_loss
            results["test_correct_rank_pct"] = correct_pct or 0.0

        print(f"\n{'='*60}")
        print(f"  Test evaluation")
        print(f"  Regression loss ({self.cfg.loss}): {reg_loss:.6f}")
        if mrl_loss is not None:
            print(f"  Margin Ranking Loss (unweighted): {mrl_loss:.6f}")
            print(f"  Correct rank %: {(correct_pct or 0.0)*100:.2f}%")
        print(f"{'='*60}\n")

        if self._wandb_run is not None:
            _wandb.summary.update({k: v for k, v in results.items()})

        return results
