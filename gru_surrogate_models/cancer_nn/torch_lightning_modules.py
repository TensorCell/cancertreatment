"""'Pythorch lightning module for cancer optimization
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from cancer_nn.datasets import Dataset, DatasetMargin
from cancer_nn.metrics import evaluate
from cancer_nn.used_networks import MultiHeadTaskRegressor, MultiTaskRegressor

LOSSES_DICT = {
    "L1": torch.nn.L1Loss(),
    "HuberLoss": torch.nn.HuberLoss(),
    "MSE": torch.nn.MSELoss(),
}


class CancerNet(pl.LightningModule):
    """Pythorch lightning cancer net module"""

    def __init__(
        self,
        train_df: Optional[pd.DataFrame] = None,
        val_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        lr: float = 0.001,
        margin_loss: bool = True,
        config_list: List[Dict[str, Any]] = [{"output_last_layer": True}],
        losses: List[str] = ["L1", "MSE", "HuberLoss"],
        main_loss: str = "L1",
        margin_loss_w: Union[float, int] = 3,
        margin: float = -0.04,
        mode: str = "unet",
        w_losses: List[Union[int, float]] = [1],
        name: str = "MultiHeadTaskRegressor",  # config key, accepted so configs pass through
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        scheduler: str = "plateau",
        hard_pair_frac: float = 0.0,
        hard_pair_min_diff: float = 4.0,
        hard_pair_max_diff: float = 15.0,
    ):
        super().__init__()

        self.lr = lr
        self.margin_loss = margin_loss
        self.margin_loss_w = margin_loss_w
        self.w_losses = w_losses
        self.optimizer_name = optimizer.lower()
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler.lower()
        self.hard_pair_frac = hard_pair_frac
        self.hard_pair_min_diff = hard_pair_min_diff
        self.hard_pair_max_diff = hard_pair_max_diff

        # data
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.prepare()

        # model
        self.val_loss = []
        self.test_loss = []
        self.val_predictions = []
        self.val_true = []
        self.test_true = []
        self.test_predictions = []
        if self.margin_loss:
            self.losses = [LOSSES_DICT[loss] for loss in losses] + [
                torch.nn.MarginRankingLoss(margin=margin)
            ]
            self.head_output_size = len(self.losses)
            self.model = MultiHeadTaskRegressor(config_list, mode=mode)

        else:
            self.losses = [LOSSES_DICT[loss] for loss in losses]
            self.head_output_size = len(self.losses) - 1
            self.model = MultiTaskRegressor(config_list, mode=mode)
        self.losses = [LOSSES_DICT[main_loss]] + self.losses
        # event-based subnetworks need the raw feature scale to tell doses from padding
        for layer in self.model.layers:
            if hasattr(layer, "set_feature_scaling"):
                layer.set_feature_scaling(self.scalers)

    def prepare(self) -> None:
        """Prepare data with torch dataset,
        depends on whether margin_loss is used

        """
        if self.margin_loss:
            self.train_set = DatasetMargin(
                self.train_df.copy(),
                hard_pair_frac=self.hard_pair_frac,
                hard_pair_min_diff=self.hard_pair_min_diff,
                hard_pair_max_diff=self.hard_pair_max_diff,
            )
        else:
            self.train_set = Dataset(self.train_df.copy())
        self.scalers = self.train_set.scalers
        if self.margin_loss:
            self.val_set = DatasetMargin(self.val_df.copy(), scalers=self.scalers)
            self.test_set = DatasetMargin(self.test_df.copy(), scalers=self.scalers)
        else:
            self.val_set = Dataset(self.val_df.copy(), scalers=self.scalers)
            self.test_set = Dataset(self.test_df.copy(), scalers=self.scalers)
        self.target_scale = self.train_set.target_scale

    def train_dataloader(self) -> DataLoader:
        """prepare train dataloader

        Returns:
        ---------
            DataLoader: train dataloader
        """
        return DataLoader(
            self.train_set,
            batch_size=256,
            num_workers=7,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """prepare val dataloader

        Returns:
        ---------
            DataLoader: val dataloader
        """
        return DataLoader(
            self.val_set,
            batch_size=256,
            num_workers=7,
        )

    def test_dataloader(self) -> DataLoader:
        """prepare test dataloader

        Returns:
        ---------
            DataLoader: test dataloader
        """
        return DataLoader(
            self.test_set,
            batch_size=256,
            num_workers=7,
        )

    def forward(self, x: Tuple[torch.Tensor]) -> torch.Tensor:
        """forward function

        Args:
        ---------
            x (Tuple[torch.Tensor]): Tuple
            from single batch

        Returns:
        ---------
            torch.Tensor: torch tensor w
            ith predicted values
        """
        y_pred = self.model(x)
        return y_pred

    def training_step(
        self, batch: Tuple[torch.Tensor], batch_idx: int
    ) -> torch.FloatTensor:
        """training step (see pythorch-lightning
        documenation for more information)

        Args:
        ---------
            batch (Tuple[torch.Tensor]): single batch
            batch_idx (int): batch index

        Returns:
        ---------
            torch.FloatTensor: loss value
        """
        output = self(batch)
        loss_weights = self.w_losses
        loss = torch.sum(
            torch.stack(
                [
                    self.losses[i](batch[1].squeeze(), output[:, i].squeeze())
                    * loss_weights[i]
                    for i in range(self.head_output_size)
                ],
            ),
        )

        wandb_log = {
            "train/loss_mae": torch.mean(
                torch.abs(
                    batch[1] * self.target_scale - output[:, 0] * self.target_scale
                )
            ).item()
        }
        if self.margin_loss:
            margin_loss = (
                self.losses[-1](
                    output[:, 0], output[:, -1], (batch[1] - batch[3]).sign()
                )
                * self.margin_loss_w
            )
            loss += margin_loss
            wandb_log["train/margin_loss"] = margin_loss.item()

        self.logger.experiment.log(wandb_log)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor], batch_idx: int
    ) -> torch.FloatTensor:
        """validation step (see pythorch-lightning
        documenation for more information)

        Args:
        ---------
            batch (Tuple[torch.Tensor]): single batch
            batch_idx (int): batch index

        Returns:
        ---------
            torch.FloatTensor: loss value
        """
        output = self.model(batch)
        loss = torch.sum(
            torch.stack(
                [
                    self.losses[i](batch[1].squeeze(), output[:, i].squeeze())
                    * self.w_losses[i]
                    for i in range(self.head_output_size)
                ],
            ),
        )
        if self.margin_loss:
            loss += (
                self.losses[-1](
                    output[:, 0], output[:, -1], (batch[1] - batch[3]).sign()
                )
                * self.margin_loss_w
            )
        self.val_loss.append(loss.detach().cpu())
        self.val_true.append(batch[1].flatten().detach().cpu())
        self.val_predictions.append(output[:, 0].flatten().detach().cpu())
        self.log_dict({"val_loss": loss})
        return loss

    def on_validation_epoch_end(self):
        y_true = torch.cat(self.val_true).numpy() * self.target_scale
        y_pred = torch.cat(self.val_predictions).numpy() * self.target_scale
        wandb_log = {
            "lr": self.optimizers().param_groups[0]["lr"],
            "val/loss": float(np.mean(np.array(self.val_loss))),
        }
        wandb_log.update({f"val/{k}": v for k, v in evaluate(y_true, y_pred).items()})
        self.logger.experiment.log(wandb_log)
        self.val_loss, self.val_true, self.val_predictions = [], [], []

    def test_step(
        self, batch: Tuple[torch.Tensor], batch_idx: int
    ) -> torch.FloatTensor:
        """test step (see pythorch-lightning
        documenation for more information)

        Args:
        ---------
            batch (Tuple[torch.Tensor]): single batch
            batch_idx (int): batch index

        Returns:
        ---------
            torch.FloatTensor: loss value
        """
        output = self.model(batch)
        mar_loss = torch.nn.MarginRankingLoss(margin=0)
        loss = torch.sum(
            torch.stack(
                [
                    self.losses[i](batch[1].squeeze(), output[:, i].squeeze())
                    for i in range(self.head_output_size)
                ],
            ),
        )
        if self.margin_loss:
            loss += (
                mar_loss(output[:, 0], output[:, -1], (batch[1] - batch[3]).sign())
                * self.margin_loss_w
            )
        self.test_loss.append(loss.detach().cpu())
        self.test_true.append(batch[1].flatten().detach().cpu())
        self.test_predictions.append(output[:, 0].flatten().detach().cpu())
        self.log_dict({"test_loss": loss})
        return loss

    def on_test_epoch_end(self):
        y_true = torch.cat(self.test_true).numpy() * self.target_scale
        y_pred = torch.cat(self.test_predictions).numpy() * self.target_scale
        wandb_log = {"test/loss": float(np.mean(np.array(self.test_loss)))}
        wandb_log.update({f"test/{k}": v for k, v in evaluate(y_true, y_pred).items()})
        self.logger.experiment.log(wandb_log)
        self.test_loss = []

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[Dict[str, Any]]]:
        """configure optimizers  (see pythorch-lightning
        documenation for more information)

        Returns:
        ---------
            (Tuple(List[torch.optim], List[Dict[str, Any]])): List
            with optimizers, List with schedulers
        """
        optimizer_classes = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}
        if self.optimizer_name not in optimizer_classes:
            raise ValueError(
                f"unsupported optimizer {self.optimizer_name!r}, "
                f"expected one of {list(optimizer_classes)}"
            )
        optimizers = [
            optimizer_classes[self.optimizer_name](
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        ]

        if self.scheduler_name == "plateau":
            schedulers = [
                {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizers[0], mode="min", factor=0.7, patience=5
                    ),
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            ]
        elif self.scheduler_name == "onecycle":
            steps_per_epoch = len(self.train_dataloader())
            schedulers = [
                {
                    "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                        optimizers[0],
                        max_lr=self.lr,
                        steps_per_epoch=steps_per_epoch,
                        epochs=self.trainer.max_epochs,
                    ),
                    "interval": "step",
                    "frequency": 1,
                },
            ]
        elif self.scheduler_name == "cosine":
            schedulers = [
                {
                    "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        optimizers[0], T_0=10
                    ),
                    "interval": "epoch",
                    "frequency": 1,
                },
            ]
        else:
            raise ValueError(
                f"unsupported scheduler {self.scheduler_name!r}, "
                "expected 'plateau', 'onecycle', or 'cosine'"
            )
        return (optimizers, schedulers)
