"""Dataset utilities wrapper for NHiTS cancer treatment forecasting.

Re-exports the core EMT6/Ro dataset pipeline from ``tide.datasets.cancer_dataset``
to ensure identical pre-processing, scaling, and train/val/test protocol splits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tide.datasets.cancer_dataset import (
    CancerProtocolDataset,
    ScaleStats,

    load_cancer_data,
)
from tide.datasets.cancer_dataset import (
    build_dataloaders as _tide_build_dataloaders,
)

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from nhits.config import CancerNHiTSConfig


def build_dataloaders(
    cfg: CancerNHiTSConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test DataLoaders for NHiTS.

    Passes the ``CancerNHiTSConfig`` to the dataset builder.

    Parameters
    ----------
    cfg:
        :class:`CancerNHiTSConfig` instance.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        Train, validation, test loaders.
    """
    # tide's build_dataloaders relies on data_path, input_len, train_frac, val_frac, seed, batch_size, horizon
    return _tide_build_dataloaders(cfg)  # type: ignore[arg-type]


__all__ = [
    "CancerProtocolDataset",
    "ScaleStats",
    "build_dataloaders",
    "load_cancer_data",
]
