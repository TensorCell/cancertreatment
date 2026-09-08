"""Dataset and DataLoader utilities for cancer treatment tumour cell survival forecasting.

EMT6/Ro radiotherapy simulation dataset:
  - 200,000 independent protocols, each with 20 dose steps (zero-padded)
  - Features per step: dose (past_target), time + time_gap (hist_covs)
  - Single scalar label: normalised tumour cell count after 10 days

Provides
--------
load_cancer_data
    Load and validate the CSV, returning one DataFrame per protocol.
CancerProtocolDataset
    PyTorch Dataset — each item is one protocol (one sample).
build_dataloaders
    Construct train / val / test DataLoaders split by protocol ID.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from tide.config import CancerTiDEConfig


# ---------------------------------------------------------------------------
# Scale statistics container
# ---------------------------------------------------------------------------

class ScaleStats(NamedTuple):
    """Min/max and mean/std statistics fitted on the training split."""

    dose_min: float
    dose_max: float
    target_mean: float
    target_std: float
    time_min: float
    time_max: float
    time_gap_min: float
    time_gap_max: float


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_cancer_data(path: str | Path) -> pd.DataFrame:
    """Load the EMT6/Ro simulation CSV and validate its schema.

    Parameters
    ----------
    path:
        Path to ``data.csv``.

    Returns
    -------
    pd.DataFrame
        Raw dataframe sorted by (``series``, ``time_idx``).

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    required_cols = {"time", "dose", "series", "time_idx", "is_target", "target", "time_gap"}

    df = pd.read_csv(path, index_col=0)

    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Ensure chronological order within each protocol
    df = df.sort_values(["series", "time_idx"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Protocol extraction helpers
# ---------------------------------------------------------------------------

def _extract_protocol_arrays(
    group: pd.DataFrame,
    input_len: int = 20,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Extract arrays for one protocol group.

    Parameters
    ----------
    group:
        Rows belonging to a single ``series`` value, sorted by ``time_idx``.
    input_len:
        Expected number of dose-schedule steps (default 20).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        ``(dose_seq, cov_seq, target_value)`` where:
          - ``dose_seq``  — shape ``(input_len,)`` dose values (raw)
          - ``cov_seq``   — shape ``(input_len, 2)`` [time, time_gap] (raw)
          - ``target_value`` — scalar normalised tumour cell count
    """
    # Separate schedule rows (is_target == 0) from the outcome row (is_target == 1)
    schedule = group[group["is_target"] == 0].iloc[:input_len]
    outcome = group[group["is_target"] == 1].iloc[0]

    dose_seq = schedule["dose"].values.astype(np.float32)     # (input_len,)
    time_seq = schedule["time"].values.astype(np.float32)     # (input_len,)
    gap_seq = schedule["time_gap"].values.astype(np.float32)  # (input_len,)

    cov_seq = np.stack([time_seq, gap_seq], axis=-1)          # (input_len, 2)
    target_value = float(outcome["target"])

    return dose_seq, cov_seq, target_value


# ---------------------------------------------------------------------------
# CancerProtocolDataset
# ---------------------------------------------------------------------------

class CancerProtocolDataset(Dataset):
    """PyTorch Dataset where each sample is one radiotherapy protocol.

    Each item contains:
      - ``past_target``  — scaled dose sequence, shape ``(input_len,)``
      - ``hist_covs``    — scaled [time, time_gap], shape ``(input_len, 2)``
      - ``future_covs``  — empty tensor, shape ``(1, 0)`` (no future inputs)
      - ``target``       — normalised tumour cell count, shape ``(1,)``

    Parameters
    ----------
    dose_seqs:
        Array of dose sequences, shape ``(N, input_len)``, already scaled.
    cov_seqs:
        Array of covariate sequences, shape ``(N, input_len, 2)``, already scaled.
    targets:
        Array of target values, shape ``(N,)``.
    horizon:
        Forecast horizon (always 1 for this dataset).
    """

    def __init__(
        self,
        dose_seqs: np.ndarray,
        cov_seqs: np.ndarray,
        targets: np.ndarray,
        horizon: int = 1,
    ) -> None:
        super().__init__()
        self.dose_seqs = torch.tensor(dose_seqs, dtype=torch.float32)   # (N, L)
        self.cov_seqs = torch.tensor(cov_seqs, dtype=torch.float32)     # (N, L, 2)
        self.targets = torch.tensor(targets, dtype=torch.float32)       # (N,)
        self.horizon = horizon

        # Placeholder for scale_stats attached by build_dataloaders
        self.scale_stats: ScaleStats | None = None

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Return a single protocol sample.

        Returns
        -------
        dict with keys:
            ``past_target``  — ``(input_len,)``
            ``hist_covs``    — ``(input_len, 2)``
            ``future_covs``  — ``(horizon, 0)`` empty tensor
            ``target``       — ``(horizon,)``
        """
        L = self.dose_seqs.shape[1]
        return {
            "past_target": self.dose_seqs[idx],                       # (L,)
            "hist_covs": self.cov_seqs[idx],                          # (L, 2)
            # No future covariates — create a correctly-shaped empty tensor
            "future_covs": torch.zeros(self.horizon, 0, dtype=torch.float32),
            "target": self.targets[idx : idx + 1],                    # (1,)
        }


# ---------------------------------------------------------------------------
# MinMax scaling helpers
# ---------------------------------------------------------------------------

def _fit_scaler(
    dose_seqs: np.ndarray,
    cov_seqs: np.ndarray,
    targets: np.ndarray,
) -> ScaleStats:
    """Compute MinMax scale statistics from the training arrays.

    Parameters
    ----------
    dose_seqs:
        Shape ``(N, L)`` — raw training dose values.
    cov_seqs:
        Shape ``(N, L, 2)`` — raw [time, time_gap] training values.
    targets:
        Shape ``(N,)`` — training target values.

    Returns
    -------
    ScaleStats
        Min/max for features, and mean/std for target.
    """
    dose_min = float(dose_seqs.min())
    dose_max = float(dose_seqs.max())
    target_mean = float(targets.mean())
    target_std = float(targets.std())
    time_min = float(cov_seqs[:, :, 0].min())
    time_max = float(cov_seqs[:, :, 0].max())
    gap_min = float(cov_seqs[:, :, 1].min())
    gap_max = float(cov_seqs[:, :, 1].max())

    return ScaleStats(
        dose_min=dose_min,
        dose_max=dose_max,
        target_mean=target_mean,
        target_std=target_std,
        time_min=time_min,
        time_max=time_max,
        time_gap_min=gap_min,
        time_gap_max=gap_max,
    )


def _apply_minmax(
    dose_seqs: np.ndarray,
    cov_seqs: np.ndarray,
    stats: ScaleStats,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply MinMax scaling using pre-fitted statistics.

    Parameters
    ----------
    dose_seqs:
        Raw dose array, shape ``(N, L)``.
    cov_seqs:
        Raw covariate array, shape ``(N, L, 2)``.
    stats:
        Pre-fitted :class:`ScaleStats`.
    eps:
        Small constant to avoid division by zero on constant features.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Scaled ``(dose_seqs, cov_seqs)``.
    """
    dose_range = stats.dose_max - stats.dose_min + eps
    scaled_dose = (dose_seqs - stats.dose_min) / dose_range

    scaled_cov = cov_seqs.copy()
    time_range = stats.time_max - stats.time_min + eps
    gap_range = stats.time_gap_max - stats.time_gap_min + eps

    scaled_cov[:, :, 0] = (cov_seqs[:, :, 0] - stats.time_min) / time_range
    scaled_cov[:, :, 1] = (cov_seqs[:, :, 1] - stats.time_gap_min) / gap_range

    return scaled_dose.astype(np.float32), scaled_cov.astype(np.float32)


# ---------------------------------------------------------------------------
# build_dataloaders
# ---------------------------------------------------------------------------

def build_dataloaders(
    cfg: CancerTiDEConfig,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test DataLoaders from config.

    Protocols are split by ID (not by time) because each protocol is an
    independent sample.  A fixed random seed ensures reproducibility.

    Parameters
    ----------
    cfg:
        :class:`CancerTiDEConfig` instance.

    Returns
    -------
    tuple[DataLoader, DataLoader, DataLoader]
        Train, validation, test loaders.  Each loader's ``.dataset`` has a
        ``scale_stats`` attribute (:class:`ScaleStats`) attached.
    """
    if cfg.data_path is None:
        raise ValueError("data_path must be set in CancerTiDEConfig.")

    # ── Load and group by protocol ────────────────────────────────────────
    print("Loading CSV …")
    df = load_cancer_data(cfg.data_path)

    print("Extracting protocol arrays …")
    all_series = df["series"].unique()
    n_total = len(all_series)

    dose_all = np.zeros((n_total, cfg.input_len), dtype=np.float32)
    cov_all = np.zeros((n_total, cfg.input_len, 2), dtype=np.float32)
    target_all = np.zeros(n_total, dtype=np.float32)

    grouped = df.groupby("series")
    for i, sid in enumerate(all_series):
        group = grouped.get_group(sid)
        dose_seq, cov_seq, target_val = _extract_protocol_arrays(group, cfg.input_len)
        dose_all[i] = dose_seq
        cov_all[i] = cov_seq
        target_all[i] = target_val

    print(f"  Loaded {n_total:,} protocols.")

    # ── Split by protocol ID (shuffled, seed-fixed) ───────────────────────
    rng = np.random.default_rng(cfg.seed)
    indices = rng.permutation(n_total)

    n_train = int(n_total * cfg.train_frac)
    n_val = int(n_total * cfg.val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    train_dose, train_cov = dose_all[train_idx], cov_all[train_idx]
    val_dose, val_cov = dose_all[val_idx], cov_all[val_idx]
    test_dose, test_cov = dose_all[test_idx], cov_all[test_idx]

    # ── Fit scaler on training data only ──────────────────────────────────
    stats = _fit_scaler(train_dose, train_cov, target_all[train_idx])

    train_dose_s, train_cov_s = _apply_minmax(train_dose, train_cov, stats)
    val_dose_s, val_cov_s = _apply_minmax(val_dose, val_cov, stats)
    test_dose_s, test_cov_s = _apply_minmax(test_dose, test_cov, stats)

    # ── Build PyTorch Datasets ────────────────────────────────────────────
    train_ds = CancerProtocolDataset(
        train_dose_s, train_cov_s, target_all[train_idx], cfg.horizon
    )
    val_ds = CancerProtocolDataset(
        val_dose_s, val_cov_s, target_all[val_idx], cfg.horizon
    )
    test_ds = CancerProtocolDataset(
        test_dose_s, test_cov_s, target_all[test_idx], cfg.horizon
    )

    # Attach scale stats so callers can inverse-transform if needed
    for ds in (train_ds, val_ds, test_ds):
        ds.scale_stats = stats

    # ── Build DataLoaders ─────────────────────────────────────────────────
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,       # shuffle independent protocol samples
        drop_last=True,
        num_workers=0,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=pin,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=pin,
    )

    print(
        f"  Split — train: {len(train_ds):,}  "
        f"val: {len(val_ds):,}  "
        f"test: {len(test_ds):,}"
    )
    return train_loader, val_loader, test_loader
