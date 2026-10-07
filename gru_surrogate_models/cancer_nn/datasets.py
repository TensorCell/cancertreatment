"""Script for torch dataset classes"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


class Dataset(torch.utils.data.Dataset):
    """Cancer torch dataset maker"""

    def __init__(
        self,
        data: pd.DataFrame,
        columns: List[str] = ["time", "dose", "time_gap"],
        target_column: str = "target",
        scalers: Optional[
            Dict[str, StandardScaler]
        ] = None,  # for training, for validation the same scalers as for training should be used
    ) -> None:
        """Init function

        Args:
        ------------
            data (pd.DataFrame): data frame witch protocls and target column
            columns (List[str], optional): list with features columns
            which will be taken to model. Defaults to ["time", "dose", "time_gap"].
            target_column (str, optional): target column name. Defaults to "target".
            scalers (Optional[ List[StandardScaler] ], optional): Dict with
            feature_name: fitted StandardScaler or None (new StandardScalers
            will be fitted). Defaults to None.
        """
        self.data = data
        self.columns = columns
        self.target_column = target_column
        self.indices = list(set(data.series.values))
        self.dataset_len = len(self.indices)
        self.series_len = 21
        self.target_scale = 500
        if not scalers:
            self.scalers = {
                column: StandardScaler().fit(self.data[column].values.reshape(-1, 1))
                for column in self.columns
            }
        else:
            self.scalers = scalers
        for column in self.columns:
            self.data[column] = self.scalers[column].transform(
                self.data[column].values.reshape(-1, 1)
            )
        self.data[self.target_column] = (
            self.data[self.target_column] / self.target_scale
        )

    def __len__(self) -> int:
        """Denotes the total number of samples

        Returns:
        ------------
            int: number of samples
        """
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Select ith series

        Args:
        ------------
            idx (int): sample index

        Returns:
        ------------
            Tuple[torch.FloatTensor, torch.FloatTensor]: (input_features, target)
        """
        x = self.data.iloc[
            self.series_len * idx : (self.series_len * idx) + self.series_len - 1
        ][self.columns].to_numpy()
        y = self.data.iloc[
            (self.series_len * idx)
            + self.series_len
            - 1 : (self.series_len * idx)
            + self.series_len
        ][self.target_column].to_numpy()
        return (torch.FloatTensor(x.astype(float)), torch.FloatTensor(y.astype(float)))


class DatasetMargin(Dataset):
    """Dataset that also returns a second series to rank against.

    With probability ``hard_pair_frac`` the partner is drawn from the series whose target
    is within ``hard_pair_min_diff``..``hard_pair_max_diff`` cells of this one's (the pairs
    the model actually gets wrong); otherwise it is uniformly random.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        columns: List[str] = ["time", "dose", "time_gap"],
        target_column: str = "target",
        scalers: Optional[Dict[str, StandardScaler]] = None,
        hard_pair_frac: float = 0.0,
        hard_pair_min_diff: float = 4.0,
        hard_pair_max_diff: float = 15.0,
    ) -> None:
        super().__init__(data, columns, target_column, scalers)
        self.hard_pair_frac = hard_pair_frac
        self.hard_pair_min_diff = hard_pair_min_diff / self.target_scale
        self.hard_pair_max_diff = hard_pair_max_diff / self.target_scale
        if hard_pair_frac > 0:
            # the target of series i is on the last row of its block
            self.targets = self.data[self.target_column].values[
                self.series_len - 1 :: self.series_len
            ].astype(float)
            self.sorted_idx = np.argsort(self.targets)
            self.sorted_targets = self.targets[self.sorted_idx]

    def partner_idx(self, idx: int) -> int:
        if self.hard_pair_frac > 0 and random.random() < self.hard_pair_frac:
            t = self.targets[idx]
            side = 1 if random.random() < 0.5 else -1
            lo, hi = sorted(
                (t + side * self.hard_pair_min_diff, t + side * self.hard_pair_max_diff)
            )
            a = np.searchsorted(self.sorted_targets, lo, side="left")
            b = np.searchsorted(self.sorted_targets, hi, side="right")
            if b - a == 0:  # nothing on that side, try the other one
                lo, hi = sorted(
                    (t - side * self.hard_pair_min_diff, t - side * self.hard_pair_max_diff)
                )
                a = np.searchsorted(self.sorted_targets, lo, side="left")
                b = np.searchsorted(self.sorted_targets, hi, side="right")
            if b - a > 0:
                return int(self.sorted_idx[random.randrange(a, b)])
        return random.randint(0, self.dataset_len - 1)

    def get_from_idx(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Select ith series

        Args:
        ------------
            idx (int): sample index

        Returns:
        ------------
            Tuple[torch.FloatTensor, torch.FloatTensor]: (input_features, target)
        """
        x = self.data.iloc[
            self.series_len * idx : (self.series_len * idx) + self.series_len - 1
        ][self.columns].to_numpy()
        y = self.data.iloc[
            (self.series_len * idx)
            + self.series_len
            - 1 : (self.series_len * idx)
            + self.series_len
        ][self.target_column].to_numpy()
        return (
            torch.FloatTensor(x.astype(float)),
            torch.FloatTensor(y.astype(float)),
        )

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor
    ]:
        """Select series by given idx
        and rand another sample

        Args:
        ------------
            idx (int): series index

        Returns:
        ------------
            Tuple[ torch.FloatTensor, torch.FloatTensor,
            torch.FloatTensor, torch.FloatTensor ]: (input_features, target,
            other_sample_input_features, other_sample_target)
        """
        x = self.data.iloc[
            self.series_len * idx : (self.series_len * idx) + self.series_len - 1
        ][self.columns].to_numpy()
        y = self.data.iloc[
            (self.series_len * idx)
            + self.series_len
            - 1 : (self.series_len * idx)
            + self.series_len
        ][self.target_column].to_numpy()
        another_sample = self.get_from_idx(self.partner_idx(idx))
        return (
            torch.FloatTensor(x.astype(float)),
            torch.FloatTensor(y.astype(float)),
            another_sample[0],
            another_sample[1],
        )
