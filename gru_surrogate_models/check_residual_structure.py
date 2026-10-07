"""Correlate a checkpoint's test residuals with coarse protocol features."""
import argparse

import numpy as np
import pandas as pd
import torch
from scipy import stats

from cancer_nn.experiments_helpers import (
    read_subnetworks_configs,
    read_yml_config,
    train_val_test_split,
)
from cancer_nn.torch_lightning_modules import CancerNet


def protocol_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, sub in df.groupby("series"):
        sub = sub.sort_values("time_idx")
        doses = sub["dose"].to_numpy()[:20]
        times = sub["time"].to_numpy()[:20]
        rows.append(
            {
                "series": sid,
                "total_dose": doses.sum(),
                "n_doses": int((doses > 0).sum()),
                "last_time": times[times > 0].max() if (times > 0).any() else 0.0,
                "mean_gap": (
                    np.diff(times[times > 0]).mean() if (times > 0).sum() > 1 else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="data/data_400k.csv")
    parser.add_argument("--config_folder", default="configs")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    args = parser.parse_args()

    train, val, test = train_val_test_split(args.csv_path, "series")
    config = read_yml_config(args.config_folder, args.config)
    config["network"]["config_list"] = read_subnetworks_configs(args.config_folder, config)

    network = CancerNet.load_from_checkpoint(
        checkpoint_path=args.ckpt,
        train_df=train,
        val_df=val,
        test_df=test,
        map_location="cpu",
        **config["network"],
    )
    network.eval()

    loader = network.test_dataloader()
    all_true, all_pred, all_series = [], [], []
    test_indices = sorted(network.test_set.indices)
    with torch.no_grad():
        offset = 0
        for batch in loader:
            out = network.model(batch)
            n = batch[1].shape[0]
            all_true.append(batch[1].flatten() * network.target_scale)
            all_pred.append(out[:, 0].flatten() * network.target_scale)
            all_series.extend(test_indices[offset : offset + n])
            offset += n

    y_true = torch.cat(all_true).numpy()
    y_pred = torch.cat(all_pred).numpy()
    residual = np.abs(y_true - y_pred)

    res_df = pd.DataFrame({"series": all_series, "residual": residual, "y_true": y_true})
    feats = protocol_features(test)
    merged = res_df.merge(feats, on="series", how="left")

    print(f"n test series: {len(merged)}")
    print(f"MAE: {merged['residual'].mean():.4f}")
    print()
    print("Pearson correlation of |residual| with protocol features:")
    for col in ["total_dose", "n_doses", "last_time", "mean_gap", "y_true"]:
        r, p = stats.pearsonr(merged[col], merged["residual"])
        print(f"  residual vs {col:<12}: r={r:+.3f}  p={p:.4g}")

    print()
    print("Residual by n_doses bucket (mean |residual|, count):")
    print(merged.groupby("n_doses")["residual"].agg(["mean", "count"]))


if __name__ == "__main__":
    main()
