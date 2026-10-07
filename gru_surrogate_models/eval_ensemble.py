"""Score trained checkpoints on the test split, alone and as prediction averages."""
import argparse
import sys

import numpy as np
import torch

from cancer_nn.experiments_helpers import (
    read_subnetworks_configs,
    read_yml_config,
    train_val_test_split,
)
from cancer_nn.metrics import evaluate
from cancer_nn.torch_lightning_modules import CancerNet


def predict(config_folder, config_name, ckpt, train, val, test, device, split="test"):
    config = read_yml_config(config_folder, config_name)
    config["network"]["config_list"] = read_subnetworks_configs(config_folder, config)
    net = CancerNet.load_from_checkpoint(
        checkpoint_path=ckpt,
        train_df=train.copy(),
        val_df=val.copy(),
        test_df=test.copy(),
        map_location="cpu",
        **config["network"],
    )
    net.eval().to(device)
    loader = net.val_dataloader() if split == "val" else net.test_dataloader()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            batch = [b.to(device) for b in batch]
            out = net.model(batch)
            y_true.append(batch[1].flatten().cpu())
            y_pred.append(out[:, 0].flatten().cpu())
    scale = net.target_scale
    return torch.cat(y_true).numpy() * scale, torch.cat(y_pred).numpy() * scale


def fmt(name, m, val=None, ci=None):
    line = (
        f"{name:<42} mae={m['mae']:.3f}  rmse={m['rmse']:.3f}  "
        f"mape={100 * m['mape']:.3f}%  correct={100 * m['percent']:.2f}%  "
        f"err99={m['max_error_99']:.2f}"
    )
    if ci is not None:
        line += (
            f"  | 95% CI mae [{ci['mae'][0]:.3f}, {ci['mae'][1]:.3f}]"
            f" correct [{100 * ci['percent'][0]:.2f}%, {100 * ci['percent'][1]:.2f}%]"
        )
    if val is not None:
        line += f"  | val mae={val['mae']:.3f} correct={100 * val['percent']:.2f}%"
    return line


def bootstrap_ci(y_true, y_pred, n, seed=0):
    """95% bootstrap CI of MAE and correct fraction, resampling series."""
    rng = np.random.default_rng(seed)
    maes, pcts = [], []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        m = evaluate(y_true[idx], y_pred[idx])
        maes.append(m["mae"])
        pcts.append(m["percent"])
    return {
        "mae": tuple(np.percentile(maes, [2.5, 97.5])),
        "percent": tuple(np.percentile(pcts, [2.5, 97.5])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="data/data_400k.csv")
    parser.add_argument("--config_folder", default="configs")
    parser.add_argument(
        "--member", action="append", required=True, help="config.yml:checkpoint.ckpt"
    )
    parser.add_argument("--group", action="append", default=[], help="NAME=i,j,k")
    parser.add_argument("--save_preds", default=None, help="optional .npz output")
    parser.add_argument("--with_val", action="store_true", help="also score the val split")
    parser.add_argument("--bootstrap", type=int, default=0, help="bootstrap resamples (0 = off)")
    parser.add_argument(
        "--train_on_val",
        action="append",
        default=[],
        type=int,
        help="member index trained with -train_on_val (val score is in-sample)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val, test = train_val_test_split(args.csv_path, "series")

    preds, val_preds, names, y_true, yv_true = [], [], [], None, None
    for i, member in enumerate(args.member):
        config_name, ckpt = member.split(":")
        yt, yp = predict(args.config_folder, config_name, ckpt, train, val, test, device)
        if y_true is None:
            y_true = yt
        elif not np.allclose(y_true, yt):
            sys.exit(f"test targets differ for {member}; test order is not stable")
        preds.append(yp)
        names.append(f"{config_name.split('/')[-1][:-4]}:{ckpt.split('/')[-1][:-5]}")
        val_m = None
        if args.with_val:
            yv, vp = predict(
                args.config_folder, config_name, ckpt, train, val, test, device, split="val"
            )
            yv_true = yv if yv_true is None else yv_true
            val_preds.append(vp)
            val_m = evaluate(yv_true, vp)
            if i in args.train_on_val:
                names[-1] += " (val in-sample)"
        print(fmt(names[-1], evaluate(y_true, yp), val=val_m), flush=True)

    preds = np.stack(preds)
    print()
    all_idx = ",".join(map(str, range(len(preds))))
    groups = args.group or [f"avg of all {len(preds)}={all_idx}"]
    for group in groups:
        name, idx = group.split("=")
        idx = [int(i) for i in idx.split(",")]
        yp = preds[idx].mean(0)
        val_m = evaluate(yv_true, np.stack(val_preds)[idx].mean(0)) if args.with_val else None
        ci = bootstrap_ci(y_true, yp, args.bootstrap) if args.bootstrap > 0 else None
        print(fmt(f"[{name}]", evaluate(y_true, yp), val=val_m, ci=ci))

    if args.save_preds:
        extra = {"val_true": yv_true, "val_preds": np.stack(val_preds)} if args.with_val else {}
        np.savez(args.save_preds, y_true=y_true, preds=preds, names=np.array(names), **extra)


if __name__ == "__main__":
    main()
