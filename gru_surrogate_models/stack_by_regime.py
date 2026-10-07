"""Per-dose-count NNLS stacking of the predictions saved by eval_ensemble.py --save_preds.

Weights are fitted on the validation predictions only.
"""
import argparse

import numpy as np
from scipy.optimize import nnls

from cancer_nn.experiments_helpers import train_val_test_split
from cancer_nn.metrics import evaluate


def n_doses(df):
    rows = df[df.is_target == 0]
    return rows.groupby("series", sort=False).dose.apply(lambda s: (s > 0).sum()).values


def metrics(y, p):
    m = evaluate(y, p)
    return m["mae"], m["rmse"], 100 * m["percent"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="logs/next_batch_preds.npz")
    ap.add_argument("--csv_path", default="data/data_400k.csv")
    ap.add_argument("--bins", default="1-4,5-6,7-8,9-10,11-19", help="dose-count bins lo-hi")
    args = ap.parse_args()

    d = np.load(args.preds)
    if "val_preds" not in d:
        raise SystemExit("npz has no val_preds; re-run eval_ensemble.py with --with_val")
    P, y, V, yv, names = d["preds"], d["y_true"], d["val_preds"], d["val_true"], d["names"]
    _, val, test = train_val_test_split(args.csv_path, "series")
    nt, nv = n_doses(test), n_doses(val)
    assert len(nt) == len(y) and len(nv) == len(yv), "split order does not match the npz"

    mean = P.mean(0)
    gated = np.zeros_like(y)
    print(f"members: {list(names)}\n")
    for b in args.bins.split(","):
        lo, hi = map(int, b.split("-"))
        mt, mv = (nt >= lo) & (nt <= hi), (nv >= lo) & (nv <= hi)
        w, _ = nnls(V[:, mv].T, yv[mv])
        gated[mt] = w @ P[:, mt]
        print(
            f"doses {lo:2d}-{hi:2d}  n_test {mt.sum():5d}  mean {metrics(y[mt], mean[mt])[0]:.3f}"
            f"  gated {metrics(y[mt], gated[mt])[0]:.3f}  w={np.round(w, 2)}"
        )
    w_all, _ = nnls(V.T, yv)
    stacks = [
        ("plain mean", mean),
        ("global NNLS stack", w_all @ P),
        ("regime-gated stack", gated),
    ]
    for label, p in stacks:
        m = metrics(y, p)
        print(f"\n{label:<20} test mae={m[0]:.3f} rmse={m[1]:.3f} correct={m[2]:.2f}%")


if __name__ == "__main__":
    main()
