"""Write nested whole-series subsets of a data csv (data_400k.csv -> data_400k_25pct.csv)."""
import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="data/data_400k.csv")
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    all_series = sorted(df["series"].unique())
    n_total = len(all_series)

    # one fixed random order; every subset is a prefix of it, so they nest
    ordered_series = pd.Series(all_series).sample(frac=1.0, random_state=args.seed).to_numpy()

    for frac in sorted(args.fractions):
        n_keep = int(n_total * frac)
        keep_series = set(ordered_series[:n_keep])
        subset = df[df["series"].isin(keep_series)]
        src = Path(args.csv_path)
        out_path = src.parent / f"{src.stem}_{int(frac * 100)}pct.csv"
        subset.to_csv(out_path, index=False)
        print(f"{out_path}: {n_keep}/{n_total} series ({frac:.0%}), {len(subset)} rows")


if __name__ == "__main__":
    main()
