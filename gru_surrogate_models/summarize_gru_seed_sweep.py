"""Mean +/- std per config and pairwise Welch t-tests for the seed sweep manifest."""
import csv
import itertools
import json
import statistics
import sys
from pathlib import Path

from scipy import stats

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "logs" / "gru_seed_sweep_manifest.csv"
METRICS = ["test/mae", "test/rmse", "test/mape", "test/percent", "test/max_error_99"]
LOWER_IS_BETTER = {
    "test/mae": True,
    "test/rmse": True,
    "test/mape": True,
    "test/percent": False,
    "test/max_error_99": True,
}


def load_run_summary(run_id):
    matches = list((ROOT / "wandb").glob(f"*-{run_id}"))
    if not matches:
        return None
    summary_path = matches[0] / "files" / "wandb-summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text())


def main():
    if not MANIFEST.exists():
        print(f"ERROR: {MANIFEST} not found; run run_gru_seed_sweep.sh first", file=sys.stderr)
        sys.exit(1)

    rows = list(csv.DictReader(MANIFEST.open()))
    by_config = {}
    for row in rows:
        if row["status"] != "OK" or not row["run_id"]:
            continue
        summary = load_run_summary(row["run_id"])
        if summary is None:
            print(f"WARNING: no wandb-summary.json for run_id={row['run_id']} "
                  f"({row['config']}, seed={row['seed']})", file=sys.stderr)
            continue
        by_config.setdefault(row["config"], []).append(
            {"seed": row["seed"], "run_id": row["run_id"], **summary}
        )

    if not by_config:
        print("No usable runs found.", file=sys.stderr)
        sys.exit(1)

    print(f"{'config':<45} {'n':>3}  " + "  ".join(f"{m:>14}" for m in METRICS))
    for config, runs in sorted(by_config.items()):
        n = len(runs)
        line = f"{config:<45} {n:>3}  "
        for metric in METRICS:
            values = [r[metric] for r in runs if metric in r]
            if not values:
                line += f"{'--':>14}  "
                continue
            mean = statistics.mean(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            line += f"{mean:>7.4f}+-{std:<5.4f} "
        print(line)

    print("\nPairwise comparisons (Welch's t-test, test/mae and test/percent):")
    for (c1, c2) in itertools.combinations(sorted(by_config), 2):
        r1, r2 = by_config[c1], by_config[c2]
        if len(r1) < 2 or len(r2) < 2:
            print(f"  {c1} vs {c2}: not enough seeds yet (n={len(r1)}, n={len(r2)})")
            continue
        for metric in ("test/mae", "test/percent"):
            v1 = [r[metric] for r in r1 if metric in r]
            v2 = [r[metric] for r in r2 if metric in r]
            if len(v1) < 2 or len(v2) < 2:
                continue
            t, p = stats.ttest_ind(v1, v2, equal_var=False)
            m1, m2 = statistics.mean(v1), statistics.mean(v2)
            lower_better = LOWER_IS_BETTER[metric]
            better = c1 if (m1 < m2) == lower_better else c2
            sig = "significant (p<0.05)" if p < 0.05 else "not significant"
            print(f"  {c1} vs {c2} [{metric}]: {m1:.4f} vs {m2:.4f}, "
                  f"p={p:.4f} ({sig}, {better} better)")


if __name__ == "__main__":
    main()
