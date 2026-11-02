"""Create Figure 7 from saved 24-hour and 10-hour evaluation summaries.

- Plots baselines/models results for normal/10h modes.
- Saves a comparison.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import seaborn as sns


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
DEFAULT_24H = RESULTS_DIR / "fig03_summary_final_ppo_vs_reference_protocols_24h_30x10.csv"
DEFAULT_10H = RESULTS_DIR / "fig07_summary_10h_treatment_window_30x10.csv"
DEFAULT_OUTPUT = RESULTS_DIR / "fig07_treatment_window_effect_24h_vs_10h_30x10.png"
EXCLUDED_MODES = {"paper_ppo", "random", "no_treatment"}


def load_summary(path: Path) -> dict[str, dict[str, str]]:
    required = {
        "mode",
        "protocol",
        "mean_leftover_cells",
        "std_leftover_cells",
    }
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = {
            row["mode"]: row
            for row in reader
            if row["mode"] not in EXCLUDED_MODES
        }
    if not rows:
        raise ValueError(f"{path} contains no included protocols")
    return rows


def create_plot(hours_24_path: Path, hours_10_path: Path):
    hours_24 = load_summary(hours_24_path)
    hours_10 = load_summary(hours_10_path)
    if hours_24.keys() != hours_10.keys():
        only_24h = sorted(hours_24.keys() - hours_10.keys())
        only_10h = sorted(hours_10.keys() - hours_24.keys())
        raise ValueError(
            "The summaries contain different protocols after filtering: "
            f"24-hour only={only_24h}, 10-hour only={only_10h}"
        )

    
    order = sorted(
        hours_24, key=lambda mode: float(hours_24[mode]["mean_leftover_cells"])
    )
    labels = [hours_24[mode]["protocol"] for mode in order]
    means_24h = np.asarray(
        [float(hours_24[mode]["mean_leftover_cells"]) for mode in order]
    )
    means_10h = np.asarray(
        [float(hours_10[mode]["mean_leftover_cells"]) for mode in order]
    )
    std_24h = np.asarray(
        [float(hours_24[mode]["std_leftover_cells"]) for mode in order]
    )
    std_10h = np.asarray(
        [float(hours_10[mode]["std_leftover_cells"]) for mode in order]
    )

    sns.set_theme(context="paper", style="whitegrid", font="DejaVu Sans")
    colors = sns.color_palette("colorblind", 2)
    x = np.arange(len(order))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))
    error_style = {"elinewidth": 1.15, "capthick": 1.15}
    ax.bar(
        x - width / 2,
        means_24h,
        width,
        yerr=std_24h,
        capsize=3,
        color=colors[0],
        edgecolor="#333333",
        linewidth=0.7,
        error_kw=error_style,
        label="24-hour treatment window",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        means_10h,
        width,
        yerr=std_10h,
        capsize=3,
        color=colors[1],
        edgecolor="#333333",
        linewidth=0.7,
        hatch="//",
        error_kw=error_style,
        label="10-hour treatment window",
        zorder=3,
    )

    ax.set_xlabel("Model / protocol (ordered by mean for the 24-hour treatment window)")
    ax.set_ylabel("Mean leftover cancer cells")
    ax.set_xticks(x, labels, rotation=28, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.grid(axis="x", visible=False)
    ax.grid(axis="y", which="major", linestyle="--", alpha=0.55, zorder=0)
    ax.grid(axis="y", which="minor", linestyle="--", alpha=0.22, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return fig


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours-24", type=Path, default=DEFAULT_24H)
    parser.add_argument("--hours-10", type=Path, default=DEFAULT_10H)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    figure = create_plot(args.hours_24, args.hours_10)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        args.output,
        dpi=args.dpi,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Title": "Normal baseline comparison for 24-hour and 10-hour treatment windows"},
    )
    plt.close(figure)
    print(f"Saved treatment-window comparison to {args.output}")


if __name__ == "__main__":
    main()
