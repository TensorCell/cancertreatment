"""Create Figure 6 from saved normal and treatment-delay summaries.

- Plots baselines/models results for normal/treatment-delay modes.
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
DEFAULT_NORMAL = RESULTS_DIR / "fig03_summary_final_ppo_vs_reference_protocols_24h_30x10.csv"
DEFAULT_TREATMENT_DELAY = RESULTS_DIR / "fig06_summary_with_treatment_delay_24h_30x10.csv"
DEFAULT_OUTPUT = RESULTS_DIR / "fig06_treatment_delay_effect_24h_30x10.png"
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


def create_plot(normal_path: Path, treatment_delay_path: Path):
    normal = load_summary(normal_path)
    treatment_delay = load_summary(treatment_delay_path)
    if normal.keys() != treatment_delay.keys():
        only_normal = sorted(normal.keys() - treatment_delay.keys())
        only_treatment_delay = sorted(treatment_delay.keys() - normal.keys())
        raise ValueError(
            "The summaries contain different protocols after filtering: "
            f"normal only={only_normal}, "
            f"treatment-delay only={only_treatment_delay}"
        )

    
    order = sorted(normal, key=lambda mode: float(normal[mode]["mean_leftover_cells"]))
    labels = [normal[mode]["protocol"] for mode in order]
    normal_means = np.asarray(
        [float(normal[mode]["mean_leftover_cells"]) for mode in order]
    )
    treatment_delay_means = np.asarray(
        [float(treatment_delay[mode]["mean_leftover_cells"]) for mode in order]
    )
    normal_std = np.asarray(
        [float(normal[mode]["std_leftover_cells"]) for mode in order]
    )
    treatment_delay_std = np.asarray(
        [float(treatment_delay[mode]["std_leftover_cells"]) for mode in order]
    )

    sns.set_theme(context="paper", style="whitegrid", font="DejaVu Sans")
    colors = sns.color_palette("colorblind", 2)
    x = np.arange(len(order))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))
    error_style = {"elinewidth": 1.15, "capthick": 1.15}
    ax.bar(
        x - width / 2,
        normal_means,
        width,
        yerr=normal_std,
        capsize=3,
        color=colors[0],
        edgecolor="#333333",
        linewidth=0.7,
        error_kw=error_style,
        label="No treatment delay",
        zorder=3,
    )
    ax.bar(
        x + width / 2,
        treatment_delay_means,
        width,
        yerr=treatment_delay_std,
        capsize=3,
        color=colors[1],
        edgecolor="#333333",
        linewidth=0.7,
        hatch="//",
        error_kw=error_style,
        label="Treatment delay",
        zorder=3,
    )

    ax.set_xlabel("Model / protocol (ordered by mean without treatment delay)")
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
    parser.add_argument("--normal", type=Path, default=DEFAULT_NORMAL)
    parser.add_argument(
        "--treatment-delay",
        type=Path,
        default=DEFAULT_TREATMENT_DELAY,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    figure = create_plot(args.normal, args.treatment_delay)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        args.output,
        dpi=args.dpi,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Title": "Baseline comparison with and without treatment delays"},
    )
    plt.close(figure)
    print(f"Saved baseline comparison to {args.output}")


if __name__ == "__main__":
    main()
