"""Scientific figures for local data-processing benchmark measurements."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOG = logging.getLogger("visualize")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
COLORS = ["#0072B2", "#D55E00"]
MARKERS = ["o", "s"]
LINESTYLES = ["-", "--"]


def publication_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.labelsize": 10.5, "axes.labelweight": "bold",
        "legend.fontsize": 9, "lines.linewidth": 2,
        "lines.markersize": 6, "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig1_latency_vs_load(summary: pd.DataFrame,
                         out_dir: Path = ROOT / "reports/figures") -> None:
    publication_style()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    variants = list(summary["variant"].drop_duplicates())
    for i, variant in enumerate(variants):
        group = summary[summary["variant"] == variant].sort_values("load_level")
        x = group["input_rows"].to_numpy()
        y = group["mean"].to_numpy()
        low = group["ci95_low"].to_numpy()
        high = group["ci95_high"].to_numpy()
        ax.plot(x, y, color=COLORS[i], marker=MARKERS[i], linestyle=LINESTYLES[i], label=variant)
        ax.fill_between(x, low, high, color=COLORS[i], alpha=0.18)
    ax.set_title("Aggregation latency by input size")
    ax.set_xlabel("Input size (graded-test rows)")
    ax.set_ylabel("Aggregation latency (ms)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig1_latency_vs_load")


def fig2_distribution_boxplot(frame: pd.DataFrame,
                              out_dir: Path = ROOT / "reports/figures") -> None:
    publication_style()
    variants = list(frame["variant"].drop_duplicates())
    loads = sorted(frame["load_level"].unique())
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    positions = []
    series = []
    colors_for_boxes = []
    labels = []
    group_width = 0.72
    for load_index, load in enumerate(loads):
        for variant_index, variant in enumerate(variants):
            sample = frame[(frame["load_level"] == load)
                           & (frame["variant"] == variant)
                           & (frame["status"] == "ok")]["latency_ms"].dropna().to_numpy()
            position = load_index + (variant_index - (len(variants) - 1) / 2) * group_width / len(variants)
            positions.append(position)
            series.append(sample)
            colors_for_boxes.append(COLORS[variant_index])
            labels.append(str(load))
    boxes = ax.boxplot(series, positions=positions, widths=group_width / len(variants) * 0.82,
                       notch=True, patch_artist=True, showmeans=True, manage_ticks=False,
                       meanprops={"marker": "D", "markerfacecolor": "white",
                                  "markeredgecolor": "black", "markersize": 4},
                       flierprops={"marker": ".", "markersize": 2.5, "alpha": 0.35})
    for index, patch in enumerate(boxes["boxes"]):
        patch.set_facecolor(colors_for_boxes[index])
        patch.set_alpha(0.62)
        patch.set_hatch("//" if index % len(variants) else "")
    centers = np.arange(len(loads))
    ax.set_xticks(centers, [str(load) for load in loads])
    ax.set_title("Latency distribution across input sizes")
    ax.set_xlabel("Input size (graded-test rows)")
    ax.set_ylabel("Aggregation latency (ms)")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    handles = [plt.Line2D([0], [0], color=COLORS[i], marker=MARKERS[i],
                           linestyle="None", label=variants[i]) for i in range(len(variants))]
    ax.legend(handles=handles, frameon=True)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig2_distribution_boxplot")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "data/processed/metrics_summary.csv")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/test_run.jsonl")
    parser.add_argument("--outdir", type=Path, default=ROOT / "reports/figures")
    args = parser.parse_args()
    summary = pd.read_csv(args.summary)
    with args.raw.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    frame = pd.DataFrame(records)
    frame = frame[frame["phase"] == "benchmark"].copy()
    fig1_latency_vs_load(summary, args.outdir)
    fig2_distribution_boxplot(frame, args.outdir)
    LOG.info("Exported measured benchmark figures to %s", args.outdir)


if __name__ == "__main__":
    main()
