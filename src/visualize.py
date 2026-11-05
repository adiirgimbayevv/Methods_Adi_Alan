"""Scientific visualization module (src/visualize.py).

Combines Domain Research Outcomes (Critical Thinking & Academic Performance under GenAI modes)
with System Benchmark Metrics (Response Latency p50/p95 & Throughput) across two multi-panel figures,
strictly enforcing the 3 Core Rules of Scientific Visualization:
  1. Explicit physical units on all axes (`AI Usage Intensity (%)`, `Score (points, 0-100)`,
     `Workload Intensity (RPS)`, `Response Latency (ms)`).
  2. Uncertainty quantification via shaded 95% bootstrapped confidence bands and notched median
     IQR (p25-p75) boxplots.
  3. Colorblind-safe palette (`sns.color_palette("colorblind")`) + distinct markers (`o`, `s`, `^`, `D`),
     line styles (`-`, `--`, `-.`, `:`), and box hatching for monochrome / B&W print legibility.
"""
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
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("visualize")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def apply_publication_style() -> None:
    sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.15)
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 12.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 11.5,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 9.8,
        "legend.title_fontsize": 10.2,
        "lines.linewidth": 2.2,
        "lines.markersize": 7.0,
    })


PALETTE = sns.color_palette("colorblind")


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        log.info("Exported figure -> %s", path)
    plt.close(fig)


def fig1_latency_vs_load(summary: pd.DataFrame,
                         out_dir: Path = ROOT / "reports" / "figures") -> None:
    """Figure 1: (a) Domain Outcome: Critical Thinking & Final Grade vs. AI Intensity (%) with 95% CI;
    (b) Pipeline Benchmark: Response Latency (Mean + 95% CI & p95 Tail) vs. Workload (RPS)."""
    apply_publication_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.2))

    variants = sorted(summary["variant"].unique())
    markers = ["o", "s", "^", "D"]

    # Panel (a): Critical Thinking Score & Final Grade vs. AI Usage Intensity (%)
    for idx, var in enumerate(variants):
        sub = summary[summary["variant"] == var].sort_values("load_level")
        x_ai = sub["ai_share_mean"].values
        y_ct = sub["ct_mean"].values
        ct_lo = sub["ct_ci95_low"].values
        ct_hi = sub["ct_ci95_high"].values
        y_gr = sub["grade_mean"].values
        gr_lo = sub["grade_ci95_low"].values
        gr_hi = sub["grade_ci95_high"].values
        color = PALETTE[idx]

        ax1.plot(
            x_ai, y_ct,
            color=color, linestyle="-", marker=markers[idx],
            label=f"{var} — Critical Thinking (95% CI)",
        )
        ax1.fill_between(x_ai, ct_lo, ct_hi, color=color, alpha=0.22)

        ax1.plot(
            x_ai, y_gr,
            color=color, linestyle="--", marker=markers[idx + 2], alpha=0.88,
            label=f"{var} — Final Grade (95% CI)",
        )
        ax1.fill_between(x_ai, gr_lo, gr_hi, color=color, alpha=0.12)

    ax1.set_title("(a) RQ2 & RQ3: Critical Thinking & Grade vs. AI Intensity")
    ax1.set_xlabel("GenAI Usage Intensity / Assisted Code Steps (%)")
    ax1.set_ylabel("Assessment Score (points, 0–100)")
    ax1.set_ylim(0, 100)
    ax1.legend(loc="lower left", frameon=True)

    # Panel (b): Pipeline Latency (Mean + 95% CI & p95 Tail) vs. Workload (RPS)
    linestyles_mean = ["-", "--"]
    linestyles_p95 = ["-.", ":"]
    for idx, var in enumerate(variants):
        sub = summary[summary["variant"] == var].sort_values("load_level")
        x_rps = sub["load_level"].values
        y_lat = sub["mean"].values
        lat_lo = sub["ci95_low"].values
        lat_hi = sub["ci95_high"].values
        y_p95 = sub["p95"].values
        color = PALETTE[idx]

        ax2.plot(
            x_rps, y_lat,
            color=color, linestyle=linestyles_mean[idx], marker=markers[idx],
            label=f"{var} — Mean Latency (95% CI)",
        )
        ax2.fill_between(x_rps, lat_lo, lat_hi, color=color, alpha=0.22)

        ax2.plot(
            x_rps, y_p95,
            color=color, linestyle=linestyles_p95[idx], marker=markers[idx + 2], alpha=0.85,
            label=f"{var} — p95 Tail Latency",
        )

    ax2.set_title("(b) Pipeline Response Latency vs. Workload Intensity")
    ax2.set_xlabel("Request Rate / Workload Intensity (RPS)")
    ax2.set_ylabel("Response Latency (ms)")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="upper left", frameon=True)

    sns.despine(fig=fig)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig1_latency_vs_load")


def fig2_distribution_boxplot(df_bench: pd.DataFrame,
                              out_dir: Path = ROOT / "reports" / "figures") -> None:
    """Figure 2: Two-panel Notched Boxplot & IQR comparison:
    (a) Critical Thinking Score (ct_score, points) by GenAI Mode & Intensity Tier;
    (b) Pipeline Response Latency (latency_ms, ms) by Workload Intensity (RPS)."""
    apply_publication_style()
    df_ok = df_bench[df_bench["status"] == "ok"].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.2))
    hatches = ["", "//"]

    # Panel (a): Critical Thinking Score Boxplot
    sns.boxplot(
        data=df_ok,
        x="load_level",
        y="ct_score",
        hue="variant",
        palette="colorblind",
        notch=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 5.5},
        flierprops={"marker": "o", "markersize": 3.2, "alpha": 0.45},
        ax=ax1,
    )
    patches1 = [p for p in ax1.patches if isinstance(p, matplotlib.patches.PathPatch)]
    for i, patch in enumerate(patches1):
        patch.set_hatch(hatches[i % len(hatches)])
        patch.set_alpha(0.82)

    ax1.set_title("(a) Critical Thinking Score Distribution (RQ1 & RQ3)")
    ax1.set_xlabel("AI Usage Tier / Workload Level (RPS)")
    ax1.set_ylabel("Critical Thinking Score (points, 0–100)")
    ax1.set_ylim(0, 100)
    ax1.legend(title="GenAI Mode (♦ = Mean, Notch = 95% CI)", loc="lower left", frameon=True)

    # Panel (b): Pipeline Latency Boxplot
    sns.boxplot(
        data=df_ok,
        x="load_level",
        y="latency_ms",
        hue="variant",
        palette="colorblind",
        notch=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "white", "markeredgecolor": "black", "markersize": 5.5},
        flierprops={"marker": "o", "markersize": 3.2, "alpha": 0.45},
        ax=ax2,
    )
    patches2 = [p for p in ax2.patches if isinstance(p, matplotlib.patches.PathPatch)]
    for i, patch in enumerate(patches2):
        patch.set_hatch(hatches[i % len(hatches)])
        patch.set_alpha(0.82)

    ax2.set_title("(b) Pipeline Latency Distribution & Tail Outliers")
    ax2.set_xlabel("Request Rate / Workload Intensity (RPS)")
    ax2.set_ylabel("Response Latency (ms)")
    ax2.set_ylim(bottom=0)
    ax2.legend(title="System Variant (♦ = Mean, Notch = 95% CI)", loc="upper left", frameon=True)

    sns.despine(fig=fig)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig2_distribution_boxplot")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default=str(ROOT / "data" / "processed" / "metrics_summary.csv"))
    ap.add_argument("--raw", default=str(ROOT / "data" / "raw" / "test_run.jsonl"))
    ap.add_argument("--outdir", default=str(ROOT / "reports" / "figures"))
    args = ap.parse_args()

    summary = pd.read_csv(args.summary)
    out_dir = Path(args.outdir)

    records = []
    with open(args.raw, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    df_raw = pd.DataFrame(records)
    df_bench = df_raw[df_raw["phase"].isin({"benchmark", "eval"})].copy()

    fig1_latency_vs_load(summary, out_dir=out_dir)
    fig2_distribution_boxplot(df_bench, out_dir=out_dir)
    log.info("All publication-ready figures exported to %s", out_dir)


if __name__ == "__main__":
    main()
