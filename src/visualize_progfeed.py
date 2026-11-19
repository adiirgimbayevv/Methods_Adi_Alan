"""Create publication-ready plots from the real ProgFeed analysis."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_progfeed import DEFAULT_INPUT, submission_outcomes, summarize

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "figures"
ORDER = ["no_feedback", "tc", "nl"]
LABELS = ["No feedback", "Test-case feedback", "Natural-language feedback"]
COLORS = ["#0072B2", "#E69F00", "#009E73"]


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_figures(data_path: Path = DEFAULT_INPUT) -> None:
    units = submission_outcomes(data_path)
    summary = summarize(units)
    summary = summary.set_index("feedback_type").reindex(ORDER).reset_index()
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})

    # Fig. 3: per-student cluster-bootstrap CI for score gain after first failure.
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    means = summary["mean_score_gain_pp_after_first_failure"].to_numpy()
    lo = summary["score_gain_cluster_ci95_low"].to_numpy()
    hi = summary["score_gain_cluster_ci95_high"].to_numpy()
    yerr = np.vstack([means - lo, hi - means])
    ax.errorbar(np.arange(3), means, yerr=yerr, fmt="o", markersize=8,
                color="#222222", ecolor=COLORS[1], capsize=5, linewidth=2)
    ax.set_xticks(np.arange(3), LABELS, rotation=10, ha="right")
    ax.set_ylabel("Score gain (percentage points)")
    ax.set_title("Programming score gain after an initial failed submission")
    ax.text(0.01, 0.99, "Randomized labs only; 95% CI resamples students",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, "fig3_feedback_score_gain")

    # Fig. 4: distribution of final student-lab scores, with notched medians.
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    series = [units.loc[units["feedback_type"] == arm, "final_score_pct"].dropna().to_numpy()
              for arm in ORDER]
    bp = ax.boxplot(series, tick_labels=LABELS, notch=True, patch_artist=True, showmeans=True,
                    meanprops={"marker": "D", "markerfacecolor": "white",
                               "markeredgecolor": "black", "markersize": 5},
                    flierprops={"marker": ".", "markersize": 3, "alpha": 0.3})
    for patch, color, hatch in zip(bp["boxes"], COLORS, ["", "//", "xx"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
        patch.set_hatch(hatch)
    ax.set_ylabel("Final submission score (% of available test points)")
    ax.set_title("Final score distribution by assigned feedback condition")
    ax.set_ylim(-3, 103)
    ax.tick_params(axis="x", rotation=10)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save(fig, "fig4_final_score_distribution")


if __name__ == "__main__":
    make_figures()
