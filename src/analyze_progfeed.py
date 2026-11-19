"""Prepare student-by-lab outcomes from the public ProgFeed dataset.

The analysis is restricted to the four labs whose feedback arm was randomized.
Test-level rows are first collapsed to one row per submission, then to one row
per student and lab. Feedback assignment and feedback actually delivered are
kept as separate variables because hints were shown only after failed tests.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data/raw/progfeed/all_submissions_consolidated.csv"
DEFAULT_OUTPUT = ROOT / "data/processed/student_outcomes_summary.csv"
RANDOMIZED_LABS = {"lab05", "lab06", "lab07", "lab09"}
ARMS = ("no_feedback", "tc", "nl")


def submission_outcomes(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, keep_default_na=False)
    required = {
        "student_id", "lab", "submission_timestamp", "score", "max_score",
        "ai_feedback_type", "ai_feedback_text", "status", "consented",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"ProgFeed schema is missing columns: {sorted(missing)}")

    data = data[(data["consented"].astype(str).str.lower() == "true")
                & data["lab"].isin(RANDOMIZED_LABS)].copy()
    data["score"] = pd.to_numeric(data["score"], errors="coerce")
    data["max_score"] = pd.to_numeric(data["max_score"], errors="coerce")
    keys = ["student_id", "lab", "submission_timestamp"]
    submissions = data.groupby(keys, as_index=False).agg(
        score=("score", "sum"), max_score=("max_score", "sum"),
        feedback_type=("ai_feedback_type", "first"),
        feedback_delivered=("ai_feedback_text", lambda s: s.astype(str).str.strip().ne("").any()),
        failed_tests=("status", lambda s: s.astype(str).str.lower().eq("failed").sum()),
        n_tests=("status", "size"),
    )
    submissions = submissions[submissions["max_score"] > 0].copy()
    submissions["score_pct"] = 100 * submissions["score"] / submissions["max_score"]
    submissions = submissions.sort_values(keys).reset_index(drop=True)
    submissions["attempt"] = submissions.groupby(["student_id", "lab"]).cumcount() + 1

    # Arm assignment should be stable within a student-lab unit.
    arm_counts = submissions.groupby(["student_id", "lab"])["feedback_type"].nunique()
    if (arm_counts > 1).any():
        raise ValueError("Feedback assignment changes within a student-lab unit")

    units = submissions.groupby(["student_id", "lab"], as_index=False).agg(
        feedback_type=("feedback_type", "first"), attempts=("attempt", "max"),
        first_score_pct=("score_pct", "first"), final_score_pct=("score_pct", "last"),
        first_failed_tests=("failed_tests", "first"),
        feedback_delivered=("feedback_delivered", "max"),
    )
    full_pass = submissions.assign(full_pass=submissions["score_pct"] >= 99.999)
    first_pass = (full_pass[full_pass["full_pass"]]
                  .groupby(["student_id", "lab"], as_index=False)["attempt"].min()
                  .rename(columns={"attempt": "attempts_to_full_pass"}))
    units = units.merge(first_pass, on=["student_id", "lab"], how="left")
    units["first_attempt_failed"] = units["first_score_pct"] < 99.999
    units["eventual_full_pass"] = units["attempts_to_full_pass"].notna()
    units["score_gain_pp"] = units["final_score_pct"] - units["first_score_pct"]
    return units


def cluster_bootstrap_ci(units: pd.DataFrame, value: str, seed: int = 42,
                         n_resamples: int = 5000) -> tuple[float, float]:
    """Percentile CI from resampling students, retaining each student's labs."""
    clusters = {
        student: group[value].dropna().to_numpy(dtype=float)
        for student, group in units.groupby("student_id")
    }
    clusters = {student: values for student, values in clusters.items() if len(values)}
    if len(clusters) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    student_ids = np.asarray(list(clusters))
    draws = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        sampled = rng.choice(student_ids, size=len(student_ids), replace=True)
        # Keep every observed lab for each selected student. The mean remains
        # student-lab weighted, matching the point estimate in the summary.
        values = np.concatenate([clusters[student] for student in sampled])
        draws[i] = values.mean()
    return tuple(float(x) for x in np.quantile(draws, [0.025, 0.975]))


def summarize(units: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    failed_first = units[units["first_attempt_failed"]]
    rows = []
    for arm in ARMS:
        group = units[units["feedback_type"] == arm]
        failed = failed_first[failed_first["feedback_type"] == arm]
        lo, hi = cluster_bootstrap_ci(failed, "score_gain_pp", seed=seed)
        rows.append({
            "feedback_type": arm,
            "student_lab_units": int(len(group)),
            "first_attempt_failed": int(len(failed)),
            "retry_rate_after_first_failure": float((failed["attempts"] > 1).mean()) if len(failed) else np.nan,
            "feedback_delivered_units": int(group["feedback_delivered"].sum()),
            "eventual_full_pass_rate_after_first_failure": float(failed["eventual_full_pass"].mean()) if len(failed) else np.nan,
            "mean_score_gain_pp_after_first_failure": float(failed["score_gain_pp"].mean()) if len(failed) else np.nan,
            "score_gain_cluster_ci95_low": lo,
            "score_gain_cluster_ci95_high": hi,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    units = submission_outcomes(args.input)
    summary = summarize(units, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(f"Randomized-lab student units: {len(units)}")
    print(f"Saved outcome summary: {args.output}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
