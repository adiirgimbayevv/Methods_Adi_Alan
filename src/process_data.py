"""Data processing, schema validation, and statistical aggregation module (src/process_data.py).

Processes both:
  - Domain Study Metrics: Critical Thinking (`ct_score`), Final Grade (`final_grade`),
    Assignment Score (`assignment_score`), AI Usage Intensity (`ai_share_pct`).
  - Benchmark Performance Metrics: Response Latency (`latency_ms`), Throughput (`throughput_rps`),
    Guardrail Error Rate (`error_rate`), and 1.5x IQR Outlier Flagging (`outlier_rate`).

Exports clean aggregated summary to `data/processed/metrics_summary.csv`.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("process_data")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

REQUIRED_COLUMNS = {
    "run_id", "timestamp_us", "phase", "variant", "load_level", "latency_ms", "throughput_rps", "status"
}


def load_jsonl(path: Path) -> pd.DataFrame:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Schema validation failed: missing columns {missing}")
    log.info("Loaded %d raw records from %s", len(df), path)
    return df


def filter_warmup(df: pd.DataFrame, phase_col: str = "phase") -> pd.DataFrame:
    before = len(df)
    valid_phases = {"benchmark", "eval"}
    df_bench = df[df[phase_col].isin(valid_phases)].copy()
    log.info("Warm-up filtering: %d total -> %d benchmark rows (%d warmup discarded)",
             before, len(df_bench), before - len(df_bench))
    return df_bench


def flag_outliers_iqr(df: pd.DataFrame,
                      group_cols: list[str],
                      metric_col: str = "latency_ms",
                      iqr_mult: float = 1.5) -> pd.DataFrame:
    df = df.copy()
    df["is_outlier"] = False
    for _, idx in df.groupby(group_cols).groups.items():
        vals = df.loc[idx, metric_col]
        q25, q75 = np.percentile(vals, [25, 75])
        iqr = q75 - q25
        upper = q75 + iqr_mult * iqr
        lower = q25 - iqr_mult * iqr
        df.loc[idx, "is_outlier"] = (vals < lower) | (vals > upper)
    log.info("Flagged %d non-destructive IQR outliers (%.2f%%)",
             int(df["is_outlier"].sum()), 100.0 * df["is_outlier"].mean())
    return df


def bootstrap_ci_95(data: np.ndarray,
                    stat_fn=np.mean,
                    n_resamples: int = 1000,
                    seed: int = 42) -> tuple[float, float]:
    x = np.asarray(data, dtype=float)
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, len(x), size=(n_resamples, len(x)))
    boot_stats = stat_fn(x[boot_idx], axis=1)
    lo, hi = np.percentile(boot_stats, [2.5, 97.5])
    return float(lo), float(hi)


def compute_summary(df_bench: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    group_cols = ["variant", "load_level"]
    df_flagged = flag_outliers_iqr(df_bench, group_cols=group_cols, metric_col="latency_ms")

    rows = []
    for (variant, load_level), grp in df_flagged.groupby(group_cols):
        n_total = len(grp)
        n_errors = int((grp["status"] != "ok").sum())
        error_rate = n_errors / n_total if n_total > 0 else 0.0

        ok_grp = grp[grp["status"] == "ok"]
        lat = ok_grp["latency_ms"].dropna().values
        thr = ok_grp["throughput_rps"].dropna().values
        outlier_rate = float(ok_grp["is_outlier"].mean()) if len(ok_grp) > 0 else 0.0

        lat_ci_lo, lat_ci_hi = bootstrap_ci_95(lat, stat_fn=np.mean, seed=seed)
        med_ci_lo, med_ci_hi = bootstrap_ci_95(lat, stat_fn=np.median, seed=seed)
        thr_ci_lo, thr_ci_hi = bootstrap_ci_95(thr, stat_fn=np.mean, seed=seed)

        # Domain metrics (Critical Thinking & Academic Performance)
        ct = ok_grp["ct_score"].dropna().values if "ct_score" in ok_grp else np.zeros(len(lat))
        fg = ok_grp["final_grade"].dropna().values if "final_grade" in ok_grp else np.zeros(len(lat))
        ai_sh = ok_grp["ai_share_pct"].dropna().values if "ai_share_pct" in ok_grp else np.zeros(len(lat))
        ct_ci_lo, ct_ci_hi = bootstrap_ci_95(ct, stat_fn=np.mean, seed=seed)
        fg_ci_lo, fg_ci_hi = bootstrap_ci_95(fg, stat_fn=np.mean, seed=seed)

        rows.append({
            "variant": variant,
            "load_level": int(load_level),
            "ai_share_mean": round(float(np.mean(ai_sh)), 1),
            "n_total": n_total,
            "n_valid": len(lat),
            "error_count": n_errors,
            "error_rate": round(error_rate, 4),
            "outlier_rate": round(outlier_rate, 4),
            # Critical Thinking & Academic Performance Metrics (Domain Study)
            "ct_mean": round(float(np.mean(ct)), 2),
            "ct_sd": round(float(np.std(ct, ddof=1)), 2),
            "ct_median": round(float(np.median(ct)), 2),
            "ct_p25": round(float(np.percentile(ct, 25)), 2),
            "ct_p75": round(float(np.percentile(ct, 75)), 2),
            "ct_ci95_low": round(ct_ci_lo, 2),
            "ct_ci95_high": round(ct_ci_hi, 2),
            "grade_mean": round(float(np.mean(fg)), 2),
            "grade_sd": round(float(np.std(fg, ddof=1)), 2),
            "grade_median": round(float(np.median(fg)), 2),
            "grade_ci95_low": round(fg_ci_lo, 2),
            "grade_ci95_high": round(fg_ci_hi, 2),
            # Pipeline Benchmark Metrics (Latency & Throughput)
            "mean": round(float(np.mean(lat)), 3),
            "sd": round(float(np.std(lat, ddof=1)), 3),
            "median": round(float(np.median(lat)), 3),
            "p25": round(float(np.percentile(lat, 25)), 3),
            "p75": round(float(np.percentile(lat, 75)), 3),
            "iqr": round(float(np.percentile(lat, 75) - np.percentile(lat, 25)), 3),
            "p95": round(float(np.percentile(lat, 95)), 3),
            "p99": round(float(np.percentile(lat, 99)), 3),
            "ci95_low": round(lat_ci_lo, 3),
            "ci95_high": round(lat_ci_hi, 3),
            "median_ci95_low": round(med_ci_lo, 3),
            "median_ci95_high": round(med_ci_hi, 3),
            "throughput_mean": round(float(np.mean(thr)), 2),
            "throughput_sd": round(float(np.std(thr, ddof=1)), 2),
            "throughput_median": round(float(np.median(thr)), 2),
            "throughput_ci95_low": round(thr_ci_lo, 2),
            "throughput_ci95_high": round(thr_ci_hi, 2),
            "min": round(float(np.min(lat)), 3),
            "max": round(float(np.max(lat)), 3),
        })

    summary = pd.DataFrame(rows).sort_values(["load_level", "variant"]).reset_index(drop=True)
    log.info("Computed summary table with %d condition rows", len(summary))
    return summary


def main() -> None:
    np.random.seed(42)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(ROOT / "data" / "raw" / "test_run.jsonl"))
    ap.add_argument("--output", default=str(ROOT / "data" / "processed" / "metrics_summary.csv"))
    args = ap.parse_args()

    df_raw = load_jsonl(Path(args.input))
    df_bench = filter_warmup(df_raw)
    summary = compute_summary(df_bench, seed=42)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    log.info("Saved processed metrics -> %s", out_path)
    print(summary[["variant", "load_level", "ct_mean", "ct_sd", "grade_mean", "mean", "median", "p95", "error_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
