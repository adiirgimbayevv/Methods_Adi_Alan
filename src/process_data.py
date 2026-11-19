"""Warm-up filtering and robust aggregation for measured pipeline runs."""
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
    "run_id", "timestamp_us", "phase", "variant", "load_level",
    "latency_ms", "throughput_rows_s", "status",
}


def load_jsonl(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    frame = pd.DataFrame(records)
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Benchmark schema is missing columns: {sorted(missing)}")
    log.info("Loaded %d measured runs from %s", len(frame), path)
    return frame


def filter_warmup(frame: pd.DataFrame, phase_col: str = "phase") -> pd.DataFrame:
    valid = frame[frame[phase_col] == "benchmark"].copy()
    log.info("Warm-up filter: %d total -> %d measured (%d warm-up removed)",
             len(frame), len(valid), len(frame) - len(valid))
    return valid


def flag_outliers_iqr(frame: pd.DataFrame, group_cols: list[str],
                      metric_col: str = "latency_ms", iqr_mult: float = 1.5) -> pd.DataFrame:
    result = frame.copy()
    result["is_outlier"] = False
    for _, indices in result.groupby(group_cols).groups.items():
        values = result.loc[indices, metric_col]
        q25, q75 = np.percentile(values, [25, 75])
        iqr = q75 - q25
        result.loc[indices, "is_outlier"] = (
            (values < q25 - iqr_mult * iqr) | (values > q75 + iqr_mult * iqr)
        )
    return result


def bootstrap_ci(data: np.ndarray, statistic=np.mean, n_resamples: int = 2000,
                 seed: int = 42) -> tuple[float, float]:
    values = np.asarray(data, dtype=float)
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_resamples, len(values)), replace=True)
    low, high = np.quantile(statistic(draws, axis=1), [0.025, 0.975])
    return float(low), float(high)


def compute_summary(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    keys = ["variant", "load_level"]
    flagged = flag_outliers_iqr(frame, keys)
    rows = []
    for (variant, load_rows), group in flagged.groupby(keys, sort=True):
        valid = group[(group["status"] == "ok") & group["latency_ms"].notna()]
        latency = valid["latency_ms"].to_numpy(dtype=float)
        throughput = valid["throughput_rows_s"].dropna().to_numpy(dtype=float)
        latency_low, latency_high = bootstrap_ci(latency, seed=seed)
        median_low, median_high = bootstrap_ci(latency, statistic=np.median, seed=seed)
        throughput_low, throughput_high = bootstrap_ci(throughput, seed=seed)
        rows.append({
            "variant": variant,
            "load_level": int(load_rows),
            "input_rows": int(load_rows),
            "n_total": int(len(group)),
            "n_valid": int(len(valid)),
            "error_count": int((group["status"] != "ok").sum()),
            "error_rate": float((group["status"] != "ok").mean()),
            "outlier_rate": float(valid["is_outlier"].mean()) if len(valid) else 0.0,
            "mean": float(np.mean(latency)) if len(latency) else float("nan"),
            "sd": float(np.std(latency, ddof=1)) if len(latency) > 1 else 0.0,
            "median": float(np.median(latency)) if len(latency) else float("nan"),
            "p25": float(np.percentile(latency, 25)) if len(latency) else float("nan"),
            "p75": float(np.percentile(latency, 75)) if len(latency) else float("nan"),
            "iqr": float(np.percentile(latency, 75) - np.percentile(latency, 25)) if len(latency) else float("nan"),
            "p95": float(np.percentile(latency, 95)) if len(latency) else float("nan"),
            "p99": float(np.percentile(latency, 99)) if len(latency) else float("nan"),
            "ci95_low": latency_low,
            "ci95_high": latency_high,
            "median_ci95_low": median_low,
            "median_ci95_high": median_high,
            "throughput_mean_rows_s": float(np.mean(throughput)) if len(throughput) else float("nan"),
            "throughput_sd_rows_s": float(np.std(throughput, ddof=1)) if len(throughput) > 1 else 0.0,
            "throughput_ci95_low_rows_s": throughput_low,
            "throughput_ci95_high_rows_s": throughput_high,
            "min": float(np.min(latency)) if len(latency) else float("nan"),
            "max": float(np.max(latency)) if len(latency) else float("nan"),
        })
    summary = pd.DataFrame(rows).sort_values(["load_level", "variant"]).reset_index(drop=True)
    pivot = summary.pivot(index="load_level", columns="variant", values="median")
    baseline, proposed = "Baseline (Python row loop)", "Proposed (pandas groupby)"
    speedup = (pivot[baseline] / pivot[proposed]).replace([np.inf, -np.inf], np.nan)
    summary["speedup_vs_baseline"] = summary.apply(
        lambda row: 1.0 if row["variant"] == baseline
        else float(speedup.loc[row["load_level"]]), axis=1
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/test_run.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/metrics_summary.csv")
    args = parser.parse_args()
    raw = load_jsonl(args.input)
    measured = filter_warmup(raw)
    summary = compute_summary(measured, seed=42)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary[["variant", "load_level", "n_valid", "mean", "median", "p95", "p99",
                   "error_rate", "throughput_mean_rows_s", "speedup_vs_baseline"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
