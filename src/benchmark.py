"""Measure two actual aggregation implementations on the public ProgFeed data.

This benchmark measures local CPU time while aggregating actual source rows.
It does not simulate student outcomes, AI requests, or HTTP failures.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
GROUP_COLS = ["lab", "ai_feedback_type"]


def python_row_loop(frame: pd.DataFrame) -> list[tuple]:
    totals: dict[tuple[str, str], list[float]] = {}
    for row in frame.to_dict(orient="records"):
        key = (str(row["lab"]), str(row["ai_feedback_type"]))
        bucket = totals.setdefault(key, [0.0, 0.0, 0.0])
        bucket[0] += float(row["score"])
        bucket[1] += float(row["max_score"])
        bucket[2] += 1
    return [(key[0], key[1], *values) for key, values in sorted(totals.items())]


def pandas_groupby(frame: pd.DataFrame) -> list[tuple]:
    grouped = frame.groupby(GROUP_COLS, dropna=False).agg(
        score=("score", "sum"),
        max_score=("max_score", "sum"),
        n_rows=("score", "size"),
    ).reset_index().sort_values(GROUP_COLS)
    return [tuple(row) for row in grouped.itertuples(index=False, name=None)]


def run_benchmark(config_path: Path = ROOT / "configs" / "config.yaml") -> None:
    with config_path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    bench = cfg["benchmark"]
    source = ROOT / bench["source_data_path"]
    output = ROOT / bench["raw_log_path"]
    data = pd.read_csv(source, usecols=["lab", "ai_feedback_type", "score", "max_score"])
    data = data.dropna(subset=["lab", "ai_feedback_type", "score", "max_score"]).reset_index(drop=True)
    data["score"] = pd.to_numeric(data["score"], errors="coerce")
    data["max_score"] = pd.to_numeric(data["max_score"], errors="coerce")
    data = data.dropna(subset=["score", "max_score"]).reset_index(drop=True)
    if data.empty:
        raise ValueError(f"No usable records in {source}")

    variants = {
        "Baseline (Python row loop)": python_row_loop,
        "Proposed (pandas groupby)": pandas_groupby,
    }
    levels = [min(int(n), len(data)) for n in bench["load_levels_rows"]]
    levels = list(dict.fromkeys(n for n in levels if n > 0))
    warmups = int(bench["warmup_runs_per_level"])
    repetitions = int(bench["benchmark_runs_per_level"])
    environment = {
        "os": f"{platform.system()}-{platform.release()}",
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "source_rows": int(len(data)),
    }

    records: list[dict] = []
    run_id = 0
    for load_rows in levels:
        subset = data.iloc[:load_rows].copy()
        for variant, function in variants.items():
            for phase, count in (("warmup", warmups), ("benchmark", repetitions)):
                for _ in range(count):
                    started_ns = time.perf_counter_ns()
                    status = "ok"
                    try:
                        function(subset)
                    except Exception:
                        status = "error"
                    elapsed_ns = time.perf_counter_ns() - started_ns
                    latency_ms = elapsed_ns / 1_000_000
                    records.append({
                        "id": run_id + 1,
                        "run_id": run_id,
                        "timestamp_us": time.time_ns() // 1000,
                        "phase": phase,
                        "variant": variant,
                        "load_level": load_rows,
                        "input_rows": load_rows,
                        "latency_ms": round(latency_ms, 6),
                        "throughput_rows_s": round(load_rows / (elapsed_ns / 1_000_000_000), 3)
                            if elapsed_ns > 0 else None,
                        "status": status,
                        **environment,
                    })
                    run_id += 1

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")
    print(f"Measured {len(records)} actual aggregation runs from {source.relative_to(ROOT)}")
    print(f"Warm-up runs: {warmups} per condition; benchmark runs: {repetitions} per condition")
    print(f"Saved JSONL measurements: {output.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "config.yaml")
    args = parser.parse_args()
    run_benchmark(args.config)


if __name__ == "__main__":
    main()
