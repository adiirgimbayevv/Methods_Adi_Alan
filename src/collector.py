"""Measurement testbed: times the analysis pipeline on synthetic cohorts.

What it does
    1. Builds synthetic student tables (realistic distributions + edge cases).
    2. Runs the target function `analysis.run_all` repeatedly:
       - `warmup_runs` cold-start runs (logged, flagged phase="warmup", excluded from statistics),
       - `measured_runs` working runs.
    3. Writes one JSON object per line (JSONL) to data/raw/test_run.jsonl
       (line 1 = environment metadata, then one line per run).
    4. Prints / saves summary statistics: median, p95, p99, error rate.

Measurement isolation: all file/console output happens AFTER the timed section;
data generation is also outside the timer; logging is disabled and the garbage
collector is paused while the clock is running.

Usage:  python src/collector.py [--config configs/config.yaml]
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis as an  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- synthetic data
def _sample(spec: dict, rng: np.random.Generator, n: int) -> np.ndarray:
    if spec["dist"] == "normal":
        x = rng.normal(spec["mean"], spec["sd"], n)
    elif spec["dist"] == "pareto":          # heavy tail: many light users, few very heavy ones
        x = spec["scale"] * rng.pareto(spec["shape"], n)
    else:
        raise ValueError(f"unknown distribution: {spec['dist']}")
    return np.clip(x, spec["min"], spec["max"])


def make_cohort(rng: np.random.Generator, bcfg: dict, edge: str | None = None) -> pd.DataFrame:
    n = 8 if edge == "tiny_sample" else bcfg["cohort_size"]
    d = bcfg["distributions"]
    used = rng.random(n) < bcfg["share_ai_users"]
    explanation = rng.random(n) < bcfg["share_explanation_among_users"]
    mode = np.where(used, np.where(explanation, "explanation", "solution"), "none")

    gpa = _sample(d["gpa"], rng, n)
    sessions = np.where(used, np.round(_sample(d["ai_sessions_per_week"], rng, n)), 0)
    share = np.where(used, np.clip(sessions * 7 + rng.normal(0, 8, n), 0, 100), 0)
    exp = np.round(_sample(d["prior_exp_years"], rng, n), 1)

    # Invented outcome model, ONLY to give the synthetic data some structure.
    bonus = np.where(mode == "explanation", 4.0, np.where(mode == "solution", -3.0, 0.0))
    ct = np.clip(50 + 8 * (gpa - 3) + 1.5 * exp + bonus + rng.normal(0, 6, n), 0, 100)
    assign = np.clip(60 + 10 * (gpa - 3) + np.where(mode == "solution", 6, 0) + rng.normal(0, 6, n), 0, 100)
    prac = np.clip(55 + 9 * (gpa - 3) + 0.4 * ct / 10 + rng.normal(0, 6, n), 0, 100)
    final = np.clip(0.4 * assign + 0.3 * prac + 0.3 * ct + rng.normal(0, 3, n), 0, 100)

    df = pd.DataFrame({
        "id": [f"S{i:04d}" for i in range(n)],
        "used_ai": used.astype(int), "ai_sessions_per_week": sessions, "ai_share_pct": share.round(1),
        "ai_mode": mode, "gpa": gpa.round(2), "prior_exp_years": exp,
        "year_of_study": rng.integers(1, 5, n), "study_hours_week": _sample(d["study_hours_week"], rng, n).round(1),
        "task_time_min": _sample(d["task_time_min"], rng, n).round(0), "task_difficulty": rng.integers(1, 6, n),
        "ct_score": ct.round(1), "assignment_score": assign.round(1),
        "practical_score": prac.round(1), "final_grade": final.round(1),
    })

    if edge == "missing_values":
        for col in ("gpa", "ct_score"):
            df.loc[rng.random(n) < 0.15, col] = np.nan
    elif edge == "constant_column":
        df["gpa"] = 3.0
    elif edge == "empty_group":
        df.loc[df["ai_mode"] == "solution", "ai_mode"] = "explanation"
    elif edge == "extreme_outliers":
        df.loc[df.sample(3, random_state=int(rng.integers(1e6))).index, "ct_score"] = 1e6
    return df


# ---------------------------------------------------------------------- metadata
def _cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _ram_gb() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            return round(int(f.readline().split()[1]) / 1024 / 1024, 2)
    except (OSError, ValueError, IndexError):
        pass
    try:  # Windows
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("l", ctypes.c_ulong), ("m", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("a", ctypes.c_ulonglong), ("p", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong),
                        ("v", ctypes.c_ulonglong), ("w", ctypes.c_ulonglong), ("x", ctypes.c_ulonglong)]
        ms = MS(); ms.l = ctypes.sizeof(MS)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return round(ms.total / 1024 ** 3, 2)
    except Exception:
        return None


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, timeout=3, check=True).stdout.strip()
    except Exception:
        return None


def environment_meta(config_path: Path, bcfg: dict) -> dict:
    import scipy, statsmodels  # noqa: E401
    return {
        "type": "meta",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()[:60]})",
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_count_logical": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "in_container": Path("/.dockerenv").exists() or Path("/run/.containerenv").exists(),
        "libs": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                 "statsmodels": statsmodels.__version__},
        "seed": bcfg["seed"],
        "benchmark_params": {k: bcfg[k] for k in ("warmup_runs", "measured_runs", "cohort_size", "edge_case_share")},
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest()[:16],
        "git_commit": _git_commit(),
    }


# ---------------------------------------------------------------------- timing
def timed_run(df: pd.DataFrame, cfg: dict) -> dict:
    """Time ONE call of the target function. No I/O inside the timed section."""
    status, err = "ok", None
    logging.disable(logging.CRITICAL)
    gc.collect()
    gc.disable()
    started = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    t0 = time.perf_counter_ns()
    try:
        an.run_all(df, cfg)
    except Exception as exc:  # failures are RECORDED, never silently dropped (selection bias)
        status, err = "error", f"{type(exc).__name__}: {exc}"[:200]
    dt = time.perf_counter_ns() - t0
    gc.enable()
    logging.disable(logging.NOTSET)
    return {"started_at": started, "duration_ns": dt, "duration_ms": round(dt / 1e6, 4),
            "status": status, "error": err}


def stats(ms: list[float]) -> dict:
    if not ms:
        return {"n": 0}
    a = np.array(ms)
    return {"n": int(a.size), "mean_ms": round(float(a.mean()), 3), "median_ms": round(float(np.median(a)), 3),
            "p95_ms": round(float(np.percentile(a, 95)), 3), "p99_ms": round(float(np.percentile(a, 99)), 3),
            "min_ms": round(float(a.min()), 3), "max_ms": round(float(a.max()), 3),
            "std_ms": round(float(a.std(ddof=1)), 3) if a.size > 1 else 0.0}


def summarise(records: list[dict]) -> dict:
    warm = [r for r in records if r["phase"] == "warmup"]
    meas = [r for r in records if r["phase"] == "measured"]
    ok = [r for r in meas if r["status"] == "ok"]
    err_by_edge: dict[str, int] = {}
    for r in meas:
        if r["status"] == "error":
            key = r["edge_case"] or "none"
            err_by_edge[key] = err_by_edge.get(key, 0) + 1
    return {
        "warmup_runs_discarded": len(warm),
        "warmup_median_ms": stats([r["duration_ms"] for r in warm]).get("median_ms"),
        "measured_runs": len(meas),
        "ok_runs": len(ok),
        "error_runs": len(meas) - len(ok),
        "error_rate": round((len(meas) - len(ok)) / max(len(meas), 1), 4),
        "errors_by_edge_case": err_by_edge,
        "latency_ok_runs": stats([r["duration_ms"] for r in ok]),
        "latency_all_measured_runs": stats([r["duration_ms"] for r in meas]),
        "note": "Percentiles are reported with and without failed runs so errors cannot bias the result silently.",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    args = ap.parse_args()
    config_path = Path(args.config)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    b = cfg["benchmark"]
    warnings.simplefilter("ignore")
    rng = np.random.default_rng(b["seed"])

    n_warm, n_meas = b["warmup_runs"], b["measured_runs"]
    n_edge = int(round(n_meas * b["edge_case_share"]))
    edge_plan = [None] * (n_meas - n_edge) + [b["edge_cases"][i % len(b["edge_cases"])] for i in range(n_edge)]
    rng.shuffle(edge_plan)  # object array shuffle keeps edge cases spread through the run
    plan = [("warmup", None)] * n_warm + [("measured", e) for e in edge_plan]

    records: list[dict] = []
    for i, (phase, edge) in enumerate(plan):
        df = make_cohort(rng, b, edge)                    # outside the timer
        rec = {"type": "run", "run_id": i, "phase": phase, "edge_case": edge, "n_rows": int(len(df))}
        rec.update(timed_run(df, cfg))                    # only this call is timed
        records.append(rec)

    # ---- all I/O below this line (never inside a measurement) ----
    out = ROOT / b["output_jsonl"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(environment_meta(config_path, b)) + "\n")
        for r in records:
            f.write(json.dumps(r) + "\n")

    summ = summarise(records)
    sp = ROOT / b["summary_file"]
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(summ, indent=2), encoding="utf-8")

    ok = summ["latency_ok_runs"]
    print(f"Wrote {len(records)} run records (+1 meta line) -> {out.relative_to(ROOT)}")
    print(f"Warm-up runs discarded: {summ['warmup_runs_discarded']} (median {summ['warmup_median_ms']} ms)")
    print(f"Measured: {summ['measured_runs']} | ok: {summ['ok_runs']} | errors: {summ['error_runs']} "
          f"({summ['error_rate']:.1%}) {summ['errors_by_edge_case']}")
    print(f"Latency of ok runs: median={ok.get('median_ms')} ms, p95={ok.get('p95_ms')} ms, p99={ok.get('p99_ms')} ms")
    print(f"Summary -> {sp.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
