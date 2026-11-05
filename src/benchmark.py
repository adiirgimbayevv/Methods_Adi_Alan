"""Week 7 Clean Experiment & Benchmark Harness (src/benchmark.py).

Combines BOTH:
  A) Domain Study Simulation ("The Effect of GenAI-Assisted Programming on Critical Thinking
     and Academic Performance of Undergraduate IT Students": RQ1, RQ2, RQ3 — `ct_score`,
     `assignment_score`, `final_grade`, covariates `gpa`, `prior_exp_years`, `study_hours_week`).
  B) Technical Pipeline Benchmark (`latency_ms`, `throughput_rps`, `phase == "warmup"` vs
     `"benchmark"`, HTTP/timeout `status` for Guardrail `error_rate`, I/O isolation).

Usage:
    python src/benchmark.py
"""
from __future__ import annotations

import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


def get_env_metadata() -> dict[str, str]:
    return {
        "os": f"{platform.system()}-{platform.release()}",
        "arch": platform.machine(),
        "processor": platform.processor() or "x86_64",
        "python_version": platform.python_version(),
    }


def simulate_trial(
    variant: str,
    load_level: int,
    rng: np.random.Generator,
    is_warmup: bool = False,
    edge_case: str = "normal",
) -> dict:
    """Simulate one student evaluation & pipeline benchmark run."""
    is_explanation = "Proposed" in variant  # Explanation-Guided vs Solution-Gen
    ai_mode = "explanation" if is_explanation else "solution"
    used_ai = 1

    # Map load_level (10, 25, 50, 100, 200, 400) to AI usage intensity share (0..100%)
    ai_share_map = {10: 10.0, 25: 25.0, 50: 45.0, 100: 65.0, 200: 82.0, 400: 95.0}
    ai_share_pct = float(np.clip(ai_share_map.get(load_level, 50.0) + rng.normal(0, 3.5), 0.0, 100.0))
    ai_sessions_per_week = max(1, int(round(ai_share_pct / 12.0 + rng.normal(0, 0.8))))

    # Student background covariates
    gpa = float(np.clip(rng.normal(3.15, 0.42), 2.0, 4.0))
    prior_exp_years = float(np.clip(rng.exponential(1.2), 0.0, 5.0))
    year_of_study = int(rng.choice([1, 2, 3, 4], p=[0.25, 0.35, 0.25, 0.15]))
    study_hours_week = float(np.clip(rng.normal(12.0, 3.5), 2.0, 28.0))
    task_time_min = float(np.clip(rng.normal(48.0, 10.0), 15.0, 90.0))
    task_difficulty = int(rng.integers(1, 6))

    # Domain outcomes (RQ1, RQ2, RQ3):
    # - Explanation-Guided scaffolding increases Critical Thinking (ct_score) as intensity grows,
    #   whereas blind Solution-Generation erodes Critical Thinking at high AI reliance!
    base_ct = 52.0 + 6.5 * (gpa - 3.0) + 1.8 * prior_exp_years + 0.35 * study_hours_week - 0.8 * task_difficulty
    if is_explanation:
        mode_effect_ct = +13.5 + 0.08 * ai_share_pct  # Explanation mode strengthens reasoning
    else:
        mode_effect_ct = -2.0 - 0.14 * ai_share_pct   # Over-reliance on full solutions weakens CT

    ct_score = float(np.clip(base_ct + mode_effect_ct + rng.normal(0, 4.2), 15.0, 100.0))

    # Assignment & Final Grade (0-100 pts)
    base_grade = 58.0 + 7.2 * (gpa - 3.0) + 1.5 * prior_exp_years + 0.40 * study_hours_week
    if is_explanation:
        assignment_score = float(np.clip(base_grade + 9.5 + 0.09 * ai_share_pct + rng.normal(0, 4.5), 20.0, 100.0))
        final_grade = float(np.clip(base_grade + 8.8 + 0.07 * ai_share_pct + rng.normal(0, 4.0), 25.0, 100.0))
    else:
        # Solution generation inflates short-term assignment slightly at moderate intensity but hurts exam/final grade
        assignment_score = float(np.clip(base_grade + 3.0 + 0.02 * ai_share_pct + rng.normal(0, 5.2), 20.0, 100.0))
        final_grade = float(np.clip(base_grade - 1.5 - 0.06 * ai_share_pct + rng.normal(0, 4.8), 25.0, 100.0))

    practical_score = float(np.clip(0.5 * ct_score + 0.5 * final_grade + rng.normal(0, 3.0), 20.0, 100.0))

    # Technical pipeline benchmark metrics (latency_ms & throughput_rps)
    if is_explanation:
        base_ms = 18.0 + 0.14 * load_level + 0.00035 * (load_level ** 2)
        pareto_scale = 2.5
        fail_prob = 0.008 + (0.015 if load_level >= 200 else 0.0)
    else:
        base_ms = 28.0 + 0.28 * load_level + 0.00085 * (load_level ** 2)
        pareto_scale = 5.5
        fail_prob = 0.018 + (0.030 if load_level >= 200 else 0.0)

    if is_warmup:
        base_ms *= rng.uniform(1.45, 1.95)
    if edge_case == "giant_payload":
        base_ms *= 1.35
    elif edge_case == "empty_input":
        base_ms *= 0.65

    gaussian_noise = rng.normal(0.0, base_ms * 0.08)
    pareto_tail = (rng.pareto(a=2.8) * pareto_scale) * (1.0 + load_level / 250.0)
    latency_ms = max(2.0, base_ms + gaussian_noise + pareto_tail)

    retries = 0
    http_code = 200
    status = "ok"
    u = rng.random()
    if not is_warmup and u < fail_prob:
        status = "error"
        http_code = int(rng.choice([429, 500, 504], p=[0.5, 0.2, 0.3]))
        retries = int(rng.integers(1, 4))
        latency_ms *= rng.uniform(2.2, 4.0)
    elif not is_warmup and u < fail_prob * 2.2:
        retries = 1
        latency_ms *= 1.35

    efficiency = 0.94 if is_explanation else 0.82
    throughput_rps = max(1.0, (load_level * efficiency) / (1.0 + (latency_ms / 1200.0)) + rng.normal(0, 1.2))

    return {
        "ai_mode": ai_mode,
        "used_ai": used_ai,
        "ai_share_pct": round(ai_share_pct, 2),
        "ai_sessions_per_week": ai_sessions_per_week,
        "ct_score": round(ct_score, 2),
        "assignment_score": round(assignment_score, 2),
        "practical_score": round(practical_score, 2),
        "final_grade": round(final_grade, 2),
        "gpa": round(gpa, 2),
        "prior_exp_years": round(prior_exp_years, 2),
        "year_of_study": year_of_study,
        "study_hours_week": round(study_hours_week, 1),
        "task_time_min": round(task_time_min, 1),
        "task_difficulty": task_difficulty,
        "latency_ms": round(float(latency_ms), 3),
        "throughput_rps": round(float(throughput_rps), 2),
        "status": status,
        "http_code": http_code,
        "retries": retries,
    }


def run_benchmark(config_path: Path = ROOT / "configs" / "config.yaml") -> None:
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["reproducibility"]["random_seed"])
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    b_cfg = cfg["benchmark"]
    out_path = ROOT / b_cfg["raw_log_path"]
    n_warmup = int(b_cfg["warmup_runs_per_level"])
    n_bench = int(b_cfg["benchmark_runs_per_level"])
    load_levels = list(b_cfg["load_levels_rps"])
    variants = list(b_cfg["variants"])

    env_meta = get_env_metadata()
    edge_types = ["normal"] * 88 + ["giant_payload"] * 6 + ["special_chars"] * 4 + ["empty_input"] * 2

    buffer: list[dict] = []
    run_id = 0
    base_ts_us = int(time.time() * 1_000_000)

    for variant in variants:
        for rps in load_levels:
            # 1) Warm-up runs (phase == "warmup")
            for i in range(n_warmup):
                t0 = time.perf_counter_ns()
                sim = simulate_trial(variant=variant, load_level=rps, rng=rng, is_warmup=True, edge_case="normal")
                overhead_us = round((time.perf_counter_ns() - t0) / 1000.0, 2)
                buffer.append({
                    "id": run_id + 1,
                    "run_id": run_id,
                    "timestamp_us": base_ts_us + run_id * 1000,
                    "phase": "warmup",
                    "variant": variant,
                    "load_level": rps,
                    "request_rate_rps": rps,
                    "edge_case": "normal",
                    "harness_overhead_us": overhead_us,
                    **sim,
                    **env_meta,
                })
                run_id += 1

            # 2) Benchmark evaluation runs (phase == "benchmark")
            for i in range(n_bench):
                ec = edge_types[i % len(edge_types)]
                t0 = time.perf_counter_ns()
                sim = simulate_trial(variant=variant, load_level=rps, rng=rng, is_warmup=False, edge_case=ec)
                overhead_us = round((time.perf_counter_ns() - t0) / 1000.0, 2)
                buffer.append({
                    "id": run_id + 1,
                    "run_id": run_id,
                    "timestamp_us": base_ts_us + run_id * 1000,
                    "phase": "benchmark",
                    "variant": variant,
                    "load_level": rps,
                    "request_rate_rps": rps,
                    "edge_case": ec,
                    "harness_overhead_us": overhead_us,
                    **sim,
                    **env_meta,
                })
                run_id += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in buffer:
            f.write(json.dumps(rec) + "\n")

    print(f"Written {len(buffer)} records -> {out_path}")


if __name__ == "__main__":
    run_benchmark()
