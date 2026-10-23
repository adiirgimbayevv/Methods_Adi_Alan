"""Entry point: load config -> check data -> test hypotheses -> write log + metrics.

Usage:
    python src/main.py                              # uses configs/config.yaml
    python src/main.py --config configs/config.yaml
    python src/main.py --strict                     # exit code 1 if any guardrail fails
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analysis as an  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("study")


def setup_logging(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode="w", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_columns(df: pd.DataFrame, cfg: dict) -> None:
    v = cfg["variables"]
    need = list(v["independent"].values()) + list(v["dependent"].values()) + list(v["controls"])
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: columns missing in data file: {missing}")


def fmt(res: dict) -> str:
    return (f"beta={res['beta']:+.3f} [{res['ci_low']:+.2f}, {res['ci_high']:+.2f}], "
            f"p={res['p_reported']:.3f} ({res['alternative']}), dR2={res['delta_r2']:.3f}, n={res['n']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    ap.add_argument("--strict", action="store_true", help="exit 1 if any guardrail fails")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    out_dir = ROOT / cfg["paths"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(ROOT / cfg["paths"]["log_file"])

    seed = cfg["reproducibility"]["random_seed"]
    random.seed(seed)
    np.random.seed(seed)
    t0 = time.time()

    data_path = ROOT / cfg["paths"]["data_file"]
    log.info("Project: %s v%s", cfg["project"]["name"], cfg["project"]["version"])
    log.info("Seed=%d | data=%s", seed, data_path.relative_to(ROOT))
    df = pd.read_csv(data_path)
    validate_columns(df, cfg)
    log.info("Loaded %d rows x %d columns", *df.shape)

    # 1) guardrails ---------------------------------------------------------
    gr = an.check_guardrails(df, cfg)
    log.info("--- Guardrails ---")
    for k in ("sample_size_ok", "missing_ok", "vif_ok", "cronbach_ok", "anonymized_ok"):
        val = gr[k]
        log.info("%-16s %s", k, "n/a" if val is None else ("PASS" if val else "FAIL"))
    smoke = not gr["sample_size_ok"]
    if smoke:
        log.warning("n=%d < %d: SMOKE-TEST MODE. The pipeline runs, but results are not interpretable.",
                    gr["n"], cfg["guardrails"]["min_sample_size"])

    # 2) hypothesis tests ---------------------------------------------------
    log.info("--- Primary hypothesis (RQ3): explanation vs solution -> critical thinking ---")
    primary = an.test_primary_rq3(df, cfg)
    log.info(fmt(primary))
    if smoke:
        primary["supported"] = None  # never report a conclusion from the smoke-test sample
    log.info("Cohen's d (raw)=%.2f | H1 supported: %s", primary["cohens_d_raw"],
             "n/a (smoke test)" if smoke else primary["supported"])

    log.info("--- Secondary hypotheses ---")
    rq1, rq2, rq3b = an.test_rq1(df, cfg), an.test_rq2(df, cfg), an.test_rq3b_assignment(df, cfg)
    log.info("RQ1  AI use -> CT:             %s", fmt(rq1))
    log.info("RQ2  intensity -> final grade: %s | rho=%.2f", fmt(rq2), rq2["spearman_rho"])
    log.info("RQ3b mode -> assignment:       %s", fmt(rq3b))
    corrected = an.holm({"RQ1": rq1["p_reported"], "RQ2": rq2["p_reported"], "RQ3b": rq3b["p_reported"]},
                        cfg["analysis"]["alpha"])
    for k, r in corrected.items():
        log.info("Holm %-5s p_raw=%.3f p_adj=%.3f reject_H0=%s", k, r["p_raw"], r["p_holm"], r["reject_h0"])

    # 3) save ---------------------------------------------------------------
    metrics = {
        "smoke_test_mode": smoke,
        "guardrails": gr,
        "primary_rq3": primary,
        "secondary": {"rq1": rq1, "rq2": rq2, "rq3b": rq3b, "holm": corrected},
        "descriptives": an.descriptives(df, cfg),
        "runtime_seconds": round(time.time() - t0, 3),
    }
    metrics_path = ROOT / cfg["paths"]["metrics_file"]
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    log.info("Saved metrics -> %s", metrics_path.relative_to(ROOT))
    log.info("Done in %.2fs", time.time() - t0)

    failed = [k for k in ("sample_size_ok", "missing_ok", "vif_ok", "anonymized_ok") if gr[k] is False]
    return 1 if (args.strict and failed) else 0


if __name__ == "__main__":
    sys.exit(main())
