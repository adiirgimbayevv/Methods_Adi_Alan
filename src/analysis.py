"""Statistical routines for the GenAI / critical thinking study.

Every function is pure (data in, dict out) so it can be unit-tested and
reused from notebooks.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.outliers_influence import variance_inflation_factor

log = logging.getLogger("study")

FORBIDDEN_COLUMNS = {"name", "full_name", "surname", "email", "phone", "username"}


# ----------------------------------------------------------------- guardrails
def cronbach_alpha(items: pd.DataFrame) -> float:
    """Cronbach's alpha for a set of item columns (rows = respondents)."""
    items = items.dropna()
    k = items.shape[1]
    if k < 2 or len(items) < 2:
        return float("nan")
    item_var = items.var(axis=0, ddof=1).sum()
    total_var = items.sum(axis=1).var(ddof=1)
    return float(k / (k - 1) * (1 - item_var / total_var))


def compute_vif(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    X = sm.add_constant(df[cols].astype(float).dropna(), has_constant="add")
    if len(X) <= X.shape[1]:
        return {c: float("nan") for c in cols}
    return {c: float(variance_inflation_factor(X.values, i + 1)) for i, c in enumerate(cols)}


def check_guardrails(df: pd.DataFrame, cfg: dict) -> dict:
    g, v = cfg["guardrails"], cfg["variables"]
    used = (
        list(v["independent"].values()) + list(v["dependent"].values()) + list(v["controls"])
    )
    missing = {c: float(df[c].isna().mean()) for c in used}
    vif = compute_vif(df, list(v["controls"]))
    items = v.get("ct_items") or []
    alpha = cronbach_alpha(df[items]) if items else None

    res = {
        "n": int(len(df)),
        "sample_size_ok": len(df) >= g["min_sample_size"],
        "max_missing_share": max(missing.values()),
        "missing_ok": max(missing.values()) <= g["max_missing_share"],
        "vif": vif,
        "vif_ok": all((x <= g["max_vif"]) for x in vif.values() if not np.isnan(x)),
        "cronbach_alpha": alpha,
        "cronbach_ok": (alpha >= g["min_cronbach_alpha"]) if alpha is not None else None,
        "anonymized_ok": not (FORBIDDEN_COLUMNS & {c.lower() for c in df.columns}),
    }
    return res


# --------------------------------------------------------------------- models
def _trim_controls(n_obs: int, n_focal_params: int, controls: list[str]) -> list[str]:
    """Drop trailing controls until at least 1 residual degree of freedom remains.

    Only triggers on tiny samples (e.g. the 10-row smoke-test sample). With a
    real sample (n >= 100) all controls are kept.
    """
    keep = list(controls)
    while keep and n_obs - (1 + n_focal_params + len(keep)) < 1:
        keep.pop()
    if len(keep) < len(controls):
        log.warning(
            "Small sample (n=%d): using %d of %d controls (%s). Results are NOT interpretable.",
            n_obs, len(keep), len(controls), ", ".join(keep) or "none",
        )
    return keep


def fit_ols(df: pd.DataFrame, y: str, focal: str, controls: list[str], categorical: bool = False,
            reference: str | None = None, n_focal_params: int = 1):
    d = df.dropna(subset=[y, focal] + controls)
    ctrl = _trim_controls(len(d), n_focal_params, controls)
    term = f"C({focal}, Treatment(reference='{reference}'))" if categorical else focal
    rhs = " + ".join([term] + ctrl)
    model = smf.ols(f"{y} ~ {rhs}", data=d).fit()
    null = smf.ols(f"{y} ~ " + (" + ".join(ctrl) if ctrl else "1"), data=d).fit()
    return model, null, ctrl, d


def _focal_coef(model, focal: str, level: str | None = None):
    for name in model.params.index:
        if name == focal or (level and name.endswith(f"[T.{level}]")):
            return name
    raise KeyError(f"focal term for {focal} not found in {list(model.params.index)}")


def _one_sided(beta: float, p_two: float, alternative: str) -> float:
    if alternative == "two-sided":
        return p_two
    if alternative == "greater":
        return p_two / 2 if beta > 0 else 1 - p_two / 2
    return p_two / 2 if beta < 0 else 1 - p_two / 2


def _summarise(model, null, coef: str, alt: str, conf: float) -> dict:
    beta = float(model.params[coef])
    p_two = float(model.pvalues[coef])
    lo, hi = model.conf_int(alpha=1 - conf).loc[coef]
    out = {
        "beta": beta,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_two_sided": p_two,
        "p_reported": _one_sided(beta, p_two, alt),
        "alternative": alt,
        "r2": float(model.rsquared),
        "r2_null": float(null.rsquared),
        "delta_r2": float(model.rsquared - null.rsquared),
        "n": int(model.nobs),
        "df_resid": int(model.df_resid),
    }
    try:
        f, p, _ = model.compare_f_test(null)
        out["delta_r2_F"], out["delta_r2_p"] = float(f), float(p)
    except Exception:  # not enough df on tiny samples
        out["delta_r2_F"] = out["delta_r2_p"] = None
    return out


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else float("nan")


# ----------------------------------------------------------------- hypotheses
def test_primary_rq3(df: pd.DataFrame, cfg: dict) -> dict:
    """H1: explanation-mode users score higher on critical thinking than solution-mode users."""
    v, a = cfg["variables"], cfg["analysis"]
    mode, y = v["independent"]["ai_mode"], v["dependent"]["critical_thinking"]
    ref = a["reference_mode"]
    users = df[df[mode].isin(["explanation", "solution"])]
    model, null, ctrl, d = fit_ols(users, y, mode, v["controls"], categorical=True, reference=ref)
    level = "explanation" if ref == "solution" else "solution"
    coef = _focal_coef(model, mode, level)
    out = _summarise(model, null, coef, a["alternative_primary"], a["confidence_level"])
    out["cohens_d_raw"] = cohens_d(d.loc[d[mode] == level, y], d.loc[d[mode] == ref, y])
    out["controls_used"] = ctrl
    out["supported"] = bool(out["beta"] > 0 and out["p_reported"] < a["alpha"])
    return out


def test_rq3b_assignment(df: pd.DataFrame, cfg: dict) -> dict:
    v, a = cfg["variables"], cfg["analysis"]
    mode, y = v["independent"]["ai_mode"], v["dependent"]["assignment"]
    ref = a["reference_mode"]
    users = df[df[mode].isin(["explanation", "solution"])]
    model, null, ctrl, _ = fit_ols(users, y, mode, v["controls"], categorical=True, reference=ref)
    level = "explanation" if ref == "solution" else "solution"
    out = _summarise(model, null, _focal_coef(model, mode, level), a["alternative_primary"], a["confidence_level"])
    out["controls_used"] = ctrl
    return out


def test_rq1(df: pd.DataFrame, cfg: dict) -> dict:
    v, a = cfg["variables"], cfg["analysis"]
    x, y = v["independent"]["ai_used"], v["dependent"]["critical_thinking"]
    model, null, ctrl, _ = fit_ols(df, y, x, v["controls"])
    out = _summarise(model, null, _focal_coef(model, x), "two-sided", a["confidence_level"])
    out["controls_used"] = ctrl
    return out


def test_rq2(df: pd.DataFrame, cfg: dict) -> dict:
    v, a = cfg["variables"], cfg["analysis"]
    x, y = v["independent"]["ai_intensity_share"], v["dependent"]["final_grade"]
    d = df.dropna(subset=[x, y])
    rho, p_rho = stats.spearmanr(d[x], d[y])
    model, null, ctrl, _ = fit_ols(df, y, x, v["controls"])
    out = _summarise(model, null, _focal_coef(model, x), "two-sided", a["confidence_level"])
    out.update({"spearman_rho": float(rho), "spearman_p": float(p_rho), "controls_used": ctrl})
    return out


def holm(p_values: dict[str, float], alpha: float) -> dict[str, dict]:
    keys = list(p_values)
    reject, p_adj, _, _ = multipletests([p_values[k] for k in keys], alpha=alpha, method="holm")
    return {k: {"p_raw": p_values[k], "p_holm": float(pa), "reject_h0": bool(r)}
            for k, pa, r in zip(keys, p_adj, reject)}


def descriptives(df: pd.DataFrame, cfg: dict) -> dict:
    v = cfg["variables"]
    cols = list(v["dependent"].values())
    mode = v["independent"]["ai_mode"]
    return {
        "overall": df[cols].describe().round(3).to_dict(),
        "by_mode": df.groupby(mode)[cols].mean().round(3).to_dict(orient="index"),
        "group_sizes": df[mode].value_counts().to_dict(),
    }


# ------------------------------------------------------------- full pipeline
def run_all(df: pd.DataFrame, cfg: dict) -> dict:
    """One complete analysis pass (guardrails + all hypothesis tests).

    This is the *target function* timed by src/collector.py. It does no file
    or console I/O, so a measurement contains only computation.
    """
    a = cfg["analysis"]
    gr = check_guardrails(df, cfg)
    primary = test_primary_rq3(df, cfg)
    rq1, rq2, rq3b = test_rq1(df, cfg), test_rq2(df, cfg), test_rq3b_assignment(df, cfg)
    corrected = holm({"RQ1": rq1["p_reported"], "RQ2": rq2["p_reported"], "RQ3b": rq3b["p_reported"]}, a["alpha"])
    return {"guardrails": gr, "primary_rq3": primary,
            "secondary": {"rq1": rq1, "rq2": rq2, "rq3b": rq3b, "holm": corrected},
            "descriptives": descriptives(df, cfg)}
