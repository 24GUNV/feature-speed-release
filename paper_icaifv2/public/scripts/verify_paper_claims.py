#!/usr/bin/env python3
"""Verify ICAIF v2 paper claims against canonical CSVs and source guardrails."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "paper_icaifv2" / "public" / "results"
DEFAULT_REPORT = RESULTS_DIR / "verification_report.md"
DEFAULT_JSON = RESULTS_DIR / "computed_claims.json"


@dataclass
class Check:
    name: str
    status: str
    severity: str
    detail: str


def rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def read_text(path: str | Path) -> str:
    full = REPO_ROOT / path
    return full.read_text(encoding="utf-8")


def load_csv(path: str | Path) -> list[dict[str, str]]:
    full = REPO_ROOT / path
    with full.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def fval(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def matches(row: dict[str, str], filters: dict[str, Any]) -> bool:
    for key, expected in filters.items():
        if isinstance(expected, float):
            try:
                if abs(fval(row, key) - expected) > 1e-9:
                    return False
            except ValueError:
                return False
        else:
            if row.get(key) != expected:
                return False
    return True


def group_mean(
    rows: list[dict[str, str]],
    group_key: str,
    metric: str,
    filters: dict[str, Any],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if matches(row, filters):
            grouped[row[group_key]].append(fval(row, metric))
    return {key: mean(vals) for key, vals in grouped.items() if vals}


def add_close_check(
    checks: list[Check],
    name: str,
    actual: dict[str, float],
    expected: dict[str, float],
    tol: float = 0.015,
    severity: str = "blocking",
) -> None:
    misses = []
    parts = []
    for key, exp in expected.items():
        act = actual.get(key)
        if act is None:
            misses.append(f"{key}: missing")
            parts.append(f"{key}=missing")
            continue
        parts.append(f"{key}={act:.3f}")
        if abs(act - exp) > tol:
            misses.append(f"{key}: got {act:.4f}, expected {exp:.4f}")
    status = "PASS" if not misses else "FAIL"
    detail = "; ".join(misses) if misses else ", ".join(parts)
    checks.append(Check(name, status, severity, detail))


def add_scalar_check(
    checks: list[Check],
    name: str,
    actual: float,
    expected: float,
    tol: float = 0.015,
    severity: str = "blocking",
) -> None:
    status = "PASS" if abs(actual - expected) <= tol else "FAIL"
    detail = f"got {actual:.4f}, expected {expected:.4f}"
    checks.append(Check(name, status, severity, detail))


def pass_fail(condition: bool, fail_status: str = "FAIL") -> str:
    return "PASS" if condition else fail_status


def source_has(path: str, pattern: str, flags: int = re.S) -> bool:
    return re.search(pattern, read_text(path), flags) is not None


def main_table_checks(checks: list[Check], computed: dict[str, Any]) -> None:
    rows = load_csv("scripts/concern_experiments/expAX_full_comparison.csv")
    filters = {"Period": "OOS", "CostMult": 3.0}
    net = group_mean(rows, "Strategy", "Net_SR", filters)
    gross = group_mean(rows, "Strategy", "Gross_SR", filters)
    turnover = group_mean(rows, "Strategy", "Turnover", filters)
    computed["expAX_oos_3x"] = {
        key: {
            "Net_SR": round(net.get(key, float("nan")), 4),
            "Gross_SR": round(gross.get(key, float("nan")), 4),
            "Turnover": round(turnover.get(key, float("nan")), 4),
        }
        for key in sorted(net)
    }
    add_close_check(
        checks,
        "Main comparison net SRs match expAX CSV",
        net,
        {
            "CompositeSort": 0.324,
            "LightGBM": 0.127,
            "MomentumML_Blend": 0.383,
            "MomentumSort": 0.360,
            "Ridge": 0.353,
            "XGBoost": 0.169,
        },
    )

    by_market: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if matches(row, filters):
            by_market[row["Market"]][row["Strategy"]] = fval(row, "Net_SR")
    wins = sum(
        1
        for values in by_market.values()
        if "Ridge" in values
        and "MomentumSort" in values
        and values["Ridge"] > values["MomentumSort"]
    )
    checks.append(
        Check(
            "Ridge market wins vs MomentumSort match paper",
            "PASS" if wins == 8 else "FAIL",
            "blocking",
            f"Ridge wins {wins}/15 markets at OOS 3x",
        )
    )


def feature_speed_checks(checks: list[Check], computed: dict[str, Any]) -> None:
    rows = load_csv("scripts/concern_experiments/expAH_feature_speed_test.csv")
    filters = {"Period": "OOS", "CostMult": 3.0}
    net = group_mean(rows, "Strategy", "Net_SR", filters)
    turnover = group_mean(rows, "Strategy", "Turnover", filters)
    computed["expAH_oos_3x"] = {
        key: {
            "Net_SR": round(net.get(key, float("nan")), 4),
            "Turnover": round(turnover.get(key, float("nan")), 4),
        }
        for key in sorted(net)
    }
    add_close_check(
        checks,
        "Feature-speed net SRs match expAH CSV",
        net,
        {
            "MomentumSort": 0.360,
            "MomentumSort_Q": 0.373,
            "Ridge_All": 0.353,
            "Ridge_All_Q": 0.481,
            "Ridge_Fast": 0.208,
            "Ridge_Slow": 0.472,
            "Ridge_SlowMed": 0.393,
            "Ridge_Slow_Q": 0.488,
        },
    )
    add_close_check(
        checks,
        "Feature-speed turnovers match expAH CSV",
        turnover,
        {
            "MomentumSort": 0.170,
            "Ridge_All": 0.233,
            "Ridge_Fast": 0.301,
            "Ridge_Slow": 0.102,
        },
        tol=0.01,
    )


def supporting_result_checks(checks: list[Check], computed: dict[str, Any]) -> None:
    an_rows = load_csv("scripts/concern_experiments/expAN_persistence_filter.csv")
    an_net = group_mean(an_rows, "Strategy", "Net_SR", {"Period": "OOS", "CostMult": 3.0})
    computed["expAN_oos_3x_net_sr"] = {key: round(value, 4) for key, value in sorted(an_net.items())}
    add_close_check(
        checks,
        "Persistence-filter net SRs match expAN CSV",
        an_net,
        {
            "MomentumSort": 0.360,
            "Ridge_AdhocSlow": 0.472,
            "Ridge_All": 0.353,
            "Ridge_EmpSlow_FS": 0.470,
            "Ridge_PersFiltered": 0.472,
        },
    )

    ar_rows = load_csv("scripts/concern_experiments/expAR_ema_robustness.csv")
    ar_net = group_mean(ar_rows, "Strategy", "Net_SR", {"Period": "OOS", "CostMult": 3.0})
    ar_to = group_mean(ar_rows, "Strategy", "Turnover", {"Period": "OOS", "CostMult": 3.0})
    computed["expAR_oos_3x"] = {
        key: {
            "Net_SR": round(ar_net.get(key, float("nan")), 4),
            "Turnover": round(ar_to.get(key, float("nan")), 4),
        }
        for key in sorted(ar_net)
    }
    add_close_check(
        checks,
        "EMA smoothing net SRs match expAR CSV",
        ar_net,
        {
            "MomentumSort": 0.360,
            "Ridge_EMA_0.35": 0.474,
            "Ridge_WF_EMA": 0.451,
        },
    )
    add_close_check(
        checks,
        "EMA smoothing turnover matches expAR CSV",
        ar_to,
        {
            "Ridge_EMA_0.35": 0.097,
            "Ridge_WF_EMA": 0.103,
        },
        tol=0.01,
    )

    ai_rows = load_csv("scripts/concern_experiments/expAI_aggregation_vs_speed.csv")
    ai_to = group_mean(ai_rows, "Strategy", "Turnover", {"Period": "OOS", "CostMult": 3.0})
    computed["expAI_oos_3x_turnover"] = {
        key: round(value, 4) for key, value in sorted(ai_to.items())
    }
    add_close_check(
        checks,
        "Aggregation-vs-speed N=5 turnover matches expAI CSV (5.1% vs 5.6%)",
        ai_to,
        {
            "Ridge_5": 0.051,
            "CompositeSort_5": 0.056,
        },
        tol=0.005,
    )


def ic_and_decay_checks(checks: list[Check], computed: dict[str, Any]) -> None:
    ak_rows = load_csv("scripts/concern_experiments/expAK_information_coefficient.csv")
    ak_ic = group_mean(ak_rows, "Strategy", "Mean_IC", {})
    ak_stab = group_mean(ak_rows, "Strategy", "Pred_Stability", {})
    computed["expAK_ic"] = {
        key: {
            "Mean_IC": round(ak_ic.get(key, float("nan")), 4),
            "Pred_Stability": round(ak_stab.get(key, float("nan")), 4),
        }
        for key in sorted(ak_ic)
    }
    add_close_check(
        checks,
        "IC table means match expAK CSV",
        ak_ic,
        {
            "MomentumSort": 0.031,
            "Ridge_All": 0.022,
            "Ridge_Slow": 0.031,
            "Ridge_Fast": 0.013,
        },
        tol=0.001,
    )
    add_close_check(
        checks,
        "IC table stabilities match expAK CSV",
        ak_stab,
        {
            "MomentumSort": 0.894,
            "Ridge_All": 0.665,
            "Ridge_Slow": 0.939,
            "Ridge_Fast": 0.539,
        },
        tol=0.001,
    )
    ic_x_stab = {
        key: ak_ic[key] * ak_stab[key]
        for key in ("MomentumSort", "Ridge_All", "Ridge_Slow", "Ridge_Fast")
        if key in ak_ic and key in ak_stab
    }
    best = max(ic_x_stab, key=ic_x_stab.get)
    checks.append(
        Check(
            "Ridge_Slow has the highest IC x stability",
            "PASS" if best == "Ridge_Slow" else "FAIL",
            "blocking",
            ", ".join(f"{k}={v:.4f}" for k, v in sorted(ic_x_stab.items())),
        )
    )

    dg_rows = load_csv("scripts/concern_experiments/expDG_ic_decay.csv")
    dg_h1 = {
        row["speed_class"]: fval(row, "mean_aligned_ic")
        for row in dg_rows
        if row["horizon"] == "1"
    }
    dg_ret12 = {
        row["speed_class"]: fval(row, "retention")
        for row in dg_rows
        if row["horizon"] == "12"
    }
    computed["expDG_ic_decay"] = {
        "h1_ic": {key: round(value, 4) for key, value in sorted(dg_h1.items())},
        "retention_h12": {key: round(value, 4) for key, value in sorted(dg_ret12.items())},
    }
    add_close_check(
        checks,
        "Model-free IC decay h=1 ICs match expDG CSV",
        dg_h1,
        {"SLOW": 0.027, "FAST": 0.025},
        tol=0.001,
    )
    add_close_check(
        checks,
        "Model-free IC retention at h=12 matches paper claim (69% vs 57%)",
        dg_ret12,
        {"SLOW": 0.693, "FAST": 0.566},
        tol=0.005,
    )


def tree_and_transfer_checks(checks: list[Check], computed: dict[str, Any]) -> None:
    cp_rows = load_csv("scripts/concern_experiments/expCP_tree_smoothing.csv")
    cp_net = group_mean(cp_rows, "Strategy", "Net_SR", {"CostMult": 3.0})
    cp_gross = group_mean(cp_rows, "Strategy", "Gross_SR", {"CostMult": 3.0})
    cp_to = group_mean(cp_rows, "Strategy", "Turnover", {"CostMult": 3.0})
    computed["expCP_3x"] = {
        key: {
            "Net_SR": round(cp_net.get(key, float("nan")), 4),
            "Gross_SR": round(cp_gross.get(key, float("nan")), 4),
            "Turnover": round(cp_to.get(key, float("nan")), 4),
        }
        for key in sorted(cp_net)
    }
    add_close_check(
        checks,
        "Tree smoothing net SRs match expCP CSV",
        cp_net,
        {
            "LGBM_All": 0.265,
            "Ridge_All": 0.357,
            "Ridge_Slow": 0.479,
            "XGB_All": 0.300,
            "XGB_Slow": 0.263,
        },
    )

    df_rows = load_csv("scripts/concern_experiments/expDF_tree_ema.csv")
    df_net = group_mean(df_rows, "Strategy", "Net_SR", {"Period": "OOS", "CostMult": 3.0})
    df_gross = group_mean(df_rows, "Strategy", "Gross_SR", {"Period": "OOS", "CostMult": 3.0})
    df_to = group_mean(df_rows, "Strategy", "Turnover", {"Period": "OOS", "CostMult": 3.0})
    computed["expDF_oos_3x"] = {
        key: {
            "Net_SR": round(df_net.get(key, float("nan")), 4),
            "Gross_SR": round(df_gross.get(key, float("nan")), 4),
            "Turnover": round(df_to.get(key, float("nan")), 4),
        }
        for key in sorted(df_net)
    }
    add_close_check(
        checks,
        "Tree EMA net SRs match expDF CSV",
        df_net,
        {"XGB_EMA_0.5": 0.526},
    )
    add_close_check(
        checks,
        "Tree EMA gross SR and turnover match expDF CSV",
        {
            "XGB_EMA_0.5_gross": df_gross.get("XGB_EMA_0.5", float("nan")),
            "XGB_EMA_0.5_turnover": df_to.get("XGB_EMA_0.5", float("nan")),
        },
        {
            "XGB_EMA_0.5_gross": 0.911,
            "XGB_EMA_0.5_turnover": 0.261,
        },
        tol=0.015,
    )

    transfer = load_csv("scripts/concern_experiments/expDE_transfer_test.csv")[0]
    computed["expDE_transfer"] = {
        key: float(value) if key != "universal_slow_count" else int(float(value))
        for key, value in transfer.items()
    }
    add_scalar_check(checks, "Bound in-sample R2 matches expDE CSV", fval(transfer, "in_sample_r2"), 0.724)
    add_scalar_check(checks, "Bound LOO R2 matches expDE CSV", fval(transfer, "loo_mean_r2"), 0.596)
    add_scalar_check(checks, "Bound leave-5-out R2 matches expDE CSV", fval(transfer, "l5o_mean_r2"), 0.688)
    checks.append(
        Check(
            "Universal slow feature count matches expDE CSV",
            "PASS" if int(float(transfer["universal_slow_count"])) == 52 else "FAIL",
            "blocking",
            f"universal_slow_count={transfer['universal_slow_count']}",
        )
    )


def generated_figure_checks(checks: list[Check], computed: dict[str, Any]) -> None:
    summary_path = REPO_ROOT / "paper_icaifv2" / "public" / "results" / "cost_speed_bound_summary.csv"
    per_kappa_path = REPO_ROOT / "paper_icaifv2" / "public" / "results" / "cost_speed_bound_by_kappa.csv"
    points_path = REPO_ROOT / "paper_icaifv2" / "public" / "results" / "cost_speed_bound_points.csv"
    frontier_path = REPO_ROOT / "paper_icaifv2" / "public" / "results" / "cost_speed_frontier_points.csv"

    if not summary_path.exists() or not per_kappa_path.exists() or not points_path.exists():
        checks.append(
            Check(
                "Generated bound outputs exist",
                "FAIL",
                "blocking",
                "run scripts/concern_experiments/generate_bound_figure.py before verification",
            )
        )
    else:
        summary = load_csv(summary_path.relative_to(REPO_ROOT))[0]
        per_kappa = load_csv(per_kappa_path.relative_to(REPO_ROOT))
        points = load_csv(points_path.relative_to(REPO_ROOT))
        computed["generated_bound_summary"] = {
            "fsi_definition": summary["fsi_definition"],
            "lambda": round(fval(summary, "lambda"), 4),
            "r2": round(fval(summary, "r2"), 4),
            "n": int(float(summary["n"])),
            "per_kappa_lambda_mean": round(fval(summary, "per_kappa_lambda_mean"), 4),
            "per_kappa_lambda_std": round(fval(summary, "per_kappa_lambda_std"), 4),
        }
        checks.append(
            Check(
                "Generated bound uses equal-weight feature-set FSI",
                "PASS" if summary["fsi_definition"] == "equal_weight_feature_set" else "FAIL",
                "blocking",
                f"fsi_definition={summary['fsi_definition']}",
            )
        )
        checks.append(
            Check(
                "Generated bound has 300 observations",
                "PASS" if len(points) == 300 and int(float(summary["n"])) == 300 else "FAIL",
                "blocking",
                f"points={len(points)}, summary_n={summary['n']}",
            )
        )
        add_scalar_check(checks, "Generated bound R2 matches paper claim", fval(summary, "r2"), 0.724)
        add_scalar_check(checks, "Generated bound lambda matches paper claim", fval(summary, "lambda"), 0.392)
        add_scalar_check(
            checks,
            "Generated per-kappa lambda mean matches paper claim",
            fval(summary, "per_kappa_lambda_mean"),
            0.334,
        )
        max_lambda_std = max(float(row["lambda"]) for row in per_kappa) - min(float(row["lambda"]) for row in per_kappa)
        checks.append(
            Check(
                "Generated per-kappa lambda is stable",
                "PASS" if max_lambda_std < 0.005 else "FAIL",
                "blocking",
                f"max-min lambda spread={max_lambda_std:.4f}",
            )
        )

    if not frontier_path.exists():
        checks.append(
            Check(
                "Generated frontier output exists",
                "FAIL",
                "blocking",
                "run scripts/concern_experiments/generate_frontier_figure.py before verification",
            )
        )
    else:
        points = load_csv(frontier_path.relative_to(REPO_ROOT))
        computed["generated_frontier_points"] = {
            row["name"]: {
                "Net_SR": round(float(row["net_sr"]), 4),
                "Turnover": round(float(row["turnover"]), 4),
                "Source": row["source"],
            }
            for row in points
        }
        names = {row["name"] for row in points}
        required = {"Ridge_Slow", "WF-EMA", "Ridge_Slow_Q", "MomentumSort", "Blend", "XGB_All", "LGBM_All"}
        checks.append(
            Check(
                "Generated frontier includes required paper points",
                "PASS" if required.issubset(names) else "FAIL",
                "blocking",
                f"missing={sorted(required - names)}",
            )
        )
        max_turnover = max(float(row["turnover"]) for row in points)
        checks.append(
            Check(
                "Generated frontier turnover units are fractional before plotting",
                "PASS" if max_turnover < 1.0 else "FAIL",
                "blocking",
                f"max raw turnover={max_turnover:.4f}",
            )
        )
        by_name = {row["name"]: row for row in points}
        if "Ridge_Slow" in by_name:
            add_scalar_check(checks, "Frontier Ridge_Slow net SR matches generated point", float(by_name["Ridge_Slow"]["net_sr"]), 0.472)
        if "Ridge_Slow_Q" in by_name:
            add_scalar_check(checks, "Frontier Ridge_Slow_Q net SR matches generated point", float(by_name["Ridge_Slow_Q"]["net_sr"]), 0.488)
        if "WF-EMA" in by_name:
            add_scalar_check(checks, "Frontier WF-EMA net SR matches generated point", float(by_name["WF-EMA"]["net_sr"]), 0.451)


def robustness_checks(checks: list[Check], computed: dict[str, Any]) -> None:
    boot = load_csv("scripts/concern_experiments/expDH_pooled_bootstrap.csv")[0]
    computed["expDH_pooled_bootstrap"] = {key: float(value) for key, value in boot.items()}
    add_scalar_check(checks, "Pooled bootstrap one-sided p matches paper (0.10)", fval(boot, "boot_p_onesided"), 0.099, tol=0.01)
    add_scalar_check(checks, "Pooled bootstrap CI low matches paper (-0.08)", fval(boot, "boot_ci_lo"), -0.08, tol=0.02)
    add_scalar_check(checks, "Pooled bootstrap CI high matches paper (0.51)", fval(boot, "boot_ci_hi"), 0.51, tol=0.02)
    add_scalar_check(checks, "Bootstrap fraction positive matches paper (90%)", fval(boot, "boot_frac_positive"), 0.90, tol=0.02)
    checks.append(
        Check(
            "Ridge_Slow market wins match paper (11/15)",
            "PASS" if int(float(boot["market_wins"])) == 11 else "FAIL",
            "blocking",
            f"market_wins={boot['market_wins']}",
        )
    )
    add_scalar_check(checks, "Pooled t-stat matches paper (0.77)", fval(boot, "t_stat"), 0.77, tol=0.05)
    add_scalar_check(checks, "Effect size d matches paper (0.08)", fval(boot, "effect_size_d"), 0.08, tol=0.005)

    ay_rows = load_csv("scripts/concern_experiments/expAY_long_short.csv")
    short_net = group_mean(
        ay_rows, "Strategy", "Net_SR",
        {"PortfolioType": "ShortOnly", "CostMult": 3.0, "Period": "OOS"},
    )
    computed["expAY_short_only_3x"] = {key: round(value, 4) for key, value in sorted(short_net.items())}
    add_close_check(
        checks,
        "Short-only net SRs match expAY CSV",
        short_net,
        {"MomentumSort": -0.44, "Ridge": -0.76, "XGBoost": -0.98},
        tol=0.02,
    )
    checks.append(
        Check(
            "Short-only net SR negative for every strategy",
            "PASS" if all(value < 0 for value in short_net.values()) else "FAIL",
            "blocking",
            "all short-only strategies negative at 3x + borrow" if all(v < 0 for v in short_net.values()) else "some short-only strategy is positive",
        )
    )

    decomp_md = read_text("scripts/concern_experiments/expAX_turnover_decomposition_summary.md")
    t3_markers = ["SLOW: 33.6%", "FAST: 43.0%", "FAST: 46.8%", "63.4% of month-to-month"]
    missing_markers = [marker for marker in t3_markers if marker not in decomp_md]
    checks.append(
        Check(
            "Feature-speed decomposition summary matches Table 3 (lag=1 rerun)",
            "PASS" if not missing_markers else "FAIL",
            "blocking",
            f"missing markers: {missing_markers}" if missing_markers else "Ridge 43.0%/XGB 46.8% FAST importance, 63.4% pred changes",
        )
    )


def frontier_diagnostics_checks(checks: list[Check], computed: dict[str, Any]) -> None:
    """expDI: cross-sectional vs mechanical decomposition of the frontier fit.

    The paper quotes lambda = 0.33 as the per-kappa cross-sectional slope and
    reports (i) cross-sectional R^2 = 0.64-0.65 at each fixed kappa, (ii)
    kappa-alone pooled R^2 = 0.37 vs 0.72 for kappa*FSI, and (iii)
    R^2(turnover ~ FSI) = 0.73 with partial corr(FSI, deg | TO) ~ 0.25.
    """
    rows = load_csv("scripts/concern_experiments/expDI_frontier_diagnostics.csv")
    diag = {row["diagnostic"]: row for row in rows}
    computed["expDI_frontier_diagnostics"] = {
        name: round(fval(row, "value"), 4) for name, row in sorted(diag.items())
    }

    cross_r2 = {
        "kappa_1": 0.638, "kappa_1.5": 0.640, "kappa_2": 0.644, "kappa_3": 0.654,
    }
    for kappa_key, expected_r2 in cross_r2.items():
        name = f"cross_sectional_{kappa_key}"
        add_scalar_check(
            checks,
            f"Cross-sectional R2 of degradation~FSI at {kappa_key} (paper: 64-65%)",
            fval(diag[name], "value"), expected_r2, tol=0.01,
        )
    for kappa_key in cross_r2:
        name = f"cross_sectional_{kappa_key}"
        add_scalar_check(
            checks,
            f"Per-kappa slope/kappa at {kappa_key} matches lambda=0.33 convention",
            fval(diag[name], "slope_over_kappa"), 0.334, tol=0.005,
        )
    add_scalar_check(
        checks,
        "Pooled R2 on kappa alone (paper: 37%)",
        fval(diag["pooled_kappa_only_r2"], "value"), 0.372, tol=0.01,
    )
    add_scalar_check(
        checks,
        "Pooled R2 on kappa*FSI (paper: 0.72)",
        fval(diag["pooled_kappa_fsi_r2"], "value"), 0.724, tol=0.01,
    )
    add_scalar_check(
        checks,
        "R2 of realized turnover on FSI (paper: 0.73)",
        fval(diag["turnover_on_fsi_r2"], "value"), 0.727, tol=0.01,
    )
    add_scalar_check(
        checks,
        "Partial corr(FSI, degradation | turnover) at kappa=3 (paper: ~0.25)",
        fval(diag["partial_corr_fsi_given_turnover_kappa_3"], "value"), 0.263, tol=0.02,
    )
    add_scalar_check(
        checks,
        "Within-strategy R2 at kappa=3 (paper: 0.07 -- strategy-level resolution)",
        fval(diag["within_strategy_r2_kappa_3"], "value"), 0.065, tol=0.01,
    )

    fsi_diag = load_csv("scripts/concern_experiments/expDA_fsi_diagnostic.csv")
    class_ac: dict[str, list[float]] = {}
    for row in fsi_diag:
        class_ac.setdefault(row["speed_class"], []).append(fval(row, "rank_ac"))
    class_means = {name: sum(vals) / len(vals) for name, vals in class_ac.items()}
    computed["expDA_class_rank_ac"] = {name: round(v, 4) for name, v in sorted(class_means.items())}
    add_close_check(
        checks,
        "Table 3 / Sec 4.1 class rank AC matches expDA diagnostic (0.97/0.83/0.68)",
        class_means,
        {"SLOW": 0.965, "MEDIUM": 0.830, "FAST": 0.675},
        tol=0.006,
    )
    add_scalar_check(
        checks,
        "FSI coefficient with spread control at kappa=3 (paper: unchanged)",
        fval(diag["fsi_coef_spread_controlled_kappa_3"], "value"), 1.002, tol=0.01,
    )
    add_scalar_check(
        checks,
        "Two-channel pooled R2 with non-smoothness indicator (paper: 0.73)",
        fval(diag["pooled_2ch_full_r2"], "value"), 0.734, tol=0.01,
    )
    add_scalar_check(
        checks,
        "Two-channel non-smoothness coefficient (paper: +0.22 SR)",
        fval(diag["pooled_2ch_nonsmooth_coef"], "value"), 0.223, tol=0.01,
    )
    add_scalar_check(
        checks,
        "Lambda cluster-market CI low (paper: 0.27)",
        fval(diag["lambda_ci_lo_cluster_market"], "value"), 0.266, tol=0.005,
    )
    add_scalar_check(
        checks,
        "Lambda cluster-market CI high (paper: 0.40)",
        fval(diag["lambda_ci_hi_cluster_market"], "value"), 0.402, tol=0.005,
    )
    add_scalar_check(
        checks,
        "Lambda cluster-strategy CI low (paper: 0.26)",
        fval(diag["lambda_ci_lo_cluster_strategy"], "value"), 0.256, tol=0.005,
    )
    add_scalar_check(
        checks,
        "Lambda cluster-strategy CI high (paper: 0.41)",
        fval(diag["lambda_ci_hi_cluster_strategy"], "value"), 0.411, tol=0.005,
    )


def joint_and_exclusion_checks(checks: list[Check], computed: dict[str, Any]) -> None:
    """expDJ joint sign-flip test and expDK GRC-exclusion robustness."""
    dj = load_csv("scripts/concern_experiments/expDJ_joint_interventions.csv")
    joint = next(row for row in dj if row["test"] == "JOINT_all_eight")
    interventions = [row for row in dj if row["test"] != "JOINT_all_eight"]
    computed["expDJ_joint_interventions"] = {
        row["test"]: {
            "mean_sr_diff": round(fval(row, "mean_sr_diff"), 4),
            "signflip_p": round(fval(row, "signflip_p_onesided"), 4),
        }
        for row in dj
    }
    add_scalar_check(
        checks,
        "Joint sign-flip p across eight interventions (paper: 0.008)",
        fval(joint, "signflip_p_onesided"), 0.0079, tol=0.001,
    )
    add_scalar_check(
        checks,
        "Joint mean SR advantage over MomSort (paper: +0.11)",
        fval(joint, "mean_sr_diff"), 0.112, tol=0.005,
    )
    checks.append(
        Check(
            "Eight interventions enter the joint test",
            "PASS" if len(interventions) == 8 else "FAIL",
            "blocking",
            f"{len(interventions)} interventions in expDJ CSV",
        )
    )
    off_band = {
        row["test"]: round(fval(row, "mean_sr_diff"), 3)
        for row in interventions
        if not 0.0 < fval(row, "mean_sr_diff")
    }
    checks.append(
        Check(
            "Every intervention beats MomSort on mean SR",
            "PASS" if not off_band else "FAIL",
            "blocking",
            f"non-positive mean diffs: {off_band}" if off_band else "all eight mean SR diffs positive",
        )
    )

    dk = load_csv("scripts/concern_experiments/expDK_grc_exclusion.csv")
    by_strategy = {row["strategy"]: row for row in dk}
    computed["expDK_grc_exclusion"] = {
        name: round(fval(row, "net_sr_ex_grc"), 4) for name, row in sorted(by_strategy.items())
    }
    add_scalar_check(
        checks,
        "Ex-GRC Ridge_Slow net SR (paper: 0.45)",
        fval(by_strategy["Ridge_Slow"], "net_sr_ex_grc"), 0.453, tol=0.005,
    )
    add_scalar_check(
        checks,
        "Ex-GRC MomentumSort net SR (paper: 0.36)",
        fval(by_strategy["MomentumSort"], "net_sr_ex_grc"), 0.355, tol=0.005,
    )
    frontier_names = [
        "Ridge_Slow", "PersistenceFilter", "SpeedTercile", "RT_Slow",
        "EMA_0.35", "WF_EMA", "Ridge_All_Q", "Ridge_Slow_Q",
    ]
    band = [fval(by_strategy[name], "net_sr_ex_grc") for name in frontier_names]
    band_ok = abs(min(band) - 0.424) <= 0.005 and abs(max(band) - 0.470) <= 0.005
    checks.append(
        Check(
            "Ex-GRC frontier band matches paper (0.42-0.47)",
            "PASS" if band_ok else "FAIL",
            "blocking",
            f"band [{min(band):.3f}, {max(band):.3f}]",
        )
    )


def full_oos_tree_ema_and_construction_checks(checks: list[Check], computed: dict[str, Any]) -> None:
    """expDM full-OOS XGB+EMA (replaces post-2020 tree-EMA claims) and expDN value-weighted construction."""
    dm = load_csv("scripts/concern_experiments/expDM_xgb_ema_full_oos.csv")
    dm_oos3 = [r for r in dm if r["Period"] == "OOS" and float(r["CostMult"]) == 3.0]

    def mean_by_strategy(rows: list[dict[str, str]], col: str) -> dict[str, float]:
        out: dict[str, list[float]] = {}
        for r in rows:
            out.setdefault(r["Strategy"], []).append(fval(r, col))
        return {k: sum(v) / len(v) for k, v in out.items()}

    dm_sr = mean_by_strategy(dm_oos3, "Net_SR")
    dm_to = mean_by_strategy(dm_oos3, "Turnover")
    computed["expDM_full_oos_tree_ema"] = {k: round(v, 4) for k, v in sorted(dm_sr.items())}
    add_scalar_check(
        checks,
        "Full-OOS XGBoost net SR at 3x (paper: 0.24)",
        dm_sr["XGBoost"], 0.237, tol=0.005,
    )
    add_scalar_check(
        checks,
        "Full-OOS XGB walk-forward EMA net SR at 3x (paper: 0.35)",
        dm_sr["XGB_WF_EMA"], 0.351, tol=0.005,
    )
    add_scalar_check(
        checks,
        "Full-OOS best post-hoc fixed alpha=0.2 net SR (paper: 0.45)",
        dm_sr["XGB_EMA_0.2"], 0.446, tol=0.005,
    )
    add_scalar_check(
        checks,
        "Full-OOS XGB turnover reduction 36%->25% (WF EMA)",
        dm_to["XGB_WF_EMA"], 0.246, tol=0.005,
    )
    fixed_grid = {k: v for k, v in dm_sr.items() if k.startswith("XGB_EMA_")}
    best_fixed = max(fixed_grid, key=fixed_grid.get)
    checks.append(
        Check(
            "Best fixed EMA alpha on full OOS is 0.2 (post hoc)",
            "PASS" if best_fixed == "XGB_EMA_0.2" else "FAIL",
            "blocking",
            f"best fixed grid point: {best_fixed} ({fixed_grid[best_fixed]:.3f})",
        )
    )
    ax = load_csv("scripts/concern_experiments/expAX_full_comparison.csv")
    ax_mom = [
        fval(r, "Net_SR") for r in ax
        if r["Strategy"] == "MomentumSort" and r["Period"] == "OOS" and float(r["CostMult"]) == 3.0
    ]
    mom_match = abs(dm_sr["MomentumSort"] - sum(ax_mom) / len(ax_mom)) < 1e-6
    checks.append(
        Check(
            "expDM MomentumSort exactly reproduces expAX (pipeline fidelity)",
            "PASS" if mom_match else "FAIL",
            "blocking",
            f"expDM {dm_sr['MomentumSort']:.4f} vs expAX {sum(ax_mom) / len(ax_mom):.4f}",
        )
    )
    paper_text = read_text("paper_icaifv2/main_v6_frontier.tex")
    replicate_disclosed = "does not replicate on the full sample" in paper_text
    stale_claim = "net SR rises to 0.53" in paper_text or "achieves net SR 0.53" in paper_text
    checks.append(
        Check(
            "Paper discloses post-2020 tree-EMA 0.53 does not replicate on full OOS",
            "PASS" if replicate_disclosed and not stale_claim else "FAIL",
            "blocking",
            "non-replication disclosed and stale 0.53 claim removed" if replicate_disclosed and not stale_claim else "paper still asserts the post-2020 0.53 result or omits the non-replication disclosure",
        )
    )

    dn = load_csv("scripts/concern_experiments/expDN_construction_robustness.csv")
    dn_oos3 = [r for r in dn if r["Period"] == "OOS" and float(r["CostMult"]) == 3.0]
    dn_sr = mean_by_strategy(dn_oos3, "Net_SR")
    computed["expDN_value_weighted"] = {k: round(v, 4) for k, v in sorted(dn_sr.items())}
    add_scalar_check(
        checks,
        "Value-weighted Ridge_Slow net SR at 3x (paper: 0.40)",
        dn_sr["Ridge_Slow"], 0.400, tol=0.005,
    )
    add_scalar_check(
        checks,
        "Value-weighted MomentumSort net SR at 3x (paper: 0.31)",
        dn_sr["MomentumSort"], 0.305, tol=0.005,
    )
    add_scalar_check(
        checks,
        "Value-weighted Ridge_All net SR at 3x (paper: 0.30)",
        dn_sr["Ridge_All"], 0.298, tol=0.005,
    )
    vw_slow = {r["Market"]: fval(r, "Net_SR") for r in dn_oos3 if r["Strategy"] == "Ridge_Slow"}
    vw_mom = {r["Market"]: fval(r, "Net_SR") for r in dn_oos3 if r["Strategy"] == "MomentumSort"}
    wins = sum(1 for m in vw_slow if vw_slow[m] > vw_mom[m])
    positive = sum(1 for v in vw_slow.values() if v > 0)
    checks.append(
        Check(
            "Value-weighted Ridge_Slow wins 10/15, positive 14/15",
            "PASS" if wins == 10 and positive == 14 else "FAIL",
            "blocking",
            f"wins {wins}/15, positive {positive}/15",
        )
    )
    hierarchy_ok = dn_sr["Ridge_Slow"] > dn_sr["MomentumSort"] > dn_sr["Ridge_All"]
    checks.append(
        Check(
            "Value-weighted hierarchy Ridge_Slow > MomSort > Ridge_All holds",
            "PASS" if hierarchy_ok else "FAIL",
            "blocking",
            f"Ridge_Slow {dn_sr['Ridge_Slow']:.3f} > MomSort {dn_sr['MomentumSort']:.3f} > Ridge_All {dn_sr['Ridge_All']:.3f}" if hierarchy_ok else "hierarchy violated under value weighting",
        )
    )


def us_validation_checks(checks: list[Check], computed: dict[str, Any]) -> None:
    rows = load_csv("scripts/concern_experiments/expCM_us_comparison.csv")
    computed_rows: dict[str, dict[str, float]] = {}
    expected = {
        ("MomentumSort", 0.0): 0.775,
        ("MomentumSort", 1.0): 0.693,
        ("MomentumSort", 2.0): 0.612,
        ("MomentumSort", 3.0): 0.530,
        ("Ridge_All", 0.0): 0.580,
        ("Ridge_All", 1.0): 0.445,
        ("Ridge_All", 2.0): 0.310,
        ("Ridge_All", 3.0): 0.175,
        ("Ridge_Slow", 0.0): 0.638,
        ("Ridge_Slow", 1.0): 0.596,
        ("Ridge_Slow", 2.0): 0.553,
        ("Ridge_Slow", 3.0): 0.510,
        ("Ridge_Fast", 0.0): 0.404,
        ("Ridge_Fast", 1.0): 0.206,
        ("Ridge_Fast", 2.0): 0.008,
        ("Ridge_Fast", 3.0): -0.189,
    }
    actual: dict[str, float] = {}
    for strategy, cost in expected:
        vals = [
            fval(row, "Net_SR")
            for row in rows
            if row["Period"] == "OOS"
            and row["Strategy"] == strategy
            and abs(fval(row, "CostMult") - cost) <= 1e-9
        ]
        label = f"{strategy}@{cost:g}x"
        if vals:
            actual[label] = vals[0]
            computed_rows[label] = {"Net_SR": round(vals[0], 4)}
    computed["expCM_us_oos"] = computed_rows
    add_close_check(
        checks,
        "US validation table net SRs match expCM CSV",
        actual,
        {f"{strategy}@{cost:g}x": value for (strategy, cost), value in expected.items()},
        tol=0.015,
    )


def source_and_provenance_checks(checks: list[Check]) -> None:
    for script in [
        "scripts/concern_experiments/expAX_full_comparison.py",
        "scripts/concern_experiments/expAH_feature_speed_test.py",
    ]:
        name = Path(script).name
        checks.append(
            Check(
                f"{name} excludes market cap from predictors",
                pass_fail(source_has(script, r"EXCLUDE_COLS\s*=.*['\"]me['\"].*['\"]market_equity['\"]")),
                "blocking",
                "EXCLUDE_COLS contains me and market_equity",
            )
        )
        checks.append(
            Check(
                f"{name} lags market cap before universe filter",
                pass_fail(source_has(script, r"df\['me'\]\s*=\s*df\.groupby\(id_col\)\['me'\]\.shift\(1\)")),
                "blocking",
                "market cap is shifted one month before top-N filtering",
            )
        )
        checks.append(
            Check(
                f"{name} uses past-only walk-forward training dates",
                pass_fail("train_dates = unique_dates[:t_idx]" in read_text(script)),
                "blocking",
                "train_dates are strictly before the current test date",
            )
        )
        checks.append(
            Check(
                f"{name} fits preprocessing inside the walk-forward loop",
                pass_fail("imputer.fit_transform" in read_text(script)),
                "blocking",
                "imputer is fit on training data inside walk-forward execution",
            )
        )

    bound_text = read_text("scripts/concern_experiments/generate_bound_figure.py")
    paper_text = read_text("paper_icaifv2/main_v6_frontier.tex")
    checks.append(
        Check(
            "Bound figure FSI formula matches paper text",
            pass_fail(
                "equal-weight feature-set FSI" in bound_text
                and "equal-weight average instability" in paper_text
                and "coefficient-weighted average feature" not in paper_text
            ),
            "blocking",
            "script and paper both use equal-weight feature-set FSI",
        )
    )
    checks.append(
        Check(
            "Bound figure header is current",
            "FAIL" if "R² = 0.95" in bound_text or "R2 = 0.95" in bound_text else "PASS",
            "advisory",
            "script header still references an older R2=0.95 claim" if "R² = 0.95" in bound_text or "R2 = 0.95" in bound_text else "header does not contain stale R2=0.95 claim",
        )
    )

    frontier_text = read_text("scripts/concern_experiments/generate_frontier_figure.py")
    double_scaled = "add_point(name, net_sr, to * 100" in frontier_text
    checks.append(
        Check(
            "Frontier figure does not double-scale tree turnover",
            "FAIL" if double_scaled else "PASS",
            "blocking",
            "tree turnover is double-scaled" if double_scaled else "turnover is stored as fractions and scaled only at plotting time",
        )
    )
    checks.append(
        Check(
            "Frontier figure is generated from CSVs rather than hard-coded paper values",
            "FAIL" if "tree_data = {" in frontier_text or "add_point('WF-EMA', 0.451" in frontier_text else "PASS",
            "blocking",
            "frontier script reads paper points from canonical CSVs",
        )
    )

    df_text = read_text("scripts/concern_experiments/expDF_tree_ema.py")
    checks.append(
        Check(
            "Tree EMA post-2020 OOS window is labeled in paper",
            "PASS" if "2020-12-31" in df_text and re.search(r"post-2020\s+(OOS\s+)?validation", paper_text) else ("PASS" if "2020-12-31" not in df_text else "WARN"),
            "advisory",
            "paper labels XGB_EMA as a separate post-2020 validation sample" if re.search(r"post-2020\s+(OOS\s+)?validation", paper_text) else "expDF_tree_ema uses a fixed post-2020 OOS boundary and should be labeled separately",
        )
    )

    ar_text = read_text("scripts/concern_experiments/expAR_ema_robustness.py")
    wf_precise = "not a nested out-of-fold validation" in ar_text and "does not use future OOS months" in paper_text
    checks.append(
        Check(
            "WF-EMA alpha-selection wording matches implementation",
            "PASS" if wf_precise else ("WARN" if "is_eval_dates = train_dates[-IS_EVAL_MONTHS:]" in ar_text and "model.predict(X_is_scaled)" in ar_text else "PASS"),
            "advisory",
            "paper states no future OOS months and script notes this is not nested OOF validation" if wf_precise else "alpha selection uses recent in-sample months from the current training model; wording should be precise",
        )
    )
    stale_lag_header = "dual-lag (accounting=4, market=1)" in ar_text
    checks.append(
        Check(
            "EMA script lag header matches implementation",
            "WARN" if stale_lag_header else "PASS",
            "advisory",
            "header says accounting=4 while implementation applies lag=1 to current pre-lagged data convention" if stale_lag_header else "header documents lag=1 on pre-lagged JKP/GFD accounting fields",
        )
    )

    stale_summary = read_text("paper_icaifv2/results_summary.md")
    stale_markers = ["Ridge_All | -0.06", "sorts dominate", "Momentum sort beats Ridge in 14/15"]
    has_stale_summary = any(marker in stale_summary for marker in stale_markers)
    checks.append(
        Check(
            "Stale paper results summary is removed from evidence path",
            "FAIL" if has_stale_summary else "PASS",
            "blocking",
            "paper_icaifv2/results_summary.md contains old negative Ridge/sorts-dominate narrative" if has_stale_summary else "paper_icaifv2/results_summary.md points to the public verification outputs",
        )
    )

    stat_summary = read_text("scripts/concern_experiments/expAX_statistical_tests_summary.md")
    has_stale_stats = "+0.4195" in stat_summary or ("SR_Ridge" in stat_summary and "-0.5047" in stat_summary)
    checks.append(
        Check(
            "Statistical-test summary matches current v6 outputs",
            "FAIL" if has_stale_stats else "PASS",
            "blocking",
            "expAX_statistical_tests_summary.md is stale relative to current expAX CSV" if has_stale_stats else "statistical summary was regenerated from current expAX CSV",
        )
    )


def markdown_table(mapping: dict[str, Any]) -> str:
    if not mapping:
        return "_No data._\n"
    keys = sorted(mapping)
    metric_keys = sorted({metric for value in mapping.values() if isinstance(value, dict) for metric in value})
    if not metric_keys:
        return "\n".join(f"- `{key}`: `{mapping[key]}`" for key in keys) + "\n"
    lines = ["| Item | " + " | ".join(metric_keys) + " |"]
    lines.append("|---|" + "|".join("---" for _ in metric_keys) + "|")
    for key in keys:
        value = mapping[key]
        cells = []
        for metric in metric_keys:
            val = value.get(metric, "")
            cells.append(f"{val:.4f}" if isinstance(val, float) else str(val))
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_report(checks: list[Check], computed: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = defaultdict(int)
    for check in checks:
        counts[check.status] += 1
    lines = [
        "# ICAIF v2 Verification Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Summary",
        "",
        f"- PASS: {counts['PASS']}",
        f"- WARN: {counts['WARN']}",
        f"- FAIL: {counts['FAIL']}",
        "",
        "## Checks",
        "",
        "| Status | Severity | Check | Detail |",
        "|---|---|---|---|",
    ]
    for check in checks:
        detail = check.detail.replace("|", "\\|")
        lines.append(f"| {check.status} | {check.severity} | {check.name} | {detail} |")
    lines.extend(["", "## Computed Claim Tables", ""])
    for name, table in computed.items():
        lines.extend([f"### {name}", "", markdown_table(table), ""])
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def print_checks(checks: list[Check]) -> None:
    for check in checks:
        print(f"[{check.status}] {check.name}: {check.detail}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify ICAIF v2 paper claims against canonical result CSVs."
    )
    parser.add_argument(
        "--write-report",
        nargs="?",
        const=str(DEFAULT_REPORT),
        default=None,
        help="Write a markdown report. Optional path defaults to paper_icaifv2/public/results/verification_report.md.",
    )
    parser.add_argument(
        "--write-json",
        nargs="?",
        const=str(DEFAULT_JSON),
        default=None,
        help="Write computed claim tables as JSON. Optional path defaults to paper_icaifv2/public/results/computed_claims.json.",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Return exit code 0 even if blocking checks fail.",
    )
    args = parser.parse_args()

    checks: list[Check] = []
    computed: dict[str, Any] = {}

    main_table_checks(checks, computed)
    feature_speed_checks(checks, computed)
    supporting_result_checks(checks, computed)
    ic_and_decay_checks(checks, computed)
    tree_and_transfer_checks(checks, computed)
    generated_figure_checks(checks, computed)
    robustness_checks(checks, computed)
    frontier_diagnostics_checks(checks, computed)
    joint_and_exclusion_checks(checks, computed)
    full_oos_tree_ema_and_construction_checks(checks, computed)
    us_validation_checks(checks, computed)
    source_and_provenance_checks(checks)

    print_checks(checks)

    if args.write_report:
        write_report(checks, computed, REPO_ROOT / args.write_report if not Path(args.write_report).is_absolute() else Path(args.write_report))
    if args.write_json:
        out = REPO_ROOT / args.write_json if not Path(args.write_json).is_absolute() else Path(args.write_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(computed, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    has_blocking_failure = any(check.status == "FAIL" and check.severity == "blocking" for check in checks)
    if has_blocking_failure and not args.allow_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
