#!/usr/bin/env python3
"""Build paper tables from the verified ICAIF v2 CSV outputs."""

from __future__ import annotations

import sys
from math import isnan
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = REPO_ROOT / "paper_icaifv2" / "public" / "results" / "tables"

sys.path.insert(0, str(SCRIPT_DIR))
import verify_paper_claims as verify  # noqa: E402


def fmt_num(value: float) -> str:
    return f"{value:.3f}"


def fmt_pct(value: float) -> str:
    return f"{100 * value:.1f}\\%"


def fmt_optional(value: float | None) -> str:
    if value is None or isnan(value):
        return ""
    return fmt_num(value)


def latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def write_latex_table(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    col_spec = "l" + "r" * (len(headers) - 1)
    lines = [
        "\\begin{tabular}{" + col_spec + "}",
        "\\toprule",
        " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown_table(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row).replace("\\%", "%") + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_table(stem: str, headers: list[str], rows: list[list[str]]) -> None:
    write_latex_table(OUT_DIR / f"{stem}.tex", headers, rows)
    write_markdown_table(OUT_DIR / f"{stem}.md", headers, rows)


def main_comparison() -> None:
    csv_rows = verify.load_csv("scripts/concern_experiments/expAX_full_comparison.csv")
    filters = {"Period": "OOS", "CostMult": 3.0}
    net = verify.group_mean(csv_rows, "Strategy", "Net_SR", filters)
    gross = verify.group_mean(csv_rows, "Strategy", "Gross_SR", filters)
    turnover = verify.group_mean(csv_rows, "Strategy", "Turnover", filters)
    order = [
        "MomentumML_Blend",
        "MomentumSort",
        "Ridge",
        "CompositeSort",
        "XGBoost",
        "LightGBM",
        "LowVolSort",
    ]
    rows = [
        [latex_escape(strategy), fmt_num(net[strategy]), fmt_num(gross[strategy]), fmt_pct(turnover[strategy])]
        for strategy in order
        if strategy in net
    ]
    write_table("table_main_comparison_3x", ["Strategy", "Net SR", "Gross SR", "Turnover"], rows)


def feature_speed() -> None:
    csv_rows = verify.load_csv("scripts/concern_experiments/expAH_feature_speed_test.csv")
    filters = {"Period": "OOS", "CostMult": 3.0}
    net = verify.group_mean(csv_rows, "Strategy", "Net_SR", filters)
    gross = verify.group_mean(csv_rows, "Strategy", "Gross_SR", filters)
    turnover = verify.group_mean(csv_rows, "Strategy", "Turnover", filters)
    order = [
        "MomentumSort",
        "Ridge_All",
        "Ridge_Slow",
        "Ridge_SlowMed",
        "Ridge_Fast",
        "Ridge_All_Q",
        "Ridge_Slow_Q",
    ]
    rows = [
        [latex_escape(strategy), fmt_num(net[strategy]), fmt_num(gross[strategy]), fmt_pct(turnover[strategy])]
        for strategy in order
        if strategy in net
    ]
    write_table("table_feature_speed_3x", ["Strategy", "Net SR", "Gross SR", "Turnover"], rows)


def ema_table() -> None:
    csv_rows = verify.load_csv("scripts/concern_experiments/expAR_ema_robustness.csv")
    filters = {"Period": "OOS", "CostMult": 3.0}
    net = verify.group_mean(csv_rows, "Strategy", "Net_SR", filters)
    gross = verify.group_mean(csv_rows, "Strategy", "Gross_SR", filters)
    turnover = verify.group_mean(csv_rows, "Strategy", "Turnover", filters)
    stability = verify.group_mean(csv_rows, "Strategy", "Pred_Stability", filters)
    order = ["MomentumSort", "Ridge_EMA_0.25", "Ridge_EMA_0.35", "Ridge_WF_EMA"]
    rows = []
    for strategy in order:
        if strategy not in net:
            continue
        stab = fmt_optional(stability.get(strategy))
        rows.append(
            [
                latex_escape(strategy),
                fmt_num(net[strategy]),
                fmt_num(gross[strategy]),
                fmt_pct(turnover[strategy]),
                stab,
            ]
        )
    write_table("table_ema_robustness_3x", ["Strategy", "Net SR", "Gross SR", "Turnover", "Stability"], rows)


def us_validation() -> None:
    csv_rows = verify.load_csv("scripts/concern_experiments/expCM_us_comparison.csv")
    strategies = ["MomentumSort", "Ridge_All", "Ridge_Slow", "Ridge_Fast"]
    costs = [0.0, 1.0, 2.0, 3.0]
    rows = []
    for strategy in strategies:
        row = [latex_escape(strategy)]
        for cost in costs:
            vals = [
                verify.fval(item, "Net_SR")
                for item in csv_rows
                if item["Period"] == "OOS"
                and item["Strategy"] == strategy
                and abs(verify.fval(item, "CostMult") - cost) <= 1e-9
            ]
            row.append(fmt_num(vals[0]) if vals else "")
        rows.append(row)
    write_table("table_us_validation", ["Strategy", "0x", "1x", "2x", "3x"], rows)


def transfer_summary() -> None:
    row = verify.load_csv("scripts/concern_experiments/expDE_transfer_test.csv")[0]
    rows = [
        ["In-sample R2", fmt_num(verify.fval(row, "in_sample_r2"))],
        ["Leave-one-market-out R2", fmt_num(verify.fval(row, "loo_mean_r2"))],
        ["Leave-five-markets-out R2", fmt_num(verify.fval(row, "l5o_mean_r2"))],
        ["Lambda std. across LOMO", fmt_num(verify.fval(row, "loo_lambda_std"))],
        ["Slow-feature Jaccard", fmt_num(verify.fval(row, "slow_jaccard_mean"))],
        ["Universal slow features", str(int(float(row["universal_slow_count"])))],
    ]
    write_table("table_transfer_summary", ["Diagnostic", "Value"], rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    main_comparison()
    feature_speed()
    ema_table()
    us_validation()
    transfer_summary()
    print(f"Wrote paper tables to {OUT_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
