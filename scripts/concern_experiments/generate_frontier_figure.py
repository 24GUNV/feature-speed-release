#!/usr/bin/env python3
"""Generate the verified cost-speed frontier figure for the ICAIF v2 paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "paper_icaifv2" / "figures"
PUBLIC_RESULTS = REPO_ROOT / "paper_icaifv2" / "public" / "results"
COST_MULT = 3.0


def mean_point(
    csv_path: str,
    source: str,
    strategy: str,
    label: str,
    mechanism: str,
    marker: str,
    color: str,
    period_required: bool = True,
) -> dict[str, object] | None:
    df = pd.read_csv(REPO_ROOT / csv_path)
    mask = (df["Strategy"] == strategy) & (df["CostMult"] == COST_MULT)
    if period_required and "Period" in df.columns:
        mask &= df["Period"] == "OOS"
    sub = df[mask].copy()
    if sub.empty:
        return None
    return {
        "name": label,
        "source": source,
        "source_strategy": strategy,
        "net_sr": float(sub["Net_SR"].mean()),
        "gross_sr": float(sub["Gross_SR"].mean()) if "Gross_SR" in sub.columns else None,
        "turnover": float(sub["Turnover"].mean()),
        "mechanism": mechanism,
        "marker": marker,
        "color": color,
        "n_markets": int(sub["Market"].nunique()) if "Market" in sub.columns else len(sub),
    }


def collect_points() -> pd.DataFrame:
    specs = [
        # Feature selection and rebalancing from expAH.
        (
            "scripts/concern_experiments/expAH_feature_speed_test.csv",
            "expAH",
            "Ridge_Slow",
            "Ridge_Slow",
            "Feature selection",
            "o",
            "#2166ac",
        ),
        (
            "scripts/concern_experiments/expAH_feature_speed_test.csv",
            "expAH",
            "Ridge_SlowMed",
            "Ridge_SlowMed",
            "Feature selection",
            "o",
            "#67a9cf",
        ),
        (
            "scripts/concern_experiments/expAH_feature_speed_test.csv",
            "expAH",
            "Ridge_All",
            "Ridge_All",
            "Feature selection",
            "o",
            "#92c5de",
        ),
        (
            "scripts/concern_experiments/expAH_feature_speed_test.csv",
            "expAH",
            "Ridge_Fast",
            "Ridge_Fast",
            "Feature selection",
            "o",
            "#d6604d",
        ),
        (
            "scripts/concern_experiments/expAH_feature_speed_test.csv",
            "expAH",
            "MomentumSort",
            "MomentumSort",
            "Factor sort",
            "D",
            "#333333",
        ),
        (
            "scripts/concern_experiments/expAH_feature_speed_test.csv",
            "expAH",
            "Ridge_All_Q",
            "Ridge_All_Q",
            "Quarterly rebalancing",
            "p",
            "#7b3294",
        ),
        (
            "scripts/concern_experiments/expAH_feature_speed_test.csv",
            "expAH",
            "Ridge_Slow_Q",
            "Ridge_Slow_Q",
            "Quarterly rebalancing",
            "p",
            "#542788",
        ),
        # Blend from expAX.
        (
            "scripts/concern_experiments/expAX_full_comparison.csv",
            "expAX",
            "MomentumML_Blend",
            "Blend",
            "Blend",
            "h",
            "#666666",
        ),
        # Persistence and empirical slow-feature filters from expAN.
        (
            "scripts/concern_experiments/expAN_persistence_filter.csv",
            "expAN",
            "Ridge_PersFiltered",
            "PersistFilter",
            "Feature selection",
            "o",
            "#2166ac",
        ),
        (
            "scripts/concern_experiments/expAN_persistence_filter.csv",
            "expAN",
            "Ridge_EmpSlow_FS",
            "SpeedTercile",
            "Feature selection",
            "o",
            "#2166ac",
        ),
        # EMA from expAR.
        (
            "scripts/concern_experiments/expAR_ema_robustness.csv",
            "expAR",
            "Ridge_WF_EMA",
            "WF-EMA",
            "EMA smoothing",
            "*",
            "#1b7837",
        ),
        # Tree models from expCP.
        (
            "scripts/concern_experiments/expCP_tree_smoothing.csv",
            "expCP",
            "XGB_All",
            "XGB_All",
            "Tree model",
            "^",
            "#d6604d",
        ),
        (
            "scripts/concern_experiments/expCP_tree_smoothing.csv",
            "expCP",
            "XGB_Slow",
            "XGB_Slow",
            "Tree model",
            "^",
            "#f4a582",
        ),
        (
            "scripts/concern_experiments/expCP_tree_smoothing.csv",
            "expCP",
            "LGBM_All",
            "LGBM_All",
            "Tree model",
            "^",
            "#b2182b",
        ),
        (
            "scripts/concern_experiments/expCP_tree_smoothing.csv",
            "expCP",
            "LGBM_Slow",
            "LGBM_Slow",
            "Tree model",
            "^",
            "#ef8a62",
        ),
        # Cost-aware neural-network variants from expDB.
        (
            "scripts/concern_experiments/expDB_cost_aware_nn.csv",
            "expDB",
            "CostAware_Linear",
            "CostAware_Linear",
            "Cost-aware NN",
            "s",
            "#e08214",
        ),
        (
            "scripts/concern_experiments/expDB_cost_aware_nn.csv",
            "expDB",
            "CostAware_Lipschitz",
            "CostAware_Lip",
            "Cost-aware NN",
            "s",
            "#b35806",
        ),
        (
            "scripts/concern_experiments/expDB_cost_aware_nn.csv",
            "expDB",
            "Lipschitz_NN",
            "Lipschitz_NN",
            "Cost-aware NN",
            "s",
            "#fdb863",
        ),
    ]

    points = []
    for spec in specs:
        point = mean_point(*spec)
        if point is not None:
            points.append(point)
    if not points:
        raise ValueError("No frontier points were collected")
    return pd.DataFrame(points)


def plot(points: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PUBLIC_RESULTS.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhspan(0.44, 0.50, alpha=0.12, color="#2166ac", zorder=0)
    ax.axhline(y=0.47, color="#2166ac", linestyle="--", alpha=0.5, linewidth=1, zorder=1)

    highlight = {"Ridge_Slow", "WF-EMA", "Ridge_Slow_Q"}
    for _, row in points.iterrows():
        turnover_pct = row["turnover"] * 100.0
        size = 120 if row["name"] in highlight else 70
        edgecolor = "black" if row["name"] in highlight else row["color"]
        linewidth = 1.5 if row["name"] in highlight else 0.5
        ax.scatter(
            turnover_pct,
            row["net_sr"],
            c=row["color"],
            marker=row["marker"],
            s=size,
            edgecolors=edgecolor,
            linewidths=linewidth,
            zorder=5,
        )

    label_offsets = {
        "Ridge_Slow": (2.0, 0.048),
        "Ridge_All": (1.5, 0.02),
        "Ridge_Fast": (-2.5, -0.055),
        "MomentumSort": (1.5, -0.04),
        "WF-EMA": (-7.8, -0.028),
        "Ridge_Slow_Q": (-5.0, 0.035),
        "CostAware_Linear": (1.5, 0.02),
        "Lipschitz_NN": (1.5, 0.02),
        "XGB_All": (1.5, 0.02),
        "LGBM_All": (0.8, -0.045),
    }
    for _, row in points.iterrows():
        if row["name"] not in label_offsets:
            continue
        x = row["turnover"] * 100.0
        y = row["net_sr"]
        dx, dy = label_offsets[row["name"]]
        ax.annotate(
            row["name"],
            (x, y),
            xytext=(x + dx, y + dy),
            fontsize=8 if row["name"] in highlight else 7,
            fontweight="bold" if row["name"] in highlight else "normal",
            color=row["color"],
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.5) if abs(dx) > 3 else None,
        )

    ax.text(38, 0.48, "Performance ceiling", fontsize=9, fontstyle="italic", color="#2166ac", alpha=0.8, ha="right")

    handles = [
        mpatches.Patch(color="#2166ac", label="Feature selection"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#1b7837", markersize=10, label="EMA smoothing"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="#e08214", markersize=8, label="Cost-aware NN"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#d6604d", markersize=8, label="Tree model"),
        plt.Line2D([0], [0], marker="p", color="w", markerfacecolor="#7b3294", markersize=8, label="Quarterly rebalancing"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="#333333", markersize=7, label="Factor sort"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.9, edgecolor="gray")
    ax.set_xlabel("Monthly turnover (%)", fontsize=11)
    ax.set_ylabel("Net Sharpe ratio (3x costs)", fontsize=11)
    ax.set_xlim(-1, 42)
    ax.set_ylim(-0.1, 0.6)
    ax.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "cost_speed_frontier.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / "cost_speed_frontier.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> int:
    points = collect_points()
    PUBLIC_RESULTS.mkdir(parents=True, exist_ok=True)
    points.to_csv(PUBLIC_RESULTS / "cost_speed_frontier_points.csv", index=False)
    plot(points)
    print(f"Frontier points: {len(points)}")
    print("Wrote cost_speed_frontier.pdf/.png and public frontier points CSV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

