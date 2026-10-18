#!/usr/bin/env python3
"""Generate the EMA smoothing figure (alpha vs. net SR) for the ICAIF v2 paper.

Reads the expAR EMA robustness grid and plots mean OOS net Sharpe ratio at
3x costs across 15 markets for each fixed smoothing parameter, the
walk-forward EMA, and the MomentumSort benchmark. Styling follows the
cost-speed bound figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "paper_icaifv2" / "figures"

EMA_SOURCE = REPO_ROOT / "scripts" / "concern_experiments" / "expAR_ema_robustness.csv"
COST_MULT = 3.0

COLORS = {
    "Ridge": "#2166ac",
    "WF_EMA": "#1b7837",
    "MomentumSort": "#333333",
}


def load_grid() -> tuple[pd.DataFrame, float, float]:
    df = pd.read_csv(EMA_SOURCE)
    df = df[(df["Period"] == "OOS") & (df["CostMult"] == COST_MULT)]
    means = df.groupby("Strategy")["Net_SR"].mean()

    fixed = means[means.index.str.fullmatch(r"Ridge_EMA_[0-9.]+")]
    grid = pd.DataFrame(
        {
            "alpha": fixed.index.str.replace("Ridge_EMA_", "", regex=False).astype(float),
            "net_sr": fixed.values,
        }
    ).sort_values("alpha")
    return grid, float(means["Ridge_WF_EMA"]), float(means["MomentumSort"])


def plot(grid: pd.DataFrame, wf_sr: float, mom_sr: float) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    best = grid.loc[grid["net_sr"].idxmax()]

    fig, ax = plt.subplots(figsize=(4.7, 3.2))

    ax.axhline(
        mom_sr,
        color=COLORS["MomentumSort"],
        alpha=0.55,
        linewidth=1.2,
        linestyle="--",
        label=f"MomentumSort (SR = {mom_sr:.2f})",
    )
    ax.plot(
        grid["alpha"],
        grid["net_sr"],
        color=COLORS["Ridge"],
        linewidth=1.5,
        marker="o",
        markersize=5,
        markeredgecolor="white",
        markeredgewidth=0.3,
        label=r"Fixed $\alpha$",
        zorder=4,
    )
    ax.scatter(
        best["alpha"],
        best["net_sr"],
        s=110,
        facecolors="none",
        edgecolors="black",
        linewidths=1.2,
        label=rf"Best fixed ($\alpha$ = {best['alpha']:.2f}, SR = {best['net_sr']:.2f})",
        zorder=5,
    )
    ax.scatter(
        best["alpha"],
        wf_sr,
        s=130,
        marker="*",
        c=COLORS["WF_EMA"],
        edgecolors="white",
        linewidths=0.3,
        label=f"Walk-forward EMA (SR = {wf_sr:.2f})",
        zorder=5,
    )

    ax.set_xlabel(r"EMA smoothing parameter $\alpha$ (1.0 = no smoothing)", fontsize=11)
    ax.set_ylabel("Mean OOS net SR (3x costs)", fontsize=11)
    ax.set_xlim(-0.02, 1.04)
    ax.set_ylim(0.25, 0.55)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.9, edgecolor="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "ema_alpha_curve.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / "ema_alpha_curve.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> int:
    grid, wf_sr, mom_sr = load_grid()
    plot(grid, wf_sr, mom_sr)
    print(f"Best fixed alpha: {grid.loc[grid['net_sr'].idxmax(), 'alpha']:.2f}")
    print(f"WF-EMA net SR: {wf_sr:.4f}; MomentumSort net SR: {mom_sr:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
