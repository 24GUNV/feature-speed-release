#!/usr/bin/env python3
"""Generate the verified cost-speed bound figure for the ICAIF v2 paper.

The paper uses an ex-ante, equal-weight feature-set FSI:

    FSI(S) = mean_{j in S}(1 - rank_autocorrelation_j)

This is the quantity available before model fitting for every smooth strategy
in the bound. The script writes both the figure and machine-readable summaries
that the public verifier can check.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "paper_icaifv2" / "figures"
PUBLIC_RESULTS = REPO_ROOT / "paper_icaifv2" / "public" / "results"

FSI_SOURCE = REPO_ROOT / "scripts" / "concern_experiments" / "expDA_fsi_diagnostic.csv"
PERF_SOURCE = REPO_ROOT / "scripts" / "concern_experiments" / "expAH_feature_speed_test.csv"

SMOOTH_STRATEGIES = {
    "Ridge_All": None,
    "Ridge_Slow": ["SLOW"],
    "Ridge_SlowMed": ["SLOW", "MEDIUM"],
    "Ridge_Fast": ["FAST"],
}
COST_LEVELS = [1.0, 1.5, 2.0, 3.0]


def compute_fsi_equal_weight(features: pd.DataFrame) -> float:
    values = features["rank_ac"].dropna()
    if values.empty:
        return np.nan
    return float((1.0 - values).mean())


def feature_subset(fsi_raw: pd.DataFrame, market: str, speed_filter: list[str] | None) -> pd.DataFrame:
    features = fsi_raw[fsi_raw["market"] == market]
    if speed_filter is None:
        return features
    return features[features["speed_class"].isin(speed_filter)]


def momentum_fsi(fsi_raw: pd.DataFrame, market: str) -> float:
    market_features = fsi_raw[fsi_raw["market"] == market]
    momentum = market_features[market_features["feature"].str.fullmatch("ret_12_1", na=False)]
    if momentum.empty:
        momentum = market_features[market_features["feature"].str.contains("ret_12_1", na=False)]
    if momentum.empty:
        raise ValueError(f"Missing ret_12_1 rank autocorrelation for {market}")
    return float(1.0 - momentum.iloc[0]["rank_ac"])


def build_bound_panel() -> pd.DataFrame:
    fsi_raw = pd.read_csv(FSI_SOURCE)
    perf = pd.read_csv(PERF_SOURCE)
    perf = perf[perf["Period"] == "OOS"].copy()

    rows = []
    for market in sorted(fsi_raw["market"].unique()):
        for strategy, speed_filter in SMOOTH_STRATEGIES.items():
            features = feature_subset(fsi_raw, market, speed_filter)
            fsi = compute_fsi_equal_weight(features)
            for kappa in COST_LEVELS:
                match = perf[
                    (perf["Market"] == market)
                    & (perf["Strategy"] == strategy)
                    & (perf["CostMult"] == kappa)
                ]
                if match.empty:
                    continue
                item = match.iloc[0]
                rows.append(
                    {
                        "market": market,
                        "strategy": strategy,
                        "kappa": kappa,
                        "fsi": fsi,
                        "kappa_fsi": kappa * fsi,
                        "gross_sr": item["Gross_SR"],
                        "net_sr": item["Net_SR"],
                        "degradation": item["Gross_SR"] - item["Net_SR"],
                        "turnover": item["Turnover"],
                    }
                )

        mom_fsi = momentum_fsi(fsi_raw, market)
        for kappa in COST_LEVELS:
            match = perf[
                (perf["Market"] == market)
                & (perf["Strategy"] == "MomentumSort")
                & (perf["CostMult"] == kappa)
            ]
            if match.empty:
                continue
            item = match.iloc[0]
            rows.append(
                {
                    "market": market,
                    "strategy": "MomentumSort",
                    "kappa": kappa,
                    "fsi": mom_fsi,
                    "kappa_fsi": kappa * mom_fsi,
                    "gross_sr": item["Gross_SR"],
                    "net_sr": item["Net_SR"],
                    "degradation": item["Gross_SR"] - item["Net_SR"],
                    "turnover": item["Turnover"],
                }
            )

    panel = pd.DataFrame(rows).dropna(subset=["fsi", "gross_sr", "net_sr", "degradation"])
    expected = 5 * 15 * len(COST_LEVELS)
    if len(panel) != expected:
        raise ValueError(f"Expected {expected} bound observations, found {len(panel)}")
    return panel


def fit_line(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[LinearRegression, float]:
    reg = LinearRegression().fit(df[[x_col]], df[y_col])
    pred = reg.predict(df[[x_col]])
    ss_res = float(((df[y_col] - pred) ** 2).sum())
    ss_tot = float(((df[y_col] - df[y_col].mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot
    return reg, r2


def write_outputs(panel: pd.DataFrame, reg: LinearRegression, r2: float) -> pd.DataFrame:
    PUBLIC_RESULTS.mkdir(parents=True, exist_ok=True)

    per_kappa = []
    for kappa in COST_LEVELS:
        sub = panel[panel["kappa"] == kappa].copy()
        reg_k, r2_k = fit_line(sub, "kappa_fsi", "degradation")
        per_kappa.append(
            {
                "kappa": kappa,
                "lambda": float(reg_k.coef_[0]),
                "intercept": float(reg_k.intercept_),
                "r2": r2_k,
                "n": len(sub),
            }
        )

    summary = pd.DataFrame(
        [
            {
                "spec": "degradation_on_kappa_fsi",
                "fsi_definition": "equal_weight_feature_set",
                "lambda": float(reg.coef_[0]),
                "intercept": float(reg.intercept_),
                "r2": r2,
                "n": len(panel),
                "per_kappa_lambda_mean": float(np.mean([row["lambda"] for row in per_kappa])),
                "per_kappa_lambda_std": float(np.std([row["lambda"] for row in per_kappa], ddof=0)),
            }
        ]
    )

    panel.to_csv(PUBLIC_RESULTS / "cost_speed_bound_points.csv", index=False)
    summary.to_csv(PUBLIC_RESULTS / "cost_speed_bound_summary.csv", index=False)
    pd.DataFrame(per_kappa).to_csv(PUBLIC_RESULTS / "cost_speed_bound_by_kappa.csv", index=False)
    return summary


def plot(panel: pd.DataFrame, reg: LinearRegression, r2: float) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    colors = {
        "Ridge_Slow": "#2166ac",
        "Ridge_SlowMed": "#67a9cf",
        "Ridge_All": "#92c5de",
        "Ridge_Fast": "#d6604d",
        "MomentumSort": "#333333",
    }
    markers = {
        "Ridge_Slow": "o",
        "Ridge_SlowMed": "s",
        "Ridge_All": "^",
        "Ridge_Fast": "v",
        "MomentumSort": "D",
    }
    alphas = {1.0: 0.35, 1.5: 0.50, 2.0: 0.65, 3.0: 1.0}

    fig, ax = plt.subplots(figsize=(5.5, 5))

    x_line = np.linspace(0, panel["kappa_fsi"].max() * 1.05, 100)
    y_line = reg.predict(pd.DataFrame({"kappa_fsi": x_line}))
    ax.plot(
        x_line,
        y_line,
        color="black",
        alpha=0.55,
        linewidth=1.5,
        label=f"pooled fit: slope = {reg.coef_[0]:.2f}, R^2 = {r2:.2f}",
    )

    for strategy in colors:
        for kappa in COST_LEVELS:
            sub = panel[(panel["strategy"] == strategy) & (panel["kappa"] == kappa)]
            if sub.empty:
                continue
            ax.scatter(
                sub["kappa_fsi"],
                sub["degradation"],
                c=colors[strategy],
                marker=markers[strategy],
                s=40,
                alpha=alphas[kappa],
                edgecolors="white",
                linewidths=0.3,
                label=strategy if kappa == COST_LEVELS[-1] else None,
                zorder=5,
            )

    ax.text(
        0.96,
        0.04,
        "Darker = higher cost multiplier",
        transform=ax.transAxes,
        fontsize=8,
        ha="right",
        color="gray",
        fontstyle="italic",
    )
    ax.set_xlabel("Cost multiplier x FSI", fontsize=11)
    ax.set_ylabel("SR degradation (gross - net)", fontsize=11)
    ax.set_xlim(-0.02, panel["kappa_fsi"].max() * 1.08)
    ax.set_ylim(-0.05, panel["degradation"].max() * 1.1)
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9, edgecolor="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "cost_speed_bound.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / "cost_speed_bound.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> int:
    panel = build_bound_panel()
    reg, r2 = fit_line(panel, "kappa_fsi", "degradation")
    summary = write_outputs(panel, reg, r2)
    plot(panel, reg, r2)

    row = summary.iloc[0]
    print(f"Bound observations: {int(row['n'])}")
    print(f"FSI definition: {row['fsi_definition']}")
    print(f"R2: {row['r2']:.4f}")
    print(f"lambda: {row['lambda']:.4f}")
    print(f"Per-kappa lambda mean: {row['per_kappa_lambda_mean']:.4f}")
    print("Wrote cost_speed_bound.pdf/.png and public bound summary CSVs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

