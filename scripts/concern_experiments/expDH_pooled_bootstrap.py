#!/usr/bin/env python3
"""
expDH -- Pinned pooled block bootstrap: Ridge_Slow vs MomentumSort
==================================================================

Reproducible source for the statistical-significance paragraph in the
ICAIF v2 paper (Section: Robustness / Statistical significance).

Methodology (as described in the paper):
  - Monthly net returns at 3x costs, OOS period, from expCS_monthly_returns.csv.
  - Pool across all 15 markets: equal-weight mean of available markets per month.
  - Moving-block bootstrap (block = 6 months, 10,000 replications, seed 42),
    resampling DATES so cross-market dependence is preserved.
  - Statistic: annualized SR(Ridge_Slow pooled) - SR(MomentumSort pooled).

Also reports:
  - Per-market SR wins (Ridge_Slow > MomentumSort).
  - Pooled monthly-difference effect size d, its t-statistic, and the number
    of months needed for 80% power at one-sided alpha = 0.05.

Output: expDH_pooled_bootstrap.csv
"""

import os
import numpy as np
import pandas as pd
from scipy import stats

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

SEED = 42
B = 10_000
BLOCK = 6
COST_MULT = 3.0

mr = pd.read_csv('scripts/concern_experiments/expCS_monthly_returns.csv')
mr = mr[(mr.CostMult == COST_MULT) & (mr.period == 'OOS')]

rs = mr[mr.Strategy == 'Ridge_Slow'].pivot_table(index='date', columns='Market', values='net_ret')
ms = mr[mr.Strategy == 'MomentumSort'].pivot_table(index='date', columns='Market', values='net_ret')
assert (rs.columns == ms.columns).all() and len(rs.columns) == 15

# Equal-weight pool across available markets per month
pool_rs = rs.mean(axis=1)
pool_ms = ms.mean(axis=1)
common = pool_rs.dropna().index.intersection(pool_ms.dropna().index)
x = pool_rs[common].values
y = pool_ms[common].values
T = len(x)


def ann_sr(v):
    return v.mean() / v.std(ddof=1) * np.sqrt(12)


obs_diff = ann_sr(x) - ann_sr(y)

rng = np.random.default_rng(SEED)
nblocks = int(np.ceil(T / BLOCK))
boot = np.empty(B)
for b in range(B):
    starts = rng.integers(0, T - BLOCK + 1, nblocks)
    idx = np.concatenate([np.arange(s, s + BLOCK) for s in starts])[:T]
    boot[b] = ann_sr(x[idx]) - ann_sr(y[idx])

p_onesided = float((boot <= 0).mean())
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
frac_positive = float((boot > 0).mean())

# Per-market wins
sr_by_mkt_rs = {m: ann_sr(rs[m].dropna().values) for m in rs.columns}
sr_by_mkt_ms = {m: ann_sr(ms[m].dropna().values) for m in ms.columns}
wins = sum(sr_by_mkt_rs[m] > sr_by_mkt_ms[m] for m in rs.columns)

# Pooled monthly-difference effect size and power
diff = x - y
d = diff.mean() / diff.std(ddof=1)
t_stat = d * np.sqrt(T)
p_t_onesided = float(1 - stats.t.cdf(t_stat, df=T - 1))
n_needed = ((stats.norm.ppf(0.95) + stats.norm.ppf(0.80)) / d) ** 2

out = pd.DataFrame([{
    'obs_sr_diff': obs_diff,
    'boot_p_onesided': p_onesided,
    'boot_ci_lo': ci_lo,
    'boot_ci_hi': ci_hi,
    'boot_frac_positive': frac_positive,
    'market_wins': wins,
    'n_markets': len(rs.columns),
    'pooled_T': T,
    'effect_size_d': d,
    't_stat': t_stat,
    'p_t_onesided': p_t_onesided,
    'months_for_80pct_power': n_needed,
    'block': BLOCK,
    'replications': B,
    'seed': SEED,
}])
out.to_csv('scripts/concern_experiments/expDH_pooled_bootstrap.csv', index=False)

print(f"Observed annualized SR diff (Ridge_Slow - MomSort, pooled): {obs_diff:.3f}")
print(f"Block bootstrap one-sided p: {p_onesided:.3f}")
print(f"95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]")
print(f"Fraction of bootstrap samples positive: {frac_positive:.2f}")
print(f"Per-market wins: {wins}/{len(rs.columns)}")
print(f"Pooled T={T}, d={d:.3f}, t={t_stat:.2f} (one-sided p={p_t_onesided:.2f})")
print(f"Months for 80% power at one-sided 5%: {n_needed:.0f} (~{n_needed/12:.0f} years)")
