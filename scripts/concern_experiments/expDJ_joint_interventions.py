#!/usr/bin/env python3
"""
expDJ -- Joint significance of the eight frontier interventions
================================================================

The paper's statistical-significance paragraph reports a single-comparison
test (Ridge_Slow vs MomentumSort, pooled block bootstrap p = 0.10). The
frontier claim, however, is that EIGHT independent low-FSI interventions
all reach the same ceiling above the sort. This script tests that claim
jointly.

Method (market-level randomization inference):
  - For each intervention s and market m, compute the paired difference
    d[s, m] = NetSR(s, m) - NetSR(MomentumSort, m) at 3x costs, OOS,
    using the MomentumSort control from the SAME experiment file.
  - Collapse to one value per market: D[m] = mean over the 8 interventions
    (the market is the independent unit; interventions within a market are
    highly correlated and are never treated as independent draws).
  - Exact sign-flip test: under H0 (no improvement, symmetric differences),
    flip the sign of D[m] market-by-market; all 2^15 = 32,768 patterns are
    enumerated. One-sided p = P(mean(D_flipped) >= mean(D_observed)).
    Flipping at the market level preserves the within-market dependence
    across interventions exactly.
  - The same exact test is reported per intervention for comparison, plus
    a Wilcoxon signed-rank test on the 15 market-level means.

Inputs (canonical per-market CSVs):
  expAH_feature_speed_test.csv   Ridge_Slow, Ridge_All_Q, Ridge_Slow_Q (+ control)
  expAN_persistence_filter.csv   Ridge_PersFiltered, Ridge_EmpSlow_FS (+ control)
  expCL_ranktarget_feature_speed.csv        RT_Slow (+ control 'MomSort')
  expAR_ema_robustness.csv       Ridge_EMA_0.35, Ridge_WF_EMA (+ control)

Output: scripts/concern_experiments/expDJ_joint_interventions.csv
"""

import os

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
import pandas as pd
from scipy import stats

COST = 3.0
BASE = 'scripts/concern_experiments'


def per_market_sr(path, strategy, period_col='Period', period='OOS'):
    df = pd.read_csv(f'{BASE}/{path}')
    df = df[(df.Strategy == strategy) & (np.isclose(df.CostMult, COST))]
    if period_col in df.columns:
        df = df[df[period_col] == period]
    out = df.set_index('Market')['Net_SR']
    if out.index.duplicated().any():
        raise ValueError(f'duplicate markets for {strategy} in {path}')
    return out


# (label, file, strategy, control file, control strategy)
INTERVENTIONS = [
    ('Ridge_Slow',        'expAH_feature_speed_test.csv', 'Ridge_Slow',
     'expAH_feature_speed_test.csv', 'MomentumSort'),
    ('PersistenceFilter', 'expAN_persistence_filter.csv', 'Ridge_PersFiltered',
     'expAN_persistence_filter.csv', 'MomentumSort'),
    ('SpeedTercile',      'expAN_persistence_filter.csv', 'Ridge_EmpSlow_FS',
     'expAN_persistence_filter.csv', 'MomentumSort'),
    ('RT_Slow',           'expCL_ranktarget_feature_speed.csv', 'RT_Slow',
     'expCL_ranktarget_feature_speed.csv', 'MomSort'),
    ('EMA_0.35',          'expAR_ema_robustness.csv', 'Ridge_EMA_0.35',
     'expAR_ema_robustness.csv', 'MomentumSort'),
    ('WF_EMA',            'expAR_ema_robustness.csv', 'Ridge_WF_EMA',
     'expAR_ema_robustness.csv', 'MomentumSort'),
    ('Ridge_All_Q',       'expAH_feature_speed_test.csv', 'Ridge_All_Q',
     'expAH_feature_speed_test.csv', 'MomentumSort'),
    ('Ridge_Slow_Q',      'expAH_feature_speed_test.csv', 'Ridge_Slow_Q',
     'expAH_feature_speed_test.csv', 'MomentumSort'),
]


def exact_signflip_p(values):
    """One-sided exact sign-flip p-value for mean(values) > 0."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    signs = np.array([[1 if (pattern >> bit) & 1 else -1 for bit in range(n)]
                      for pattern in range(2 ** n)])
    t_null = signs @ values / n
    t_obs = values.mean()
    return float((t_null >= t_obs - 1e-12).mean())


diffs = {}
markets = None
for label, path, strategy, cpath, cstrategy in INTERVENTIONS:
    sr = per_market_sr(path, strategy)
    control = per_market_sr(cpath, cstrategy)
    common = sr.index.intersection(control.index)
    if markets is None:
        markets = sorted(common)
    if sorted(common) != markets:
        raise ValueError(f'market mismatch for {label}: {sorted(common)}')
    diffs[label] = (sr - control).loc[markets]
    print(f'{label:>18}: mean SR {sr.loc[markets].mean():.3f}, '
          f'mean diff vs sort {diffs[label].mean():+.3f}, '
          f'wins {(diffs[label] > 0).sum()}/{len(markets)}')

panel = pd.DataFrame(diffs)          # 15 markets x 8 interventions
market_mean = panel.mean(axis=1)     # one value per market (independent unit)

rows = []
for label in panel.columns:
    rows.append({
        'test': label,
        'mean_sr_diff': panel[label].mean(),
        'market_wins': int((panel[label] > 0).sum()),
        'n_markets': len(markets),
        'signflip_p_onesided': exact_signflip_p(panel[label].values),
    })

joint_p = exact_signflip_p(market_mean.values)
wilcoxon_p = float(stats.wilcoxon(market_mean.values, alternative='greater').pvalue)
rows.append({
    'test': 'JOINT_all_eight',
    'mean_sr_diff': market_mean.mean(),
    'market_wins': int((market_mean > 0).sum()),
    'n_markets': len(markets),
    'signflip_p_onesided': joint_p,
    'wilcoxon_p_onesided': wilcoxon_p,
})

print(f'\nJoint (market-level mean over 8 interventions):')
print(f'  mean SR diff = {market_mean.mean():+.3f}, '
      f'markets positive = {(market_mean > 0).sum()}/{len(markets)}')
print(f'  exact sign-flip p (one-sided) = {joint_p:.4f}')
print(f'  Wilcoxon signed-rank p (one-sided) = {wilcoxon_p:.4f}')

out = pd.DataFrame(rows)
out.to_csv(f'{BASE}/expDJ_joint_interventions.csv', index=False)
print('\nSaved expDJ_joint_interventions.csv')
