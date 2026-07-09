#!/usr/bin/env python3
"""
expDI -- Frontier diagnostics: separating cross-sectional content from
mechanical kappa-scaling in the cost-speed frontier.
========================================================================

Referee-driven decomposition of the pooled fit
    SR_gross - SR_net ~ lambda * kappa * FSI + c   (R^2 = 0.72, n = 300).

Because costs are c = kappa * HalfSpread and no strategy adapts its trades
to kappa, degradation is mechanically proportional to kappa for a fixed
strategy. The informative content of the frontier is therefore
CROSS-SECTIONAL: does FSI rank strategy-market cells by degradation at a
fixed cost level? This script reports:

  1. Per-kappa cross-sectional R^2 and slope of degradation on FSI
     (slope should be ~lambda*kappa under the linear cost model).
  2. Pooled R^2 of degradation on kappa alone (how much the pooled fit
     owes to cost scaling rather than feature speed).
  3. R^2 of realized turnover on FSI at the strategy-market level
     (FSI as an ex-ante turnover proxy for smooth models).
  4. Partial correlation of FSI with degradation controlling for
     realized turnover (what FSI adds beyond ex-post turnover).

Input:  paper_icaifv2/public/results/cost_speed_bound_points.csv
Output: scripts/concern_experiments/expDI_frontier_diagnostics.csv
"""

import os

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

bp = pd.read_csv('paper_icaifv2/public/results/cost_speed_bound_points.csv')


def fit_r2(X, y):
    reg = LinearRegression().fit(X, y)
    r2 = 1 - ((y - reg.predict(X)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return reg, float(r2)


rows = []

# 1. Per-kappa cross-sectional fits: degradation ~ FSI
for kappa in sorted(bp.kappa.unique()):
    sub = bp[bp.kappa == kappa]
    reg, r2 = fit_r2(sub[['fsi']], sub.degradation)
    rows.append({
        'diagnostic': f'cross_sectional_kappa_{kappa:g}',
        'value': r2,
        'slope': float(reg.coef_[0]),
        'slope_over_kappa': float(reg.coef_[0]) / kappa,
        'n': len(sub),
    })
    print(f'kappa={kappa:g}: cross-sectional R2 = {r2:.3f}, '
          f'slope = {reg.coef_[0]:.3f} (slope/kappa = {reg.coef_[0]/kappa:.3f})')

# 2. Pooled decomposition
_, r2_kappa_only = fit_r2(bp[['kappa']], bp.degradation)
_, r2_pooled = fit_r2(bp[['kappa_fsi']], bp.degradation)
rows.append({'diagnostic': 'pooled_kappa_only_r2', 'value': r2_kappa_only, 'n': len(bp)})
rows.append({'diagnostic': 'pooled_kappa_fsi_r2', 'value': r2_pooled, 'n': len(bp)})
print(f'\npooled R2 (kappa alone):   {r2_kappa_only:.3f}')
print(f'pooled R2 (kappa x FSI):   {r2_pooled:.3f}')

# 3. FSI as ex-ante turnover proxy (turnover is kappa-invariant; use kappa=1)
cells = bp[bp.kappa == 1.0]
_, r2_turnover = fit_r2(cells[['fsi']], cells.turnover)
rows.append({'diagnostic': 'turnover_on_fsi_r2', 'value': r2_turnover, 'n': len(cells)})
print(f'\nR2(turnover ~ FSI), strategy-market cells: {r2_turnover:.3f}')

# 4. Partial correlation of FSI with degradation given realized turnover
for kappa in [1.0, 3.0]:
    sub = bp[bp.kappa == kappa]
    reg_f, _ = fit_r2(sub[['turnover']], sub.fsi)
    fsi_resid = sub.fsi - reg_f.predict(sub[['turnover']])
    reg_d, _ = fit_r2(sub[['turnover']], sub.degradation)
    deg_resid = sub.degradation - reg_d.predict(sub[['turnover']])
    pcorr = float(np.corrcoef(fsi_resid, deg_resid)[0, 1])
    rows.append({'diagnostic': f'partial_corr_fsi_given_turnover_kappa_{kappa:g}',
                 'value': pcorr, 'n': len(sub)})
    print(f'kappa={kappa:g}: partial corr(FSI, degradation | turnover) = {pcorr:.3f}')

# 5. Between- vs within-strategy resolution at kappa = 3.
#    FSI is a feature-set property (nearly constant across markets within a
#    strategy), so the cross-sectional fit separates STRATEGIES; the residual
#    within-strategy variation across markets carries little signal.
k3 = bp[bp.kappa == 3.0].copy()
k3['deg_within'] = k3.degradation - k3.groupby('strategy').degradation.transform('mean')
k3['fsi_within'] = k3.fsi - k3.groupby('strategy').fsi.transform('mean')
_, r2_within = fit_r2(k3[['fsi_within']], k3.deg_within)
rows.append({'diagnostic': 'within_strategy_r2_kappa_3', 'value': r2_within, 'n': len(k3)})
print(f'\nwithin-strategy (demeaned) R2 at kappa=3: {r2_within:.3f}')

# 6. FSI vs market cost level: control for each market's per-trade cost at
#    kappa = 1 (MomentumSort Avg_Cost_bps, a clipped half-spread proxy).
ah = pd.read_csv('scripts/concern_experiments/expAH_feature_speed_test.csv')
spread = ah[(ah.Strategy == 'MomentumSort') & (ah.CostMult == 1.0)
            & (ah.Period == 'OOS')].set_index('Market').Avg_Cost_bps
k3['spread'] = k3.market.map(spread)
reg_c, r2_ctrl = fit_r2(k3[['fsi', 'spread']], k3.degradation)
rows.append({'diagnostic': 'fsi_coef_spread_controlled_kappa_3',
             'value': float(reg_c.coef_[0]), 'n': len(k3)})
rows.append({'diagnostic': 'r2_fsi_plus_spread_kappa_3', 'value': r2_ctrl, 'n': len(k3)})
print(f'kappa=3 with spread control: FSI coef = {reg_c.coef_[0]:.3f} '
      f'(spread coef {reg_c.coef_[1]:.5f}), R2 = {r2_ctrl:.3f}')

# 7. Pooled two-channel regression: add tree models (expAX, same OOS window)
#    with a non-smoothness indicator. Makes the "second mechanism" claim a
#    joint specification rather than a residual description.
ax = pd.read_csv('scripts/concern_experiments/expAX_full_comparison.csv')
trees = ax[(ax.Strategy.isin(['XGBoost', 'LightGBM'])) & (ax.CostMult == 3.0)
           & (ax.Period == 'OOS')].copy()
fsi_all = k3[k3.strategy == 'Ridge_All'].set_index('market').fsi
trees['fsi'] = trees.Market.map(fsi_all)
trees['degradation'] = trees.Gross_SR - trees.Net_SR
pool = pd.concat([
    k3[['fsi', 'degradation']].assign(nonsmooth=0),
    trees[['fsi', 'degradation']].assign(nonsmooth=1),
], ignore_index=True)
_, r2_fsi_only = fit_r2(pool[['fsi']], pool.degradation)
reg_2ch, r2_2ch = fit_r2(pool[['fsi', 'nonsmooth']], pool.degradation)
rows.append({'diagnostic': 'pooled_2ch_fsi_only_r2', 'value': r2_fsi_only, 'n': len(pool)})
rows.append({'diagnostic': 'pooled_2ch_full_r2', 'value': r2_2ch, 'n': len(pool)})
rows.append({'diagnostic': 'pooled_2ch_nonsmooth_coef',
             'value': float(reg_2ch.coef_[1]), 'n': len(pool)})
rows.append({'diagnostic': 'pooled_2ch_fsi_coef',
             'value': float(reg_2ch.coef_[0]), 'n': len(pool)})
print(f'two-channel pooled (7 strategies x 15 markets, kappa=3): '
      f'FSI-only R2 = {r2_fsi_only:.3f}, +nonsmooth R2 = {r2_2ch:.3f}, '
      f'nonsmooth coef = {reg_2ch.coef_[1]:.3f}, FSI coef = {reg_2ch.coef_[0]:.3f}')

# 8. Cluster-robust (Liang-Zeger) 95% CIs for the kappa=3 slope, per unit kappa.
from scipy import stats as scipy_stats

X = np.column_stack([np.ones(len(k3)), k3.fsi.values])
y = k3.degradation.values
beta = np.linalg.solve(X.T @ X, X.T @ y)
resid = y - X @ beta
XtXi = np.linalg.inv(X.T @ X)
for cluster_name in ['market', 'strategy']:
    clusters = k3[cluster_name].values
    meat = np.zeros((2, 2))
    for c in np.unique(clusters):
        Xg = X[clusters == c]
        eg = resid[clusters == c]
        meat += np.outer(Xg.T @ eg, Xg.T @ eg)
    G, (n, k) = len(np.unique(clusters)), X.shape
    V = XtXi @ meat @ XtXi * (G / (G - 1)) * ((n - 1) / (n - k))
    tcrit = scipy_stats.t.ppf(0.975, G - 1)
    lo, hi = (beta[1] - tcrit * np.sqrt(V[1, 1])) / 3, (beta[1] + tcrit * np.sqrt(V[1, 1])) / 3
    rows.append({'diagnostic': f'lambda_ci_lo_cluster_{cluster_name}', 'value': lo, 'n': G})
    rows.append({'diagnostic': f'lambda_ci_hi_cluster_{cluster_name}', 'value': hi, 'n': G})
    print(f'lambda per unit kappa, cluster-{cluster_name} 95% CI: [{lo:.3f}, {hi:.3f}]')

out = pd.DataFrame(rows)
out.to_csv('scripts/concern_experiments/expDI_frontier_diagnostics.csv', index=False)
print('\nSaved expDI_frontier_diagnostics.csv')
