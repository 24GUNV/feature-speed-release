#!/usr/bin/env python3
"""
expDK -- Small-universe robustness: exclude GRC (N = 50)
=========================================================

GRC is the smallest universe in the sample (top 50 by market cap vs 100-300
elsewhere), so quintile portfolios hold only ~10 names and single stocks can
move the market's SR. This script recomputes the paper's headline means at
3x costs, OOS, over the remaining 14 markets to check that no ranking or
headline number depends on GRC.

Inputs: canonical per-market CSVs (expAX, expAH, expAR, expAN, expCL).
Output: scripts/concern_experiments/expDK_grc_exclusion.csv
"""

import os

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
import pandas as pd

COST = 3.0
BASE = 'scripts/concern_experiments'

# (label, file, strategy)
STRATEGIES = [
    ('MomentumSort', 'expAX_full_comparison.csv', 'MomentumSort'),
    ('CompositeSort', 'expAX_full_comparison.csv', 'CompositeSort'),
    ('Ridge_All', 'expAX_full_comparison.csv', 'Ridge'),
    ('XGBoost', 'expAX_full_comparison.csv', 'XGBoost'),
    ('LightGBM', 'expAX_full_comparison.csv', 'LightGBM'),
    ('Blend', 'expAX_full_comparison.csv', 'MomentumML_Blend'),
    ('Ridge_Slow', 'expAH_feature_speed_test.csv', 'Ridge_Slow'),
    ('Ridge_Fast', 'expAH_feature_speed_test.csv', 'Ridge_Fast'),
    ('Ridge_All_Q', 'expAH_feature_speed_test.csv', 'Ridge_All_Q'),
    ('Ridge_Slow_Q', 'expAH_feature_speed_test.csv', 'Ridge_Slow_Q'),
    ('PersistenceFilter', 'expAN_persistence_filter.csv', 'Ridge_PersFiltered'),
    ('SpeedTercile', 'expAN_persistence_filter.csv', 'Ridge_EmpSlow_FS'),
    ('RT_Slow', 'expCL_ranktarget_feature_speed.csv', 'RT_Slow'),
    ('EMA_0.35', 'expAR_ema_robustness.csv', 'Ridge_EMA_0.35'),
    ('WF_EMA', 'expAR_ema_robustness.csv', 'Ridge_WF_EMA'),
]


def per_market(path, strategy):
    df = pd.read_csv(f'{BASE}/{path}')
    df = df[(df.Strategy == strategy) & (np.isclose(df.CostMult, COST))]
    if 'Period' in df.columns:
        df = df[df.Period == 'OOS']
    return df.set_index('Market')[['Net_SR', 'Turnover']]


rows = []
for label, path, strategy in STRATEGIES:
    pm = per_market(path, strategy)
    if 'GRC' not in pm.index:
        raise ValueError(f'GRC missing for {label}')
    ex = pm.drop('GRC')
    rows.append({
        'strategy': label,
        'net_sr_15mkt': pm.Net_SR.mean(),
        'net_sr_ex_grc': ex.Net_SR.mean(),
        'delta': ex.Net_SR.mean() - pm.Net_SR.mean(),
        'turnover_ex_grc': ex.Turnover.mean(),
        'grc_net_sr': pm.loc['GRC', 'Net_SR'],
        'n_markets_ex': len(ex),
    })
    print(f'{label:>18}: 15-mkt {pm.Net_SR.mean():.3f} -> ex-GRC '
          f'{ex.Net_SR.mean():.3f} (delta {ex.Net_SR.mean() - pm.Net_SR.mean():+.3f})')

out = pd.DataFrame(rows)

slow = out.set_index('strategy').loc['Ridge_Slow', 'net_sr_ex_grc']
mom = out.set_index('strategy').loc['MomentumSort', 'net_sr_ex_grc']
frontier = out[out.strategy.isin(
    ['Ridge_Slow', 'PersistenceFilter', 'SpeedTercile', 'RT_Slow',
     'EMA_0.35', 'WF_EMA', 'Ridge_All_Q', 'Ridge_Slow_Q'])]
print(f'\nEx-GRC: Ridge_Slow {slow:.3f} vs MomSort {mom:.3f} '
      f'(gap {slow - mom:+.3f})')
print(f'Ex-GRC frontier band: {frontier.net_sr_ex_grc.min():.3f}'
      f'-{frontier.net_sr_ex_grc.max():.3f}')

out.to_csv(f'{BASE}/expDK_grc_exclusion.csv', index=False)
print('\nSaved expDK_grc_exclusion.csv')
