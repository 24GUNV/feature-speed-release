"""
Experiment CM: US vs EM Cost-Regime Interaction Hypothesis

Tests the hypothesis that ML's failure in EM is COST-REGIME DEPENDENT, not universal.

Approach:
  1. Run Ridge_All, Ridge_Slow, Ridge_Fast, and MomentumSort on US equities (GFD format)
     using the SAME pipeline as the EM experiments (expAH).
  2. Sweep cost multipliers from 0x to 3x in fine increments.
  3. Show that at low costs (US-realistic), Ridge beats MomentumSort;
     at high costs (EM-equivalent), Ridge loses — proving cost-regime interaction.
  4. Decompose feature importance by speed class (SLOW/MEDIUM/FAST) and compare
     the turnover and feature-speed profile to EM averages.

Key predictions:
  - At 0x costs: Ridge_All should beat MomentumSort (better gross signal).
  - At ~1x costs: Ridge still competitive.
  - At 2-3x costs: Ridge loses due to higher turnover.
  - Ridge_Fast turnover > Ridge_Slow turnover in US too, but the GAP matters less
    at low costs.
  - The "breakeven cost multiplier" where Ridge = MomSort provides a clean metric.

Uses USEquityGFD.csv (same GFD feature set as EM), enabling direct comparison.
Pipeline: dual-lag, buffer, walk-forward, CS Corwin-Schultz costs.
OOS: 2021-01 to 2025-02.

Output: scripts/concern_experiments/expCM_us_comparison.csv
"""

import sys
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, 'public/src')
sys.path.insert(0, 'scripts/concern_experiments')

import numpy as np
import pandas as pd
import warnings
import time
import sys
warnings.filterwarnings('ignore')

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy import stats

from data_loader import load_equity_data

np.random.seed(42)


def lag_features_fast(df, features, id_col, date_col, lag=1):
    """Optimized lag: avoid expensive groupby on large datasets."""
    df = df.sort_values(by=[id_col, date_col]).reset_index(drop=True)
    # Use shift within groups but only on available features
    avail = [f for f in features if f in df.columns]
    print(f"  Lagging {len(avail)} features by {lag} period(s)...")
    df[avail] = df.groupby(id_col)[avail].shift(lag)
    # Drop rows where first few features are all NaN (first obs per entity)
    check_cols = avail[:min(5, len(avail))]
    initial = len(df)
    df = df.dropna(subset=check_cols, how='all')
    print(f"  Removed {initial - len(df):,} rows, final: {len(df):,}")
    return df


# ==============================================================================
# FEATURE SPEED CLASSIFICATION (identical to expAH)
# ==============================================================================

SLOW_KEYWORDS = [
    'f_score', 'o_score', 'z_score',
    'qmj', 'qmj_prof', 'qmj_safety', 'qmj_growth',
    'roa', 'roe', 'gp_at', 'gp_atl1', 'op_at', 'op_atl1',
    'niq_at', 'niq_be', 'ni_be', 'ni_me',
    'cop_at', 'cop_atl1',
    'ope_be', 'ope_bel1',
    'ocf_at', 'ocf_me',
    'oaccruals_at', 'oaccruals_ni',
    'taccruals_at', 'taccruals_ni',
    'be_me', 'bev_mev', 'at_me', 'ebit_bev', 'ebitda_mev',
    'fcf_me', 'sale_me', 'debt_me', 'netdebt_me',
    'div12m_me', 'ebit_sale', 'cash_at', 'tangibility',
    'noa_at', 'opex_at', 'rd5_at', 'rd_sale', 'rd_me',
    'sale_bev',
    'mispricing_mgmt', 'mispricing_perf',
    'eq_dur', 'age',
    'at_be', 'at_turnover',
    'pi_nix', 'kz_index', 'aliq_at', 'aliq_mat',
    'eqpo_me', 'eqnpo_me', 'ival_me',
]

MEDIUM_KEYWORDS = [
    'at_gr1', 'ppeinv_gr1a', 'noa_gr1a', 'lnoa_gr1a',
    'capx_gr1', 'capx_gr2', 'capx_gr3', 'capex_abn',
    'chcsho_12m', 'eqnpo_12m', 'debt_gr3',
    'inv_gr1', 'inv_gr1a',
    'cowc_gr1a', 'coa_gr1a', 'col_gr1a', 'nncoa_gr1a',
    'ncoa_gr1a', 'ncol_gr1a', 'nfna_gr1a',
    'sti_gr1a', 'lti_gr1a', 'fnl_gr1a', 'be_gr1a',
    'netis_at', 'eqnetis_at', 'dbnetis_at',
    'tax_gr1a',
    'niq_at_chg1', 'niq_be_chg1', 'ocf_at_chg1',
    'ni_ar1', 'ni_inc8q',
    'dsale_dinv', 'dsale_drec', 'dgp_dsale', 'dsale_dsga',
    'sale_gr1', 'sale_gr3', 'emp_gr1', 'sale_emp_gr1',
    'saleq_gr1',
    'niq_su', 'saleq_su',
    'ocfq_saleq_std', 'earnings_variability',
    'seas_1_1an', 'seas_1_1na',
    'seas_2_5an', 'seas_2_5na',
    'seas_6_10an', 'seas_6_10na',
    'seas_11_15an', 'seas_11_15na',
    'seas_16_20an', 'seas_16_20na',
]

FAST_KEYWORDS = [
    'ret_12_1', 'ret_12_7', 'ret_6_1', 'ret_9_1', 'ret_3_1', 'ret_60_12',
    'resff3_12_1', 'resff3_6_1',
    'ivol_capm_21d', 'ivol_capm_252d', 'ivol_ff3_21d', 'ivol_hxz4_21d',
    'rvol_21d',
    'beta_60m', 'beta_dimson_21d', 'betabab_1260d', 'betadown_252d',
    'ni_ivol',
    'iskew_capm_21d', 'iskew_ff3_21d', 'iskew_hxz4_21d',
    'rskew_21d', 'coskew_21d', 'corr_1260d',
    'rmax1_21d', 'rmax5_21d', 'rmax5_rvol_21d',
    'ami_126d',
    'bidaskhl_21d',
    'dolvol_126d', 'dolvol_var_126d',
    'turnover_126d', 'turnover_var_126d',
    'zero_trades_126d', 'zero_trades_21d', 'zero_trades_252d',
    'prc', 'prc_highprc_252d',
]

EXCLUDE_COLS = ['Unnamed: 0', 'id', 'eom', 'excntry', 'gvkey', 'permno',
                'size_grp', 'me', 'market_equity', 'ret_1_0', 'date']


def classify_feature_speed(feature_name):
    """Classify a single feature into SLOW/MEDIUM/FAST."""
    f = feature_name.lower()
    for kw in FAST_KEYWORDS:
        if f == kw or f == kw.lower():
            return 'FAST'
    for kw in MEDIUM_KEYWORDS:
        if f == kw or f == kw.lower():
            return 'MEDIUM'
    for kw in SLOW_KEYWORDS:
        if f == kw or f == kw.lower():
            return 'SLOW'
    if any(x in f for x in ['ret_', 'vol', 'beta', 'ivol', 'rvol', 'skew', 'ami',
                              'bidask', 'dolvol', 'turnover', 'zero_trades',
                              'prc', 'rmax', 'corr_', 'coskew']):
        return 'FAST'
    if any(x in f for x in ['_gr1', '_gr2', '_gr3', '_chg', 'seas_', '_su',
                              'variability', 'netis', 'eqnetis', 'dbnetis']):
        return 'MEDIUM'
    return 'SLOW'


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# US market config: use GFD data (same format as EM)
US_CONFIG = {
    'file': 'data/USEquityGFD.csv',
    'type': 'emerging_equity',  # same format as EM: eom/gvkey/ret_1_0
    'target': 'ret_1_0',
    'topn': 500,  # top 500 by market cap (US has ~3000+ stocks/month)
    'nq': 5,
}

# Fine cost multiplier grid: 0x to 3x in 0.5x increments, plus EM-standard levels
COST_MULTS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
BUFFER_PCT = 0.20
MIN_TRAIN_MONTHS = 60

# OOS boundary: same as EM experiments
OOS_CUTOFF = pd.Timestamp('2020-12-31')


# ==============================================================================
# HELPER FUNCTIONS (identical to expAH)
# ==============================================================================

def sharpe(s):
    s = s.dropna()
    if len(s) < 2 or s.std() == 0:
        return 0.0
    return (s.mean() / s.std()) * np.sqrt(12)


def build_cost_lookup_1x(raw_df, date_col, id_col, spread_col='bidaskhl_21d',
                         floor_bps=20, cap_bps=200):
    """Build 1x cost lookup from Corwin-Schultz half-spread."""
    if spread_col not in raw_df.columns:
        lookup = raw_df[[date_col, id_col]].copy()
        lookup['cost_bps_1x'] = 50.0
        return lookup
    lookup = raw_df[[date_col, id_col, spread_col]].copy()
    lookup['cost_bps_1x'] = (lookup[spread_col] / 2) * 10000
    monthly_median = lookup.groupby(date_col)['cost_bps_1x'].transform('median')
    lookup['cost_bps_1x'] = lookup['cost_bps_1x'].fillna(monthly_median)
    lookup['cost_bps_1x'] = lookup['cost_bps_1x'].clip(lower=floor_bps, upper=cap_bps)
    return lookup[[date_col, id_col, 'cost_bps_1x']]


def compute_cost_drag(month_data, entries, exits, new_long, id_col,
                      cost_mult, global_med_1x, prev_long):
    """Compute cost drag for a given cost multiplier."""
    if cost_mult == 0.0:
        return 0.0
    if not prev_long or not (entries | exits):
        return 0.0
    traded = entries | exits
    if len(traded) == 0:
        return 0.0
    traded_costs_1x = month_data[month_data[id_col].isin(traded)]['cost_bps_1x'].mean()
    if pd.isna(traded_costs_1x):
        traded_costs_1x = global_med_1x
    traded_costs = np.clip(traded_costs_1x * cost_mult, 20.0, 200.0)
    frac_traded = len(traded) / max(len(new_long), 1)
    cost_drag = frac_traded * (traded_costs / 10000)
    return cost_drag


# ==============================================================================
# FACTOR SORT
# ==============================================================================

def run_factor_sort(df, sort_features, date_col, id_col, target,
                    cost_lookup_1x, nq=5, buffer_pct=0.20,
                    ascending_features=None):
    """Factor sort with buffer."""
    if ascending_features is None:
        ascending_features = []

    data = df.copy()
    for feat in sort_features:
        if feat not in data.columns:
            continue
        data[f'{feat}_rank'] = data.groupby(date_col)[feat].rank(pct=True)
        data[f'{feat}_rank'] = data[f'{feat}_rank'].fillna(0.5)
        if feat in ascending_features:
            data[f'{feat}_rank'] = 1.0 - data[f'{feat}_rank']

    rank_cols = [f'{f}_rank' for f in sort_features if f in data.columns]
    if len(rank_cols) == 0:
        return None
    data['composite_score'] = data[rank_cols].mean(axis=1)
    data['pct_rank'] = data.groupby(date_col)['composite_score'].rank(pct=True)

    top_threshold = 1.0 - 1.0 / nq
    buffer = buffer_pct * (1.0 / nq)

    data = data.merge(cost_lookup_1x, on=[date_col, id_col], how='left')
    global_med_1x = cost_lookup_1x['cost_bps_1x'].median()
    data['cost_bps_1x'] = data['cost_bps_1x'].fillna(global_med_1x)

    dates = sorted(data[date_col].unique())
    prev_long = set()
    records = []

    for i, date in enumerate(dates):
        month = data[data[date_col] == date].copy()
        if len(month) < nq:
            continue

        ids = month[id_col].values
        ranks = month['pct_rank'].values
        prev_ids = np.array(list(prev_long), dtype=ids.dtype) if prev_long else np.array([], dtype=ids.dtype)
        held_mask = np.isin(ids, prev_ids)
        select_mask = np.where(held_mask,
                               ranks >= top_threshold - buffer,
                               ranks >= top_threshold + buffer)
        new_long = set(ids[select_mask])
        if len(new_long) == 0:
            new_long = set(month.nlargest(max(1, len(month) // nq), 'pct_rank')[id_col])

        held = month[month[id_col].isin(new_long)]
        gross_ret = held[target].mean() if len(held) > 0 else 0.0

        if prev_long:
            entries = new_long - prev_long
            exits = prev_long - new_long
            turn = (len(entries) + len(exits)) / (2 * max(len(new_long), 1))
        else:
            entries = new_long
            exits = set()
            turn = np.nan

        cost_drags = {}
        for cm in COST_MULTS:
            cost_drags[cm] = compute_cost_drag(
                month, entries, exits, new_long, id_col,
                cm, global_med_1x, prev_long
            )

        records.append({
            'date': date, 'gross_ret': gross_ret, 'turnover': turn,
            'n_held': len(new_long),
            **{f'cost_drag_{cm}': cost_drags[cm] for cm in COST_MULTS},
        })
        prev_long = new_long

    if len(records) == 0:
        return None
    return pd.DataFrame(records).set_index('date')


# ==============================================================================
# ML WALK-FORWARD
# ==============================================================================

def run_ml_walkforward(df, features, date_col, id_col, target,
                       cost_lookup_1x, nq=5, buffer_pct=0.20,
                       min_train=60):
    """Walk-forward Ridge with rank targets and buffer."""
    data = df.copy()
    data = data.sort_values([date_col, id_col]).reset_index(drop=True)

    # Pre-compute rank target (cross-sectional percentile ranks)
    data['_rank_target'] = data.groupby(date_col)[target].rank(pct=True)

    unique_dates = sorted(data[date_col].unique())
    n_dates = len(unique_dates)

    if n_dates < min_train + 12:
        return None

    data = data.merge(cost_lookup_1x, on=[date_col, id_col], how='left')
    global_med_1x = cost_lookup_1x['cost_bps_1x'].median()
    data['cost_bps_1x'] = data['cost_bps_1x'].fillna(global_med_1x)

    avail_features = [f for f in features if f in data.columns]
    feat_coverage = data[avail_features].notna().mean()
    avail_features = feat_coverage[feat_coverage > 0.1].index.tolist()

    if len(avail_features) == 0:
        return None

    top_threshold = 1.0 - 1.0 / nq
    buffer = buffer_pct * (1.0 / nq)

    prev_long = set()
    records = []

    for t_idx in range(min_train, n_dates):
        train_dates = unique_dates[:t_idx]
        test_date = unique_dates[t_idx]

        train_data = data[data[date_col].isin(train_dates)]
        test_data = data[data[date_col] == test_date].copy()

        if len(test_data) < nq:
            continue

        if len(train_data) < 100:
            continue

        X_train = train_data[avail_features].copy()
        y_train = train_data['_rank_target'].values
        X_test = test_data[avail_features].copy()

        imputer = SimpleImputer(strategy='median')
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        mask = ~np.isnan(y_train)
        if mask.sum() < 50:
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp[mask])
        X_test_scaled = scaler.transform(X_test_imp)

        model = Ridge(alpha=100, random_state=42)
        model.fit(X_train_scaled, y_train[mask])
        preds = model.predict(X_test_scaled)

        test_data = test_data.copy()
        test_data['prediction'] = preds
        test_data['pct_rank'] = test_data['prediction'].rank(pct=True)

        ids = test_data[id_col].values
        ranks = test_data['pct_rank'].values
        prev_ids = np.array(list(prev_long), dtype=ids.dtype) if prev_long else np.array([], dtype=ids.dtype)
        held_mask = np.isin(ids, prev_ids)
        select_mask = np.where(held_mask,
                               ranks >= top_threshold - buffer,
                               ranks >= top_threshold + buffer)
        new_long = set(ids[select_mask])
        if len(new_long) == 0:
            new_long = set(test_data.nlargest(max(1, len(test_data) // nq), 'pct_rank')[id_col])

        held = test_data[test_data[id_col].isin(new_long)]
        gross_ret = held[target].mean() if len(held) > 0 else 0.0

        if prev_long:
            entries = new_long - prev_long
            exits = prev_long - new_long
            turn = (len(entries) + len(exits)) / (2 * max(len(new_long), 1))
        else:
            entries = new_long
            exits = set()
            turn = np.nan

        cost_drags = {}
        for cm in COST_MULTS:
            cost_drags[cm] = compute_cost_drag(
                test_data, entries, exits, new_long, id_col,
                cm, global_med_1x, prev_long
            )

        records.append({
            'date': test_date, 'gross_ret': gross_ret, 'turnover': turn,
            'n_held': len(new_long),
            **{f'cost_drag_{cm}': cost_drags[cm] for cm in COST_MULTS},
        })
        prev_long = new_long

    if len(records) == 0:
        return None
    return pd.DataFrame(records).set_index('date')


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate(result_df, cost_mult, period_label, period_mask,
             strategy_name, market='US'):
    if result_df is None:
        return None
    subset = result_df[period_mask]
    if len(subset) < 6:
        return None
    cost_col = f'cost_drag_{cost_mult}'
    net_ret = subset['gross_ret'] - subset[cost_col]
    gross_ret = subset['gross_ret']
    return {
        'Market': market,
        'Strategy': strategy_name,
        'CostMult': cost_mult,
        'Period': period_label,
        'Net_SR': round(sharpe(net_ret), 4),
        'Gross_SR': round(sharpe(gross_ret), 4),
        'Ann_Ret': round(net_ret.mean() * 12, 4),
        'Turnover': round(subset['turnover'].mean(), 4),
        'Avg_Cost_bps': round(subset[cost_col].mean() * 10000, 1),
        'N_months': len(subset),
    }


# ==============================================================================
# MAIN
# ==============================================================================

print("=" * 100)
print("EXPERIMENT CM: US vs EM Cost-Regime Interaction Hypothesis")
print("=" * 100)

all_results = []
t0 = time.time()

cfg = US_CONFIG

# Load US data using GFD format (same as EM markets)
# Start from 2005 to have enough training data (60+ months before OOS 2021)
# while keeping memory manageable (full dataset is 4.2M rows from 1925)
print(f"\n{'#'*80}")
print(f"# Loading US Equities (GFD format, 2005+)")
print(f"{'#'*80}")

df, raw_features, date_col, id_col = load_equity_data(
    cfg['file'], cfg['type'], start_date='2005-01-01'
)
target = cfg['target']

# Print data summary
print(f"\n  date_col={date_col}, id_col={id_col}, target={target}")
print(f"  Raw features: {len(raw_features)}")

# Identify modeling features
all_features = [f for f in raw_features if f not in EXCLUDE_COLS and f != target]
print(f"  Modeling features: {len(all_features)}")

# Classify features by speed
slow_feats = [f for f in all_features if classify_feature_speed(f) == 'SLOW']
medium_feats = [f for f in all_features if classify_feature_speed(f) == 'MEDIUM']
fast_feats = [f for f in all_features if classify_feature_speed(f) == 'FAST']
slow_med_feats = slow_feats + medium_feats

print(f"\n  FEATURE SPEED BREAKDOWN:")
print(f"    SLOW:   {len(slow_feats):3d} features ({100*len(slow_feats)/len(all_features):.0f}%)")
print(f"    MEDIUM: {len(medium_feats):3d} features ({100*len(medium_feats)/len(all_features):.0f}%)")
print(f"    FAST:   {len(fast_feats):3d} features ({100*len(fast_feats)/len(all_features):.0f}%)")
print(f"    Total:  {len(all_features):3d}")

# CRITICAL: Lag BEFORE filter (same as EM pipeline)
print(f"\n  Applying lag=1 to all features...")
df = lag_features_fast(df, all_features, id_col, date_col, lag=1)

# Lag me to avoid look-ahead bias
if 'me' in df.columns:
    df = df.sort_values([id_col, date_col])
    df['me'] = df.groupby(id_col)['me'].shift(1)
    df = df.dropna(subset=['me'])

# Filter to top-N by market cap
topn = cfg['topn']
if 'me' in df.columns:
    df['_me_rank'] = df.groupby(date_col)['me'].rank(ascending=False, method='first')
    df = df[df['_me_rank'] <= topn].drop(columns=['_me_rank'])
    print(f"  Filtered to top-{topn} by market cap: {len(df):,} rows")

# Build cost lookup
cost_lookup_1x = build_cost_lookup_1x(df, date_col, id_col)
print(f"  Median 1x cost: {cost_lookup_1x['cost_bps_1x'].median():.1f} bps")

# Filter features to those available in the data
ml_features = [f for f in all_features if f in df.columns]
slow_feats = [f for f in slow_feats if f in df.columns]
medium_feats = [f for f in medium_feats if f in df.columns]
fast_feats = [f for f in fast_feats if f in df.columns]
slow_med_feats = [f for f in slow_med_feats if f in df.columns]

df[date_col] = pd.to_datetime(df[date_col])
nq = cfg['nq']

print(f"\n  Available features after filtering:")
print(f"    ALL:      {len(ml_features)}")
print(f"    SLOW:     {len(slow_feats)}")
print(f"    MEDIUM:   {len(medium_feats)}")
print(f"    FAST:     {len(fast_feats)}")

# ==================================================================
# RUN STRATEGIES
# ==================================================================
strategies = {}

# 1. MomentumSort (12-1 momentum, monthly)
print(f"\n  --- MomentumSort (ret_12_1, monthly) ---")
strategies['MomentumSort'] = run_factor_sort(
    df, ['ret_12_1'], date_col, id_col, target, cost_lookup_1x,
    nq=nq, buffer_pct=BUFFER_PCT
)

# 2. QualitySort (QMJ composite, monthly)
quality_features = [f for f in ['qmj', 'qmj_prof', 'qmj_safety'] if f in df.columns]
if quality_features:
    print(f"  --- QualitySort ({quality_features}, monthly) ---")
    strategies['QualitySort'] = run_factor_sort(
        df, quality_features, date_col, id_col, target, cost_lookup_1x,
        nq=nq, buffer_pct=BUFFER_PCT
    )

# 3. Ridge_All (all features, monthly)
print(f"  --- Ridge_All ({len(ml_features)} features, monthly) ---")
strategies['Ridge_All'] = run_ml_walkforward(
    df, ml_features, date_col, id_col, target, cost_lookup_1x,
    nq=nq, buffer_pct=BUFFER_PCT, min_train=MIN_TRAIN_MONTHS
)

# 4. Ridge_Slow (SLOW features only, monthly)
print(f"  --- Ridge_Slow ({len(slow_feats)} features, monthly) ---")
if len(slow_feats) >= 5:
    strategies['Ridge_Slow'] = run_ml_walkforward(
        df, slow_feats, date_col, id_col, target, cost_lookup_1x,
        nq=nq, buffer_pct=BUFFER_PCT, min_train=MIN_TRAIN_MONTHS
    )
else:
    strategies['Ridge_Slow'] = None
    print(f"    Skipping: only {len(slow_feats)} SLOW features")

# 5. Ridge_SlowMed (SLOW + MEDIUM features, monthly)
print(f"  --- Ridge_SlowMed ({len(slow_med_feats)} features, monthly) ---")
if len(slow_med_feats) >= 5:
    strategies['Ridge_SlowMed'] = run_ml_walkforward(
        df, slow_med_feats, date_col, id_col, target, cost_lookup_1x,
        nq=nq, buffer_pct=BUFFER_PCT, min_train=MIN_TRAIN_MONTHS
    )
else:
    strategies['Ridge_SlowMed'] = None

# 6. Ridge_Fast (FAST features only, monthly)
print(f"  --- Ridge_Fast ({len(fast_feats)} features, monthly) ---")
if len(fast_feats) >= 5:
    strategies['Ridge_Fast'] = run_ml_walkforward(
        df, fast_feats, date_col, id_col, target, cost_lookup_1x,
        nq=nq, buffer_pct=BUFFER_PCT, min_train=MIN_TRAIN_MONTHS
    )
else:
    strategies['Ridge_Fast'] = None
    print(f"    Skipping: only {len(fast_feats)} FAST features")


# ==================================================================
# EVALUATE ALL STRATEGIES ACROSS ALL COST MULTIPLIERS
# ==================================================================
print(f"\n{'#'*80}")
print(f"# EVALUATION")
print(f"{'#'*80}")

for strat_name, result in strategies.items():
    if result is None:
        print(f"  {strat_name}: No results")
        continue

    idx = result.index

    # Define IS/OOS periods
    periods = [
        ('IS', idx <= OOS_CUTOFF),
        ('OOS', idx > OOS_CUTOFF),
        ('Full', pd.Series(True, index=idx)),
    ]

    for period_label, mask in periods:
        for cm in COST_MULTS:
            metrics = evaluate(result, cm, period_label, mask, strat_name, 'US')
            if metrics:
                all_results.append(metrics)

    # Print OOS summary at key cost levels
    oos_mask = idx > OOS_CUTOFF
    oos = result[oos_mask]
    if len(oos) > 0:
        gsr = sharpe(oos['gross_ret'])
        sr_0x = sharpe(oos['gross_ret'] - oos['cost_drag_0.0'])
        sr_1x = sharpe(oos['gross_ret'] - oos['cost_drag_1.0'])
        sr_3x = sharpe(oos['gross_ret'] - oos['cost_drag_3.0'])
        to = oos['turnover'].mean()
        print(f"    {strat_name:20s}  Gross={gsr:.2f}  Net@0x={sr_0x:.2f}  Net@1x={sr_1x:.2f}  Net@3x={sr_3x:.2f}  TO={to:.1%}")

elapsed = time.time() - t0
print(f"\n\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")

# ==============================================================================
# SAVE RESULTS
# ==============================================================================
results_df = pd.DataFrame(all_results)
outfile = 'scripts/concern_experiments/expCM_us_comparison.csv'
results_df.to_csv(outfile, index=False)
print(f"\nSaved {len(results_df)} rows to {outfile}")


# ==============================================================================
# ANALYSIS: COST-REGIME INTERACTION
# ==============================================================================
print("\n\n" + "=" * 100)
print("ANALYSIS 1: US OOS Net SR by Cost Multiplier")
print("=" * 100)

oos = results_df[results_df['Period'] == 'OOS']
if len(oos) > 0:
    pivot = oos.pivot_table(index='CostMult', columns='Strategy', values='Net_SR')
    strat_order = ['MomentumSort', 'QualitySort', 'Ridge_All', 'Ridge_Slow',
                   'Ridge_SlowMed', 'Ridge_Fast']
    strat_order = [s for s in strat_order if s in pivot.columns]
    pivot = pivot[strat_order]
    print(pivot.round(3).to_string())

    # Find breakeven cost multiplier (where Ridge_All = MomentumSort)
    if 'Ridge_All' in pivot.columns and 'MomentumSort' in pivot.columns:
        diff = pivot['Ridge_All'] - pivot['MomentumSort']
        print(f"\n  Ridge_All - MomentumSort (Net SR difference):")
        for cm in diff.index:
            marker = " <-- BREAKEVEN" if abs(diff[cm]) < 0.05 else ""
            print(f"    {cm}x: {diff[cm]:+.3f}{marker}")

        # Interpolate breakeven
        positive = diff[diff > 0].index
        negative = diff[diff <= 0].index
        if len(positive) > 0 and len(negative) > 0:
            last_pos = max(positive)
            first_neg = min(negative[negative > last_pos]) if len(negative[negative > last_pos]) > 0 else None
            if first_neg is not None:
                # Linear interpolation
                v1, v2 = diff[last_pos], diff[first_neg]
                breakeven = last_pos + (0 - v1) / (v2 - v1) * (first_neg - last_pos)
                print(f"\n  *** Estimated breakeven cost multiplier: {breakeven:.2f}x ***")
                print(f"  (Below {breakeven:.2f}x, Ridge beats MomSort; above, it loses)")


# ==============================================================================
# ANALYSIS 2: TURNOVER COMPARISON
# ==============================================================================
print("\n\n" + "=" * 100)
print("ANALYSIS 2: US OOS Average Monthly Turnover")
print("=" * 100)

if len(oos) > 0:
    # Turnover doesn't depend on cost mult, so just pick one
    oos_0x = oos[oos['CostMult'] == 0.0]
    if len(oos_0x) > 0:
        for _, row in oos_0x.iterrows():
            print(f"  {row['Strategy']:20s}  Turnover={row['Turnover']:.1%}")


# ==============================================================================
# ANALYSIS 3: COMPARISON WITH EM AVERAGES
# ==============================================================================
print("\n\n" + "=" * 100)
print("ANALYSIS 3: US vs EM Comparison (Ridge_All vs MomentumSort)")
print("=" * 100)

# Load EM results for comparison
try:
    em_df = pd.read_csv('scripts/concern_experiments/expAH_feature_speed_test.csv')
    em_oos = em_df[(em_df['Period'] == 'OOS')]

    # EM averages at each cost level
    em_avg = em_oos.groupby(['Strategy', 'CostMult'])[['Net_SR', 'Turnover']].mean()

    us_oos = results_df[results_df['Period'] == 'OOS']

    print(f"\n  {'Cost':>6s}  {'Ridge_All':>18s}  {'MomSort':>18s}  {'Ridge-Mom':>10s}")
    print(f"  {'Mult':>6s}  {'US':>8s} {'EM_Avg':>8s}  {'US':>8s} {'EM_Avg':>8s}  {'US':>5s} {'EM':>5s}")
    print("  " + "-" * 75)

    for cm in [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        us_ridge = us_oos[(us_oos['Strategy'] == 'Ridge_All') & (us_oos['CostMult'] == cm)]
        us_mom = us_oos[(us_oos['Strategy'] == 'MomentumSort') & (us_oos['CostMult'] == cm)]

        us_r = us_ridge['Net_SR'].values[0] if len(us_ridge) > 0 else float('nan')
        us_m = us_mom['Net_SR'].values[0] if len(us_mom) > 0 else float('nan')

        em_r = em_avg.loc[('Ridge_All', cm), 'Net_SR'] if ('Ridge_All', cm) in em_avg.index else float('nan')
        em_m = em_avg.loc[('MomentumSort', cm), 'Net_SR'] if ('MomentumSort', cm) in em_avg.index else float('nan')

        us_diff = us_r - us_m if not (np.isnan(us_r) or np.isnan(us_m)) else float('nan')
        em_diff = em_r - em_m if not (np.isnan(em_r) or np.isnan(em_m)) else float('nan')

        print(f"  {cm:6.2f}  {us_r:8.3f} {em_r:8.3f}  {us_m:8.3f} {em_m:8.3f}  {us_diff:+5.2f} {em_diff:+5.2f}")

    # Turnover comparison
    print(f"\n  TURNOVER:")
    for strat in ['Ridge_All', 'Ridge_Fast', 'Ridge_Slow', 'MomentumSort']:
        us_to_row = us_oos[(us_oos['Strategy'] == strat) & (us_oos['CostMult'] == 0.0)]
        us_to = us_to_row['Turnover'].values[0] if len(us_to_row) > 0 else float('nan')

        em_to_rows = em_oos[(em_oos['Strategy'] == strat) & (em_oos['CostMult'] == 1.0)]
        em_to = em_to_rows['Turnover'].mean() if len(em_to_rows) > 0 else float('nan')

        print(f"    {strat:20s}  US={us_to:.1%}  EM_Avg={em_to:.1%}")

except FileNotFoundError:
    print("  (EM results not found — skipping comparison)")


# ==============================================================================
# ANALYSIS 4: FEATURE SPEED IMPORTANCE IN US
# ==============================================================================
print("\n\n" + "=" * 100)
print("ANALYSIS 4: Feature Speed Profile")
print("=" * 100)

if 'Ridge_All' in strategies and strategies['Ridge_All'] is not None:
    # Re-fit a final Ridge to get coefficients for feature speed analysis
    data_for_coef = df.copy()
    data_for_coef = data_for_coef.sort_values([date_col, id_col])
    data_for_coef['_rank_target'] = data_for_coef.groupby(date_col)[target].rank(pct=True)

    # Use most recent training window
    unique_dates = sorted(data_for_coef[date_col].unique())
    train_end_idx = -1  # use all data up to last date for coefficient analysis
    train_data = data_for_coef[data_for_coef[date_col].isin(unique_dates[:train_end_idx])]

    avail_ml = [f for f in ml_features if f in train_data.columns]
    feat_cov = train_data[avail_ml].notna().mean()
    avail_ml = feat_cov[feat_cov > 0.1].index.tolist()

    X = train_data[avail_ml].copy()
    y = train_data['_rank_target'].values

    imp = SimpleImputer(strategy='median')
    X_imp = imp.fit_transform(X)
    mask = ~np.isnan(y)
    sc = StandardScaler()
    X_scaled = sc.fit_transform(X_imp[mask])

    model = Ridge(alpha=100, random_state=42)
    model.fit(X_scaled, y[mask])

    # Compute absolute coefficient importance by speed class
    coef_abs = np.abs(model.coef_)
    total_importance = coef_abs.sum()

    speed_importance = {'SLOW': 0, 'MEDIUM': 0, 'FAST': 0}
    speed_counts = {'SLOW': 0, 'MEDIUM': 0, 'FAST': 0}

    for feat, coef in zip(avail_ml, coef_abs):
        speed = classify_feature_speed(feat)
        speed_importance[speed] += coef
        speed_counts[speed] += 1

    print(f"\n  Feature Speed Importance (|coef| share):")
    for speed in ['SLOW', 'MEDIUM', 'FAST']:
        share = speed_importance[speed] / total_importance * 100
        n = speed_counts[speed]
        per_feat = speed_importance[speed] / max(n, 1) / total_importance * 100
        print(f"    {speed:8s}: {share:5.1f}% total ({n:3d} features, {per_feat:.2f}% per feature)")

    # Compare with EM: in the paper, FAST features get 61% of importance
    fast_share = speed_importance['FAST'] / total_importance * 100
    print(f"\n  US FAST feature importance share: {fast_share:.1f}%")
    print(f"  EM FAST feature importance share: ~61% (paper Table 2)")
    print(f"  Gap: {fast_share - 61:.1f}pp")


# ==============================================================================
# FINAL SUMMARY
# ==============================================================================
print("\n\n" + "=" * 100)
print("FINAL SUMMARY: Cost-Regime Interaction Hypothesis")
print("=" * 100)

if len(results_df) > 0:
    oos = results_df[results_df['Period'] == 'OOS']

    for strat in ['Ridge_All', 'Ridge_Slow', 'Ridge_Fast', 'MomentumSort']:
        strat_data = oos[oos['Strategy'] == strat]
        if len(strat_data) > 0:
            print(f"\n  {strat}:")
            for _, row in strat_data.iterrows():
                marker = ""
                if row['CostMult'] == 0.0:
                    marker = " (gross)"
                elif row['CostMult'] == 1.0:
                    marker = " (US-realistic)"
                elif row['CostMult'] == 3.0:
                    marker = " (EM-equivalent)"
                print(f"    {row['CostMult']:4.2f}x: SR={row['Net_SR']:+.3f}  TO={row['Turnover']:.1%}  Cost={row['Avg_Cost_bps']:.0f}bps{marker}")

print("\nDone!")
