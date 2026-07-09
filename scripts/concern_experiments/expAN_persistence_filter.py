"""
Experiment AN: Persistence-Filtered Ridge with IN-SAMPLE Rank Autocorrelation

Motivation: Experiment AM showed Ridge restricted to empirically-slow features
(high rank autocorrelation) achieves SR=1.03 vs momentum sort SR=0.81 on 5
markets. BUT the rank AC was computed on the FULL sample, which is leakage --
the feature classification used future data.

This experiment fixes that: rank autocorrelation is computed INSIDE the
walk-forward loop using ONLY training data, recomputed every 12 months.
This is the clean, non-cheating version.

Strategies tested:
  1. MomentumSort      - Simple 12-1 momentum sort (baseline)
  2. Ridge_All         - Walk-forward Ridge with all features
  3. Ridge_PersFiltered - Ridge with persistence-filtered features (rank AC > 0.75,
                          computed on trailing 60 months of training data only)
  4. Ridge_AdhocSlow   - Ridge with ad-hoc SLOW features (accounting keywords)
  5. Ridge_EmpSlow_FS  - Ridge with full-sample rank AC > 0.75 ("cheating" version)

Key comparison: Ridge_PersFiltered vs Ridge_EmpSlow_FS reveals whether
persistence filtering is stable over time (low leakage effect) or whether
it degrades when done properly.

Pipeline: dual-lag (accounting=4, market=1), buffer, walk-forward, CS costs.
All 15 emerging markets.

Output: scripts/concern_experiments/expAN_persistence_filter.csv
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
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy import stats

from data_loader import load_equity_data, lag_features
from market_config import (MARKETS, COST_MULTS, BUFFER_PCT, MIN_TRAIN_MONTHS,
                           PERSISTENCE_THRESHOLD, PERSISTENCE_FALLBACK_THRESHOLD,
                           PERSISTENCE_MIN_FEATURES, PERSISTENCE_RECOMPUTE_INTERVAL,
                           PERSISTENCE_LOOKBACK, PERSISTENCE_MIN_MONTHS_AC)

np.random.seed(42)


# ==============================================================================
# FEATURE SPEED CLASSIFICATION (ad-hoc, from expAH)
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
    """Classify a single feature into SLOW/MEDIUM/FAST (ad-hoc)."""
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

ACCOUNTING_KEYWORDS = [
    'f_score', 'o_score', 'z_score',
    'qmj', 'qmj_prof', 'qmj_safety', 'qmj_growth',
    'roa', 'roe', 'gp_at', 'gp_atl1', 'op_at', 'op_atl1',
    'niq_at', 'niq_be', 'niq_at_chg1', 'niq_be_chg1',
    'ni_be', 'ni_me', 'ni_ar1', 'ni_inc8q',
    'cop_at', 'cop_atl1',
    'ope_be', 'ope_bel1',
    'ocf_at', 'ocf_at_chg1', 'ocf_me',
    'oaccruals_at', 'oaccruals_ni',
    'taccruals_at', 'taccruals_ni',
    'be_me', 'bev_mev', 'at_me', 'ebit_bev', 'ebitda_mev',
    'fcf_me', 'sale_me', 'debt_me', 'netdebt_me',
    'div12m_me', 'ebit_sale', 'cash_at', 'tangibility',
    'noa_at', 'opex_at', 'rd5_at', 'rd_sale', 'rd_me',
    'sale_bev', 'sale_emp_gr1',
    'mispricing_mgmt', 'mispricing_perf',
    'eq_dur', 'age',
    'at_gr1', 'ppeinv_gr1a', 'noa_gr1a', 'lnoa_gr1a',
    'capx_gr1', 'capx_gr2', 'capx_gr3', 'capex_abn',
    'chcsho_12m', 'eqnpo_12m', 'debt_gr3',
    'inv_gr1', 'inv_gr1a',
    'cowc_gr1a', 'coa_gr1a', 'col_gr1a', 'nncoa_gr1a',
    'ncoa_gr1a', 'ncol_gr1a', 'nfna_gr1a',
    'sti_gr1a', 'lti_gr1a', 'fnl_gr1a', 'be_gr1a',
    'netis_at', 'eqnetis_at', 'dbnetis_at',
    'at_be', 'at_turnover',
    'pi_nix', 'kz_index', 'aliq_at', 'aliq_mat',
    'tax_gr1a', 'niq_su', 'saleq_su', 'saleq_gr1',
    'dsale_dinv', 'dsale_drec', 'dgp_dsale', 'dsale_dsga',
    'ocfq_saleq_std', 'earnings_variability',
    'sale_gr1', 'sale_gr3', 'emp_gr1',
    'eqpo_me', 'eqnpo_me', 'ival_me',
]

# Local-only persistence filter parameters (not in market_config)
PERSISTENCE_MIN_FEATURES_HARD = 5
PERSISTENCE_MIN_STOCKS = 30  # need at least 30 stocks per month for AC


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def sharpe(s):
    s = s.dropna()
    if len(s) < 2 or s.std() == 0:
        return 0.0
    return (s.mean() / s.std()) * np.sqrt(12)


def classify_features(all_features):
    """Classify into accounting (lag=1) vs market (lag=1)."""
    accounting = []
    market = []
    for f in all_features:
        f_lower = f.lower()
        if any(kw == f_lower or kw in f_lower for kw in ACCOUNTING_KEYWORDS):
            accounting.append(f)
        else:
            market.append(f)
    return accounting, market


def build_cost_lookup_1x(raw_df, date_col, id_col, spread_col='bidaskhl_21d',
                         floor_bps=20, cap_bps=200):
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
# RANK AUTOCORRELATION (training-window only)
# ==============================================================================

def compute_rank_ac_training(df, features, date_col, id_col,
                             train_dates, lookback=60,
                             min_months=24, min_stocks=30):
    """
    Compute rank autocorrelation for each feature using ONLY training data.

    Uses last `lookback` months of training dates for efficiency and relevance.
    Returns dict: feature_name -> mean rank AC.
    """
    # Use only the last `lookback` months of training data
    if len(train_dates) > lookback:
        use_dates = train_dates[-lookback:]
    else:
        use_dates = train_dates

    if len(use_dates) < min_months:
        return {}

    # Subset data to these dates
    use_dates_set = set(use_dates)
    sub = df[df[date_col].isin(use_dates_set)].copy()

    rank_ac = {}

    for feat in features:
        if feat not in sub.columns:
            continue

        # Check NaN fraction
        nan_frac = sub[feat].isna().mean()
        if nan_frac > 0.50:
            continue

        correlations = []
        for i in range(len(use_dates) - 1):
            d1 = use_dates[i]
            d2 = use_dates[i + 1]

            # Get cross-section for each month
            cs1 = sub[sub[date_col] == d1][[id_col, feat]].dropna(subset=[feat])
            cs1 = cs1.drop_duplicates(subset=[id_col])

            cs2 = sub[sub[date_col] == d2][[id_col, feat]].dropna(subset=[feat])
            cs2 = cs2.drop_duplicates(subset=[id_col])

            if len(cs1) < min_stocks or len(cs2) < min_stocks:
                continue

            # Merge on id_col to get common stocks
            merged = cs1.merge(cs2, on=id_col, suffixes=('_t', '_t1'))

            if len(merged) < min_stocks:
                continue

            # Spearman rank correlation
            rho, _ = stats.spearmanr(merged[f'{feat}_t'], merged[f'{feat}_t1'])
            if not np.isnan(rho):
                correlations.append(rho)

        if len(correlations) >= min_months // 2:  # need enough month-pairs
            rank_ac[feat] = np.mean(correlations)

    return rank_ac


def compute_rank_ac_fullsample(df, features, date_col, id_col, min_stocks=30):
    """
    Compute rank autocorrelation on FULL sample (for the "cheating" baseline).
    Same as expAM's compute_rank_autocorrelation.
    """
    df_sorted = df.sort_values(date_col).copy()
    dates = sorted(df_sorted[date_col].unique())

    rank_ac = {}

    for feat in features:
        if feat not in df_sorted.columns:
            continue

        correlations = []
        for i in range(len(dates) - 1):
            d1 = dates[i]
            d2 = dates[i + 1]

            cs1 = df_sorted[df_sorted[date_col] == d1][[id_col, feat]].dropna(subset=[feat])
            cs1 = cs1.drop_duplicates(subset=[id_col])

            cs2 = df_sorted[df_sorted[date_col] == d2][[id_col, feat]].dropna(subset=[feat])
            cs2 = cs2.drop_duplicates(subset=[id_col])

            merged = cs1.merge(cs2, on=id_col, suffixes=('_t', '_t1'))

            if len(merged) < min_stocks:
                continue

            rho, _ = stats.spearmanr(merged[f'{feat}_t'], merged[f'{feat}_t1'])
            if not np.isnan(rho):
                correlations.append(rho)

        if len(correlations) > 0:
            rank_ac[feat] = np.mean(correlations)

    return rank_ac


def select_persistent_features(rank_ac, threshold=0.75, fallback_threshold=0.60,
                                min_features=10, min_features_hard=5,
                                all_features=None):
    """
    Select features with rank AC above threshold.
    Fallback logic: lower threshold if too few features pass.
    """
    if not rank_ac:
        return all_features if all_features else [], threshold, 'fallback_all'

    selected = [f for f, ac in rank_ac.items() if ac >= threshold]

    if len(selected) >= min_features:
        return selected, threshold, 'primary'

    # Fallback: lower threshold
    selected = [f for f, ac in rank_ac.items() if ac >= fallback_threshold]
    if len(selected) >= min_features_hard:
        return selected, fallback_threshold, 'fallback_low'

    # Ultimate fallback: use all features
    if all_features:
        return all_features, 0.0, 'fallback_all'
    return list(rank_ac.keys()), 0.0, 'fallback_all'


# ==============================================================================
# FACTOR SORT (monthly)
# ==============================================================================

def run_factor_sort(df, sort_features, date_col, id_col, target,
                    cost_lookup_1x, nq=5, buffer_pct=0.20):
    """Factor sort with monthly rebalancing."""
    data = df.copy()
    for feat in sort_features:
        if feat not in data.columns:
            continue
        data[f'{feat}_rank'] = data.groupby(date_col)[feat].rank(pct=True)
        data[f'{feat}_rank'] = data[f'{feat}_rank'].fillna(0.5)

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
# ML WALK-FORWARD (standard Ridge, fixed feature set)
# ==============================================================================

def run_ml_walkforward(df, features, date_col, id_col, target,
                       cost_lookup_1x, nq=5, buffer_pct=0.20, min_train=60):
    """Walk-forward Ridge regression with a fixed feature set."""
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
# ML WALK-FORWARD WITH PERSISTENCE FILTERING (the new one)
# ==============================================================================

def run_ml_walkforward_persistence(df, all_features, date_col, id_col, target,
                                    cost_lookup_1x, nq=5, buffer_pct=0.20,
                                    min_train=60, mkt_name=''):
    """
    Walk-forward Ridge with persistence-filtered features.
    Rank AC is computed INSIDE the loop using ONLY training data.
    Recomputed every PERSISTENCE_RECOMPUTE_INTERVAL months.
    """
    data = df.copy()
    data = data.sort_values([date_col, id_col]).reset_index(drop=True)

    # Pre-compute rank target (cross-sectional percentile ranks)
    data['_rank_target'] = data.groupby(date_col)[target].rank(pct=True)

    unique_dates = sorted(data[date_col].unique())
    n_dates = len(unique_dates)

    if n_dates < min_train + 12:
        return None, []

    data = data.merge(cost_lookup_1x, on=[date_col, id_col], how='left')
    global_med_1x = cost_lookup_1x['cost_bps_1x'].median()
    data['cost_bps_1x'] = data['cost_bps_1x'].fillna(global_med_1x)

    avail_features = [f for f in all_features if f in data.columns]
    feat_coverage = data[avail_features].notna().mean()
    avail_features = feat_coverage[feat_coverage > 0.1].index.tolist()

    if len(avail_features) == 0:
        return None, []

    top_threshold = 1.0 - 1.0 / nq
    buffer = buffer_pct * (1.0 / nq)

    prev_long = set()
    records = []
    feature_selection_stats = []  # track what features were selected

    # Cache for persistence-filtered features
    cached_features = None
    last_recompute_idx = -PERSISTENCE_RECOMPUTE_INTERVAL  # force recompute on first iteration

    for t_idx in range(min_train, n_dates):
        train_dates = unique_dates[:t_idx]
        test_date = unique_dates[t_idx]

        test_data = data[data[date_col] == test_date].copy()

        if len(test_data) < nq:
            continue

        # Decide whether to recompute rank AC
        months_since_recompute = t_idx - last_recompute_idx
        if months_since_recompute >= PERSISTENCE_RECOMPUTE_INTERVAL or cached_features is None:
            # Recompute rank AC on training data
            rank_ac = compute_rank_ac_training(
                data, avail_features, date_col, id_col,
                train_dates,
                lookback=PERSISTENCE_LOOKBACK,
                min_months=PERSISTENCE_MIN_MONTHS_AC,
                min_stocks=PERSISTENCE_MIN_STOCKS
            )

            selected_feats, threshold_used, method = select_persistent_features(
                rank_ac,
                threshold=PERSISTENCE_THRESHOLD,
                fallback_threshold=PERSISTENCE_FALLBACK_THRESHOLD,
                min_features=PERSISTENCE_MIN_FEATURES,
                min_features_hard=PERSISTENCE_MIN_FEATURES_HARD,
                all_features=avail_features
            )

            cached_features = selected_feats
            last_recompute_idx = t_idx

            # Log progress
            print(f"    {mkt_name} PersFilter t={t_idx}: "
                  f"selected {len(selected_feats)} feats "
                  f"(thresh={threshold_used:.2f}, method={method})")

            # Track stats
            if rank_ac:
                selected_acs = [rank_ac.get(f, np.nan) for f in selected_feats if f in rank_ac]
                feature_selection_stats.append({
                    'date': test_date,
                    't_idx': t_idx,
                    'n_features_selected': len(selected_feats),
                    'threshold_used': threshold_used,
                    'method': method,
                    'mean_ac_selected': np.nanmean(selected_acs) if selected_acs else np.nan,
                    'n_features_with_ac': len(rank_ac),
                })

        use_features = cached_features

        # Now run Ridge with selected features
        train_data = data[data[date_col].isin(train_dates)]

        if len(train_data) < 100:
            continue

        X_train = train_data[use_features].copy()
        y_train = train_data['_rank_target'].values
        X_test = test_data[use_features].copy()

        imputer = SimpleImputer(strategy='median')
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        mask_y = ~np.isnan(y_train)
        if mask_y.sum() < 50:
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp[mask_y])
        X_test_scaled = scaler.transform(X_test_imp)

        model = Ridge(alpha=100, random_state=42)
        model.fit(X_train_scaled, y_train[mask_y])
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
            'n_features': len(use_features),
            **{f'cost_drag_{cm}': cost_drags[cm] for cm in COST_MULTS},
        })
        prev_long = new_long

    if len(records) == 0:
        return None, feature_selection_stats
    return pd.DataFrame(records).set_index('date'), feature_selection_stats


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate(result_df, cost_mult, period_label, period_mask,
             strategy_name, market):
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
print("EXPERIMENT AN: Persistence-Filtered Ridge with IN-SAMPLE Rank AC")
print("  Rank AC computed INSIDE walk-forward loop (training data only)")
print("  Recomputed every 12 months on trailing 60 months")
print("  15 emerging markets, 5 strategies")
print("=" * 100)

all_results = []
all_feat_stats = []  # feature selection stats across markets
t0 = time.time()

for mkt_name, cfg in MARKETS.items():
    print(f"\n{'#'*80}")
    print(f"# MARKET: {mkt_name}")
    print(f"{'#'*80}")

    try:
        df, raw_features, date_col, id_col = load_equity_data(
            cfg['file'], cfg['type']
        )
    except Exception as e:
        print(f"  ERROR loading {mkt_name}: {e}")
        continue

    target = cfg['target']
    all_features = [f for f in raw_features if f not in EXCLUDE_COLS and f != target]
    accounting_feats, market_feats = classify_features(all_features)
    accounting_feats = [f for f in accounting_feats if f in df.columns]
    market_feats = [f for f in market_feats if f in df.columns]

    # CRITICAL: Lag BEFORE filter (accounting=4, market=1)
    if accounting_feats:
        df = lag_features(df, accounting_feats, id_col, date_col, lag=1)
    if market_feats:
        df = lag_features(df, market_feats, id_col, date_col, lag=1)

    # Fix: lag me to avoid look-ahead bias (me(T) embeds ret_1_0(T))
    if 'me' in df.columns:
        df = df.sort_values([id_col, date_col])
        df['me'] = df.groupby(id_col)['me'].shift(1)
        df = df.dropna(subset=['me'])

    # Filter to top-N by market cap
    topn = cfg['topn']
    if 'me' in df.columns:
        df['_me_rank'] = df.groupby(date_col)['me'].rank(ascending=False, method='first')
        df = df[df['_me_rank'] <= topn].drop(columns=['_me_rank'])

    cost_lookup_1x = build_cost_lookup_1x(df, date_col, id_col)
    ml_features = [f for f in accounting_feats + market_feats if f in df.columns]

    # Ad-hoc slow features
    adhoc_slow_feats = [f for f in ml_features if classify_feature_speed(f) == 'SLOW']

    print(f"  Features: {len(ml_features)} total, {len(adhoc_slow_feats)} ad-hoc SLOW")

    df[date_col] = pd.to_datetime(df[date_col])
    nq = cfg['nq']

    # IS/OOS split
    unique_dates = sorted(df[date_col].unique())
    n_total = len(unique_dates)
    is_split_idx = max(int(n_total * 0.8), MIN_TRAIN_MONTHS)
    is_end_date = unique_dates[min(is_split_idx - 1, n_total - 1)]

    print(f"  IS ends: {is_end_date}, OOS months: {n_total - is_split_idx}")

    def make_masks(result_df):
        idx = result_df.index
        return [('IS', idx <= is_end_date), ('OOS', idx > is_end_date),
                ('Full', pd.Series(True, index=idx))]

    # ==================================================================
    # STRATEGIES
    # ==================================================================
    strategies = {}

    # 1. MomentumSort (baseline)
    print(f"\n  --- Strategy 1: MomentumSort ---")
    strategies['MomentumSort'] = run_factor_sort(
        df, ['ret_12_1'], date_col, id_col, target, cost_lookup_1x,
        nq=nq, buffer_pct=BUFFER_PCT
    )

    # 2. Ridge_All (all features)
    print(f"  --- Strategy 2: Ridge_All ({len(ml_features)} features) ---")
    strategies['Ridge_All'] = run_ml_walkforward(
        df, ml_features, date_col, id_col, target, cost_lookup_1x,
        nq=nq, buffer_pct=BUFFER_PCT, min_train=MIN_TRAIN_MONTHS
    )

    # 3. Ridge_PersFiltered (persistence-filtered, rank AC inside loop)
    print(f"  --- Strategy 3: Ridge_PersFiltered (rank AC inside walk-forward) ---")
    result_pf, feat_stats = run_ml_walkforward_persistence(
        df, ml_features, date_col, id_col, target, cost_lookup_1x,
        nq=nq, buffer_pct=BUFFER_PCT, min_train=MIN_TRAIN_MONTHS,
        mkt_name=mkt_name
    )
    strategies['Ridge_PersFiltered'] = result_pf
    for fs in feat_stats:
        fs['Market'] = mkt_name
    all_feat_stats.extend(feat_stats)

    # 4. Ridge_AdhocSlow (ad-hoc SLOW features)
    print(f"  --- Strategy 4: Ridge_AdhocSlow ({len(adhoc_slow_feats)} features) ---")
    if len(adhoc_slow_feats) >= 5:
        strategies['Ridge_AdhocSlow'] = run_ml_walkforward(
            df, adhoc_slow_feats, date_col, id_col, target, cost_lookup_1x,
            nq=nq, buffer_pct=BUFFER_PCT, min_train=MIN_TRAIN_MONTHS
        )
    else:
        strategies['Ridge_AdhocSlow'] = None
        print(f"    Skipping: only {len(adhoc_slow_feats)} SLOW features")

    # 5. Ridge_EmpSlow_FS (full-sample rank AC > 0.75 -- "cheating" version)
    print(f"  --- Strategy 5: Ridge_EmpSlow_FS (full-sample rank AC, cheating) ---")
    fullsample_rank_ac = compute_rank_ac_fullsample(
        df, ml_features, date_col, id_col, min_stocks=PERSISTENCE_MIN_STOCKS
    )
    fs_selected, fs_thresh, fs_method = select_persistent_features(
        fullsample_rank_ac,
        threshold=PERSISTENCE_THRESHOLD,
        fallback_threshold=PERSISTENCE_FALLBACK_THRESHOLD,
        min_features=PERSISTENCE_MIN_FEATURES,
        min_features_hard=PERSISTENCE_MIN_FEATURES_HARD,
        all_features=ml_features
    )
    print(f"    Full-sample: selected {len(fs_selected)} features "
          f"(thresh={fs_thresh:.2f}, method={fs_method})")

    if len(fs_selected) >= 5:
        strategies['Ridge_EmpSlow_FS'] = run_ml_walkforward(
            df, fs_selected, date_col, id_col, target, cost_lookup_1x,
            nq=nq, buffer_pct=BUFFER_PCT, min_train=MIN_TRAIN_MONTHS
        )
    else:
        strategies['Ridge_EmpSlow_FS'] = None

    # Evaluate all
    for strat_name, result in strategies.items():
        if result is None:
            continue
        for period_label, mask in make_masks(result):
            for cm in COST_MULTS:
                metrics = evaluate(result, cm, period_label, mask,
                                   strat_name, mkt_name)
                if metrics:
                    # Add n_features info for PersFiltered
                    if strat_name == 'Ridge_PersFiltered' and 'n_features' in result.columns:
                        oos_nf = result[result.index > is_end_date]['n_features'].mean()
                        metrics['Avg_N_Features'] = round(oos_nf, 1)
                    all_results.append(metrics)

        # Print OOS summary
        oos_mask = result.index > is_end_date
        oos = result[oos_mask]
        if len(oos) > 0:
            sr3 = sharpe(oos['gross_ret'] - oos['cost_drag_3.0'])
            gsr = sharpe(oos['gross_ret'])
            to = oos['turnover'].mean()
            extra = ''
            if strat_name == 'Ridge_PersFiltered' and 'n_features' in result.columns:
                avg_nf = oos['n_features'].mean()
                extra = f'  NFeats={avg_nf:.0f}'
            print(f"    {strat_name:25s}  Gross={gsr:.2f}  Net@3x={sr3:.2f}  TO={to:.1%}{extra}")


elapsed = time.time() - t0
print(f"\n\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")


# ==============================================================================
# SAVE RESULTS
# ==============================================================================

results_df = pd.DataFrame(all_results)
outfile = 'scripts/concern_experiments/expAN_persistence_filter.csv'
results_df.to_csv(outfile, index=False)
print(f"\nSaved {len(results_df)} rows to {outfile}")


# ==============================================================================
# TABLE 1: Per-market OOS Net SR at 3x Costs
# ==============================================================================

print("\n\n" + "=" * 100)
print("TABLE 1: Per-Market OOS Net SR at 3x Costs")
print("=" * 100)

oos_3x = results_df[(results_df['Period'] == 'OOS') & (results_df['CostMult'] == 3.0)]
strat_order = ['MomentumSort', 'Ridge_All', 'Ridge_PersFiltered',
               'Ridge_AdhocSlow', 'Ridge_EmpSlow_FS']
if len(oos_3x) > 0:
    pivot = oos_3x.pivot_table(index='Market', columns='Strategy', values='Net_SR')
    strat_order = [s for s in strat_order if s in pivot.columns]
    pivot = pivot[strat_order]
    print(pivot.round(2).to_string())
    print(f"\nMean:")
    print(pivot.mean().round(3).to_string())
    print(f"\nMedian:")
    print(pivot.median().round(3).to_string())


# ==============================================================================
# TABLE 2: Cross-Market Mean at Each Cost Level
# ==============================================================================

print("\n\n" + "=" * 100)
print("TABLE 2: Cross-Market Mean OOS Net SR at Each Cost Level")
print("=" * 100)

oos_all = results_df[results_df['Period'] == 'OOS']
if len(oos_all) > 0:
    for cm in COST_MULTS:
        oos_cm = oos_all[oos_all['CostMult'] == cm]
        pivot_cm = oos_cm.pivot_table(index='Market', columns='Strategy', values='Net_SR')
        strat_order_cm = [s for s in strat_order if s in pivot_cm.columns]
        pivot_cm = pivot_cm[strat_order_cm]
        means = pivot_cm.mean()
        print(f"\nCost {cm:.1f}x: ", end='')
        for s in strat_order_cm:
            print(f"  {s}={means[s]:.3f}", end='')
    print()


# ==============================================================================
# TABLE 3: Feature Selection Stats for Ridge_PersFiltered
# ==============================================================================

print("\n\n" + "=" * 100)
print("TABLE 3: Feature Selection Stats for Ridge_PersFiltered")
print("=" * 100)

if all_feat_stats:
    feat_stats_df = pd.DataFrame(all_feat_stats)
    for mkt in feat_stats_df['Market'].unique():
        mkt_stats = feat_stats_df[feat_stats_df['Market'] == mkt]
        avg_n = mkt_stats['n_features_selected'].mean()
        min_n = mkt_stats['n_features_selected'].min()
        max_n = mkt_stats['n_features_selected'].max()
        avg_thresh = mkt_stats['threshold_used'].mean()
        avg_ac = mkt_stats['mean_ac_selected'].mean()
        methods = mkt_stats['method'].value_counts().to_dict()
        print(f"  {mkt:4s}: Avg_N={avg_n:.0f}, Min_N={min_n}, Max_N={max_n}, "
              f"Avg_Thresh={avg_thresh:.2f}, Avg_AC={avg_ac:.3f}, Methods={methods}")

    # Overall summary
    print(f"\n  Overall: Avg_N={feat_stats_df['n_features_selected'].mean():.0f}, "
          f"Min_N={feat_stats_df['n_features_selected'].min()}, "
          f"Max_N={feat_stats_df['n_features_selected'].max()}")


# ==============================================================================
# TABLE 4: Win Counts (OOS Net SR at 3x)
# ==============================================================================

print("\n\n" + "=" * 100)
print("TABLE 4: Win Counts (Best OOS Net SR at 3x Costs)")
print("=" * 100)

if len(oos_3x) > 0:
    pivot = oos_3x.pivot_table(index='Market', columns='Strategy', values='Net_SR')
    strat_order_win = [s for s in strat_order if s in pivot.columns]
    pivot = pivot[strat_order_win]

    wins = {}
    for s in strat_order_win:
        wins[s] = 0

    for mkt in pivot.index:
        row = pivot.loc[mkt]
        best = row.idxmax()
        if pd.notna(best):
            wins[best] += 1

    print(f"  {'Strategy':25s}  Wins  Mean_SR  Median_SR")
    print(f"  {'-'*60}")
    for s in strat_order_win:
        mean_sr = pivot[s].mean()
        med_sr = pivot[s].median()
        print(f"  {s:25s}  {wins[s]:4d}  {mean_sr:7.3f}  {med_sr:9.3f}")


# ==============================================================================
# TABLE 5: Turnover Comparison
# ==============================================================================

print("\n\n" + "=" * 100)
print("TABLE 5: OOS Average Monthly Turnover")
print("=" * 100)

if len(oos_3x) > 0:
    to_pivot = oos_3x.pivot_table(index='Market', columns='Strategy', values='Turnover')
    strat_order_to = [s for s in strat_order if s in to_pivot.columns]
    to_pivot = to_pivot[strat_order_to]
    print(to_pivot.round(3).to_string())
    print(f"\nMean:")
    print(to_pivot.mean().round(3).to_string())


# ==============================================================================
# TABLE 6: Leakage Effect -- PersFiltered vs EmpSlow_FS
# ==============================================================================

print("\n\n" + "=" * 100)
print("TABLE 6: Leakage Effect -- Ridge_PersFiltered vs Ridge_EmpSlow_FS")
print("=" * 100)

if len(oos_3x) > 0:
    pivot = oos_3x.pivot_table(index='Market', columns='Strategy', values='Net_SR')
    if 'Ridge_PersFiltered' in pivot.columns and 'Ridge_EmpSlow_FS' in pivot.columns:
        pivot['Leakage_Effect'] = pivot['Ridge_EmpSlow_FS'] - pivot['Ridge_PersFiltered']
        print(pivot[['Ridge_PersFiltered', 'Ridge_EmpSlow_FS', 'Leakage_Effect']].round(3).to_string())
        print(f"\nMean leakage effect: {pivot['Leakage_Effect'].mean():.3f}")
        print(f"Median leakage effect: {pivot['Leakage_Effect'].median():.3f}")
        n_positive = (pivot['Leakage_Effect'] > 0).sum()
        n_total_lk = pivot['Leakage_Effect'].notna().sum()
        print(f"Leakage helps in {n_positive}/{n_total_lk} markets")


# ==============================================================================
# KEY TAKEAWAY
# ==============================================================================

print("\n\n" + "=" * 100)
print("KEY TAKEAWAY")
print("=" * 100)

if len(oos_3x) > 0:
    pivot = oos_3x.pivot_table(index='Market', columns='Strategy', values='Net_SR')
    strat_order_final = [s for s in ['MomentumSort', 'Ridge_All', 'Ridge_PersFiltered',
                                      'Ridge_AdhocSlow', 'Ridge_EmpSlow_FS']
                         if s in pivot.columns]
    means = pivot[strat_order_final].mean()
    print(f"\nMean OOS Net SR at 3x costs:")
    for s in strat_order_final:
        print(f"  {s:25s}: {means[s]:.3f}")

    if 'Ridge_PersFiltered' in means and 'MomentumSort' in means:
        delta = means['Ridge_PersFiltered'] - means['MomentumSort']
        print(f"\n  PersFiltered vs MomentumSort: {delta:+.3f}")
    if 'Ridge_PersFiltered' in means and 'Ridge_All' in means:
        delta = means['Ridge_PersFiltered'] - means['Ridge_All']
        print(f"  PersFiltered vs Ridge_All:    {delta:+.3f}")
    if 'Ridge_PersFiltered' in means and 'Ridge_AdhocSlow' in means:
        delta = means['Ridge_PersFiltered'] - means['Ridge_AdhocSlow']
        print(f"  PersFiltered vs AdhocSlow:    {delta:+.3f}")
    if 'Ridge_PersFiltered' in means and 'Ridge_EmpSlow_FS' in means:
        delta_leak = means['Ridge_EmpSlow_FS'] - means['Ridge_PersFiltered']
        print(f"  Leakage effect (FS - PF):     {delta_leak:+.3f}")

print("\nDone!")
