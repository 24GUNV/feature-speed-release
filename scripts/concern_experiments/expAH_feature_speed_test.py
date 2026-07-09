"""
Experiment AH: Feature Speed Hypothesis Test — SLOW-Only Ridge + 3-Month Returns

Direct test of the "feature speed" mechanism:
  1. Ridge_All:       Ridge with all ~154 features (baseline)
  2. Ridge_Slow:      Ridge with only SLOW (accounting) features
  3. Ridge_SlowMed:   Ridge with SLOW + MEDIUM features (no FAST)
  4. Ridge_Fast:      Ridge with only FAST (market) features
  5. MomentumSort:    Simple 12-1 momentum sort (baseline)
  6. Ridge_3m:        Ridge (all features) predicting 3-month return, rebalance quarterly
  7. MomSort_3m:      Momentum sort, rebalance quarterly

Hypothesis: Ridge_Slow should have:
  - Lower turnover than Ridge_All (closer to sorts)
  - Comparable or better net SR at high cost levels
  - If true, confirms feature speed is the mechanism

Also tests quarterly rebalancing: predicting ret_3_0 instead of ret_1_0
should reduce turnover for ALL strategies and potentially help ML.

Pipeline matches expAX exactly: dual-lag, buffer, walk-forward, CS costs.

Output: scripts/concern_experiments/expAH_feature_speed_test.csv
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

np.random.seed(42)


# ==============================================================================
# FEATURE SPEED CLASSIFICATION (from expAX_turnover_decomposition.py)
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

from market_config import MARKETS, COST_MULTS, BUFFER_PCT, MIN_TRAIN_MONTHS

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


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def sharpe(s):
    s = s.dropna()
    if len(s) < 2 or s.std() == 0:
        return 0.0
    return (s.mean() / s.std()) * np.sqrt(12)


def classify_features(all_features):
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
# FACTOR SORT (monthly or quarterly)
# ==============================================================================

def run_factor_sort(df, sort_features, date_col, id_col, target,
                    cost_lookup_1x, nq=5, buffer_pct=0.20,
                    ascending_features=None, rebal_freq=1):
    """
    Factor sort with optional quarterly rebalancing.
    rebal_freq=1: rebalance every month (standard)
    rebal_freq=3: rebalance every 3 months (quarterly)
    """
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

        # Decide whether to rebalance this month
        do_rebalance = (i % rebal_freq == 0) or len(prev_long) == 0

        if do_rebalance:
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
        else:
            # Hold existing portfolio, only drop stocks that disappeared
            available_ids = set(month[id_col].unique())
            new_long = prev_long & available_ids
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
# ML WALK-FORWARD (monthly or quarterly rebalancing)
# ==============================================================================

def run_ml_walkforward(df, features, date_col, id_col, target,
                       cost_lookup_1x, nq=5, buffer_pct=0.20,
                       min_train=60, rebal_freq=1):
    """
    Walk-forward Ridge with optional quarterly rebalancing.
    rebal_freq=1: retrain and rebalance monthly
    rebal_freq=3: retrain and rebalance every 3 months
    """
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
    last_model = None
    last_imputer = None
    last_scaler = None
    rebal_counter = 0

    for t_idx in range(min_train, n_dates):
        train_dates = unique_dates[:t_idx]
        test_date = unique_dates[t_idx]

        train_data = data[data[date_col].isin(train_dates)]
        test_data = data[data[date_col] == test_date].copy()

        if len(test_data) < nq:
            continue

        do_rebalance = (rebal_counter % rebal_freq == 0) or last_model is None

        if do_rebalance and len(train_data) >= 100:
            X_train = train_data[avail_features].copy()
            y_train = train_data['_rank_target'].values
            X_test = test_data[avail_features].copy()

            imputer = SimpleImputer(strategy='median')
            X_train_imp = imputer.fit_transform(X_train)
            X_test_imp = imputer.transform(X_test)

            mask = ~np.isnan(y_train)
            if mask.sum() < 50:
                rebal_counter += 1
                continue

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imp[mask])
            X_test_scaled = scaler.transform(X_test_imp)

            model = Ridge(alpha=100, random_state=42)
            model.fit(X_train_scaled, y_train[mask])
            preds = model.predict(X_test_scaled)

            last_model = model
            last_imputer = imputer
            last_scaler = scaler
        elif last_model is not None:
            # Use cached model, just score this month
            X_test = test_data[avail_features].copy()
            X_test_imp = last_imputer.transform(X_test)
            X_test_scaled = last_scaler.transform(X_test_imp)
            preds = last_model.predict(X_test_scaled)
        else:
            rebal_counter += 1
            continue

        test_data = test_data.copy()
        test_data['prediction'] = preds
        test_data['pct_rank'] = test_data['prediction'].rank(pct=True)

        if do_rebalance or len(prev_long) == 0:
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
        else:
            available_ids = set(test_data[id_col].unique())
            new_long = prev_long & available_ids
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
        rebal_counter += 1

    if len(records) == 0:
        return None
    return pd.DataFrame(records).set_index('date')


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
print("EXPERIMENT AH: Feature Speed Hypothesis Test — SLOW-Only Ridge + Quarterly Rebalancing")
print("=" * 100)

all_results = []
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

    # CRITICAL: Lag BEFORE filter
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

    # Classify features by speed
    slow_feats = [f for f in ml_features if classify_feature_speed(f) == 'SLOW']
    medium_feats = [f for f in ml_features if classify_feature_speed(f) == 'MEDIUM']
    fast_feats = [f for f in ml_features if classify_feature_speed(f) == 'FAST']
    slow_med_feats = slow_feats + medium_feats

    print(f"  Features: {len(ml_features)} total = {len(slow_feats)} SLOW + {len(medium_feats)} MEDIUM + {len(fast_feats)} FAST")

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

    # 1. MomentumSort (monthly baseline)
    print(f"\n  --- MomentumSort (monthly) ---")
    strategies['MomentumSort'] = run_factor_sort(
        df, ['ret_12_1'], date_col, id_col, target, cost_lookup_1x,
        nq=nq, buffer_pct=BUFFER_PCT, rebal_freq=1
    )

    # 2. Ridge_All (all features, monthly)
    print(f"  --- Ridge_All ({len(ml_features)} features, monthly) ---")
    strategies['Ridge_All'] = run_ml_walkforward(
        df, ml_features, date_col, id_col, target, cost_lookup_1x,
        nq=nq, buffer_pct=BUFFER_PCT, min_train=MIN_TRAIN_MONTHS, rebal_freq=1
    )

    # 3. Ridge_Slow (only SLOW features, monthly)
    print(f"  --- Ridge_Slow ({len(slow_feats)} features, monthly) ---")
    if len(slow_feats) >= 5:
        strategies['Ridge_Slow'] = run_ml_walkforward(
            df, slow_feats, date_col, id_col, target, cost_lookup_1x,
            nq=nq, buffer_pct=BUFFER_PCT, min_train=MIN_TRAIN_MONTHS, rebal_freq=1
        )
    else:
        strategies['Ridge_Slow'] = None
        print(f"    Skipping: only {len(slow_feats)} SLOW features")

    # 4. Ridge_SlowMed (SLOW + MEDIUM features, monthly)
    print(f"  --- Ridge_SlowMed ({len(slow_med_feats)} features, monthly) ---")
    if len(slow_med_feats) >= 5:
        strategies['Ridge_SlowMed'] = run_ml_walkforward(
            df, slow_med_feats, date_col, id_col, target, cost_lookup_1x,
            nq=nq, buffer_pct=BUFFER_PCT, min_train=MIN_TRAIN_MONTHS, rebal_freq=1
        )
    else:
        strategies['Ridge_SlowMed'] = None

    # 5. Ridge_Fast (only FAST features, monthly)
    print(f"  --- Ridge_Fast ({len(fast_feats)} features, monthly) ---")
    if len(fast_feats) >= 5:
        strategies['Ridge_Fast'] = run_ml_walkforward(
            df, fast_feats, date_col, id_col, target, cost_lookup_1x,
            nq=nq, buffer_pct=BUFFER_PCT, min_train=MIN_TRAIN_MONTHS, rebal_freq=1
        )
    else:
        strategies['Ridge_Fast'] = None

    # 6. Ridge_All_Q (all features, quarterly rebalancing)
    print(f"  --- Ridge_All_Q ({len(ml_features)} features, quarterly) ---")
    strategies['Ridge_All_Q'] = run_ml_walkforward(
        df, ml_features, date_col, id_col, target, cost_lookup_1x,
        nq=nq, buffer_pct=BUFFER_PCT, min_train=MIN_TRAIN_MONTHS, rebal_freq=3
    )

    # 7. MomentumSort_Q (quarterly rebalancing)
    print(f"  --- MomentumSort_Q (quarterly) ---")
    strategies['MomentumSort_Q'] = run_factor_sort(
        df, ['ret_12_1'], date_col, id_col, target, cost_lookup_1x,
        nq=nq, buffer_pct=BUFFER_PCT, rebal_freq=3
    )

    # 8. Ridge_Slow_Q (SLOW features, quarterly)
    print(f"  --- Ridge_Slow_Q ({len(slow_feats)} features, quarterly) ---")
    if len(slow_feats) >= 5:
        strategies['Ridge_Slow_Q'] = run_ml_walkforward(
            df, slow_feats, date_col, id_col, target, cost_lookup_1x,
            nq=nq, buffer_pct=BUFFER_PCT, min_train=MIN_TRAIN_MONTHS, rebal_freq=3
        )
    else:
        strategies['Ridge_Slow_Q'] = None

    # Evaluate all
    for strat_name, result in strategies.items():
        if result is None:
            continue
        for period_label, mask in make_masks(result):
            for cm in COST_MULTS:
                metrics = evaluate(result, cm, period_label, mask,
                                   strat_name, mkt_name)
                if metrics:
                    all_results.append(metrics)

        # Print OOS summary
        oos_mask = result.index > is_end_date
        oos = result[oos_mask]
        if len(oos) > 0:
            sr3 = sharpe(oos['gross_ret'] - oos['cost_drag_3.0'])
            gsr = sharpe(oos['gross_ret'])
            to = oos['turnover'].mean()
            print(f"    {strat_name:20s}  Gross={gsr:.2f}  Net@3x={sr3:.2f}  TO={to:.1%}")

elapsed = time.time() - t0
print(f"\n\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")

# Save results
results_df = pd.DataFrame(all_results)
outfile = 'scripts/concern_experiments/expAH_feature_speed_test.csv'
results_df.to_csv(outfile, index=False)
print(f"\nSaved {len(results_df)} rows to {outfile}")

# ==============================================================================
# SUMMARY TABLE
# ==============================================================================
print("\n\n" + "="*100)
print("SUMMARY: OOS Net SR at 3x Costs")
print("="*100)

oos_3x = results_df[(results_df['Period'] == 'OOS') & (results_df['CostMult'] == 3.0)]
if len(oos_3x) > 0:
    pivot = oos_3x.pivot_table(index='Market', columns='Strategy', values='Net_SR')
    strat_order = ['MomentumSort', 'Ridge_All', 'Ridge_Slow', 'Ridge_SlowMed',
                   'Ridge_Fast', 'Ridge_All_Q', 'MomentumSort_Q', 'Ridge_Slow_Q']
    strat_order = [s for s in strat_order if s in pivot.columns]
    pivot = pivot[strat_order]
    print(pivot.round(2).to_string())
    print(f"\nMean:")
    print(pivot.mean().round(3).to_string())

    # Turnover comparison
    print("\n\nTURNOVER: OOS Average Monthly Turnover")
    print("="*100)
    to_pivot = oos_3x.pivot_table(index='Market', columns='Strategy', values='Turnover')
    to_pivot = to_pivot[[s for s in strat_order if s in to_pivot.columns]]
    print(to_pivot.round(3).to_string())
    print(f"\nMean:")
    print(to_pivot.mean().round(3).to_string())

print("\nDone!")
