"""
Experiment DN: Construction robustness — VALUE-WEIGHTED holdings.

Referee question: is the 0.45-0.49 net-SR ceiling an artifact of the shared
portfolio construction (top quintile, 20% buffer, EQUAL weight)? This rerun keeps
the pipeline of expAH_feature_speed_test.py IDENTICAL (same lag-before-filter, same
top-N universe, same 57-feature SLOW set, same buffer selection, same Corwin-Schultz
cost model, same rank targets) and changes ONLY the holding weights: within the long
portfolio each name is weighted proportional to its LAGGED market cap (`me`, which is
already me.shift(1) in this pipeline), renormalized to sum to 1 each month.

Strategies: Ridge_Slow (57 SLOW feats), Ridge_All (all feats), MomentumSort.

CONVENTIONS (differ from the equal-weight expAH; see comments):
  - Gross return  = sum_i w_{i,t} * ret_{i,t},  w_{i,t} ∝ lagged me_i among held names.
  - Turnover      = 0.5 * sum_i | w_{i,t} - w_{i,t-1}^drift |  (one-way weight turnover),
                    where w^drift is last month's weights carried forward and grown by
                    each name's realized return, then renormalized. This is the standard
                    value-weighted turnover; expAH used a name-COUNT turnover
                    (|entries|+|exits|)/(2*n_held), so turnover levels are not identical
                    but both measure the one-way traded fraction.
  - Cost drag(cm) = turnover_weight * clip(mean_1x_cost_of_held * cm, 20, 200) / 10000.
                    Same Corwin-Schultz half-spread * kappa, clipped 20-200 bps as expAH,
                    but applied to the weight turnover and the held names' mean cost
                    (the value-weighted analog of expAH's frac_traded * traded_cost).

Output: scripts/concern_experiments/expDN_construction_robustness.csv
Schema: Market, Strategy, CostMult, Period, Net_SR, Gross_SR, Ann_Ret, Turnover, Avg_Cost_bps, N_months

Usage:
  python scripts/concern_experiments/expDN_construction_robustness.py               # all 15
  python scripts/concern_experiments/expDN_construction_robustness.py --markets THA  # subset
"""

import sys
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, 'public/src')
sys.path.insert(0, 'scripts/concern_experiments')

import argparse
import numpy as np
import pandas as pd
import warnings
import time
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from data_loader import load_equity_data, lag_features
from market_config import MARKETS, COST_MULTS, BUFFER_PCT, MIN_TRAIN_MONTHS

np.random.seed(42)


# ==============================================================================
# FEATURE CLASSIFICATION — copied VERBATIM from expAH_feature_speed_test.py
# (SLOW/MEDIUM/FAST keyword lists + classify_feature_speed must be identical so
#  the 57-feature SLOW set matches expAH exactly.)
# ==============================================================================

SLOW_KEYWORDS = [
    'f_score', 'o_score', 'z_score',
    'qmj', 'qmj_prof', 'qmj_safety', 'qmj_growth',
    'roa', 'roe', 'gp_at', 'gp_atl1', 'op_at', 'op_atl1',
    'niq_at', 'niq_be', 'ni_be', 'ni_me',
    'cop_at', 'cop_atl1', 'ope_be', 'ope_bel1', 'ocf_at', 'ocf_me',
    'oaccruals_at', 'oaccruals_ni', 'taccruals_at', 'taccruals_ni',
    'be_me', 'bev_mev', 'at_me', 'ebit_bev', 'ebitda_mev',
    'fcf_me', 'sale_me', 'debt_me', 'netdebt_me',
    'div12m_me', 'ebit_sale', 'cash_at', 'tangibility',
    'noa_at', 'opex_at', 'rd5_at', 'rd_sale', 'rd_me', 'sale_bev',
    'mispricing_mgmt', 'mispricing_perf', 'eq_dur', 'age',
    'at_be', 'at_turnover', 'pi_nix', 'kz_index', 'aliq_at', 'aliq_mat',
    'eqpo_me', 'eqnpo_me', 'ival_me',
]
MEDIUM_KEYWORDS = [
    'at_gr1', 'ppeinv_gr1a', 'noa_gr1a', 'lnoa_gr1a',
    'capx_gr1', 'capx_gr2', 'capx_gr3', 'capex_abn',
    'chcsho_12m', 'eqnpo_12m', 'debt_gr3', 'inv_gr1', 'inv_gr1a',
    'cowc_gr1a', 'coa_gr1a', 'col_gr1a', 'nncoa_gr1a',
    'ncoa_gr1a', 'ncol_gr1a', 'nfna_gr1a',
    'sti_gr1a', 'lti_gr1a', 'fnl_gr1a', 'be_gr1a',
    'netis_at', 'eqnetis_at', 'dbnetis_at', 'tax_gr1a',
    'niq_at_chg1', 'niq_be_chg1', 'ocf_at_chg1', 'ni_ar1', 'ni_inc8q',
    'dsale_dinv', 'dsale_drec', 'dgp_dsale', 'dsale_dsga',
    'sale_gr1', 'sale_gr3', 'emp_gr1', 'sale_emp_gr1', 'saleq_gr1',
    'niq_su', 'saleq_su', 'ocfq_saleq_std', 'earnings_variability',
    'seas_1_1an', 'seas_1_1na', 'seas_2_5an', 'seas_2_5na',
    'seas_6_10an', 'seas_6_10na', 'seas_11_15an', 'seas_11_15na',
    'seas_16_20an', 'seas_16_20na',
]
FAST_KEYWORDS = [
    'ret_12_1', 'ret_12_7', 'ret_6_1', 'ret_9_1', 'ret_3_1', 'ret_60_12',
    'resff3_12_1', 'resff3_6_1',
    'ivol_capm_21d', 'ivol_capm_252d', 'ivol_ff3_21d', 'ivol_hxz4_21d', 'rvol_21d',
    'beta_60m', 'beta_dimson_21d', 'betabab_1260d', 'betadown_252d', 'ni_ivol',
    'iskew_capm_21d', 'iskew_ff3_21d', 'iskew_hxz4_21d',
    'rskew_21d', 'coskew_21d', 'corr_1260d',
    'rmax1_21d', 'rmax5_21d', 'rmax5_rvol_21d', 'ami_126d', 'bidaskhl_21d',
    'dolvol_126d', 'dolvol_var_126d', 'turnover_126d', 'turnover_var_126d',
    'zero_trades_126d', 'zero_trades_21d', 'zero_trades_252d',
    'prc', 'prc_highprc_252d',
]

# Accounting classifier used ONLY for the lag step (same as expAH/expAX).
ACCOUNTING_KEYWORDS = [
    'f_score', 'o_score', 'z_score', 'qmj', 'qmj_prof', 'qmj_safety', 'qmj_growth',
    'roa', 'roe', 'gp_at', 'gp_atl1', 'op_at', 'op_atl1',
    'niq_at', 'niq_be', 'niq_at_chg1', 'niq_be_chg1',
    'ni_be', 'ni_me', 'ni_ar1', 'ni_inc8q', 'cop_at', 'cop_atl1',
    'ope_be', 'ope_bel1', 'ocf_at', 'ocf_at_chg1', 'ocf_me',
    'oaccruals_at', 'oaccruals_ni', 'taccruals_at', 'taccruals_ni',
    'be_me', 'bev_mev', 'at_me', 'ebit_bev', 'ebitda_mev',
    'fcf_me', 'sale_me', 'debt_me', 'netdebt_me',
    'div12m_me', 'ebit_sale', 'cash_at', 'tangibility',
    'noa_at', 'opex_at', 'rd5_at', 'rd_sale', 'rd_me', 'sale_bev', 'sale_emp_gr1',
    'mispricing_mgmt', 'mispricing_perf', 'eq_dur', 'age',
    'at_gr1', 'ppeinv_gr1a', 'noa_gr1a', 'lnoa_gr1a',
    'capx_gr1', 'capx_gr2', 'capx_gr3', 'capex_abn',
    'chcsho_12m', 'eqnpo_12m', 'debt_gr3', 'inv_gr1', 'inv_gr1a',
    'cowc_gr1a', 'coa_gr1a', 'col_gr1a', 'nncoa_gr1a',
    'ncoa_gr1a', 'ncol_gr1a', 'nfna_gr1a',
    'sti_gr1a', 'lti_gr1a', 'fnl_gr1a', 'be_gr1a',
    'netis_at', 'eqnetis_at', 'dbnetis_at', 'at_be', 'at_turnover',
    'pi_nix', 'kz_index', 'aliq_at', 'aliq_mat',
    'tax_gr1a', 'niq_su', 'saleq_su', 'saleq_gr1',
    'dsale_dinv', 'dsale_drec', 'dgp_dsale', 'dsale_dsga',
    'ocfq_saleq_std', 'earnings_variability',
    'sale_gr1', 'sale_gr3', 'emp_gr1', 'eqpo_me', 'eqnpo_me', 'ival_me',
]

EXCLUDE_COLS = ['Unnamed: 0', 'id', 'eom', 'excntry', 'gvkey', 'permno',
                'size_grp', 'me', 'market_equity', 'ret_1_0', 'date']


def classify_feature_speed(feature_name):
    f = feature_name.lower()
    for kw in FAST_KEYWORDS:
        if f == kw:
            return 'FAST'
    for kw in MEDIUM_KEYWORDS:
        if f == kw:
            return 'MEDIUM'
    for kw in SLOW_KEYWORDS:
        if f == kw:
            return 'SLOW'
    if any(x in f for x in ['ret_', 'vol', 'beta', 'ivol', 'rvol', 'skew', 'ami',
                            'bidask', 'dolvol', 'turnover', 'zero_trades',
                            'prc', 'rmax', 'corr_', 'coskew']):
        return 'FAST'
    if any(x in f for x in ['_gr1', '_gr2', '_gr3', '_chg', 'seas_', '_su',
                            'variability', 'netis', 'eqnetis', 'dbnetis']):
        return 'MEDIUM'
    return 'SLOW'


def classify_features(all_features):
    accounting, market = [], []
    for f in all_features:
        fl = f.lower()
        if any(kw == fl or kw in fl for kw in ACCOUNTING_KEYWORDS):
            accounting.append(f)
        else:
            market.append(f)
    return accounting, market


# ==============================================================================
# HELPERS
# ==============================================================================

def sharpe(s):
    s = s.dropna()
    if len(s) < 2 or s.std() == 0:
        return 0.0
    return (s.mean() / s.std()) * np.sqrt(12)


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


def vw_weights(held_ids, me_map):
    """Value weights ∝ lagged me among held names, renormalized to sum 1.
    Falls back to equal weight if all me are missing/zero."""
    ids = list(held_ids)
    caps = np.array([me_map.get(i, np.nan) for i in ids], dtype=float)
    caps = np.where(np.isnan(caps) | (caps <= 0), 0.0, caps)
    tot = caps.sum()
    if tot <= 0:
        w = np.full(len(ids), 1.0 / max(len(ids), 1))
    else:
        w = caps / tot
    return dict(zip(ids, w))


def vw_turnover_and_cost(w_curr, w_prev, ret_prev_map, held_cost_1x,
                         global_med_1x, prev_long):
    """One-way weight turnover and per-cost-mult drag for value-weighted holdings.

    w_curr, w_prev: dict id -> target weight this / last month.
    ret_prev_map:   dict id -> realized return during last month (for weight drift).
    Returns (turnover, {cm: cost_drag}). turnover is NaN on the first month.
    """
    if not prev_long or not w_prev:
        return np.nan, {cm: 0.0 for cm in COST_MULTS}

    # Drift last month's weights by realized returns, renormalize
    drift = {}
    for i, w in w_prev.items():
        r = ret_prev_map.get(i, 0.0)
        if not np.isfinite(r):
            r = 0.0
        drift[i] = w * (1.0 + r)
    dtot = sum(drift.values())
    if dtot > 0:
        drift = {i: v / dtot for i, v in drift.items()}

    all_ids = set(w_curr) | set(drift)
    turnover = 0.5 * sum(abs(w_curr.get(i, 0.0) - drift.get(i, 0.0)) for i in all_ids)

    cost_drags = {}
    for cm in COST_MULTS:
        traded_cost = np.clip(held_cost_1x * cm, 20.0, 200.0)
        cost_drags[cm] = turnover * (traded_cost / 10000)
    return turnover, cost_drags


# ==============================================================================
# VALUE-WEIGHTED FACTOR SORT
# ==============================================================================

def run_factor_sort_vw(df, sort_features, date_col, id_col, target,
                       cost_lookup_1x, nq=5, buffer_pct=0.20):
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
    prev_long, w_prev, ret_prev_map = set(), {}, {}
    records = []

    for date in dates:
        month = data[data[date_col] == date].copy()
        if len(month) < nq:
            continue
        sids = month[id_col].values
        ranks = month['pct_rank'].values
        prev_ids = np.array(list(prev_long), dtype=sids.dtype) if prev_long else np.array([], dtype=sids.dtype)
        in_prev = np.isin(sids, prev_ids)
        keep = in_prev & (ranks >= top_threshold - buffer)
        enter = ~in_prev & (ranks >= top_threshold + buffer)
        new_long = set(sids[keep | enter])
        if len(new_long) == 0:
            new_long = set(month.nlargest(max(1, len(month) // nq), 'pct_rank')[id_col])

        me_map = dict(zip(month[id_col], month['me'])) if 'me' in month.columns else {}
        ret_map = dict(zip(month[id_col], month[target]))
        w_curr = vw_weights(new_long, me_map)
        gross_ret = sum(w_curr[i] * ret_map.get(i, 0.0) for i in new_long)

        held_cost_1x = month[month[id_col].isin(new_long)]['cost_bps_1x'].mean()
        if pd.isna(held_cost_1x):
            held_cost_1x = global_med_1x
        turn, cost_drags = vw_turnover_and_cost(w_curr, w_prev, ret_prev_map,
                                                held_cost_1x, global_med_1x, prev_long)

        records.append({'date': date, 'gross_ret': gross_ret, 'turnover': turn,
                        'n_held': len(new_long),
                        **{f'cost_drag_{cm}': cost_drags[cm] for cm in COST_MULTS}})
        prev_long, w_prev = new_long, w_curr
        ret_prev_map = {i: ret_map.get(i, 0.0) for i in new_long}

    if len(records) == 0:
        return None
    return pd.DataFrame(records).set_index('date')


# ==============================================================================
# VALUE-WEIGHTED RIDGE WALK-FORWARD
# (selection/prediction identical to expAH; only weighting differs)
# ==============================================================================

def run_ridge_walkforward_vw(df, features, date_col, id_col, target,
                             cost_lookup_1x, nq=5, buffer_pct=0.20, min_train=60):
    data = df.copy().sort_values([date_col, id_col]).reset_index(drop=True)
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

    prev_long, w_prev, ret_prev_map = set(), {}, {}
    records = []

    for t_idx in range(min_train, n_dates):
        train_dates = unique_dates[:t_idx]
        test_date = unique_dates[t_idx]
        train_data = data[data[date_col].isin(train_dates)]
        test_data = data[data[date_col] == test_date].copy()
        if len(train_data) < 100 or len(test_data) < nq:
            continue

        X_train = train_data[avail_features].copy()
        y_train = train_data['_rank_target'].values
        imputer = SimpleImputer(strategy='median')
        X_train_imp = imputer.fit_transform(X_train)
        mask = ~np.isnan(y_train)
        if mask.sum() < 50:
            continue
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp[mask])
        model = Ridge(alpha=100, random_state=42)
        model.fit(X_train_scaled, y_train[mask])

        X_test_imp = imputer.transform(test_data[avail_features].copy())
        preds = model.predict(scaler.transform(X_test_imp))
        test_data['prediction'] = preds
        test_data['pct_rank'] = test_data['prediction'].rank(pct=True)

        sids = test_data[id_col].values
        ranks = test_data['pct_rank'].values
        prev_ids = np.array(list(prev_long), dtype=sids.dtype) if prev_long else np.array([], dtype=sids.dtype)
        in_prev = np.isin(sids, prev_ids)
        keep = in_prev & (ranks >= top_threshold - buffer)
        enter = ~in_prev & (ranks >= top_threshold + buffer)
        new_long = set(sids[keep | enter])
        if len(new_long) == 0:
            new_long = set(test_data.nlargest(max(1, len(test_data) // nq), 'pct_rank')[id_col])

        me_map = dict(zip(test_data[id_col], test_data['me'])) if 'me' in test_data.columns else {}
        ret_map = dict(zip(test_data[id_col], test_data[target]))
        w_curr = vw_weights(new_long, me_map)
        gross_ret = sum(w_curr[i] * ret_map.get(i, 0.0) for i in new_long)

        held_cost_1x = test_data[test_data[id_col].isin(new_long)]['cost_bps_1x'].mean()
        if pd.isna(held_cost_1x):
            held_cost_1x = global_med_1x
        turn, cost_drags = vw_turnover_and_cost(w_curr, w_prev, ret_prev_map,
                                                held_cost_1x, global_med_1x, prev_long)

        records.append({'date': test_date, 'gross_ret': gross_ret, 'turnover': turn,
                        'n_held': len(new_long),
                        **{f'cost_drag_{cm}': cost_drags[cm] for cm in COST_MULTS}})
        prev_long, w_prev = new_long, w_curr
        ret_prev_map = {i: ret_map.get(i, 0.0) for i in new_long}

    if len(records) == 0:
        return None
    return pd.DataFrame(records).set_index('date')


# ==============================================================================
# EVALUATION (expAH/expAX schema)
# ==============================================================================

def evaluate(result_df, cost_mult, period_label, period_mask, strategy_name, market):
    if result_df is None:
        return None
    subset = result_df[period_mask]
    if len(subset) < 6:
        return None
    cost_col = f'cost_drag_{cost_mult}'
    net_ret = subset['gross_ret'] - subset[cost_col]
    return {
        'Market': market, 'Strategy': strategy_name, 'CostMult': cost_mult,
        'Period': period_label,
        'Net_SR': round(sharpe(net_ret), 4),
        'Gross_SR': round(sharpe(subset['gross_ret']), 4),
        'Ann_Ret': round(net_ret.mean() * 12, 4),
        'Turnover': round(subset['turnover'].mean(), 4),
        'Avg_Cost_bps': round(subset[cost_col].mean() * 10000, 1),
        'N_months': len(subset),
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--markets', type=str, default=None,
                    help='Comma-separated market subset (e.g. THA,KOR). Default: all.')
    args = ap.parse_args()

    markets = dict(MARKETS)
    if args.markets:
        want = [m.strip().upper() for m in args.markets.split(',')]
        markets = {k: v for k, v in MARKETS.items() if k in want}
        missing = [m for m in want if m not in MARKETS]
        if missing:
            print(f"WARNING: unknown markets ignored: {missing}", flush=True)

    print("=" * 100, flush=True)
    print("EXPERIMENT DN: Value-weighted construction robustness", flush=True)
    print(f"  Markets: {list(markets.keys())}", flush=True)
    print(f"  Strategies: Ridge_Slow, Ridge_All, MomentumSort  |  Cost mults: {COST_MULTS}", flush=True)
    print("=" * 100, flush=True)

    all_results = []
    t0 = time.time()

    for mkt_name, cfg in markets.items():
        mkt_t0 = time.time()
        print(f"\n{'#'*80}\n# MARKET: {mkt_name}\n{'#'*80}", flush=True)

        try:
            df, raw_features, date_col, id_col = load_equity_data(cfg['file'], cfg['type'])
        except Exception as e:
            print(f"  ERROR loading {mkt_name}: {e}", flush=True)
            continue

        target = cfg['target']
        all_features = [f for f in raw_features if f not in EXCLUDE_COLS and f != target]
        accounting_feats, market_feats = classify_features(all_features)
        accounting_feats = [f for f in accounting_feats if f in df.columns]
        market_feats = [f for f in market_feats if f in df.columns]

        # Lag BEFORE filter
        if accounting_feats:
            df = lag_features(df, accounting_feats, id_col, date_col, lag=1)
        if market_feats:
            df = lag_features(df, market_feats, id_col, date_col, lag=1)

        # Lag me (kept as a column for value weighting AND used for the filter), filter top-N
        if 'me' in df.columns:
            df = df.sort_values([id_col, date_col])
            df['me'] = df.groupby(id_col)['me'].shift(1)
            df = df.dropna(subset=['me'])
            df['_me_rank'] = df.groupby(date_col)['me'].rank(ascending=False, method='first')
            df = df[df['_me_rank'] <= cfg['topn']].drop(columns=['_me_rank'])

        cost_lookup_1x = build_cost_lookup_1x(df, date_col, id_col)
        ml_features = [f for f in accounting_feats + market_feats if f in df.columns]
        slow_feats = [f for f in ml_features if classify_feature_speed(f) == 'SLOW']
        df[date_col] = pd.to_datetime(df[date_col])
        nq = cfg['nq']

        # IS/OOS split (expAH logic)
        unique_dates = sorted(df[date_col].unique())
        n_total = len(unique_dates)
        is_split_idx = max(int(n_total * 0.8), MIN_TRAIN_MONTHS)
        is_end_date = unique_dates[min(is_split_idx - 1, n_total - 1)]
        print(f"  Features: {len(ml_features)} total ({len(slow_feats)} SLOW)  "
              f"IS ends {is_end_date}  OOS months: {n_total - is_split_idx}", flush=True)

        def make_masks(result_df):
            idx = result_df.index
            return [('IS', idx <= is_end_date), ('OOS', idx > is_end_date),
                    ('Full', pd.Series(True, index=idx))]

        strategies = {}
        print("  --- MomentumSort (VW) ---", flush=True)
        strategies['MomentumSort'] = run_factor_sort_vw(
            df, ['ret_12_1'], date_col, id_col, target, cost_lookup_1x,
            nq=nq, buffer_pct=BUFFER_PCT)

        print(f"  --- Ridge_All (VW, {len(ml_features)} feats) ---", flush=True)
        strategies['Ridge_All'] = run_ridge_walkforward_vw(
            df, ml_features, date_col, id_col, target, cost_lookup_1x,
            nq=nq, buffer_pct=BUFFER_PCT, min_train=MIN_TRAIN_MONTHS)

        print(f"  --- Ridge_Slow (VW, {len(slow_feats)} feats) ---", flush=True)
        if len(slow_feats) >= 5:
            strategies['Ridge_Slow'] = run_ridge_walkforward_vw(
                df, slow_feats, date_col, id_col, target, cost_lookup_1x,
                nq=nq, buffer_pct=BUFFER_PCT, min_train=MIN_TRAIN_MONTHS)
        else:
            strategies['Ridge_Slow'] = None
            print(f"    Skipping Ridge_Slow: only {len(slow_feats)} SLOW features", flush=True)

        for strat_name, result in strategies.items():
            if result is None:
                continue
            for period_label, mask_item in make_masks(result):
                for cm in COST_MULTS:
                    m = evaluate(result, cm, period_label, mask_item, strat_name, mkt_name)
                    if m:
                        all_results.append(m)

        mkt_elapsed = time.time() - mkt_t0
        print(f"\n  OOS Net SR @3x [{mkt_elapsed:.0f}s]:", flush=True)
        for strat_name in ['MomentumSort', 'Ridge_All', 'Ridge_Slow']:
            result = strategies.get(strat_name)
            if result is None:
                continue
            oos = result[result.index > is_end_date]
            if len(oos) > 0:
                sr3 = sharpe(oos['gross_ret'] - oos['cost_drag_3.0'])
                gsr = sharpe(oos['gross_ret'])
                to = oos['turnover'].mean()
                print(f"    {strat_name:14s}  Gross={gsr:.2f}  Net@3x={sr3:.2f}  TO={to:.1%}", flush=True)

    elapsed = time.time() - t0
    print(f"\n\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)

    results_df = pd.DataFrame(all_results)
    outfile = 'scripts/concern_experiments/expDN_construction_robustness.csv'
    results_df.to_csv(outfile, index=False)
    print(f"Saved {len(results_df)} rows to {outfile}", flush=True)

    oos_3x = results_df[(results_df['Period'] == 'OOS') & (results_df['CostMult'] == 3.0)]
    if len(oos_3x) > 0:
        print("\n" + "=" * 80, flush=True)
        print("OOS Net SR @3x by strategy (VALUE-WEIGHTED, mean across markets)", flush=True)
        print("=" * 80, flush=True)
        for s in ['MomentumSort', 'Ridge_All', 'Ridge_Slow']:
            sd = oos_3x[oos_3x['Strategy'] == s]
            if len(sd) > 0:
                print(f"  {s:14s}  mean={sd['Net_SR'].mean():.3f}  "
                      f"median={sd['Net_SR'].median():.3f}  "
                      f"wins>0={int((sd['Net_SR'] > 0).sum())}/{len(sd)}  "
                      f"TO={sd['Turnover'].mean():.1%}", flush=True)

    print("\nDone!", flush=True)


if __name__ == '__main__':
    main()
