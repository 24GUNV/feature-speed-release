"""
Experiment AI: Aggregation vs Speed Decomposition

Tests whether the real mechanism behind ML's higher turnover is:
  (a) Feature speed (noisy inputs), or
  (b) Multivariate aggregation (combining many signals destabilizes rankings)

Strategies:
  1. Single-feature sorts: Sort on each of 10 individual features
  2. Composite sorts:     Sort on 5, 10, 20, 50, all features
  3. Ridge_N:             Ridge with same N features as composite sorts (paired comparison)
  4. Rank autocorrelation: Per-feature rank AC(1) for THA, KOR, TWN

NOTE: Ridge with 1 feature produces identical rankings to a sort (monotonic
linear transform preserves rank order). So Ridge_SingleMom is NOT included.
The real test is Ridge_N vs CompositeSort_N at each dimensionality level,
which isolates the sort-vs-regression aggregation effect.

Addresses Concerns 3 (aggregation vs speed) and 4 (momentum paradox).

Pipeline alignment updated 2026-07-03 to match expAH clean pipeline:
lag=1 for accounting/market features, lagged-me top-N filter, rank targets,
buffer, walk-forward, and CS costs.

Output: scripts/concern_experiments/expAI_aggregation_vs_speed.csv
"""

import sys
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, 'scripts/concern_experiments')
sys.path.insert(0, 'public/src')

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
from market_config import MARKETS, COST_MULTS, BUFFER_PCT, MIN_TRAIN_MONTHS

np.random.seed(42)


# ==============================================================================
# FEATURE SPEED CLASSIFICATION (from expAH)
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
# HELPER FUNCTIONS (from expAH)
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
# FACTOR SORT (from expAH)
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
            new_long = set()
            for _, row in month.iterrows():
                sid = row[id_col]
                rank = row['pct_rank']
                if sid in prev_long:
                    if rank >= top_threshold - buffer:
                        new_long.add(sid)
                else:
                    if rank >= top_threshold + buffer:
                        new_long.add(sid)
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
# ML WALK-FORWARD (from expAH)
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
            new_long = set()
            for _, row in test_data.iterrows():
                sid = row[id_col]
                rank = row['pct_rank']
                if sid in prev_long:
                    if rank >= top_threshold - buffer:
                        new_long.add(sid)
                else:
                    if rank >= top_threshold + buffer:
                        new_long.add(sid)
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
# EVALUATION (from expAH)
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
# NEW: RANK AUTOCORRELATION
# ==============================================================================

def compute_rank_autocorrelation(df, feature, date_col, id_col):
    """
    For a given feature, compute cross-sectional rank AC(1).
    For each month t, compute cross-sectional rank of the feature.
    Compute Spearman correlation between rank_t and rank_{t-1}
    (for stocks present in both months).
    Average across all months to get rank AC(1).
    """
    data = df[[date_col, id_col, feature]].dropna(subset=[feature]).copy()
    if len(data) == 0:
        return np.nan

    # Compute cross-sectional rank within each month
    data['rank'] = data.groupby(date_col)[feature].rank(pct=True)

    dates = sorted(data[date_col].unique())
    correlations = []

    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        curr_date = dates[i]

        prev_month = data[data[date_col] == prev_date][[id_col, 'rank']].drop_duplicates(subset=[id_col]).set_index(id_col)
        curr_month = data[data[date_col] == curr_date][[id_col, 'rank']].drop_duplicates(subset=[id_col]).set_index(id_col)

        # Find stocks present in both months
        common_ids = prev_month.index.intersection(curr_month.index)
        if len(common_ids) < 10:
            continue

        prev_ranks = prev_month.loc[common_ids, 'rank'].values
        curr_ranks = curr_month.loc[common_ids, 'rank'].values

        corr, _ = stats.spearmanr(prev_ranks, curr_ranks)
        if not np.isnan(corr):
            correlations.append(corr)

    if len(correlations) == 0:
        return np.nan
    return np.mean(correlations)


# ==============================================================================
# MAIN
# ==============================================================================

print("=" * 100)
print("EXPERIMENT AI: Aggregation vs Speed Decomposition")
print("  Addresses Concerns 3 (aggregation vs speed) and 4 (momentum paradox)")
print("=" * 100)

all_results = []
rank_ac_results = []  # For rank autocorrelation
t0 = time.time()

# Define single-feature sort targets
SINGLE_SORT_FEATURES = {
    'ret_12_1': {'ascending': False, 'speed': 'FAST'},
    'ret_6_1': {'ascending': False, 'speed': 'FAST'},
    'ret_3_1': {'ascending': False, 'speed': 'FAST'},
    'rvol_21d': {'ascending': True, 'speed': 'FAST'},
    'ivol_capm_21d': {'ascending': True, 'speed': 'FAST'},
    'f_score': {'ascending': False, 'speed': 'SLOW'},
    'be_me': {'ascending': False, 'speed': 'SLOW'},
    'roa': {'ascending': False, 'speed': 'SLOW'},
    'gp_at': {'ascending': False, 'speed': 'SLOW'},
    'oaccruals_at': {'ascending': True, 'speed': 'SLOW'},
}

# Composite sort N values
COMPOSITE_N_VALUES = [5, 10, 20, 50]

# Markets for rank AC computation
RANK_AC_MARKETS = ['THA', 'KOR', 'TWN']

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

    # Classify features by speed for ordering
    slow_feats = [f for f in ml_features if classify_feature_speed(f) == 'SLOW']
    medium_feats = [f for f in ml_features if classify_feature_speed(f) == 'MEDIUM']
    fast_feats = [f for f in ml_features if classify_feature_speed(f) == 'FAST']

    # Ordered feature list: SLOW first, then MEDIUM, then FAST
    ordered_features = slow_feats + medium_feats + fast_feats

    print(f"  Features: {len(ml_features)} total = {len(slow_feats)} SLOW + "
          f"{len(medium_feats)} MEDIUM + {len(fast_feats)} FAST")

    df[date_col] = pd.to_datetime(df[date_col])
    nq = cfg['nq']

    # IS/OOS split (same logic as expAH)
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

    # ------------------------------------------------------------------
    # 1. BASELINES: MomentumSort and Ridge_All (same as expAH)
    # ------------------------------------------------------------------
    print(f"\n  --- MomentumSort (baseline) ---")
    strategies['MomentumSort'] = run_factor_sort(
        df, ['ret_12_1'], date_col, id_col, target, cost_lookup_1x,
        nq=nq, buffer_pct=BUFFER_PCT, rebal_freq=1
    )

    print(f"  --- Ridge_All ({len(ml_features)} features, baseline) ---")
    strategies['Ridge_All'] = run_ml_walkforward(
        df, ml_features, date_col, id_col, target, cost_lookup_1x,
        nq=nq, buffer_pct=BUFFER_PCT, min_train=MIN_TRAIN_MONTHS, rebal_freq=1
    )

    # ------------------------------------------------------------------
    # 2. Single-feature sorts
    # ------------------------------------------------------------------
    for feat_name, feat_cfg in SINGLE_SORT_FEATURES.items():
        if feat_name not in df.columns:
            print(f"  --- Sort_{feat_name}: SKIPPED (not in data) ---")
            continue

        asc_list = [feat_name] if feat_cfg['ascending'] else []
        strat_name = f"Sort_{feat_name}"
        print(f"  --- {strat_name} (speed={feat_cfg['speed']}) ---")

        strategies[strat_name] = run_factor_sort(
            df, [feat_name], date_col, id_col, target, cost_lookup_1x,
            nq=nq, buffer_pct=BUFFER_PCT, ascending_features=asc_list, rebal_freq=1
        )

    # ------------------------------------------------------------------
    # 3. Composite sorts + paired Ridge_N (SLOW-first ordering)
    #    At each N, run both CompositeSort_N and Ridge_N with the SAME
    #    features. This isolates the sort-vs-regression aggregation effect.
    # ------------------------------------------------------------------
    for N in COMPOSITE_N_VALUES:
        if N > len(ordered_features):
            continue
        subset_feats = ordered_features[:N]

        # CompositeSort_N
        sort_name = f"CompositeSort_{N}"
        print(f"  --- {sort_name} ({N} features) ---")
        asc_in_subset = []
        for f in subset_feats:
            if f in SINGLE_SORT_FEATURES and SINGLE_SORT_FEATURES[f]['ascending']:
                asc_in_subset.append(f)
        strategies[sort_name] = run_factor_sort(
            df, subset_feats, date_col, id_col, target, cost_lookup_1x,
            nq=nq, buffer_pct=BUFFER_PCT, ascending_features=asc_in_subset, rebal_freq=1
        )

        # Paired Ridge_N with same features
        ridge_name = f"Ridge_{N}"
        print(f"  --- {ridge_name} ({N} features) ---")
        strategies[ridge_name] = run_ml_walkforward(
            df, subset_feats, date_col, id_col, target, cost_lookup_1x,
            nq=nq, buffer_pct=BUFFER_PCT, min_train=MIN_TRAIN_MONTHS, rebal_freq=1
        )

    # CompositeSort_all (pairs with existing Ridge_All)
    n_all = len(ordered_features)
    strat_name = f"CompositeSort_{n_all}"
    print(f"  --- {strat_name} (all {n_all} features) ---")
    asc_all = []
    for f in ordered_features:
        if f in SINGLE_SORT_FEATURES and SINGLE_SORT_FEATURES[f]['ascending']:
            asc_all.append(f)
    strategies[strat_name] = run_factor_sort(
        df, ordered_features, date_col, id_col, target, cost_lookup_1x,
        nq=nq, buffer_pct=BUFFER_PCT, ascending_features=asc_all, rebal_freq=1
    )

    # ------------------------------------------------------------------
    # 5. Rank autocorrelation (only for THA, KOR, TWN)
    # ------------------------------------------------------------------
    if mkt_name in RANK_AC_MARKETS:
        print(f"\n  --- Computing rank AC(1) for {mkt_name} ---")
        for feat in ml_features:
            if feat not in df.columns:
                continue
            rac = compute_rank_autocorrelation(df, feat, date_col, id_col)
            speed = classify_feature_speed(feat)
            rank_ac_results.append({
                'Market': mkt_name,
                'Feature': feat,
                'Speed': speed,
                'Rank_AC1': round(rac, 4) if not np.isnan(rac) else np.nan,
            })
        print(f"    Computed rank AC for {len(ml_features)} features")

    # ------------------------------------------------------------------
    # Evaluate all strategies
    # ------------------------------------------------------------------
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
            print(f"    {strat_name:25s}  Gross={gsr:.2f}  Net@3x={sr3:.2f}  TO={to:.1%}")

elapsed = time.time() - t0
print(f"\n\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")

# ==============================================================================
# SAVE RESULTS
# ==============================================================================

results_df = pd.DataFrame(all_results)
outfile = 'scripts/concern_experiments/expAI_aggregation_vs_speed.csv'
results_df.to_csv(outfile, index=False)
print(f"\nSaved {len(results_df)} rows to {outfile}")

rank_ac_df = pd.DataFrame(rank_ac_results)


# ==============================================================================
# SUMMARY TABLES
# ==============================================================================

print("\n\n" + "=" * 100)
print("SUMMARY TABLE 1: OOS Net SR at 3x Costs — Key Strategies")
print("=" * 100)

oos_3x = results_df[(results_df['Period'] == 'OOS') & (results_df['CostMult'] == 3.0)]
if len(oos_3x) > 0:
    key_strats = ['MomentumSort', 'Ridge_All', 'Ridge_5', 'Ridge_10',
                  'Sort_ret_12_1', 'Sort_f_score']
    key_strats = [s for s in key_strats if s in oos_3x['Strategy'].unique()]
    key_data = oos_3x[oos_3x['Strategy'].isin(key_strats)]
    if len(key_data) > 0:
        pivot_sr = key_data.pivot_table(index='Market', columns='Strategy', values='Net_SR')
        pivot_sr = pivot_sr[[s for s in key_strats if s in pivot_sr.columns]]
        print(pivot_sr.round(2).to_string())
        print(f"\nMean:")
        print(pivot_sr.mean().round(3).to_string())


print("\n\n" + "=" * 100)
print("SUMMARY TABLE 2: OOS Turnover — Key Strategies")
print("=" * 100)

if len(oos_3x) > 0:
    key_data = oos_3x[oos_3x['Strategy'].isin(key_strats)]
    if len(key_data) > 0:
        pivot_to = key_data.pivot_table(index='Market', columns='Strategy', values='Turnover')
        pivot_to = pivot_to[[s for s in key_strats if s in pivot_to.columns]]
        print(pivot_to.round(3).to_string())
        print(f"\nMean:")
        print(pivot_to.mean().round(3).to_string())


print("\n\n" + "=" * 100)
print("SUMMARY TABLE 3: Composite Sort — OOS Net SR and Turnover vs N_features")
print("=" * 100)

if len(oos_3x) > 0:
    comp_strats = [s for s in oos_3x['Strategy'].unique() if s.startswith('CompositeSort_')]
    if len(comp_strats) > 0:
        comp_data = oos_3x[oos_3x['Strategy'].isin(comp_strats)]

        # Extract N from strategy name and sort
        comp_summary = comp_data.groupby('Strategy').agg(
            Mean_Net_SR=('Net_SR', 'mean'),
            Median_Net_SR=('Net_SR', 'median'),
            Mean_Turnover=('Turnover', 'mean'),
            Median_Turnover=('Turnover', 'median'),
            N_markets=('Market', 'nunique'),
        ).reset_index()

        # Extract N and sort
        comp_summary['N_features'] = comp_summary['Strategy'].str.extract(r'(\d+)').astype(int)
        comp_summary = comp_summary.sort_values('N_features')
        print(comp_summary[['N_features', 'Mean_Net_SR', 'Median_Net_SR',
                            'Mean_Turnover', 'Median_Turnover', 'N_markets']].to_string(index=False))


print("\n\n" + "=" * 100)
print("SUMMARY TABLE 4: Single-Feature Sorts — OOS Net SR and Turnover by Speed")
print("=" * 100)

if len(oos_3x) > 0:
    sort_strats = [s for s in oos_3x['Strategy'].unique() if s.startswith('Sort_')]
    if len(sort_strats) > 0:
        sort_data = oos_3x[oos_3x['Strategy'].isin(sort_strats)].copy()

        # Add speed category
        sort_data['Speed'] = sort_data['Strategy'].apply(
            lambda s: SINGLE_SORT_FEATURES.get(s.replace('Sort_', ''), {}).get('speed', 'UNKNOWN')
        )

        sort_summary = sort_data.groupby(['Strategy', 'Speed']).agg(
            Mean_Net_SR=('Net_SR', 'mean'),
            Mean_Turnover=('Turnover', 'mean'),
            N_markets=('Market', 'nunique'),
        ).reset_index().sort_values(['Speed', 'Strategy'])
        print(sort_summary.to_string(index=False))

        print("\nBy speed category:")
        speed_summary = sort_data.groupby('Speed').agg(
            Mean_Net_SR=('Net_SR', 'mean'),
            Mean_Turnover=('Turnover', 'mean'),
        )
        print(speed_summary.round(3).to_string())


print("\n\n" + "=" * 100)
print("SUMMARY TABLE 5: Rank Autocorrelation by Speed Category (THA, KOR, TWN avg)")
print("=" * 100)

if len(rank_ac_df) > 0:
    # Average across the 3 markets
    rank_ac_avg = rank_ac_df.groupby(['Feature', 'Speed'])['Rank_AC1'].mean().reset_index()
    rank_ac_avg.columns = ['Feature', 'Speed', 'Mean_Rank_AC1']

    # Merge with single-feature sort turnover (averaged across 3 markets)
    sort_turnover_data = []
    for feat_name in SINGLE_SORT_FEATURES.keys():
        strat = f'Sort_{feat_name}'
        feat_oos = oos_3x[(oos_3x['Strategy'] == strat) &
                          (oos_3x['Market'].isin(RANK_AC_MARKETS))]
        if len(feat_oos) > 0:
            sort_turnover_data.append({
                'Feature': feat_name,
                'Sort_Turnover': feat_oos['Turnover'].mean(),
            })
    sort_to_df = pd.DataFrame(sort_turnover_data)

    if len(sort_to_df) > 0:
        merged = rank_ac_avg.merge(sort_to_df, on='Feature', how='inner')
        merged = merged.sort_values('Mean_Rank_AC1', ascending=False)
        print("\nFeatures with sort turnover (scatter plot data):")
        print(merged.round(4).to_string(index=False))

    # Overall by speed
    print("\nOverall rank AC by speed category:")
    speed_ac = rank_ac_avg.groupby('Speed')['Mean_Rank_AC1'].agg(['mean', 'median', 'count'])
    print(speed_ac.round(4).to_string())

    # Highlight specific features
    print("\nSpecific features:")
    for feat in ['ret_12_1', 'f_score', 'roa', 'rvol_21d', 'be_me']:
        row = rank_ac_avg[rank_ac_avg['Feature'] == feat]
        if len(row) > 0:
            ac_val = row['Mean_Rank_AC1'].values[0]
            spd = row['Speed'].values[0]
            print(f"  {feat:20s}  Speed={spd:6s}  Rank_AC1={ac_val:.4f}")


print("\n\n" + "=" * 100)
print("SUMMARY TABLE 6: Sort vs Ridge at Each Dimensionality (Aggregation Effect)")
print("=" * 100)

if len(oos_3x) > 0:
    # For each N, compare CompositeSort_N vs Ridge_N
    print("\nNet SR @ 3x: CompositeSort_N vs Ridge_N (mean across markets)")
    print(f"{'N':>5s}  {'Sort_SR':>8s}  {'Ridge_SR':>9s}  {'Delta':>7s}  {'Sort_TO':>8s}  {'Ridge_TO':>9s}")
    print("-" * 55)
    for N in COMPOSITE_N_VALUES + [999]:  # 999 = placeholder for "all"
        sort_name = f"CompositeSort_{N}" if N != 999 else None
        ridge_name = f"Ridge_{N}" if N != 999 else 'Ridge_All'
        # Find the actual CompositeSort_all name
        if N == 999:
            comp_strats = [s for s in oos_3x['Strategy'].unique() if s.startswith('CompositeSort_')]
            comp_ns = [(s, int(s.replace('CompositeSort_', ''))) for s in comp_strats]
            if comp_ns:
                sort_name = max(comp_ns, key=lambda x: x[1])[0]
                N_label = 'all'
            else:
                continue
        else:
            N_label = str(N)
            if sort_name not in oos_3x['Strategy'].unique():
                continue
        sort_data = oos_3x[oos_3x['Strategy'] == sort_name]
        ridge_data = oos_3x[oos_3x['Strategy'] == ridge_name]
        if len(sort_data) > 0 and len(ridge_data) > 0:
            s_sr = sort_data['Net_SR'].mean()
            r_sr = ridge_data['Net_SR'].mean()
            s_to = sort_data['Turnover'].mean()
            r_to = ridge_data['Turnover'].mean()
            print(f"{N_label:>5s}  {s_sr:8.3f}  {r_sr:9.3f}  {r_sr-s_sr:+7.3f}  {s_to:8.3f}  {r_to:9.3f}")

    print("\nKey insight: If Ridge consistently has HIGHER turnover than CompositeSort")
    print("at the SAME dimensionality, then the regression aggregation method itself")
    print("creates instability beyond what feature noise alone would explain.")

print("\nDone!")
