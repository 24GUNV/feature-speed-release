"""
Experiment DA: FSI Diagnostic + Ridge_Slow + EMA + Adaptive Feature Selection

Tests the positive-result candidates that combine slow features with EMA smoothing,
plus computes the Feature Speed Index (FSI) as an ex-ante diagnostic.

Strategies (15 markets, OOS 2021-2023, 5 cost levels):

  1. MomentumSort           -- ret_12_1 sort, 20% buffer (baseline)
  2. Ridge_Slow             -- Keyword 57 slow features, no EMA, 20% buffer
  3. Ridge_Slow_EMA_{alpha} -- Keyword 57, fixed EMA grid (8 alphas), 20% buffer
  4. Ridge_Slow_WF_EMA      -- Keyword 57, walk-forward alpha selection, 20% buffer
  5. FSI_AC_{threshold}     -- Market-adaptive (3 AC thresholds), no EMA, 20% buffer
  6. Ridge_Slow_WideBuf     -- Keyword 57, no EMA, 40% buffer
  7. Ridge_Slow_WB_EMA      -- Keyword 57, WF-EMA, 40% buffer

FSI diagnostic: For each (market, feature), saves rank_AC, mean |coef|, speed_class.
Aggregate FSI = sum(|coef_j| * (1 - AC_j)) / sum(|coef_j|)

Architecture: Single walk-forward pass per market. Train Ridge ONCE per month,
compute predictions on each feature subset, apply EMA at each alpha, do WF selection.

Output:
  - scripts/concern_experiments/expDA_fsi_adaptive.csv
  - scripts/concern_experiments/expDA_fsi_diagnostic.csv
"""

import sys
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, 'public/src')
sys.path.insert(0, 'scripts/concern_experiments')

import numpy as np
import pandas as pd
import warnings
import time
from collections import Counter
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy import stats

from data_loader import load_equity_data, lag_features
from market_config import MARKETS, COST_MULTS, BUFFER_PCT, MIN_TRAIN_MONTHS

np.random.seed(42)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# EMA grid biased toward higher alphas (slow features already stable)
EMA_ALPHAS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

# FSI AC thresholds for market-adaptive feature selection (reduced for speed)
FSI_AC_THRESHOLDS = [0.65, 0.75, 0.85]
FSI_MIN_FEATURES = 15

# Wide buffer
WIDE_BUFFER_PCT = 0.40

# Walk-forward EMA selection
IS_EVAL_MONTHS = 24
COST_MULT_FOR_SELECTION = 3.0
WF_REEVAL_INTERVAL = 3  # Only re-search alpha every 3 OOS months

# Rank AC recompute interval
AC_RECOMPUTE_INTERVAL = 12
AC_LOOKBACK = 60


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


# ==============================================================================
# HELPER FUNCTIONS
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


def portfolio_from_ranks_vec(sids, ranks, prev_long, top_threshold, buffer, nq):
    """Vectorized portfolio construction with buffer."""
    prev_ids = np.array(list(prev_long), dtype=sids.dtype) if prev_long else np.array([], dtype=sids.dtype)
    in_prev = np.isin(sids, prev_ids)
    keep = in_prev & (ranks >= top_threshold - buffer)
    enter = ~in_prev & (ranks >= top_threshold + buffer)
    selected = keep | enter
    new_long = set(sids[selected])
    if len(new_long) == 0:
        n_top = max(1, len(sids) // nq)
        top_idx = np.argsort(ranks)[-n_top:]
        new_long = set(sids[top_idx])
    return new_long


def apply_ema_vec(sids, raw_ranks, ema_state, prev_month_stocks, alpha):
    """Vectorized EMA application."""
    smoothed = np.empty_like(raw_ranks)
    for i, (sid, rr) in enumerate(zip(sids, raw_ranks)):
        if sid in ema_state and sid in prev_month_stocks:
            smoothed[i] = alpha * rr + (1 - alpha) * ema_state[sid]
        else:
            smoothed[i] = rr
    return dict(zip(sids, smoothed))


def compute_rank_ac_per_feature(train_data, features, date_col, id_col):
    """
    Compute rank autocorrelation for each feature using Spearman correlation
    of cross-sectional ranks between consecutive months.
    """
    dates = sorted(train_data[date_col].unique())
    n_features = len(features)
    ac_sum = np.zeros(n_features)
    ac_count = np.zeros(n_features)

    for i in range(len(dates) - 1):
        d_t = dates[i]
        d_t1 = dates[i + 1]

        df_t = train_data[train_data[date_col] == d_t][[id_col] + features].drop_duplicates(subset=id_col).set_index(id_col)
        df_t1 = train_data[train_data[date_col] == d_t1][[id_col] + features].drop_duplicates(subset=id_col).set_index(id_col)

        common_ids = df_t.index.intersection(df_t1.index)
        if len(common_ids) < 10:
            continue

        vals_t = df_t.loc[common_ids].values
        vals_t1 = df_t1.loc[common_ids].values

        for j in range(n_features):
            col_t = vals_t[:, j]
            col_t1 = vals_t1[:, j]
            valid = ~(np.isnan(col_t) | np.isnan(col_t1))
            if valid.sum() < 10:
                continue
            rho, _ = stats.spearmanr(col_t[valid], col_t1[valid])
            if not np.isnan(rho):
                ac_sum[j] += rho
                ac_count[j] += 1

    ac_avg = np.where(ac_count > 0, ac_sum / ac_count, 0.5)
    ac_avg = np.clip(ac_avg, 0.05, 1.0)
    return ac_avg


# ==============================================================================
# FACTOR SORT
# ==============================================================================

def run_factor_sort(df, sort_features, date_col, id_col, target,
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
    prev_long = set()
    records = []

    for i, date in enumerate(dates):
        month = data[data[date_col] == date]
        if len(month) < nq:
            continue
        sids = month[id_col].values
        ranks = month['pct_rank'].values
        new_long = portfolio_from_ranks_vec(sids, ranks, prev_long, top_threshold, buffer, nq)

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
# EVALUATE HELPER
# ==============================================================================

def evaluate(result_df, cost_mult, period_label, period_mask,
             strategy_name, market, n_features=0):
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
        'N_features': n_features,
    }


# ==============================================================================
# SINGLE WALK-FORWARD ENGINE
# ==============================================================================

def evaluate_strategy_on_is(is_monthly_preds, is_eval_sorted_dates, feature_indices,
                            all_avail_features, ema_alpha, model, imputer, scaler,
                            top_threshold, buffer, nq, id_col, target,
                            global_med_1x, data, date_col):
    """
    Evaluate a single (feature_set, ema_alpha) combo on IS data.
    Returns IS net SR at 3x costs.
    """
    is_ema_state = {}
    is_prev_month_sids = set()
    is_prev_long = set()
    is_records = []

    for is_date in is_eval_sorted_dates:
        if is_date not in is_monthly_preds:
            continue
        is_sids, is_raw_preds, is_returns, is_month_df = is_monthly_preds[is_date]

        # If feature_indices is not None, retrain on subset
        # But for IS evaluation we use the full-feature model's predictions
        # and just re-rank -- this is an approximation for speed
        is_raw_ranks = pd.Series(is_raw_preds).rank(pct=True).values

        # Apply EMA
        smoothed = apply_ema_vec(is_sids, is_raw_ranks,
                                 is_ema_state, is_prev_month_sids, ema_alpha)
        is_ema_state = smoothed
        is_prev_month_sids = set(is_sids)

        smoothed_arr = np.array([smoothed[s] for s in is_sids])

        new_long = portfolio_from_ranks_vec(
            is_sids, smoothed_arr, is_prev_long, top_threshold, buffer, nq
        )

        held_ids = np.array(list(new_long), dtype=is_sids.dtype)
        held_mask = np.isin(is_sids, held_ids)
        gross_ret = is_returns[held_mask].mean() if held_mask.any() else 0.0

        if is_prev_long:
            entries = new_long - is_prev_long
            exits = is_prev_long - new_long
        else:
            entries = new_long
            exits = set()

        cost_drag = compute_cost_drag(
            is_month_df, entries, exits, new_long, id_col,
            COST_MULT_FOR_SELECTION, global_med_1x, is_prev_long
        )

        is_records.append({'gross_ret': gross_ret, 'cost_drag': cost_drag})
        is_prev_long = new_long

    if len(is_records) < 6:
        return -999.0

    is_df = pd.DataFrame(is_records)
    is_net_ret = is_df['gross_ret'] - is_df['cost_drag']
    return sharpe(is_net_ret)


def run_market(df, ml_features, slow_feats, date_col, id_col, target,
               cost_lookup_1x, nq, mkt_name):
    """
    Single walk-forward pass for one market. Returns:
      - strategies dict: {name: (result_df, n_features)}
      - fsi_diagnostic: list of dicts with per-feature FSI data
      - alpha_selections: list of dicts for WF alpha tracking
    """
    data = df.copy()
    data = data.sort_values([date_col, id_col]).reset_index(drop=True)

    # Pre-compute rank target
    data['_rank_target'] = data.groupby(date_col)[target].rank(pct=True)

    unique_dates = sorted(data[date_col].unique())
    n_dates = len(unique_dates)

    if n_dates < MIN_TRAIN_MONTHS + 12:
        return {}, [], []

    data = data.merge(cost_lookup_1x, on=[date_col, id_col], how='left')
    global_med_1x = cost_lookup_1x['cost_bps_1x'].median()
    data['cost_bps_1x'] = data['cost_bps_1x'].fillna(global_med_1x)

    # Validate features
    avail_slow = [f for f in slow_feats if f in data.columns]
    feat_coverage_slow = data[avail_slow].notna().mean()
    avail_slow = feat_coverage_slow[feat_coverage_slow > 0.1].index.tolist()

    avail_all = [f for f in ml_features if f in data.columns]
    feat_coverage_all = data[avail_all].notna().mean()
    avail_all = feat_coverage_all[feat_coverage_all > 0.1].index.tolist()

    if len(avail_slow) < 5:
        print(f"    WARNING: Only {len(avail_slow)} slow features available, skipping")
        return {}, [], []

    top_threshold = 1.0 - 1.0 / nq
    buffer_std = BUFFER_PCT * (1.0 / nq)
    buffer_wide = WIDE_BUFFER_PCT * (1.0 / nq)

    # ======================================================================
    # State tracking for all strategies
    # ======================================================================

    # Strategy 2: Ridge_Slow (no EMA, standard buffer)
    slow_prev_long = set()
    slow_records = []

    # Strategy 3: Ridge_Slow_EMA_{alpha} (fixed EMA, standard buffer)
    slow_ema_states = {a: {} for a in EMA_ALPHAS}
    slow_ema_prev_long = {a: set() for a in EMA_ALPHAS}
    slow_ema_records = {a: [] for a in EMA_ALPHAS}

    # Strategy 4: Ridge_Slow_WF_EMA (walk-forward alpha, standard buffer)
    slow_wf_ema_state = {}
    slow_wf_prev_long = set()
    slow_wf_records = []
    slow_wf_alpha_selections = []
    wf_oos_counter = 0  # Count OOS months since last alpha re-evaluation
    cached_best_alpha = 0.50  # Cached best alpha between re-evaluations

    # Strategy 5: FSI_AC_{threshold} (no EMA, standard buffer)
    fsi_prev_long = {t: set() for t in FSI_AC_THRESHOLDS}
    fsi_records = {t: [] for t in FSI_AC_THRESHOLDS}
    fsi_n_features = {t: 0 for t in FSI_AC_THRESHOLDS}
    # Cache FSI models (retrain only every 12 months)
    fsi_cached_models = {}  # threshold -> (model, imputer, scaler, features)

    # Strategy 6: Ridge_Slow_WideBuf (no EMA, wide buffer)
    slow_wb_prev_long = set()
    slow_wb_records = []

    # Strategy 7: Ridge_Slow_WB_EMA (WF-EMA, wide buffer)
    slow_wb_ema_state = {}
    slow_wb_ema_prev_long = set()
    slow_wb_ema_records = []

    prev_month_stocks = set()

    # FSI diagnostic data
    fsi_diagnostic = []
    cached_ac = None
    cached_ac_features = None
    last_ac_compute = -999

    # Coefficient accumulator for FSI
    coef_accum = {}  # feature -> list of |coef| values

    for t_idx in range(MIN_TRAIN_MONTHS, n_dates):
        train_dates = unique_dates[:t_idx]
        test_date = unique_dates[t_idx]

        train_data = data[data[date_col].isin(train_dates)]
        test_data = data[data[date_col] == test_date].copy()

        if len(test_data) < nq or len(train_data) < 100:
            continue

        # ==============================================================
        # Train Ridge on SLOW features ONCE
        # ==============================================================
        X_train = train_data[avail_slow].copy()
        y_train = train_data['_rank_target'].values

        imputer = SimpleImputer(strategy='median')
        X_train_imp = imputer.fit_transform(X_train)

        # Track which features survived imputation (not all-NaN)
        survived_mask = ~np.isnan(imputer.statistics_)
        survived_slow = [f for f, s in zip(avail_slow, survived_mask) if s]
        # Note: fit_transform already drops all-NaN columns, no need to re-index

        mask = ~np.isnan(y_train)
        if mask.sum() < 50:
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp[mask])

        model = Ridge(alpha=100, random_state=42)
        model.fit(X_train_scaled, y_train[mask])

        # Extract Ridge |coef| for FSI diagnostic
        for j, feat in enumerate(survived_slow):
            if feat not in coef_accum:
                coef_accum[feat] = []
            coef_accum[feat].append(abs(model.coef_[j]))

        # ==============================================================
        # OOS predictions with slow features
        # ==============================================================
        X_test = test_data[avail_slow].copy()
        X_test_imp = imputer.transform(X_test)
        # transform already drops same columns as fit_transform
        X_test_scaled = scaler.transform(X_test_imp)
        slow_preds = model.predict(X_test_scaled)

        test_data['slow_prediction'] = slow_preds
        test_data['slow_pct_rank'] = test_data['slow_prediction'].rank(pct=True)

        test_sids = test_data[id_col].values
        slow_raw_ranks = test_data['slow_pct_rank'].values
        test_returns = test_data[target].values
        current_month_stocks = set(test_sids)

        # ==============================================================
        # Compute rank AC periodically for FSI-based feature selection
        # ==============================================================
        if cached_ac is None or (t_idx - last_ac_compute) >= AC_RECOMPUTE_INTERVAL:
            ac_lookback_dates = train_dates[-AC_LOOKBACK:]
            ac_data = train_data[train_data[date_col].isin(ac_lookback_dates)]
            cached_ac = compute_rank_ac_per_feature(
                ac_data, avail_all, date_col, id_col
            )
            cached_ac_features = avail_all
            last_ac_compute = t_idx

        # ==============================================================
        # Strategy 2: Ridge_Slow (no EMA, standard buffer)
        # ==============================================================
        new_long = portfolio_from_ranks_vec(
            test_sids, slow_raw_ranks, slow_prev_long,
            top_threshold, buffer_std, nq
        )
        held_mask = np.isin(test_sids, np.array(list(new_long), dtype=test_sids.dtype))
        gross_ret = test_returns[held_mask].mean() if held_mask.any() else 0.0
        if slow_prev_long:
            entries = new_long - slow_prev_long
            exits = slow_prev_long - new_long
            turn = (len(entries) + len(exits)) / (2 * max(len(new_long), 1))
        else:
            entries = new_long
            exits = set()
            turn = np.nan
        cost_drags = {}
        for cm in COST_MULTS:
            cost_drags[cm] = compute_cost_drag(
                test_data, entries, exits, new_long, id_col,
                cm, global_med_1x, slow_prev_long
            )
        slow_records.append({
            'date': test_date, 'gross_ret': gross_ret, 'turnover': turn,
            'n_held': len(new_long),
            **{f'cost_drag_{cm}': cost_drags[cm] for cm in COST_MULTS},
        })
        slow_prev_long = new_long

        # ==============================================================
        # Strategy 3: Ridge_Slow_EMA_{alpha} (fixed EMA, standard buffer)
        # ==============================================================
        for alpha in EMA_ALPHAS:
            smoothed = apply_ema_vec(test_sids, slow_raw_ranks,
                                     slow_ema_states[alpha], prev_month_stocks, alpha)
            slow_ema_states[alpha] = smoothed
            smoothed_arr = np.array([smoothed[s] for s in test_sids])

            new_long = portfolio_from_ranks_vec(
                test_sids, smoothed_arr, slow_ema_prev_long[alpha],
                top_threshold, buffer_std, nq
            )
            held_ids = np.array(list(new_long), dtype=test_sids.dtype)
            held_mask = np.isin(test_sids, held_ids)
            gross_ret = test_returns[held_mask].mean() if held_mask.any() else 0.0

            prev = slow_ema_prev_long[alpha]
            if prev:
                entries = new_long - prev
                exits = prev - new_long
                turn = (len(entries) + len(exits)) / (2 * max(len(new_long), 1))
            else:
                entries = new_long
                exits = set()
                turn = np.nan

            cost_drags = {}
            for cm in COST_MULTS:
                cost_drags[cm] = compute_cost_drag(
                    test_data, entries, exits, new_long, id_col,
                    cm, global_med_1x, prev
                )
            slow_ema_records[alpha].append({
                'date': test_date, 'gross_ret': gross_ret, 'turnover': turn,
                'n_held': len(new_long),
                **{f'cost_drag_{cm}': cost_drags[cm] for cm in COST_MULTS},
            })
            slow_ema_prev_long[alpha] = new_long

        # ==============================================================
        # Strategy 4: Ridge_Slow_WF_EMA (walk-forward alpha selection)
        # Re-evaluate alpha every WF_REEVAL_INTERVAL OOS months
        # ==============================================================
        wf_oos_counter += 1
        if wf_oos_counter >= WF_REEVAL_INTERVAL or cached_best_alpha == 0.50:
            wf_oos_counter = 0
            is_eval_dates = train_dates[-IS_EVAL_MONTHS:]
            is_monthly_preds = {}
            for is_date in is_eval_dates:
                is_month = data[data[date_col] == is_date]
                if len(is_month) < nq:
                    continue
                X_is = is_month[avail_slow].copy()
                X_is_imp = imputer.transform(X_is)
                X_is_scaled = scaler.transform(X_is_imp)
                is_preds = model.predict(X_is_scaled)
                is_monthly_preds[is_date] = (
                    is_month[id_col].values,
                    is_preds,
                    is_month[target].values,
                    is_month
                )

            is_eval_sorted_dates = sorted(is_monthly_preds.keys())

            best_alpha = 0.50  # default for slow features
            best_is_sr = -999.0

            if len(is_eval_sorted_dates) >= 12:
                for candidate_alpha in EMA_ALPHAS:
                    is_sr = evaluate_strategy_on_is(
                        is_monthly_preds, is_eval_sorted_dates, None,
                        avail_slow, candidate_alpha, model, imputer, scaler,
                        top_threshold, buffer_std, nq, id_col, target,
                        global_med_1x, data, date_col
                    )
                    if is_sr > best_is_sr:
                        best_is_sr = is_sr
                        best_alpha = candidate_alpha
            cached_best_alpha = best_alpha
        else:
            best_alpha = cached_best_alpha

        slow_wf_alpha_selections.append({
            'date': test_date, 'selected_alpha': best_alpha
        })

        # Apply WF-selected alpha
        wf_smoothed = apply_ema_vec(test_sids, slow_raw_ranks,
                                     slow_wf_ema_state, prev_month_stocks, best_alpha)
        slow_wf_ema_state = wf_smoothed
        wf_smoothed_arr = np.array([wf_smoothed[s] for s in test_sids])

        new_long = portfolio_from_ranks_vec(
            test_sids, wf_smoothed_arr, slow_wf_prev_long,
            top_threshold, buffer_std, nq
        )
        held_ids = np.array(list(new_long), dtype=test_sids.dtype)
        held_mask = np.isin(test_sids, held_ids)
        gross_ret = test_returns[held_mask].mean() if held_mask.any() else 0.0

        if slow_wf_prev_long:
            entries = new_long - slow_wf_prev_long
            exits = slow_wf_prev_long - new_long
            turn = (len(entries) + len(exits)) / (2 * max(len(new_long), 1))
        else:
            entries = new_long
            exits = set()
            turn = np.nan

        cost_drags = {}
        for cm in COST_MULTS:
            cost_drags[cm] = compute_cost_drag(
                test_data, entries, exits, new_long, id_col,
                cm, global_med_1x, slow_wf_prev_long
            )
        slow_wf_records.append({
            'date': test_date, 'gross_ret': gross_ret, 'turnover': turn,
            'n_held': len(new_long),
            **{f'cost_drag_{cm}': cost_drags[cm] for cm in COST_MULTS},
        })
        slow_wf_prev_long = new_long

        # ==============================================================
        # Strategy 5: FSI_AC_{threshold} (data-driven feature selection)
        # Models retrained only when AC is recomputed (every 12 months)
        # ==============================================================
        for thresh in FSI_AC_THRESHOLDS:
            fsi_feats = [f for j, f in enumerate(cached_ac_features)
                         if cached_ac[j] >= thresh and f in avail_all]
            if len(fsi_feats) < FSI_MIN_FEATURES:
                sorted_idx = np.argsort(cached_ac)[::-1]
                fsi_feats = [cached_ac_features[j] for j in sorted_idx[:FSI_MIN_FEATURES]
                             if cached_ac_features[j] in avail_all]

            fsi_n_features[thresh] = len(fsi_feats)

            # Retrain only when AC was recomputed (last_ac_compute == t_idx)
            if thresh not in fsi_cached_models or last_ac_compute == t_idx:
                X_train_fsi = train_data[fsi_feats].copy()
                imp_fsi = SimpleImputer(strategy='median')
                X_train_fsi_imp = imp_fsi.fit_transform(X_train_fsi)
                sc_fsi = StandardScaler()
                X_train_fsi_sc = sc_fsi.fit_transform(X_train_fsi_imp[mask])
                model_fsi = Ridge(alpha=100, random_state=42)
                model_fsi.fit(X_train_fsi_sc, y_train[mask])
                fsi_cached_models[thresh] = (model_fsi, imp_fsi, sc_fsi, fsi_feats)
            else:
                model_fsi, imp_fsi, sc_fsi, fsi_feats = fsi_cached_models[thresh]

            X_test_fsi = test_data[fsi_feats].copy()
            X_test_fsi_imp = imp_fsi.transform(X_test_fsi)
            X_test_fsi_sc = sc_fsi.transform(X_test_fsi_imp)
            fsi_preds = model_fsi.predict(X_test_fsi_sc)

            fsi_ranks = pd.Series(fsi_preds, index=test_data.index).rank(pct=True).values

            new_long = portfolio_from_ranks_vec(
                test_sids, fsi_ranks, fsi_prev_long[thresh],
                top_threshold, buffer_std, nq
            )
            held_ids = np.array(list(new_long), dtype=test_sids.dtype)
            held_mask = np.isin(test_sids, held_ids)
            gross_ret = test_returns[held_mask].mean() if held_mask.any() else 0.0

            prev = fsi_prev_long[thresh]
            if prev:
                entries = new_long - prev
                exits = prev - new_long
                turn = (len(entries) + len(exits)) / (2 * max(len(new_long), 1))
            else:
                entries = new_long
                exits = set()
                turn = np.nan

            cost_drags = {}
            for cm in COST_MULTS:
                cost_drags[cm] = compute_cost_drag(
                    test_data, entries, exits, new_long, id_col,
                    cm, global_med_1x, prev
                )
            fsi_records[thresh].append({
                'date': test_date, 'gross_ret': gross_ret, 'turnover': turn,
                'n_held': len(new_long),
                **{f'cost_drag_{cm}': cost_drags[cm] for cm in COST_MULTS},
            })
            fsi_prev_long[thresh] = new_long

        # ==============================================================
        # Strategy 6: Ridge_Slow_WideBuf (no EMA, wide buffer)
        # ==============================================================
        new_long = portfolio_from_ranks_vec(
            test_sids, slow_raw_ranks, slow_wb_prev_long,
            top_threshold, buffer_wide, nq
        )
        held_ids = np.array(list(new_long), dtype=test_sids.dtype)
        held_mask = np.isin(test_sids, held_ids)
        gross_ret = test_returns[held_mask].mean() if held_mask.any() else 0.0

        if slow_wb_prev_long:
            entries = new_long - slow_wb_prev_long
            exits = slow_wb_prev_long - new_long
            turn = (len(entries) + len(exits)) / (2 * max(len(new_long), 1))
        else:
            entries = new_long
            exits = set()
            turn = np.nan

        cost_drags = {}
        for cm in COST_MULTS:
            cost_drags[cm] = compute_cost_drag(
                test_data, entries, exits, new_long, id_col,
                cm, global_med_1x, slow_wb_prev_long
            )
        slow_wb_records.append({
            'date': test_date, 'gross_ret': gross_ret, 'turnover': turn,
            'n_held': len(new_long),
            **{f'cost_drag_{cm}': cost_drags[cm] for cm in COST_MULTS},
        })
        slow_wb_prev_long = new_long

        # ==============================================================
        # Strategy 7: Ridge_Slow_WB_EMA (WF-EMA, wide buffer)
        # ==============================================================
        # Reuse standard buffer's best alpha (same model, same IS data)
        best_wb_alpha = best_alpha

        wb_smoothed = apply_ema_vec(test_sids, slow_raw_ranks,
                                     slow_wb_ema_state, prev_month_stocks, best_wb_alpha)
        slow_wb_ema_state = wb_smoothed
        wb_smoothed_arr = np.array([wb_smoothed[s] for s in test_sids])

        new_long = portfolio_from_ranks_vec(
            test_sids, wb_smoothed_arr, slow_wb_ema_prev_long,
            top_threshold, buffer_wide, nq
        )
        held_ids = np.array(list(new_long), dtype=test_sids.dtype)
        held_mask = np.isin(test_sids, held_ids)
        gross_ret = test_returns[held_mask].mean() if held_mask.any() else 0.0

        if slow_wb_ema_prev_long:
            entries = new_long - slow_wb_ema_prev_long
            exits = slow_wb_ema_prev_long - new_long
            turn = (len(entries) + len(exits)) / (2 * max(len(new_long), 1))
        else:
            entries = new_long
            exits = set()
            turn = np.nan

        cost_drags = {}
        for cm in COST_MULTS:
            cost_drags[cm] = compute_cost_drag(
                test_data, entries, exits, new_long, id_col,
                cm, global_med_1x, slow_wb_ema_prev_long
            )
        slow_wb_ema_records.append({
            'date': test_date, 'gross_ret': gross_ret, 'turnover': turn,
            'n_held': len(new_long),
            **{f'cost_drag_{cm}': cost_drags[cm] for cm in COST_MULTS},
        })
        slow_wb_ema_prev_long = new_long

        prev_month_stocks = current_month_stocks

    # ==================================================================
    # Assemble results
    # ==================================================================
    strategies = {}

    if slow_records:
        strategies['Ridge_Slow'] = (
            pd.DataFrame(slow_records).set_index('date'),
            len(avail_slow)
        )

    for alpha in EMA_ALPHAS:
        if slow_ema_records[alpha]:
            strategies[f'Ridge_Slow_EMA_{alpha}'] = (
                pd.DataFrame(slow_ema_records[alpha]).set_index('date'),
                len(avail_slow)
            )

    if slow_wf_records:
        strategies['Ridge_Slow_WF_EMA'] = (
            pd.DataFrame(slow_wf_records).set_index('date'),
            len(avail_slow)
        )

    for thresh in FSI_AC_THRESHOLDS:
        if fsi_records[thresh]:
            strategies[f'FSI_AC_{thresh}'] = (
                pd.DataFrame(fsi_records[thresh]).set_index('date'),
                fsi_n_features[thresh]
            )

    if slow_wb_records:
        strategies['Ridge_Slow_WideBuf'] = (
            pd.DataFrame(slow_wb_records).set_index('date'),
            len(avail_slow)
        )

    if slow_wb_ema_records:
        strategies['Ridge_Slow_WB_EMA'] = (
            pd.DataFrame(slow_wb_ema_records).set_index('date'),
            len(avail_slow)
        )

    # ==================================================================
    # FSI diagnostic data
    # ==================================================================
    if cached_ac is not None:
        ac_dict = dict(zip(cached_ac_features, cached_ac))
        for feat in avail_all:
            ac_val = ac_dict.get(feat, 0.5)
            mean_coef = np.mean(coef_accum[feat]) if feat in coef_accum else 0.0
            speed_class = classify_feature_speed(feat)
            fsi_diagnostic.append({
                'market': mkt_name,
                'feature': feat,
                'rank_ac': round(ac_val, 4),
                'mean_abs_coef': round(mean_coef, 6),
                'speed_class': speed_class,
            })

    return strategies, fsi_diagnostic, slow_wf_alpha_selections


# ==============================================================================
# MAIN
# ==============================================================================

print("=" * 100)
print("EXPERIMENT DA: FSI Diagnostic + Ridge_Slow + EMA + Adaptive Feature Selection")
print(f"  EMA alphas: {EMA_ALPHAS}")
print(f"  FSI AC thresholds: {FSI_AC_THRESHOLDS}")
print(f"  Wide buffer: {WIDE_BUFFER_PCT}")
print("  15 emerging markets")
print("=" * 100)
sys.stdout.flush()

all_results = []
all_fsi_diagnostic = []
all_alpha_selections = []
t0 = time.time()

for mkt_name, cfg in MARKETS.items():
    mkt_t0 = time.time()
    print(f"\n{'#'*80}")
    print(f"# MARKET: {mkt_name}")
    print(f"{'#'*80}")
    sys.stdout.flush()

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

    # Fix: lag me to avoid look-ahead bias
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

    print(f"  Features: {len(ml_features)} total, {len(slow_feats)} SLOW")

    df[date_col] = pd.to_datetime(df[date_col])
    nq = cfg['nq']

    # IS/OOS split
    unique_dates = sorted(df[date_col].unique())
    n_total = len(unique_dates)
    is_split_idx = max(int(n_total * 0.8), MIN_TRAIN_MONTHS)
    is_end_date = unique_dates[min(is_split_idx - 1, n_total - 1)]

    print(f"  IS ends: {is_end_date}, OOS months: {n_total - is_split_idx}")

    # 1. MomentumSort
    print(f"  --- MomentumSort ---")
    mom_result = run_factor_sort(
        df, ['ret_12_1'], date_col, id_col, target, cost_lookup_1x,
        nq=nq, buffer_pct=BUFFER_PCT
    )

    # 2-8. All ML strategies in single pass
    print(f"  --- Ridge_Slow + EMA + FSI (single pass) ---")
    sys.stdout.flush()
    strategies, fsi_diag, alpha_sel = run_market(
        df, ml_features, slow_feats, date_col, id_col, target,
        cost_lookup_1x, nq, mkt_name
    )

    all_fsi_diagnostic.extend(fsi_diag)
    for sel in alpha_sel:
        sel['Market'] = mkt_name
    all_alpha_selections.extend(alpha_sel)

    # Add MomentumSort
    all_strategies = {'MomentumSort': (mom_result, 1)}
    all_strategies.update(strategies)

    # Evaluate
    def make_masks(result_df):
        idx = result_df.index
        return [('IS', idx <= is_end_date), ('OOS', idx > is_end_date),
                ('Full', pd.Series(True, index=idx))]

    for strat_name, (result, n_feat) in all_strategies.items():
        if result is None:
            continue
        for period_label, mask_item in make_masks(result):
            for cm in COST_MULTS:
                metrics = evaluate(result, cm, period_label, mask_item,
                                   strat_name, mkt_name, n_features=n_feat)
                if metrics:
                    all_results.append(metrics)

    # Print OOS summary
    mkt_elapsed = time.time() - mkt_t0
    print(f"\n  OOS Summary (Net SR @ 3x costs) [{mkt_elapsed:.0f}s]:")
    strat_order = ['MomentumSort', 'Ridge_Slow'] + \
                  [f'Ridge_Slow_EMA_{a}' for a in EMA_ALPHAS] + \
                  ['Ridge_Slow_WF_EMA'] + \
                  [f'FSI_AC_{t}' for t in FSI_AC_THRESHOLDS] + \
                  ['Ridge_Slow_WideBuf', 'Ridge_Slow_WB_EMA']
    for strat_name in strat_order:
        if strat_name not in all_strategies:
            continue
        result, n_feat = all_strategies[strat_name]
        if result is None:
            continue
        oos_mask = result.index > is_end_date
        oos = result[oos_mask]
        if len(oos) > 0:
            sr3 = sharpe(oos['gross_ret'] - oos['cost_drag_3.0'])
            gsr = sharpe(oos['gross_ret'])
            to = oos['turnover'].mean()
            print(f"    {strat_name:30s}  Gross={gsr:.2f}  Net@3x={sr3:.2f}  TO={to:.1%}  N={n_feat}")

    sys.stdout.flush()


elapsed = time.time() - t0
print(f"\n\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")


# ==============================================================================
# SAVE RESULTS
# ==============================================================================

results_df = pd.DataFrame(all_results)
outfile = 'scripts/concern_experiments/expDA_fsi_adaptive.csv'
results_df.to_csv(outfile, index=False)
print(f"\nSaved {len(results_df)} rows to {outfile}")

# FSI diagnostic
fsi_df = pd.DataFrame(all_fsi_diagnostic)
fsi_outfile = 'scripts/concern_experiments/expDA_fsi_diagnostic.csv'
fsi_df.to_csv(fsi_outfile, index=False)
print(f"Saved {len(fsi_df)} rows to {fsi_outfile}")


# ==============================================================================
# SUMMARY 1: Key Strategies Comparison
# ==============================================================================

print("\n\n" + "=" * 100)
print("SUMMARY 1: Key Strategies -- OOS Net SR at 3x Costs")
print("=" * 100)

oos_3x = results_df[(results_df['Period'] == 'OOS') & (results_df['CostMult'] == 3.0)]

if len(oos_3x) > 0:
    key_strats = ['MomentumSort', 'Ridge_Slow', 'Ridge_Slow_WF_EMA',
                  'Ridge_Slow_WideBuf', 'Ridge_Slow_WB_EMA']
    key_data = oos_3x[oos_3x['Strategy'].isin(key_strats)]

    if len(key_data) > 0:
        pivot = key_data.pivot_table(index='Market', columns='Strategy', values='Net_SR')
        pivot = pivot[[s for s in key_strats if s in pivot.columns]]
        print(pivot.round(3).to_string())
        print(f"\nMean:")
        print(pivot.mean().round(3).to_string())

        # Turnover
        to_pivot = key_data.pivot_table(index='Market', columns='Strategy', values='Turnover')
        to_pivot = to_pivot[[s for s in key_strats if s in to_pivot.columns]]
        print(f"\nMean Turnover:")
        print(to_pivot.mean().round(3).to_string())


# ==============================================================================
# SUMMARY 2: Ridge_Slow_WF_EMA vs MomentumSort (per market)
# ==============================================================================

print("\n\n" + "=" * 100)
print("SUMMARY 2: Ridge_Slow_WF_EMA vs MomentumSort (OOS at 3x costs)")
print("=" * 100)

if len(oos_3x) > 0:
    wf_data = oos_3x[oos_3x['Strategy'] == 'Ridge_Slow_WF_EMA']
    mom_data = oos_3x[oos_3x['Strategy'] == 'MomentumSort']

    if len(wf_data) > 0 and len(mom_data) > 0:
        wf_by_mkt = wf_data.set_index('Market')['Net_SR']
        mom_by_mkt = mom_data.set_index('Market')['Net_SR']
        common_mkts = sorted(set(wf_by_mkt.index) & set(mom_by_mkt.index))

        print(f"\n{'Market':>6s}  {'Mom_SR':>8s}  {'SlowWF_SR':>10s}  {'Delta':>8s}  {'SlowWF_TO':>10s}")
        print("-" * 50)

        wf_wins = 0
        for mkt in common_mkts:
            m_sr = mom_by_mkt[mkt]
            w_sr = wf_by_mkt[mkt]
            delta = w_sr - m_sr
            wf_to_row = wf_data[wf_data['Market'] == mkt]
            wf_to = wf_to_row['Turnover'].values[0] if len(wf_to_row) > 0 else np.nan
            if w_sr > m_sr:
                wf_wins += 1
            print(f"{mkt:>6s}  {m_sr:8.3f}  {w_sr:10.3f}  {delta:+8.3f}  {wf_to:10.3f}")

        print("-" * 50)
        avg_mom = mom_by_mkt[common_mkts].mean()
        avg_wf = wf_by_mkt[common_mkts].mean()
        print(f"{'Mean':>6s}  {avg_mom:8.3f}  {avg_wf:10.3f}  {avg_wf-avg_mom:+8.3f}")
        print(f"\n  Win rate: {wf_wins}/{len(common_mkts)} ({wf_wins/len(common_mkts)*100:.0f}%)")


# ==============================================================================
# SUMMARY 3: EMA Alpha Curve for Slow Features
# ==============================================================================

print("\n\n" + "=" * 100)
print("SUMMARY 3: EMA Alpha Curve for Slow Features (OOS @ 3x)")
print("=" * 100)

if len(oos_3x) > 0:
    print(f"\n{'Alpha':>8s}  {'Mean_SR':>8s}  {'Med_SR':>8s}  {'Wins>0':>7s}  {'Mean_TO':>8s}")
    print("-" * 50)

    for alpha in EMA_ALPHAS:
        strat = f'Ridge_Slow_EMA_{alpha}'
        strat_data = oos_3x[oos_3x['Strategy'] == strat]
        if len(strat_data) == 0:
            continue
        mean_sr = strat_data['Net_SR'].mean()
        med_sr = strat_data['Net_SR'].median()
        wins = (strat_data['Net_SR'] > 0).sum()
        n_mkts = len(strat_data)
        mean_to = strat_data['Turnover'].mean()
        print(f"{alpha:8.2f}  {mean_sr:8.3f}  {med_sr:8.3f}  {wins:3d}/{n_mkts:<3d}  {mean_to:8.3f}")

    print("-" * 50)
    for strat in ['Ridge_Slow', 'Ridge_Slow_WF_EMA', 'MomentumSort']:
        strat_data = oos_3x[oos_3x['Strategy'] == strat]
        if len(strat_data) == 0:
            continue
        mean_sr = strat_data['Net_SR'].mean()
        med_sr = strat_data['Net_SR'].median()
        wins = (strat_data['Net_SR'] > 0).sum()
        n_mkts = len(strat_data)
        mean_to = strat_data['Turnover'].mean()
        print(f"{'WF' if 'WF' in strat else 'N/A':>8s}  {mean_sr:8.3f}  {med_sr:8.3f}  {wins:3d}/{n_mkts:<3d}  {mean_to:8.3f}  [{strat}]")


# ==============================================================================
# SUMMARY 4: FSI AC Thresholds
# ==============================================================================

print("\n\n" + "=" * 100)
print("SUMMARY 4: FSI AC Threshold Comparison (OOS @ 3x)")
print("=" * 100)

if len(oos_3x) > 0:
    print(f"\n{'Threshold':>10s}  {'Mean_SR':>8s}  {'Mean_TO':>8s}  {'Mean_N_feat':>12s}")
    print("-" * 50)

    for thresh in FSI_AC_THRESHOLDS:
        strat = f'FSI_AC_{thresh}'
        strat_data = oos_3x[oos_3x['Strategy'] == strat]
        if len(strat_data) == 0:
            continue
        mean_sr = strat_data['Net_SR'].mean()
        mean_to = strat_data['Turnover'].mean()
        mean_n = strat_data['N_features'].mean()
        print(f"{thresh:10.2f}  {mean_sr:8.3f}  {mean_to:8.3f}  {mean_n:12.0f}")


# ==============================================================================
# SUMMARY 5: FSI Aggregate Values
# ==============================================================================

print("\n\n" + "=" * 100)
print("SUMMARY 5: FSI Values per Market (weighted avg feature speed)")
print("=" * 100)

if len(fsi_df) > 0:
    # Compute FSI = sum(|coef_j| * (1 - AC_j)) / sum(|coef_j|) for slow features
    for mkt in sorted(fsi_df['market'].unique()):
        mkt_diag = fsi_df[fsi_df['market'] == mkt]
        if len(mkt_diag) == 0 or mkt_diag['mean_abs_coef'].sum() == 0:
            continue
        # All features
        w = mkt_diag['mean_abs_coef'].values
        ac = mkt_diag['rank_ac'].values
        fsi_all = np.sum(w * (1 - ac)) / np.sum(w) if np.sum(w) > 0 else np.nan

        # Slow only
        slow_diag = mkt_diag[mkt_diag['speed_class'] == 'SLOW']
        w_slow = slow_diag['mean_abs_coef'].values
        ac_slow = slow_diag['rank_ac'].values
        fsi_slow = np.sum(w_slow * (1 - ac_slow)) / np.sum(w_slow) if np.sum(w_slow) > 0 else np.nan

        print(f"  {mkt:>5s}  FSI_all={fsi_all:.3f}  FSI_slow={fsi_slow:.3f}  "
              f"N_features={len(mkt_diag)}  N_slow={len(slow_diag)}")


# ==============================================================================
# KEY TAKEAWAY
# ==============================================================================

print("\n\n" + "=" * 100)
print("KEY TAKEAWAY")
print("=" * 100)

if len(oos_3x) > 0:
    key_strats = ['MomentumSort', 'Ridge_Slow', 'Ridge_Slow_WF_EMA',
                  'Ridge_Slow_WideBuf', 'Ridge_Slow_WB_EMA']
    for strat in key_strats:
        strat_data = oos_3x[oos_3x['Strategy'] == strat]
        if len(strat_data) > 0:
            mean_sr = strat_data['Net_SR'].mean()
            mean_to = strat_data['Turnover'].mean()
            print(f"  {strat:30s}  SR={mean_sr:.3f}  TO={mean_to:.1%}")

    # Best fixed EMA alpha
    best_fixed_sr = -999
    best_fixed_alpha = None
    for alpha in EMA_ALPHAS:
        strat = f'Ridge_Slow_EMA_{alpha}'
        strat_data = oos_3x[oos_3x['Strategy'] == strat]
        if len(strat_data) > 0:
            sr = strat_data['Net_SR'].mean()
            if sr > best_fixed_sr:
                best_fixed_sr = sr
                best_fixed_alpha = alpha
    if best_fixed_alpha is not None:
        print(f"  {'Best fixed EMA (a=' + str(best_fixed_alpha) + ')':30s}  SR={best_fixed_sr:.3f}")

print("\nDone!")
