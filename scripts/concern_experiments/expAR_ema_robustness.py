"""
Experiment AR: EMA Robustness -- Fine Grid + Walk-Forward Alpha Selection

Part 1: Fine EMA Grid
  Run EMA prediction smoothing with 15 alpha values across all 15 EM markets.
  Extends Exp AQ which only tested [0.3, 0.5, 0.7, 0.9, 1.0].
  Alphas: [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
           0.60, 0.70, 0.80, 0.90, 1.00]

Part 2: Walk-Forward Alpha Selection
  In each expanding window, use recent in-sample data (last 24 training months)
  scored by the current-window model to pick the best EMA alpha at 3x costs.
  Then apply that alpha to OOS predictions. This uses no future OOS months, but
  it is not a nested out-of-fold validation.

Vectorized implementation: trains Ridge once per window, evaluates all
15 alphas in a single pass, plus walk-forward alpha selection.

Pipeline: lag=1 for both accounting and market fields because the JKP/GFD
accounting characteristics are already released with a four-month lag; buffer,
walk-forward, CS costs.
All 15 emerging markets.

Output: scripts/concern_experiments/expAR_ema_robustness.csv
"""

import sys
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, 'public/src')

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

sys.path.insert(0, 'scripts/concern_experiments')
from market_config import MARKETS, COST_MULTS, BUFFER_PCT, MIN_TRAIN_MONTHS, EMA_ALPHAS

np.random.seed(42)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Fine EMA alpha grid comes from market_config

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

EXCLUDE_COLS = ['Unnamed: 0', 'id', 'eom', 'excntry', 'gvkey', 'permno',
                'size_grp', 'me', 'market_equity', 'ret_1_0', 'date']


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


# ==============================================================================
# FACTOR SORT (vectorized)
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
# UNIFIED ML WALK-FORWARD: trains once, evaluates ALL alphas + WF selection
# ==============================================================================

def run_ml_walkforward_all_alphas(df, features, date_col, id_col, target,
                                   cost_lookup_1x, alpha_grid, nq=5,
                                   buffer_pct=0.20, min_train=60):
    """
    Single walk-forward pass that:
      1. Trains Ridge once per OOS month
      2. Applies EMA with EACH fixed alpha (Part 1)
      3. Does walk-forward alpha selection (Part 2)

    Returns dict of {strategy_name: (result_df, pred_stability)}
    plus alpha_selections list for WF.
    """
    data = df.copy()
    data = data.sort_values([date_col, id_col]).reset_index(drop=True)

    # Pre-compute rank target (cross-sectional percentile ranks)
    data['_rank_target'] = data.groupby(date_col)[target].rank(pct=True)

    unique_dates = sorted(data[date_col].unique())
    n_dates = len(unique_dates)

    if n_dates < min_train + 12:
        return {}, []

    data = data.merge(cost_lookup_1x, on=[date_col, id_col], how='left')
    global_med_1x = cost_lookup_1x['cost_bps_1x'].median()
    data['cost_bps_1x'] = data['cost_bps_1x'].fillna(global_med_1x)

    avail_features = [f for f in features if f in data.columns]
    feat_coverage = data[avail_features].notna().mean()
    avail_features = feat_coverage[feat_coverage > 0.1].index.tolist()

    if len(avail_features) == 0:
        return {}, []

    top_threshold = 1.0 - 1.0 / nq
    buffer = buffer_pct * (1.0 / nq)

    IS_EVAL_MONTHS = 24
    COST_MULT_FOR_SELECTION = 3.0

    # Per-alpha state
    ema_states = {a: {} for a in alpha_grid}
    prev_long_per_alpha = {a: set() for a in alpha_grid}
    records_per_alpha = {a: [] for a in alpha_grid}
    prev_smoothed_per_alpha = {a: {} for a in alpha_grid}
    rank_corrs_per_alpha = {a: [] for a in alpha_grid}

    # WF state
    wf_ema_state = {}
    wf_prev_long = set()
    wf_records = []
    wf_prev_smoothed = {}
    wf_rank_corrs = []
    alpha_selections = []

    prev_month_stocks = set()

    for t_idx in range(min_train, n_dates):
        train_dates = unique_dates[:t_idx]
        test_date = unique_dates[t_idx]

        train_data = data[data[date_col].isin(train_dates)]
        test_data = data[data[date_col] == test_date].copy()

        if len(test_data) < nq or len(train_data) < 100:
            continue

        # --- Train model ONCE ---
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

        # --- OOS predictions ---
        X_test = test_data[avail_features].copy()
        X_test_imp = imputer.transform(X_test)
        X_test_scaled = scaler.transform(X_test_imp)
        preds = model.predict(X_test_scaled)

        test_data['prediction'] = preds
        test_data['raw_pct_rank'] = test_data['prediction'].rank(pct=True)

        test_sids = test_data[id_col].values
        test_raw_ranks = test_data['raw_pct_rank'].values
        test_returns = test_data[target].values
        current_month_stocks = set(test_sids)

        # ==================================================================
        # Part 1: Apply each fixed alpha
        # ==================================================================
        for alpha in alpha_grid:
            smoothed = apply_ema_vec(test_sids, test_raw_ranks,
                                     ema_states[alpha], prev_month_stocks, alpha)
            ema_states[alpha] = smoothed

            smoothed_arr = np.array([smoothed[s] for s in test_sids])

            # Prediction stability
            if prev_smoothed_per_alpha[alpha]:
                common_sids = []
                prev_vals = []
                curr_vals = []
                for s in test_sids:
                    if s in prev_smoothed_per_alpha[alpha]:
                        common_sids.append(s)
                        prev_vals.append(prev_smoothed_per_alpha[alpha][s])
                        curr_vals.append(smoothed[s])
                if len(common_sids) >= 20:
                    rho, _ = stats.spearmanr(prev_vals, curr_vals)
                    if not np.isnan(rho):
                        rank_corrs_per_alpha[alpha].append(rho)
            prev_smoothed_per_alpha[alpha] = smoothed.copy()

            # Portfolio construction
            new_long = portfolio_from_ranks_vec(
                test_sids, smoothed_arr, prev_long_per_alpha[alpha],
                top_threshold, buffer, nq
            )

            held_ids = np.array(list(new_long), dtype=test_sids.dtype)
            held_mask = np.isin(test_sids, held_ids)
            gross_ret = test_returns[held_mask].mean() if held_mask.any() else 0.0

            prev = prev_long_per_alpha[alpha]
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

            records_per_alpha[alpha].append({
                'date': test_date, 'gross_ret': gross_ret, 'turnover': turn,
                'n_held': len(new_long),
                **{f'cost_drag_{cm}': cost_drags[cm] for cm in COST_MULTS},
            })
            prev_long_per_alpha[alpha] = new_long

        # ==================================================================
        # Part 2: Walk-forward alpha selection
        # ==================================================================
        # Generate IS predictions for last IS_EVAL_MONTHS
        is_eval_dates = train_dates[-IS_EVAL_MONTHS:]
        is_monthly_preds = {}  # date -> (sids_array, raw_ranks_array, returns_array, cost_data)

        for is_date in is_eval_dates:
            is_month = data[data[date_col] == is_date]
            if len(is_month) < nq:
                continue
            X_is = is_month[avail_features].copy()
            X_is_imp = imputer.transform(X_is)
            X_is_scaled = scaler.transform(X_is_imp)
            is_preds = model.predict(X_is_scaled)

            is_sids = is_month[id_col].values
            is_raw_ranks = pd.Series(is_preds).rank(pct=True).values
            is_returns = is_month[target].values

            is_monthly_preds[is_date] = (is_sids, is_raw_ranks, is_returns, is_month)

        # Evaluate each alpha on IS data
        best_alpha = 0.30  # sensible default
        best_is_sr = -999.0

        is_eval_sorted_dates = sorted(is_monthly_preds.keys())

        if len(is_eval_sorted_dates) >= 12:
            for candidate_alpha in alpha_grid:
                is_ema_state = {}
                is_prev_month_sids = set()
                is_prev_long = set()
                is_records = []

                for is_date in is_eval_sorted_dates:
                    is_sids, is_raw_ranks, is_returns, is_month_df = is_monthly_preds[is_date]

                    smoothed = apply_ema_vec(is_sids, is_raw_ranks,
                                             is_ema_state, is_prev_month_sids, candidate_alpha)
                    is_ema_state = smoothed
                    is_prev_month_sids = set(is_sids)

                    smoothed_arr = np.array([smoothed[s] for s in is_sids])

                    new_long = portfolio_from_ranks_vec(
                        is_sids, smoothed_arr, is_prev_long,
                        top_threshold, buffer, nq
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

                    is_records.append({
                        'gross_ret': gross_ret,
                        'cost_drag': cost_drag,
                    })
                    is_prev_long = new_long

                if len(is_records) >= 6:
                    is_df = pd.DataFrame(is_records)
                    is_net_ret = is_df['gross_ret'] - is_df['cost_drag']
                    is_sr = sharpe(is_net_ret)
                    if is_sr > best_is_sr:
                        best_is_sr = is_sr
                        best_alpha = candidate_alpha

        alpha_selections.append({'date': test_date, 'selected_alpha': best_alpha})

        # Apply WF-selected alpha to OOS
        wf_smoothed = apply_ema_vec(test_sids, test_raw_ranks,
                                     wf_ema_state, prev_month_stocks, best_alpha)
        wf_ema_state = wf_smoothed

        wf_smoothed_arr = np.array([wf_smoothed[s] for s in test_sids])

        # WF prediction stability
        if wf_prev_smoothed:
            common_sids = []
            prev_vals = []
            curr_vals = []
            for s in test_sids:
                if s in wf_prev_smoothed:
                    common_sids.append(s)
                    prev_vals.append(wf_prev_smoothed[s])
                    curr_vals.append(wf_smoothed[s])
            if len(common_sids) >= 20:
                rho, _ = stats.spearmanr(prev_vals, curr_vals)
                if not np.isnan(rho):
                    wf_rank_corrs.append(rho)
        wf_prev_smoothed = wf_smoothed.copy()

        # WF portfolio
        wf_new_long = portfolio_from_ranks_vec(
            test_sids, wf_smoothed_arr, wf_prev_long,
            top_threshold, buffer, nq
        )

        wf_held_ids = np.array(list(wf_new_long), dtype=test_sids.dtype)
        held_mask = np.isin(test_sids, wf_held_ids)
        gross_ret = test_returns[held_mask].mean() if held_mask.any() else 0.0

        if wf_prev_long:
            entries = wf_new_long - wf_prev_long
            exits = wf_prev_long - wf_new_long
            turn = (len(entries) + len(exits)) / (2 * max(len(wf_new_long), 1))
        else:
            entries = wf_new_long
            exits = set()
            turn = np.nan

        cost_drags = {}
        for cm in COST_MULTS:
            cost_drags[cm] = compute_cost_drag(
                test_data, entries, exits, wf_new_long, id_col,
                cm, global_med_1x, wf_prev_long
            )

        wf_records.append({
            'date': test_date, 'gross_ret': gross_ret, 'turnover': turn,
            'n_held': len(wf_new_long),
            **{f'cost_drag_{cm}': cost_drags[cm] for cm in COST_MULTS},
        })
        wf_prev_long = wf_new_long

        prev_month_stocks = current_month_stocks

    # Assemble results
    results = {}
    for alpha in alpha_grid:
        if records_per_alpha[alpha]:
            stab = np.mean(rank_corrs_per_alpha[alpha]) if rank_corrs_per_alpha[alpha] else np.nan
            results[f'Ridge_EMA_{alpha}'] = (
                pd.DataFrame(records_per_alpha[alpha]).set_index('date'),
                stab
            )

    if wf_records:
        wf_stab = np.mean(wf_rank_corrs) if wf_rank_corrs else np.nan
        results['Ridge_WF_EMA'] = (
            pd.DataFrame(wf_records).set_index('date'),
            wf_stab
        )

    return results, alpha_selections


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate(result_df, cost_mult, period_label, period_mask,
             strategy_name, market, pred_stability=np.nan):
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
        'Pred_Stability': round(pred_stability, 4) if not np.isnan(pred_stability) else np.nan,
    }


# ==============================================================================
# MAIN
# ==============================================================================

print("=" * 100)
print("EXPERIMENT AR (FAST): EMA Robustness -- Vectorized")
print(f"  Alphas: {EMA_ALPHAS}")
print("  15 emerging markets")
print("=" * 100)
sys.stdout.flush()

all_results = []
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

    print(f"  Features: {len(ml_features)} total")

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

    # 1. MomentumSort
    print(f"  --- MomentumSort ---")
    mom_result = run_factor_sort(
        df, ['ret_12_1'], date_col, id_col, target, cost_lookup_1x,
        nq=nq, buffer_pct=BUFFER_PCT
    )

    # 2+3. All EMA alphas + WF in single pass
    print(f"  --- Ridge (all alphas + WF) ---")
    sys.stdout.flush()
    ml_results, wf_alpha_sel = run_ml_walkforward_all_alphas(
        df, ml_features, date_col, id_col, target, cost_lookup_1x,
        alpha_grid=EMA_ALPHAS, nq=nq, buffer_pct=BUFFER_PCT,
        min_train=MIN_TRAIN_MONTHS
    )

    for sel in wf_alpha_sel:
        sel['Market'] = mkt_name
    all_alpha_selections.extend(wf_alpha_sel)

    # Evaluate
    all_strategies = {'MomentumSort': (mom_result, np.nan)}
    all_strategies.update(ml_results)

    for strat_name, (result, stab) in all_strategies.items():
        if result is None:
            continue
        if stab is None:
            stab = np.nan
        for period_label, mask_item in make_masks(result):
            for cm in COST_MULTS:
                metrics = evaluate(result, cm, period_label, mask_item,
                                   strat_name, mkt_name, stab)
                if metrics:
                    all_results.append(metrics)

    # Print OOS summary
    mkt_elapsed = time.time() - mkt_t0
    print(f"\n  OOS Summary (Net SR @ 3x costs) [{mkt_elapsed:.0f}s]:")
    strat_order = ['MomentumSort'] + [f'Ridge_EMA_{a}' for a in EMA_ALPHAS] + ['Ridge_WF_EMA']
    for strat_name in strat_order:
        if strat_name not in all_strategies:
            continue
        result, stab = all_strategies[strat_name]
        if result is None:
            continue
        oos_mask = result.index > is_end_date
        oos = result[oos_mask]
        if len(oos) > 0:
            sr3 = sharpe(oos['gross_ret'] - oos['cost_drag_3.0'])
            gsr = sharpe(oos['gross_ret'])
            to = oos['turnover'].mean()
            stab_str = f'  Stab={stab:.3f}' if stab is not None and not np.isnan(stab) else ''
            print(f"    {strat_name:25s}  Gross={gsr:.2f}  Net@3x={sr3:.2f}  TO={to:.1%}{stab_str}")

    # Print WF alpha distribution
    if wf_alpha_sel:
        mkt_sel = [s['selected_alpha'] for s in wf_alpha_sel if s['Market'] == mkt_name]
        if mkt_sel:
            sel_counts = Counter(mkt_sel)
            top3 = sel_counts.most_common(3)
            top3_str = ', '.join([f'{a}({c}x)' for a, c in top3])
            print(f"    WF alpha: {top3_str} (of {len(mkt_sel)} windows)")

    sys.stdout.flush()


elapsed = time.time() - t0
print(f"\n\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")


# ==============================================================================
# SAVE RESULTS
# ==============================================================================

results_df = pd.DataFrame(all_results)
outfile = 'scripts/concern_experiments/expAR_ema_robustness.csv'
results_df.to_csv(outfile, index=False)
print(f"\nSaved {len(results_df)} rows to {outfile}")


# ==============================================================================
# SUMMARY 1: Full Alpha Curve
# ==============================================================================

print("\n\n" + "=" * 100)
print("SUMMARY 1: Full Alpha Curve -- Avg OOS Net SR Across 15 Markets at 3x Costs")
print("=" * 100)

oos_3x = results_df[(results_df['Period'] == 'OOS') & (results_df['CostMult'] == 3.0)]

if len(oos_3x) > 0:
    print(f"\n{'Alpha':>8s}  {'Strategy':>25s}  {'Mean_SR':>8s}  {'Med_SR':>8s}  {'Wins>0':>7s}  {'Mean_TO':>8s}  {'Mean_Stab':>10s}")
    print("-" * 85)

    for alpha in EMA_ALPHAS:
        strat = f'Ridge_EMA_{alpha}'
        strat_data = oos_3x[oos_3x['Strategy'] == strat]
        if len(strat_data) == 0:
            continue
        mean_sr = strat_data['Net_SR'].mean()
        med_sr = strat_data['Net_SR'].median()
        wins = (strat_data['Net_SR'] > 0).sum()
        n_mkts = len(strat_data)
        mean_to = strat_data['Turnover'].mean()
        mean_stab = strat_data['Pred_Stability'].mean()
        stab_str = f'{mean_stab:.3f}' if not np.isnan(mean_stab) else 'N/A'
        print(f"{alpha:8.2f}  {strat:>25s}  {mean_sr:8.3f}  {med_sr:8.3f}  {wins:3d}/{n_mkts:<3d}  {mean_to:8.3f}  {stab_str:>10s}")

    print("-" * 85)
    for strat in ['MomentumSort', 'Ridge_WF_EMA']:
        strat_data = oos_3x[oos_3x['Strategy'] == strat]
        if len(strat_data) == 0:
            continue
        mean_sr = strat_data['Net_SR'].mean()
        med_sr = strat_data['Net_SR'].median()
        wins = (strat_data['Net_SR'] > 0).sum()
        n_mkts = len(strat_data)
        mean_to = strat_data['Turnover'].mean()
        mean_stab = strat_data['Pred_Stability'].mean()
        stab_str = f'{mean_stab:.3f}' if not np.isnan(mean_stab) else 'N/A'
        alpha_str = 'WF' if strat == 'Ridge_WF_EMA' else 'N/A'
        print(f"{alpha_str:>8s}  {strat:>25s}  {mean_sr:8.3f}  {med_sr:8.3f}  {wins:3d}/{n_mkts:<3d}  {mean_to:8.3f}  {stab_str:>10s}")


# ==============================================================================
# SUMMARY 2: WF_EMA vs MomentumSort
# ==============================================================================

print("\n\n" + "=" * 100)
print("SUMMARY 2: Ridge_WF_EMA vs MomentumSort (OOS at 3x costs)")
print("=" * 100)

if len(oos_3x) > 0:
    wf_data = oos_3x[oos_3x['Strategy'] == 'Ridge_WF_EMA']
    mom_data = oos_3x[oos_3x['Strategy'] == 'MomentumSort']

    if len(wf_data) > 0 and len(mom_data) > 0:
        wf_by_mkt = wf_data.set_index('Market')['Net_SR']
        mom_by_mkt = mom_data.set_index('Market')['Net_SR']

        common_mkts = sorted(set(wf_by_mkt.index) & set(mom_by_mkt.index))

        print(f"\n{'Market':>6s}  {'Mom_SR':>8s}  {'WF_EMA_SR':>10s}  {'Delta':>8s}")
        print("-" * 40)

        wf_wins = 0
        for mkt in common_mkts:
            m_sr = mom_by_mkt[mkt]
            w_sr = wf_by_mkt[mkt]
            delta = w_sr - m_sr
            if w_sr > m_sr:
                wf_wins += 1
            print(f"{mkt:>6s}  {m_sr:8.3f}  {w_sr:10.3f}  {delta:+8.3f}")

        print("-" * 40)
        avg_mom = mom_by_mkt[common_mkts].mean()
        avg_wf = wf_by_mkt[common_mkts].mean()
        print(f"{'Mean':>6s}  {avg_mom:8.3f}  {avg_wf:10.3f}  {avg_wf-avg_mom:+8.3f}")
        print(f"\n  WF_EMA win rate: {wf_wins}/{len(common_mkts)} ({wf_wins/len(common_mkts)*100:.0f}%)")


# ==============================================================================
# SUMMARY 3: Alpha Selection Distribution
# ==============================================================================

print("\n\n" + "=" * 100)
print("SUMMARY 3: Walk-Forward Alpha Selection Distribution")
print("=" * 100)

if all_alpha_selections:
    alpha_sel_df = pd.DataFrame(all_alpha_selections)
    all_sel = alpha_sel_df['selected_alpha'].values
    overall_counts = Counter(all_sel)

    print(f"\nOverall (N={len(all_sel)}):")
    print(f"\n{'Alpha':>8s}  {'Count':>6s}  {'Pct':>6s}  Bar")
    print("-" * 50)
    for alpha in EMA_ALPHAS:
        count = overall_counts.get(alpha, 0)
        pct = count / len(all_sel) * 100
        bar = '#' * int(pct / 2)
        print(f"{alpha:8.2f}  {count:6d}  {pct:5.1f}%  {bar}")


# ==============================================================================
# KEY TAKEAWAY
# ==============================================================================

print("\n\n" + "=" * 100)
print("KEY TAKEAWAY")
print("=" * 100)

if len(oos_3x) > 0:
    pivot = oos_3x.pivot_table(index='Market', columns='Strategy', values='Net_SR')
    ema_strats = [f'Ridge_EMA_{a}' for a in EMA_ALPHAS if f'Ridge_EMA_{a}' in pivot.columns]
    if ema_strats:
        ema_means = {s: pivot[s].mean() for s in ema_strats}
        best_fixed = max(ema_means, key=ema_means.get)
        best_fixed_sr = ema_means[best_fixed]
        best_alpha = best_fixed.replace('Ridge_EMA_', '')
    else:
        best_fixed_sr = np.nan
        best_alpha = 'N/A'

    mom_sr = pivot['MomentumSort'].mean() if 'MomentumSort' in pivot.columns else np.nan
    wf_sr = pivot['Ridge_WF_EMA'].mean() if 'Ridge_WF_EMA' in pivot.columns else np.nan

    print(f"\n  MomentumSort:          {mom_sr:.3f}")
    print(f"  Best fixed EMA (a={best_alpha}): {best_fixed_sr:.3f}  (vs Mom: {best_fixed_sr-mom_sr:+.3f})")
    print(f"  Walk-forward EMA:      {wf_sr:.3f}  (vs Mom: {wf_sr-mom_sr:+.3f})")

    if ema_strats:
        srs = [ema_means[s] for s in ema_strats]
        print(f"\n  Alpha sensitivity: range={max(srs)-min(srs):.3f}")

    if not np.isnan(wf_sr) and ema_strats:
        if wf_sr >= best_fixed_sr - 0.02:
            print("  => WF alpha selection works (no data mining concern)")
        else:
            print(f"  => WF underperforms best fixed by {best_fixed_sr-wf_sr:.3f}")

print("\nDone!")
