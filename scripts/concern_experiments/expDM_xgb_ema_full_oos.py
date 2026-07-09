"""
Experiment DM: XGBoost + EMA prediction smoothing on the FULL OOS design (Table 1 / expAX).

The paper's XGB_EMA headline (net SR ~0.53) currently comes from a post-2020
validation sample only. This rerun evaluates XGBoost + EMA on the SAME
walk-forward / IS-OOS / universe / cost design as expAX (Table 1), so the
"trees reach the ceiling with output smoothing" claim can stand on the main sample.

Design decisions (copied EXACTLY from the reference scripts for comparability):
  - XGBoost config + inside-loop top-20 feature selection: from expAX_full_comparison.py
  - Walk-forward split, universe/lag handling, buffer, cost model:      from expAX
  - EMA-on-prediction-ranks + walk-forward alpha selection machinery:   from expAR_ema_robustness.py
    (IS_EVAL_MONTHS=24, maximize IS net SR at 3x, no future OOS months)

Efficiency: ONE XGBoost fit per month per market. Prediction ranks are stored,
then the whole EMA alpha grid is applied to the stored ranks afterward (EMA
operates on predictions, so every alpha shares the same fits).

Strategies emitted:
  - XGBoost         (unsmoothed; == EMA alpha=1.0)
  - XGB_EMA_0.2 / _0.35 / _0.5 / _0.7   (fixed-alpha smoothing)
  - XGB_WF_EMA      (walk-forward-selected alpha)
  - MomentumSort    (control)

Output: scripts/concern_experiments/expDM_xgb_ema_full_oos.csv
Schema: Market, Strategy, CostMult, Period, Net_SR, Gross_SR, Ann_Ret, Turnover, Avg_Cost_bps, N_months

Usage:
  python scripts/concern_experiments/expDM_xgb_ema_full_oos.py               # all 15 markets
  python scripts/concern_experiments/expDM_xgb_ema_full_oos.py --markets THA  # subset (comma-separated)
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
from collections import Counter
warnings.filterwarnings('ignore')

from sklearn.impute import SimpleImputer
import xgboost as xgb
from scipy import stats

from data_loader import load_equity_data, lag_features
from market_config import MARKETS, COST_MULTS, BUFFER_PCT, MIN_TRAIN_MONTHS, EMA_ALPHAS

np.random.seed(42)


# ==============================================================================
# CONFIGURATION (copied from expAX_full_comparison.py)
# ==============================================================================

ACCOUNTING_KEYWORDS = [
    'f_score', 'o_score', 'z_score',
    'qmj', 'qmj_prof', 'qmj_safety', 'qmj_growth',
    'roa', 'roe', 'gp_at', 'gp_atl1', 'op_at', 'op_atl1',
    'niq_at', 'niq_be', 'niq_at_chg1', 'niq_be_chg1',
    'ni_be', 'ni_me', 'ni_ar1', 'ni_inc8q',
    'cop_at', 'cop_atl1', 'ope_be', 'ope_bel1', 'ocf_at', 'ocf_at_chg1', 'ocf_me',
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

# Fixed EMA alphas the paper reports (all present in EMA_ALPHAS). 1.0 == unsmoothed.
FIXED_ALPHAS = [0.2, 0.35, 0.5, 0.7]
TOP_K_FEATURES = 20  # expAX inside-loop feature selection


# ==============================================================================
# HELPERS (copied from expAX / expAR)
# ==============================================================================

def sharpe(s):
    s = s.dropna()
    if len(s) < 2 or s.std() == 0:
        return 0.0
    return (s.mean() / s.std()) * np.sqrt(12)


def classify_features(all_features):
    accounting, market = [], []
    for f in all_features:
        fl = f.lower()
        if any(kw == fl or kw in fl for kw in ACCOUNTING_KEYWORDS):
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
    return frac_traded * (traded_costs / 10000)


def portfolio_from_ranks_vec(sids, ranks, prev_long, top_threshold, buffer, nq):
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
    smoothed = np.empty_like(raw_ranks)
    for i, (sid, rr) in enumerate(zip(sids, raw_ranks)):
        if sid in ema_state and sid in prev_month_stocks:
            smoothed[i] = alpha * rr + (1 - alpha) * ema_state[sid]
        else:
            smoothed[i] = rr
    return dict(zip(sids, smoothed))


# ==============================================================================
# FACTOR SORT (from expAR) — MomentumSort control
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

    for date in dates:
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

        cost_drags = {cm: compute_cost_drag(month, entries, exits, new_long, id_col,
                                            cm, global_med_1x, prev_long) for cm in COST_MULTS}
        records.append({'date': date, 'gross_ret': gross_ret, 'turnover': turn,
                        'n_held': len(new_long),
                        **{f'cost_drag_{cm}': cost_drags[cm] for cm in COST_MULTS}})
        prev_long = new_long

    if len(records) == 0:
        return None
    return pd.DataFrame(records).set_index('date')


# ==============================================================================
# XGBOOST WALK-FORWARD: one fit/month, all EMA alphas + WF selection
# (structure mirrors expAR.run_ml_walkforward_all_alphas; model is XGBoost with
#  expAX's exact config + inside-loop top-20 feature selection)
# ==============================================================================

def run_xgb_walkforward_all_alphas(df, features, date_col, id_col, target,
                                   cost_lookup_1x, alpha_grid, nq=5,
                                   buffer_pct=0.20, min_train=60,
                                   top_k_features=TOP_K_FEATURES):
    data = df.copy().sort_values([date_col, id_col]).reset_index(drop=True)
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

    # WF state
    wf_ema_state = {}
    wf_prev_long = set()
    wf_records = []
    alpha_selections = []

    prev_month_stocks = set()

    for t_idx in range(min_train, n_dates):
        train_dates = unique_dates[:t_idx]
        test_date = unique_dates[t_idx]

        train_data = data[data[date_col].isin(train_dates)]
        test_data = data[data[date_col] == test_date].copy()
        if len(test_data) < nq or len(train_data) < 100:
            continue

        # --- Impute (fit on train) ---
        X_train = train_data[avail_features].copy()
        y_train = train_data['_rank_target'].values
        imputer = SimpleImputer(strategy='median')
        X_train_imp = imputer.fit_transform(X_train)

        mask = ~np.isnan(y_train)
        if mask.sum() < 50:
            continue

        # --- Inside-loop top-20 feature selection (expAX config) ---
        if len(avail_features) > top_k_features:
            selector = xgb.XGBRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0
            )
            selector.fit(X_train_imp[mask], y_train[mask])
            top_idx = np.argsort(selector.feature_importances_)[-top_k_features:]
        else:
            top_idx = np.arange(X_train_imp.shape[1])

        # --- Fit final XGBoost (expAX config) ---
        model = xgb.XGBRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0
        )
        model.fit(X_train_imp[mask][:, top_idx], y_train[mask])

        # --- OOS predictions ---
        X_test_imp = imputer.transform(test_data[avail_features].copy())
        preds = model.predict(X_test_imp[:, top_idx])
        test_data['prediction'] = preds
        test_data['raw_pct_rank'] = test_data['prediction'].rank(pct=True)

        test_sids = test_data[id_col].values
        test_raw_ranks = test_data['raw_pct_rank'].values
        test_returns = test_data[target].values
        current_month_stocks = set(test_sids)

        # ---- Part 1: apply each fixed alpha (EMA on stored prediction ranks) ----
        for alpha in alpha_grid:
            smoothed = apply_ema_vec(test_sids, test_raw_ranks,
                                     ema_states[alpha], prev_month_stocks, alpha)
            ema_states[alpha] = smoothed
            smoothed_arr = np.array([smoothed[s] for s in test_sids])

            new_long = portfolio_from_ranks_vec(
                test_sids, smoothed_arr, prev_long_per_alpha[alpha],
                top_threshold, buffer, nq)

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

            cost_drags = {cm: compute_cost_drag(test_data, entries, exits, new_long, id_col,
                                                cm, global_med_1x, prev) for cm in COST_MULTS}
            records_per_alpha[alpha].append({
                'date': test_date, 'gross_ret': gross_ret, 'turnover': turn,
                'n_held': len(new_long),
                **{f'cost_drag_{cm}': cost_drags[cm] for cm in COST_MULTS}})
            prev_long_per_alpha[alpha] = new_long

        # ---- Part 2: walk-forward alpha selection on last IS_EVAL_MONTHS ----
        is_eval_dates = train_dates[-IS_EVAL_MONTHS:]
        is_monthly_preds = {}
        for is_date in is_eval_dates:
            is_month = data[data[date_col] == is_date]
            if len(is_month) < nq:
                continue
            X_is_imp = imputer.transform(is_month[avail_features].copy())
            is_preds = model.predict(X_is_imp[:, top_idx])
            is_sids = is_month[id_col].values
            is_raw_ranks = pd.Series(is_preds).rank(pct=True).values
            is_returns = is_month[target].values
            is_monthly_preds[is_date] = (is_sids, is_raw_ranks, is_returns, is_month)

        best_alpha = 0.30
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
                        is_sids, smoothed_arr, is_prev_long, top_threshold, buffer, nq)
                    held_ids = np.array(list(new_long), dtype=is_sids.dtype)
                    held_mask = np.isin(is_sids, held_ids)
                    gross_ret = is_returns[held_mask].mean() if held_mask.any() else 0.0
                    if is_prev_long:
                        entries = new_long - is_prev_long
                        exits = is_prev_long - new_long
                    else:
                        entries = new_long
                        exits = set()
                    cost_drag = compute_cost_drag(is_month_df, entries, exits, new_long, id_col,
                                                  COST_MULT_FOR_SELECTION, global_med_1x, is_prev_long)
                    is_records.append({'gross_ret': gross_ret, 'cost_drag': cost_drag})
                    is_prev_long = new_long
                if len(is_records) >= 6:
                    is_df = pd.DataFrame(is_records)
                    is_sr = sharpe(is_df['gross_ret'] - is_df['cost_drag'])
                    if is_sr > best_is_sr:
                        best_is_sr = is_sr
                        best_alpha = candidate_alpha

        alpha_selections.append({'date': test_date, 'selected_alpha': best_alpha})

        # ---- Apply WF-selected alpha to OOS ----
        wf_smoothed = apply_ema_vec(test_sids, test_raw_ranks,
                                    wf_ema_state, prev_month_stocks, best_alpha)
        wf_ema_state = wf_smoothed
        wf_smoothed_arr = np.array([wf_smoothed[s] for s in test_sids])
        wf_new_long = portfolio_from_ranks_vec(
            test_sids, wf_smoothed_arr, wf_prev_long, top_threshold, buffer, nq)
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
        cost_drags = {cm: compute_cost_drag(test_data, entries, exits, wf_new_long, id_col,
                                            cm, global_med_1x, wf_prev_long) for cm in COST_MULTS}
        wf_records.append({'date': test_date, 'gross_ret': gross_ret, 'turnover': turn,
                           'n_held': len(wf_new_long),
                           **{f'cost_drag_{cm}': cost_drags[cm] for cm in COST_MULTS}})
        wf_prev_long = wf_new_long

        prev_month_stocks = current_month_stocks

    # Assemble
    results = {}
    for alpha in alpha_grid:
        if records_per_alpha[alpha]:
            results[f'XGB_EMA_{alpha}'] = pd.DataFrame(records_per_alpha[alpha]).set_index('date')
    if wf_records:
        results['XGB_WF_EMA'] = pd.DataFrame(wf_records).set_index('date')
    return results, alpha_selections


# ==============================================================================
# EVALUATION (expAX schema — no Pred_Stability)
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
    print("EXPERIMENT DM: XGBoost + EMA on the full OOS (expAX) design", flush=True)
    print(f"  Markets: {list(markets.keys())}", flush=True)
    print(f"  Fixed alphas: {FIXED_ALPHAS} (+ unsmoothed=1.0); WF over {EMA_ALPHAS}", flush=True)
    print(f"  Cost mults: {COST_MULTS}", flush=True)
    print("=" * 100, flush=True)

    all_results = []
    all_alpha_selections = []
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

        # Lag me, then filter top-N
        if 'me' in df.columns:
            df = df.sort_values([id_col, date_col])
            df['me'] = df.groupby(id_col)['me'].shift(1)
            df = df.dropna(subset=['me'])
            df['_me_rank'] = df.groupby(date_col)['me'].rank(ascending=False, method='first')
            df = df[df['_me_rank'] <= cfg['topn']].drop(columns=['_me_rank'])

        cost_lookup_1x = build_cost_lookup_1x(df, date_col, id_col)
        ml_features = [f for f in accounting_feats + market_feats if f in df.columns]
        df[date_col] = pd.to_datetime(df[date_col])
        nq = cfg['nq']

        # IS/OOS split (expAX logic, for Table 1 comparability)
        unique_dates = sorted(df[date_col].unique())
        n_total = len(unique_dates)
        is_split_idx = int(n_total * 0.8)
        if is_split_idx < MIN_TRAIN_MONTHS:
            is_split_idx = MIN_TRAIN_MONTHS
        is_end_date = unique_dates[min(is_split_idx - 1, n_total - 1)]
        print(f"  Features: {len(ml_features)}  IS ends {is_end_date}  "
              f"OOS months: {n_total - is_split_idx}", flush=True)

        def make_masks(result_df):
            idx = result_df.index
            return [('IS', idx <= is_end_date), ('OOS', idx > is_end_date),
                    ('Full', pd.Series(True, index=idx))]

        # MomentumSort control
        print("  --- MomentumSort ---", flush=True)
        mom_result = run_factor_sort(df, ['ret_12_1'], date_col, id_col, target,
                                     cost_lookup_1x, nq=nq, buffer_pct=BUFFER_PCT)

        # XGBoost + all alphas + WF (one fit/month)
        print("  --- XGBoost (one fit/month; all alphas + WF) ---", flush=True)
        xgb_results, wf_alpha_sel = run_xgb_walkforward_all_alphas(
            df, ml_features, date_col, id_col, target, cost_lookup_1x,
            alpha_grid=EMA_ALPHAS, nq=nq, buffer_pct=BUFFER_PCT,
            min_train=MIN_TRAIN_MONTHS)

        for sel in wf_alpha_sel:
            sel['Market'] = mkt_name
        all_alpha_selections.extend(wf_alpha_sel)

        # Build emitted strategy set: unsmoothed + requested fixed alphas + WF + Mom
        emit = {'MomentumSort': mom_result}
        if 'XGB_EMA_1.0' in xgb_results:
            emit['XGBoost'] = xgb_results['XGB_EMA_1.0']
        for a in FIXED_ALPHAS:
            key = f'XGB_EMA_{a}'
            if key in xgb_results:
                emit[key] = xgb_results[key]
        if 'XGB_WF_EMA' in xgb_results:
            emit['XGB_WF_EMA'] = xgb_results['XGB_WF_EMA']

        for strat_name, result in emit.items():
            if result is None:
                continue
            for period_label, mask_item in make_masks(result):
                for cm in COST_MULTS:
                    m = evaluate(result, cm, period_label, mask_item, strat_name, mkt_name)
                    if m:
                        all_results.append(m)

        # OOS summary @3x
        mkt_elapsed = time.time() - mkt_t0
        print(f"\n  OOS Net SR @3x [{mkt_elapsed:.0f}s]:", flush=True)
        for strat_name in ['MomentumSort', 'XGBoost'] + [f'XGB_EMA_{a}' for a in FIXED_ALPHAS] + ['XGB_WF_EMA']:
            if strat_name not in emit or emit[strat_name] is None:
                continue
            result = emit[strat_name]
            oos = result[result.index > is_end_date]
            if len(oos) > 0:
                sr3 = sharpe(oos['gross_ret'] - oos['cost_drag_3.0'])
                gsr = sharpe(oos['gross_ret'])
                to = oos['turnover'].mean()
                print(f"    {strat_name:16s}  Gross={gsr:.2f}  Net@3x={sr3:.2f}  TO={to:.1%}", flush=True)
        if wf_alpha_sel:
            mkt_sel = [s['selected_alpha'] for s in wf_alpha_sel if s['Market'] == mkt_name]
            if mkt_sel:
                top3 = Counter(mkt_sel).most_common(3)
                print(f"    WF alpha: {', '.join(f'{a}({c}x)' for a, c in top3)} "
                      f"(of {len(mkt_sel)} windows)", flush=True)

    elapsed = time.time() - t0
    print(f"\n\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)

    results_df = pd.DataFrame(all_results)
    outfile = 'scripts/concern_experiments/expDM_xgb_ema_full_oos.csv'
    results_df.to_csv(outfile, index=False)
    print(f"Saved {len(results_df)} rows to {outfile}", flush=True)

    # Summary: OOS Net SR @3x by strategy
    oos_3x = results_df[(results_df['Period'] == 'OOS') & (results_df['CostMult'] == 3.0)]
    if len(oos_3x) > 0:
        print("\n" + "=" * 80, flush=True)
        print("OOS Net SR @3x by strategy (mean across markets)", flush=True)
        print("=" * 80, flush=True)
        order = ['MomentumSort', 'XGBoost'] + [f'XGB_EMA_{a}' for a in FIXED_ALPHAS] + ['XGB_WF_EMA']
        for s in order:
            sd = oos_3x[oos_3x['Strategy'] == s]
            if len(sd) > 0:
                print(f"  {s:16s}  mean={sd['Net_SR'].mean():.3f}  "
                      f"median={sd['Net_SR'].median():.3f}  "
                      f"wins>0={int((sd['Net_SR'] > 0).sum())}/{len(sd)}  "
                      f"TO={sd['Turnover'].mean():.1%}", flush=True)

    print("\nDone!", flush=True)


if __name__ == '__main__':
    main()
