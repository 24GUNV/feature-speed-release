"""
Experiment CP: Gross-Net SR Inversion for Tree Models
=====================================================

The key finding from expCO was that Ridge_All has the BEST gross SR but WORST
net SR -- a "gross-net SR inversion" where the ranking flips between gross and
net returns. This experiment tests whether the same pattern holds for XGBoost
and LightGBM.

Strategies per model (3 each):
    1. Model_All          -- All features, no smoothing (baseline)
    2. Model_Slow         -- Slow features only, no smoothing
    3. Model_AllSmoothed  -- All features, EMA smoothed at alpha=0.3

Models:
    - Ridge (alpha=100)
    - XGBoost (n_estimators=100, max_depth=3)
    - LightGBM (n_estimators=100, max_depth=3)

Plus MomentumSort as benchmark.

Total: 9 ML strategies + 1 sort = 10 strategies.

Pipeline: lag=1 all, top-N filter, walk-forward expanding window,
rank-target, top quintile with 20% buffer, equal weight, long-only.

15 EM markets, cost multipliers 0 and 3 only (gross and net).

Output: scripts/concern_experiments/expCP_tree_smoothing.csv
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
warnings.filterwarnings('ignore')

from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import xgboost as xgb
import lightgbm as lgb

from data_loader import load_equity_data, lag_features
from market_config import MARKETS, BUFFER_PCT, MIN_TRAIN_MONTHS

np.random.seed(42)

RIDGE_ALPHA = 100.0
COST_MULTS = [0.0, 3.0]
EMA_ALPHA = 0.3  # Only one alpha -- heaviest smoothing

# ==============================================================================
# FEATURE SPEED CLASSIFICATION (from expAH/expCO)
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


def classify_features_acct_mkt(all_features):
    accounting, market = [], []
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
    if cost_mult == 0.0 or not prev_long or not (entries | exits):
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


def portfolio_from_ranks(ranks, ids, actuals, prev_long,
                         top_threshold, buffer, nq,
                         test_data, id_col, cost_mults, global_med_1x):
    prev_ids = np.array(list(prev_long), dtype=ids.dtype) if prev_long else np.array([], dtype=ids.dtype)
    held_mask = np.isin(ids, prev_ids)
    select_mask = np.where(held_mask,
                           ranks >= top_threshold - buffer,
                           ranks >= top_threshold + buffer)
    new_long = set(ids[select_mask])
    if len(new_long) == 0:
        top_k = max(1, len(ids) // nq)
        top_idx = np.argsort(ranks)[-top_k:]
        new_long = set(ids[top_idx])

    held_ret = np.isin(ids, np.array(list(new_long), dtype=ids.dtype))
    gross_ret = float(np.nanmean(actuals[held_ret])) if held_ret.sum() > 0 else 0.0

    if prev_long:
        entries = new_long - prev_long
        exits = prev_long - new_long
        turn = (len(entries) + len(exits)) / (2 * max(len(new_long), 1))
    else:
        entries, exits, turn = new_long, set(), np.nan

    cost_drags = {}
    for cm in cost_mults:
        cost_drags[cm] = compute_cost_drag(
            test_data, entries, exits, new_long, id_col,
            cm, global_med_1x, prev_long)
    return new_long, gross_ret, turn, cost_drags


def ridge_fit_predict(X_train, y_train, X_test, alpha=RIDGE_ALPHA):
    p = X_train.shape[1]
    XtX = X_train.T @ X_train
    Xty = X_train.T @ y_train
    reg = alpha * np.eye(p)
    try:
        beta = np.linalg.solve(XtX + reg, Xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(XtX + reg, Xty, rcond=None)[0]
    preds = X_test @ beta
    return np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)


def xgb_fit_predict(X_train, y_train, X_test):
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1,
        verbosity=0
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)


def lgbm_fit_predict(X_train, y_train, X_test):
    model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=1,
        verbose=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)


def smooth_features_ema(df, feature_cols, alpha, id_col, date_col):
    """Apply per-stock EMA to specified features. EMA is causal -- no look-ahead."""
    if alpha >= 1.0:
        return df
    df = df.sort_values([id_col, date_col]).copy()
    for col in feature_cols:
        if col in df.columns:
            df[col] = df.groupby(id_col)[col].transform(
                lambda x: x.ewm(alpha=alpha, adjust=False).mean()
            )
    return df


# ==============================================================================
# MODEL DISPATCH
# ==============================================================================

MODEL_FIT_PREDICT = {
    'Ridge': ridge_fit_predict,
    'XGB': xgb_fit_predict,
    'LGBM': lgbm_fit_predict,
}


# ==============================================================================
# MAIN EXPERIMENT -- UNIFIED WALK-FORWARD
# ==============================================================================

def run_single_market(mkt_name, cfg):
    """
    Run all strategies for a single market in a unified walk-forward loop.

    Key optimization: pre-compute smoothed feature arrays upfront,
    then in the walk-forward loop train/predict all strategies in one pass.
    """
    t0 = time.time()

    df, raw_features, date_col, id_col = load_equity_data(cfg['file'], cfg['type'])
    target = cfg['target']
    all_features = [f for f in raw_features if f not in EXCLUDE_COLS and f != target]
    accounting_feats, market_feats = classify_features_acct_mkt(all_features)
    accounting_feats = [f for f in accounting_feats if f in df.columns]
    market_feats = [f for f in market_feats if f in df.columns]

    # CRITICAL: Lag BEFORE filter (lag=1 for all features)
    if accounting_feats:
        df = lag_features(df, accounting_feats, id_col, date_col, lag=1)
    if market_feats:
        df = lag_features(df, market_feats, id_col, date_col, lag=1)

    # Lag me
    if 'me' in df.columns:
        df = df.sort_values([id_col, date_col])
        df['me'] = df.groupby(id_col)['me'].shift(1)
        df = df.dropna(subset=['me'])

    # Filter to top-N
    topn = cfg['topn']
    if 'me' in df.columns:
        df['_me_rank'] = df.groupby(date_col)['me'].rank(ascending=False, method='first')
        df = df[df['_me_rank'] <= topn].drop(columns=['_me_rank'])

    cost_lookup_1x = build_cost_lookup_1x(df, date_col, id_col)
    ml_features = [f for f in accounting_feats + market_feats if f in df.columns]
    ml_features = [f for f in ml_features if f not in EXCLUDE_COLS]

    # Classify features by speed
    slow_feats = [f for f in ml_features if classify_feature_speed(f) == 'SLOW']
    medium_feats = [f for f in ml_features if classify_feature_speed(f) == 'MEDIUM']
    fast_feats = [f for f in ml_features if classify_feature_speed(f) == 'FAST']

    df[date_col] = pd.to_datetime(df[date_col])
    data_base = df.copy().sort_values([date_col, id_col]).reset_index(drop=True)

    unique_dates = sorted(data_base[date_col].unique())
    n_dates = len(unique_dates)

    if n_dates < MIN_TRAIN_MONTHS + 12:
        print(f"  {mkt_name}: too few dates ({n_dates})", flush=True)
        return []

    nq = cfg['nq']
    top_threshold = 1.0 - 1.0 / nq
    buffer = BUFFER_PCT * (1.0 / nq)

    print(f"  {mkt_name}: {len(data_base):,} rows, {n_dates} months, "
          f"{len(ml_features)} features "
          f"(slow={len(slow_feats)}, med={len(medium_feats)}, fast={len(fast_feats)})",
          flush=True)

    # ==================================================================
    # Step 1: Pre-compute smoothed dataset (alpha=0.3 only)
    # ==================================================================
    print(f"    Pre-computing smoothed dataset (alpha={EMA_ALPHA})...", flush=True)
    t_smooth = time.time()

    all_feat_idx = {f: i for i, f in enumerate(ml_features)}
    slow_col_idx = [all_feat_idx[f] for f in slow_feats if f in all_feat_idx]

    # Rank target
    data_base['_rank_target'] = data_base.groupby(date_col)[target].rank(pct=True)

    # Merge cost info
    data_base = data_base.merge(cost_lookup_1x, on=[date_col, id_col], how='left')
    global_med_1x = cost_lookup_1x['cost_bps_1x'].median()
    data_base['cost_bps_1x'] = data_base['cost_bps_1x'].fillna(global_med_1x)
    data_base = data_base.sort_values([date_col, id_col]).reset_index(drop=True)

    # Extract numpy arrays
    raw_feat_matrix = data_base[ml_features].values.copy()
    rank_target = data_base['_rank_target'].values.copy()
    target_vals = data_base[target].values.copy()
    ids_arr = data_base[id_col].values.copy()
    dates_arr = data_base[date_col].values.copy()
    cost_arr = data_base['cost_bps_1x'].values.copy()

    # Pre-compute date boundaries
    date_start_idx = {}
    date_end_idx = {}
    for d in unique_dates:
        mask_d = dates_arr == d
        indices = np.where(mask_d)[0]
        if len(indices) > 0:
            date_start_idx[d] = indices[0]
            date_end_idx[d] = indices[-1] + 1

    # Smooth all features at alpha=0.3
    df_temp = data_base[[id_col, date_col] + ml_features].copy()
    df_temp = smooth_features_ema(df_temp, ml_features, EMA_ALPHA, id_col, date_col)
    df_temp = df_temp.sort_values([date_col, id_col]).reset_index(drop=True)
    all_smoothed_matrix = df_temp[ml_features].values.copy()
    del df_temp

    print(f"    Smoothing done in {time.time()-t_smooth:.0f}s", flush=True)

    # ==================================================================
    # Step 2: Define strategy configs
    # ==================================================================
    # 3 models x 3 strategies = 9 ML strategies + MomentumSort
    strategies = {}

    for model_name in ['Ridge', 'XGB', 'LGBM']:
        # 1. Model_All (raw, all features)
        def _make_getter_all_raw(mn=model_name):
            def get_mat(tr_sl, te_sl):
                return (raw_feat_matrix[tr_sl], raw_feat_matrix[te_sl])
            return get_mat

        strategies[f'{model_name}_All'] = {
            'get_matrix': _make_getter_all_raw(),
            'model': model_name,
        }

        # 2. Model_Slow (raw, slow features only)
        if len(slow_col_idx) >= 3:
            def _make_getter_slow(mn=model_name, _idx=slow_col_idx):
                def get_mat(tr_sl, te_sl):
                    return (raw_feat_matrix[tr_sl][:, _idx], raw_feat_matrix[te_sl][:, _idx])
                return get_mat

            strategies[f'{model_name}_Slow'] = {
                'get_matrix': _make_getter_slow(),
                'model': model_name,
            }

        # 3. Model_AllSmoothed (all features, EMA alpha=0.3)
        def _make_getter_all_sm(mn=model_name):
            def get_mat(tr_sl, te_sl):
                return (all_smoothed_matrix[tr_sl], all_smoothed_matrix[te_sl])
            return get_mat

        strategies[f'{model_name}_AllSmoothed'] = {
            'get_matrix': _make_getter_all_sm(),
            'model': model_name,
        }

    # ==================================================================
    # Step 3: Unified walk-forward loop
    # ==================================================================
    print(f"    Running walk-forward ({n_dates - MIN_TRAIN_MONTHS} steps, "
          f"{len(strategies)} ML strategies + MomSort)...", flush=True)

    prev_longs = {s: set() for s in strategies}
    prev_longs['MomentumSort'] = set()
    month_records = {s: [] for s in strategies}
    month_records['MomentumSort'] = []

    feat_masks = {}
    n_steps = n_dates - MIN_TRAIN_MONTHS

    for step_i, t_idx in enumerate(range(MIN_TRAIN_MONTHS, n_dates)):
        if step_i % 50 == 0:
            print(f"      step {step_i}/{n_steps}", flush=True)

        test_date = unique_dates[t_idx]
        if test_date not in date_start_idx:
            continue

        tr_end = date_start_idx[test_date]
        te_start = date_start_idx[test_date]
        te_end = date_end_idx[test_date]
        tr_slice = slice(0, tr_end)
        te_slice = slice(te_start, te_end)

        if tr_end < 100 or (te_end - te_start) < nq:
            continue

        ids = ids_arr[te_slice].copy()
        actuals = target_vals[te_slice].copy()
        y_raw = rank_target[tr_slice].copy()

        test_data_df = pd.DataFrame({
            id_col: ids,
            'cost_bps_1x': cost_arr[te_slice],
        })

        # --- ML strategies ---
        for strat_label, strat_cfg in strategies.items():
            X_train, X_test = strat_cfg['get_matrix'](tr_slice, te_slice)
            model_name = strat_cfg['model']

            # Feature coverage mask
            if step_i % 50 == 0 or strat_label not in feat_masks:
                cov = 1.0 - np.isnan(X_train).mean(axis=0)
                feat_masks[strat_label] = cov > 0.1

            fmask = feat_masks[strat_label]
            p = fmask.sum()
            valid = ~np.isnan(y_raw)
            if valid.sum() < 50 or p == 0:
                continue

            imp = SimpleImputer(strategy='median')
            X_tr = imp.fit_transform(X_train[:, fmask])
            X_te = imp.transform(X_test[:, fmask])
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr[valid])
            X_te = sc.transform(X_te)

            # Dispatch to appropriate model
            fit_predict_fn = MODEL_FIT_PREDICT[model_name]
            preds = fit_predict_fn(X_tr, y_raw[valid], X_te)
            ranks = rankdata(preds, method='average') / len(preds)

            new_long, gross_ret, turn, cost_drags = portfolio_from_ranks(
                ranks, ids, actuals, prev_longs[strat_label],
                top_threshold, buffer, nq, test_data_df, id_col, COST_MULTS, global_med_1x)
            month_records[strat_label].append({
                'date': test_date, 'gross_ret': gross_ret, 'turnover': turn,
                'n_held': len(new_long),
                **{f'cost_drag_{cm}': cost_drags[cm] for cm in COST_MULTS}})
            prev_longs[strat_label] = new_long

        # --- MomentumSort ---
        mom_feat = 'ret_12_1'
        if mom_feat in ml_features:
            mom_idx = all_feat_idx[mom_feat]
            mom_vals = raw_feat_matrix[te_slice, mom_idx]
            mom_valid = ~np.isnan(mom_vals)
            mom_ranks = np.full(len(mom_vals), 0.5)
            if mom_valid.sum() >= nq:
                mom_ranks[mom_valid] = rankdata(mom_vals[mom_valid], method='average') / mom_valid.sum()
            new_long, gross_ret, turn, cost_drags = portfolio_from_ranks(
                mom_ranks, ids, actuals, prev_longs['MomentumSort'],
                top_threshold, buffer, nq, test_data_df, id_col, COST_MULTS, global_med_1x)
            month_records['MomentumSort'].append({
                'date': test_date, 'gross_ret': gross_ret, 'turnover': turn,
                'n_held': len(new_long),
                **{f'cost_drag_{cm}': cost_drags[cm] for cm in COST_MULTS}})
            prev_longs['MomentumSort'] = new_long

    # ==================================================================
    # Step 4: Compute OOS results
    # ==================================================================
    all_results = []
    all_strats = list(strategies.keys()) + ['MomentumSort']

    for strat in all_strats:
        recs = month_records[strat]
        if len(recs) < 6:
            continue
        result_df = pd.DataFrame(recs).set_index('date')
        unique_d = sorted(result_df.index)
        is_end = unique_d[int(len(unique_d) * 0.7)]
        oos_mask = result_df.index > is_end
        subset = result_df[oos_mask]
        if len(subset) < 6:
            continue

        # Get model name
        if strat == 'MomentumSort':
            model_label = 'Sort'
        else:
            model_label = strat.split('_')[0]  # Ridge, XGB, or LGBM

        # Get strategy type
        if strat == 'MomentumSort':
            strat_type = 'MomentumSort'
        elif '_AllSmoothed' in strat:
            strat_type = 'AllSmoothed'
        elif '_Slow' in strat:
            strat_type = 'Slow'
        elif '_All' in strat:
            strat_type = 'All'
        else:
            strat_type = strat

        # Get n_features
        if strat == 'MomentumSort':
            n_feats = 1
        elif '_Slow' in strat:
            n_feats = len(slow_feats)
        else:
            n_feats = len(ml_features)

        for cm in COST_MULTS:
            cost_col = f'cost_drag_{cm}'
            net_ret = subset['gross_ret'] - subset[cost_col]
            mean_to = subset['turnover'].mean()

            all_results.append({
                'Market': mkt_name,
                'Model': model_label,
                'Strategy': strat,
                'StrategyType': strat_type,
                'CostMult': cm,
                'Net_SR': round(sharpe(net_ret), 4),
                'Gross_SR': round(sharpe(subset['gross_ret']), 4),
                'Turnover': round(mean_to, 4),
                'N_months': len(subset),
                'N_features': n_feats,
            })

    elapsed = time.time() - t0
    print(f"  {mkt_name} done in {elapsed:.0f}s", flush=True)
    return all_results


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    t0_global = time.time()

    print("=" * 90)
    print("EXPERIMENT CP: Gross-Net SR Inversion for Tree Models")
    print(f"  Models: Ridge, XGBoost, LightGBM")
    print(f"  Strategies per model: All, Slow, AllSmoothed (alpha={EMA_ALPHA})")
    print(f"  Cost multipliers: {COST_MULTS}")
    print(f"  15 emerging markets")
    print("=" * 90)
    sys.stdout.flush()

    all_results = []

    for mkt_name, cfg in MARKETS.items():
        print(f"\n{'='*60}")
        print(f"  {mkt_name}")
        print(f"{'='*60}")
        sys.stdout.flush()
        try:
            results = run_single_market(mkt_name, cfg)
            all_results.extend(results)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            continue

    results_df = pd.DataFrame(all_results)
    out_path = 'scripts/concern_experiments/expCP_tree_smoothing.csv'
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(results_df)} rows to {out_path}")

    # ======================================================================
    # SUMMARY
    # ======================================================================
    print("\n" + "=" * 90)
    print("SUMMARY: Gross-Net SR Inversion Across Model Classes")
    print("=" * 90)

    # ------------------------------------------------------------------
    # 1. Per-model summary: All vs Slow vs AllSmoothed
    # ------------------------------------------------------------------
    print("\n\n--- 1. GROSS-NET SR INVERSION TEST ---\n")
    print("For each model, compare Gross SR ranking vs Net SR (3x) ranking:\n")

    for model_name in ['Ridge', 'XGB', 'LGBM']:
        print(f"\n  {'='*60}")
        print(f"  {model_name}")
        print(f"  {'='*60}")

        model_data = results_df[results_df['Model'] == model_name]
        if len(model_data) == 0:
            print(f"  No data for {model_name}")
            continue

        # Gross SR (CostMult=0)
        gross_data = model_data[model_data['CostMult'] == 0.0]
        # Net SR (CostMult=3)
        net_data = model_data[model_data['CostMult'] == 3.0]

        print(f"\n  {'Strategy':<25s} {'Gross SR':>10s} {'Net SR(3x)':>12s} {'Turnover':>10s} {'SR Lost':>10s}")
        print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")

        for strat_type in ['All', 'Slow', 'AllSmoothed']:
            strat_label = f'{model_name}_{strat_type}'

            g = gross_data[gross_data['Strategy'] == strat_label]
            n = net_data[net_data['Strategy'] == strat_label]

            if len(g) == 0 or len(n) == 0:
                continue

            gross_sr = g['Gross_SR'].mean()
            net_sr = n['Net_SR'].mean()
            to = g['Turnover'].mean()
            sr_lost = gross_sr - net_sr

            print(f"  {strat_type:<25s} {gross_sr:>+10.3f} {net_sr:>+12.3f} {to:>10.1%} {sr_lost:>+10.3f}")

        # Check for inversion
        all_gross = gross_data[gross_data['StrategyType'] == 'All']['Gross_SR'].mean() if len(gross_data[gross_data['StrategyType'] == 'All']) > 0 else -99
        slow_gross = gross_data[gross_data['StrategyType'] == 'Slow']['Gross_SR'].mean() if len(gross_data[gross_data['StrategyType'] == 'Slow']) > 0 else -99
        smooth_gross = gross_data[gross_data['StrategyType'] == 'AllSmoothed']['Gross_SR'].mean() if len(gross_data[gross_data['StrategyType'] == 'AllSmoothed']) > 0 else -99

        all_net = net_data[net_data['StrategyType'] == 'All']['Net_SR'].mean() if len(net_data[net_data['StrategyType'] == 'All']) > 0 else -99
        slow_net = net_data[net_data['StrategyType'] == 'Slow']['Net_SR'].mean() if len(net_data[net_data['StrategyType'] == 'Slow']) > 0 else -99
        smooth_net = net_data[net_data['StrategyType'] == 'AllSmoothed']['Net_SR'].mean() if len(net_data[net_data['StrategyType'] == 'AllSmoothed']) > 0 else -99

        gross_rank = sorted([(all_gross, 'All'), (slow_gross, 'Slow'), (smooth_gross, 'AllSmoothed')], reverse=True)
        net_rank = sorted([(all_net, 'All'), (slow_net, 'Slow'), (smooth_net, 'AllSmoothed')], reverse=True)

        print(f"\n  Gross SR ranking: {' > '.join([f'{name}({sr:+.3f})' for sr, name in gross_rank])}")
        print(f"  Net SR ranking:   {' > '.join([f'{name}({sr:+.3f})' for sr, name in net_rank])}")

        if gross_rank[0][1] != net_rank[0][1]:
            print(f"  ** INVERSION DETECTED: Best gross ({gross_rank[0][1]}) != Best net ({net_rank[0][1]}) **")
        else:
            print(f"  No inversion: {gross_rank[0][1]} is best on both gross and net")

    # ------------------------------------------------------------------
    # 2. Cross-model comparison table
    # ------------------------------------------------------------------
    print("\n\n--- 2. CROSS-MODEL COMPARISON (Mean across 15 markets) ---\n")

    print(f"  {'Strategy':<25s} {'Gross SR':>10s} {'Net SR(3x)':>12s} {'Turnover':>10s} {'SR Lost':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*10} {'-'*10}")

    # MomentumSort first
    mom_gross = results_df[(results_df['Strategy'] == 'MomentumSort') & (results_df['CostMult'] == 0.0)]
    mom_net = results_df[(results_df['Strategy'] == 'MomentumSort') & (results_df['CostMult'] == 3.0)]
    if len(mom_gross) > 0 and len(mom_net) > 0:
        g_sr = mom_gross['Gross_SR'].mean()
        n_sr = mom_net['Net_SR'].mean()
        to = mom_gross['Turnover'].mean()
        print(f"  {'MomentumSort':<25s} {g_sr:>+10.3f} {n_sr:>+12.3f} {to:>10.1%} {g_sr - n_sr:>+10.3f}")

    print()

    for model_name in ['Ridge', 'XGB', 'LGBM']:
        for strat_type in ['All', 'Slow', 'AllSmoothed']:
            strat_label = f'{model_name}_{strat_type}'
            g = results_df[(results_df['Strategy'] == strat_label) & (results_df['CostMult'] == 0.0)]
            n = results_df[(results_df['Strategy'] == strat_label) & (results_df['CostMult'] == 3.0)]
            if len(g) == 0 or len(n) == 0:
                continue
            g_sr = g['Gross_SR'].mean()
            n_sr = n['Net_SR'].mean()
            to = g['Turnover'].mean()
            print(f"  {strat_label:<25s} {g_sr:>+10.3f} {n_sr:>+12.3f} {to:>10.1%} {g_sr - n_sr:>+10.3f}")
        print()

    # ------------------------------------------------------------------
    # 3. Inversion summary
    # ------------------------------------------------------------------
    print("\n--- 3. INVERSION SUMMARY ---\n")
    print("  Does the model with the highest Gross SR have the lowest Net SR?\n")

    for strat_type in ['All', 'Slow', 'AllSmoothed']:
        print(f"\n  Strategy: {strat_type}")
        for metric, cm, col in [('Gross SR', 0.0, 'Gross_SR'), ('Net SR(3x)', 3.0, 'Net_SR')]:
            subset = results_df[(results_df['StrategyType'] == strat_type) & (results_df['CostMult'] == cm)]
            if len(subset) == 0:
                continue
            by_model = subset.groupby('Model')[col].mean().sort_values(ascending=False)
            ranking_str = ' > '.join([f'{m}({v:+.3f})' for m, v in by_model.items()])
            print(f"    {metric}: {ranking_str}")

    # ------------------------------------------------------------------
    # 4. Per-market detail for the "All" strategy
    # ------------------------------------------------------------------
    print("\n\n--- 4. PER-MARKET DETAIL: Model_All at Net SR (3x) ---\n")

    all_strats_detail = ['MomentumSort'] + [f'{m}_All' for m in ['Ridge', 'XGB', 'LGBM']]
    detail_data = results_df[(results_df['Strategy'].isin(all_strats_detail)) & (results_df['CostMult'] == 3.0)]
    if len(detail_data) > 0:
        pivot = detail_data.pivot_table(index='Market', columns='Strategy', values='Net_SR')
        ordered = [c for c in all_strats_detail if c in pivot.columns]
        pivot = pivot[ordered]
        print(pivot.round(3).to_string())
        print(f"\nMean:\n{pivot.mean().round(3).to_string()}")

    # ------------------------------------------------------------------
    # 5. Turnover comparison across models
    # ------------------------------------------------------------------
    print("\n\n--- 5. TURNOVER BY MODEL AND STRATEGY ---\n")

    for strat_type in ['All', 'Slow', 'AllSmoothed']:
        subset = results_df[(results_df['StrategyType'] == strat_type) & (results_df['CostMult'] == 0.0)]
        if len(subset) == 0:
            continue
        by_model = subset.groupby('Model')['Turnover'].mean()
        print(f"  {strat_type}: " + "  ".join([f"{m}={v:.1%}" for m, v in by_model.items()]))

    mom_to = results_df[(results_df['Strategy'] == 'MomentumSort') & (results_df['CostMult'] == 0.0)]['Turnover'].mean()
    print(f"  MomentumSort: {mom_to:.1%}")

    elapsed = time.time() - t0_global
    print(f"\n\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("\nDone!")
