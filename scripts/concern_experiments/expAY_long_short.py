"""
Experiment AY: Long-Short and Short-Only Analysis
==================================================

Tests whether ML's advantage concentrates on the short side (identifying
stocks to AVOID) rather than fine-tuning the long portfolio.

Motivation: Hanauer & Kalsbach (2023) find ML works in EM using long-short
portfolios. We find momentum sort beats ML in long-only top-quintile.
The hypothesis: ML's excess alpha is on the short side.

For each of 7 strategies, construct THREE portfolios:
  1. Long-only:  Top quintile (Q5) equal-weight
  2. Short-only: Bottom quintile (Q1) equal-weight, returns NEGATED
  3. Long-short: Long Q5, Short Q1 (return = Q5_ret - Q1_ret)

Cost treatment:
  - Long-only:  costs on long-side turnover
  - Short-only: costs on short-side turnover + 50 bps/month borrowing cost
  - Long-short: costs on BOTH legs + 50 bps/month borrowing cost on short leg

Markets: THA, KOR, TWN, IND, BRA (5 representative markets)
Cost levels: 1x, 2x, 3x

Strategies:
  1. MomentumSort     2. LowVolSort     3. CompositeSort
  4. Ridge            5. XGBoost        6. LightGBM
  7. MomentumML_Blend (0.8*mom + 0.2*Ridge)

Output: scripts/concern_experiments/expAY_long_short.csv

Usage: python3 scripts/concern_experiments/expAY_long_short.py
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
warnings.filterwarnings('ignore')

from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from scipy import stats

from data_loader import load_equity_data

np.random.seed(42)


def fast_lag_features(df, features, id_col, date_col, lag=1):
    """Faster lag using sort + shift, avoiding expensive groupby on many columns."""
    df = df.sort_values(by=[id_col, date_col]).reset_index(drop=True)
    existing = [f for f in features if f in df.columns]
    if not existing:
        return df
    df[existing] = df.groupby(id_col)[existing].shift(lag)
    # Drop rows where ALL lagged features are NaN (first obs per entity)
    check_cols = existing[:5] if len(existing) > 5 else existing
    df = df.dropna(subset=check_cols, how='all')
    print(f"  Lagged {len(existing)} features by {lag}, shape: {df.shape}")
    return df


# ==============================================================================
# CONFIGURATION
# ==============================================================================

MARKETS = {
    'THA': {'file': 'public/data/ThaiEquitiesData.csv', 'type': 'thai_equity',
            'target': 'ret_1_0', 'topn': 100, 'nq': 5},
    'KOR': {'file': 'data/KOREquitiesData.csv', 'type': 'emerging_equity',
            'target': 'ret_1_0', 'topn': 200, 'nq': 5},
    'TWN': {'file': 'data/TWNEquitiesData.csv', 'type': 'emerging_equity',
            'target': 'ret_1_0', 'topn': 150, 'nq': 5},
    'IND': {'file': 'data/INDEquitiesData.csv', 'type': 'emerging_equity',
            'target': 'ret_1_0', 'topn': 200, 'nq': 5},
    'BRA': {'file': 'data/BRAEquitiesData.csv', 'type': 'emerging_equity',
            'target': 'ret_1_0', 'topn': 100, 'nq': 5},
}

COST_MULTS = [1.0, 2.0, 3.0]
BUFFER_PCT = 0.20
MIN_TRAIN_MONTHS = 60
SHORT_BORROW_COST_MONTHLY = 50.0 / 10000  # 50 bps/month

STRATEGIES = [
    'MomentumSort', 'LowVolSort', 'CompositeSort',
    'Ridge', 'XGBoost', 'LightGBM',
    'MomentumML_Blend',
]
SORT_STRATEGIES = ['MomentumSort', 'LowVolSort', 'CompositeSort']
ML_STRATEGIES = ['Ridge', 'XGBoost', 'LightGBM']

# Feature classification
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

# Sort features
MOMENTUM_FEATURES = ['ret_12_1']
LOWVOL_FEATURES = ['ivol_capm_21d']
QUALITY_FEATURES = ['f_score', 'gp_at', 'oaccruals_at', 'niq_at']


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def sharpe(s):
    """Annualized Sharpe ratio from monthly returns."""
    s = s.dropna()
    if len(s) < 2 or s.std() == 0:
        return 0.0
    return (s.mean() / s.std()) * np.sqrt(12)


def monthly_ic(predictions, actuals):
    """Rank IC (Spearman correlation)."""
    if len(predictions) < 5:
        return np.nan
    ic, _ = stats.spearmanr(predictions, actuals)
    return ic


def classify_features(all_features):
    """Classify features into accounting (lag=4) vs market (lag=1)."""
    accounting = []
    market = []
    for f in all_features:
        f_lower = f.lower()
        if any(kw == f_lower or kw in f_lower for kw in ACCOUNTING_KEYWORDS):
            accounting.append(f)
        else:
            market.append(f)
    return accounting, market


def buffer_select_long(month_df, id_col, rank_col, prev_set, threshold, buffer, nq):
    """Vectorized buffer-based selection for TOP quintile."""
    ids = month_df[id_col].values
    ranks = month_df[rank_col].values
    new_set = set()
    for i in range(len(ids)):
        sid = ids[i]
        rank = ranks[i]
        if sid in prev_set:
            if rank >= threshold - buffer:
                new_set.add(sid)
        else:
            if rank >= threshold + buffer:
                new_set.add(sid)
    if len(new_set) == 0:
        new_set = set(month_df.nlargest(max(1, len(month_df) // nq), rank_col)[id_col])
    return new_set


def buffer_select_short(month_df, id_col, rank_col, prev_set, threshold, buffer, nq):
    """Vectorized buffer-based selection for BOTTOM quintile."""
    ids = month_df[id_col].values
    ranks = month_df[rank_col].values
    new_set = set()
    for i in range(len(ids)):
        sid = ids[i]
        rank = ranks[i]
        if sid in prev_set:
            if rank <= threshold + buffer:
                new_set.add(sid)
        else:
            if rank <= threshold - buffer:
                new_set.add(sid)
    if len(new_set) == 0:
        new_set = set(month_df.nsmallest(max(1, len(month_df) // nq), rank_col)[id_col])
    return new_set


def build_cost_lookup_1x(raw_df, date_col, id_col, spread_col='bidaskhl_21d',
                         floor_bps=20, cap_bps=200):
    """Build per-stock per-month transaction cost lookup at 1x CS."""
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


def compute_cost_drag(month_data, entries, exits, new_portfolio, id_col,
                      cost_mult, global_med_1x, prev_portfolio):
    """Compute per-month cost drag given a cost multiplier."""
    if cost_mult == 0.0:
        return 0.0
    if not prev_portfolio or not (entries | exits):
        return 0.0
    traded = entries | exits
    if len(traded) == 0:
        return 0.0
    traded_costs_1x = month_data[month_data[id_col].isin(traded)]['cost_bps_1x'].mean()
    if pd.isna(traded_costs_1x):
        traded_costs_1x = global_med_1x
    traded_costs = np.clip(traded_costs_1x * cost_mult, 20.0, 200.0)
    frac_traded = len(traded) / max(len(new_portfolio), 1)
    cost_drag = frac_traded * (traded_costs / 10000)
    return cost_drag


# ==============================================================================
# DATA LOADING
# ==============================================================================

def load_and_prepare(mkt_name, cfg):
    """Load data, apply dual-lag, filter to top-N."""
    print(f"\n{'='*80}")
    print(f"Loading {mkt_name}...")
    print(f"{'='*80}")

    df, raw_features, date_col, id_col = load_equity_data(
        cfg['file'], cfg['type']
    )
    target = cfg['target']

    all_features = [f for f in raw_features if f not in EXCLUDE_COLS and f != target]
    accounting_feats, market_feats = classify_features(all_features)
    accounting_feats = [f for f in accounting_feats if f in df.columns]
    market_feats = [f for f in market_feats if f in df.columns]

    print(f"  Accounting features (lag=1, JKP data is pre-lagged 4m): {len(accounting_feats)}")
    print(f"  Market features (lag=1): {len(market_feats)}")

    # CRITICAL: Lag BEFORE filter. All features lag=1: JKP applies the 4-month
    # accounting lag internally, so lag=4 here would double-lag (8m effective).
    if accounting_feats:
        df = fast_lag_features(df, accounting_feats, id_col, date_col, lag=1)
    if market_feats:
        df = fast_lag_features(df, market_feats, id_col, date_col, lag=1)

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
        print(f"  Filtered to top {topn} by market cap")

    # Build 1x cost lookup
    cost_lookup_1x = build_cost_lookup_1x(df, date_col, id_col)

    # ML features
    ml_features = [f for f in accounting_feats + market_feats if f in df.columns]

    df[date_col] = pd.to_datetime(df[date_col])
    n_months = df[date_col].nunique()
    avg_stocks = df.groupby(date_col)[id_col].nunique().mean()
    print(f"  Final: {len(df):,} rows, {n_months} months, avg {avg_stocks:.0f} stocks/month")
    print(f"  ML features: {len(ml_features)}")

    return df, ml_features, date_col, id_col, target, cost_lookup_1x


# ==============================================================================
# FACTOR SORT STRATEGY — tracks BOTH Q5 (top) and Q1 (bottom)
# ==============================================================================

def run_factor_sort_ls(df, sort_features, date_col, id_col, target,
                       cost_lookup_1x, nq=5, buffer_pct=0.20,
                       ascending_features=None):
    """
    Run factor sort tracking both top quintile (Q5) and bottom quintile (Q1).
    Returns per-month records with long_gross_ret, short_gross_ret, and
    cost info for both legs.
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

    top_threshold = 1.0 - 1.0 / nq       # e.g. 0.8 for quintile
    bottom_threshold = 1.0 / nq           # e.g. 0.2 for quintile
    buffer = buffer_pct * (1.0 / nq)

    data = data.merge(cost_lookup_1x, on=[date_col, id_col], how='left')
    global_med_1x = cost_lookup_1x['cost_bps_1x'].median()
    data['cost_bps_1x'] = data['cost_bps_1x'].fillna(global_med_1x)

    dates = sorted(data[date_col].unique())
    prev_long = set()
    prev_short = set()
    records = []

    for date in dates:
        month = data[data[date_col] == date].copy()
        if len(month) < nq:
            continue

        # --- TOP QUINTILE (Q5) with buffer ---
        new_long = buffer_select_long(month, id_col, 'pct_rank', prev_long,
                                       top_threshold, buffer, nq)

        # --- BOTTOM QUINTILE (Q1) with buffer ---
        new_short = buffer_select_short(month, id_col, 'pct_rank', prev_short,
                                         bottom_threshold, buffer, nq)

        # Returns
        long_held = month[month[id_col].isin(new_long)]
        short_held = month[month[id_col].isin(new_short)]
        long_gross_ret = long_held[target].mean() if len(long_held) > 0 else 0.0
        short_gross_ret = short_held[target].mean() if len(short_held) > 0 else 0.0

        # Turnover — long leg
        if prev_long:
            long_entries = new_long - prev_long
            long_exits = prev_long - new_long
            long_turn = (len(long_entries) + len(long_exits)) / (2 * max(len(new_long), 1))
        else:
            long_entries = new_long
            long_exits = set()
            long_turn = np.nan

        # Turnover — short leg
        if prev_short:
            short_entries = new_short - prev_short
            short_exits = prev_short - new_short
            short_turn = (len(short_entries) + len(short_exits)) / (2 * max(len(new_short), 1))
        else:
            short_entries = new_short
            short_exits = set()
            short_turn = np.nan

        # IC
        ic = monthly_ic(month['composite_score'].values, month[target].values)

        # Cost drags per cost multiplier
        long_cost_drags = {}
        short_cost_drags = {}
        for cm in COST_MULTS:
            long_cost_drags[cm] = compute_cost_drag(
                month, long_entries, long_exits, new_long, id_col,
                cm, global_med_1x, prev_long
            )
            short_cost_drags[cm] = compute_cost_drag(
                month, short_entries, short_exits, new_short, id_col,
                cm, global_med_1x, prev_short
            )

        records.append({
            'date': date,
            'long_gross_ret': long_gross_ret,
            'short_gross_ret': short_gross_ret,
            'long_turnover': long_turn,
            'short_turnover': short_turn,
            'n_long': len(new_long),
            'n_short': len(new_short),
            'ic': ic,
            **{f'long_cost_drag_{cm}': long_cost_drags[cm] for cm in COST_MULTS},
            **{f'short_cost_drag_{cm}': short_cost_drags[cm] for cm in COST_MULTS},
        })
        prev_long = new_long
        prev_short = new_short

    if len(records) == 0:
        return None
    return pd.DataFrame(records).set_index('date')


# ==============================================================================
# ML WALK-FORWARD — tracks BOTH Q5 (top) and Q1 (bottom)
# ==============================================================================

def run_ml_walkforward_ls(df, features, date_col, id_col, target,
                          cost_lookup_1x, model_type='ridge',
                          nq=5, buffer_pct=0.20, min_train=60,
                          top_k_features=20):
    """
    Walk-forward ML prediction tracking both top and bottom quintiles.
    Returns per-month result DataFrame and per-month prediction ranks.
    """
    data = df.copy()
    data = data.sort_values([date_col, id_col]).reset_index(drop=True)

    unique_dates = sorted(data[date_col].unique())
    n_dates = len(unique_dates)

    if n_dates < min_train + 12:
        print(f"    WARNING: Only {n_dates} months, need {min_train + 12}. Skipping.")
        return None, None

    data = data.merge(cost_lookup_1x, on=[date_col, id_col], how='left')
    global_med_1x = cost_lookup_1x['cost_bps_1x'].median()
    data['cost_bps_1x'] = data['cost_bps_1x'].fillna(global_med_1x)

    avail_features = [f for f in features if f in data.columns]
    feat_coverage = data[avail_features].notna().mean()
    avail_features = feat_coverage[feat_coverage > 0.1].index.tolist()

    if len(avail_features) == 0:
        print(f"    WARNING: No features with >10% coverage. Skipping.")
        return None, None

    top_threshold = 1.0 - 1.0 / nq
    bottom_threshold = 1.0 / nq
    buffer = buffer_pct * (1.0 / nq)

    prev_long = set()
    prev_short = set()
    records = []
    all_pred_ranks = []

    for t_idx in range(min_train, n_dates):
        train_dates = unique_dates[:t_idx]
        test_date = unique_dates[t_idx]

        train_data = data[data[date_col].isin(train_dates)]
        test_data = data[data[date_col] == test_date].copy()

        if len(train_data) < 100 or len(test_data) < nq:
            continue

        X_train = train_data[avail_features].copy()
        y_train = train_data[target].values
        X_test = test_data[avail_features].copy()

        imputer = SimpleImputer(strategy='median')
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        # Feature selection inside walk-forward loop
        if model_type in ['xgboost', 'lightgbm'] and len(avail_features) > top_k_features:
            selector = xgb.XGBRegressor(
                n_estimators=100, max_depth=3, learning_rate=0.1,
                min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0
            )
            mask = ~np.isnan(y_train)
            if mask.sum() > 50:
                selector.fit(X_train_imp[mask], y_train[mask])
                importances = selector.feature_importances_
                top_idx = np.argsort(importances)[-top_k_features:]
            else:
                top_idx = np.arange(min(top_k_features, X_train_imp.shape[1]))
        else:
            top_idx = np.arange(X_train_imp.shape[1])

        X_train_sel = X_train_imp[:, top_idx]
        X_test_sel = X_test_imp[:, top_idx]

        mask = ~np.isnan(y_train)
        X_train_clean = X_train_sel[mask]
        y_train_clean = y_train[mask]

        if len(y_train_clean) < 50:
            continue

        try:
            if model_type == 'ridge':
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_sel[mask])
                X_test_scaled = scaler.transform(X_test_sel)
                model = Ridge(alpha=100, random_state=42)
                model.fit(X_train_scaled, y_train_clean)
                preds = model.predict(X_test_scaled)

            elif model_type == 'xgboost':
                model = xgb.XGBRegressor(
                    n_estimators=200, max_depth=3, learning_rate=0.05,
                    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                    random_state=42, verbosity=0
                )
                model.fit(X_train_clean, y_train_clean)
                preds = model.predict(X_test_sel)

            elif model_type == 'lightgbm':
                model = lgb.LGBMRegressor(
                    n_estimators=200, max_depth=3, learning_rate=0.05,
                    min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
                    random_state=42, verbosity=-1
                )
                model.fit(X_train_clean, y_train_clean)
                preds = model.predict(X_test_sel)

            else:
                raise ValueError(f"Unknown model type: {model_type}")

        except Exception as e:
            print(f"    Model error at {test_date}: {e}")
            continue

        test_data = test_data.copy()
        test_data['prediction'] = preds
        test_data['pct_rank'] = test_data['prediction'].rank(pct=True)

        # Save prediction ranks for blend
        pred_rank_df = test_data[[date_col, id_col, 'pct_rank']].copy()
        pred_rank_df = pred_rank_df.rename(columns={'pct_rank': 'ml_rank'})
        all_pred_ranks.append(pred_rank_df)

        # --- TOP QUINTILE (Q5) with buffer ---
        new_long = buffer_select_long(test_data, id_col, 'pct_rank', prev_long,
                                       top_threshold, buffer, nq)

        # --- BOTTOM QUINTILE (Q1) with buffer ---
        new_short = buffer_select_short(test_data, id_col, 'pct_rank', prev_short,
                                         bottom_threshold, buffer, nq)

        # Returns
        long_held = test_data[test_data[id_col].isin(new_long)]
        short_held = test_data[test_data[id_col].isin(new_short)]
        long_gross_ret = long_held[target].mean() if len(long_held) > 0 else 0.0
        short_gross_ret = short_held[target].mean() if len(short_held) > 0 else 0.0

        # Turnover — long leg
        if prev_long:
            long_entries = new_long - prev_long
            long_exits = prev_long - new_long
            long_turn = (len(long_entries) + len(long_exits)) / (2 * max(len(new_long), 1))
        else:
            long_entries = new_long
            long_exits = set()
            long_turn = np.nan

        # Turnover — short leg
        if prev_short:
            short_entries = new_short - prev_short
            short_exits = prev_short - new_short
            short_turn = (len(short_entries) + len(short_exits)) / (2 * max(len(new_short), 1))
        else:
            short_entries = new_short
            short_exits = set()
            short_turn = np.nan

        ic = monthly_ic(test_data['prediction'].values, test_data[target].values)

        long_cost_drags = {}
        short_cost_drags = {}
        for cm in COST_MULTS:
            long_cost_drags[cm] = compute_cost_drag(
                test_data, long_entries, long_exits, new_long, id_col,
                cm, global_med_1x, prev_long
            )
            short_cost_drags[cm] = compute_cost_drag(
                test_data, short_entries, short_exits, new_short, id_col,
                cm, global_med_1x, prev_short
            )

        records.append({
            'date': test_date,
            'long_gross_ret': long_gross_ret,
            'short_gross_ret': short_gross_ret,
            'long_turnover': long_turn,
            'short_turnover': short_turn,
            'n_long': len(new_long),
            'n_short': len(new_short),
            'ic': ic,
            **{f'long_cost_drag_{cm}': long_cost_drags[cm] for cm in COST_MULTS},
            **{f'short_cost_drag_{cm}': short_cost_drags[cm] for cm in COST_MULTS},
        })
        prev_long = new_long
        prev_short = new_short

    if len(records) == 0:
        return None, None

    result_df = pd.DataFrame(records).set_index('date')
    pred_ranks_df = pd.concat(all_pred_ranks, ignore_index=True) if all_pred_ranks else None
    return result_df, pred_ranks_df


# ==============================================================================
# MOMENTUM-ML BLEND — tracks BOTH Q5 and Q1
# ==============================================================================

def run_momentum_ml_blend_ls(df, ridge_pred_ranks, date_col, id_col, target,
                              cost_lookup_1x, nq=5, buffer_pct=0.20,
                              mom_weight=0.8, ml_weight=0.2):
    """Blend: 0.8*momentum_rank + 0.2*Ridge_rank. Track both Q5 and Q1."""
    if ridge_pred_ranks is None or len(ridge_pred_ranks) == 0:
        return None

    data = df.copy()

    mom_feat = 'ret_12_1'
    if mom_feat not in data.columns:
        return None
    data['mom_rank'] = data.groupby(date_col)[mom_feat].rank(pct=True)
    data['mom_rank'] = data['mom_rank'].fillna(0.5)

    data = data.merge(ridge_pred_ranks, on=[date_col, id_col], how='left')
    data['ml_rank'] = data['ml_rank'].fillna(0.5)

    data['blend_score'] = mom_weight * data['mom_rank'] + ml_weight * data['ml_rank']
    data['pct_rank'] = data.groupby(date_col)['blend_score'].rank(pct=True)

    top_threshold = 1.0 - 1.0 / nq
    bottom_threshold = 1.0 / nq
    buffer = buffer_pct * (1.0 / nq)

    data = data.merge(cost_lookup_1x, on=[date_col, id_col], how='left')
    global_med_1x = cost_lookup_1x['cost_bps_1x'].median()
    data['cost_bps_1x'] = data['cost_bps_1x'].fillna(global_med_1x)

    blend_dates = set(ridge_pred_ranks[date_col].unique())
    dates = sorted([d for d in data[date_col].unique() if d in blend_dates])

    prev_long = set()
    prev_short = set()
    records = []

    for date in dates:
        month = data[data[date_col] == date].copy()
        if len(month) < nq:
            continue

        # --- TOP QUINTILE ---
        new_long = buffer_select_long(month, id_col, 'pct_rank', prev_long,
                                       top_threshold, buffer, nq)

        # --- BOTTOM QUINTILE ---
        new_short = buffer_select_short(month, id_col, 'pct_rank', prev_short,
                                         bottom_threshold, buffer, nq)

        long_held = month[month[id_col].isin(new_long)]
        short_held = month[month[id_col].isin(new_short)]
        long_gross_ret = long_held[target].mean() if len(long_held) > 0 else 0.0
        short_gross_ret = short_held[target].mean() if len(short_held) > 0 else 0.0

        if prev_long:
            long_entries = new_long - prev_long
            long_exits = prev_long - new_long
            long_turn = (len(long_entries) + len(long_exits)) / (2 * max(len(new_long), 1))
        else:
            long_entries = new_long
            long_exits = set()
            long_turn = np.nan

        if prev_short:
            short_entries = new_short - prev_short
            short_exits = prev_short - new_short
            short_turn = (len(short_entries) + len(short_exits)) / (2 * max(len(new_short), 1))
        else:
            short_entries = new_short
            short_exits = set()
            short_turn = np.nan

        ic = monthly_ic(month['blend_score'].values, month[target].values)

        long_cost_drags = {}
        short_cost_drags = {}
        for cm in COST_MULTS:
            long_cost_drags[cm] = compute_cost_drag(
                month, long_entries, long_exits, new_long, id_col,
                cm, global_med_1x, prev_long
            )
            short_cost_drags[cm] = compute_cost_drag(
                month, short_entries, short_exits, new_short, id_col,
                cm, global_med_1x, prev_short
            )

        records.append({
            'date': date,
            'long_gross_ret': long_gross_ret,
            'short_gross_ret': short_gross_ret,
            'long_turnover': long_turn,
            'short_turnover': short_turn,
            'n_long': len(new_long),
            'n_short': len(new_short),
            'ic': ic,
            **{f'long_cost_drag_{cm}': long_cost_drags[cm] for cm in COST_MULTS},
            **{f'short_cost_drag_{cm}': short_cost_drags[cm] for cm in COST_MULTS},
        })
        prev_long = new_long
        prev_short = new_short

    if len(records) == 0:
        return None
    return pd.DataFrame(records).set_index('date')


# ==============================================================================
# EVALUATION — compute metrics for LongOnly, ShortOnly, LongShort
# ==============================================================================

def evaluate_portfolio_types(result_df, cost_mult, period_label, period_mask,
                              strategy_name, market):
    """
    Compute summary metrics for three portfolio types at a given cost level.
    Returns a list of result dicts (one per portfolio type).
    """
    if result_df is None:
        return []

    subset = result_df[period_mask]
    if len(subset) < 6:
        return []

    long_cost_col = f'long_cost_drag_{cost_mult}'
    short_cost_col = f'short_cost_drag_{cost_mult}'

    results = []

    # --- LONG-ONLY: top quintile, costs on long side only ---
    long_net = subset['long_gross_ret'] - subset[long_cost_col]
    results.append({
        'Market': market,
        'Strategy': strategy_name,
        'PortfolioType': 'LongOnly',
        'CostMult': cost_mult,
        'Period': period_label,
        'Net_SR': round(sharpe(long_net), 4),
        'Gross_SR': round(sharpe(subset['long_gross_ret']), 4),
        'Ann_Ret': round(long_net.mean() * 12, 4),
        'Turnover': round(subset['long_turnover'].mean(), 4),
        'Avg_Cost_bps': round(subset[long_cost_col].mean() * 10000, 1),
        'N_months': len(subset),
    })

    # --- SHORT-ONLY: bottom quintile, returns negated, costs + borrow ---
    # Short position profit = -Q1_ret (you profit when Q1 drops)
    # Short costs = trading costs on short leg + borrowing cost
    short_gross = -subset['short_gross_ret']  # negate: profit when shorts fall
    short_trading_cost = subset[short_cost_col]
    short_borrow = SHORT_BORROW_COST_MONTHLY  # 50 bps/month flat
    short_net = short_gross - short_trading_cost - short_borrow

    results.append({
        'Market': market,
        'Strategy': strategy_name,
        'PortfolioType': 'ShortOnly',
        'CostMult': cost_mult,
        'Period': period_label,
        'Net_SR': round(sharpe(short_net), 4),
        'Gross_SR': round(sharpe(short_gross), 4),
        'Ann_Ret': round(short_net.mean() * 12, 4),
        'Turnover': round(subset['short_turnover'].mean(), 4),
        'Avg_Cost_bps': round((short_trading_cost.mean() + short_borrow) * 10000, 1),
        'N_months': len(subset),
    })

    # --- LONG-SHORT: Q5 - Q1, costs on both legs + borrow on short ---
    ls_gross = subset['long_gross_ret'] - subset['short_gross_ret']
    ls_cost = subset[long_cost_col] + subset[short_cost_col] + short_borrow
    ls_net = ls_gross - ls_cost

    results.append({
        'Market': market,
        'Strategy': strategy_name,
        'PortfolioType': 'LongShort',
        'CostMult': cost_mult,
        'Period': period_label,
        'Net_SR': round(sharpe(ls_net), 4),
        'Gross_SR': round(sharpe(ls_gross), 4),
        'Ann_Ret': round(ls_net.mean() * 12, 4),
        'Turnover': round((subset['long_turnover'].mean() + subset['short_turnover'].mean()) / 2, 4),
        'Avg_Cost_bps': round(ls_cost.mean() * 10000, 1),
        'N_months': len(subset),
    })

    return results


# ==============================================================================
# MAIN
# ==============================================================================

print("=" * 110)
print("EXPERIMENT AY: Long-Short and Short-Only Analysis")
print(f"  Markets: {list(MARKETS.keys())}")
print(f"  Strategies: {STRATEGIES}")
print(f"  Cost multipliers: {COST_MULTS}")
print(f"  Portfolio types: LongOnly, ShortOnly, LongShort")
print(f"  Short borrowing cost: {SHORT_BORROW_COST_MONTHLY*10000:.0f} bps/month")
print(f"  Buffer: {BUFFER_PCT:.0%}")
print(f"  ML min training: {MIN_TRAIN_MONTHS} months")
print("=" * 110)

all_results = []
t0 = time.time()

for mkt_name, cfg in MARKETS.items():
    print(f"\n\n{'#'*110}")
    print(f"# MARKET: {mkt_name}")
    print(f"{'#'*110}")

    try:
        df, ml_features, date_col, id_col, target, cost_lookup_1x = \
            load_and_prepare(mkt_name, cfg)
    except Exception as e:
        print(f"  ERROR loading {mkt_name}: {e}")
        continue

    nq = cfg['nq']

    # IS/OOS split: 80/20
    unique_dates = sorted(df[date_col].unique())
    n_total_months = len(unique_dates)
    is_split_idx = int(n_total_months * 0.8)
    if is_split_idx < MIN_TRAIN_MONTHS:
        is_split_idx = MIN_TRAIN_MONTHS
    is_end_date = unique_dates[min(is_split_idx - 1, n_total_months - 1)]
    oos_start_date = unique_dates[min(is_split_idx, n_total_months - 1)]

    print(f"  IS/OOS split: IS ends {is_end_date}, OOS starts {oos_start_date}")
    print(f"  IS months: {is_split_idx}, OOS months: {n_total_months - is_split_idx}")

    def make_masks(result_df):
        idx = result_df.index
        oos_mask = idx > is_end_date
        full_mask = pd.Series(True, index=idx)
        return [('OOS', oos_mask), ('Full', full_mask)]

    strat_results = {}

    # ------------------------------------------------------------------
    # STRATEGY 1: MomentumSort
    # ------------------------------------------------------------------
    print(f"\n  --- MomentumSort ---")
    mom_feats = [f for f in MOMENTUM_FEATURES if f in df.columns]
    result = run_factor_sort_ls(
        df, mom_feats, date_col, id_col, target, cost_lookup_1x,
        nq=nq, buffer_pct=BUFFER_PCT
    )
    strat_results['MomentumSort'] = result
    if result is not None:
        for period_label, mask in make_masks(result):
            for cm in COST_MULTS:
                metrics_list = evaluate_portfolio_types(result, cm, period_label, mask,
                                                        'MomentumSort', mkt_name)
                all_results.extend(metrics_list)
        oos = result[result.index > is_end_date]
        if len(oos) > 0:
            sr_l = sharpe(oos['long_gross_ret'] - oos['long_cost_drag_3.0'])
            sr_s = sharpe(-oos['short_gross_ret'] - oos['short_cost_drag_3.0'] - SHORT_BORROW_COST_MONTHLY)
            print(f"    OOS LongOnly SR@3x: {sr_l:.3f}, ShortOnly SR@3x: {sr_s:.3f}")

    # ------------------------------------------------------------------
    # STRATEGY 2: LowVolSort
    # ------------------------------------------------------------------
    print(f"\n  --- LowVolSort ---")
    lowvol_feats = [f for f in LOWVOL_FEATURES if f in df.columns]
    result = run_factor_sort_ls(
        df, lowvol_feats, date_col, id_col, target, cost_lookup_1x,
        nq=nq, buffer_pct=BUFFER_PCT, ascending_features=lowvol_feats
    )
    strat_results['LowVolSort'] = result
    if result is not None:
        for period_label, mask in make_masks(result):
            for cm in COST_MULTS:
                metrics_list = evaluate_portfolio_types(result, cm, period_label, mask,
                                                        'LowVolSort', mkt_name)
                all_results.extend(metrics_list)
        oos = result[result.index > is_end_date]
        if len(oos) > 0:
            sr_l = sharpe(oos['long_gross_ret'] - oos['long_cost_drag_3.0'])
            sr_s = sharpe(-oos['short_gross_ret'] - oos['short_cost_drag_3.0'] - SHORT_BORROW_COST_MONTHLY)
            print(f"    OOS LongOnly SR@3x: {sr_l:.3f}, ShortOnly SR@3x: {sr_s:.3f}")

    # ------------------------------------------------------------------
    # STRATEGY 3: CompositeSort
    # ------------------------------------------------------------------
    print(f"\n  --- CompositeSort ---")
    quality_feats = [f for f in QUALITY_FEATURES if f in df.columns]
    composite_feats = mom_feats + quality_feats + lowvol_feats
    ascending_comp = [f for f in composite_feats
                      if 'accruals' in f.lower() or 'ivol' in f.lower() or 'rvol' in f.lower()]
    result = run_factor_sort_ls(
        df, composite_feats, date_col, id_col, target, cost_lookup_1x,
        nq=nq, buffer_pct=BUFFER_PCT, ascending_features=ascending_comp
    )
    strat_results['CompositeSort'] = result
    if result is not None:
        for period_label, mask in make_masks(result):
            for cm in COST_MULTS:
                metrics_list = evaluate_portfolio_types(result, cm, period_label, mask,
                                                        'CompositeSort', mkt_name)
                all_results.extend(metrics_list)
        oos = result[result.index > is_end_date]
        if len(oos) > 0:
            sr_l = sharpe(oos['long_gross_ret'] - oos['long_cost_drag_3.0'])
            sr_s = sharpe(-oos['short_gross_ret'] - oos['short_cost_drag_3.0'] - SHORT_BORROW_COST_MONTHLY)
            print(f"    OOS LongOnly SR@3x: {sr_l:.3f}, ShortOnly SR@3x: {sr_s:.3f}")

    # ------------------------------------------------------------------
    # STRATEGY 4: Ridge
    # ------------------------------------------------------------------
    print(f"\n  --- Ridge ---")
    result, ridge_pred_ranks = run_ml_walkforward_ls(
        df, ml_features, date_col, id_col, target, cost_lookup_1x,
        model_type='ridge', nq=nq, buffer_pct=BUFFER_PCT,
        min_train=MIN_TRAIN_MONTHS
    )
    strat_results['Ridge'] = result
    if result is not None:
        for period_label, mask in make_masks(result):
            for cm in COST_MULTS:
                metrics_list = evaluate_portfolio_types(result, cm, period_label, mask,
                                                        'Ridge', mkt_name)
                all_results.extend(metrics_list)
        oos = result[result.index > is_end_date]
        if len(oos) > 0:
            sr_l = sharpe(oos['long_gross_ret'] - oos['long_cost_drag_3.0'])
            sr_s = sharpe(-oos['short_gross_ret'] - oos['short_cost_drag_3.0'] - SHORT_BORROW_COST_MONTHLY)
            print(f"    OOS LongOnly SR@3x: {sr_l:.3f}, ShortOnly SR@3x: {sr_s:.3f}")

    # ------------------------------------------------------------------
    # STRATEGY 5: XGBoost
    # ------------------------------------------------------------------
    print(f"\n  --- XGBoost ---")
    result, _ = run_ml_walkforward_ls(
        df, ml_features, date_col, id_col, target, cost_lookup_1x,
        model_type='xgboost', nq=nq, buffer_pct=BUFFER_PCT,
        min_train=MIN_TRAIN_MONTHS, top_k_features=20
    )
    strat_results['XGBoost'] = result
    if result is not None:
        for period_label, mask in make_masks(result):
            for cm in COST_MULTS:
                metrics_list = evaluate_portfolio_types(result, cm, period_label, mask,
                                                        'XGBoost', mkt_name)
                all_results.extend(metrics_list)
        oos = result[result.index > is_end_date]
        if len(oos) > 0:
            sr_l = sharpe(oos['long_gross_ret'] - oos['long_cost_drag_3.0'])
            sr_s = sharpe(-oos['short_gross_ret'] - oos['short_cost_drag_3.0'] - SHORT_BORROW_COST_MONTHLY)
            print(f"    OOS LongOnly SR@3x: {sr_l:.3f}, ShortOnly SR@3x: {sr_s:.3f}")

    # ------------------------------------------------------------------
    # STRATEGY 6: LightGBM
    # ------------------------------------------------------------------
    print(f"\n  --- LightGBM ---")
    result, _ = run_ml_walkforward_ls(
        df, ml_features, date_col, id_col, target, cost_lookup_1x,
        model_type='lightgbm', nq=nq, buffer_pct=BUFFER_PCT,
        min_train=MIN_TRAIN_MONTHS, top_k_features=20
    )
    strat_results['LightGBM'] = result
    if result is not None:
        for period_label, mask in make_masks(result):
            for cm in COST_MULTS:
                metrics_list = evaluate_portfolio_types(result, cm, period_label, mask,
                                                        'LightGBM', mkt_name)
                all_results.extend(metrics_list)
        oos = result[result.index > is_end_date]
        if len(oos) > 0:
            sr_l = sharpe(oos['long_gross_ret'] - oos['long_cost_drag_3.0'])
            sr_s = sharpe(-oos['short_gross_ret'] - oos['short_cost_drag_3.0'] - SHORT_BORROW_COST_MONTHLY)
            print(f"    OOS LongOnly SR@3x: {sr_l:.3f}, ShortOnly SR@3x: {sr_s:.3f}")

    # ------------------------------------------------------------------
    # STRATEGY 7: MomentumML_Blend
    # ------------------------------------------------------------------
    print(f"\n  --- MomentumML_Blend ---")
    result = run_momentum_ml_blend_ls(
        df, ridge_pred_ranks, date_col, id_col, target, cost_lookup_1x,
        nq=nq, buffer_pct=BUFFER_PCT, mom_weight=0.8, ml_weight=0.2
    )
    strat_results['MomentumML_Blend'] = result
    if result is not None:
        for period_label, mask in make_masks(result):
            for cm in COST_MULTS:
                metrics_list = evaluate_portfolio_types(result, cm, period_label, mask,
                                                        'MomentumML_Blend', mkt_name)
                all_results.extend(metrics_list)
        oos = result[result.index > is_end_date]
        if len(oos) > 0:
            sr_l = sharpe(oos['long_gross_ret'] - oos['long_cost_drag_3.0'])
            sr_s = sharpe(-oos['short_gross_ret'] - oos['short_cost_drag_3.0'] - SHORT_BORROW_COST_MONTHLY)
            print(f"    OOS LongOnly SR@3x: {sr_l:.3f}, ShortOnly SR@3x: {sr_s:.3f}")

    # --- Market summary ---
    mkt_oos_3x = [r for r in all_results
                   if r['Market'] == mkt_name and r['CostMult'] == 3.0 and r['Period'] == 'OOS']
    if mkt_oos_3x:
        print(f"\n  {'='*90}")
        print(f"  {mkt_name} OOS Summary @ 3x costs")
        print(f"  {'='*90}")
        print(f"  {'Strategy':<22s} {'Type':<12s} {'Net SR':>8s} {'Ann Ret':>9s} {'Turn':>7s} {'Cost':>8s}")
        print(f"  {'-'*70}")
        for r in sorted(mkt_oos_3x, key=lambda x: (x['Strategy'], x['PortfolioType'])):
            print(f"  {r['Strategy']:<22s} {r['PortfolioType']:<12s} {r['Net_SR']:>8.3f} "
                  f"{r['Ann_Ret']:>8.1%} {r['Turnover']:>6.1%} {r['Avg_Cost_bps']:>7.1f}")


# ==============================================================================
# SAVE RESULTS
# ==============================================================================

elapsed = time.time() - t0
print(f"\n\nTotal runtime: {elapsed/60:.1f} minutes")

df_results = pd.DataFrame(all_results)
out_path = 'scripts/concern_experiments/expAY_long_short.csv'
df_results.to_csv(out_path, index=False)
print(f"Saved {len(df_results)} rows to {out_path}")


# ==============================================================================
# ANALYSIS 1: OOS Net SR by Portfolio Type, averaged across markets (3x costs)
# ==============================================================================

print("\n\n" + "=" * 120)
print("ANALYSIS 1: Average OOS Net Sharpe Ratio by Strategy x Portfolio Type (3x costs)")
print("=" * 120)

oos_3x = [r for r in all_results if r['Period'] == 'OOS' and r['CostMult'] == 3.0]

header = f"  {'Strategy':<22s} {'LongOnly':>10s} {'ShortOnly':>10s} {'LongShort':>10s}"
print(header)
print(f"  {'-'*55}")

for s in STRATEGIES:
    row = f"  {s:<22s}"
    for pt in ['LongOnly', 'ShortOnly', 'LongShort']:
        vals = [r['Net_SR'] for r in oos_3x if r['Strategy'] == s and r['PortfolioType'] == pt]
        avg = np.mean(vals) if vals else np.nan
        row += f" {avg:>10.3f}" if not np.isnan(avg) else f" {'N/A':>10s}"
    print(row)

# Averages by group
print(f"  {'-'*55}")
for grp_name, grp_list in [('Sort Avg', SORT_STRATEGIES), ('ML Avg', ML_STRATEGIES)]:
    row = f"  {grp_name:<22s}"
    for pt in ['LongOnly', 'ShortOnly', 'LongShort']:
        vals = [r['Net_SR'] for r in oos_3x if r['Strategy'] in grp_list and r['PortfolioType'] == pt]
        avg = np.mean(vals) if vals else np.nan
        row += f" {avg:>10.3f}" if not np.isnan(avg) else f" {'N/A':>10s}"
    print(row)


# ==============================================================================
# ANALYSIS 2: Per-Market Breakdown (3x costs, OOS)
# ==============================================================================

print("\n\n" + "=" * 120)
print("ANALYSIS 2: Per-Market OOS Net SR at 3x Costs")
print("=" * 120)

for pt in ['LongOnly', 'ShortOnly', 'LongShort']:
    print(f"\n  --- {pt} ---")
    header = f"  {'Market':<6s}"
    for s in STRATEGIES:
        header += f" {s[:12]:>12s}"
    print(header)
    print(f"  {'-'*95}")

    for mkt in MARKETS.keys():
        row = f"  {mkt:<6s}"
        for s in STRATEGIES:
            match = [r for r in oos_3x
                     if r['Market'] == mkt and r['Strategy'] == s and r['PortfolioType'] == pt]
            if match:
                row += f" {match[0]['Net_SR']:>12.3f}"
            else:
                row += f" {'N/A':>12s}"
        print(row)

    # Average row
    row = f"  {'Avg':<6s}"
    for s in STRATEGIES:
        vals = [r['Net_SR'] for r in oos_3x if r['Strategy'] == s and r['PortfolioType'] == pt]
        avg = np.mean(vals) if vals else np.nan
        row += f" {avg:>12.3f}" if not np.isnan(avg) else f" {'N/A':>12s}"
    print(row)


# ==============================================================================
# ANALYSIS 3: THE KEY QUESTION — Where does ML's advantage concentrate?
# ==============================================================================

print("\n\n" + "=" * 120)
print("ANALYSIS 3: ML's Advantage by Portfolio Type (THE KEY TEST)")
print("  ML_alpha - MomentumSort_alpha for each portfolio type")
print("=" * 120)

for cm in COST_MULTS:
    oos_cm = [r for r in all_results if r['Period'] == 'OOS' and r['CostMult'] == cm]

    print(f"\n  --- Cost Level: {cm:.0f}x ---")
    print(f"  {'Strategy':<22s} {'Long Adv':>10s} {'Short Adv':>10s} {'LS Adv':>10s} {'Where?':>12s}")
    print(f"  {'-'*70}")

    # For each ML strategy, compute advantage over MomentumSort
    for ml_strat in ML_STRATEGIES + ['MomentumML_Blend']:
        long_diffs = []
        short_diffs = []
        ls_diffs = []

        for mkt in MARKETS.keys():
            for pt, diff_list in [('LongOnly', long_diffs), ('ShortOnly', short_diffs),
                                   ('LongShort', ls_diffs)]:
                mom_match = [r for r in oos_cm
                             if r['Market'] == mkt and r['Strategy'] == 'MomentumSort'
                             and r['PortfolioType'] == pt]
                ml_match = [r for r in oos_cm
                            if r['Market'] == mkt and r['Strategy'] == ml_strat
                            and r['PortfolioType'] == pt]
                if mom_match and ml_match:
                    diff_list.append(ml_match[0]['Net_SR'] - mom_match[0]['Net_SR'])

        avg_long = np.mean(long_diffs) if long_diffs else np.nan
        avg_short = np.mean(short_diffs) if short_diffs else np.nan
        avg_ls = np.mean(ls_diffs) if ls_diffs else np.nan

        # Determine where advantage concentrates
        if not np.isnan(avg_long) and not np.isnan(avg_short):
            if avg_short > avg_long and avg_short > 0:
                where = "SHORT SIDE"
            elif avg_long > avg_short and avg_long > 0:
                where = "LONG SIDE"
            elif avg_long < 0 and avg_short < 0:
                where = "NEITHER"
            else:
                where = "MIXED"
        else:
            where = "N/A"

        print(f"  {ml_strat:<22s} {avg_long:>+10.3f} {avg_short:>+10.3f} "
              f"{avg_ls:>+10.3f} {where:>12s}")


# ==============================================================================
# ANALYSIS 4: Detailed per-market ML vs Momentum on each side
# ==============================================================================

print("\n\n" + "=" * 120)
print("ANALYSIS 4: Per-Market ML Advantage (Best ML - MomentumSort) @ 3x costs")
print("=" * 120)

oos_3x_data = [r for r in all_results if r['Period'] == 'OOS' and r['CostMult'] == 3.0]

print(f"\n  {'Market':<6s} {'Long(ML)':>10s} {'Long(Mom)':>10s} {'L_Diff':>8s} "
      f"{'Short(ML)':>10s} {'Short(Mom)':>11s} {'S_Diff':>8s} "
      f"{'LS(ML)':>8s} {'LS(Mom)':>8s} {'LS_Diff':>8s}")
print(f"  {'-'*100}")

long_diffs_all = []
short_diffs_all = []
ls_diffs_all = []

for mkt in MARKETS.keys():
    # Best ML for each portfolio type
    row = f"  {mkt:<6s}"

    for pt, diff_list in [('LongOnly', long_diffs_all), ('ShortOnly', short_diffs_all),
                           ('LongShort', ls_diffs_all)]:
        mom = [r for r in oos_3x_data
               if r['Market'] == mkt and r['Strategy'] == 'MomentumSort' and r['PortfolioType'] == pt]
        ml_candidates = [r for r in oos_3x_data
                         if r['Market'] == mkt and r['Strategy'] in ML_STRATEGIES
                         and r['PortfolioType'] == pt]
        if mom and ml_candidates:
            best_ml = max(ml_candidates, key=lambda x: x['Net_SR'])
            diff = best_ml['Net_SR'] - mom[0]['Net_SR']
            diff_list.append(diff)
            row += f" {best_ml['Net_SR']:>10.3f} {mom[0]['Net_SR']:>10.3f} {diff:>+8.3f}"
        else:
            row += f" {'N/A':>10s} {'N/A':>10s} {'N/A':>8s}"

    print(row)

print(f"  {'-'*100}")
avg_l = np.mean(long_diffs_all) if long_diffs_all else np.nan
avg_s = np.mean(short_diffs_all) if short_diffs_all else np.nan
avg_ls = np.mean(ls_diffs_all) if ls_diffs_all else np.nan
print(f"  {'Avg':<6s} {'':>10s} {'':>10s} {avg_l:>+8.3f} "
      f"{'':>10s} {'':>11s} {avg_s:>+8.3f} "
      f"{'':>8s} {'':>8s} {avg_ls:>+8.3f}")


# ==============================================================================
# ANALYSIS 5: Cost Sensitivity by Portfolio Type
# ==============================================================================

print("\n\n" + "=" * 120)
print("ANALYSIS 5: Cost Sensitivity — Sort vs ML by Cost Level (OOS, averaged)")
print("=" * 120)

for pt in ['LongOnly', 'ShortOnly', 'LongShort']:
    print(f"\n  --- {pt} ---")
    print(f"  {'Strategy':<22s}", end='')
    for cm in COST_MULTS:
        print(f" {f'{cm:.0f}x':>8s}", end='')
    print(f" {'Degrade':>10s}")
    print(f"  {'-'*55}")

    for s in STRATEGIES:
        print(f"  {s:<22s}", end='')
        sr_by_cost = {}
        for cm in COST_MULTS:
            vals = [r['Net_SR'] for r in all_results
                    if r['Period'] == 'OOS' and r['CostMult'] == cm
                    and r['Strategy'] == s and r['PortfolioType'] == pt]
            avg = np.mean(vals) if vals else np.nan
            sr_by_cost[cm] = avg
            print(f" {avg:>8.3f}" if not np.isnan(avg) else f" {'N/A':>8s}", end='')

        # Degradation from 1x to 3x
        if COST_MULTS[0] in sr_by_cost and COST_MULTS[-1] in sr_by_cost:
            deg = sr_by_cost[COST_MULTS[0]] - sr_by_cost[COST_MULTS[-1]]
            if not np.isnan(deg):
                print(f" {deg:>+10.3f}")
            else:
                print(f" {'N/A':>10s}")
        else:
            print(f" {'N/A':>10s}")


# ==============================================================================
# ANALYSIS 6: Turnover Comparison (Long vs Short leg)
# ==============================================================================

print("\n\n" + "=" * 120)
print("ANALYSIS 6: Average Monthly Turnover — Long vs Short Leg (OOS)")
print("=" * 120)

print(f"  {'Strategy':<22s} {'Long Turn':>10s} {'Short Turn':>11s} {'Ratio(S/L)':>11s}")
print(f"  {'-'*56}")

for s in STRATEGIES:
    long_turns = [r['Turnover'] for r in all_results
                  if r['Period'] == 'OOS' and r['CostMult'] == COST_MULTS[0]
                  and r['Strategy'] == s and r['PortfolioType'] == 'LongOnly']
    short_turns = [r['Turnover'] for r in all_results
                   if r['Period'] == 'OOS' and r['CostMult'] == COST_MULTS[0]
                   and r['Strategy'] == s and r['PortfolioType'] == 'ShortOnly']
    avg_l = np.mean(long_turns) if long_turns else np.nan
    avg_s = np.mean(short_turns) if short_turns else np.nan
    ratio = avg_s / avg_l if avg_l and avg_l > 0 and not np.isnan(avg_s) else np.nan
    print(f"  {s:<22s} {avg_l:>9.1%} {avg_s:>10.1%} {ratio:>10.2f}x"
          if not np.isnan(ratio) else f"  {s:<22s} {'N/A':>10s} {'N/A':>11s} {'N/A':>11s}")


# ==============================================================================
# ANALYSIS 7: Statistical Tests
# ==============================================================================

print("\n\n" + "=" * 120)
print("ANALYSIS 7: Statistical Tests — Does ML Have MORE Short-Side Advantage?")
print("=" * 120)

# For each market, compute: (best_ML_short_SR - mom_short_SR) vs (best_ML_long_SR - mom_long_SR)
oos_3x_data = [r for r in all_results if r['Period'] == 'OOS' and r['CostMult'] == 3.0]

long_advantages = []  # ML advantage on long side per market
short_advantages = []  # ML advantage on short side per market

for mkt in MARKETS.keys():
    # Long side
    mom_long = [r for r in oos_3x_data
                if r['Market'] == mkt and r['Strategy'] == 'MomentumSort'
                and r['PortfolioType'] == 'LongOnly']
    ml_long = [r for r in oos_3x_data
               if r['Market'] == mkt and r['Strategy'] in ML_STRATEGIES
               and r['PortfolioType'] == 'LongOnly']

    # Short side
    mom_short = [r for r in oos_3x_data
                 if r['Market'] == mkt and r['Strategy'] == 'MomentumSort'
                 and r['PortfolioType'] == 'ShortOnly']
    ml_short = [r for r in oos_3x_data
                if r['Market'] == mkt and r['Strategy'] in ML_STRATEGIES
                and r['PortfolioType'] == 'ShortOnly']

    if mom_long and ml_long and mom_short and ml_short:
        best_ml_long = max(ml_long, key=lambda x: x['Net_SR'])['Net_SR']
        best_ml_short = max(ml_short, key=lambda x: x['Net_SR'])['Net_SR']
        long_adv = best_ml_long - mom_long[0]['Net_SR']
        short_adv = best_ml_short - mom_short[0]['Net_SR']
        long_advantages.append(long_adv)
        short_advantages.append(short_adv)
        print(f"  {mkt}: ML long advantage = {long_adv:+.3f}, ML short advantage = {short_adv:+.3f}, "
              f"difference (short-long) = {short_adv - long_adv:+.3f}")

if len(long_advantages) >= 3:
    diffs = [s - l for s, l in zip(short_advantages, long_advantages)]
    avg_long_adv = np.mean(long_advantages)
    avg_short_adv = np.mean(short_advantages)
    avg_diff = np.mean(diffs)

    print(f"\n  Summary:")
    print(f"    Avg ML long-side advantage over MomentumSort:  {avg_long_adv:+.3f}")
    print(f"    Avg ML short-side advantage over MomentumSort: {avg_short_adv:+.3f}")
    print(f"    Avg difference (short - long advantage):       {avg_diff:+.3f}")
    print(f"    ML has MORE short-side advantage: {'YES' if avg_diff > 0 else 'NO'}")

    n_short_wins = sum(1 for d in diffs if d > 0)
    print(f"    Short-side advantage larger in {n_short_wins}/{len(diffs)} markets")

    if len(diffs) >= 5:
        from scipy.stats import wilcoxon
        try:
            stat_w, pval_w = wilcoxon(diffs)
            print(f"    Wilcoxon signed-rank (short_adv - long_adv): stat={stat_w:.1f}, p={pval_w:.4f}")
        except Exception as e:
            print(f"    Wilcoxon test failed: {e}")

    # t-test
    from scipy.stats import ttest_rel
    try:
        t_stat, t_pval = ttest_rel(short_advantages, long_advantages)
        print(f"    Paired t-test (short vs long advantage): t={t_stat:.3f}, p={t_pval:.4f}")
    except Exception as e:
        print(f"    t-test failed: {e}")


# ==============================================================================
# KEY FINDINGS
# ==============================================================================

print("\n\n" + "=" * 120)
print("KEY FINDINGS")
print("=" * 120)

oos_3x_data = [r for r in all_results if r['Period'] == 'OOS' and r['CostMult'] == 3.0]

# 1. Avg SR by strategy group x portfolio type
print(f"\n  1. AVERAGE OOS NET SR @ 3x COSTS:")
print(f"     {'Group':<12s} {'LongOnly':>10s} {'ShortOnly':>10s} {'LongShort':>10s}")
for grp_name, grp_list in [('Sorts', SORT_STRATEGIES), ('ML', ML_STRATEGIES),
                             ('Blend', ['MomentumML_Blend'])]:
    vals = {}
    for pt in ['LongOnly', 'ShortOnly', 'LongShort']:
        v = [r['Net_SR'] for r in oos_3x_data if r['Strategy'] in grp_list and r['PortfolioType'] == pt]
        vals[pt] = np.mean(v) if v else np.nan
    print(f"     {grp_name:<12s} {vals['LongOnly']:>10.3f} {vals['ShortOnly']:>10.3f} "
          f"{vals['LongShort']:>10.3f}")

# 2. ML advantage decomposition
print(f"\n  2. ML ADVANTAGE DECOMPOSITION (vs MomentumSort, avg across markets):")
if long_advantages and short_advantages:
    print(f"     Long-side advantage:  {np.mean(long_advantages):+.3f} SR units")
    print(f"     Short-side advantage: {np.mean(short_advantages):+.3f} SR units")
    net_diff = np.mean(short_advantages) - np.mean(long_advantages)
    print(f"     Short - Long diff:    {net_diff:+.3f} SR units")
    if net_diff > 0:
        print(f"     => ML's advantage concentrates MORE on the SHORT side")
        print(f"        This reconciles with H&K (2023): ML shines in long-short because")
        print(f"        it better identifies stocks to AVOID, not stocks to buy.")
    else:
        print(f"     => ML's advantage does NOT concentrate more on the short side")
        print(f"        ML may have comparable or better long-side selection.")

# 3. Turnover story
print(f"\n  3. TURNOVER BY LEG:")
for s in STRATEGIES:
    lt = [r['Turnover'] for r in oos_3x_data if r['Strategy'] == s and r['PortfolioType'] == 'LongOnly']
    st = [r['Turnover'] for r in oos_3x_data if r['Strategy'] == s and r['PortfolioType'] == 'ShortOnly']
    if lt and st:
        print(f"     {s:<22s}: Long {np.mean(lt):.1%}, Short {np.mean(st):.1%}")

# 4. Is short-only profitable at all?
print(f"\n  4. IS SHORT-ONLY PROFITABLE? (OOS, 3x costs, incl. 50bps/mo borrow)")
for s in STRATEGIES:
    vals = [r['Net_SR'] for r in oos_3x_data if r['Strategy'] == s and r['PortfolioType'] == 'ShortOnly']
    if vals:
        avg = np.mean(vals)
        n_pos = sum(1 for v in vals if v > 0)
        print(f"     {s:<22s}: Avg SR = {avg:>+.3f}, Positive in {n_pos}/{len(vals)} markets")

# 5. Long-short vs long-only
print(f"\n  5. LONG-SHORT vs LONG-ONLY (OOS, 3x costs):")
for s in STRATEGIES:
    lo_vals = [r['Net_SR'] for r in oos_3x_data if r['Strategy'] == s and r['PortfolioType'] == 'LongOnly']
    ls_vals = [r['Net_SR'] for r in oos_3x_data if r['Strategy'] == s and r['PortfolioType'] == 'LongShort']
    if lo_vals and ls_vals:
        lo_avg = np.mean(lo_vals)
        ls_avg = np.mean(ls_vals)
        diff = ls_avg - lo_avg
        print(f"     {s:<22s}: LO={lo_avg:>+.3f}, LS={ls_avg:>+.3f}, Diff={diff:>+.3f} "
              f"{'(LS better)' if diff > 0 else '(LO better)'}")

print(f"\n  6. IMPLICATION:")
print(f"     If ML's advantage is on the short side but shorting is expensive (50bps/mo borrow),")
print(f"     then ML's theoretical superiority is not implementable in EM.")
print(f"     This explains why simple momentum sort dominates in long-only top-quintile.")

print("\n\nDone.")
