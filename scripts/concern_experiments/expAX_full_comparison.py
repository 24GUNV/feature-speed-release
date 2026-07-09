"""
Experiment AX: ML vs Simple Sorts across ALL 15 EM Markets — Extended OOS

Core experiment for ICAIF 2026 paper: "The Implementable Frontier in Emerging Markets."

Tests 7 strategies across 15 EM markets with:
  - Extended OOS: 2016-01 to 2023-12 (~95 months), IS ends ~2015-12 (80/20 split)
  - Multiple cost levels: 1x, 1.5x, 2x, 2.5x, 3x half-spread
  - Walk-forward expanding window, min 60 months training

Strategies:
  1. MomentumSort     — sort by ret_12_1, top quintile, 20% buffer
  2. LowVolSort       — sort by ivol_capm_21d ASCENDING (low vol = good), top quintile, 20% buffer
  3. CompositeSort     — equal-weight rank of momentum + quality + low-vol, top quintile, 20% buffer
  4. Ridge             — Ridge regression, walk-forward, top quintile
  5. XGBoost           — XGBoost, walk-forward, top-20 feature selection, top quintile
  6. LightGBM          — LightGBM, walk-forward, top-20 feature selection, top quintile
  7. MomentumML_Blend  — 0.8 * momentum_rank + 0.2 * Ridge_rank, top quintile

Pipeline rules (from CLAUDE.md):
  - Lag BEFORE filter (lag on full dataset, then filter to top-N by mcap)
  - Accounting features lag=1 (JKP already applies 4-month lag), market features lag=1
  - Drop 'me' from modeling features
  - Feature selection INSIDE walk-forward loop (ML strategies only)
  - Transaction costs: (bidaskhl_21d / 2) * 10000 * COST_MULT, clipped [20, 200] bps
  - np.random.seed(42)

Output: scripts/concern_experiments/expAX_full_comparison.csv

Usage: python3 scripts/concern_experiments/expAX_full_comparison.py
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
import xgboost as xgb
import lightgbm as lgb
from scipy import stats

from data_loader import load_equity_data, lag_features

np.random.seed(42)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

from market_config import MARKETS, COST_MULTS, BUFFER_PCT, MIN_TRAIN_MONTHS

# IS/OOS split: use 80/20 where IS ends ~2015-12, OOS = 2016-01 to 2023-12
# Applied dynamically per market based on available data.

STRATEGIES = [
    'MomentumSort', 'LowVolSort', 'CompositeSort',
    'Ridge', 'XGBoost', 'LightGBM',
    'MomentumML_Blend',
]
SORT_STRATEGIES = ['MomentumSort', 'LowVolSort', 'CompositeSort']
ML_STRATEGIES = ['Ridge', 'XGBoost', 'LightGBM']

# Feature classification: accounting (lag=1) vs market (lag=1)
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
LOWVOL_FEATURES = ['ivol_capm_21d']  # lower = better
# Quality features for composite
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


def max_drawdown(s):
    """Max drawdown from monthly return series."""
    s = s.dropna()
    if len(s) == 0:
        return 0.0
    cum = (1 + s).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    return dd.min()


def monthly_ic(predictions, actuals):
    """Rank IC (Spearman correlation)."""
    if len(predictions) < 5:
        return np.nan
    ic, _ = stats.spearmanr(predictions, actuals)
    return ic


def classify_features(all_features):
    """Classify features into accounting (lag=1) vs market (lag=1)."""
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
    """
    Build per-stock per-month transaction cost lookup at 1x Corwin-Schultz.
    Caller scales by COST_MULT at portfolio construction time.
    """
    if spread_col not in raw_df.columns:
        lookup = raw_df[[date_col, id_col]].copy()
        lookup['cost_bps_1x'] = 50.0
        return lookup
    lookup = raw_df[[date_col, id_col, spread_col]].copy()
    lookup['cost_bps_1x'] = (lookup[spread_col] / 2) * 10000
    monthly_median = lookup.groupby(date_col)['cost_bps_1x'].transform('median')
    lookup['cost_bps_1x'] = lookup['cost_bps_1x'].fillna(monthly_median)
    # Clip at a reasonable range for 1x (will be re-scaled+re-clipped by cost_mult)
    lookup['cost_bps_1x'] = lookup['cost_bps_1x'].clip(lower=floor_bps, upper=cap_bps)
    return lookup[[date_col, id_col, 'cost_bps_1x']]


def compute_cost_drag(month_data, entries, exits, new_long, id_col,
                      cost_mult, global_med_1x, prev_long):
    """Compute per-month cost drag given a cost multiplier."""
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
# DATA LOADING
# ==============================================================================

def load_and_prepare(mkt_name, cfg):
    """Load data, apply dual-lag (accounting=4, market=1), filter to top-N."""
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

    print(f"  Accounting features (lag=1): {len(accounting_feats)}")
    print(f"  Market features (lag=1): {len(market_feats)}")

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
        print(f"  Filtered to top {topn} by market cap")

    # Build 1x cost lookup
    cost_lookup_1x = build_cost_lookup_1x(df, date_col, id_col)

    # ML features: everything except excluded columns and me
    ml_features = [f for f in accounting_feats + market_feats if f in df.columns]

    df[date_col] = pd.to_datetime(df[date_col])
    n_months = df[date_col].nunique()
    avg_stocks = df.groupby(date_col)[id_col].nunique().mean()
    print(f"  Final: {len(df):,} rows, {n_months} months, avg {avg_stocks:.0f} stocks/month")
    print(f"  ML features: {len(ml_features)}")

    return df, ml_features, date_col, id_col, target, cost_lookup_1x


# ==============================================================================
# FACTOR SORT STRATEGY (computes gross returns + turnover)
# ==============================================================================

def run_factor_sort(df, sort_features, date_col, id_col, target,
                    cost_lookup_1x, nq=5, buffer_pct=0.20,
                    ascending_features=None):
    """
    Run a simple factor sort, returning per-month gross returns and
    enough info to apply variable costs.

    ascending_features: features where LOWER is BETTER (e.g., volatility, accruals).
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

    for date in dates:
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

        ic = monthly_ic(month['composite_score'].values, month[target].values)

        cost_drags = {}
        for cm in COST_MULTS:
            cost_drags[cm] = compute_cost_drag(
                month, entries, exits, new_long, id_col,
                cm, global_med_1x, prev_long
            )

        records.append({
            'date': date, 'gross_ret': gross_ret, 'turnover': turn,
            'n_held': len(new_long), 'ic': ic,
            **{f'cost_drag_{cm}': cost_drags[cm] for cm in COST_MULTS},
        })
        prev_long = new_long

    if len(records) == 0:
        return None
    return pd.DataFrame(records).set_index('date')


# ==============================================================================
# ML WALK-FORWARD STRATEGY (computes gross returns + turnover)
# ==============================================================================

def run_ml_walkforward(df, features, date_col, id_col, target,
                       cost_lookup_1x, model_type='ridge',
                       nq=5, buffer_pct=0.20, min_train=60,
                       top_k_features=20):
    """
    Walk-forward ML prediction with gross returns and variable-cost info.
    Returns per-month result DataFrame and also per-month prediction ranks
    (needed for MomentumML_Blend).
    """
    data = df.copy()
    data = data.sort_values([date_col, id_col]).reset_index(drop=True)

    # Pre-compute rank target (cross-sectional percentile ranks)
    data['_rank_target'] = data.groupby(date_col)[target].rank(pct=True)

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
    buffer = buffer_pct * (1.0 / nq)

    prev_long = set()
    records = []
    # Store per-stock prediction ranks for blend strategy
    all_pred_ranks = []

    for t_idx in range(min_train, n_dates):
        train_dates = unique_dates[:t_idx]
        test_date = unique_dates[t_idx]

        train_data = data[data[date_col].isin(train_dates)]
        test_data = data[data[date_col] == test_date].copy()

        if len(train_data) < 100 or len(test_data) < nq:
            continue

        X_train = train_data[avail_features].copy()
        y_train = train_data['_rank_target'].values
        X_test = test_data[avail_features].copy()

        imputer = SimpleImputer(strategy='median')
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        # Feature selection INSIDE walk-forward loop (for tree models)
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

        # Save per-stock prediction ranks for blend
        pred_rank_df = test_data[[date_col, id_col, 'pct_rank']].copy()
        pred_rank_df = pred_rank_df.rename(columns={'pct_rank': 'ml_rank'})
        all_pred_ranks.append(pred_rank_df)

        # Buffer-based selection
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

        ic = monthly_ic(test_data['prediction'].values, test_data[target].values)

        cost_drags = {}
        for cm in COST_MULTS:
            cost_drags[cm] = compute_cost_drag(
                test_data, entries, exits, new_long, id_col,
                cm, global_med_1x, prev_long
            )

        records.append({
            'date': test_date, 'gross_ret': gross_ret, 'turnover': turn,
            'n_held': len(new_long), 'ic': ic,
            **{f'cost_drag_{cm}': cost_drags[cm] for cm in COST_MULTS},
        })
        prev_long = new_long

    if len(records) == 0:
        return None, None

    result_df = pd.DataFrame(records).set_index('date')
    pred_ranks_df = pd.concat(all_pred_ranks, ignore_index=True) if all_pred_ranks else None
    return result_df, pred_ranks_df


# ==============================================================================
# MOMENTUM-ML BLEND STRATEGY
# ==============================================================================

def run_momentum_ml_blend(df, ridge_pred_ranks, date_col, id_col, target,
                          cost_lookup_1x, nq=5, buffer_pct=0.20,
                          mom_weight=0.8, ml_weight=0.2):
    """
    Blend: composite_rank = mom_weight * momentum_rank + ml_weight * Ridge_rank.
    Top quintile, buffer-based selection.
    """
    if ridge_pred_ranks is None or len(ridge_pred_ranks) == 0:
        return None

    data = df.copy()

    # Compute momentum rank
    mom_feat = 'ret_12_1'
    if mom_feat not in data.columns:
        return None
    data['mom_rank'] = data.groupby(date_col)[mom_feat].rank(pct=True)
    data['mom_rank'] = data['mom_rank'].fillna(0.5)

    # Merge Ridge prediction ranks
    data = data.merge(ridge_pred_ranks, on=[date_col, id_col], how='left')
    data['ml_rank'] = data['ml_rank'].fillna(0.5)

    # Composite rank
    data['blend_score'] = mom_weight * data['mom_rank'] + ml_weight * data['ml_rank']
    data['pct_rank'] = data.groupby(date_col)['blend_score'].rank(pct=True)

    top_threshold = 1.0 - 1.0 / nq
    buffer = buffer_pct * (1.0 / nq)

    data = data.merge(cost_lookup_1x, on=[date_col, id_col], how='left')
    global_med_1x = cost_lookup_1x['cost_bps_1x'].median()
    data['cost_bps_1x'] = data['cost_bps_1x'].fillna(global_med_1x)

    # Only use dates where we have Ridge predictions
    blend_dates = set(ridge_pred_ranks[date_col].unique())
    dates = sorted([d for d in data[date_col].unique() if d in blend_dates])

    prev_long = set()
    records = []

    for date in dates:
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

        ic = monthly_ic(month['blend_score'].values, month[target].values)

        cost_drags = {}
        for cm in COST_MULTS:
            cost_drags[cm] = compute_cost_drag(
                month, entries, exits, new_long, id_col,
                cm, global_med_1x, prev_long
            )

        records.append({
            'date': date, 'gross_ret': gross_ret, 'turnover': turn,
            'n_held': len(new_long), 'ic': ic,
            **{f'cost_drag_{cm}': cost_drags[cm] for cm in COST_MULTS},
        })
        prev_long = new_long

    if len(records) == 0:
        return None
    return pd.DataFrame(records).set_index('date')


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate_at_cost_level(result_df, cost_mult, period_label, period_mask,
                           strategy_name, market):
    """Compute summary metrics for a strategy at a given cost level and period."""
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

print("=" * 110)
print("EXPERIMENT AX: ML vs Simple Sorts across ALL 15 EM Markets — Extended OOS")
print(f"  Markets: {list(MARKETS.keys())}")
print(f"  Strategies: {STRATEGIES}")
print(f"  Cost multipliers: {COST_MULTS}")
print(f"  Buffer: {BUFFER_PCT:.0%}")
print(f"  ML min training: {MIN_TRAIN_MONTHS} months")
print(f"  Total configs: {len(MARKETS)} x {len(STRATEGIES)} x {len(COST_MULTS)} x 3 periods = "
      f"{len(MARKETS)*len(STRATEGIES)*len(COST_MULTS)*3}")
print("=" * 110)

all_results = []
t0 = time.time()

for mkt_name, cfg in MARKETS.items():
    print(f"\n\n{'#'*110}")
    print(f"# MARKET: {mkt_name}")
    print(f"{'#'*110}")

    # Load and prepare data
    try:
        df, ml_features, date_col, id_col, target, cost_lookup_1x = \
            load_and_prepare(mkt_name, cfg)
    except Exception as e:
        print(f"  ERROR loading {mkt_name}: {e}")
        continue

    nq = cfg['nq']

    # Determine IS/OOS split: 80/20 based on available months
    unique_dates = sorted(df[date_col].unique())
    n_total_months = len(unique_dates)
    is_split_idx = int(n_total_months * 0.8)
    if is_split_idx < MIN_TRAIN_MONTHS:
        is_split_idx = MIN_TRAIN_MONTHS
    is_end_date = unique_dates[min(is_split_idx - 1, n_total_months - 1)]
    oos_start_date = unique_dates[min(is_split_idx, n_total_months - 1)]

    print(f"  IS/OOS split: IS ends {is_end_date}, OOS starts {oos_start_date}")
    print(f"  IS months: {is_split_idx}, OOS months: {n_total_months - is_split_idx}")

    # Helper to create period masks for a result_df
    def make_masks(result_df):
        idx = result_df.index
        is_mask = idx <= is_end_date
        oos_mask = idx > is_end_date
        full_mask = pd.Series(True, index=idx)
        return [('IS', is_mask), ('OOS', oos_mask), ('Full', full_mask)]

    # Store raw results per strategy for this market (keyed by strategy name)
    strat_results = {}

    # ------------------------------------------------------------------
    # STRATEGY 1: MomentumSort
    # ------------------------------------------------------------------
    print(f"\n  --- MomentumSort ---")
    mom_feats = [f for f in MOMENTUM_FEATURES if f in df.columns]
    print(f"    Features: {mom_feats}")
    result = run_factor_sort(
        df, mom_feats, date_col, id_col, target, cost_lookup_1x,
        nq=nq, buffer_pct=BUFFER_PCT
    )
    strat_results['MomentumSort'] = result
    if result is not None:
        for period_label, mask in make_masks(result):
            for cm in COST_MULTS:
                metrics = evaluate_at_cost_level(result, cm, period_label, mask,
                                                 'MomentumSort', mkt_name)
                if metrics:
                    all_results.append(metrics)
        oos_mask = result.index > is_end_date
        oos_subset = result[oos_mask]
        if len(oos_subset) > 0:
            sr3 = sharpe(oos_subset['gross_ret'] - oos_subset['cost_drag_3.0'])
            print(f"    OOS Net SR @3x: {sr3:.3f}, Turnover: {oos_subset['turnover'].mean():.1%}")

    # ------------------------------------------------------------------
    # STRATEGY 2: LowVolSort
    # ------------------------------------------------------------------
    print(f"\n  --- LowVolSort ---")
    lowvol_feats = [f for f in LOWVOL_FEATURES if f in df.columns]
    print(f"    Features: {lowvol_feats}")
    # Low vol: ASCENDING = lower is better
    result = run_factor_sort(
        df, lowvol_feats, date_col, id_col, target, cost_lookup_1x,
        nq=nq, buffer_pct=BUFFER_PCT, ascending_features=lowvol_feats
    )
    strat_results['LowVolSort'] = result
    if result is not None:
        for period_label, mask in make_masks(result):
            for cm in COST_MULTS:
                metrics = evaluate_at_cost_level(result, cm, period_label, mask,
                                                 'LowVolSort', mkt_name)
                if metrics:
                    all_results.append(metrics)
        oos_mask = result.index > is_end_date
        oos_subset = result[oos_mask]
        if len(oos_subset) > 0:
            sr3 = sharpe(oos_subset['gross_ret'] - oos_subset['cost_drag_3.0'])
            print(f"    OOS Net SR @3x: {sr3:.3f}, Turnover: {oos_subset['turnover'].mean():.1%}")

    # ------------------------------------------------------------------
    # STRATEGY 3: CompositeSort (momentum + quality + low-vol)
    # ------------------------------------------------------------------
    print(f"\n  --- CompositeSort ---")
    quality_feats = [f for f in QUALITY_FEATURES if f in df.columns]
    composite_feats = mom_feats + quality_feats + lowvol_feats
    # Features where lower is better: accruals and volatility
    ascending_comp = [f for f in composite_feats
                      if 'accruals' in f.lower() or 'ivol' in f.lower() or 'rvol' in f.lower()]
    print(f"    Features: {composite_feats}")
    print(f"    Ascending (lower=better): {ascending_comp}")
    result = run_factor_sort(
        df, composite_feats, date_col, id_col, target, cost_lookup_1x,
        nq=nq, buffer_pct=BUFFER_PCT, ascending_features=ascending_comp
    )
    strat_results['CompositeSort'] = result
    if result is not None:
        for period_label, mask in make_masks(result):
            for cm in COST_MULTS:
                metrics = evaluate_at_cost_level(result, cm, period_label, mask,
                                                 'CompositeSort', mkt_name)
                if metrics:
                    all_results.append(metrics)
        oos_mask = result.index > is_end_date
        oos_subset = result[oos_mask]
        if len(oos_subset) > 0:
            sr3 = sharpe(oos_subset['gross_ret'] - oos_subset['cost_drag_3.0'])
            print(f"    OOS Net SR @3x: {sr3:.3f}, Turnover: {oos_subset['turnover'].mean():.1%}")

    # ------------------------------------------------------------------
    # STRATEGY 4: Ridge
    # ------------------------------------------------------------------
    print(f"\n  --- Ridge ---")
    result, ridge_pred_ranks = run_ml_walkforward(
        df, ml_features, date_col, id_col, target, cost_lookup_1x,
        model_type='ridge', nq=nq, buffer_pct=BUFFER_PCT,
        min_train=MIN_TRAIN_MONTHS
    )
    strat_results['Ridge'] = result
    if result is not None:
        for period_label, mask in make_masks(result):
            for cm in COST_MULTS:
                metrics = evaluate_at_cost_level(result, cm, period_label, mask,
                                                 'Ridge', mkt_name)
                if metrics:
                    all_results.append(metrics)
        oos_mask = result.index > is_end_date
        oos_subset = result[oos_mask]
        if len(oos_subset) > 0:
            sr3 = sharpe(oos_subset['gross_ret'] - oos_subset['cost_drag_3.0'])
            print(f"    OOS Net SR @3x: {sr3:.3f}, Turnover: {oos_subset['turnover'].mean():.1%}")

    # ------------------------------------------------------------------
    # STRATEGY 5: XGBoost
    # ------------------------------------------------------------------
    print(f"\n  --- XGBoost ---")
    result, _ = run_ml_walkforward(
        df, ml_features, date_col, id_col, target, cost_lookup_1x,
        model_type='xgboost', nq=nq, buffer_pct=BUFFER_PCT,
        min_train=MIN_TRAIN_MONTHS, top_k_features=20
    )
    strat_results['XGBoost'] = result
    if result is not None:
        for period_label, mask in make_masks(result):
            for cm in COST_MULTS:
                metrics = evaluate_at_cost_level(result, cm, period_label, mask,
                                                 'XGBoost', mkt_name)
                if metrics:
                    all_results.append(metrics)
        oos_mask = result.index > is_end_date
        oos_subset = result[oos_mask]
        if len(oos_subset) > 0:
            sr3 = sharpe(oos_subset['gross_ret'] - oos_subset['cost_drag_3.0'])
            print(f"    OOS Net SR @3x: {sr3:.3f}, Turnover: {oos_subset['turnover'].mean():.1%}")

    # ------------------------------------------------------------------
    # STRATEGY 6: LightGBM
    # ------------------------------------------------------------------
    print(f"\n  --- LightGBM ---")
    result, _ = run_ml_walkforward(
        df, ml_features, date_col, id_col, target, cost_lookup_1x,
        model_type='lightgbm', nq=nq, buffer_pct=BUFFER_PCT,
        min_train=MIN_TRAIN_MONTHS, top_k_features=20
    )
    strat_results['LightGBM'] = result
    if result is not None:
        for period_label, mask in make_masks(result):
            for cm in COST_MULTS:
                metrics = evaluate_at_cost_level(result, cm, period_label, mask,
                                                 'LightGBM', mkt_name)
                if metrics:
                    all_results.append(metrics)
        oos_mask = result.index > is_end_date
        oos_subset = result[oos_mask]
        if len(oos_subset) > 0:
            sr3 = sharpe(oos_subset['gross_ret'] - oos_subset['cost_drag_3.0'])
            print(f"    OOS Net SR @3x: {sr3:.3f}, Turnover: {oos_subset['turnover'].mean():.1%}")

    # ------------------------------------------------------------------
    # STRATEGY 7: MomentumML_Blend (0.8*mom + 0.2*Ridge)
    # ------------------------------------------------------------------
    print(f"\n  --- MomentumML_Blend ---")
    result = run_momentum_ml_blend(
        df, ridge_pred_ranks, date_col, id_col, target, cost_lookup_1x,
        nq=nq, buffer_pct=BUFFER_PCT, mom_weight=0.8, ml_weight=0.2
    )
    strat_results['MomentumML_Blend'] = result
    if result is not None:
        for period_label, mask in make_masks(result):
            for cm in COST_MULTS:
                metrics = evaluate_at_cost_level(result, cm, period_label, mask,
                                                 'MomentumML_Blend', mkt_name)
                if metrics:
                    all_results.append(metrics)
        oos_mask = result.index > is_end_date
        oos_subset = result[oos_mask]
        if len(oos_subset) > 0:
            sr3 = sharpe(oos_subset['gross_ret'] - oos_subset['cost_drag_3.0'])
            print(f"    OOS Net SR @3x: {sr3:.3f}, Turnover: {oos_subset['turnover'].mean():.1%}")

    # Print market summary at 3x costs OOS
    mkt_oos_3x = [r for r in all_results
                   if r['Market'] == mkt_name and r['CostMult'] == 3.0 and r['Period'] == 'OOS']
    if mkt_oos_3x:
        print(f"\n  {'='*80}")
        print(f"  {mkt_name} OOS Summary @ 3x costs")
        print(f"  {'='*80}")
        print(f"  {'Strategy':<22s} {'Net SR':>8s} {'Ann Ret':>9s} {'Turn':>7s} {'Cost':>8s} {'N_mo':>6s}")
        print(f"  {'-'*65}")
        for r in sorted(mkt_oos_3x, key=lambda x: x['Net_SR'], reverse=True):
            print(f"  {r['Strategy']:<22s} {r['Net_SR']:>8.3f} {r['Ann_Ret']:>8.1%} "
                  f"{r['Turnover']:>6.1%} {r['Avg_Cost_bps']:>7.1f} {r['N_months']:>6d}")


# ==============================================================================
# SAVE RESULTS
# ==============================================================================

elapsed = time.time() - t0
print(f"\n\nTotal runtime: {elapsed/60:.1f} minutes")

df_results = pd.DataFrame(all_results)
out_path = 'scripts/concern_experiments/expAX_full_comparison.csv'
df_results.to_csv(out_path, index=False)
print(f"Saved {len(df_results)} rows to {out_path}")


# ==============================================================================
# ANALYSIS 1: OOS Net SR by Strategy x Cost Level (averaged across markets)
# ==============================================================================

print("\n\n" + "=" * 120)
print("ANALYSIS 1: Average OOS Net Sharpe Ratio by Strategy x Cost Level")
print("=" * 120)

oos_results = [r for r in all_results if r['Period'] == 'OOS']

header = f"  {'CostMult':>8s}"
for s in STRATEGIES:
    label = s[:14]
    header += f" {label:>14s}"
header += f" {'Sort Avg':>10s} {'ML Avg':>10s} {'ML-Sort':>10s}"
print(header)
print(f"  {'-'*120}")

for cm in COST_MULTS:
    row = f"  {cm:>8.1f}"
    sort_vals = []
    ml_vals = []
    for s in STRATEGIES:
        vals = [r['Net_SR'] for r in oos_results
                if r['Strategy'] == s and r['CostMult'] == cm]
        avg = np.mean(vals) if vals else np.nan
        row += f" {avg:>14.3f}" if not np.isnan(avg) else f" {'N/A':>14s}"
        if s in SORT_STRATEGIES and vals:
            sort_vals.extend(vals)
        if s in ML_STRATEGIES and vals:
            ml_vals.extend(vals)

    sort_avg = np.mean(sort_vals) if sort_vals else np.nan
    ml_avg = np.mean(ml_vals) if ml_vals else np.nan
    diff = ml_avg - sort_avg if not (np.isnan(ml_avg) or np.isnan(sort_avg)) else np.nan
    row += f" {sort_avg:>10.3f}" if not np.isnan(sort_avg) else f" {'N/A':>10s}"
    row += f" {ml_avg:>10.3f}" if not np.isnan(ml_avg) else f" {'N/A':>10s}"
    row += f" {diff:>+10.3f}" if not np.isnan(diff) else f" {'N/A':>10s}"
    print(row)


# ==============================================================================
# ANALYSIS 2: Per-Market OOS Net SR Table at 3x Costs
# ==============================================================================

print("\n\n" + "=" * 120)
print("ANALYSIS 2: Per-Market OOS Net Sharpe Ratio at 3x Costs")
print("=" * 120)

header = f"  {'Market':<6s}"
for s in STRATEGIES:
    label = s[:14]
    header += f" {label:>14s}"
header += f" {'Best':>16s}"
print(header)
print(f"  {'-'*120}")

market_winners = {'sort': 0, 'ml': 0, 'blend': 0}

for mkt in MARKETS.keys():
    row = f"  {mkt:<6s}"
    srs = {}
    for s in STRATEGIES:
        match = [r for r in oos_results
                 if r['Market'] == mkt and r['Strategy'] == s and r['CostMult'] == 3.0]
        if match:
            sr = match[0]['Net_SR']
            srs[s] = sr
            row += f" {sr:>14.3f}"
        else:
            row += f" {'N/A':>14s}"

    if srs:
        best_name = max(srs, key=srs.get)
        row += f" {best_name:>16s}"
        if best_name in SORT_STRATEGIES:
            market_winners['sort'] += 1
        elif best_name in ML_STRATEGIES:
            market_winners['ml'] += 1
        else:
            market_winners['blend'] += 1
    print(row)

# Averages
print(f"  {'-'*120}")
row = f"  {'Avg':<6s}"
for s in STRATEGIES:
    vals = [r['Net_SR'] for r in oos_results
            if r['Strategy'] == s and r['CostMult'] == 3.0]
    avg = np.mean(vals) if vals else np.nan
    row += f" {avg:>14.3f}" if not np.isnan(avg) else f" {'N/A':>14s}"
print(row)
row = f"  {'Med':<6s}"
for s in STRATEGIES:
    vals = [r['Net_SR'] for r in oos_results
            if r['Strategy'] == s and r['CostMult'] == 3.0]
    med = np.median(vals) if vals else np.nan
    row += f" {med:>14.3f}" if not np.isnan(med) else f" {'N/A':>14s}"
print(row)

print(f"\n  Winners @3x: Sorts={market_winners['sort']}, ML={market_winners['ml']}, "
      f"Blend={market_winners['blend']} (of {sum(market_winners.values())} markets)")


# ==============================================================================
# ANALYSIS 3: Per-Market OOS Net SR at Each Cost Level — Identify Best Strategy
# ==============================================================================

print("\n\n" + "=" * 120)
print("ANALYSIS 3: Best Strategy per Market at Each Cost Level (OOS)")
print("=" * 120)

header = f"  {'Market':<6s}"
for cm in COST_MULTS:
    header += f" {f'{cm:.1f}x':>20s}"
print(header)
print(f"  {'-'*110}")

for mkt in MARKETS.keys():
    row = f"  {mkt:<6s}"
    for cm in COST_MULTS:
        mkt_data = [r for r in oos_results
                    if r['Market'] == mkt and r['CostMult'] == cm]
        if mkt_data:
            best = max(mkt_data, key=lambda r: r['Net_SR'])
            label = f"{best['Strategy'][:10]}({best['Net_SR']:.2f})"
            row += f" {label:>20s}"
        else:
            row += f" {'N/A':>20s}"
    print(row)


# ==============================================================================
# ANALYSIS 4: Turnover Comparison (cost-level independent)
# ==============================================================================

print("\n\n" + "=" * 120)
print("ANALYSIS 4: Average Monthly Turnover (OOS)")
print("=" * 120)

header = f"  {'Market':<6s}"
for s in STRATEGIES:
    label = s[:14]
    header += f" {label:>14s}"
print(header)
print(f"  {'-'*110}")

for mkt in MARKETS.keys():
    row = f"  {mkt:<6s}"
    for s in STRATEGIES:
        match = [r for r in oos_results
                 if r['Market'] == mkt and r['Strategy'] == s and r['CostMult'] == COST_MULTS[0]]
        if match:
            turn = match[0]['Turnover']
            row += f" {turn:>13.1%}"
        else:
            row += f" {'N/A':>14s}"
    print(row)

# Averages
print(f"  {'-'*110}")
row = f"  {'Avg':<6s}"
for s in STRATEGIES:
    vals = [r['Turnover'] for r in oos_results
            if r['Strategy'] == s and r['CostMult'] == COST_MULTS[0]]
    row += f" {np.mean(vals):>13.1%}" if vals else f" {'N/A':>14s}"
print(row)


# ==============================================================================
# ANALYSIS 5: Crossover Point — At What Cost Level Do Sorts First Beat ML?
# ==============================================================================

print("\n\n" + "=" * 120)
print("ANALYSIS 5: Crossover Analysis — Cost Level Where Sorts First Beat ML")
print("=" * 120)

print(f"\n  For each market, find the lowest cost multiplier where the best sort")
print(f"  strategy has a higher OOS Net SR than the best ML strategy.\n")

print(f"  {'Market':<6s} {'Best Sort':>14s} {'Best ML':>12s} {'Crossover':>12s} "
      f"{'Sort@1x':>10s} {'ML@1x':>10s} {'Sort@3x':>10s} {'ML@3x':>10s}")
print(f"  {'-'*85}")

crossover_points = {}

for mkt in MARKETS.keys():
    mkt_oos = [r for r in oos_results if r['Market'] == mkt]
    if not mkt_oos:
        continue

    # Find best sort and best ML at 1x (lowest cost)
    sort_1x = [r for r in mkt_oos if r['Strategy'] in SORT_STRATEGIES and r['CostMult'] == 1.0]
    ml_1x = [r for r in mkt_oos if r['Strategy'] in ML_STRATEGIES and r['CostMult'] == 1.0]

    if not sort_1x or not ml_1x:
        continue

    best_sort_name = max(sort_1x, key=lambda r: r['Net_SR'])['Strategy']
    best_ml_name = max(ml_1x, key=lambda r: r['Net_SR'])['Strategy']

    sort_curve = {}
    ml_curve = {}
    for cm in COST_MULTS:
        s_match = [r for r in mkt_oos if r['Strategy'] == best_sort_name and r['CostMult'] == cm]
        m_match = [r for r in mkt_oos if r['Strategy'] == best_ml_name and r['CostMult'] == cm]
        if s_match:
            sort_curve[cm] = s_match[0]['Net_SR']
        if m_match:
            ml_curve[cm] = m_match[0]['Net_SR']

    # Find crossover: first cost_mult where sort SR > ML SR
    crossover = None
    for cm in sorted(COST_MULTS):
        if cm in sort_curve and cm in ml_curve:
            if sort_curve[cm] >= ml_curve[cm]:
                crossover = cm
                break

    crossover_points[mkt] = crossover

    sort_1 = sort_curve.get(1.0, np.nan)
    ml_1 = ml_curve.get(1.0, np.nan)
    sort_3 = sort_curve.get(3.0, np.nan)
    ml_3 = ml_curve.get(3.0, np.nan)

    if crossover is not None:
        be_str = f"{crossover:.1f}x"
    else:
        be_str = ">3.0x (ML always)"

    sort_label = best_sort_name[:14]
    ml_label = best_ml_name[:12]

    print(f"  {mkt:<6s} {sort_label:>14s} {ml_label:>12s} {be_str:>12s} "
          f"{sort_1:>10.3f} {ml_1:>10.3f} {sort_3:>10.3f} {ml_3:>10.3f}")


# ==============================================================================
# ANALYSIS 6: ML Advantage Curve (Best ML SR - Best Sort SR at each cost level)
# ==============================================================================

print("\n\n" + "=" * 120)
print("ANALYSIS 6: ML Advantage Curve (Best ML SR - Best Sort SR, OOS)")
print("  Positive = ML wins, Negative = Sort wins")
print("=" * 120)

header = f"  {'CostMult':>8s}"
for mkt in MARKETS.keys():
    header += f" {mkt:>8s}"
header += f" {'Average':>10s} {'ML>Sort':>8s}"
print(header)
print(f"  {'-'*120}")

for cm in COST_MULTS:
    row = f"  {cm:>8.1f}"
    diffs = []
    ml_wins = 0
    n_valid = 0
    for mkt in MARKETS.keys():
        mkt_data = [r for r in oos_results if r['Market'] == mkt and r['CostMult'] == cm]
        sort_srs = [r['Net_SR'] for r in mkt_data if r['Strategy'] in SORT_STRATEGIES]
        ml_srs = [r['Net_SR'] for r in mkt_data if r['Strategy'] in ML_STRATEGIES]
        if sort_srs and ml_srs:
            best_sort = max(sort_srs)
            best_ml = max(ml_srs)
            d = best_ml - best_sort
            diffs.append(d)
            n_valid += 1
            if d > 0:
                ml_wins += 1
            row += f" {d:>+8.3f}"
        else:
            row += f" {'N/A':>8s}"

    avg_diff = np.mean(diffs) if diffs else np.nan
    row += f" {avg_diff:>+10.3f}" if not np.isnan(avg_diff) else f" {'N/A':>10s}"
    row += f" {ml_wins:>3d}/{n_valid:<3d}"
    print(row)


# ==============================================================================
# ANALYSIS 7: IS-to-OOS Decay (Overfitting Indicator)
# ==============================================================================

print("\n\n" + "=" * 120)
print("ANALYSIS 7: IS-to-OOS Decay (at 3x costs)")
print("=" * 120)

print(f"  {'Strategy':<22s} {'Avg IS SR':>10s} {'Avg OOS SR':>11s} {'Decay':>10s} {'Decay%':>8s}")
print(f"  {'-'*65}")

for s in STRATEGIES:
    is_srs = [r['Net_SR'] for r in all_results
              if r['Strategy'] == s and r['CostMult'] == 3.0 and r['Period'] == 'IS']
    oos_srs = [r['Net_SR'] for r in all_results
               if r['Strategy'] == s and r['CostMult'] == 3.0 and r['Period'] == 'OOS']
    if is_srs and oos_srs:
        avg_is = np.mean(is_srs)
        avg_oos = np.mean(oos_srs)
        decay = avg_oos - avg_is
        decay_pct = (decay / abs(avg_is) * 100) if avg_is != 0 else 0
        print(f"  {s:<22s} {avg_is:>10.3f} {avg_oos:>11.3f} {decay:>+10.3f} {decay_pct:>+7.1f}%")


# ==============================================================================
# ANALYSIS 8: Cost Sensitivity — SR Degradation Per Unit Cost
# ==============================================================================

print("\n\n" + "=" * 120)
print("ANALYSIS 8: Cost Sensitivity — SR Degradation (1x vs 3x, OOS)")
print("=" * 120)

print(f"  {'Strategy':<22s} {'SR@1x':>8s} {'SR@3x':>8s} {'Degradation':>12s} {'Per 1x':>8s}")
print(f"  {'-'*65}")

for s in STRATEGIES:
    vals_1 = [r['Net_SR'] for r in oos_results if r['Strategy'] == s and r['CostMult'] == 1.0]
    vals_3 = [r['Net_SR'] for r in oos_results if r['Strategy'] == s and r['CostMult'] == 3.0]
    if vals_1 and vals_3:
        avg_1 = np.mean(vals_1)
        avg_3 = np.mean(vals_3)
        degrade = avg_1 - avg_3
        per_1x = degrade / 2.0  # difference is over 2 units (1x to 3x)
        print(f"  {s:<22s} {avg_1:>8.3f} {avg_3:>8.3f} {degrade:>+12.3f} {per_1x:>8.3f}")


# ==============================================================================
# ANALYSIS 9: Statistical Tests
# ==============================================================================

print("\n\n" + "=" * 120)
print("ANALYSIS 9: Statistical Tests (OOS, 3x costs)")
print("=" * 120)

# Sort vs ML aggregate comparison
sort_srs = [r['Net_SR'] for r in oos_results
            if r['Strategy'] in SORT_STRATEGIES and r['CostMult'] == 3.0]
ml_srs = [r['Net_SR'] for r in oos_results
          if r['Strategy'] in ML_STRATEGIES and r['CostMult'] == 3.0]

if sort_srs and ml_srs:
    print(f"\n  Sort strategies (N={len(sort_srs)}): Mean SR = {np.mean(sort_srs):.3f}, "
          f"Median = {np.median(sort_srs):.3f}")
    print(f"  ML strategies   (N={len(ml_srs)}):   Mean SR = {np.mean(ml_srs):.3f}, "
          f"Median = {np.median(ml_srs):.3f}")
    print(f"  Difference: {np.mean(ml_srs) - np.mean(sort_srs):+.3f}")

    if len(sort_srs) >= 3 and len(ml_srs) >= 3:
        stat, pval = stats.ranksums(ml_srs, sort_srs)
        print(f"  Wilcoxon rank-sum (ML vs Sort): z={stat:.3f}, p={pval:.4f}")

# Per-market: best ML vs best sort paired test
print(f"\n  Paired comparison: Best ML vs Best Sort per market @3x:")
paired_diffs = []
for mkt in MARKETS.keys():
    mkt_data = [r for r in oos_results if r['Market'] == mkt and r['CostMult'] == 3.0]
    sort_best = [r['Net_SR'] for r in mkt_data if r['Strategy'] in SORT_STRATEGIES]
    ml_best = [r['Net_SR'] for r in mkt_data if r['Strategy'] in ML_STRATEGIES]
    if sort_best and ml_best:
        d = max(ml_best) - max(sort_best)
        paired_diffs.append(d)
        winner = "ML" if d > 0 else "Sort"
        print(f"    {mkt}: Best Sort={max(sort_best):.3f}, Best ML={max(ml_best):.3f} "
              f"-> {winner} ({d:+.3f})")

if len(paired_diffs) >= 5:
    from scipy.stats import wilcoxon
    try:
        stat_w, pval_w = wilcoxon(paired_diffs)
        print(f"\n  Wilcoxon signed-rank (paired, Best ML - Best Sort): "
              f"stat={stat_w:.1f}, p={pval_w:.4f}")
    except Exception:
        pass
    n_ml_wins = sum(1 for d in paired_diffs if d > 0)
    print(f"  ML wins in {n_ml_wins}/{len(paired_diffs)} markets")
    print(f"  Mean diff: {np.mean(paired_diffs):+.3f}, Median: {np.median(paired_diffs):+.3f}")

# Blend vs pure momentum
print(f"\n  MomentumML_Blend vs MomentumSort @3x:")
for mkt in MARKETS.keys():
    mkt_data = [r for r in oos_results if r['Market'] == mkt and r['CostMult'] == 3.0]
    mom = [r for r in mkt_data if r['Strategy'] == 'MomentumSort']
    blend = [r for r in mkt_data if r['Strategy'] == 'MomentumML_Blend']
    if mom and blend:
        d = blend[0]['Net_SR'] - mom[0]['Net_SR']
        print(f"    {mkt}: Mom={mom[0]['Net_SR']:.3f}, Blend={blend[0]['Net_SR']:.3f} ({d:+.3f})")


# ==============================================================================
# KEY FINDINGS
# ==============================================================================

print("\n\n" + "=" * 120)
print("KEY FINDINGS")
print("=" * 120)

# 1. At 3x costs
sort_3x = [r['Net_SR'] for r in oos_results
           if r['Strategy'] in SORT_STRATEGIES and r['CostMult'] == 3.0]
ml_3x = [r['Net_SR'] for r in oos_results
         if r['Strategy'] in ML_STRATEGIES and r['CostMult'] == 3.0]
blend_3x = [r['Net_SR'] for r in oos_results
            if r['Strategy'] == 'MomentumML_Blend' and r['CostMult'] == 3.0]

print(f"\n  1. AT 3x COSTS (conservative assumption):")
if sort_3x:
    print(f"     Sort avg OOS SR: {np.mean(sort_3x):.3f} (median: {np.median(sort_3x):.3f})")
if ml_3x:
    print(f"     ML avg OOS SR:   {np.mean(ml_3x):.3f} (median: {np.median(ml_3x):.3f})")
if blend_3x:
    print(f"     Blend avg OOS SR: {np.mean(blend_3x):.3f} (median: {np.median(blend_3x):.3f})")
print(f"     Winner: {'Sorts' if np.mean(sort_3x) > np.mean(ml_3x) else 'ML'} "
      f"(diff: {np.mean(ml_3x) - np.mean(sort_3x):+.3f})")
print(f"     Sorts win {market_winners['sort']}/{sum(market_winners.values())} markets, "
      f"ML wins {market_winners['ml']}/{sum(market_winners.values())}, "
      f"Blend wins {market_winners['blend']}/{sum(market_winners.values())}")

# 2. Crossover summary
cross_vals = [v for v in crossover_points.values() if v is not None]
if cross_vals:
    print(f"\n  2. CROSSOVER POINTS:")
    print(f"     Markets where sorts beat ML at 1x: "
          f"{sum(1 for v in crossover_points.values() if v == 1.0)}/{len(crossover_points)}")
    for cm in COST_MULTS:
        n_sort_wins = sum(1 for v in crossover_points.values() if v is not None and v <= cm)
        n_never = sum(1 for v in crossover_points.values() if v is None)
        print(f"     At {cm:.1f}x: Sorts win in {n_sort_wins} markets, "
              f"ML always wins in {n_never} markets")

# 3. Turnover story
print(f"\n  3. TURNOVER:")
for s in STRATEGIES:
    turns = [r['Turnover'] for r in oos_results
             if r['Strategy'] == s and r['CostMult'] == COST_MULTS[0]]
    if turns:
        print(f"     {s:<22s}: Avg monthly turnover = {np.mean(turns):.1%}")

# 4. Best strategy per cost level
print(f"\n  4. BEST STRATEGY BY COST LEVEL (avg across markets):")
for cm in COST_MULTS:
    best_avg = None
    best_name = None
    for s in STRATEGIES:
        vals = [r['Net_SR'] for r in oos_results
                if r['Strategy'] == s and r['CostMult'] == cm]
        if vals:
            avg = np.mean(vals)
            if best_avg is None or avg > best_avg:
                best_avg = avg
                best_name = s
    if best_name:
        print(f"     @{cm:.1f}x: {best_name} (avg SR={best_avg:.3f})")

print(f"\n  5. IMPLICATION FOR PAPER:")
print(f"     The 'implementable frontier' maps strategy complexity vs net performance")
print(f"     across realistic cost levels. If sorts dominate at high costs, the frontier")
print(f"     favors simplicity. If ML dominates at low costs but sorts dominate at high costs,")
print(f"     the crossover point defines the implementability boundary.")

print("\n\nDone.")
