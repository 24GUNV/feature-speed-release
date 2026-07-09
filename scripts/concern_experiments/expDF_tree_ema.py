"""
Experiment DF: EMA smoothing on tree model predictions.

Tests whether smoothing XGBoost/LightGBM predictions can bring trees
closer to the performance ceiling that Ridge_Slow achieves.

Key question: Is the tree penalty due to piecewise-constant surfaces
(fixable by output smoothing) or something deeper (not fixable)?

Strategies:
  - XGBoost (no smoothing, baseline)
  - XGBoost + EMA (alpha = 0.2, 0.35, 0.5, 0.7, 1.0)
  - LightGBM (no smoothing, baseline)
  - LightGBM + EMA (alpha = 0.2, 0.35, 0.5, 0.7, 1.0)
  - Ridge_Slow (ceiling reference)
  - MomentumSort (sort reference)

Uses rank targets, lag=1, all 15 EM markets, walk-forward.
EMA applied to cross-sectional prediction RANKS (same as expAQ).

Refactored: each model is trained ONCE per walk-forward step, then all
EMA alphas are applied to the same raw predictions. This is ~5x faster
for the tree models compared to the original version.

Output: scripts/concern_experiments/expDF_tree_ema.csv
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

from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy import stats

from data_loader import load_equity_data, lag_features
from market_config import MARKETS

np.random.seed(42)

COST_MULTS = [1.0, 2.0, 3.0]
BUFFER_PCT = 0.20
MIN_TRAIN_MONTHS = 60
EMA_ALPHAS = [0.2, 0.35, 0.5, 0.7, 1.0]
EXCLUDE_COLS = ['Unnamed: 0', 'id', 'eom', 'excntry', 'gvkey', 'permno',
                'size_grp', 'me', 'market_equity', 'ret_1_0', 'date']

# Slow features (accounting fundamentals with AC > 0.90)
SLOW_FEATURES = [
    'f_score', 'o_score', 'z_score', 'qmj', 'qmj_prof', 'qmj_safety', 'qmj_growth',
    'roa', 'roe', 'gp_at', 'gp_atl1', 'op_at', 'op_atl1',
    'niq_at', 'niq_be', 'ni_be', 'ni_me',
    'cop_at', 'cop_atl1', 'ope_be', 'ope_bel1',
    'ocf_at', 'ocf_me', 'oaccruals_at', 'oaccruals_ni',
    'taccruals_at', 'taccruals_ni',
    'be_me', 'bev_mev', 'at_me', 'ebit_bev', 'ebitda_mev',
    'fcf_me', 'sale_me', 'debt_me', 'netdebt_me',
    'div12m_me', 'ebit_sale', 'cash_at', 'tangibility',
    'noa_at', 'opex_at', 'rd5_at', 'rd_sale', 'rd_me',
    'sale_bev', 'eq_dur', 'age',
    'at_gr1', 'ppeinv_gr1a', 'noa_gr1a', 'lnoa_gr1a',
    'capx_gr1', 'capx_gr2', 'capx_gr3', 'capex_abn',
    'chcsho_12m', 'eqnpo_12m', 'debt_gr3',
    'inv_gr1', 'inv_gr1a', 'at_be', 'at_turnover',
    'kz_index', 'aliq_at', 'aliq_mat',
]


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


def walk_forward_predict(df, features, date_col, id_col, target,
                         cost_lookup_1x, model_type='ridge', min_train=60):
    """Walk-forward training: train model once per step, return raw predictions per month.

    Returns a list of dicts, one per OOS month, each containing:
      - 'date': the test date
      - 'test_data': DataFrame with columns including id_col, target, 'cost_bps_1x', 'raw_rank'
    Returns None if insufficient data.
    """
    data = df.copy().sort_values([date_col, id_col]).reset_index(drop=True)
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

    monthly_predictions = []

    for t_idx in range(min_train, n_dates):
        train_dates = unique_dates[:t_idx]
        test_date = unique_dates[t_idx]

        train_data = data[data[date_col].isin(train_dates)]
        test_data = data[data[date_col] == test_date].copy()

        nq_check = 5  # minimum stocks needed
        if len(test_data) < nq_check or len(train_data) < 100:
            continue

        X_train = train_data[avail_features].copy()
        X_test = test_data[avail_features].copy()

        # Rank targets
        y_rank = train_data.groupby(date_col)[target].rank(pct=True).values

        imputer = SimpleImputer(strategy='median')
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        mask = ~np.isnan(y_rank)
        if mask.sum() < 50:
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp[mask])
        X_test_scaled = scaler.transform(X_test_imp)

        # Fit model
        if model_type == 'ridge':
            model = Ridge(alpha=100, random_state=42)
            model.fit(X_train_scaled, y_rank[mask])
            preds = model.predict(X_test_scaled)
        elif model_type == 'xgboost':
            model = XGBRegressor(
                n_estimators=200, max_depth=4, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0
            )
            model.fit(X_train_scaled, y_rank[mask])
            preds = model.predict(X_test_scaled)
        elif model_type == 'lightgbm':
            model = LGBMRegressor(
                n_estimators=200, num_leaves=31, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbose=-1
            )
            model.fit(X_train_scaled, y_rank[mask])
            preds = model.predict(X_test_scaled)

        test_data = test_data.copy()
        test_data['prediction'] = preds
        test_data['raw_rank'] = test_data['prediction'].rank(pct=True)

        monthly_predictions.append({
            'date': test_date,
            'test_data': test_data[[id_col, target, 'cost_bps_1x', 'raw_rank']].copy(),
        })

    if len(monthly_predictions) == 0:
        return None
    return monthly_predictions


def apply_ema_and_portfolio(monthly_predictions, id_col, target,
                            ema_alpha=1.0, nq=5, buffer_pct=0.20):
    """Apply EMA smoothing to raw prediction ranks and construct portfolio.

    Takes the output of walk_forward_predict() and applies a specific EMA alpha,
    then runs portfolio construction with buffer.

    Returns a DataFrame of monthly records (same format as the old run_model_with_ema).
    """
    top_threshold = 1.0 - 1.0 / nq
    buffer = buffer_pct * (1.0 / nq)

    prev_long = set()
    smoothed_ranks = {}  # {stock_id: smoothed_rank}
    records = []

    global_med_1x = np.median([
        row['test_data']['cost_bps_1x'].median()
        for row in monthly_predictions
    ])

    for month_pred in monthly_predictions:
        test_date = month_pred['date']
        test_data = month_pred['test_data'].copy()

        # Apply EMA smoothing on ranks
        current_smoothed = {}
        for _, row in test_data.iterrows():
            sid = row[id_col]
            raw_r = row['raw_rank']
            if sid in smoothed_ranks:
                current_smoothed[sid] = ema_alpha * raw_r + (1 - ema_alpha) * smoothed_ranks[sid]
            else:
                current_smoothed[sid] = raw_r
        smoothed_ranks = current_smoothed

        test_data['pct_rank'] = test_data[id_col].map(current_smoothed)
        test_data['pct_rank'] = test_data['pct_rank'].fillna(test_data['raw_rank'])

        # Portfolio construction with buffer
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

        held = test_data[test_data[id_col].isin(new_long)]
        gross_ret = held[target].mean() if len(held) > 0 else 0.0

        if prev_long:
            entries = new_long - prev_long
            exits = prev_long - new_long
            turn = (len(entries) + len(exits)) / (2 * max(len(new_long), 1))
        else:
            entries, exits = new_long, set()
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
# MAIN
# ==============================================================================

all_results = []
t0 = time.time()

for mkt_code, mkt_config in MARKETS.items():
    t_mkt = time.time()
    print(f"\n{'='*60}", flush=True)
    print(f"Market: {mkt_code}", flush=True)
    print(f"{'='*60}", flush=True)

    try:
        result = load_equity_data(mkt_config['file'], mkt_config['type'])
        if isinstance(result, tuple):
            raw_df, _, date_col, id_col = result
        else:
            raw_df = result
            date_col = 'eom' if 'eom' in raw_df.columns else 'date'
            id_col = 'gvkey' if 'gvkey' in raw_df.columns else 'permno'
    except Exception as e:
        print(f"  ERROR loading {mkt_code}: {e}", flush=True)
        continue

    target = mkt_config['target']
    topn = mkt_config['topn']
    nq = mkt_config['nq']

    # Lag features BEFORE filtering
    all_feat_cols = [c for c in raw_df.columns if c not in EXCLUDE_COLS
                     and c != target and c != 'me' and c != '_me' and c != '_me_rank']
    raw_df = lag_features(raw_df, features=all_feat_cols, id_col=id_col,
                          date_col=date_col, lag=1)

    # Lag me to avoid look-ahead bias: me(T) = me(T-1) * (1+ret), so
    # unlagged me embeds current return into universe selection
    if 'me' in raw_df.columns:
        raw_df = raw_df.sort_values([id_col, date_col])
        raw_df['me'] = raw_df.groupby(id_col)['me'].shift(1)
        raw_df = raw_df.dropna(subset=['me'])

    # Universe filter
    raw_df['_me_rank'] = raw_df.groupby(date_col)['me'].rank(ascending=False, method='first')
    df = raw_df[raw_df['_me_rank'] <= topn].copy()

    cost_lookup = build_cost_lookup_1x(raw_df, date_col, id_col)

    # Feature sets
    all_features = [c for c in df.columns if c not in EXCLUDE_COLS
                    and c != target and c != '_me' and c != '_me_rank'
                    and not c.startswith('_')]
    slow_features = [f for f in SLOW_FEATURES if f in df.columns]

    # Determine IS/OOS split — fixed date to match paper
    unique_dates = sorted(df[date_col].unique())
    oos_boundary = pd.Timestamp('2020-12-31')
    oos_start_idx = None
    for i, d in enumerate(unique_dates):
        if pd.Timestamp(d) > oos_boundary:
            oos_start_idx = i
            break
    if oos_start_idx is None:
        oos_start_idx = max(MIN_TRAIN_MONTHS, int(len(unique_dates) * 0.8))

    # ==========================================
    # Define model runs: (model_type, features)
    # Each model is trained ONCE, then all EMA alphas applied to same predictions
    # ==========================================
    model_runs = [
        ('ridge', slow_features, 'Ridge_Slow', [1.0]),  # ceiling reference, no EMA sweep
        ('xgboost', all_features, 'XGB', EMA_ALPHAS),
        ('lightgbm', all_features, 'LGBM', EMA_ALPHAS),
    ]

    for model_type, features, base_name, alphas in model_runs:
        t_s = time.time()

        # Step 1: Train model ONCE across all walk-forward steps
        print(f"  Training {model_type}...", flush=True)
        monthly_preds = walk_forward_predict(
            df, features, date_col, id_col, target, cost_lookup,
            model_type=model_type, min_train=MIN_TRAIN_MONTHS
        )

        if monthly_preds is None:
            print(f"  {base_name}: FAILED (no predictions)", flush=True)
            continue

        train_elapsed = time.time() - t_s
        print(f"  {base_name} training done ({train_elapsed:.0f}s), applying {len(alphas)} EMA alphas...", flush=True)

        # Step 2: Apply each EMA alpha to the SAME predictions
        for ema_alpha in alphas:
            if base_name == 'Ridge_Slow':
                strat_name = 'Ridge_Slow'
            elif ema_alpha < 1.0:
                strat_name = f'{base_name}_EMA_{ema_alpha}'
            else:
                # alpha=1.0 means no smoothing = baseline
                strat_name = 'XGBoost' if model_type == 'xgboost' else 'LightGBM'

            result = apply_ema_and_portfolio(
                monthly_preds, id_col, target,
                ema_alpha=ema_alpha, nq=nq, buffer_pct=BUFFER_PCT
            )

            if result is None:
                print(f"  {strat_name}: FAILED (no portfolio)", flush=True)
                continue

            # Split IS/OOS
            oos_date = unique_dates[oos_start_idx]
            result_oos = result[result.index >= oos_date]

            if len(result_oos) < 12:
                print(f"  {strat_name}: too few OOS months ({len(result_oos)})", flush=True)
                continue

            for cm in COST_MULTS:
                net_ret = result_oos['gross_ret'] - result_oos[f'cost_drag_{cm}']
                net_sr = sharpe(net_ret)
                gross_sr = sharpe(result_oos['gross_ret'])
                avg_to = result_oos['turnover'].mean()

                all_results.append({
                    'Market': mkt_code, 'Strategy': strat_name,
                    'CostMult': cm, 'Period': 'OOS',
                    'Net_SR': net_sr, 'Gross_SR': gross_sr,
                    'Turnover': avg_to, 'N_months': len(result_oos),
                    'EMA_alpha': ema_alpha,
                })

            oos_sr3 = [r for r in all_results if r['Market'] == mkt_code
                       and r['Strategy'] == strat_name and r['CostMult'] == 3.0
                       and r['Period'] == 'OOS']
            sr3 = oos_sr3[0]['Net_SR'] if oos_sr3 else 0
            avg_to = result_oos['turnover'].mean()
            print(f"    {strat_name}: Net SR@3x={sr3:.3f}, TO={avg_to:.1%}", flush=True)

        print(f"  {base_name} total time: {time.time() - t_s:.0f}s", flush=True)

    print(f"  Market time: {time.time() - t_mkt:.0f}s", flush=True)

# Save
results_df = pd.DataFrame(all_results)
results_df.to_csv('scripts/concern_experiments/expDF_tree_ema.csv', index=False)

# Summary
print(f"\n\n{'='*60}", flush=True)
print("SUMMARY: Mean OOS Net SR at 3x costs across 15 markets", flush=True)
print("="*60, flush=True)

oos3 = results_df[(results_df['Period'] == 'OOS') & (results_df['CostMult'] == 3.0)]
for strat in ['Ridge_Slow', 'XGBoost', 'XGB_EMA_0.5', 'XGB_EMA_0.35', 'XGB_EMA_0.2',
              'LightGBM', 'LGBM_EMA_0.5', 'LGBM_EMA_0.35', 'LGBM_EMA_0.2']:
    sub = oos3[oos3['Strategy'] == strat]
    if len(sub) > 0:
        print(f"  {strat:20s}: SR={sub['Net_SR'].mean():.3f}, TO={sub['Turnover'].mean():.1%}", flush=True)

print(f"\nTotal time: {time.time() - t0:.0f}s", flush=True)
