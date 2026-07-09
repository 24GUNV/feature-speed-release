"""
Experiment AK: Stock-Level Information Coefficient

Computes stock-level prediction quality metrics for sorts vs ML:
  1. Monthly rank IC (Spearman correlation with realized returns)
  2. Top-quintile precision (hit rate of top-Q predictions)
  3. Prediction stability (month-to-month rank correlation of predictions)

Strengthens the mechanism story: ML predicts better cross-sectionally
but with less stable predictions, causing high turnover.

Pipeline matches expAH exactly: dual-lag, buffer, walk-forward, CS costs.

Output: scripts/concern_experiments/expAK_information_coefficient.csv
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
from scipy import stats

from data_loader import load_equity_data, lag_features

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
    'CHN': {'file': 'data/CHNEquitiesData.csv', 'type': 'emerging_equity',
            'target': 'ret_1_0', 'topn': 300, 'nq': 5},
    'MYS': {'file': 'data/MYSEquitiesData.csv', 'type': 'emerging_equity',
            'target': 'ret_1_0', 'topn': 100, 'nq': 5},
    'POL': {'file': 'data/POLEquitiesData.csv', 'type': 'emerging_equity',
            'target': 'ret_1_0', 'topn': 50, 'nq': 5},
    'IDN': {'file': 'data/IDNEquitiesData.csv', 'type': 'emerging_equity',
            'target': 'ret_1_0', 'topn': 100, 'nq': 5},
    'ZAF': {'file': 'data/ZAFEquitiesData.csv', 'type': 'emerging_equity',
            'target': 'ret_1_0', 'topn': 100, 'nq': 5},
    'TUR': {'file': 'data/TUREquitiesData.csv', 'type': 'emerging_equity',
            'target': 'ret_1_0', 'topn': 100, 'nq': 5},
    'BRA': {'file': 'data/BRAEquitiesData.csv', 'type': 'emerging_equity',
            'target': 'ret_1_0', 'topn': 100, 'nq': 5},
    'PHL': {'file': 'data/PHLEquitiesData.csv', 'type': 'emerging_equity',
            'target': 'ret_1_0', 'topn': 50, 'nq': 5},
    'GRC': {'file': 'data/GRCEquitiesData.csv', 'type': 'emerging_equity',
            'target': 'ret_1_0', 'topn': 50, 'nq': 5},
    'SAU': {'file': 'data/SAUEquitiesData.csv', 'type': 'emerging_equity',
            'target': 'ret_1_0', 'topn': 100, 'nq': 5},
    'MEX': {'file': 'data/MEXEquitiesData.csv', 'type': 'emerging_equity',
            'target': 'ret_1_0', 'topn': 50, 'nq': 5},
}

COST_MULTS = [1.0, 1.5, 2.0, 2.5, 3.0]
BUFFER_PCT = 0.20
MIN_TRAIN_MONTHS = 60

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


# ==============================================================================
# WALK-FORWARD THAT RETURNS STOCK-LEVEL PREDICTIONS
# ==============================================================================

def run_ml_predictions(df, features, date_col, id_col, target, min_train=60,
                       oos_start_date=None):
    """
    Walk-forward Ridge that returns stock-level predictions per OOS month.
    Returns DataFrame: [date_col, id_col, 'prediction', target]

    If oos_start_date is given, only produce predictions for dates > oos_start_date
    (but still use all prior data for training). This avoids wasting time on IS predictions.
    """
    data = df.copy()
    data = data.sort_values([date_col, id_col]).reset_index(drop=True)

    unique_dates = sorted(data[date_col].unique())
    n_dates = len(unique_dates)

    if n_dates < min_train + 12:
        return None

    avail_features = [f for f in features if f in data.columns]
    feat_coverage = data[avail_features].notna().mean()
    avail_features = feat_coverage[feat_coverage > 0.1].index.tolist()

    if len(avail_features) == 0:
        return None

    # Determine start index for walk-forward
    start_idx = min_train
    if oos_start_date is not None:
        for i, d in enumerate(unique_dates):
            if d > oos_start_date:
                start_idx = max(i, min_train)
                break

    all_preds = []
    n_iter = n_dates - start_idx

    for step, t_idx in enumerate(range(start_idx, n_dates)):
        test_date = unique_dates[t_idx]
        cutoff_date = unique_dates[t_idx - 1]

        train_data = data[data[date_col] <= cutoff_date]
        test_data = data[data[date_col] == test_date].copy()

        if len(test_data) < 5:
            continue
        if len(train_data) < 100:
            continue

        X_train = train_data[avail_features].copy()
        y_train = train_data[target].values
        X_test = test_data[avail_features].copy()

        imputer = SimpleImputer(strategy='median')
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        mask = ~np.isnan(y_train)
        if mask.sum() < 50:
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp[mask])
        X_test_scaled = scaler.transform(X_test_imp)

        model = Ridge(alpha=100, random_state=42)
        model.fit(X_train_scaled, y_train[mask])
        preds = model.predict(X_test_scaled)

        pred_df = test_data[[date_col, id_col, target]].copy()
        pred_df['prediction'] = preds
        all_preds.append(pred_df)

    if len(all_preds) == 0:
        return None
    return pd.concat(all_preds, ignore_index=True)


# ==============================================================================
# IC, PRECISION, STABILITY FUNCTIONS
# ==============================================================================

def compute_monthly_ic(pred_df, date_col, score_col, target):
    """
    Compute monthly Spearman rank IC between score and realized return.
    Returns DataFrame with one row per month: date, ic
    """
    records = []
    for date, group in pred_df.groupby(date_col):
        valid = group[[score_col, target]].dropna()
        if len(valid) < 10:
            continue
        ic, _ = stats.spearmanr(valid[score_col], valid[target])
        records.append({'date': date, 'ic': ic})
    return pd.DataFrame(records)


def compute_topq_precision(pred_df, date_col, score_col, target, nq=5):
    """
    Among stocks ranked in the top quintile by score, what fraction
    are actually in the top quintile of realized returns?
    """
    records = []
    for date, group in pred_df.groupby(date_col):
        valid = group[[score_col, target]].dropna()
        if len(valid) < nq * 2:
            continue
        # Top quintile by prediction
        score_threshold = valid[score_col].quantile(1.0 - 1.0/nq)
        pred_top = valid[valid[score_col] >= score_threshold]
        # Top quintile by realized return
        ret_threshold = valid[target].quantile(1.0 - 1.0/nq)
        # Fraction of predicted top that are actually in realized top
        if len(pred_top) == 0:
            continue
        precision = (pred_top[target] >= ret_threshold).mean()
        records.append({'date': date, 'precision': precision, 'n_top': len(pred_top)})
    return pd.DataFrame(records)


def compute_prediction_stability(pred_df, date_col, id_col, score_col):
    """
    Month-to-month Spearman rank correlation of predictions.
    For each consecutive pair of months, compute corr(rank_t, rank_{t+1})
    for stocks present in both months.
    """
    dates = sorted(pred_df[date_col].unique())
    records = []
    for i in range(1, len(dates)):
        prev_date = dates[i-1]
        curr_date = dates[i]
        prev = pred_df[pred_df[date_col] == prev_date][[id_col, score_col]].drop_duplicates(subset=[id_col])
        curr = pred_df[pred_df[date_col] == curr_date][[id_col, score_col]].drop_duplicates(subset=[id_col])
        merged = prev.merge(curr, on=id_col, suffixes=('_prev', '_curr'))
        if len(merged) < 10:
            continue
        rho, _ = stats.spearmanr(merged[f'{score_col}_prev'].values, merged[f'{score_col}_curr'].values)
        records.append({'date': curr_date, 'stability': rho, 'n_common': len(merged)})
    return pd.DataFrame(records)


# ==============================================================================
# MAIN
# ==============================================================================

print("=" * 100)
print("EXPERIMENT AK: Stock-Level Information Coefficient")
print("=" * 100)

all_results = []
all_ic_details = []  # Monthly IC details for optional deeper analysis
t0 = time.time()

for mkt_name, cfg in MARKETS.items():
    print(f"\n{'#'*80}")
    print(f"# MARKET: {mkt_name}")
    print(f"{'#'*80}")

    try:
        df, raw_features, date_col, id_col = load_equity_data(cfg['file'], cfg['type'])
    except Exception as e:
        print(f"  ERROR loading {mkt_name}: {e}")
        continue

    target = cfg['target']
    all_features = [f for f in raw_features if f not in EXCLUDE_COLS and f != target]
    accounting_feats, market_feats = classify_features(all_features)
    accounting_feats = [f for f in accounting_feats if f in df.columns]
    market_feats = [f for f in market_feats if f in df.columns]

    # CRITICAL: Lag BEFORE filter. All features lag=1: JKP applies the 4-month
    # accounting lag internally, so lag=4 here would double-lag (8m effective).
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

    ml_features = [f for f in accounting_feats + market_feats if f in df.columns]
    slow_feats = [f for f in ml_features if classify_feature_speed(f) == 'SLOW']
    fast_feats = [f for f in ml_features if classify_feature_speed(f) == 'FAST']

    print(f"  Features: {len(ml_features)} total = {len(slow_feats)} SLOW + {len(fast_feats)} FAST")

    df[date_col] = pd.to_datetime(df[date_col])
    nq = cfg['nq']

    # IS/OOS split (same logic as expAH)
    unique_dates = sorted(df[date_col].unique())
    n_total = len(unique_dates)
    is_split_idx = max(int(n_total * 0.8), MIN_TRAIN_MONTHS)
    is_end_date = unique_dates[min(is_split_idx - 1, n_total - 1)]

    print(f"  IS ends: {is_end_date}, OOS months: {n_total - is_split_idx}")

    # Build prediction DataFrames for each strategy
    strategies = {}

    # MomentumSort: score = ret_12_1
    if 'ret_12_1' in df.columns:
        mom_pred = df[[date_col, id_col, target, 'ret_12_1']].copy()
        mom_pred = mom_pred.rename(columns={'ret_12_1': 'prediction'})
        mom_pred = mom_pred.dropna(subset=['prediction', target])
        strategies['MomentumSort'] = mom_pred

    # Ridge_All
    print(f"  Training Ridge_All ({len(ml_features)} features)...")
    pred_all = run_ml_predictions(df, ml_features, date_col, id_col, target,
                                   min_train=MIN_TRAIN_MONTHS,
                                   oos_start_date=is_end_date)
    if pred_all is not None:
        strategies['Ridge_All'] = pred_all

    # Ridge_Slow
    if len(slow_feats) >= 5:
        print(f"  Training Ridge_Slow ({len(slow_feats)} features)...")
        pred_slow = run_ml_predictions(df, slow_feats, date_col, id_col, target,
                                        min_train=MIN_TRAIN_MONTHS,
                                        oos_start_date=is_end_date)
        if pred_slow is not None:
            strategies['Ridge_Slow'] = pred_slow

    # Ridge_Fast
    if len(fast_feats) >= 5:
        print(f"  Training Ridge_Fast ({len(fast_feats)} features)...")
        pred_fast = run_ml_predictions(df, fast_feats, date_col, id_col, target,
                                        min_train=MIN_TRAIN_MONTHS,
                                        oos_start_date=is_end_date)
        if pred_fast is not None:
            strategies['Ridge_Fast'] = pred_fast

    # CompositeSort: z-score composite of all features (no fitting)
    # This is the sort-based analog of Ridge_All — same features, different aggregation.
    # Provides an apples-to-apples IC comparison (no in-sample optimization).
    if len(ml_features) >= 5:
        print(f"  Computing CompositeSort ({len(ml_features)} features, no fitting)...")
        comp_pred = df[[date_col, id_col, target]].copy()
        z_cols = []
        for feat in ml_features:
            if feat in df.columns:
                col_name = f'_z_{feat}'
                comp_pred[col_name] = df.groupby(date_col)[feat].transform(
                    lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0
                )
                z_cols.append(col_name)
        if len(z_cols) > 0:
            comp_pred['prediction'] = comp_pred[z_cols].mean(axis=1)
            comp_pred = comp_pred[[date_col, id_col, target, 'prediction']].dropna(
                subset=['prediction', target]
            )
            strategies['CompositeSort'] = comp_pred

    # Evaluate each strategy
    for strat_name, pred_df in strategies.items():
        # Filter to OOS
        oos_pred = pred_df[pred_df[date_col] > is_end_date].copy()
        if len(oos_pred) < 50:
            print(f"    {strat_name}: Not enough OOS data")
            continue

        # Monthly IC
        ic_df = compute_monthly_ic(oos_pred, date_col, 'prediction', target)
        mean_ic = ic_df['ic'].mean() if len(ic_df) > 0 else np.nan
        ic_std = ic_df['ic'].std() if len(ic_df) > 1 else np.nan
        ic_ir = mean_ic / ic_std if ic_std > 0 else 0.0
        hit_rate = (ic_df['ic'] > 0).mean() if len(ic_df) > 0 else np.nan

        # Top-Q precision
        prec_df = compute_topq_precision(oos_pred, date_col, 'prediction', target, nq=nq)
        mean_precision = prec_df['precision'].mean() if len(prec_df) > 0 else np.nan

        # Prediction stability
        stab_df = compute_prediction_stability(oos_pred, date_col, id_col, 'prediction')
        mean_stability = stab_df['stability'].mean() if len(stab_df) > 0 else np.nan

        result = {
            'Market': mkt_name,
            'Strategy': strat_name,
            'Mean_IC': round(mean_ic, 4) if not np.isnan(mean_ic) else np.nan,
            'IC_Std': round(ic_std, 4) if not np.isnan(ic_std) else np.nan,
            'IC_IR': round(ic_ir, 4),
            'Hit_Rate': round(hit_rate, 4) if not np.isnan(hit_rate) else np.nan,
            'TopQ_Precision': round(mean_precision, 4) if not np.isnan(mean_precision) else np.nan,
            'Pred_Stability': round(mean_stability, 4) if not np.isnan(mean_stability) else np.nan,
            'N_OOS_Months': len(ic_df),
        }
        all_results.append(result)

        print(f"    {strat_name:15s}  IC={mean_ic:.4f}  IC_IR={ic_ir:.2f}  "
              f"HitRate={hit_rate:.1%}  Precision={mean_precision:.3f}  "
              f"Stability={mean_stability:.3f}")

        # Store monthly IC details
        for _, row in ic_df.iterrows():
            all_ic_details.append({
                'Market': mkt_name, 'Strategy': strat_name,
                'Date': row['date'], 'IC': row['ic']
            })

elapsed = time.time() - t0
print(f"\n\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")


# ==============================================================================
# SAVE AND SUMMARY
# ==============================================================================

# Save main results
results_df = pd.DataFrame(all_results)
outfile = 'scripts/concern_experiments/expAK_information_coefficient.csv'
results_df.to_csv(outfile, index=False)
print(f"\nSaved {len(results_df)} rows to {outfile}")

# Save IC details
ic_details_df = pd.DataFrame(all_ic_details)
ic_outfile = 'scripts/concern_experiments/expAK_ic_monthly_details.csv'
ic_details_df.to_csv(ic_outfile, index=False)
print(f"Saved {len(ic_details_df)} IC detail rows to {ic_outfile}")

# Summary tables
print("\n\n" + "="*100)
print("SUMMARY: Mean Information Coefficient (OOS)")
print("="*100)
if len(results_df) > 0:
    for metric in ['Mean_IC', 'IC_IR', 'Hit_Rate', 'TopQ_Precision', 'Pred_Stability']:
        print(f"\n--- {metric} ---")
        pivot = results_df.pivot_table(index='Market', columns='Strategy', values=metric)
        strat_order = ['MomentumSort', 'CompositeSort', 'Ridge_All', 'Ridge_Slow', 'Ridge_Fast']
        strat_order = [s for s in strat_order if s in pivot.columns]
        pivot = pivot[strat_order]
        print(pivot.round(4).to_string())
        print(f"\nMean:")
        print(pivot.mean().round(4).to_string())

# Cross-strategy comparison (the key table for the paper)
print("\n\n" + "="*100)
print("KEY TABLE: IC vs Stability Tradeoff (Average across 15 markets)")
print("="*100)
if len(results_df) > 0:
    summary = results_df.groupby('Strategy')[['Mean_IC', 'IC_IR', 'Hit_Rate',
                                                'TopQ_Precision', 'Pred_Stability']].mean()
    strat_order = ['MomentumSort', 'CompositeSort', 'Ridge_All', 'Ridge_Slow', 'Ridge_Fast']
    strat_order = [s for s in strat_order if s in summary.index]
    summary = summary.loc[strat_order]
    print(summary.round(4).to_string())
    print("\nNote: CompositeSort uses the same features as Ridge_All but with equal-weighted")
    print("z-scores (no fitting). Comparing its IC with Ridge_All isolates the benefit of")
    print("learned weights vs the cost of in-sample optimization.")

print("\nDone!")
