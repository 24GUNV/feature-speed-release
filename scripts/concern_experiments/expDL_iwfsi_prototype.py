"""
Experiment DL: Importance-Weighted FSI (IW-FSI) prototype.

Tests whether weighting the Feature Speed Index by ML feature importance
(instead of equal-weighting) adds within-strategy, across-market resolution.

Static in-sample version (IS window <= 2020-12). Everything used to BUILD
IW-FSI comes from the IS window only; OOS outcomes (2021+) are correlated
against as ex-post realizations.

Output: expDL_iwfsi_prototype.csv
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
from scipy import stats

from data_loader import load_equity_data, lag_features
from market_config import MARKETS, MIN_TRAIN_MONTHS

# NOTE: expAH_feature_speed_test.py has no __main__ guard, so importing it would
# rerun the entire 15-market expAH experiment. The classification lists/functions
# below are copied VERBATIM from expAH_feature_speed_test.py (keep in sync).

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


np.random.seed(42)

IS_END = pd.Timestamp('2020-12-31')  # OOS boundary is 2020-12
XGB_AVAILABLE = True
try:
    from xgboost import XGBRegressor
except Exception as e:
    XGB_AVAILABLE = False
    print(f"XGBoost unavailable: {e}")


def compute_ac(is_data, feats, date_col, id_col):
    """Mean cross-sectional Spearman rank autocorrelation of each feature
    between consecutive months, over the IS window.

    Vectorized: (1) cross-sectional pct-rank per month; (2) align each stock's
    current-month rank with its rank one month earlier via an ordinal month key;
    (3) per-feature, take the mean over month-pairs of the within-month Pearson
    correlation of (prev_rank, curr_rank) -- Pearson on ranks == Spearman.
    """
    # Cross-sectional percentile rank per month
    ranked = is_data[[date_col, id_col]].copy()
    ranked[feats] = is_data.groupby(date_col)[feats].rank(pct=True)

    # Ordinal month index (0..M-1) so 'previous month' is robust to month-end spacing
    dates_sorted = sorted(ranked[date_col].unique())
    ord_map = {d: i for i, d in enumerate(dates_sorted)}
    ranked['_ord'] = ranked[date_col].map(ord_map)

    curr = ranked.drop(columns=[date_col])
    prev = curr.copy()
    prev['_ord'] = prev['_ord'] + 1  # prev row now carries the month that precedes _ord
    merged = curr.merge(prev, on=[id_col, '_ord'], suffixes=('_c', '_p'))

    ordv = merged['_ord']
    C = merged[[f + '_c' for f in feats]].to_numpy(dtype=float).copy()
    P = merged[[f + '_p' for f in feats]].to_numpy(dtype=float).copy()
    # Joint-mask so per-feature sums only count rows where both months are present
    joint_nan = np.isnan(C) | np.isnan(P)
    C[joint_nan] = np.nan
    P[joint_nan] = np.nan

    Cdf = pd.DataFrame(C, columns=feats); Cdf['_ord'] = ordv.values
    Pdf = pd.DataFrame(P, columns=feats); Pdf['_ord'] = ordv.values
    CPdf = pd.DataFrame(C * P, columns=feats); CPdf['_ord'] = ordv.values
    C2df = pd.DataFrame(C * C, columns=feats); C2df['_ord'] = ordv.values
    P2df = pd.DataFrame(P * P, columns=feats); P2df['_ord'] = ordv.values

    n = Cdf.groupby('_ord').count()          # per-month non-null count, per feature
    sx = Cdf.groupby('_ord').sum(min_count=1)
    sy = Pdf.groupby('_ord').sum(min_count=1)
    sxy = CPdf.groupby('_ord').sum(min_count=1)
    sxx = C2df.groupby('_ord').sum(min_count=1)
    syy = P2df.groupby('_ord').sum(min_count=1)

    cov = sxy - sx * sy / n
    vx = sxx - sx * sx / n
    vy = syy - sy * sy / n
    vxy = vx * vy
    corr = cov / np.sqrt(vxy.where(vxy > 0))  # guard zero/neg variance -> NaN
    corr = corr.where(n >= 10)                # drop thin months
    corr = corr.replace([np.inf, -np.inf], np.nan)
    ac = corr.mean(axis=0, skipna=True)       # average over month-pairs, per feature
    return ac.reindex(feats)


def _valid_feats(is_data, feats):
    """Features with at least one non-NaN value in the IS window (SimpleImputer
    silently drops all-NaN columns, which would misalign coefficients)."""
    return [f for f in feats if is_data[f].notna().any()]


def fit_ridge_weights(is_data, feats, date_col, target):
    """Ridge alpha=100 on rank target, IS window. Return |std coef| weights,
    reindexed to the full feature set (dropped all-NaN cols get weight 0)."""
    vf = _valid_feats(is_data, feats)
    X = is_data[vf].copy()
    y = is_data.groupby(date_col)[target].rank(pct=True).values
    imp = SimpleImputer(strategy='median')
    Xi = imp.fit_transform(X)
    mask = ~np.isnan(y)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xi[mask])
    m = Ridge(alpha=100, random_state=42)
    m.fit(Xs, y[mask])
    return pd.Series(np.abs(m.coef_), index=vf).reindex(feats, fill_value=0.0)


def fit_xgb_weights(is_data, feats, date_col, target):
    """XGBoost gain importances on rank target, IS window."""
    vf = _valid_feats(is_data, feats)
    X = is_data[vf].copy()
    y = is_data.groupby(date_col)[target].rank(pct=True).values
    imp = SimpleImputer(strategy='median')
    Xi = imp.fit_transform(X)
    mask = ~np.isnan(y)
    m = XGBRegressor(n_estimators=200, max_depth=4, random_state=42,
                     n_jobs=4, verbosity=0)
    m.fit(Xi[mask], y[mask])
    gain = m.get_booster().get_score(importance_type='gain')  # keys f0,f1,...
    w = np.zeros(len(vf))
    for k, v in gain.items():
        w[int(k[1:])] = v
    return pd.Series(w, index=vf).reindex(feats, fill_value=0.0)


def iw_fsi(weights, one_minus_ac):
    common = weights.index.intersection(one_minus_ac.dropna().index)
    w = weights[common].values.astype(float)
    speed = one_minus_ac[common].values.astype(float)
    if w.sum() <= 0:
        return np.nan
    w = w / w.sum()
    return float(np.sum(w * speed))


# ==========================================================================
# Load realized OOS outcomes
# ==========================================================================
realized = pd.read_csv('scripts/concern_experiments/expAH_feature_speed_test.csv')


def get_realized(market, strategy):
    sub = realized[(realized.Market == market) & (realized.Strategy == strategy) &
                   (realized.CostMult == 3.0) & (realized.Period == 'OOS')]
    if len(sub) == 0:
        return None
    r = sub.iloc[0]
    return {'turnover': r.Turnover, 'net_sr': r.Net_SR, 'gross_sr': r.Gross_SR,
            'degradation': r.Gross_SR - r.Net_SR}


# ==========================================================================
# Main loop
# ==========================================================================
rows = []
timings = {}

for mkt, cfg in MARKETS.items():
    print(f"\n{'='*60}\n{mkt}\n{'='*60}")
    try:
        df, raw_features, date_col, id_col = load_equity_data(cfg['file'], cfg['type'])
    except Exception as e:
        print(f"  SKIP {mkt}: load error {e}")
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

    df[date_col] = pd.to_datetime(df[date_col])
    ml_features = [f for f in accounting_feats + market_feats if f in df.columns]
    slow_feats = [f for f in ml_features if classify_feature_speed(f) == 'SLOW']
    fast_feats = set(f for f in ml_features if classify_feature_speed(f) == 'FAST')

    # IS window only
    is_data = df[df[date_col] <= IS_END].copy()
    n_is_months = is_data[date_col].nunique()
    print(f"  IS months: {n_is_months}, features: {len(ml_features)} "
          f"({len(slow_feats)} slow, {len(fast_feats)} fast)")
    if n_is_months < MIN_TRAIN_MONTHS or len(is_data) < 100:
        print(f"  SKIP {mkt}: insufficient IS data")
        continue

    t = time.time()
    one_minus_ac = 1.0 - compute_ac(is_data, ml_features, date_col, id_col)
    timings.setdefault('ac', []).append(time.time() - t)

    ew_fsi = float(one_minus_ac.mean(skipna=True))

    t = time.time()
    ridge_w = fit_ridge_weights(is_data, ml_features, date_col, target)
    timings.setdefault('ridge', []).append(time.time() - t)
    iw_fsi_ridge = iw_fsi(ridge_w, one_minus_ac)

    # fast weight share (Ridge)
    total_w = ridge_w.sum()
    fast_share = float(ridge_w[[f for f in ml_features if f in fast_feats]].sum() / total_w) if total_w > 0 else np.nan

    iw_fsi_xgb = np.nan
    if XGB_AVAILABLE:
        t = time.time()
        xgb_w = fit_xgb_weights(is_data, ml_features, date_col, target)
        timings.setdefault('xgb', []).append(time.time() - t)
        iw_fsi_xgb = iw_fsi(xgb_w, one_minus_ac)

    # IW-FSI restricted to slow-feature Ridge (Ridge_Slow analog)
    iw_fsi_ridge_slow = np.nan
    if len(slow_feats) >= 5:
        rw_slow = fit_ridge_weights(is_data, slow_feats, date_col, target)
        iw_fsi_ridge_slow = iw_fsi(rw_slow, one_minus_ac)

    r_all = get_realized(mkt, 'Ridge_All')
    r_slow = get_realized(mkt, 'Ridge_Slow')

    rows.append({
        'market': mkt,
        'n_is_months': n_is_months,
        'n_features': len(ml_features),
        'ew_fsi': ew_fsi,
        'iw_fsi_ridge': iw_fsi_ridge,
        'iw_fsi_xgb': iw_fsi_xgb,
        'iw_fsi_ridge_slow': iw_fsi_ridge_slow,
        'fast_weight_share': fast_share,
        'realized_turnover': r_all['turnover'] if r_all else np.nan,
        'degradation': r_all['degradation'] if r_all else np.nan,
        'net_sr': r_all['net_sr'] if r_all else np.nan,
        'slow_turnover': r_slow['turnover'] if r_slow else np.nan,
        'slow_degradation': r_slow['degradation'] if r_slow else np.nan,
        'slow_net_sr': r_slow['net_sr'] if r_slow else np.nan,
    })
    print(f"  EW-FSI={ew_fsi:.4f}  IW-FSI_ridge={iw_fsi_ridge:.4f}  "
          f"IW-FSI_xgb={iw_fsi_xgb:.4f}  fast_share={fast_share:.3f}")

out = pd.DataFrame(rows)
outfile = 'scripts/concern_experiments/expDL_iwfsi_prototype.csv'
out.to_csv(outfile, index=False)
print(f"\nSaved {len(out)} markets to {outfile}")

# ==========================================================================
# PAYOFF ANALYSIS
# ==========================================================================
def corr_r2(x, y):
    d = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(d) < 4:
        return (np.nan, np.nan, np.nan, len(d))
    pear = d['x'].corr(d['y'])
    spear = d['x'].corr(d['y'], method='spearman')
    return (pear, pear**2, spear, len(d))

print("\n" + "="*78)
print("PAYOFF: within-strategy across-market association (Ridge_All, OOS @3x)")
print("="*78)
print(f"{'predictor':<22}{'target':<14}{'pearson':>9}{'R2':>8}{'spearman':>10}{'n':>4}")
for pred in ['ew_fsi', 'iw_fsi_ridge', 'iw_fsi_xgb']:
    for tgt in ['realized_turnover', 'degradation']:
        p, r2, sp, n = corr_r2(out[pred], out[tgt])
        print(f"{pred:<22}{tgt:<14}{p:>9.3f}{r2:>8.3f}{sp:>10.3f}{n:>4}")

print("\n--- Ridge_Slow outcomes vs IW-FSI_ridge_slow ---")
for tgt in ['slow_turnover', 'slow_degradation']:
    p, r2, sp, n = corr_r2(out['iw_fsi_ridge_slow'], out[tgt])
    print(f"{'iw_fsi_ridge_slow':<22}{tgt:<16}{p:>9.3f}{r2:>8.3f}{sp:>10.3f}{n:>4}")

print("\n--- Sanity checks ---")
print(f"IW-FSI_ridge > EW-FSI in {int((out.iw_fsi_ridge > out.ew_fsi).sum())}/{len(out)} markets")
if XGB_AVAILABLE:
    print(f"IW-FSI_xgb   > EW-FSI in {int((out.iw_fsi_xgb > out.ew_fsi).sum())}/{len(out)} markets")
print(f"Ridge fast-weight share: mean={out.fast_weight_share.mean():.3f} "
      f"range=[{out.fast_weight_share.min():.3f}, {out.fast_weight_share.max():.3f}]")
print(f"EW-FSI across markets: mean={out.ew_fsi.mean():.4f} std={out.ew_fsi.std():.4f} "
      f"(CV={out.ew_fsi.std()/out.ew_fsi.mean():.3f})")
print(f"IW-FSI_ridge across markets: mean={out.iw_fsi_ridge.mean():.4f} "
      f"std={out.iw_fsi_ridge.std():.4f} (CV={out.iw_fsi_ridge.std()/out.iw_fsi_ridge.mean():.3f})")

print("\n--- Timings (sum over markets) ---")
for k, v in timings.items():
    print(f"  {k}: {np.sum(v):.1f}s total, {np.mean(v):.2f}s/market")
