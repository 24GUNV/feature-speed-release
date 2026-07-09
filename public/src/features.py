"""
Corrected bond feature engineering and normalization utilities.

This module contains the corrected bond feature pipeline using only the 7 clean
bond features (no ret_exc_lead-derived features) plus 22 engineered features.
See BOND_FEATURE_AUDIT.md for the full timing leak story.

Extracted from scripts/all_markets_engineered.py to centralize the corrected logic.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# ── Clean Bond Features (no ret_exc_lead contamination) ─────────────────────

CLEAN_BOND_FEATURES = [
    'ig_ratio', 'ig_ratio_vw', 'ig_ratio_change',
    'bond_mkt_avg_duration', 'bond_mkt_avg_yield',
    'bond_mkt_duration_change', 'bond_mkt_yield_change',
]


# ── Feature Engineering ─────────────────────────────────────────────────────

def engineer_bond_features(bond_df):
    """
    Engineer 22 features from the 7 clean bond level features.

    Creates momentum, volatility, credit quality, yield curve, and PCA features
    from yield, duration, and IG ratio data.

    Parameters
    ----------
    bond_df : pd.DataFrame
        Bond data with 'date' column and CLEAN_BOND_FEATURES columns.

    Returns
    -------
    engineered : pd.DataFrame
        DataFrame with 'date' + 22 engineered feature columns.
    eng_feature_names : list of str
        Names of the 22 engineered features.
    """
    df = bond_df.copy().sort_values('date').reset_index(drop=True)
    engineered = pd.DataFrame({'date': df['date']})

    # 1. Rolling yield changes
    for window in [3, 6, 12]:
        engineered[f'yield_change_{window}m'] = df['bond_mkt_avg_yield'].diff(window)

    # 2. Rolling yield volatility
    for window in [3, 6, 12]:
        engineered[f'yield_vol_{window}m'] = df['bond_mkt_avg_yield'].rolling(
            window=window, min_periods=max(2, window // 2)).std()

    # 3. Credit quality momentum
    for window in [3, 6]:
        engineered[f'ig_ratio_momentum_{window}m'] = df['ig_ratio'].diff(window)

    # 4. Duration shift
    for window in [3, 6]:
        engineered[f'duration_shift_{window}m'] = df['bond_mkt_avg_duration'].diff(window)

    # 5. Yield-level regime z-score
    rolling_mean_24 = df['bond_mkt_avg_yield'].rolling(window=24, min_periods=12).mean()
    rolling_std_24 = df['bond_mkt_avg_yield'].rolling(window=24, min_periods=12).std()
    engineered['yield_regime_zscore'] = (
        (df['bond_mkt_avg_yield'] - rolling_mean_24) / rolling_std_24.replace(0, np.nan)
    )

    # 6. Credit spread proxies
    engineered['credit_quality_diff'] = df['ig_ratio_vw'] - df['ig_ratio']
    engineered['hy_yield_proxy'] = df['bond_mkt_avg_yield'] * (1 - df['ig_ratio'])
    engineered['hy_yield_proxy_change_3m'] = engineered['hy_yield_proxy'].diff(3)
    engineered['hy_yield_proxy_change_6m'] = engineered['hy_yield_proxy'].diff(6)

    # 7. Yield curve steepness proxies
    engineered['duration_yield_product'] = df['bond_mkt_avg_duration'] * df['bond_mkt_avg_yield']
    engineered['dur_yield_change_3m'] = engineered['duration_yield_product'].diff(3)
    engineered['dur_yield_change_6m'] = engineered['duration_yield_product'].diff(6)
    yield_chg = df['bond_mkt_avg_yield'].diff(3)
    dur_chg = df['bond_mkt_avg_duration'].diff(3)
    engineered['yield_duration_ratio_3m'] = yield_chg / dur_chg.replace(0, np.nan)

    # 8. Rolling PCA composite
    clean_cols = ['bond_mkt_avg_yield', 'bond_mkt_avg_duration', 'ig_ratio',
                  'ig_ratio_vw', 'bond_mkt_yield_change', 'bond_mkt_duration_change',
                  'ig_ratio_change']
    pca_input = df[clean_cols].copy()
    pca_scores = np.full(len(df), np.nan)
    window_pca = 12
    for i in range(window_pca, len(df)):
        block = pca_input.iloc[i - window_pca:i].dropna()
        if len(block) < window_pca // 2:
            continue
        block_z = (block - block.mean()) / block.std().replace(0, 1)
        block_z = block_z.fillna(0)
        pca = PCA(n_components=1)
        scores = pca.fit_transform(block_z.values)
        pca_scores[i] = scores[-1, 0]
    engineered['bond_pca_credit_index'] = pca_scores

    # 9. Yield vol acceleration
    engineered['yield_vol_accel_3m'] = engineered['yield_vol_3m'].diff(3)

    # 10. IG ratio range position
    ig_roll_min = df['ig_ratio'].rolling(24, min_periods=12).min()
    ig_roll_max = df['ig_ratio'].rolling(24, min_periods=12).max()
    ig_range = (ig_roll_max - ig_roll_min).replace(0, np.nan)
    engineered['ig_ratio_range_pct'] = (df['ig_ratio'] - ig_roll_min) / ig_range

    eng_feature_names = [c for c in engineered.columns if c != 'date']
    return engineered, eng_feature_names


# ── Normalization ────────────────────────────────────────────────────────────

def cross_sectional_rank_normalize(df, equity_features, date_col):
    """
    Rank-normalize equity features cross-sectionally within each month.

    Parameters
    ----------
    df : pd.DataFrame
        Data with equity features.
    equity_features : list of str
        Equity feature column names to normalize.
    date_col : str
        Date column name.

    Returns
    -------
    df : pd.DataFrame
        Copy with rank-normalized equity features (values in [0, 1]).
    """
    df = df.copy()
    for feat in equity_features:
        if feat in df.columns:
            df[feat] = df.groupby(date_col)[feat].rank(pct=True)
            df[feat] = df[feat].fillna(0.5)
    return df


def time_series_normalize_bonds(df, bond_features, date_col, window=60):
    """
    Z-score normalize bond features using a rolling time-series window.

    Since bond features are uniform across all stocks in a given month,
    this extracts the monthly level and z-scores against a rolling window.

    Parameters
    ----------
    df : pd.DataFrame
        Data with bond feature columns.
    bond_features : list of str
        Bond feature column names to normalize.
    date_col : str
        Date column name.
    window : int
        Rolling window size in months (default: 60).

    Returns
    -------
    df : pd.DataFrame
        Copy with z-scored bond features.
    """
    df = df.copy()
    monthly_bonds = df.groupby(date_col)[bond_features].first()
    rolling_mean = monthly_bonds.rolling(window=window, min_periods=12).mean()
    rolling_std = monthly_bonds.rolling(window=window, min_periods=12).std()
    z_scored = (monthly_bonds - rolling_mean) / rolling_std.replace(0, np.nan)
    z_scored = z_scored.fillna(0)
    z_map = z_scored.to_dict()
    for feat in bond_features:
        if feat in df.columns:
            df[feat] = df[date_col].map(z_map[feat])
    return df


# ── Filtering ────────────────────────────────────────────────────────────────

def filter_top_n_by_marketcap(df, date_col, id_col, top_n):
    """
    Filter to the top N stocks by market cap each month.

    Parameters
    ----------
    df : pd.DataFrame
        Data with 'me' (market equity) column.
    date_col : str
        Date column name.
    id_col : str
        Entity identifier column name.
    top_n : int
        Number of top stocks to keep per month.

    Returns
    -------
    df : pd.DataFrame
        Filtered copy.
    """
    df = df.copy()
    df['_me_rank'] = df.groupby(date_col)['me'].rank(
        ascending=False, method='first', na_option='bottom'
    )
    return df[df['_me_rank'] <= top_n].drop(columns=['_me_rank'])


# ── Feature Selection ────────────────────────────────────────────────────────

def select_features_by_importance(importances, feature_names, keep_pct=0.7,
                                  always_keep=None):
    """
    Select top features by importance score.

    Parameters
    ----------
    importances : array-like
        Feature importance scores.
    feature_names : list of str
        Corresponding feature names.
    keep_pct : float
        Fraction of features to keep (default: 0.7).
    always_keep : list of str, optional
        Features to always include regardless of importance.

    Returns
    -------
    selected : list of str
        Selected feature names (in original order).
    imp_df : pd.DataFrame
        Full importance ranking.
    """
    if always_keep is None:
        always_keep = []
    imp_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
    }).sort_values('importance', ascending=False)
    n_keep = max(1, int(len(feature_names) * keep_pct))
    top_features = set(imp_df.head(n_keep)['feature'].tolist())
    top_features.update([f for f in always_keep if f in feature_names])
    selected = [f for f in feature_names if f in top_features]
    return selected, imp_df
