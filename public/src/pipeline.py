"""
High-level convenience functions for the corrected bond-augmented pipeline.

Combines data loading, bond feature engineering, merging, and lagging
into a single call. Uses only CLEAN_BOND_FEATURES + 22 engineered features
(no ret_exc_lead contamination).
"""

import pandas as pd

from data_loader import (
    load_equity_data,
    load_bond_factors,
    merge_bond_features,
    lag_features,
)
from features import (
    CLEAN_BOND_FEATURES,
    engineer_bond_features,
    cross_sectional_rank_normalize,
    time_series_normalize_bonds,
)


def load_dataset(equity_path, bond_path, market_type='thai_equity',
                 start_date='2008-01-01', include_bond=True, lag=1):
    """
    Load equity data with corrected bond features in one call.

    Uses only CLEAN_BOND_FEATURES (7) + 22 engineered features.
    All features are lagged by ``lag`` periods.

    Parameters
    ----------
    equity_path : str
        Path to equity CSV file.
    bond_path : str
        Path to bond_market_factors.csv.
    market_type : str
        One of 'us_equity', 'thai_equity', 'emerging_equity'.
    start_date : str
        Start date filter (YYYY-MM-DD).
    include_bond : bool
        If True, merge corrected bond features (default: True).
    lag : int
        Number of periods to lag features (default: 1).

    Returns
    -------
    df : pd.DataFrame
        Lagged dataset ready for modeling.
    features : list of str
        Feature column names (equity + bond if included).
    target_col : str
        Target column name.
    date_col : str
        Date column name.
    id_col : str
        Entity identifier column name.
    """
    # Load equity data
    df_equity, equity_features, date_col, id_col = load_equity_data(
        equity_path, market_type, start_date
    )

    # Determine target column
    from data_loader import COLUMN_MAPPINGS
    config = COLUMN_MAPPINGS.get(market_type, COLUMN_MAPPINGS['emerging_equity'])
    target_col = config['target_col']

    if not include_bond:
        df_lagged = lag_features(df_equity, equity_features, id_col, date_col, lag=lag)
        return df_lagged, equity_features, target_col, date_col, id_col

    # Load bond factors and engineer features
    df_bonds, _ = load_bond_factors(bond_path)
    eng_df, eng_feature_names = engineer_bond_features(df_bonds)

    # Merge engineered features into bond dataframe
    bond_with_eng = df_bonds.merge(eng_df, on='date', how='left')

    # Keep only clean bond features + engineered (drop old contaminated ones)
    keep_cols = ['date'] + CLEAN_BOND_FEATURES + eng_feature_names
    bond_clean = bond_with_eng[[c for c in keep_cols if c in bond_with_eng.columns]]

    # Merge with equity data
    df_merged, _, _ = merge_bond_features(
        df_equity, bond_clean, date_col, filter_to_overlap=True
    )

    # Build feature list
    all_bond_features = CLEAN_BOND_FEATURES + eng_feature_names
    all_features = equity_features + [f for f in all_bond_features if f in df_merged.columns]

    # Lag all features
    df_lagged = lag_features(df_merged, all_features, id_col, date_col, lag=lag)

    return df_lagged, all_features, target_col, date_col, id_col
