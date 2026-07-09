"""
Data loading and preprocessing utilities.

This module handles loading equity and bond data, standardizing column names,
and preparing features for model training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict


# Standard column mappings for different data sources
COLUMN_MAPPINGS = {
    'us_equity': {
        'date_col': 'date',
        'id_col': 'permno',
        'target_col': 'return',
        'date_format': '%Y%m%d',
    },
    'thai_equity': {
        'date_col': 'eom',
        'id_col': 'gvkey',
        'target_col': 'ret_1_0',
        'date_format': '%Y-%m-%d',
    },
    'emerging_equity': {
        'date_col': 'eom',
        'id_col': 'gvkey',
        'target_col': 'ret_1_0',
        'date_format': '%Y-%m-%d',
    },
}

# Identifier columns to exclude from features
IDENTIFIER_COLUMNS = ['date', 'eom', 'permno', 'gvkey', 'id', 'excntry', 'Unnamed: 0']


def load_equity_data(
    filepath: str,
    market_type: str = 'us_equity',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, List[str], str, str]:
    """
    Load equity data from CSV file with standardized preprocessing.

    Parameters
    ----------
    filepath : str
        Path to the equity data CSV file
    market_type : str
        Type of market ('us_equity', 'thai_equity', or 'emerging_equity')
    start_date : str, optional
        Start date filter (format: 'YYYY-MM-DD')
    end_date : str, optional
        End date filter (format: 'YYYY-MM-DD')

    Returns
    -------
    df : pd.DataFrame
        Loaded and preprocessed dataframe
    features : list
        List of feature column names
    date_col : str
        Name of date column
    id_col : str
        Name of entity identifier column
    """
    config = COLUMN_MAPPINGS.get(market_type, COLUMN_MAPPINGS['emerging_equity'])

    # Load data
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} rows from {filepath}")

    # Standardize date column
    date_col = config['date_col']
    if date_col not in df.columns and 'eom' in df.columns:
        df.rename(columns={'eom': 'date'}, inplace=True)
        date_col = 'date'

    # Convert date
    if df[date_col].dtype in ['int64', 'float64']:
        # Handle numeric dates (YYYYMMDD format)
        df[date_col] = pd.to_datetime(df[date_col].astype(int).astype(str), format=config['date_format'])
    else:
        df[date_col] = pd.to_datetime(df[date_col])

    # Apply date filters
    if start_date:
        df = df[df[date_col] >= start_date].copy()
    if end_date:
        df = df[df[date_col] <= end_date].copy()

    # Get target column
    target_col = config['target_col']
    if target_col not in df.columns:
        # Try alternate target names
        for alt in ['return', 'ret_1_0', 'ret']:
            if alt in df.columns:
                target_col = alt
                break

    # Remove rows without target
    df = df.dropna(subset=[target_col]).copy()

    # Handle categorical variables (encode size_grp if present)
    if 'size_grp' in df.columns:
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        df['size_grp'] = label_encoder.fit_transform(df['size_grp'].astype(str))
        print(f"Encoded categorical variable 'size_grp' ({len(label_encoder.classes_)} categories)")

    # Identify feature columns
    id_col = config['id_col']
    exclude_cols = IDENTIFIER_COLUMNS + [target_col]
    features = [col for col in df.columns if col not in exclude_cols]

    print(f"Date range: {df[date_col].min()} to {df[date_col].max()}")
    print(f"Unique dates: {df[date_col].nunique()}")
    print(f"Unique entities: {df[id_col].nunique()}")
    print(f"Features: {len(features)}")

    return df, features, date_col, id_col


def load_bond_factors(filepath: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load bond market factors from CSV file.

    Parameters
    ----------
    filepath : str
        Path to bond factors CSV file

    Returns
    -------
    df : pd.DataFrame
        Bond factors dataframe
    bond_cols : list
        List of bond factor column names
    """
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])

    bond_cols = [col for col in df.columns if col != 'date']

    print(f"Loaded {len(df)} months of bond factors")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Bond factors: {len(bond_cols)}")

    return df, bond_cols


def merge_bond_features(
    equity_df: pd.DataFrame,
    bond_df: pd.DataFrame,
    date_col: str = 'date',
    filter_to_overlap: bool = True,
) -> Tuple[pd.DataFrame, str, str]:
    """
    Merge bond factors with equity data on date.

    Parameters
    ----------
    equity_df : pd.DataFrame
        Equity data
    bond_df : pd.DataFrame
        Bond factors
    date_col : str
        Date column name in equity data
    filter_to_overlap : bool
        If True, filter to overlapping date range only

    Returns
    -------
    merged_df : pd.DataFrame
        Merged dataframe
    overlap_start : str
        Start of overlap period
    overlap_end : str
        End of overlap period
    """
    # Find overlap period
    overlap_start = max(equity_df[date_col].min(), bond_df['date'].min())
    overlap_end = min(equity_df[date_col].max(), bond_df['date'].max())

    print(f"Equity data range: {equity_df[date_col].min()} to {equity_df[date_col].max()}")
    print(f"Bond data range: {bond_df['date'].min()} to {bond_df['date'].max()}")
    print(f"Overlap period: {overlap_start} to {overlap_end}")

    if filter_to_overlap:
        equity_df = equity_df[
            (equity_df[date_col] >= overlap_start) &
            (equity_df[date_col] <= overlap_end)
        ].copy()
        print(f"Filtered equity data to {len(equity_df):,} rows")

    # Merge
    if date_col != 'date':
        bond_df = bond_df.rename(columns={'date': date_col})

    merged_df = equity_df.merge(bond_df, on=date_col, how='left')

    # Verify merge quality
    bond_cols = [col for col in bond_df.columns if col != date_col]
    coverage = merged_df[bond_cols[0]].notna().mean() * 100
    print(f"Bond factor coverage: {coverage:.1f}%")

    return merged_df, str(overlap_start), str(overlap_end)


def lag_features(
    df: pd.DataFrame,
    features: List[str],
    id_col: str,
    date_col: str,
    lag: int = 1,
) -> pd.DataFrame:
    """
    Lag features by specified number of periods within each entity.

    This ensures features from month T-1 are used to predict returns at month T,
    preventing look-ahead bias.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    features : list
        Feature columns to lag
    id_col : str
        Entity identifier column
    date_col : str
        Date column
    lag : int
        Number of periods to lag (default: 1)

    Returns
    -------
    df_lagged : pd.DataFrame
        Dataframe with lagged features
    """
    # Sort by entity and date
    df = df.sort_values(by=[id_col, date_col]).reset_index(drop=True)

    # Lag features within each entity
    df_lagged = df.copy()
    df_lagged[features] = df.groupby(id_col)[features].shift(lag)

    # Remove rows where lagging created NaN (first observation per entity)
    initial_rows = len(df_lagged)
    df_lagged = df_lagged.dropna(subset=features[:5], how='all')
    removed = initial_rows - len(df_lagged)

    print(f"Lagged {len(features)} features by {lag} period(s)")
    print(f"Removed {removed:,} rows (first obs per entity)")
    print(f"Final shape: {df_lagged.shape}")

    return df_lagged


def get_feature_groups(features: List[str]) -> Dict[str, List[str]]:
    """
    Categorize features into groups based on naming conventions.

    Parameters
    ----------
    features : list
        List of feature names

    Returns
    -------
    groups : dict
        Dictionary mapping group names to feature lists
    """
    groups = {
        'momentum': [],
        'value': [],
        'profitability': [],
        'risk': [],
        'liquidity': [],
        'quality': [],
        'bond': [],
        'other': [],
    }

    # Bond features
    bond_prefixes = ['bond_', 'ig_', 'hy_', 'credit_', 'term_']

    for feat in features:
        feat_lower = feat.lower()

        if any(feat_lower.startswith(p) for p in bond_prefixes):
            groups['bond'].append(feat)
        elif any(x in feat_lower for x in ['ret_', 'mom', 'r12', 'r6', 'r36', 'r60']):
            groups['momentum'].append(feat)
        elif any(x in feat_lower for x in ['me', 'be', 'bm', 'ep', 'cf', 'sale']):
            groups['value'].append(feat)
        elif any(x in feat_lower for x in ['roa', 'roe', 'prof', 'gp', 'ni', 'op']):
            groups['profitability'].append(feat)
        elif any(x in feat_lower for x in ['vol', 'var', 'beta', 'ivol', 'rvol']):
            groups['risk'].append(feat)
        elif any(x in feat_lower for x in ['turn', 'ami', 'spread', 'zero', 'dolvol']):
            groups['liquidity'].append(feat)
        elif any(x in feat_lower for x in ['score', 'quality', 'qmj']):
            groups['quality'].append(feat)
        else:
            groups['other'].append(feat)

    return groups
