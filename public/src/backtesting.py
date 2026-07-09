"""
Walk-forward validation and backtesting utilities.

This module provides the core backtesting framework used in the paper,
including walk-forward validation with expanding windows and proper
train-only imputation.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Callable
from sklearn.impute import SimpleImputer
from tqdm import tqdm


def create_walk_forward_splits(
    df: pd.DataFrame,
    date_col: str = 'date',
    min_train_months: int = 60,
    test_months: int = 12,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create walk-forward validation splits with expanding training windows.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with date column
    date_col : str
        Name of date column
    min_train_months : int
        Minimum number of months for initial training window (default: 60 = 5 years)
    test_months : int
        Number of months in each test window (default: 12 = 1 year)

    Returns
    -------
    splits : list of tuples
        List of (train_months, test_months) arrays

    Example
    -------
    >>> splits = create_walk_forward_splits(df, min_train_months=60, test_months=12)
    >>> for train_months, test_months in splits:
    ...     train_data = df[df['date'].isin(train_months)]
    ...     test_data = df[df['date'].isin(test_months)]
    """
    unique_months = np.sort(df[date_col].unique())
    total_months = len(unique_months)

    splits = []
    current_test_start = min_train_months

    while current_test_start + test_months <= total_months:
        train_months = unique_months[:current_test_start]
        test_months_range = unique_months[current_test_start:current_test_start + test_months]

        splits.append((train_months, test_months_range))
        current_test_start += test_months

    print(f"Created {len(splits)} walk-forward folds")
    print(f"  Initial training: {min_train_months} months")
    print(f"  Test window: {test_months} months")
    print(f"  First fold: {splits[0][0][0]} to {splits[0][0][-1]} (train)")
    print(f"             {splits[0][1][0]} to {splits[0][1][-1]} (test)")

    return splits


def impute_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    strategy: str = 'median',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Impute missing values using training set statistics only.

    This prevents data leakage by fitting the imputer only on training data
    and then transforming both train and test sets.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    strategy : str
        Imputation strategy ('median', 'mean', or 'most_frequent')

    Returns
    -------
    X_train_imputed : pd.DataFrame
        Imputed training features
    X_test_imputed : pd.DataFrame
        Imputed test features
    """
    imputer = SimpleImputer(strategy=strategy)

    # Fit on training data only
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Convert back to DataFrame to preserve column names
    X_train_imputed = pd.DataFrame(
        X_train_imputed,
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_imputed = pd.DataFrame(
        X_test_imputed,
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_imputed, X_test_imputed


def run_walk_forward_experiment(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    date_col: str,
    id_col: str,
    model_fn: Callable,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    impute_strategy: str = 'median',
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """
    Run walk-forward backtesting experiment.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with features, target, dates, and entity IDs
    features : list
        Feature column names
    target_col : str
        Target column name
    date_col : str
        Date column name
    id_col : str
        Entity identifier column name
    model_fn : callable
        Function that returns a fresh model instance
    splits : list
        Walk-forward splits from create_walk_forward_splits()
    impute_strategy : str
        Imputation strategy (default: 'median')
    verbose : bool
        Whether to show progress bar

    Returns
    -------
    predictions_df : pd.DataFrame
        DataFrame with columns: date, id, return, prediction, fold
    feature_importances : np.ndarray or None
        Average feature importances (if model supports it)
    """
    all_predictions = []
    fold_importances = []

    iterator = tqdm(splits, desc="Folds") if verbose else splits

    for fold_idx, (train_months, test_months) in enumerate(iterator):
        # Split data
        train_data = df[df[date_col].isin(train_months)].copy()
        test_data = df[df[date_col].isin(test_months)].copy()

        if len(train_data) < 100 or len(test_data) < 10:
            continue

        # Prepare features and target
        X_train = train_data[features]
        y_train = train_data[target_col]
        X_test = test_data[features]
        y_test = test_data[target_col]

        # Impute missing values using TRAINING statistics only
        X_train_imp, X_test_imp = impute_features(X_train, X_test, strategy=impute_strategy)

        # Create fresh model for this fold
        model = model_fn()

        # Train and predict
        model.fit(X_train_imp, y_train)
        predictions = model.predict(X_test_imp)

        # Store feature importances if available
        if hasattr(model, 'feature_importances_'):
            fold_importances.append(model.feature_importances_)
        elif hasattr(model, 'coef_'):
            fold_importances.append(np.abs(model.coef_))

        # Store predictions with metadata
        pred_df = test_data[[date_col, id_col, target_col]].copy()
        pred_df = pred_df.rename(columns={target_col: 'return'})
        pred_df['prediction'] = predictions
        pred_df['fold'] = fold_idx
        all_predictions.append(pred_df)

    # Combine all predictions
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    # Average feature importances
    feature_importances = None
    if fold_importances:
        feature_importances = np.mean(fold_importances, axis=0)

    return predictions_df, feature_importances


def run_baseline_vs_augmented(
    df: pd.DataFrame,
    equity_features: List[str],
    bond_features: List[str],
    target_col: str,
    date_col: str,
    id_col: str,
    model_configs: Dict[str, Callable],
    min_train_months: int = 60,
    test_months: int = 12,
) -> Dict[str, Dict]:
    """
    Run both baseline (equity-only) and augmented (equity+bond) experiments.

    Parameters
    ----------
    df : pd.DataFrame
        Data with equity features, bond features, target, dates, and IDs
    equity_features : list
        List of equity feature column names
    bond_features : list
        List of bond feature column names
    target_col : str
        Target column name
    date_col : str
        Date column name
    id_col : str
        Entity identifier column name
    model_configs : dict
        Dictionary mapping model names to model factory functions
    min_train_months : int
        Minimum training window size
    test_months : int
        Test window size

    Returns
    -------
    results : dict
        Nested dictionary with structure:
        {model_name: {'baseline': {...}, 'augmented': {...}}}
    """
    # Handle both package import (from src.backtesting) and direct import (sys.path.append)
    try:
        from . import portfolio
        calculate_portfolio_metrics = portfolio.calculate_portfolio_metrics
    except (ImportError, SystemError, ValueError):
        # Fallback for direct import via sys.path.append
        try:
            import portfolio
            calculate_portfolio_metrics = portfolio.calculate_portfolio_metrics
        except ImportError:
            # Try importing from same directory
            import sys
            from pathlib import Path
            src_dir = Path(__file__).parent
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            import portfolio
            calculate_portfolio_metrics = portfolio.calculate_portfolio_metrics

    splits = create_walk_forward_splits(df, date_col, min_train_months, test_months)

    all_features = equity_features + bond_features

    results = {}

    for model_name, model_fn in model_configs.items():
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")

        results[model_name] = {}

        # Baseline (equity only)
        print(f"\n  Baseline (equity only, {len(equity_features)} features)...")
        pred_df, importances = run_walk_forward_experiment(
            df, equity_features, target_col, date_col, id_col, model_fn, splits
        )
        monthly_returns, metrics = calculate_portfolio_metrics(pred_df)
        results[model_name]['baseline'] = {
            'predictions': pred_df,
            'monthly_returns': monthly_returns,
            'metrics': metrics,
            'importances': importances,
        }
        print(f"    Long-Short Sharpe: {metrics['Long_Short']['sharpe']:.3f}")

        # Augmented (equity + bond)
        print(f"\n  Augmented (equity + bond, {len(all_features)} features)...")
        pred_df, importances = run_walk_forward_experiment(
            df, all_features, target_col, date_col, id_col, model_fn, splits
        )
        monthly_returns, metrics = calculate_portfolio_metrics(pred_df)
        results[model_name]['augmented'] = {
            'predictions': pred_df,
            'monthly_returns': monthly_returns,
            'metrics': metrics,
            'importances': importances,
        }
        print(f"    Long-Short Sharpe: {metrics['Long_Short']['sharpe']:.3f}")

        # Calculate improvement
        baseline_sharpe = results[model_name]['baseline']['metrics']['Long_Short']['sharpe']
        augmented_sharpe = results[model_name]['augmented']['metrics']['Long_Short']['sharpe']
        improvement = augmented_sharpe - baseline_sharpe
        improvement_pct = (improvement / baseline_sharpe * 100) if baseline_sharpe != 0 else 0

        results[model_name]['improvement'] = improvement
        results[model_name]['improvement_pct'] = improvement_pct

        print(f"\n    Improvement: {improvement:+.3f} ({improvement_pct:+.1f}%)")

    return results
