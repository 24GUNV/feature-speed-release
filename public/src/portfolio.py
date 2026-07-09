"""
Portfolio construction and performance calculation utilities.

This module implements the decile-based long-short portfolio strategy
used in the paper for evaluating stock return predictions.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional


def calculate_quantile_returns(
    predictions_df: pd.DataFrame,
    date_col: str = 'date',
    n_quantiles: int = 10,
) -> pd.DataFrame:
    """
    Calculate monthly returns for each prediction quantile.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with columns: date, return, prediction
    date_col : str
        Date column name
    n_quantiles : int
        Number of quantiles (default: 10 for deciles)

    Returns
    -------
    monthly_quantile_returns : pd.DataFrame
        DataFrame with columns Q1, Q2, ..., Qn indexed by date
    """
    df = predictions_df.copy()

    # Create quantiles based on predictions for each month
    df['quantile'] = df.groupby(date_col)['prediction'].transform(
        lambda x: pd.qcut(x, n_quantiles, labels=False, duplicates='drop')
    )

    # Calculate monthly returns for each quantile
    monthly_returns = df.groupby([date_col, 'quantile'])['return'].mean().unstack()

    # Rename columns to Q1, Q2, etc.
    monthly_returns.columns = [f'Q{int(i)+1}' for i in monthly_returns.columns]

    return monthly_returns


def calculate_long_short_returns(
    monthly_quantile_returns: pd.DataFrame,
) -> pd.Series:
    """
    Calculate long-short portfolio returns (top quantile - bottom quantile).

    Parameters
    ----------
    monthly_quantile_returns : pd.DataFrame
        DataFrame with quantile returns from calculate_quantile_returns()

    Returns
    -------
    long_short : pd.Series
        Monthly long-short returns
    """
    top_col = monthly_quantile_returns.columns[-1]  # e.g., Q10
    bottom_col = monthly_quantile_returns.columns[0]  # e.g., Q1

    return monthly_quantile_returns[top_col] - monthly_quantile_returns[bottom_col]


def calculate_turnover(
    predictions_df: pd.DataFrame,
    date_col: str = 'date',
    id_col: str = 'permno',
    n_quantiles: int = 10,
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate monthly portfolio turnover for top and bottom quantiles.

    Turnover is the fraction of stocks that change each month.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with columns: date, id, return, prediction
    date_col : str
        Date column name
    id_col : str
        Entity identifier column name
    n_quantiles : int
        Number of quantiles

    Returns
    -------
    top_turnover : pd.Series
        Monthly turnover for top quantile (long leg)
    bottom_turnover : pd.Series
        Monthly turnover for bottom quantile (short leg)
    """
    df = predictions_df.copy()

    # Assign quantiles each month
    df['quantile'] = df.groupby(date_col)['prediction'].transform(
        lambda x: pd.qcut(x, n_quantiles, labels=False, duplicates='drop')
    )

    dates = sorted(df[date_col].unique())

    top_turnover = []
    bottom_turnover = []

    top_q = n_quantiles - 1  # Top quantile index
    bottom_q = 0  # Bottom quantile index

    prev_top_stocks = None
    prev_bottom_stocks = None

    for date in dates:
        month_data = df[df[date_col] == date]

        # Get stocks in top and bottom quantiles
        top_stocks = set(month_data[month_data['quantile'] == top_q][id_col])
        bottom_stocks = set(month_data[month_data['quantile'] == bottom_q][id_col])

        if prev_top_stocks is not None and len(top_stocks) > 0:
            # Turnover = fraction of new stocks not in previous portfolio
            # Plus fraction of old stocks no longer in portfolio
            # Divided by 2 (one-way turnover)
            top_new = len(top_stocks - prev_top_stocks)
            top_dropped = len(prev_top_stocks - top_stocks)
            top_turn = (top_new + top_dropped) / (2 * len(top_stocks))
            top_turnover.append(top_turn)
        else:
            top_turnover.append(np.nan)

        if prev_bottom_stocks is not None and len(bottom_stocks) > 0:
            bottom_new = len(bottom_stocks - prev_bottom_stocks)
            bottom_dropped = len(prev_bottom_stocks - bottom_stocks)
            bottom_turn = (bottom_new + bottom_dropped) / (2 * len(bottom_stocks))
            bottom_turnover.append(bottom_turn)
        else:
            bottom_turnover.append(np.nan)

        prev_top_stocks = top_stocks
        prev_bottom_stocks = bottom_stocks

    return (
        pd.Series(top_turnover, index=dates),
        pd.Series(bottom_turnover, index=dates)
    )


def apply_transaction_costs(
    monthly_returns: pd.DataFrame,
    predictions_df: pd.DataFrame,
    cost_bps: float = 20.0,
    date_col: str = 'date',
    id_col: str = 'permno',
    n_quantiles: int = 10,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Apply transaction costs to portfolio returns based on turnover.

    For long-short portfolios, costs are applied to both legs.
    Cost = turnover × cost_bps × 2 (round-trip)

    Parameters
    ----------
    monthly_returns : pd.DataFrame
        Monthly returns from calculate_quantile_returns() with Long_Short column
    predictions_df : pd.DataFrame
        Predictions DataFrame for calculating turnover
    cost_bps : float
        One-way transaction cost in basis points (default: 20 bps)
        Realistic values:
        - US large-cap: 5-10 bps
        - US all-cap: 10-20 bps
        - Emerging markets: 30-50 bps
    date_col : str
        Date column name
    id_col : str
        Entity identifier column name
    n_quantiles : int
        Number of quantiles used for portfolio formation (default: 10)

    Returns
    -------
    adjusted_returns : pd.DataFrame
        Returns adjusted for transaction costs
    cost_stats : dict
        Statistics about transaction costs
    """
    # Calculate turnover
    top_turnover, bottom_turnover = calculate_turnover(
        predictions_df, date_col, id_col, n_quantiles=n_quantiles
    )

    # Convert bps to decimal (20 bps = 0.0020)
    cost_decimal = cost_bps / 10000

    # Align turnover with returns
    adjusted_returns = monthly_returns.copy()

    top_col = monthly_returns.columns[-2]  # e.g., Q10
    bottom_col = monthly_returns.columns[0]  # e.g., Q1

    # Apply costs: cost = turnover × cost_decimal × 2 (round-trip)
    # Both legs incur transaction costs which reduce overall portfolio returns.
    #
    # For long-short portfolio: Net_Return = Long_Return - Short_Return - Total_Costs
    # Algebraically equivalent to:
    #   Net_Return = (Long - Cost_Long) - (Short + Cost_Short)
    #              = Long - Short - Cost_Long - Cost_Short
    #
    # We ADD cost to bottom_col so that when we subtract it in Long_Short calculation,
    # the cost is properly deducted from the overall portfolio return.
    top_costs = top_turnover.reindex(adjusted_returns.index).fillna(0) * cost_decimal * 2
    bottom_costs = bottom_turnover.reindex(adjusted_returns.index).fillna(0) * cost_decimal * 2

    adjusted_returns[top_col] = adjusted_returns[top_col] - top_costs
    adjusted_returns[bottom_col] = adjusted_returns[bottom_col] + bottom_costs  # See comment above

    # Recalculate long-short with costs
    # Result: Long_Short = (top - top_costs) - (bottom + bottom_costs) = gross_LS - total_costs
    adjusted_returns['Long_Short'] = adjusted_returns[top_col] - adjusted_returns[bottom_col]

    # Calculate cost statistics
    avg_top_turnover = top_turnover.mean()
    avg_bottom_turnover = bottom_turnover.mean()
    avg_monthly_cost = (avg_top_turnover + avg_bottom_turnover) * cost_decimal * 2

    cost_stats = {
        'cost_bps': cost_bps,
        'avg_top_turnover': avg_top_turnover,
        'avg_bottom_turnover': avg_bottom_turnover,
        'avg_turnover': (avg_top_turnover + avg_bottom_turnover) / 2,
        'avg_monthly_cost': avg_monthly_cost,
        'annual_cost_drag': avg_monthly_cost * 12,
    }

    return adjusted_returns, cost_stats


def calculate_portfolio_metrics(
    predictions_df: pd.DataFrame,
    date_col: str = 'date',
    n_quantiles: int = 10,
    transaction_cost_bps: float = 0.0,
    id_col: str = 'permno',
) -> Tuple[pd.DataFrame, Dict]:
    """
    Calculate comprehensive portfolio performance metrics.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with columns: date, return, prediction
    date_col : str
        Date column name
    n_quantiles : int
        Number of quantiles for portfolio formation
    transaction_cost_bps : float
        One-way transaction cost in basis points (default: 0 = no costs)
        Realistic values:
        - US large-cap: 5-10 bps
        - US all-cap: 10-20 bps
        - Emerging markets: 30-50 bps
    id_col : str
        Entity identifier column (needed for turnover calculation if costs > 0)

    Returns
    -------
    monthly_returns : pd.DataFrame
        Monthly returns for all quantiles plus Long_Short
    metrics : dict
        Performance metrics for top, bottom, and long-short portfolios
    """
    # Handle potential column naming variations
    if date_col not in predictions_df.columns:
        for col in ['date', 'eom']:
            if col in predictions_df.columns:
                date_col = col
                break

    # Find id column if not specified
    if id_col not in predictions_df.columns:
        for col in ['permno', 'gvkey', 'id']:
            if col in predictions_df.columns:
                id_col = col
                break

    # Calculate quantile returns
    monthly_returns = calculate_quantile_returns(predictions_df, date_col, n_quantiles)

    if monthly_returns.empty:
        return None, None

    # Add long-short portfolio
    monthly_returns['Long_Short'] = calculate_long_short_returns(monthly_returns)

    # Apply transaction costs if specified
    cost_stats = None
    if transaction_cost_bps > 0:
        monthly_returns, cost_stats = apply_transaction_costs(
            monthly_returns, predictions_df, transaction_cost_bps, date_col, id_col,
            n_quantiles=n_quantiles,
        )

    # Calculate metrics for key portfolios
    top_col = monthly_returns.columns[-2]  # Top quantile (before Long_Short)
    bottom_col = monthly_returns.columns[0]  # Bottom quantile

    metrics = {}
    for col in [top_col, bottom_col, 'Long_Short']:
        returns = monthly_returns[col]
        mean_ret = returns.mean()
        std_ret = returns.std()
        sharpe = (mean_ret / std_ret) * np.sqrt(12) if std_ret != 0 else 0

        metrics[col] = {
            'mean': mean_ret,
            'std': std_ret,
            'sharpe': sharpe,
            'min': returns.min(),
            'max': returns.max(),
            'skew': returns.skew(),
            'n_months': len(returns),
        }

    # Add cost statistics if applicable
    if cost_stats is not None:
        metrics['transaction_costs'] = cost_stats

    return monthly_returns, metrics


def calculate_decile_spread(
    predictions_df: pd.DataFrame,
    date_col: str = 'date',
) -> pd.DataFrame:
    """
    Calculate the average return spread between adjacent deciles.

    This helps assess the monotonicity of the predictive signal.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with predictions and returns
    date_col : str
        Date column name

    Returns
    -------
    spread_df : pd.DataFrame
        DataFrame with decile returns and spreads
    """
    monthly_returns = calculate_quantile_returns(predictions_df, date_col, n_quantiles=10)

    # Calculate average return per decile
    avg_returns = monthly_returns.mean()

    # Calculate spreads between adjacent deciles
    spreads = avg_returns.diff().dropna()

    spread_df = pd.DataFrame({
        'decile': avg_returns.index,
        'avg_return': avg_returns.values,
    })

    return spread_df


def create_comparison_table(
    results: Dict,
    models: Optional[list] = None,
) -> pd.DataFrame:
    """
    Create a comparison table for baseline vs augmented results.

    Parameters
    ----------
    results : dict
        Results dictionary from run_baseline_vs_augmented()
    models : list, optional
        List of model names to include (default: all)

    Returns
    -------
    comparison_df : pd.DataFrame
        Formatted comparison table
    """
    if models is None:
        models = list(results.keys())

    rows = []
    for model_name in models:
        if model_name not in results:
            continue

        result = results[model_name]
        baseline_metrics = result['baseline']['metrics']
        augmented_metrics = result['augmented']['metrics']

        rows.append({
            'Model': model_name,
            'Baseline_Sharpe': baseline_metrics['Long_Short']['sharpe'],
            'Augmented_Sharpe': augmented_metrics['Long_Short']['sharpe'],
            'Improvement': result['improvement'],
            'Improvement_%': result['improvement_pct'],
            'Baseline_Mean': baseline_metrics['Long_Short']['mean'],
            'Augmented_Mean': augmented_metrics['Long_Short']['mean'],
            'Baseline_Std': baseline_metrics['Long_Short']['std'],
            'Augmented_Std': augmented_metrics['Long_Short']['std'],
        })

    return pd.DataFrame(rows)


# ── Long-Only Portfolio Construction (corrected pipeline) ────────────────────
# Extracted from scripts/all_markets_engineered.py


def build_cost_lookup(raw_df, date_col, id_col, spread_col='bidaskhl_21d',
                      impact_multiplier=1.5, floor_bps=20, cap_bps=200):
    """
    Build per-stock per-month transaction cost lookup from bid-ask spreads.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Raw data with spread column.
    date_col : str
        Date column name.
    id_col : str
        Entity identifier column name.
    spread_col : str
        Bid-ask spread column name (default: 'bidaskhl_21d').
    impact_multiplier : float
        Market impact multiplier (default: 1.5).
    floor_bps : float
        Minimum cost in basis points (default: 20).
    cap_bps : float
        Maximum cost in basis points (default: 200).

    Returns
    -------
    lookup : pd.DataFrame
        DataFrame with [date_col, id_col, 'cost_bps'] columns.
    """
    lookup = raw_df[[date_col, id_col, spread_col]].copy()
    lookup['cost_bps'] = (lookup[spread_col] / 2) * 10000 * impact_multiplier
    monthly_median = lookup.groupby(date_col)['cost_bps'].transform('median')
    lookup['cost_bps'] = lookup['cost_bps'].fillna(monthly_median)
    lookup['cost_bps'] = lookup['cost_bps'].clip(lower=floor_bps, upper=cap_bps)
    return lookup[[date_col, id_col, 'cost_bps']]


def create_ensemble_predictions(pred_dfs, date_col, id_col, weights=None):
    """
    Create ensemble predictions from multiple model prediction DataFrames.

    Parameters
    ----------
    pred_dfs : list of pd.DataFrame
        Each with columns [date_col, id_col, 'return', 'prediction'].
    date_col : str
        Date column name.
    id_col : str
        Entity identifier column name.
    weights : list of float, optional
        Model weights (default: equal weighting).

    Returns
    -------
    merged : pd.DataFrame
        Ensemble predictions with columns [date_col, id_col, 'return', 'prediction'].
    """
    if weights is None:
        weights = [1.0 / len(pred_dfs)] * len(pred_dfs)
    merged = pred_dfs[0][[date_col, id_col, 'return']].copy()
    merged['prediction'] = pred_dfs[0]['prediction'] * weights[0]
    for i, pdf in enumerate(pred_dfs[1:], 1):
        sub = pdf[[date_col, id_col, 'prediction']].rename(columns={'prediction': f'_pred_{i}'})
        merged = merged.merge(sub, on=[date_col, id_col], how='inner')
        merged['prediction'] += merged[f'_pred_{i}'] * weights[i]
        merged.drop(columns=[f'_pred_{i}'], inplace=True)
    return merged


def construct_long_only_portfolio(predictions_df, universe_df, date_col, id_col,
                                  n_quantiles=5, buffer_pct=0.2,
                                  cost_lookup=None, target_col='ret_1_0'):
    """
    Construct a long-only top-quintile portfolio with turnover buffer.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Ensemble predictions with [date_col, id_col, 'return', 'prediction'].
    universe_df : pd.DataFrame
        Full universe (for benchmark calculation).
    date_col : str
        Date column name.
    id_col : str
        Entity identifier column name.
    n_quantiles : int
        Number of quantiles (default: 5, top quintile = long).
    buffer_pct : float
        Turnover buffer as fraction of quantile width (default: 0.2).
    cost_lookup : pd.DataFrame, optional
        Per-stock costs from build_cost_lookup().
    target_col : str
        Return column name in universe_df (default: 'ret_1_0').

    Returns
    -------
    result : dict
        Dictionary with portfolio return series and performance metrics.
    """
    pred = predictions_df.copy()
    dates = sorted(pred[date_col].unique())
    pred['pct_rank'] = pred.groupby(date_col)['prediction'].rank(pct=True)
    top_threshold = 1.0 - 1.0 / n_quantiles
    buffer = buffer_pct * (1.0 / n_quantiles)
    prev_long = set()
    monthly_long_ret, monthly_bench_ret = [], []
    monthly_turnover, monthly_cost_drag = [], []
    valid_dates = []

    if cost_lookup is not None:
        pred = pred.merge(cost_lookup, on=[date_col, id_col], how='left')
        global_med = cost_lookup['cost_bps'].median()
        pred['cost_bps'] = pred['cost_bps'].fillna(global_med)

    bench_returns = universe_df.groupby(date_col)[target_col].mean()

    for date in dates:
        month = pred[pred[date_col] == date].copy()
        if len(month) < n_quantiles:
            continue

        new_long = set()
        for _, row in month.iterrows():
            sid = row[id_col]
            rank = row['pct_rank']
            if sid in prev_long:
                if rank >= top_threshold - buffer:
                    new_long.add(sid)
            else:
                if rank >= top_threshold + buffer:
                    new_long.add(sid)

        if len(new_long) == 0:
            new_long = set(month.nlargest(max(1, len(month) // n_quantiles), 'pct_rank')[id_col])

        long_ret = month[month[id_col].isin(new_long)]['return'].mean()
        monthly_long_ret.append(long_ret)
        monthly_bench_ret.append(bench_returns.get(date, np.nan))

        if prev_long:
            entries = new_long - prev_long
            exits = prev_long - new_long
            turn = (len(entries) + len(exits)) / (2 * max(len(new_long), 1))
        else:
            entries = new_long
            exits = set()
            turn = np.nan
        monthly_turnover.append(turn)

        if cost_lookup is not None and prev_long:
            traded = entries | exits
            if traded:
                traded_costs = month[month[id_col].isin(traded)]['cost_bps'].mean()
                frac_traded = len(traded) / max(len(new_long), 1)
                drag = frac_traded * (traded_costs / 10000)
            else:
                drag = 0
        else:
            drag = 0
        monthly_cost_drag.append(drag)
        valid_dates.append(date)
        prev_long = new_long

    idx = pd.Index(valid_dates, name=date_col)
    long_s = pd.Series(monthly_long_ret, index=idx)
    bench_s = pd.Series(monthly_bench_ret, index=idx)
    turn_s = pd.Series(monthly_turnover, index=idx)
    cost_s = pd.Series(monthly_cost_drag, index=idx)
    long_net = long_s - cost_s
    excess_net = long_net - bench_s

    def sharpe(s):
        s = s.dropna()
        return (s.mean() / s.std()) * np.sqrt(12) if s.std() > 0 else 0

    return {
        'long_net': long_net,
        'benchmark': bench_s,
        'long_net_sharpe': sharpe(long_net),
        'bench_sharpe': sharpe(bench_s),
        'excess_net_sharpe': sharpe(excess_net),
        'ann_ret_long': long_net.mean() * 12,
        'ann_ret_bench': bench_s.mean() * 12,
        'avg_turnover': turn_s.mean(),
        'avg_cost_drag_bps': cost_s.mean() * 10000,
    }


def run_walk_forward_variant(df, features, target_col, date_col, id_col,
                             model_configs, splits, bond_feats_to_keep=None,
                             keep_pct=0.7, ensemble_models=None):
    """
    Walk-forward with within-fold feature selection and ensemble.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared dataset.
    features : list of str
        Feature columns.
    target_col : str
        Target column.
    date_col : str
        Date column.
    id_col : str
        Entity identifier column.
    model_configs : dict
        Model factory functions from get_model_configs().
    splits : list of tuples
        Walk-forward splits.
    bond_feats_to_keep : list of str, optional
        Bond features to always keep in selection.
    keep_pct : float
        Fraction of features to keep (default: 0.7).
    ensemble_models : list of str, optional
        Model names to ensemble (default: XGBoost, LightGBM, LinearRegression).

    Returns
    -------
    ens : pd.DataFrame
        Ensemble predictions with [date_col, id_col, 'return', 'prediction'].
    """
    from backtesting import impute_features
    from features import select_features_by_importance

    if ensemble_models is None:
        ensemble_models = ['XGBoost', 'LightGBM', 'LinearRegression']

    model_preds = {m: [] for m in ensemble_models}

    for fold_idx, (train_months, test_months) in enumerate(splits):
        train_data = df[df[date_col].isin(train_months)].copy()
        test_data = df[df[date_col].isin(test_months)].copy()
        if len(train_data) < 100 or len(test_data) < 10:
            continue

        X_train = train_data[features]
        y_train = train_data[target_col].values
        X_test = test_data[features]

        X_train_imp, X_test_imp = impute_features(X_train, X_test, strategy='median')

        # Phase 1: importances from XGB + LGBM
        fold_importances = []
        for mn in ['XGBoost', 'LightGBM']:
            model = model_configs[mn]()
            model.fit(X_train_imp.values if hasattr(X_train_imp, 'values') else X_train_imp, y_train)
            fold_importances.append(model.feature_importances_)
        avg_imp = np.mean(fold_importances, axis=0)

        sel_feats, _ = select_features_by_importance(
            avg_imp, features, keep_pct=keep_pct,
            always_keep=bond_feats_to_keep
        )

        if hasattr(X_train_imp, 'columns'):
            X_train_sel = X_train_imp[sel_feats].values
            X_test_sel = X_test_imp[sel_feats].values
        else:
            sel_idx = [features.index(f) for f in sel_feats]
            X_train_sel = X_train_imp[:, sel_idx]
            X_test_sel = X_test_imp[:, sel_idx]

        # Phase 2: train all models
        for mn in ensemble_models:
            model = model_configs[mn]()
            model.fit(X_train_sel, y_train)
            preds = model.predict(X_test_sel)
            pdf = test_data[[date_col, id_col, target_col]].copy()
            pdf = pdf.rename(columns={target_col: 'return'})
            pdf['prediction'] = preds
            model_preds[mn].append(pdf)

    combined = {m: pd.concat(model_preds[m], ignore_index=True) for m in ensemble_models}
    ens = create_ensemble_predictions([combined[m] for m in ensemble_models], date_col, id_col)
    return ens
