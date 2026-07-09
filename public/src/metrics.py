"""
Performance metrics calculation utilities.

This module provides functions for calculating various financial
performance metrics including Sharpe ratio, Information Coefficient,
CAPM alpha/beta, and drawdown statistics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from scipy import stats


def calculate_sharpe_ratio(
    returns: pd.Series,
    annualization_factor: float = 12,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Monthly returns
    annualization_factor : float
        Factor to annualize returns (default: 12 for monthly data)
    risk_free_rate : float
        Annual risk-free rate (default: 0)

    Returns
    -------
    sharpe : float
        Annualized Sharpe ratio
    """
    excess_returns = returns - risk_free_rate / annualization_factor
    mean_ret = excess_returns.mean()
    std_ret = excess_returns.std()

    if std_ret == 0:
        return 0.0

    return (mean_ret / std_ret) * np.sqrt(annualization_factor)


def calculate_information_coefficient(
    predictions: pd.Series,
    returns: pd.Series,
    method: str = 'spearman',
) -> float:
    """
    Calculate Information Coefficient (rank correlation between predictions and returns).

    Parameters
    ----------
    predictions : pd.Series
        Predicted returns
    returns : pd.Series
        Realized returns
    method : str
        Correlation method ('spearman' or 'pearson')

    Returns
    -------
    ic : float
        Information Coefficient
    """
    if method == 'spearman':
        ic, _ = stats.spearmanr(predictions, returns)
    else:
        ic = predictions.corr(returns)

    return ic


def calculate_monthly_ic(
    predictions_df: pd.DataFrame,
    date_col: str = 'date',
) -> pd.Series:
    """
    Calculate Information Coefficient for each month.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with date, prediction, and return columns
    date_col : str
        Date column name

    Returns
    -------
    monthly_ic : pd.Series
        IC for each month indexed by date
    """
    def calc_ic(group):
        if len(group) < 10:
            return np.nan
        ic, _ = stats.spearmanr(group['prediction'], group['return'])
        return ic

    monthly_ic = predictions_df.groupby(date_col).apply(calc_ic)
    return monthly_ic


def calculate_alpha_beta(
    portfolio_returns: pd.Series,
    market_returns: pd.Series,
) -> Dict[str, float]:
    """
    Calculate CAPM alpha and beta with t-statistics.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Portfolio returns
    market_returns : pd.Series
        Market benchmark returns

    Returns
    -------
    results : dict
        Dictionary with alpha, beta, t_alpha, t_beta, r_squared
    """
    # Align returns
    aligned = pd.concat([portfolio_returns, market_returns], axis=1).dropna()
    if len(aligned) < 3:
        return {'alpha': np.nan, 'beta': np.nan, 't_alpha': np.nan, 't_beta': np.nan}

    aligned.columns = ['portfolio', 'market']

    y = aligned['portfolio'].values
    X = aligned['market'].values
    n = len(X)

    # Calculate regression parameters
    X_mean = X.mean()
    y_mean = y.mean()

    beta = np.sum((X - X_mean) * (y - y_mean)) / np.sum((X - X_mean)**2)
    alpha = y_mean - beta * X_mean

    # Calculate standard errors and t-stats
    y_pred = alpha + beta * X
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    se = np.sqrt(ss_res / (n - 2))

    se_beta = se / np.sqrt(np.sum((X - X_mean)**2))
    se_alpha = se * np.sqrt(1/n + X_mean**2 / np.sum((X - X_mean)**2))

    t_beta = beta / se_beta if se_beta != 0 else np.nan
    t_alpha = alpha / se_alpha if se_alpha != 0 else np.nan

    # R-squared
    ss_tot = np.sum((y - y_mean)**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else 0

    return {
        'alpha': alpha,
        'beta': beta,
        't_alpha': t_alpha,
        't_beta': t_beta,
        'r_squared': r_squared,
        'se_alpha': se_alpha,
        'se_beta': se_beta,
    }


def calculate_max_drawdown(returns: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
    """
    Calculate maximum drawdown and the period it occurred.

    Parameters
    ----------
    returns : pd.Series
        Returns series (not cumulative)

    Returns
    -------
    max_dd : float
        Maximum drawdown (negative value)
    peak_date : pd.Timestamp
        Date of the peak before drawdown
    trough_date : pd.Timestamp
        Date of the trough (maximum drawdown point)
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1

    max_dd = drawdown.min()
    trough_idx = drawdown.idxmin()

    # Find peak date (last date when cumulative was at running max before trough)
    peak_idx = cumulative[:trough_idx].idxmax()

    return max_dd, peak_idx, trough_idx


def calculate_rolling_sharpe(
    returns: pd.Series,
    window: int = 12,
    annualization_factor: float = 12,
) -> pd.Series:
    """
    Calculate rolling Sharpe ratio.

    Parameters
    ----------
    returns : pd.Series
        Returns series
    window : int
        Rolling window size in periods
    annualization_factor : float
        Annualization factor

    Returns
    -------
    rolling_sharpe : pd.Series
        Rolling Sharpe ratio
    """
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()

    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(annualization_factor)

    return rolling_sharpe


def calculate_comprehensive_metrics(
    predictions_df: pd.DataFrame,
    date_col: str = 'date',
) -> pd.DataFrame:
    """
    Calculate comprehensive metrics for top, bottom, and long-short portfolios.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with date, prediction, return columns
    date_col : str
        Date column name

    Returns
    -------
    metrics_df : pd.DataFrame
        DataFrame with all metrics for each portfolio
    """
    from portfolio import calculate_quantile_returns

    monthly_returns = calculate_quantile_returns(predictions_df, date_col, n_quantiles=10)

    # Add long-short
    top_col = monthly_returns.columns[-1]
    bottom_col = monthly_returns.columns[0]
    monthly_returns['Long_Short'] = monthly_returns[top_col] - monthly_returns[bottom_col]

    # Calculate market returns
    market_returns = predictions_df.groupby(date_col)['return'].mean()

    results = {}
    for portfolio in [top_col, bottom_col, 'Long_Short']:
        returns = monthly_returns[portfolio]

        # Basic metrics
        sharpe = calculate_sharpe_ratio(returns)
        max_dd, _, _ = calculate_max_drawdown(returns)

        # CAPM metrics
        capm = calculate_alpha_beta(returns, market_returns)

        results[portfolio] = {
            'Mean (Monthly)': returns.mean(),
            'Std (Monthly)': returns.std(),
            'Sharpe (Annual)': sharpe,
            'Max Drawdown': max_dd,
            'Skewness': returns.skew(),
            'Kurtosis': returns.kurtosis(),
            'Alpha': capm['alpha'],
            'Beta': capm['beta'],
            't(Alpha)': capm['t_alpha'],
            'R-squared': capm['r_squared'],
            'N Months': len(returns),
        }

    return pd.DataFrame(results).T


def statistical_significance_test(
    baseline_sharpe: float,
    augmented_sharpe: float,
    baseline_returns: pd.Series,
    augmented_returns: pd.Series,
) -> Dict[str, float]:
    """
    Test statistical significance of Sharpe ratio difference.

    Uses the Jobson-Korkie test for comparing Sharpe ratios.

    Parameters
    ----------
    baseline_sharpe : float
        Baseline Sharpe ratio
    augmented_sharpe : float
        Augmented Sharpe ratio
    baseline_returns : pd.Series
        Baseline portfolio returns
    augmented_returns : pd.Series
        Augmented portfolio returns

    Returns
    -------
    results : dict
        Test statistic, p-value, and significance indicator
    """
    n = len(baseline_returns)

    # Correlation between strategies
    rho = baseline_returns.corr(augmented_returns)

    # Jobson-Korkie test statistic
    se = np.sqrt(
        (2 / n) * (1 - rho + 0.5 * (baseline_sharpe**2 + augmented_sharpe**2 -
        2 * baseline_sharpe * augmented_sharpe * rho))
    )

    if se == 0:
        return {'z_stat': np.nan, 'p_value': np.nan, 'significant': False}

    z_stat = (augmented_sharpe - baseline_sharpe) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return {
        'z_stat': z_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
    }
