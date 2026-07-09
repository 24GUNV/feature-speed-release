"""
Cross-Market Hyperparameter Optimization Module.

This module provides Bayesian hyperparameter optimization for finding
parameters that generalize across multiple emerging markets.

The key innovation is optimizing for both magnitude AND consistency:
    objective = mean(improvements) + beta * (% markets improved)

This prevents cherry-picking parameters that work well on one market
but fail on others.

Example usage:
    from optimization import run_cross_market_optimization

    best_params, results_df = run_cross_market_optimization(
        market_data_cache=market_data,
        beta=0.3,
        n_trials=50,
    )
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from sklearn.impute import SimpleImputer
import lightgbm as lgb
from pathlib import Path
import json

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def calculate_turnover(
    predictions_df: pd.DataFrame,
    id_col: str = 'id',
    n_quantiles: int = 10,
) -> Tuple[float, float]:
    """
    Calculate average portfolio turnover for top and bottom quantiles.

    Returns
    -------
    avg_top_turnover, avg_bottom_turnover : tuple of floats
    """
    df = predictions_df.copy()

    def qcut_safe(x):
        if len(x) < n_quantiles:
            return np.nan
        try:
            return pd.qcut(x, n_quantiles, labels=False, duplicates='drop')
        except:
            return np.nan

    df['quantile'] = df.groupby('date')['prediction'].transform(qcut_safe)
    df = df.dropna(subset=['quantile'])

    dates = sorted(df['date'].unique())

    top_turnover = []
    bottom_turnover = []

    top_q = n_quantiles - 1
    bottom_q = 0

    prev_top_stocks = None
    prev_bottom_stocks = None

    for date in dates:
        month_data = df[df['date'] == date]
        top_stocks = set(month_data[month_data['quantile'] == top_q][id_col])
        bottom_stocks = set(month_data[month_data['quantile'] == bottom_q][id_col])

        if prev_top_stocks is not None and len(top_stocks) > 0:
            top_new = len(top_stocks - prev_top_stocks)
            top_dropped = len(prev_top_stocks - top_stocks)
            top_turn = (top_new + top_dropped) / (2 * max(len(top_stocks), 1))
            top_turnover.append(top_turn)

        if prev_bottom_stocks is not None and len(bottom_stocks) > 0:
            bottom_new = len(bottom_stocks - prev_bottom_stocks)
            bottom_dropped = len(prev_bottom_stocks - bottom_stocks)
            bottom_turn = (bottom_new + bottom_dropped) / (2 * max(len(bottom_stocks), 1))
            bottom_turnover.append(bottom_turn)

        prev_top_stocks = top_stocks
        prev_bottom_stocks = bottom_stocks

    avg_top = np.mean(top_turnover) if top_turnover else 0.5
    avg_bottom = np.mean(bottom_turnover) if bottom_turnover else 0.5

    return avg_top, avg_bottom


def calculate_portfolio_sharpe(
    predictions_df: pd.DataFrame,
    target_col: str = 'return',
    n_quantiles: int = 10,
    cost_bps: float = 0.0,
    id_col: str = 'id',
) -> float:
    """
    Calculate Sharpe ratio of long-short portfolio, optionally with transaction costs.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        DataFrame with 'date', 'prediction', and target columns
    target_col : str
        Name of the target/return column
    n_quantiles : int
        Number of quantiles for portfolio formation
    cost_bps : float
        One-way transaction cost in basis points (default: 0 = no costs)
        Recommended: 30 bps for emerging markets
    id_col : str
        Entity identifier column (needed for turnover if cost_bps > 0)

    Returns
    -------
    sharpe : float
        Annualized Sharpe ratio (net of transaction costs if specified)
    """
    df = predictions_df.copy()
    if target_col != 'return':
        df.rename(columns={target_col: 'return'}, inplace=True)

    def qcut_safe(x):
        if len(x) < n_quantiles:
            return np.nan
        try:
            return pd.qcut(x, n_quantiles, labels=False, duplicates='drop')
        except:
            return np.nan

    df['quantile'] = df.groupby('date')['prediction'].transform(qcut_safe)
    df = df.dropna(subset=['quantile'])

    if len(df) == 0:
        return np.nan

    monthly_returns = df.groupby(['date', 'quantile'])['return'].mean().unstack()

    if monthly_returns.shape[1] < n_quantiles:
        return np.nan

    # Long top quantile, short bottom quantile
    long_short = monthly_returns.iloc[:, -1] - monthly_returns.iloc[:, 0]

    # Apply transaction costs if specified
    if cost_bps > 0:
        avg_top_turnover, avg_bottom_turnover = calculate_turnover(
            predictions_df, id_col=id_col, n_quantiles=n_quantiles
        )
        cost_decimal = cost_bps / 10000
        # Monthly cost = turnover × cost × 2 (round-trip) for each leg
        monthly_cost = (avg_top_turnover + avg_bottom_turnover) * cost_decimal * 2
        long_short = long_short - monthly_cost

    if long_short.std() == 0:
        return np.nan

    # Annualized Sharpe
    sharpe = (long_short.mean() / long_short.std()) * np.sqrt(12)
    return sharpe


def evaluate_single_market(
    country: str,
    params: Dict,
    market_data_cache: Dict,
    cost_bps: float = 0.0,
    holdout_date: str = None,
    use_holdout_only: bool = False,
) -> float:
    """
    Evaluate a single market with given LightGBM parameters.

    Returns improvement (augmented - baseline) for this market.

    Parameters
    ----------
    country : str
        Country code
    params : dict
        LightGBM hyperparameters
    market_data_cache : dict
        Pre-loaded market data from load_all_markets()
    cost_bps : float
        One-way transaction cost in basis points (default: 0)
        Recommended: 30 bps for emerging markets
    holdout_date : str, optional
        Date string (e.g., '2020-01-01') to split optimization vs evaluation folds.
        If provided, only folds with test periods before this date are used for optimization.
    use_holdout_only : bool
        If True and holdout_date is set, evaluate ONLY on folds >= holdout_date (for final eval).
        If False and holdout_date is set, evaluate ONLY on folds < holdout_date (for optimization).

    Returns
    -------
    improvement : float
        Sharpe improvement (augmented - baseline), net of transaction costs
    """
    try:
        data = market_data_cache[country]
        df = data['df']
        equity_features = data['equity_features']
        all_features = data['all_features']
        id_col = data['id_col']
        target = data['target']
        splits = data['splits']

        # Filter splits based on holdout_date
        if holdout_date is not None:
            holdout_ts = pd.Timestamp(holdout_date)
            if use_holdout_only:
                # Evaluation: only use folds where test period starts >= holdout_date
                splits = [(tr, te) for tr, te in splits if pd.Timestamp(te[0]) >= holdout_ts]
            else:
                # Optimization: only use folds where test period starts < holdout_date
                splits = [(tr, te) for tr, te in splits if pd.Timestamp(te[0]) < holdout_ts]

        if len(splits) == 0:
            return np.nan

        # Run baseline (equity features only)
        baseline_preds = []
        for train_months, test_months in splits:
            train_data = df[df['date'].isin(train_months)]
            test_data = df[df['date'].isin(test_months)]

            X_train = train_data[equity_features]
            y_train = train_data[target]
            X_test = test_data[equity_features]

            imputer = SimpleImputer(strategy='median')
            X_train_imp = imputer.fit_transform(X_train)
            X_test_imp = imputer.transform(X_test)

            model = lgb.LGBMRegressor(**params)
            model.fit(X_train_imp, y_train)
            predictions = model.predict(X_test_imp)

            pred_df = test_data[['date', id_col, target]].copy()
            pred_df['prediction'] = predictions
            baseline_preds.append(pred_df)

        baseline_sharpe = calculate_portfolio_sharpe(
            pd.concat(baseline_preds), target, cost_bps=cost_bps, id_col=id_col
        )

        # Run augmented (equity + bond features)
        augmented_preds = []
        for train_months, test_months in splits:
            train_data = df[df['date'].isin(train_months)]
            test_data = df[df['date'].isin(test_months)]

            X_train = train_data[all_features]
            y_train = train_data[target]
            X_test = test_data[all_features]

            imputer = SimpleImputer(strategy='median')
            X_train_imp = imputer.fit_transform(X_train)
            X_test_imp = imputer.transform(X_test)

            model = lgb.LGBMRegressor(**params)
            model.fit(X_train_imp, y_train)
            predictions = model.predict(X_test_imp)

            pred_df = test_data[['date', id_col, target]].copy()
            pred_df['prediction'] = predictions
            augmented_preds.append(pred_df)

        augmented_sharpe = calculate_portfolio_sharpe(
            pd.concat(augmented_preds), target, cost_bps=cost_bps, id_col=id_col
        )

        if np.isnan(augmented_sharpe) or np.isnan(baseline_sharpe):
            return np.nan

        return augmented_sharpe - baseline_sharpe

    except Exception as e:
        return np.nan


def evaluate_all_markets(
    params: Dict,
    market_data_cache: Dict,
    beta: float = 0.3,
    cost_bps: float = 0.0,
    holdout_date: str = None,
    use_holdout_only: bool = False,
    verbose: bool = False,
) -> float:
    """
    Evaluate all markets and compute composite optimization objective.

    The objective balances average improvement with consistency:
        composite = mean(improvements) + beta * (% markets improved)

    Parameters
    ----------
    params : dict
        LightGBM hyperparameters
    market_data_cache : dict
        Pre-loaded market data
    beta : float
        Weight for % positive markets (default 0.3)
        Higher beta prioritizes consistency across markets
    cost_bps : float
        One-way transaction cost in basis points (default: 0)
        Recommended: 30 bps for emerging markets
    holdout_date : str, optional
        Date to split optimization vs evaluation periods
    use_holdout_only : bool
        If True, evaluate only on holdout period (for final results)
    verbose : bool
        Print detailed results per market

    Returns
    -------
    composite_score : float
        Optimization objective value
    """
    improvements = []

    for country in market_data_cache.keys():
        improvement = evaluate_single_market(
            country, params, market_data_cache, cost_bps=cost_bps,
            holdout_date=holdout_date, use_holdout_only=use_holdout_only
        )
        if not np.isnan(improvement):
            improvements.append(improvement)
            if verbose:
                print(f'  {country}: {improvement:+.3f}')

    if len(improvements) == 0:
        return np.nan

    mean_improvement = np.mean(improvements)
    pct_positive = np.mean([imp > 0 for imp in improvements])
    composite_score = mean_improvement + beta * pct_positive

    if verbose:
        print(f'\n  Mean improvement: {mean_improvement:.3f}')
        print(f'  % markets improved: {pct_positive*100:.1f}%')
        print(f'  Composite score: {composite_score:.3f}')

    return composite_score


def create_cross_market_objective(
    market_data_cache: Dict,
    beta: float = 0.3,
    cost_bps: float = 0.0,
    n_jobs: int = 4,
    holdout_date: str = None,
) -> Callable:
    """
    Create Optuna objective function for cross-market optimization.

    The objective maximizes: mean(improvements) + beta * (% markets improved)

    Parameters
    ----------
    market_data_cache : dict
        Pre-loaded market data
    beta : float
        Weight for consistency (% positive markets)
    cost_bps : float
        One-way transaction cost in basis points (default: 0)
        Recommended: 30 bps for emerging markets
    n_jobs : int
        Number of parallel jobs for LightGBM
    holdout_date : str, optional
        Date to split optimization vs evaluation. If set, optimization uses
        only test folds BEFORE this date. Final evaluation uses folds >= this date.

    Returns
    -------
    objective : callable
        Optuna-compatible objective function
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required. Install with: pip install optuna")

    def objective(trial):
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'n_jobs': n_jobs,
            'verbosity': -1
        }

        # Evaluate across all markets (with transaction costs if specified)
        # If holdout_date is set, only use pre-holdout folds for optimization
        composite_score = evaluate_all_markets(
            params, market_data_cache, beta=beta, cost_bps=cost_bps,
            holdout_date=holdout_date, use_holdout_only=False, verbose=False
        )

        # Return negative (Optuna minimizes by default)
        return -composite_score if not np.isnan(composite_score) else 1e6

    return objective


def run_cross_market_optimization(
    market_data_cache: Dict,
    beta: float = 0.3,
    cost_bps: float = 0.0,
    n_trials: int = 50,
    n_jobs_optuna: int = 1,
    n_jobs_lgbm: int = 4,
    storage_path: Optional[str] = None,
    study_name: str = 'cross_market_optimization',
    holdout_date: str = None,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Run Bayesian hyperparameter optimization across multiple markets.

    Parameters
    ----------
    market_data_cache : dict
        Pre-loaded market data from load_all_markets()
    beta : float
        Weight for % positive markets in objective (default 0.3)
    cost_bps : float
        One-way transaction cost in basis points (default: 0)
        Recommended: 30 bps for emerging markets
        NOTE: Including transaction costs in optimization finds parameters
        that are robust to trading costs, not just maximizing gross returns.
    n_trials : int
        Number of optimization trials
    n_jobs_optuna : int
        Parallel Optuna trials (default 1 for reproducibility)
    n_jobs_lgbm : int
        Parallel jobs per LightGBM model
    storage_path : str, optional
        Path for Optuna SQLite storage (enables checkpointing)
    study_name : str
        Name for the Optuna study
    holdout_date : str, optional
        Date string (e.g., '2020-01-01') for temporal holdout validation.
        If provided:
        - Optimization uses only test folds BEFORE this date
        - Final results are evaluated ONLY on folds >= this date (true out-of-sample)
        This prevents data leakage from hyperparameter selection.

    Returns
    -------
    best_params : dict
        Optimized LightGBM hyperparameters
    results_df : pd.DataFrame
        Detailed results for each market with best params (on holdout period if specified)
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required. Install with: pip install optuna")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print('=' * 70)
    print('CROSS-MARKET HYPERPARAMETER OPTIMIZATION')
    print('=' * 70)
    print(f'\nObjective: mean(improvements) + {beta} * (% markets improved)')
    print(f'Markets: {len(market_data_cache)}')
    print(f'Trials: {n_trials}')
    if cost_bps > 0:
        print(f'Transaction costs: {cost_bps} bps one-way (INCLUDED in optimization)')
    else:
        print('Transaction costs: None (gross returns)')

    if holdout_date:
        print(f'\n*** TEMPORAL HOLDOUT VALIDATION ***')
        print(f'Optimization period: test folds < {holdout_date}')
        print(f'Evaluation period: test folds >= {holdout_date} (TRUE OUT-OF-SAMPLE)')
    else:
        print('\nWARNING: No holdout_date specified. Results may have selection bias.')

    # Create objective (uses only pre-holdout data for optimization)
    objective_func = create_cross_market_objective(
        market_data_cache, beta=beta, cost_bps=cost_bps, n_jobs=n_jobs_lgbm,
        holdout_date=holdout_date
    )

    # Setup study
    storage_url = f'sqlite:///{storage_path}' if storage_path else None

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        direction='minimize',
        sampler=TPESampler(seed=42),
        load_if_exists=True
    )

    # Run optimization
    print('\nStarting optimization (this may take several hours)...\n')
    study.optimize(
        objective_func,
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=n_jobs_optuna,
    )

    # Extract best parameters
    best_params = study.best_params.copy()
    best_params.update({
        'random_state': 42,
        'n_jobs': n_jobs_lgbm,
        'verbosity': -1
    })

    print('\n' + '=' * 70)
    print('BEST PARAMETERS (from optimization period)')
    print('=' * 70)
    print(f'\nComposite score (optimization period): {-study.best_value:.3f}\n')
    for key, value in best_params.items():
        print(f'  {key}: {value}')

    # Evaluate all markets with best params ON HOLDOUT PERIOD ONLY
    print('\n' + '=' * 70)
    if holdout_date:
        print(f'OUT-OF-SAMPLE RESULTS (test folds >= {holdout_date})')
    else:
        print('DETAILED RESULTS WITH OPTIMAL PARAMETERS')
    print('=' * 70)

    results = []
    for country in market_data_cache.keys():
        data = market_data_cache[country]
        df = data['df']
        equity_features = data['equity_features']
        all_features = data['all_features']
        id_col = data['id_col']
        target = data['target']
        splits = data['splits']

        # Filter to holdout period for final evaluation
        if holdout_date:
            holdout_ts = pd.Timestamp(holdout_date)
            eval_splits = [(tr, te) for tr, te in splits if pd.Timestamp(te[0]) >= holdout_ts]
        else:
            eval_splits = splits

        if len(eval_splits) == 0:
            print(f'  {country}: No holdout data available, skipping')
            continue

        # Calculate baseline and augmented Sharpe
        baseline_preds = []
        augmented_preds = []

        for train_months, test_months in eval_splits:
            train_data = df[df['date'].isin(train_months)]
            test_data = df[df['date'].isin(test_months)]

            # Baseline
            X_train = train_data[equity_features]
            y_train = train_data[target]
            X_test = test_data[equity_features]

            imputer = SimpleImputer(strategy='median')
            X_train_imp = imputer.fit_transform(X_train)
            X_test_imp = imputer.transform(X_test)

            model = lgb.LGBMRegressor(**best_params)
            model.fit(X_train_imp, y_train)

            pred_df = test_data[['date', id_col, target]].copy()
            pred_df['prediction'] = model.predict(X_test_imp)
            baseline_preds.append(pred_df)

            # Augmented
            X_train = train_data[all_features]
            X_test = test_data[all_features]

            imputer = SimpleImputer(strategy='median')
            X_train_imp = imputer.fit_transform(X_train)
            X_test_imp = imputer.transform(X_test)

            model = lgb.LGBMRegressor(**best_params)
            model.fit(X_train_imp, y_train)

            pred_df = test_data[['date', id_col, target]].copy()
            pred_df['prediction'] = model.predict(X_test_imp)
            augmented_preds.append(pred_df)

        baseline_sharpe = calculate_portfolio_sharpe(
            pd.concat(baseline_preds), target, cost_bps=cost_bps, id_col=id_col
        )
        augmented_sharpe = calculate_portfolio_sharpe(
            pd.concat(augmented_preds), target, cost_bps=cost_bps, id_col=id_col
        )

        improvement = augmented_sharpe - baseline_sharpe
        pct_improvement = (improvement / abs(baseline_sharpe) * 100
                         if baseline_sharpe != 0 else np.nan)

        results.append({
            'Country': country,
            'Baseline_Sharpe': baseline_sharpe,
            'Augmented_Sharpe': augmented_sharpe,
            'Improvement': improvement,
            'Pct_Improvement': pct_improvement,
        })

        print(f'  {country}: {baseline_sharpe:.3f} → {augmented_sharpe:.3f} '
              f'({improvement:+.3f}, {pct_improvement:+.1f}%)')

    results_df = pd.DataFrame(results)

    # Summary statistics
    print('\n' + '=' * 70)
    if holdout_date:
        print(f'SUMMARY (Out-of-Sample: >= {holdout_date})')
    else:
        print('SUMMARY')
    print('=' * 70)
    n_positive = (results_df['Improvement'] > 0).sum()
    print(f'\nMarkets with positive transfer: {n_positive}/{len(results_df)}')
    print(f'Mean improvement: {results_df["Improvement"].mean():.3f}')
    print(f'Median improvement: {results_df["Improvement"].median():.3f}')

    return best_params, results_df


def save_optimization_results(
    best_params: Dict,
    results_df: pd.DataFrame,
    output_dir: str = 'results',
) -> None:
    """
    Save optimization results to files.

    Parameters
    ----------
    best_params : dict
        Best hyperparameters
    results_df : pd.DataFrame
        Per-market results
    output_dir : str
        Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save parameters
    params_path = output_path / 'best_params_cross_market.json'
    with open(params_path, 'w') as f:
        json.dump(best_params, f, indent=2)
    print(f'Saved parameters to: {params_path}')

    # Save results
    results_path = output_path / 'cross_market_optimized_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f'Saved results to: {results_path}')


# Pre-computed optimal parameters from cross-market optimization
# These were found by optimizing across 24 emerging markets
# NOTE: These parameters were optimized WITHOUT transaction costs (cost_bps=0).
# For parameters optimized with transaction costs, re-run optimization with cost_bps=30.
CROSS_MARKET_OPTIMAL_PARAMS = {
    'n_estimators': 434,
    'learning_rate': 0.0237,
    'max_depth': 8,
    'num_leaves': 55,
    'subsample': 0.859,
    'colsample_bytree': 0.743,
    'min_child_samples': 52,
    'reg_alpha': 0.164,
    'reg_lambda': 6.686,
    'random_state': 42,
    'verbosity': -1,
}
