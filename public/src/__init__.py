"""
Cross-Market Transfer Learning for Stock Prediction

This package provides utilities for replicating the experiments in:
"Cross-Market Transfer Learning for Emerging Market Stock Prediction:
Evidence from Bond Market Augmentation"

Modules:
    - data_loader: Load and preprocess equity and bond data
    - features: Corrected bond feature engineering (7 clean + 22 engineered)
    - pipeline: High-level load_dataset() convenience function
    - backtesting: Walk-forward validation framework
    - portfolio: Portfolio construction (long-short + long-only)
    - metrics: Performance metrics (Sharpe, IC, alpha/beta)
    - models: Model configurations (Linear, XGBoost, LightGBM, MLP, LSTM)
    - plotting: Publication-quality figure generation
"""

# Handle both package imports and direct imports via sys.path.append
try:
    # Package imports (when installed or imported as package)
    from .data_loader import (
        load_equity_data,
        load_bond_factors,
        merge_bond_features,
        lag_features,
    )

    from .features import (
        CLEAN_BOND_FEATURES,
        engineer_bond_features,
        cross_sectional_rank_normalize,
        time_series_normalize_bonds,
        filter_top_n_by_marketcap,
        select_features_by_importance,
    )

    from .pipeline import load_dataset

    from .backtesting import (
        create_walk_forward_splits,
        impute_features,
        run_walk_forward_experiment,
    )

    from .portfolio import (
        calculate_portfolio_metrics,
        calculate_quantile_returns,
        calculate_long_short_returns,
        build_cost_lookup,
        create_ensemble_predictions,
        construct_long_only_portfolio,
        run_walk_forward_variant,
    )

    from .metrics import (
        calculate_sharpe_ratio,
        calculate_information_coefficient,
        calculate_alpha_beta,
        calculate_max_drawdown,
    )

    from .models import (
        get_model_configs,
        create_mlp_model,
        create_lstm_model,
    )

    from .plotting import (
        setup_publication_style,
        plot_performance_comparison,
        plot_cumulative_returns,
        plot_feature_importance,
        plot_rolling_sharpe,
    )
except ImportError:
    # Direct imports (when using sys.path.append from notebooks)
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
        filter_top_n_by_marketcap,
        select_features_by_importance,
    )

    from pipeline import load_dataset

    from backtesting import (
        create_walk_forward_splits,
        impute_features,
        run_walk_forward_experiment,
    )

    from portfolio import (
        calculate_portfolio_metrics,
        calculate_quantile_returns,
        calculate_long_short_returns,
        build_cost_lookup,
        create_ensemble_predictions,
        construct_long_only_portfolio,
        run_walk_forward_variant,
    )

    from metrics import (
        calculate_sharpe_ratio,
        calculate_information_coefficient,
        calculate_alpha_beta,
        calculate_max_drawdown,
    )

    from models import (
        get_model_configs,
        create_mlp_model,
        create_lstm_model,
    )

    from plotting import (
        setup_publication_style,
        plot_performance_comparison,
        plot_cumulative_returns,
        plot_feature_importance,
        plot_rolling_sharpe,
    )

__version__ = "1.0.0"
__author__ = "Authors"
