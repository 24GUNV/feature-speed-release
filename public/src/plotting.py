"""
Publication-quality plotting utilities.

This module provides functions for creating figures that match
the style and quality of the paper's figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path


# Publication color palette
COLORS = {
    'baseline': '#3498db',      # Blue
    'augmented': '#27ae60',     # Green (positive transfer)
    'negative': '#e74c3c',      # Red (negative transfer)
    'neutral': '#95a5a6',       # Gray
    'accent': '#9b59b6',        # Purple
    'us': '#e74c3c',            # Red for US
    'thai': '#27ae60',          # Green for Thai/Emerging
}


def setup_publication_style():
    """
    Configure matplotlib for publication-quality figures.

    Call this at the start of your notebook to ensure consistent styling.
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.axisbelow': True,
    })


def plot_performance_comparison(
    comparison_df: pd.DataFrame,
    title: str = 'Baseline vs Augmented Performance',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    show_improvement: bool = True,
) -> plt.Figure:
    """
    Create a bar chart comparing baseline and augmented Sharpe ratios.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        DataFrame with columns: Model, Baseline_Sharpe, Augmented_Sharpe
    title : str
        Figure title
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    show_improvement : bool
        Whether to annotate improvement percentages

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(comparison_df))
    width = 0.35

    # Bars
    bars1 = ax.bar(
        x - width/2,
        comparison_df['Baseline_Sharpe'],
        width,
        label='Baseline (Equity Only)',
        color=COLORS['baseline'],
        edgecolor='black',
        linewidth=1.2,
        alpha=0.85,
    )
    bars2 = ax.bar(
        x + width/2,
        comparison_df['Augmented_Sharpe'],
        width,
        label='Bond-Augmented',
        color=COLORS['augmented'],
        edgecolor='black',
        linewidth=1.2,
        alpha=0.85,
    )

    # Add improvement annotations
    if show_improvement and 'Improvement_%' in comparison_df.columns:
        for i, (idx, row) in enumerate(comparison_df.iterrows()):
            improvement = row['Improvement_%']
            color = COLORS['augmented'] if improvement > 0 else COLORS['negative']
            y_pos = max(row['Baseline_Sharpe'], row['Augmented_Sharpe']) + 0.1
            ax.annotate(
                f'{improvement:+.1f}%',
                xy=(i, y_pos),
                ha='center',
                fontsize=10,
                fontweight='bold',
                color=color,
            )

    ax.set_xlabel('Model')
    ax.set_ylabel('Long-Short Sharpe Ratio')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Model'])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def plot_cumulative_returns(
    monthly_returns: pd.DataFrame,
    title: str = 'Cumulative Returns by Quantile',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """
    Plot cumulative returns for each quantile portfolio.

    Parameters
    ----------
    monthly_returns : pd.DataFrame
        Monthly returns with columns for each quantile
    title : str
        Figure title
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    cumulative = monthly_returns.cumsum()

    # Plot all quantiles
    for col in cumulative.columns:
        if col == 'Long_Short':
            ax.plot(cumulative.index, cumulative[col], label=col, linewidth=2.5, color='black')
        else:
            ax.plot(cumulative.index, cumulative[col], label=col, alpha=0.7)

    # Add Sharpe annotation for Long-Short
    if 'Long_Short' in monthly_returns.columns:
        ls_returns = monthly_returns['Long_Short']
        sharpe = (ls_returns.mean() / ls_returns.std()) * np.sqrt(12)
        ax.text(
            0.02, 0.98,
            f'Long-Short Sharpe: {sharpe:.2f}',
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        )

    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def plot_rolling_sharpe(
    monthly_returns: pd.DataFrame,
    window: int = 12,
    columns: Optional[List[str]] = None,
    title: str = 'Rolling 12-Month Sharpe Ratio',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Plot rolling Sharpe ratios.

    Parameters
    ----------
    monthly_returns : pd.DataFrame
        Monthly returns DataFrame
    window : int
        Rolling window size
    columns : list, optional
        Columns to plot (default: Q1, Q10, Long_Short)
    title : str
        Figure title
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if columns is None:
        # Default to key portfolios
        columns = [col for col in ['Q1', 'Q10', 'Long_Short'] if col in monthly_returns.columns]

    rolling_mean = monthly_returns[columns].rolling(window=window).mean()
    rolling_std = monthly_returns[columns].rolling(window=window).std()
    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(12)

    for col in columns:
        ax.plot(rolling_sharpe.index, rolling_sharpe[col], label=col, linewidth=1.5)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Date')
    ax.set_ylabel('Rolling Sharpe Ratio')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
    title: str = 'Feature Importance',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    highlight_bond: bool = True,
) -> plt.Figure:
    """
    Plot feature importance as horizontal bar chart.

    Parameters
    ----------
    importances : np.ndarray
        Feature importance values
    feature_names : list
        Feature names
    top_n : int
        Number of top features to show
    title : str
        Figure title
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size
    highlight_bond : bool
        Whether to highlight bond features in different color

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances,
    }).sort_values('importance', ascending=False).head(top_n)

    # Determine colors (highlight bond features)
    bond_prefixes = ['bond_', 'ig_', 'hy_', 'credit_', 'term_']
    colors = []
    for feat in importance_df['feature']:
        if highlight_bond and any(feat.lower().startswith(p) for p in bond_prefixes):
            colors.append(COLORS['augmented'])
        else:
            colors.append(COLORS['baseline'])

    # Plot
    bars = ax.barh(
        importance_df['feature'],
        importance_df['importance'],
        color=colors,
        edgecolor='black',
        linewidth=0.5,
    )

    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')

    # Add legend if highlighting bond features
    if highlight_bond:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS['baseline'], label='Equity Features'),
            Patch(facecolor=COLORS['augmented'], label='Bond Features'),
        ]
        ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def plot_asymmetric_transfer(
    us_results: Dict,
    thai_results: Dict,
    title: str = 'Asymmetric Transfer Learning Effect',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
) -> plt.Figure:
    """
    Create the key paper figure showing asymmetric transfer effects.

    Parameters
    ----------
    us_results : dict
        US market results from run_baseline_vs_augmented()
    thai_results : dict
        Thai market results from run_baseline_vs_augmented()
    title : str
        Figure title
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    models = list(us_results.keys())
    width = 0.35
    x = np.arange(len(models))

    # US Market (subplot a)
    ax = axes[0]
    us_baseline = [us_results[m]['baseline']['metrics']['Long_Short']['sharpe'] for m in models]
    us_augmented = [us_results[m]['augmented']['metrics']['Long_Short']['sharpe'] for m in models]

    ax.bar(x - width/2, us_baseline, width, label='Baseline', color=COLORS['baseline'],
           edgecolor='black', linewidth=1.2, alpha=0.85)
    ax.bar(x + width/2, us_augmented, width, label='Bond-Augmented', color=COLORS['negative'],
           edgecolor='black', linewidth=1.2, alpha=0.85)

    # Add improvement annotations
    for i, m in enumerate(models):
        improvement = us_results[m]['improvement_pct']
        y_pos = max(us_baseline[i], us_augmented[i]) + 0.15
        ax.annotate(f'{improvement:+.1f}%', xy=(i, y_pos), ha='center', fontsize=10,
                   fontweight='bold', color=COLORS['negative'])

    ax.set_title('(a) US Market: Negative Transfer', fontweight='bold')
    ax.set_xlabel('Model')
    ax.set_ylabel('Long-Short Sharpe Ratio')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.25, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.5)

    # Thai Market (subplot b)
    ax = axes[1]
    thai_baseline = [thai_results[m]['baseline']['metrics']['Long_Short']['sharpe'] for m in models]
    thai_augmented = [thai_results[m]['augmented']['metrics']['Long_Short']['sharpe'] for m in models]

    ax.bar(x - width/2, thai_baseline, width, label='Baseline', color=COLORS['baseline'],
           edgecolor='black', linewidth=1.2, alpha=0.85)
    ax.bar(x + width/2, thai_augmented, width, label='Bond-Augmented', color=COLORS['augmented'],
           edgecolor='black', linewidth=1.2, alpha=0.85)

    # Add improvement annotations
    for i, m in enumerate(models):
        improvement = thai_results[m]['improvement_pct']
        y_pos = max(thai_baseline[i], thai_augmented[i]) + 0.08
        ax.annotate(f'{improvement:+.1f}%', xy=(i, y_pos), ha='center', fontsize=10,
                   fontweight='bold', color=COLORS['augmented'])

    ax.set_title('(b) Thai Market: Positive Transfer', fontweight='bold')
    ax.set_xlabel('Model')
    ax.set_ylabel('Long-Short Sharpe Ratio')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(True, alpha=0.25, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.5)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        # Save both PNG and PDF
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        pdf_path = save_path.replace('.png', '.pdf')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Saved figure to {save_path} and {pdf_path}")

    return fig


def plot_monthly_ic(
    monthly_ic: pd.Series,
    title: str = 'Information Coefficient Over Time',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
) -> plt.Figure:
    """
    Plot Information Coefficient over time with average line.

    Parameters
    ----------
    monthly_ic : pd.Series
        Monthly IC values indexed by date
    title : str
        Figure title
    save_path : str, optional
        Path to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    fig : plt.Figure
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.bar(monthly_ic.index, monthly_ic.values, alpha=0.7, color=COLORS['baseline'])
    ax.axhline(y=monthly_ic.mean(), color=COLORS['negative'], linestyle='--',
               linewidth=2, label=f'Mean IC: {monthly_ic.mean():.4f}')
    ax.axhline(y=0, color='black', linewidth=0.5)

    ax.set_xlabel('Date')
    ax.set_ylabel('Information Coefficient')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add IC statistics
    stats_text = f'Mean: {monthly_ic.mean():.4f}\nStd: {monthly_ic.std():.4f}\nIR: {monthly_ic.mean()/monthly_ic.std():.2f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def save_figure(
    fig: plt.Figure,
    name: str,
    output_dir: str = 'plots',
    formats: List[str] = ['png', 'pdf'],
    dpi: int = 300,
):
    """
    Save figure in multiple formats.

    Parameters
    ----------
    fig : plt.Figure
        Matplotlib figure
    name : str
        Base filename (without extension)
    output_dir : str
        Output directory
    formats : list
        File formats to save
    dpi : int
        Resolution for raster formats
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        filepath = output_path / f"{name}.{fmt}"
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {filepath}")
