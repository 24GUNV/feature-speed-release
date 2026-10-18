"""
Experiment DG: Model-free information-coefficient decay by feature speed.

This script tests whether SLOW features retain cross-sectional predictive
signal longer than FAST features. It deliberately performs no model training:
the only signal measure is monthly cross-sectional Spearman rank correlation
between a lagged feature observed at month T and a one-month return realized
at future horizon h.

Pipeline matches expAH_feature_speed_test.py for the clean feature panel:
load the 15 canonical EM markets, lag accounting and market features on the
full panel before filtering, shift market equity by one month to avoid
look-ahead, then filter to the canonical top-N universe.

Outputs:
  scripts/concern_experiments/expDG_ic_decay.csv
  scripts/concern_experiments/expDG_ic_decay_detail.csv
  paper_icaifv2/figures/ic_decay_by_speed.pdf
  paper_icaifv2/figures/ic_decay_by_speed.png
"""

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
sys.path.insert(0, 'public/src')
sys.path.insert(0, 'scripts/concern_experiments')

from data_loader import load_equity_data, lag_features  # noqa: E402
from market_config import MARKETS  # noqa: E402


np.random.seed(42)
warnings.filterwarnings('ignore')


# ==============================================================================
# FEATURE SPEED CLASSIFICATION (copied from expAH_feature_speed_test.py)
# ==============================================================================

SLOW_KEYWORDS = [
    'f_score', 'o_score', 'z_score',
    'qmj', 'qmj_prof', 'qmj_safety', 'qmj_growth',
    'roa', 'roe', 'gp_at', 'gp_atl1', 'op_at', 'op_atl1',
    'niq_at', 'niq_be', 'ni_be', 'ni_me',
    'cop_at', 'cop_atl1',
    'ope_be', 'ope_bel1',
    'ocf_at', 'ocf_me',
    'oaccruals_at', 'oaccruals_ni',
    'taccruals_at', 'taccruals_ni',
    'be_me', 'bev_mev', 'at_me', 'ebit_bev', 'ebitda_mev',
    'fcf_me', 'sale_me', 'debt_me', 'netdebt_me',
    'div12m_me', 'ebit_sale', 'cash_at', 'tangibility',
    'noa_at', 'opex_at', 'rd5_at', 'rd_sale', 'rd_me',
    'sale_bev',
    'mispricing_mgmt', 'mispricing_perf',
    'eq_dur', 'age',
    'at_be', 'at_turnover',
    'pi_nix', 'kz_index', 'aliq_at', 'aliq_mat',
    'eqpo_me', 'eqnpo_me', 'ival_me',
]

MEDIUM_KEYWORDS = [
    'at_gr1', 'ppeinv_gr1a', 'noa_gr1a', 'lnoa_gr1a',
    'capx_gr1', 'capx_gr2', 'capx_gr3', 'capex_abn',
    'chcsho_12m', 'eqnpo_12m', 'debt_gr3',
    'inv_gr1', 'inv_gr1a',
    'cowc_gr1a', 'coa_gr1a', 'col_gr1a', 'nncoa_gr1a',
    'ncoa_gr1a', 'ncol_gr1a', 'nfna_gr1a',
    'sti_gr1a', 'lti_gr1a', 'fnl_gr1a', 'be_gr1a',
    'netis_at', 'eqnetis_at', 'dbnetis_at',
    'tax_gr1a',
    'niq_at_chg1', 'niq_be_chg1', 'ocf_at_chg1',
    'ni_ar1', 'ni_inc8q',
    'dsale_dinv', 'dsale_drec', 'dgp_dsale', 'dsale_dsga',
    'sale_gr1', 'sale_gr3', 'emp_gr1', 'sale_emp_gr1',
    'saleq_gr1',
    'niq_su', 'saleq_su',
    'ocfq_saleq_std', 'earnings_variability',
    'seas_1_1an', 'seas_1_1na',
    'seas_2_5an', 'seas_2_5na',
    'seas_6_10an', 'seas_6_10na',
    'seas_11_15an', 'seas_11_15na',
    'seas_16_20an', 'seas_16_20na',
]

FAST_KEYWORDS = [
    'ret_12_1', 'ret_12_7', 'ret_6_1', 'ret_9_1', 'ret_3_1', 'ret_60_12',
    'resff3_12_1', 'resff3_6_1',
    'ivol_capm_21d', 'ivol_capm_252d', 'ivol_ff3_21d', 'ivol_hxz4_21d',
    'rvol_21d',
    'beta_60m', 'beta_dimson_21d', 'betabab_1260d', 'betadown_252d',
    'ni_ivol',
    'iskew_capm_21d', 'iskew_ff3_21d', 'iskew_hxz4_21d',
    'rskew_21d', 'coskew_21d', 'corr_1260d',
    'rmax1_21d', 'rmax5_21d', 'rmax5_rvol_21d',
    'ami_126d',
    'bidaskhl_21d',
    'dolvol_126d', 'dolvol_var_126d',
    'turnover_126d', 'turnover_var_126d',
    'zero_trades_126d', 'zero_trades_21d', 'zero_trades_252d',
    'prc', 'prc_highprc_252d',
]

EXCLUDE_COLS = ['Unnamed: 0', 'id', 'eom', 'excntry', 'gvkey', 'permno',
                'size_grp', 'me', 'market_equity', 'ret_1_0', 'date']


def classify_feature_speed(feature_name):
    """Classify a single feature into SLOW/MEDIUM/FAST."""
    f = feature_name.lower()
    for kw in FAST_KEYWORDS:
        if f == kw or f == kw.lower():
            return 'FAST'
    for kw in MEDIUM_KEYWORDS:
        if f == kw or f == kw.lower():
            return 'MEDIUM'
    for kw in SLOW_KEYWORDS:
        if f == kw or f == kw.lower():
            return 'SLOW'
    if any(x in f for x in ['ret_', 'vol', 'beta', 'ivol', 'rvol', 'skew', 'ami',
                            'bidask', 'dolvol', 'turnover', 'zero_trades',
                            'prc', 'rmax', 'corr_', 'coskew']):
        return 'FAST'
    if any(x in f for x in ['_gr1', '_gr2', '_gr3', '_chg', 'seas_', '_su',
                            'variability', 'netis', 'eqnetis', 'dbnetis']):
        return 'MEDIUM'
    return 'SLOW'


ACCOUNTING_KEYWORDS = [
    'f_score', 'o_score', 'z_score',
    'qmj', 'qmj_prof', 'qmj_safety', 'qmj_growth',
    'roa', 'roe', 'gp_at', 'gp_atl1', 'op_at', 'op_atl1',
    'niq_at', 'niq_be', 'niq_at_chg1', 'niq_be_chg1',
    'ni_be', 'ni_me', 'ni_ar1', 'ni_inc8q',
    'cop_at', 'cop_atl1',
    'ope_be', 'ope_bel1',
    'ocf_at', 'ocf_at_chg1', 'ocf_me',
    'oaccruals_at', 'oaccruals_ni',
    'taccruals_at', 'taccruals_ni',
    'be_me', 'bev_mev', 'at_me', 'ebit_bev', 'ebitda_mev',
    'fcf_me', 'sale_me', 'debt_me', 'netdebt_me',
    'div12m_me', 'ebit_sale', 'cash_at', 'tangibility',
    'noa_at', 'opex_at', 'rd5_at', 'rd_sale', 'rd_me',
    'sale_bev', 'sale_emp_gr1',
    'mispricing_mgmt', 'mispricing_perf',
    'eq_dur', 'age',
    'at_gr1', 'ppeinv_gr1a', 'noa_gr1a', 'lnoa_gr1a',
    'capx_gr1', 'capx_gr2', 'capx_gr3', 'capex_abn',
    'chcsho_12m', 'eqnpo_12m', 'debt_gr3',
    'inv_gr1', 'inv_gr1a',
    'cowc_gr1a', 'coa_gr1a', 'col_gr1a', 'nncoa_gr1a',
    'ncoa_gr1a', 'ncol_gr1a', 'nfna_gr1a',
    'sti_gr1a', 'lti_gr1a', 'fnl_gr1a', 'be_gr1a',
    'netis_at', 'eqnetis_at', 'dbnetis_at',
    'at_be', 'at_turnover',
    'pi_nix', 'kz_index', 'aliq_at', 'aliq_mat',
    'tax_gr1a', 'niq_su', 'saleq_su', 'saleq_gr1',
    'dsale_dinv', 'dsale_drec', 'dgp_dsale', 'dsale_dsga',
    'ocfq_saleq_std', 'earnings_variability',
    'sale_gr1', 'sale_gr3', 'emp_gr1',
    'eqpo_me', 'eqnpo_me', 'ival_me',
]


# ==============================================================================
# CONFIGURATION
# ==============================================================================

HORIZONS = [1, 2, 3, 6, 9, 12]
MIN_PAIRS = 30
SPEED_ORDER = ['SLOW', 'MEDIUM', 'FAST']

DETAIL_OUTPUT = 'scripts/concern_experiments/expDG_ic_decay_detail.csv'
AGG_OUTPUT = 'scripts/concern_experiments/expDG_ic_decay.csv'
FIG_OUTPUT = 'paper_icaifv2/figures/ic_decay_by_speed.pdf'


# ==============================================================================
# HELPERS
# ==============================================================================

def classify_features(all_features):
    """Split features into accounting and market groups exactly as expAH does."""
    accounting = []
    market = []
    for f in all_features:
        f_lower = f.lower()
        if any(kw == f_lower or kw in f_lower for kw in ACCOUNTING_KEYWORDS):
            accounting.append(f)
        else:
            market.append(f)
    return accounting, market


def prepare_market_panel(market, cfg):
    """
    Load and prepare a market panel using the canonical expAH clean pipeline.

    The important ordering is:
      1. identify accounting and market features,
      2. lag both groups on the full panel,
      3. shift market equity by one month,
      4. filter to the top-N market-cap universe.
    """
    df, raw_features, date_col, id_col = load_equity_data(cfg['file'], cfg['type'])
    target = cfg['target']
    if target not in df.columns:
        raise KeyError(f"{market}: target {target} not found")

    all_features = [f for f in raw_features if f not in EXCLUDE_COLS and f != target]
    accounting_feats, market_feats = classify_features(all_features)
    accounting_feats = [f for f in accounting_feats if f in df.columns]
    market_feats = [f for f in market_feats if f in df.columns]

    # CRITICAL: lag features before filtering the market-cap universe.
    if accounting_feats:
        df = lag_features(df, accounting_feats, id_col, date_col, lag=1)
    if market_feats:
        df = lag_features(df, market_feats, id_col, date_col, lag=1)

    # Look-ahead fix: me(T) embeds ret_1_0(T), so use me(T-1) for top-N ranking.
    if 'me' in df.columns:
        df = df.sort_values([id_col, date_col])
        df['me'] = df.groupby(id_col)['me'].shift(1)
        df = df.dropna(subset=['me'])

    topn = cfg['topn']
    if 'me' in df.columns:
        df['_me_rank'] = df.groupby(date_col)['me'].rank(ascending=False, method='first')
        df = df[df['_me_rank'] <= topn].drop(columns=['_me_rank'])

    features = [f for f in accounting_feats + market_feats
                if f in df.columns and f not in EXCLUDE_COLS and f != target]
    numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
    dropped = sorted(set(features) - set(numeric_features))
    if dropped:
        print(f"  Dropped {len(dropped)} nonnumeric features: {dropped[:5]}")

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([id_col, date_col]).reset_index(drop=True)

    for h in HORIZONS:
        ret_col = f'ret_fwd_{h}'
        if h == 1:
            df[ret_col] = df[target]
        else:
            df[ret_col] = df.groupby(id_col)[target].shift(-(h - 1))

    return df, numeric_features, date_col, id_col, target


def _pearson_by_feature(block, features):
    """Compute Pearson correlations against _ret_rank for all feature columns."""
    x = block[features].to_numpy(dtype=float, copy=False)
    y = block['_ret_rank'].to_numpy(dtype=float, copy=False)
    y_ok = np.isfinite(y)
    mask = np.isfinite(x) & y_ok[:, None]
    n = mask.sum(axis=0).astype(float)

    corr = np.full(len(features), np.nan)
    valid = n >= MIN_PAIRS
    if not valid.any():
        return corr, n

    x0 = np.where(mask, x, 0.0)
    y0 = np.where(mask, y[:, None], 0.0)
    y2 = y * y

    sum_x = x0.sum(axis=0)
    sum_y = y0.sum(axis=0)
    sum_x2 = np.where(mask, x * x, 0.0).sum(axis=0)
    sum_y2 = np.where(mask, y2[:, None], 0.0).sum(axis=0)
    sum_xy = np.where(mask, x * y[:, None], 0.0).sum(axis=0)

    numerator = sum_xy - (sum_x * sum_y / n)
    ss_x = sum_x2 - (sum_x * sum_x / n)
    ss_y = sum_y2 - (sum_y * sum_y / n)
    denom = np.sqrt(ss_x * ss_y)
    good = valid & np.isfinite(denom) & (denom > 0)
    corr[good] = numerator[good] / denom[good]
    return corr, n


def mean_monthly_rank_ic(df, features, date_col, ret_col):
    """
    Average monthly cross-sectional Spearman IC for each feature.

    For each horizon, the feature panel and future one-month return are ranked
    within month once, then Pearson correlations on ranks are computed
    vectorized across features.
    """
    work = df[[date_col, ret_col] + features].copy()
    work = work[work[ret_col].notna()]
    feature_ranks = work.groupby(date_col)[features].rank(method='average')
    ret_rank = work.groupby(date_col)[ret_col].rank(method='average')
    ranked = feature_ranks
    ranked[date_col] = work[date_col].values
    ranked['_ret_rank'] = ret_rank.values

    monthly_corrs = []
    monthly_pairs = []
    for _, block in ranked.groupby(date_col, sort=True):
        corr, n = _pearson_by_feature(block, features)
        monthly_corrs.append(corr)
        monthly_pairs.append(n)

    if not monthly_corrs:
        return pd.DataFrame({
            'feature': features,
            'raw_ic': np.nan,
            'n_months': 0,
            'avg_pairs': np.nan,
        })

    corr_arr = np.vstack(monthly_corrs)
    pairs_arr = np.vstack(monthly_pairs)
    valid_corr = np.isfinite(corr_arr)
    n_months = valid_corr.sum(axis=0)
    ic_sum = np.where(valid_corr, corr_arr, 0.0).sum(axis=0)
    pair_sum = np.where(valid_corr, pairs_arr, 0.0).sum(axis=0)

    raw_ic = np.divide(
        ic_sum, n_months,
        out=np.full(len(features), np.nan),
        where=n_months > 0,
    )
    avg_pairs = np.divide(
        pair_sum, n_months,
        out=np.full(len(features), np.nan),
        where=n_months > 0,
    )

    return pd.DataFrame({
        'feature': features,
        'raw_ic': raw_ic,
        'n_months': n_months.astype(int),
        'avg_pairs': avg_pairs,
    })


def add_sign_alignment(detail):
    """
    Align each market-feature IC curve by the sign of its mean IC across
    all horizons.

    Aligning by the horizon-1 IC alone inflates IC(1) mechanically (the
    alignment conditions on the same noisy estimate being positive), which
    overstates decay for noisier feature classes. The all-horizon mean sign
    spreads alignment noise evenly across horizons.
    """
    detail = detail.copy()
    mean_ic = detail.groupby(['market', 'feature'])['raw_ic'].transform('mean')
    h1 = (
        detail[detail['horizon'] == 1]
        .set_index(['market', 'feature'])['raw_ic']
    )
    detail['h1_ic'] = detail.set_index(['market', 'feature']).index.map(h1)
    detail['align_sign'] = np.sign(mean_ic).replace(0, 1).fillna(1.0)
    detail['aligned_ic'] = detail['raw_ic'] * detail['align_sign']
    return detail


def estimate_half_life(class_df):
    """
    Estimate the first horizon where normalized retention reaches 0.5.

    Uses linear interpolation between observed horizons. Returns NaN if the
    plotted horizon range never crosses 0.5.
    """
    curve = class_df.sort_values('horizon')[['horizon', 'retention']].dropna()
    if curve.empty:
        return np.nan
    horizons = curve['horizon'].to_numpy(dtype=float)
    retention = curve['retention'].to_numpy(dtype=float)

    for i in range(1, len(horizons)):
        r0, r1 = retention[i - 1], retention[i]
        h0, h1 = horizons[i - 1], horizons[i]
        if r1 <= 0.5 <= r0:
            if r0 == r1:
                return h1
            return h0 + (0.5 - r0) * (h1 - h0) / (r1 - r0)
    return np.nan


def aggregate_detail(detail):
    """Aggregate sign-aligned IC curves by feature speed class and horizon."""
    agg = (
        detail
        .groupby(['speed_class', 'horizon'], as_index=False)
        .agg(
            mean_aligned_ic=('aligned_ic', 'mean'),
            mean_raw_ic=('raw_ic', 'mean'),
            n_market_features=('aligned_ic', 'count'),
            mean_months=('n_months', 'mean'),
            mean_pairs=('avg_pairs', 'mean'),
        )
    )

    h1_mean = (
        agg[agg['horizon'] == 1]
        .set_index('speed_class')['mean_aligned_ic']
    )
    agg['ic_h1'] = agg['speed_class'].map(h1_mean).astype(float)
    agg['retention'] = agg['mean_aligned_ic'] / agg['ic_h1']

    half_lives = {
        speed: estimate_half_life(agg[agg['speed_class'] == speed])
        for speed in SPEED_ORDER
    }
    agg['half_life_months'] = agg['speed_class'].map(half_lives).astype(float)
    agg['speed_class'] = pd.Categorical(
        agg['speed_class'], categories=SPEED_ORDER, ordered=True
    )
    return agg.sort_values(['speed_class', 'horizon']).reset_index(drop=True)


def plot_ic_decay(agg):
    """Create the publication figure for mean IC decay and retention."""
    os.makedirs(os.path.dirname(FIG_OUTPUT), exist_ok=True)

    # Same palette and markers as the cost-speed bound figure
    # (Ridge_Slow / Ridge_SlowMed / Ridge_Fast).
    colors = {
        'SLOW': '#2166ac',
        'MEDIUM': '#67a9cf',
        'FAST': '#d6604d',
    }
    markers = {'SLOW': 'o', 'MEDIUM': 's', 'FAST': 'v'}

    plt.rcParams.update({
        'font.size': 8.5,
        'axes.labelsize': 8.5,
        'legend.fontsize': 7,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
    })

    fig, axes = plt.subplots(2, 1, figsize=(3.35, 4.2), sharex=True)

    for speed in SPEED_ORDER:
        data = agg[agg['speed_class'] == speed]
        axes[0].plot(
            data['horizon'], data['mean_aligned_ic'],
            marker=markers[speed], color=colors[speed], linewidth=1.4,
            markersize=4, markeredgecolor='white', markeredgewidth=0.3,
            label=speed,
        )
        axes[1].plot(
            data['horizon'], data['retention'],
            marker=markers[speed], color=colors[speed], linewidth=1.4,
            markersize=4, markeredgecolor='white', markeredgewidth=0.3,
        )

    axes[0].axhline(0, color='black', alpha=0.55, linewidth=0.6)
    axes[0].set_ylabel('Mean aligned rank IC')
    axes[1].legend(
        *axes[0].get_legend_handles_labels(),
        ncol=3, loc='lower left', framealpha=0.9, edgecolor='gray',
    )

    axes[1].axhline(0.5, color='black', alpha=0.55, linewidth=0.7, linestyle='--')
    axes[1].set_xlabel('Horizon (months)')
    axes[1].set_ylabel('Retention vs h=1')
    axes[1].set_xticks(HORIZONS)
    axes[1].set_ylim(bottom=min(0, agg['retention'].min() * 1.05))

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(FIG_OUTPUT, bbox_inches='tight', dpi=300)
    fig.savefig(FIG_OUTPUT.replace('.pdf', '.png'), bbox_inches='tight', dpi=150)
    plt.close(fig)


def print_summary(agg, detail, elapsed):
    """Print compact summary tables to stdout."""
    print("\n" + "=" * 80)
    print("Experiment DG: Model-free IC decay by feature speed")
    print("=" * 80)

    counts = (
        detail[['market', 'feature', 'speed_class']]
        .drop_duplicates()
        .groupby('speed_class')['feature']
        .nunique()
        .reindex(SPEED_ORDER)
    )
    print("\nUnique feature counts by speed class:")
    for speed, count in counts.items():
        print(f"  {speed:<6}: {int(count):>3}")

    mean_ic = agg.pivot(index='horizon', columns='speed_class', values='mean_aligned_ic')
    retention = agg.pivot(index='horizon', columns='speed_class', values='retention')
    print("\nMean sign-aligned rank IC:")
    print(mean_ic[SPEED_ORDER].round(5).to_string())
    print("\nNormalized retention IC(h)/IC(1):")
    print(retention[SPEED_ORDER].round(3).to_string())

    print("\nHalf-life estimates:")
    for speed in SPEED_ORDER:
        hl = agg.loc[agg['speed_class'] == speed, 'half_life_months'].dropna()
        if len(hl):
            print(f"  {speed:<6}: {hl.iloc[0]:.2f} months")
        else:
            print(f"  {speed:<6}: > {max(HORIZONS)} months")

    # Cross-market test: is SLOW retention at the longest horizon higher
    # than FAST/MEDIUM? Per-market retention = mean aligned IC at max
    # horizon / mean aligned IC at horizon 1, one observation per market.
    from scipy import stats as _stats
    h_max = max(HORIZONS)
    per_mkt = (
        detail.groupby(['market', 'speed_class', 'horizon'], observed=True)['aligned_ic']
        .mean()
        .reset_index()
        .pivot_table(index=['market', 'speed_class'], columns='horizon',
                     values='aligned_ic', observed=True)
    )
    ret_last = (per_mkt[h_max] / per_mkt[1]).unstack(1)
    print(f"\nPer-market retention at h={h_max} (mean across markets):")
    for speed in SPEED_ORDER:
        if speed in ret_last:
            print(f"  {speed:<6}: {ret_last[speed].mean():.3f}")
    for other in ['FAST', 'MEDIUM']:
        if 'SLOW' in ret_last and other in ret_last:
            _, p = _stats.wilcoxon(ret_last['SLOW'], ret_last[other])
            print(f"  Wilcoxon SLOW > {other} retention@{h_max}: p = {p:.4f}")

    print(f"\nWrote aggregate: {AGG_OUTPUT}")
    print(f"Wrote detail:    {DETAIL_OUTPUT}")
    print(f"Wrote figure:    {FIG_OUTPUT} and .png")
    print(f"Elapsed: {elapsed:.1f}s")


def main():
    t0 = time.time()
    all_rows = []

    for market, cfg in MARKETS.items():
        print("\n" + "#" * 80)
        print(f"# MARKET: {market}")
        print("#" * 80)

        try:
            df, features, date_col, id_col, target = prepare_market_panel(market, cfg)
        except Exception as exc:
            print(f"  ERROR preparing {market}: {exc}")
            continue

        speed_counts = pd.Series(
            [classify_feature_speed(f) for f in features]
        ).value_counts().reindex(SPEED_ORDER, fill_value=0)
        print(
            f"  Features: {len(features)} total = "
            f"{speed_counts['SLOW']} SLOW + "
            f"{speed_counts['MEDIUM']} MEDIUM + "
            f"{speed_counts['FAST']} FAST"
        )
        print(f"  Panel after top-N filter: {len(df):,} rows")

        for h in HORIZONS:
            ret_col = f'ret_fwd_{h}'
            ic_df = mean_monthly_rank_ic(df, features, date_col, ret_col)
            ic_df['market'] = market
            ic_df['horizon'] = h
            ic_df['speed_class'] = ic_df['feature'].map(classify_feature_speed)
            ic_df['topn'] = cfg['topn']
            ic_df['date_col'] = date_col
            ic_df['id_col'] = id_col
            ic_df['target'] = target
            all_rows.append(ic_df)

            valid_features = int(ic_df['raw_ic'].notna().sum())
            print(f"  h={h:>2}: {valid_features:>3}/{len(features)} features with IC")

    if not all_rows:
        raise RuntimeError("No market IC results were produced.")

    detail = pd.concat(all_rows, ignore_index=True)
    detail = add_sign_alignment(detail)
    detail = detail[[
        'market', 'feature', 'speed_class', 'horizon',
        'raw_ic', 'h1_ic', 'align_sign', 'aligned_ic',
        'n_months', 'avg_pairs', 'topn', 'date_col', 'id_col', 'target',
    ]]
    detail['speed_class'] = pd.Categorical(
        detail['speed_class'], categories=SPEED_ORDER, ordered=True
    )
    detail = detail.sort_values(['market', 'feature', 'horizon']).reset_index(drop=True)

    os.makedirs(os.path.dirname(DETAIL_OUTPUT), exist_ok=True)
    detail.to_csv(DETAIL_OUTPUT, index=False)

    agg = aggregate_detail(detail)
    agg.to_csv(AGG_OUTPUT, index=False)
    plot_ic_decay(agg)
    print_summary(agg, detail, time.time() - t0)


if __name__ == '__main__':
    if '--plot-only' in sys.argv:
        # Redraw the figure from the saved aggregate without recomputing ICs.
        plot_ic_decay(pd.read_csv(AGG_OUTPUT))
    else:
        main()
