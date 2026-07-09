"""
expCL: Feature Speed Decomposition on Rank Targets
====================================================

The paper's core contribution is the FEATURE SPEED mechanism.
Now that rank targets are the new ML baseline (SR 0.42 vs MomSort 0.36),
we need to show that feature speed STILL matters with rank targets.

This experiment replicates expAH (feature speed test) with rank targets:

1. RankTarget_All: all features, rank target
2. RankTarget_Slow: slow features only (rank AC > 0.75)
3. RankTarget_Fast: fast features only (rank AC < 0.50)
4. RankTarget_SlowMed: slow + medium features
5. Each with and without EMA (alpha=0.20)
6. MomSort (control)
7. Ridge_All (raw return baseline, for comparison)

KEY QUESTION: Does feature speed still drive excess turnover
when the target is cross-sectional ranks instead of raw returns?

If YES → feature speed is truly structural (stronger paper)
If NO → feature speed was an artifact of raw returns (weaker paper)

15 markets, OOS 2021-2023, 5 cost levels.
"""

import sys, os, time, warnings
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, 'public/src')
sys.path.insert(0, 'scripts/concern_experiments')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from market_config import MARKETS, COST_MULTS, BUFFER_PCT, MIN_TRAIN_MONTHS
from data_loader import load_equity_data, lag_features

np.random.seed(42)

RIDGE_ALPHA = 100.0
EMA_ALPHA = 0.20

ACCOUNTING_KEYWORDS = [
    'f_score', 'o_score', 'z_score', 'qmj', 'roa', 'roe',
    'niq_at', 'niq_be', 'ni_be', 'ni_me', 'be_me', 'fcf_me',
    'sale_me', 'at_me', 'ope_be', 'ope_at', 'gp_at', 'cop_at',
    'ebit_sale', 'sale_gr1', 'at_turnover', 'age', 'aliq_at',
    'at_be', 'bev_mev', 'cash_at', 'cowc_gr1a',
    'dbnetis_at', 'debt_at', 'debt_me', 'dgp_dsale', 'div12m_me',
    'ebitda_mev', 'emp_gr1', 'eq_dur', 'eqnpo_12m', 'eqnpo_me',
    'eqpo_me', 'fnl_gr1a',
    'inv_gr1', 'inv_gr1a', 'lnoa_gr1a', 'lti_gr1a', 'mispricing_mgmt',
    'mispricing_perf', 'ncoa_gr1a', 'ncol_gr1a', 'netdebt_me',
    'netis_at', 'ni_ar1', 'ni_at', 'ni_inc8q', 'ni_ivol',
    'niq_at_chg1', 'niq_be_chg1',
    'niq_su', 'noa_at', 'noa_gr1a', 'oaccruals_at',
    'oaccruals_ni', 'ocf_at', 'ocf_at_chg1', 'ocf_me', 'ope_bel1',
    'pi_nix', 'ppeinv_gr1a', 'prc_highprc_252d',
    'qmj_growth', 'qmj_prof', 'qmj_safety', 'rd_me',
    'rd_sale', 'rd5_at', 'sale_bev', 'sale_emp_gr1',
    'sale_gr3', 'saleq_gr1', 'saleq_su',
    'taccruals_at', 'taccruals_ni', 'tax_gr1a',
]

EXCLUDE_COLS = ['me', 'ret_1_0', 'ret_exc_lead1m', 'size_grp', 'exret']


def classify_features(all_features):
    accounting, market = [], []
    for f in all_features:
        fl = f.lower()
        if any(kw == fl or kw in fl for kw in ACCOUNTING_KEYWORDS):
            accounting.append(f)
        else:
            market.append(f)
    return accounting, market


def sharpe(s):
    s = s.dropna()
    if len(s) < 2 or s.std() == 0:
        return 0.0
    return (s.mean() / s.std()) * np.sqrt(12)


def build_cost_lookup_1x(raw_df, date_col, id_col, spread_col='bidaskhl_21d',
                         floor_bps=20, cap_bps=200):
    if spread_col not in raw_df.columns:
        lookup = raw_df[[date_col, id_col]].copy()
        lookup['cost_bps_1x'] = 50.0
        return lookup
    lookup = raw_df[[date_col, id_col, spread_col]].copy()
    lookup['cost_bps_1x'] = (lookup[spread_col] / 2) * 10000
    monthly_median = lookup.groupby(date_col)['cost_bps_1x'].transform('median')
    lookup['cost_bps_1x'] = lookup['cost_bps_1x'].fillna(monthly_median)
    lookup['cost_bps_1x'] = lookup['cost_bps_1x'].clip(lower=floor_bps, upper=cap_bps)
    return lookup[[date_col, id_col, 'cost_bps_1x']]


def compute_cost_drag(month_data, entries, exits, new_long, id_col,
                      cost_mult, global_med_1x, prev_long):
    if cost_mult == 0.0 or not prev_long or not (entries | exits):
        return 0.0
    traded = entries | exits
    if len(traded) == 0:
        return 0.0
    traded_costs_1x = month_data[month_data[id_col].isin(traded)]['cost_bps_1x'].mean()
    if pd.isna(traded_costs_1x):
        traded_costs_1x = global_med_1x
    traded_costs = np.clip(traded_costs_1x * cost_mult, 20.0, 200.0)
    frac_traded = len(traded) / max(len(new_long), 1)
    return frac_traded * (traded_costs / 10000)


def portfolio_from_ranks(ranks, ids, actuals, prev_long,
                         top_threshold, buffer, nq,
                         test_data, id_col, cost_mults, global_med_1x):
    prev_ids = np.array(list(prev_long), dtype=ids.dtype) if prev_long else np.array([], dtype=ids.dtype)
    held_mask = np.isin(ids, prev_ids)
    select_mask = np.where(held_mask,
                           ranks >= top_threshold - buffer,
                           ranks >= top_threshold + buffer)
    new_long = set(ids[select_mask])
    if len(new_long) == 0:
        top_k = max(1, len(ids) // nq)
        top_idx = np.argsort(ranks)[-top_k:]
        new_long = set(ids[top_idx])

    held_ret = np.isin(ids, np.array(list(new_long), dtype=ids.dtype))
    gross_ret = float(np.nanmean(actuals[held_ret])) if held_ret.sum() > 0 else 0.0

    if prev_long:
        entries = new_long - prev_long
        exits = prev_long - new_long
        turn = (len(entries) + len(exits)) / (2 * max(len(new_long), 1))
    else:
        entries, exits, turn = new_long, set(), np.nan

    cost_drags = {}
    for cm in cost_mults:
        cost_drags[cm] = compute_cost_drag(
            test_data, entries, exits, new_long, id_col,
            cm, global_med_1x, prev_long)
    return new_long, gross_ret, turn, cost_drags


def apply_ema(ranks, ids, ema_state, alpha):
    smoothed = np.zeros_like(ranks)
    for i, sid in enumerate(ids):
        if sid in ema_state:
            smoothed[i] = alpha * ranks[i] + (1 - alpha) * ema_state[sid]
        else:
            smoothed[i] = ranks[i]
        ema_state[sid] = smoothed[i]
    return rankdata(smoothed, method='average') / len(smoothed)


def ridge_fit_predict(X_train, y_train, X_test, alpha=RIDGE_ALPHA):
    p = X_train.shape[1]
    XtX = X_train.T @ X_train
    Xty = X_train.T @ y_train
    reg = alpha * np.eye(p)
    try:
        beta = np.linalg.solve(XtX + reg, Xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(XtX + reg, Xty, rcond=None)[0]
    preds = X_test @ beta
    return np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)


def compute_feature_rank_ac(data, features, date_col, id_col, min_months=24):
    """Compute rank autocorrelation for each feature across stocks."""
    rank_acs = {}
    dates = sorted(data[date_col].unique())
    if len(dates) < min_months + 1:
        return {f: 0.5 for f in features}

    # Use last min_months for estimation
    est_dates = dates[-min_months-1:]
    for feat in features:
        acs = []
        for i in range(1, len(est_dates)):
            d_prev = est_dates[i-1]
            d_curr = est_dates[i]
            prev_data = data[data[date_col] == d_prev][[id_col, feat]].dropna()
            curr_data = data[data[date_col] == d_curr][[id_col, feat]].dropna()
            merged = prev_data.merge(curr_data, on=id_col, suffixes=('_prev', '_curr'))
            if len(merged) >= 10:
                rho, _ = spearmanr(merged[f'{feat}_prev'], merged[f'{feat}_curr'])
                if not np.isnan(rho):
                    acs.append(rho)
        rank_acs[feat] = np.mean(acs) if acs else 0.5
    return rank_acs


def run_single_market(mkt_name, cfg):
    t0 = time.time()

    df, raw_features, date_col, id_col = load_equity_data(cfg['file'], cfg['type'])
    target = cfg['target']
    all_features = [f for f in raw_features if f not in EXCLUDE_COLS and f != target]
    accounting_feats, market_feats = classify_features(all_features)
    accounting_feats = [f for f in accounting_feats if f in df.columns]
    market_feats = [f for f in market_feats if f in df.columns]

    if accounting_feats:
        df = lag_features(df, accounting_feats, id_col, date_col, lag=1)
    if market_feats:
        df = lag_features(df, market_feats, id_col, date_col, lag=1)

    if 'me' in df.columns:
        df = df.sort_values([id_col, date_col])
        df['me'] = df.groupby(id_col)['me'].shift(1)
        df = df.dropna(subset=['me'])

    topn = cfg['topn']
    if 'me' in df.columns:
        df['_me_rank'] = df.groupby(date_col)['me'].rank(ascending=False, method='first')
        df = df[df['_me_rank'] <= topn].drop(columns=['_me_rank'])

    cost_lookup_1x = build_cost_lookup_1x(df, date_col, id_col)
    ml_features = [f for f in accounting_feats + market_feats if f in df.columns]
    ml_features = [f for f in ml_features if f not in EXCLUDE_COLS]

    df[date_col] = pd.to_datetime(df[date_col])
    data = df.copy().sort_values([date_col, id_col]).reset_index(drop=True)
    unique_dates = sorted(data[date_col].unique())
    n_dates = len(unique_dates)

    if n_dates < MIN_TRAIN_MONTHS + 12:
        print(f"  {mkt_name}: too few dates ({n_dates})", flush=True)
        return []

    data = data.merge(cost_lookup_1x, on=[date_col, id_col], how='left')
    global_med_1x = cost_lookup_1x['cost_bps_1x'].median()
    data['cost_bps_1x'] = data['cost_bps_1x'].fillna(global_med_1x)

    # Pre-compute rank target
    data['_rank_target'] = data.groupby(date_col)[target].rank(pct=True)

    avail_features = [f for f in ml_features if f in data.columns]
    feat_coverage = data[avail_features].notna().mean()
    avail_features = feat_coverage[feat_coverage > 0.1].index.tolist()

    # Compute feature rank autocorrelations for speed classification
    # Use first 70% of data (IS period) for feature classification
    is_end_idx = int(n_dates * 0.7)
    is_data = data[data[date_col] <= unique_dates[is_end_idx]]
    rank_acs = compute_feature_rank_ac(is_data, avail_features, date_col, id_col)

    # Classify features by speed
    slow_feats = [f for f in avail_features if rank_acs.get(f, 0.5) >= 0.75]
    fast_feats = [f for f in avail_features if rank_acs.get(f, 0.5) < 0.50]
    med_feats = [f for f in avail_features if 0.50 <= rank_acs.get(f, 0.5) < 0.75]
    slowmed_feats = slow_feats + med_feats

    nq = cfg['nq']
    top_threshold = 1.0 - 1.0 / nq
    buffer = BUFFER_PCT * (1.0 / nq)

    print(f"  {len(data):,} rows, {n_dates} months, {len(avail_features)} features "
          f"(slow={len(slow_feats)}, med={len(med_feats)}, fast={len(fast_feats)})",
          flush=True)

    # Print mean rank AC by speed bucket
    if slow_feats:
        mean_slow_ac = np.mean([rank_acs[f] for f in slow_feats])
        mean_fast_ac = np.mean([rank_acs[f] for f in fast_feats]) if fast_feats else 0
        print(f"    Mean rank AC: slow={mean_slow_ac:.2f}, fast={mean_fast_ac:.2f}", flush=True)

    # Pre-compute date boundaries
    date_start_idx = {}
    date_end_idx = {}
    for d in unique_dates:
        mask_d = data[date_col] == d
        indices = np.where(mask_d.values)[0]
        date_start_idx[d] = indices[0]
        date_end_idx[d] = indices[-1] + 1

    # Feature subsets for each strategy
    feature_sets = {
        'RT_All': avail_features,
        'RT_Slow': slow_feats,
        'RT_Fast': fast_feats,
        'RT_SlowMed': slowmed_feats,
        'Ridge_All': avail_features,  # raw return baseline
    }

    # Filter out empty feature sets
    feature_sets = {k: v for k, v in feature_sets.items() if len(v) >= 3}

    # Separate dict for feature masks (avoid modifying feature_sets during iteration)
    feature_masks = {}

    # Strategy names: raw + EMA versions + MomSort
    raw_strategies = list(feature_sets.keys())
    ema_strategies = [s + '_EMA' for s in raw_strategies if s != 'Ridge_All']
    # Also add Ridge_All_EMA for comparison
    ema_strategies.append('Ridge_All_EMA')
    all_strategies = raw_strategies + ema_strategies + ['MomSort']

    prev_longs = {s: set() for s in all_strategies}
    month_records = {s: [] for s in all_strategies}
    ema_states = {s: {} for s in all_strategies}

    # Track prediction stability (rank correlation between consecutive months)
    prev_ranks = {s: None for s in raw_strategies}
    stability_records = {s: [] for s in raw_strategies}

    all_results = []
    n_steps = n_dates - MIN_TRAIN_MONTHS

    for step_i, t_idx in enumerate(range(MIN_TRAIN_MONTHS, n_dates)):
        if step_i % 50 == 0:
            print(f"    step {step_i}/{n_steps}", flush=True)

        test_date = unique_dates[t_idx]
        train_end = date_start_idx[test_date]
        train_data = data.iloc[:train_end]
        test_data = data.iloc[date_start_idx[test_date]:date_end_idx[test_date]].copy()

        if len(train_data) < 100 or len(test_data) < nq:
            continue

        ids = test_data[id_col].values
        actuals = test_data[target].values

        # For each feature set, fit Ridge and get predictions
        for strat_name, feat_list in feature_sets.items():
            if len(feat_list) < 3:
                continue

            X_train_raw = train_data[feat_list].values
            X_test_raw = test_data[feat_list].values

            # Target depends on strategy
            if strat_name == 'Ridge_All':
                y_raw = train_data[target].values  # raw returns
            else:
                y_raw = train_data['_rank_target'].values  # rank target

            if step_i % 50 == 0:
                cov = 1.0 - np.isnan(X_train_raw).mean(axis=0)
                feature_masks[strat_name] = cov > 0.1

            if strat_name not in feature_masks:
                cov = 1.0 - np.isnan(X_train_raw).mean(axis=0)
                feature_masks[strat_name] = cov > 0.1

            step_mask = feature_masks[strat_name]
            p = step_mask.sum()

            valid = ~np.isnan(y_raw)
            if valid.sum() < 50 or p == 0:
                continue

            imp = SimpleImputer(strategy='median')
            X_tr = imp.fit_transform(X_train_raw[:, step_mask])
            X_te = imp.transform(X_test_raw[:, step_mask])
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr[valid])
            X_te = sc.transform(X_te)

            preds = ridge_fit_predict(X_tr, y_raw[valid], X_te)
            ranks = rankdata(preds, method='average') / len(preds)

            # Track prediction stability
            if prev_ranks[strat_name] is not None:
                # Find common stocks
                prev_ids_set, prev_r = prev_ranks[strat_name]
                common = set(ids) & set(prev_ids_set)
                if len(common) >= 10:
                    curr_map = dict(zip(ids, ranks))
                    prev_map = dict(zip(prev_ids_set, prev_r))
                    common_list = list(common)
                    r_curr = [curr_map[s] for s in common_list]
                    r_prev = [prev_map[s] for s in common_list]
                    rho, _ = spearmanr(r_curr, r_prev)
                    if not np.isnan(rho):
                        stability_records[strat_name].append(rho)
            prev_ranks[strat_name] = (ids.copy(), ranks.copy())

            # Raw strategy portfolio
            new_long, gross_ret, turn, cost_drags = portfolio_from_ranks(
                ranks, ids, actuals, prev_longs[strat_name],
                top_threshold, buffer, nq, test_data, id_col, COST_MULTS, global_med_1x)
            month_records[strat_name].append({
                'date': test_date, 'gross_ret': gross_ret, 'turnover': turn,
                'n_held': len(new_long),
                **{f'cost_drag_{cm}': cost_drags[cm] for cm in COST_MULTS}})
            prev_longs[strat_name] = new_long

            # EMA version
            ema_name = strat_name + '_EMA'
            if ema_name in all_strategies:
                ranks_ema = apply_ema(ranks.copy(), ids, ema_states[ema_name], EMA_ALPHA)
                new_long, gross_ret, turn, cost_drags = portfolio_from_ranks(
                    ranks_ema, ids, actuals, prev_longs[ema_name],
                    top_threshold, buffer, nq, test_data, id_col, COST_MULTS, global_med_1x)
                month_records[ema_name].append({
                    'date': test_date, 'gross_ret': gross_ret, 'turnover': turn,
                    'n_held': len(new_long),
                    **{f'cost_drag_{cm}': cost_drags[cm] for cm in COST_MULTS}})
                prev_longs[ema_name] = new_long

        # MomSort
        mom_feat = 'ret_12_1'
        if mom_feat in test_data.columns:
            mom_vals = test_data[mom_feat].values
            mom_valid = ~np.isnan(mom_vals)
            mom_ranks = np.full(len(mom_vals), 0.5)
            if mom_valid.sum() >= nq:
                mom_ranks[mom_valid] = rankdata(mom_vals[mom_valid], method='average') / mom_valid.sum()
            new_long, gross_ret, turn, cost_drags = portfolio_from_ranks(
                mom_ranks, ids, actuals, prev_longs['MomSort'],
                top_threshold, buffer, nq, test_data, id_col, COST_MULTS, global_med_1x)
            month_records['MomSort'].append({
                'date': test_date, 'gross_ret': gross_ret, 'turnover': turn,
                'n_held': len(new_long),
                **{f'cost_drag_{cm}': cost_drags[cm] for cm in COST_MULTS}})
            prev_longs['MomSort'] = new_long

    # ---- Compute OOS results ----
    for strat in all_strategies:
        recs = month_records[strat]
        if len(recs) < 6:
            continue
        result_df = pd.DataFrame(recs).set_index('date')
        unique_d = sorted(result_df.index)
        is_end = unique_d[int(len(unique_d) * 0.7)]
        oos_mask = result_df.index > is_end
        subset = result_df[oos_mask]
        if len(subset) < 6:
            continue

        # Get stability for this strategy (if tracked)
        base_strat = strat.replace('_EMA', '')
        stab = np.mean(stability_records.get(base_strat, [0.5]))

        for cm in COST_MULTS:
            cost_col = f'cost_drag_{cm}'
            net_ret = subset['gross_ret'] - subset[cost_col]
            mean_to = subset['turnover'].mean()

            # Count features
            feat_key = base_strat
            n_feats = len(feature_sets.get(feat_key, []))

            all_results.append({
                'Market': mkt_name,
                'Strategy': strat,
                'CostMult': cm,
                'Net_SR': round(sharpe(net_ret), 4),
                'Gross_SR': round(sharpe(subset['gross_ret']), 4),
                'Turnover': round(mean_to, 4),
                'Stability': round(stab, 3),
                'N_features': n_feats,
                'N_months': len(subset),
            })

    elapsed = time.time() - t0
    print(f"  {mkt_name} done in {elapsed:.0f}s", flush=True)
    return all_results


if __name__ == '__main__':
    t0_global = time.time()
    all_results = []

    for mkt_name, cfg in MARKETS.items():
        print(f"\n{'='*60}")
        print(f"  {mkt_name}")
        print(f"{'='*60}")
        try:
            results = run_single_market(mkt_name, cfg)
            all_results.extend(results)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            continue

    results_df = pd.DataFrame(all_results)
    out_path = 'scripts/concern_experiments/expCL_ranktarget_feature_speed.csv'
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # ======================================================================
    # SUMMARY
    # ======================================================================
    print("\n" + "="*80)
    print("FEATURE SPEED ON RANK TARGETS: OOS Net SR at 3x costs")
    print("="*80)

    oos_3x = results_df[results_df['CostMult'] == 3.0]
    if len(oos_3x) > 0:
        col_order = ['MomSort', 'Ridge_All', 'Ridge_All_EMA',
                     'RT_All', 'RT_All_EMA',
                     'RT_Slow', 'RT_Slow_EMA',
                     'RT_Fast', 'RT_Fast_EMA',
                     'RT_SlowMed', 'RT_SlowMed_EMA']

        pivot = oos_3x.pivot_table(index='Market', columns='Strategy', values='Net_SR')
        pivot = pivot[[c for c in col_order if c in pivot.columns]]
        print("\n--- Net SR (3x costs) ---")
        print(pivot.round(3).to_string())
        print(f"\nMean:\n{pivot.mean().round(3).to_string()}")

        pivot_to = oos_3x.pivot_table(index='Market', columns='Strategy', values='Turnover')
        pivot_to = pivot_to[[c for c in col_order if c in pivot_to.columns]]
        print("\n--- Mean Monthly Turnover ---")
        print(pivot_to.round(3).to_string())
        print(f"\nMean:\n{pivot_to.mean().round(3).to_string()}")

        pivot_gross = oos_3x.pivot_table(index='Market', columns='Strategy', values='Gross_SR')
        pivot_gross = pivot_gross[[c for c in col_order if c in pivot_gross.columns]]
        print("\n--- Gross SR ---")
        print(pivot_gross.round(3).to_string())
        print(f"\nMean:\n{pivot_gross.mean().round(3).to_string()}")

        # Stability
        pivot_stab = oos_3x.pivot_table(index='Market', columns='Strategy', values='Stability')
        pivot_stab = pivot_stab[[c for c in col_order if c in pivot_stab.columns]]
        print("\n--- Prediction Stability (rank AC) ---")
        print(pivot_stab.round(3).to_string())
        print(f"\nMean:\n{pivot_stab.mean().round(3).to_string()}")

        # Feature counts
        pivot_nf = oos_3x.pivot_table(index='Market', columns='Strategy', values='N_features')
        pivot_nf = pivot_nf[[c for c in col_order if c in pivot_nf.columns]]
        print("\n--- N Features ---")
        print(pivot_nf.round(0).to_string())

        if 'MomSort' in pivot.columns:
            print("\n--- vs MomSort ---")
            for strat in col_order[1:]:
                if strat in pivot.columns:
                    wins = (pivot[strat] > pivot['MomSort']).sum()
                    total = pivot[strat].notna().sum()
                    delta = (pivot[strat] - pivot['MomSort']).mean()
                    to_mean = pivot_to[strat].mean() if strat in pivot_to.columns else np.nan
                    print(f"  {strat:20s}: {wins}/{total} wins, "
                          f"delta={delta:+.3f}, TO={to_mean:.1%}")

        # Key comparison: feature speed effect
        print("\n--- FEATURE SPEED EFFECT (rank target) ---")
        for pair in [('RT_Slow', 'RT_Fast'), ('RT_Slow_EMA', 'RT_Fast_EMA')]:
            s, f = pair
            if s in pivot.columns and f in pivot.columns:
                sr_slow = pivot[s].mean()
                sr_fast = pivot[f].mean()
                to_slow = pivot_to[s].mean()
                to_fast = pivot_to[f].mean()
                stab_slow = pivot_stab[s].mean() if s in pivot_stab.columns else 0
                stab_fast = pivot_stab[f].mean() if f in pivot_stab.columns else 0
                print(f"  {s:20s}: SR={sr_slow:.3f}, TO={to_slow:.1%}, stab={stab_slow:.3f}")
                print(f"  {f:20s}: SR={sr_fast:.3f}, TO={to_fast:.1%}, stab={stab_fast:.3f}")
                print(f"  --> Slow-Fast gap: SR={sr_slow-sr_fast:+.3f}, "
                      f"TO={to_slow-to_fast:+.1%}")
                print()

    elapsed = time.time() - t0_global
    print(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
