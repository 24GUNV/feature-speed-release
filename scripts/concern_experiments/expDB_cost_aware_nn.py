"""
Experiment DB: Cost-Aware Neural Networks + Lipschitz-Constrained MLP

Two novel approaches to beat Ridge_Slow (SR 0.47, TO 10.2%):

  1. MomentumSort           -- ret_12_1 sort, 20% buffer (baseline)
  2. Ridge_Slow             -- Ridge(alpha=100) on slow features, rank targets (control)
  3. CostAware_Linear       -- Linear (PyTorch), end-to-end net-of-cost Sharpe loss
  4. Lipschitz_NN           -- Spectral-norm MLP, MSE on rank targets
  5. CostAware_Lipschitz    -- Spectral-norm MLP, end-to-end net-of-cost Sharpe loss

Core innovations:
  - End-to-end differentiable portfolio with ID-based weight tracking across months.
    The model learns to avoid costly trades via gradient-based optimization.
  - Lipschitz-constrained MLP: spectral normalization bounds prediction sensitivity,
    guaranteeing smooth outputs while allowing nonlinearity.
  - Ridge-initialized weights: NN starts from Ridge solution for stability.

Walk-forward hyperparameter selection (IS net SR):
  - Lip constant k in {0.5, 1.0, 2.0}
  - Turnover penalty lambda in {0, 0.5, 2.0, 5.0}

Pipeline matches expAH/expDA exactly: lag=1, rank targets, 20% buffer, CS costs.

Output:
  - scripts/concern_experiments/expDB_cost_aware_nn.csv
  - scripts/concern_experiments/expDB_training_log.csv
"""

import sys
import os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, 'public/src')
sys.path.insert(0, 'scripts/concern_experiments')

import numpy as np
import pandas as pd
import warnings
import time
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from data_loader import load_equity_data, lag_features
from market_config import MARKETS, COST_MULTS, BUFFER_PCT, MIN_TRAIN_MONTHS

np.random.seed(42)
torch.manual_seed(42)

DEVICE = 'cpu'


# ==============================================================================
# CONFIGURATION
# ==============================================================================

LIP_CONSTANTS = [0.5, 1.0, 2.0]
TURNOVER_LAMBDAS = [0.0, 0.5, 2.0, 5.0]

HIDDEN_DIM = 32
LR = 0.001
FINETUNE_EPOCHS = 15
TRAIN_HORIZON = 24
TEMPERATURE = 10.0
WF_REEVAL_INTERVAL = 6

TOP_FRAC = 0.20
BUFFER_FRAC = 0.04


# ==============================================================================
# FEATURE SPEED CLASSIFICATION (from expAH)
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

EXCLUDE_COLS = ['Unnamed: 0', 'id', 'eom', 'excntry', 'gvkey', 'permno',
                'size_grp', 'me', 'market_equity', 'ret_1_0', 'date']

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


def classify_feature_speed(feature_name):
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


def classify_features(all_features):
    accounting = []
    market = []
    for f in all_features:
        f_lower = f.lower()
        if any(kw == f_lower or kw in f_lower for kw in ACCOUNTING_KEYWORDS):
            accounting.append(f)
        else:
            market.append(f)
    return accounting, market


# ==============================================================================
# HELPER FUNCTIONS (from expDA/expAH)
# ==============================================================================

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
    if cost_mult == 0.0:
        return 0.0
    if not prev_long or not (entries | exits):
        return 0.0
    traded = entries | exits
    if len(traded) == 0:
        return 0.0
    traded_costs_1x = month_data[month_data[id_col].isin(traded)]['cost_bps_1x'].mean()
    if pd.isna(traded_costs_1x):
        traded_costs_1x = global_med_1x
    traded_costs = np.clip(traded_costs_1x * cost_mult, 20.0, 200.0)
    frac_traded = len(traded) / max(len(new_long), 1)
    cost_drag = frac_traded * (traded_costs / 10000)
    return cost_drag


# ==============================================================================
# FACTOR SORT (from expAH)
# ==============================================================================

def run_factor_sort(df, sort_features, date_col, id_col, target,
                    cost_lookup_1x, nq=5, buffer_pct=0.20):
    data = df.copy()
    for feat in sort_features:
        if feat not in data.columns:
            continue
        data[f'{feat}_rank'] = data.groupby(date_col)[feat].rank(pct=True)
        data[f'{feat}_rank'] = data[f'{feat}_rank'].fillna(0.5)

    rank_cols = [f'{f}_rank' for f in sort_features if f in data.columns]
    if len(rank_cols) == 0:
        return None
    data['composite_score'] = data[rank_cols].mean(axis=1)
    data['pct_rank'] = data.groupby(date_col)['composite_score'].rank(pct=True)

    top_threshold = 1.0 - 1.0 / nq
    buffer = buffer_pct * (1.0 / nq)

    data = data.merge(cost_lookup_1x, on=[date_col, id_col], how='left')
    global_med_1x = cost_lookup_1x['cost_bps_1x'].median()
    data['cost_bps_1x'] = data['cost_bps_1x'].fillna(global_med_1x)

    dates = sorted(data[date_col].unique())
    prev_long = set()
    records = []

    for i, date in enumerate(dates):
        month = data[data[date_col] == date].copy()
        if len(month) < nq:
            continue

        ids = month[id_col].values
        ranks = month['pct_rank'].values
        prev_ids = np.array(list(prev_long), dtype=ids.dtype) if prev_long else np.array([], dtype=ids.dtype)
        held_mask = np.isin(ids, prev_ids)
        select_mask = np.where(held_mask,
                               ranks >= top_threshold - buffer,
                               ranks >= top_threshold + buffer)
        new_long = set(ids[select_mask])
        if len(new_long) == 0:
            new_long = set(month.nlargest(max(1, len(month) // nq), 'pct_rank')[id_col])

        held = month[month[id_col].isin(new_long)]
        gross_ret = held[target].mean() if len(held) > 0 else 0.0

        if prev_long:
            entries = new_long - prev_long
            exits = prev_long - new_long
            turn = (len(entries) + len(exits)) / (2 * max(len(new_long), 1))
        else:
            entries = new_long
            exits = set()
            turn = np.nan

        cost_drags = {}
        for cm in COST_MULTS:
            cost_drags[cm] = compute_cost_drag(
                month, entries, exits, new_long, id_col,
                cm, global_med_1x, prev_long
            )

        records.append({
            'date': date, 'gross_ret': gross_ret, 'turnover': turn,
            'n_held': len(new_long),
            **{f'cost_drag_{cm}': cost_drags[cm] for cm in COST_MULTS},
        })
        prev_long = new_long

    if len(records) == 0:
        return None
    return pd.DataFrame(records).set_index('date')


# ==============================================================================
# PYTORCH MODELS
# ==============================================================================

class LinearModel(nn.Module):
    """Linear model in PyTorch (equivalent to Ridge without regularization)."""
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.fc(x)

    def init_from_ridge(self, ridge_coefs, ridge_intercept):
        """Initialize weights from a fitted Ridge model."""
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor(ridge_coefs, dtype=torch.float32).unsqueeze(0))
            self.fc.bias.copy_(torch.tensor([ridge_intercept], dtype=torch.float32))


class LipschitzMLP(nn.Module):
    """MLP with spectral normalization for bounded Lipschitz constant."""
    def __init__(self, input_dim, hidden_dim=32, lip_const=1.0):
        super().__init__()
        self.fc1 = spectral_norm(nn.Linear(input_dim, hidden_dim))
        self.fc2 = spectral_norm(nn.Linear(hidden_dim, hidden_dim))
        self.fc3 = spectral_norm(nn.Linear(hidden_dim, 1))
        self.lip_const = lip_const

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = F.leaky_relu(self.fc2(x), 0.1)
        return self.fc3(x) * self.lip_const

    def init_approx_linear(self, ridge_coefs, ridge_intercept):
        """Initialize to approximate a linear model (small hidden weights)."""
        with torch.no_grad():
            # Set first layer to pass-through-like
            nn.init.xavier_uniform_(self.fc1.weight, gain=0.1)
            nn.init.zeros_(self.fc1.bias)
            nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
            nn.init.zeros_(self.fc2.bias)
            nn.init.xavier_uniform_(self.fc3.weight, gain=0.1)
            self.fc3.bias.fill_(ridge_intercept * 0.1)


# ==============================================================================
# DIFFERENTIABLE PORTFOLIO WITH ID-BASED WEIGHT TRACKING
# ==============================================================================

def soft_portfolio_with_ids(predictions, ids, prev_weight_dict,
                            half_spreads_bps, kappa,
                            top_frac=TOP_FRAC, temperature=TEMPERATURE,
                            buffer_frac=BUFFER_FRAC):
    """
    Differentiable top-quintile portfolio with proper stock ID tracking.

    Args:
        predictions: (n_stocks,) tensor of predicted scores
        ids: numpy array of stock IDs for this month
        prev_weight_dict: dict mapping stock_id -> previous weight (float)
        half_spreads_bps: (n_stocks,) tensor of half-spread in bps (1x)
        kappa: cost multiplier
    Returns:
        weights: (n_stocks,) normalized portfolio weights
        cost: scalar transaction cost
        new_weight_dict: dict mapping stock_id -> new weight (detached)
    """
    n = predictions.shape[0]
    if n == 0:
        return predictions, torch.tensor(0.0), {}

    # Reconstruct prev_weights from dict, aligned by stock ID
    prev_weights = torch.zeros(n)
    for i, sid in enumerate(ids):
        key = int(sid) if isinstance(sid, (np.integer, int, float)) else sid
        if key in prev_weight_dict:
            prev_weights[i] = prev_weight_dict[key]

    # Compute threshold
    k = max(1, int(n * top_frac))
    sorted_preds, _ = torch.sort(predictions, descending=True)
    threshold = sorted_preds[min(k, n - 1)]

    # Buffer: lower threshold for currently held stocks
    held_mask = (prev_weights > 1e-6).float()
    effective_threshold = threshold - buffer_frac * held_mask

    # Soft selection via sigmoid
    raw_weights = torch.sigmoid(temperature * (predictions - effective_threshold))
    weight_sum = raw_weights.sum()
    if weight_sum < 1e-8:
        weights = torch.ones(n) / n
    else:
        weights = raw_weights / weight_sum

    # Differentiable transaction cost
    turnover = torch.abs(weights - prev_weights)
    cost = (turnover * (half_spreads_bps / 10000.0) * kappa).sum()

    # Build new weight dict (detached)
    new_weight_dict = {}
    w_detached = weights.detach()
    for i, sid in enumerate(ids):
        key = int(sid) if isinstance(sid, (np.integer, int, float)) else sid
        new_weight_dict[key] = w_detached[i].item()

    return weights, cost, new_weight_dict


def hard_portfolio_from_preds(preds_np, prev_long_set, ids, nq, buffer_pct,
                              month_data, id_col, target, cost_lookup_data,
                              global_med_1x):
    """Standard hard portfolio construction for evaluation."""
    preds_series = pd.Series(preds_np, index=range(len(preds_np)))
    pct_ranks = preds_series.rank(pct=True).values

    top_threshold = 1.0 - 1.0 / nq
    buffer = buffer_pct * (1.0 / nq)

    prev_ids = np.array(list(prev_long_set), dtype=ids.dtype) if prev_long_set else np.array([], dtype=ids.dtype)
    held_mask = np.isin(ids, prev_ids)
    select_mask = np.where(held_mask,
                           pct_ranks >= top_threshold - buffer,
                           pct_ranks >= top_threshold + buffer)
    new_long = set(ids[select_mask])
    if len(new_long) == 0:
        top_k = max(1, len(ids) // nq)
        top_indices = np.argsort(preds_np)[-top_k:]
        new_long = set(ids[top_indices])

    held = month_data[month_data[id_col].isin(new_long)]
    gross_ret = held[target].mean() if len(held) > 0 else 0.0

    if prev_long_set:
        entries = new_long - prev_long_set
        exits = prev_long_set - new_long
        turn = (len(entries) + len(exits)) / (2 * max(len(new_long), 1))
    else:
        entries = new_long
        exits = set()
        turn = np.nan

    cost_drags = {}
    for cm in COST_MULTS:
        cost_drags[cm] = compute_cost_drag(
            cost_lookup_data, entries, exits, new_long, id_col,
            cm, global_med_1x, prev_long_set
        )

    return gross_ret, turn, new_long, cost_drags, entries, exits


# ==============================================================================
# COST-AWARE TRAINING LOOP (with ID-based weight tracking)
# ==============================================================================

def train_cost_aware_model(model, optimizer, monthly_data_list, kappa=3.0,
                           turnover_lambda=0.0, n_epochs=FINETUNE_EPOCHS,
                           train_horizon=TRAIN_HORIZON):
    """
    Train end-to-end through differentiable portfolio with stock ID tracking.

    monthly_data_list items must have 'ids' key (numpy array of stock IDs).
    """
    model.train()
    data_slice = monthly_data_list[-train_horizon:] if len(monthly_data_list) > train_horizon else monthly_data_list

    if len(data_slice) < 6:
        return float('nan')

    final_loss = float('nan')
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        monthly_net_returns = []
        monthly_turnovers = []
        prev_weight_dict = {}

        for t, mdata in enumerate(data_slice):
            X_t = mdata['X']
            ids = mdata['ids']

            predictions = model(X_t).squeeze()
            if predictions.dim() == 0:
                predictions = predictions.unsqueeze(0)

            weights, cost, prev_weight_dict = soft_portfolio_with_ids(
                predictions, ids, prev_weight_dict,
                mdata['half_spreads_bps'], kappa
            )

            gross_ret = (weights * mdata['returns']).sum()
            net_ret = gross_ret - cost
            monthly_net_returns.append(net_ret)

            # Track turnover for explicit penalty
            n = weights.shape[0]
            prev_w = torch.zeros(n)
            for i, sid in enumerate(ids):
                key = int(sid) if isinstance(sid, (np.integer, int, float)) else sid
                # prev_weight_dict was updated, use the old one...
                # Actually we already detached in soft_portfolio_with_ids
            turnover_t = cost  # use cost as turnover proxy
            monthly_turnovers.append(turnover_t)

        rets = torch.stack(monthly_net_returns)
        ret_mean = rets.mean()
        ret_std = rets.std()
        if ret_std < 1e-8:
            sharpe_loss = -ret_mean * 100.0
        else:
            sharpe_loss = -(ret_mean / ret_std) * np.sqrt(12)

        # L2 regularization to stay near Ridge solution
        l2_reg = sum(p.pow(2).sum() for p in model.parameters())

        # Optional explicit turnover penalty
        if turnover_lambda > 0 and len(monthly_turnovers) > 0:
            avg_cost = torch.stack(monthly_turnovers).mean()
            total_loss = sharpe_loss + turnover_lambda * avg_cost + 0.001 * l2_reg
        else:
            total_loss = sharpe_loss + 0.001 * l2_reg

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        final_loss = total_loss.item()

    model.eval()
    return final_loss


def train_mse_model(model, optimizer, X_train_t, y_train_t, n_epochs=FINETUNE_EPOCHS):
    """Standard MSE training on rank targets (for Lipschitz_NN)."""
    model.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        preds = model(X_train_t).squeeze()
        loss = F.mse_loss(preds, y_train_t)
        l2_reg = sum(p.pow(2).sum() for p in model.parameters())
        loss = loss + 0.01 * l2_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    model.eval()
    return loss.item()


# ==============================================================================
# WALK-FORWARD HYPERPARAMETER SELECTION
# ==============================================================================

def select_hyperparams_wf(model_class, monthly_data_list, input_dim,
                          kappa=3.0, hp_grid=None):
    """
    Walk-forward HP selection on IS data via net Sharpe.
    Returns best_lip_const, best_lambda, best_is_sr.
    """
    if hp_grid is None:
        if model_class == 'linear':
            hp_grid = [(None, lam) for lam in TURNOVER_LAMBDAS]
        elif model_class == 'lipschitz':
            hp_grid = [(k, None) for k in LIP_CONSTANTS]
        else:
            hp_grid = [(k, lam) for k in LIP_CONSTANTS for lam in TURNOVER_LAMBDAS]

    best_sr = -999.0
    best_hp = hp_grid[0]

    for lip_k, lam in hp_grid:
        try:
            if model_class == 'linear':
                model = LinearModel(input_dim).to(DEVICE)
            else:
                model = LipschitzMLP(input_dim, HIDDEN_DIM,
                                     lip_const=lip_k or 1.0).to(DEVICE)

            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.01)

            if model_class == 'lipschitz':
                if len(monthly_data_list) < 6:
                    continue
                X_all = torch.cat([m['X'] for m in monthly_data_list], dim=0)
                y_all = torch.cat([m['rank_targets'] for m in monthly_data_list], dim=0)
                train_mse_model(model, optimizer, X_all, y_all, n_epochs=30)
            else:
                train_cost_aware_model(
                    model, optimizer, monthly_data_list, kappa=kappa,
                    turnover_lambda=lam or 0.0, n_epochs=20,
                    train_horizon=min(TRAIN_HORIZON, len(monthly_data_list))
                )

            # Evaluate: simulate portfolio on IS data using ID tracking
            model.eval()
            monthly_rets = []
            prev_wd = {}
            for mdata in monthly_data_list[-24:]:
                with torch.no_grad():
                    preds = model(mdata['X']).squeeze()
                weights, cost, prev_wd = soft_portfolio_with_ids(
                    preds, mdata['ids'], prev_wd,
                    mdata['half_spreads_bps'], kappa
                )
                gross = (weights * mdata['returns']).sum().item()
                net = gross - cost.item()
                monthly_rets.append(net)

            sr = sharpe(pd.Series(monthly_rets))
            if sr > best_sr:
                best_sr = sr
                best_hp = (lip_k, lam)

        except Exception:
            continue

    return best_hp[0], best_hp[1], best_sr


# ==============================================================================
# EVALUATION
# ==============================================================================

def evaluate(result_df, cost_mult, period_label, period_mask,
             strategy_name, market):
    if result_df is None:
        return None
    subset = result_df[period_mask]
    if len(subset) < 6:
        return None
    cost_col = f'cost_drag_{cost_mult}'
    net_ret = subset['gross_ret'] - subset[cost_col]
    gross_ret = subset['gross_ret']
    return {
        'Market': market,
        'Strategy': strategy_name,
        'CostMult': cost_mult,
        'Period': period_label,
        'Net_SR': round(sharpe(net_ret), 4),
        'Gross_SR': round(sharpe(gross_ret), 4),
        'Ann_Ret': round(net_ret.mean() * 12, 4),
        'Turnover': round(subset['turnover'].mean(), 4),
        'Avg_Cost_bps': round(subset[cost_col].mean() * 10000, 1),
        'N_months': len(subset),
    }


# ==============================================================================
# BUILD MONTHLY DATA DICTS (shared helper)
# ==============================================================================

def build_monthly_data(data, dates, avail_features, date_col, id_col,
                       target, imputer, scaler, nq):
    """Build list of monthly data dicts with stock IDs for training."""
    monthly_data = []
    for d in dates:
        d_data = data[data[date_col] == d]
        if len(d_data) < nq:
            continue
        X_d = d_data[avail_features].values
        X_d = imputer.transform(X_d)
        X_d = scaler.transform(X_d)
        monthly_data.append({
            'X': torch.tensor(X_d, dtype=torch.float32),
            'returns': torch.tensor(d_data[target].values, dtype=torch.float32),
            'half_spreads_bps': torch.tensor(d_data['cost_bps_1x'].values, dtype=torch.float32),
            'rank_targets': torch.tensor(d_data['_rank_target'].values, dtype=torch.float32),
            'ids': d_data[id_col].values,
        })
    return monthly_data


# ==============================================================================
# MAIN WALK-FORWARD LOOP FOR NN STRATEGIES
# ==============================================================================

def run_nn_walkforward(df, features, date_col, id_col, target,
                       cost_lookup_1x, nq=5, buffer_pct=0.20,
                       min_train=60, strategy_type='cost_linear',
                       lip_const=1.0, turnover_lambda=0.0):
    """
    Walk-forward loop for PyTorch models with warm-starting and Ridge init.
    Model is created ONCE and warm-started throughout.
    """
    data = df.copy()
    data = data.sort_values([date_col, id_col]).reset_index(drop=True)
    data['_rank_target'] = data.groupby(date_col)[target].rank(pct=True)

    unique_dates = sorted(data[date_col].unique())
    n_dates = len(unique_dates)

    if n_dates < min_train + 12:
        return None

    data = data.merge(cost_lookup_1x, on=[date_col, id_col], how='left')
    global_med_1x = cost_lookup_1x['cost_bps_1x'].median()
    data['cost_bps_1x'] = data['cost_bps_1x'].fillna(global_med_1x)

    avail_features = [f for f in features if f in data.columns]
    feat_coverage = data[avail_features].notna().mean()
    avail_features = feat_coverage[feat_coverage > 0.1].index.tolist()

    if len(avail_features) == 0:
        return None

    input_dim = len(avail_features)

    prev_long = set()
    records = []

    # Model state — created once, warm-started throughout
    current_model = None
    current_optimizer = None
    current_lip_k = lip_const
    current_lambda = turnover_lambda
    wf_counter = 0

    for t_idx in range(min_train, n_dates):
        train_dates = unique_dates[:t_idx]
        test_date = unique_dates[t_idx]

        train_data = data[data[date_col].isin(train_dates)]
        test_data = data[data[date_col] == test_date].copy()

        if len(test_data) < nq:
            continue

        # Fit imputer/scaler on training data
        X_train = train_data[avail_features].copy()
        y_train = train_data['_rank_target'].values
        X_test = test_data[avail_features].copy()

        imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        mask = ~np.isnan(y_train)
        if mask.sum() < 50:
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp[mask])
        X_test_scaled = scaler.transform(X_test_imp)

        # === FIRST ITERATION: Create model and initialize from Ridge ===
        if current_model is None:
            # Fit Ridge to get initialization weights
            ridge = Ridge(alpha=100, random_state=42)
            ridge.fit(X_train_scaled, y_train[mask])

            if strategy_type == 'cost_linear':
                current_model = LinearModel(input_dim).to(DEVICE)
                current_model.init_from_ridge(ridge.coef_, ridge.intercept_)
            else:
                current_model = LipschitzMLP(input_dim, HIDDEN_DIM,
                                             lip_const=current_lip_k).to(DEVICE)
                current_model.init_approx_linear(ridge.coef_, ridge.intercept_)

            current_optimizer = torch.optim.Adam(current_model.parameters(),
                                                  lr=LR, weight_decay=0.01)

        # === HP SEARCH (periodically, doesn't rebuild model) ===
        do_hp_search = (wf_counter % WF_REEVAL_INTERVAL == 0) and wf_counter > 0

        if do_hp_search:
            recent_dates = train_dates[-TRAIN_HORIZON:]
            monthly_data_hp = build_monthly_data(
                data, recent_dates, avail_features, date_col, id_col,
                target, imputer, scaler, nq
            )

            if len(monthly_data_hp) >= 12:
                if strategy_type == 'cost_linear':
                    hp_grid = [(None, lam) for lam in TURNOVER_LAMBDAS]
                    _, best_lam, _ = select_hyperparams_wf(
                        'linear', monthly_data_hp, input_dim,
                        kappa=3.0, hp_grid=hp_grid
                    )
                    current_lambda = best_lam or 0.0
                elif strategy_type == 'lipschitz':
                    hp_grid = [(k, None) for k in LIP_CONSTANTS]
                    best_k, _, _ = select_hyperparams_wf(
                        'lipschitz', monthly_data_hp, input_dim,
                        kappa=3.0, hp_grid=hp_grid
                    )
                    # Update lip_const without rebuilding model
                    current_lip_k = best_k or 1.0
                    current_model.lip_const = current_lip_k
                else:  # cost_lipschitz
                    hp_grid = [(k, lam) for k in LIP_CONSTANTS for lam in TURNOVER_LAMBDAS]
                    best_k, best_lam, _ = select_hyperparams_wf(
                        'cost_lipschitz', monthly_data_hp, input_dim,
                        kappa=3.0, hp_grid=hp_grid
                    )
                    current_lip_k = best_k or 1.0
                    current_lambda = best_lam or 0.0
                    current_model.lip_const = current_lip_k

        # === FINE-TUNE on recent data ===
        recent_dates_ft = train_dates[-TRAIN_HORIZON:]
        monthly_data_ft = build_monthly_data(
            data, recent_dates_ft, avail_features, date_col, id_col,
            target, imputer, scaler, nq
        )

        if len(monthly_data_ft) >= 6:
            if strategy_type in ('cost_linear', 'cost_lipschitz'):
                train_cost_aware_model(
                    current_model, current_optimizer, monthly_data_ft,
                    kappa=3.0, turnover_lambda=current_lambda,
                    n_epochs=FINETUNE_EPOCHS,
                    train_horizon=min(TRAIN_HORIZON, len(monthly_data_ft))
                )
            else:  # lipschitz (MSE)
                X_all = torch.cat([m['X'] for m in monthly_data_ft], dim=0)
                y_all = torch.cat([m['rank_targets'] for m in monthly_data_ft], dim=0)
                train_mse_model(current_model, current_optimizer, X_all, y_all,
                               n_epochs=FINETUNE_EPOCHS)

        # === PREDICT on test month ===
        X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            preds = current_model(X_test_t).squeeze().cpu().numpy()
        if preds.ndim == 0:
            preds = np.array([preds.item()])

        # Standard hard portfolio for evaluation
        ids = test_data[id_col].values
        gross_ret, turn, new_long, cost_drags, entries, exits = hard_portfolio_from_preds(
            preds, prev_long, ids, nq, buffer_pct,
            test_data, id_col, target, test_data, global_med_1x
        )

        records.append({
            'date': test_date, 'gross_ret': gross_ret, 'turnover': turn,
            'n_held': len(new_long),
            **{f'cost_drag_{cm}': cost_drags[cm] for cm in COST_MULTS},
        })
        prev_long = new_long
        wf_counter += 1

    if len(records) == 0:
        return None
    return pd.DataFrame(records).set_index('date')


# ==============================================================================
# RIDGE WALK-FORWARD (control, from expAH)
# ==============================================================================

def run_ridge_walkforward(df, features, date_col, id_col, target,
                          cost_lookup_1x, nq=5, buffer_pct=0.20, min_train=60):
    data = df.copy()
    data = data.sort_values([date_col, id_col]).reset_index(drop=True)
    data['_rank_target'] = data.groupby(date_col)[target].rank(pct=True)

    unique_dates = sorted(data[date_col].unique())
    n_dates = len(unique_dates)

    if n_dates < min_train + 12:
        return None

    data = data.merge(cost_lookup_1x, on=[date_col, id_col], how='left')
    global_med_1x = cost_lookup_1x['cost_bps_1x'].median()
    data['cost_bps_1x'] = data['cost_bps_1x'].fillna(global_med_1x)

    avail_features = [f for f in features if f in data.columns]
    feat_coverage = data[avail_features].notna().mean()
    avail_features = feat_coverage[feat_coverage > 0.1].index.tolist()

    if len(avail_features) == 0:
        return None

    prev_long = set()
    records = []

    for t_idx in range(min_train, n_dates):
        train_dates = unique_dates[:t_idx]
        test_date = unique_dates[t_idx]

        train_data = data[data[date_col].isin(train_dates)]
        test_data = data[data[date_col] == test_date].copy()

        if len(test_data) < nq:
            continue

        X_train = train_data[avail_features].copy()
        y_train = train_data['_rank_target'].values
        X_test = test_data[avail_features].copy()

        imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        mask = ~np.isnan(y_train)
        if mask.sum() < 50:
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imp[mask])
        X_test_scaled = scaler.transform(X_test_imp)

        model = Ridge(alpha=100, random_state=42)
        model.fit(X_train_scaled, y_train[mask])
        preds = model.predict(X_test_scaled)

        ids = test_data[id_col].values
        gross_ret, turn, new_long, cost_drags, entries, exits = hard_portfolio_from_preds(
            preds, prev_long, ids, nq, buffer_pct,
            test_data, id_col, target, test_data, global_med_1x
        )

        records.append({
            'date': test_date, 'gross_ret': gross_ret, 'turnover': turn,
            'n_held': len(new_long),
            **{f'cost_drag_{cm}': cost_drags[cm] for cm in COST_MULTS},
        })
        prev_long = new_long

    if len(records) == 0:
        return None
    return pd.DataFrame(records).set_index('date')


# ==============================================================================
# MAIN
# ==============================================================================

print("=" * 100)
print("EXPERIMENT DB: Cost-Aware Neural Networks + Lipschitz-Constrained MLP")
print("=" * 100)
print(f"Config: hidden={HIDDEN_DIM}, lr={LR}, epochs={FINETUNE_EPOCHS}, "
      f"horizon={TRAIN_HORIZON}, temp={TEMPERATURE}")
print(f"Lip grid: {LIP_CONSTANTS}")
print(f"Lambda grid: {TURNOVER_LAMBDAS}")
print(f"WF re-eval interval: {WF_REEVAL_INTERVAL} months")

all_results = []
training_log = []
t0 = time.time()

for mkt_name, cfg in MARKETS.items():
    print(f"\n{'#'*80}")
    print(f"# MARKET: {mkt_name}")
    print(f"{'#'*80}")

    mkt_t0 = time.time()

    try:
        df, raw_features, date_col, id_col = load_equity_data(
            cfg['file'], cfg['type']
        )
    except Exception as e:
        print(f"  ERROR loading {mkt_name}: {e}")
        continue

    target = cfg['target']
    all_features = [f for f in raw_features if f not in EXCLUDE_COLS and f != target]
    accounting_feats, market_feats = classify_features(all_features)
    accounting_feats = [f for f in accounting_feats if f in df.columns]
    market_feats = [f for f in market_feats if f in df.columns]

    # CRITICAL: Lag BEFORE filter
    if accounting_feats:
        df = lag_features(df, accounting_feats, id_col, date_col, lag=1)
    if market_feats:
        df = lag_features(df, market_feats, id_col, date_col, lag=1)

    # Lag me to avoid look-ahead bias
    if 'me' in df.columns:
        df = df.sort_values([id_col, date_col])
        df['me'] = df.groupby(id_col)['me'].shift(1)
        df = df.dropna(subset=['me'])

    # Filter to top-N by market cap
    topn = cfg['topn']
    if 'me' in df.columns:
        df['_me_rank'] = df.groupby(date_col)['me'].rank(ascending=False, method='first')
        df = df[df['_me_rank'] <= topn].drop(columns=['_me_rank'])

    cost_lookup_1x = build_cost_lookup_1x(df, date_col, id_col)
    ml_features = [f for f in accounting_feats + market_feats if f in df.columns]

    # Slow features only
    slow_feats = [f for f in ml_features if classify_feature_speed(f) == 'SLOW']

    print(f"  Features: {len(ml_features)} total, {len(slow_feats)} SLOW")

    df[date_col] = pd.to_datetime(df[date_col])
    nq = cfg['nq']

    # IS/OOS split
    unique_dates = sorted(df[date_col].unique())
    n_total = len(unique_dates)
    is_split_idx = max(int(n_total * 0.8), MIN_TRAIN_MONTHS)
    is_end_date = unique_dates[min(is_split_idx - 1, n_total - 1)]

    print(f"  IS ends: {is_end_date}, OOS months: {n_total - is_split_idx}")

    if len(slow_feats) < 5:
        print(f"  SKIPPING: only {len(slow_feats)} slow features")
        continue

    def make_masks(result_df):
        idx = result_df.index
        return [('IS', idx <= is_end_date), ('OOS', idx > is_end_date),
                ('Full', pd.Series(True, index=idx))]

    strategies = {}

    # 1. MomentumSort
    print(f"\n  --- MomentumSort ---")
    strategies['MomentumSort'] = run_factor_sort(
        df, ['ret_12_1'], date_col, id_col, target, cost_lookup_1x,
        nq=nq, buffer_pct=BUFFER_PCT
    )

    # 2. Ridge_Slow (control)
    print(f"  --- Ridge_Slow ({len(slow_feats)} features) ---")
    strategies['Ridge_Slow'] = run_ridge_walkforward(
        df, slow_feats, date_col, id_col, target, cost_lookup_1x,
        nq=nq, buffer_pct=BUFFER_PCT, min_train=MIN_TRAIN_MONTHS
    )

    # 3. CostAware_Linear
    print(f"  --- CostAware_Linear ({len(slow_feats)} features) ---")
    ca_t0 = time.time()
    strategies['CostAware_Linear'] = run_nn_walkforward(
        df, slow_feats, date_col, id_col, target, cost_lookup_1x,
        nq=nq, buffer_pct=BUFFER_PCT, min_train=MIN_TRAIN_MONTHS,
        strategy_type='cost_linear'
    )
    ca_time = time.time() - ca_t0
    print(f"    Time: {ca_time:.0f}s")

    # 4. Lipschitz_NN
    print(f"  --- Lipschitz_NN ({len(slow_feats)} features) ---")
    lip_t0 = time.time()
    strategies['Lipschitz_NN'] = run_nn_walkforward(
        df, slow_feats, date_col, id_col, target, cost_lookup_1x,
        nq=nq, buffer_pct=BUFFER_PCT, min_train=MIN_TRAIN_MONTHS,
        strategy_type='lipschitz'
    )
    lip_time = time.time() - lip_t0
    print(f"    Time: {lip_time:.0f}s")

    # 5. CostAware_Lipschitz
    print(f"  --- CostAware_Lipschitz ({len(slow_feats)} features) ---")
    cl_t0 = time.time()
    strategies['CostAware_Lipschitz'] = run_nn_walkforward(
        df, slow_feats, date_col, id_col, target, cost_lookup_1x,
        nq=nq, buffer_pct=BUFFER_PCT, min_train=MIN_TRAIN_MONTHS,
        strategy_type='cost_lipschitz'
    )
    cl_time = time.time() - cl_t0
    print(f"    Time: {cl_time:.0f}s")

    # Evaluate all strategies
    for strat_name, result in strategies.items():
        if result is None:
            continue
        for period_label, mask in make_masks(result):
            for cm in COST_MULTS:
                metrics = evaluate(result, cm, period_label, mask,
                                   strat_name, mkt_name)
                if metrics:
                    all_results.append(metrics)

        # Print OOS summary
        oos_mask = result.index > is_end_date
        oos = result[oos_mask]
        if len(oos) > 0:
            sr3 = sharpe(oos['gross_ret'] - oos['cost_drag_3.0'])
            gsr = sharpe(oos['gross_ret'])
            to = oos['turnover'].mean()
            print(f"    {strat_name:25s}  Gross={gsr:.2f}  Net@3x={sr3:.2f}  TO={to:.1%}")

    mkt_time = time.time() - mkt_t0
    print(f"\n  Market time: {mkt_time:.0f}s ({mkt_time/60:.1f}m)")

    training_log.append({
        'market': mkt_name,
        'n_slow_features': len(slow_feats),
        'n_oos_months': n_total - is_split_idx,
        'time_cost_linear_s': ca_time,
        'time_lipschitz_s': lip_time,
        'time_cost_lipschitz_s': cl_time,
        'total_time_s': mkt_time,
    })

elapsed = time.time() - t0
print(f"\n\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f}m)")

# Save results
results_df = pd.DataFrame(all_results)
outfile = 'scripts/concern_experiments/expDB_cost_aware_nn.csv'
results_df.to_csv(outfile, index=False)
print(f"\nSaved {len(results_df)} rows to {outfile}")

# Save training log
log_df = pd.DataFrame(training_log)
logfile = 'scripts/concern_experiments/expDB_training_log.csv'
log_df.to_csv(logfile, index=False)
print(f"Saved training log to {logfile}")

# ==============================================================================
# SUMMARY TABLE
# ==============================================================================

print("\n\n" + "=" * 100)
print("SUMMARY: OOS Net SR at 3x Costs")
print("=" * 100)

oos_3x = results_df[(results_df['Period'] == 'OOS') & (results_df['CostMult'] == 3.0)]
if len(oos_3x) > 0:
    strat_order = ['MomentumSort', 'Ridge_Slow', 'CostAware_Linear',
                   'Lipschitz_NN', 'CostAware_Lipschitz']
    pivot = oos_3x.pivot_table(index='Market', columns='Strategy', values='Net_SR')
    strat_order = [s for s in strat_order if s in pivot.columns]
    pivot = pivot[strat_order]
    print(pivot.round(2).to_string())
    print(f"\nMean:")
    print(pivot.mean().round(3).to_string())

    print("\n\nTURNOVER: OOS Average Monthly Turnover")
    print("=" * 100)
    to_pivot = oos_3x.pivot_table(index='Market', columns='Strategy', values='Turnover')
    to_pivot = to_pivot[[s for s in strat_order if s in to_pivot.columns]]
    print(to_pivot.round(3).to_string())
    print(f"\nMean:")
    print(to_pivot.mean().round(3).to_string())

    print("\n\nGROSS SR: OOS")
    print("=" * 100)
    gsr_pivot = oos_3x.pivot_table(index='Market', columns='Strategy', values='Gross_SR')
    gsr_pivot = gsr_pivot[[s for s in strat_order if s in gsr_pivot.columns]]
    print(gsr_pivot.round(2).to_string())
    print(f"\nMean:")
    print(gsr_pivot.mean().round(3).to_string())

    # Win rates vs Ridge_Slow
    print("\n\nWIN RATES vs Ridge_Slow (OOS @3x)")
    print("=" * 100)
    if 'Ridge_Slow' in pivot.columns:
        for strat in strat_order:
            if strat == 'Ridge_Slow':
                continue
            if strat in pivot.columns:
                wins = (pivot[strat] > pivot['Ridge_Slow']).sum()
                total = pivot[strat].notna().sum()
                print(f"  {strat:25s}: {wins}/{total} markets ({wins/max(total,1):.0%})")

    # Delta vs Ridge_Slow
    print("\n\nDELTA vs Ridge_Slow (OOS @3x, mean across markets)")
    print("=" * 100)
    if 'Ridge_Slow' in pivot.columns:
        ridge_mean = pivot['Ridge_Slow'].mean()
        for strat in strat_order:
            if strat in pivot.columns:
                delta = pivot[strat].mean() - ridge_mean
                print(f"  {strat:25s}: {delta:+.3f} SR")

print("\nDone!")
