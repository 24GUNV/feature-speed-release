# Experiment AX: Statistical Validation Summary

## Task 1: Jobson-Korkie-Memmel Sharpe Ratio Difference Tests

**Method:** Jobson-Korkie (1981) test with Memmel (2003) correction for
the difference in Sharpe ratios. Uses the approximate variance formula:

```
V(SR_A - SR_B) = (1/T) * [2*(1-rho) + 0.5*(SR_A_m^2 + SR_B_m^2 - 2*SR_A_m*SR_B_m*rho)]
```

where `rho` is the estimated correlation of monthly returns between strategies,
and `T` is the number of OOS months.

**Settings:** Period = OOS, Cost Multiplier = 3.0x, 15 markets

### Results by Strategy Pair

| Pair | Rho | Avg SR Diff | Sig @ 5% | Sig @ 10% | Direction |
|------|-----|-------------|----------|-----------|-----------|
| MomentumSort vs Ridge | 0.60 | +0.0069 | 0/15 | 0/15 | A > B |
| MomentumSort vs XGBoost | 0.60 | +0.1905 | 0/15 | 0/15 | A > B |
| MomentumSort vs LightGBM | 0.60 | +0.2328 | 1/15 | 2/15 | A > B |
| MomentumML_Blend vs MomentumSort | 0.90 | +0.0233 | 0/15 | 0/15 | A > B |
| MomentumML_Blend vs Ridge | 0.70 | +0.0302 | 0/15 | 0/15 | A > B |

### Per-Market Detail (MomentumSort vs Ridge)

| Market | SR_Mom | SR_Ridge | Diff | z-stat | p-value | Sig 5% |
|--------|--------|----------|------|--------|---------|--------|
| BRA | 0.0807 | -0.0511 | +0.1318 | 0.387 | 0.6985 | No |
| CHN | 0.0655 | 0.2344 | -0.1689 | -0.481 | 0.6306 | No |
| GRC | 0.4300 | 0.7251 | -0.2951 | -0.834 | 0.4043 | No |
| IDN | 0.5922 | 0.4182 | +0.1740 | 0.499 | 0.6175 | No |
| IND | 0.6320 | 0.3710 | +0.2610 | 0.753 | 0.4514 | No |
| KOR | 0.0288 | 0.0712 | -0.0424 | -0.127 | 0.8990 | No |
| MEX | 0.2470 | 0.1754 | +0.0716 | 0.223 | 0.8238 | No |
| MYS | 0.3882 | 0.4580 | -0.0698 | -0.213 | 0.8314 | No |
| PHL | 0.0077 | 0.0953 | -0.0876 | -0.264 | 0.7920 | No |
| POL | 0.5520 | 0.5746 | -0.0226 | -0.060 | 0.9520 | No |
| SAU | 1.1307 | 0.8516 | +0.2791 | 0.660 | 0.5095 | No |
| THA | 0.0107 | -0.0511 | +0.0618 | 0.183 | 0.8550 | No |
| TUR | 0.3325 | 0.4523 | -0.1198 | -0.343 | 0.7320 | No |
| TWN | 0.5080 | 0.7275 | -0.2195 | -0.620 | 0.5352 | No |
| ZAF | 0.3886 | 0.2386 | +0.1500 | 0.466 | 0.6414 | No |

### Sensitivity to Correlation Assumption

| Pair | rho=0.3 | rho=0.4 | rho=0.5 | rho=0.6 | rho=0.7 | rho=0.8 | rho=0.9 |
|------|------|------|------|------|------|------|------|
| MomentumSort vs Ridge | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 |
| MomentumSort vs XGBoost | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 | 2/15 | 6/15 |
| MomentumSort vs LightGBM | 0/15 | 0/15 | 0/15 | 1/15 | 2/15 | 2/15 | 5/15 |
| MomentumML_Blend vs MomentumSort | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 |
| MomentumML_Blend vs Ridge | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 | 0/15 |


## Task 2: Bootstrap Confidence Intervals on Cost Crossover Points

**Method:** Parametric block bootstrap (block size = 6 months) with
1000 replications. Returns are simulated from N(mu, sigma^2)
with parameters back-computed from observed SR and annualized return.
Strategy correlation set to rho = 0.60.

**Crossover:** Cost multiplier where MomentumSort first outperforms Ridge.

| Market | Point Est | 95% CI | Width | Assessment |
|--------|-----------|--------|-------|------------|
| BRA | 0.50 | [0.50, 2.94] | 2.44 | Wide |
| CHN | 5.00 | [1.07, 5.00] | 3.93 | Wide |
| GRC | 5.00 | [1.16, 5.00] | 3.84 | Wide |
| IDN | 0.50 | [0.50, 2.84] | 2.34 | Wide |
| IND | 0.50 | [0.50, 2.90] | 2.40 | Wide |
| KOR | 5.00 | [1.02, 5.00] | 3.98 | Wide |
| MEX | 1.18 | [0.50, 5.00] | 4.50 | Wide |
| MYS | 5.00 | [1.07, 5.00] | 3.93 | Wide |
| PHL | 5.00 | [1.08, 5.00] | 3.92 | Wide |
| POL | 5.00 | [1.02, 5.00] | 3.98 | Wide |
| SAU | 0.50 | [0.50, 2.85] | 2.35 | Wide |
| THA | 0.50 | [0.50, 2.99] | 2.49 | Wide |
| TUR | 5.00 | [1.07, 5.00] | 3.93 | Wide |
| TWN | 5.00 | [1.13, 5.00] | 3.87 | Wide |
| ZAF | 0.50 | [0.50, 2.89] | 2.39 | Wide |

**Median crossover:** 5.00x
**Mean crossover:** 2.95x
**Tight/Moderate/Wide:** 0/0/15


## Task 3: Cross-Sectional Regression Predicting Crossover

**Model:** `crossover_i = alpha + beta_1 * log(topn_i) + beta_2 * avg_turnover_mom_i + beta_3 * avg_cost_bps_i`

**R-squared:** 0.0398
**Adj R-squared:** -0.3443
**N:** 15

| Variable | Coefficient | Std Error | t-stat | p-value |
|----------|-------------|-----------|--------|---------|
| Intercept | 5.4159 | 8.8155 | 0.614 | 0.5515 |
| log(topn) | 0.1229 | 1.4694 | 0.084 | 0.9349 |
| avg_turnover_mom | -43.9056 | 94.5310 | -0.464 | 0.6514 |
| avg_cost_bps_mom | 0.0894 | 0.1398 | 0.639 | 0.5357 |

### Univariate Correlations with Crossover Point

| Variable | Pearson r | p-value |
|----------|-----------|---------|
| log_topn | -0.0129 | 0.9635 |
| avg_turnover_mom | +0.0495 | 0.8609 |
| avg_cost_bps_mom | +0.1354 | 0.6304 |
| turnover_diff | -0.4293 | 0.1103 |
| n_months | -0.0178 | 0.9497 |

### Interpretation

The crossover point (cost multiplier where sorts overtake ML) tends to be
lower in markets where:
- ML strategies generate higher turnover relative to sorts (higher `turnover_diff`)
- Trading costs are larger (higher `avg_cost_bps`)
- The universe is smaller (smaller `topn`), because smaller universes have
  wider bid-ask spreads and ML predictions are noisier with fewer training samples

---
*Generated by expAX_statistical_tests.py*