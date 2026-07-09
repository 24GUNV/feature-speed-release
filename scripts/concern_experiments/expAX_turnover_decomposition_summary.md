# Experiment AX: Factor Turnover Decomposition

## Research Question
WHY do ML models generate ~2.7x the turnover of simple factor sorts in EM equity?

## Markets Tested
- THA, KOR, TWN (3 representative EM markets)
- OOS: 2021-01-01 to 2023-12-31, 3x transaction costs, 20% buffer

## Feature Speed Classification
- **SLOW**: 57 features
- **MEDIUM**: 56 features
- **FAST**: 38 features

## Empirical Feature Speed (Rank Autocorrelation)

| Speed | Mean AC | Median AC |
|-------|---------|-----------|
| SLOW | 0.778 | 0.786 |
| MEDIUM | 0.705 | 0.799 |
| FAST | 0.436 | 0.571 |

## Portfolio Stability Comparison

| Strategy | Avg Turnover | Avg Jaccard | Cost Drag (bps/yr) | Net SR |
|----------|-------------|------------|-------------------|--------|
| LowVol_Sort | 4.6% | 0.914 | 88 | 0.317 |
| Value_Sort | 4.7% | 0.912 | 116 | 0.448 |
| Quality_Sort | 8.1% | 0.861 | 230 | -0.009 |
| Composite_Sort | 11.5% | 0.803 | 319 | 0.143 |
| Momentum_Sort | 17.2% | 0.713 | 533 | 0.185 |
| Ridge | 35.3% | 0.487 | 958 | 0.051 |
| XGBoost | 57.3% | 0.284 | 1609 | -0.385 |

## Feature Importance by Speed Category

### RIDGE
- SLOW: 33.6%
- MEDIUM: 23.4%
- FAST: 43.0%

### XGBOOST
- SLOW: 31.1%
- MEDIUM: 22.1%
- FAST: 46.8%

## Ridge Prediction-Change Decomposition

- SLOW features drive 12.6% of month-to-month prediction changes
- MEDIUM features drive 24.1% of month-to-month prediction changes
- FAST features drive 63.4% of month-to-month prediction changes

## Feature Selection Instability (XGBoost)

- **THA**: Adjacent-window Jaccard = 0.409 (std=0.110)
- **KOR**: Adjacent-window Jaccard = 0.484 (std=0.099)
- **TWN**: Adjacent-window Jaccard = 0.474 (std=0.109)

## Key Finding

ML generates excess turnover through a three-part mechanism:

1. **Feature Overweighting**: ML allocates disproportionate importance to FAST 
   (market-based) features like momentum, volatility, and liquidity because they 
   have higher short-term predictive signal-to-noise ratios.

2. **Prediction Instability**: FAST features change substantially every month 
   (low cross-sectional rank autocorrelation), causing cross-sectional stock 
   rankings to shift rapidly. Ridge decomposition shows FAST features drive 
   the majority of month-to-month prediction changes.

3. **Cost Amplification**: The resulting portfolio churn (low Jaccard similarity) 
   generates 2-3x the turnover of simple sorts, which at 3x transaction costs 
   in EM markets erases any gross-of-cost advantage ML may have.

This is not a fixable bug -- it is a structural consequence of the features 
ML naturally selects. The tension between predictive power and tradability 
is fundamental in emerging markets.