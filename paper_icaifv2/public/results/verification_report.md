# ICAIF v2 Verification Report

Generated: 2026-07-07T14:22:50

## Summary

- PASS: 99
- WARN: 0
- FAIL: 0

## Checks

| Status | Severity | Check | Detail |
|---|---|---|---|
| PASS | blocking | Main comparison net SRs match expAX CSV | CompositeSort=0.324, LightGBM=0.127, MomentumML_Blend=0.383, MomentumSort=0.360, Ridge=0.353, XGBoost=0.169 |
| PASS | blocking | Ridge market wins vs MomentumSort match paper | Ridge wins 8/15 markets at OOS 3x |
| PASS | blocking | Feature-speed net SRs match expAH CSV | MomentumSort=0.360, MomentumSort_Q=0.373, Ridge_All=0.353, Ridge_All_Q=0.481, Ridge_Fast=0.208, Ridge_Slow=0.472, Ridge_SlowMed=0.393, Ridge_Slow_Q=0.488 |
| PASS | blocking | Feature-speed turnovers match expAH CSV | MomentumSort=0.170, Ridge_All=0.233, Ridge_Fast=0.301, Ridge_Slow=0.102 |
| PASS | blocking | Persistence-filter net SRs match expAN CSV | MomentumSort=0.360, Ridge_AdhocSlow=0.472, Ridge_All=0.353, Ridge_EmpSlow_FS=0.470, Ridge_PersFiltered=0.472 |
| PASS | blocking | EMA smoothing net SRs match expAR CSV | MomentumSort=0.360, Ridge_EMA_0.35=0.474, Ridge_WF_EMA=0.451 |
| PASS | blocking | EMA smoothing turnover matches expAR CSV | Ridge_EMA_0.35=0.097, Ridge_WF_EMA=0.103 |
| PASS | blocking | Aggregation-vs-speed N=5 turnover matches expAI CSV (5.1% vs 5.6%) | Ridge_5=0.051, CompositeSort_5=0.056 |
| PASS | blocking | IC table means match expAK CSV | MomentumSort=0.031, Ridge_All=0.022, Ridge_Slow=0.031, Ridge_Fast=0.013 |
| PASS | blocking | IC table stabilities match expAK CSV | MomentumSort=0.894, Ridge_All=0.665, Ridge_Slow=0.939, Ridge_Fast=0.539 |
| PASS | blocking | Ridge_Slow has the highest IC x stability | MomentumSort=0.0278, Ridge_All=0.0147, Ridge_Fast=0.0068, Ridge_Slow=0.0295 |
| PASS | blocking | Model-free IC decay h=1 ICs match expDG CSV | SLOW=0.027, FAST=0.025 |
| PASS | blocking | Model-free IC retention at h=12 matches paper claim (69% vs 57%) | SLOW=0.693, FAST=0.566 |
| PASS | blocking | Tree smoothing net SRs match expCP CSV | LGBM_All=0.265, Ridge_All=0.357, Ridge_Slow=0.479, XGB_All=0.300, XGB_Slow=0.263 |
| PASS | blocking | Tree EMA net SRs match expDF CSV | XGB_EMA_0.5=0.526 |
| PASS | blocking | Tree EMA gross SR and turnover match expDF CSV | XGB_EMA_0.5_gross=0.911, XGB_EMA_0.5_turnover=0.261 |
| PASS | blocking | Bound in-sample R2 matches expDE CSV | got 0.7239, expected 0.7240 |
| PASS | blocking | Bound LOO R2 matches expDE CSV | got 0.5955, expected 0.5960 |
| PASS | blocking | Bound leave-5-out R2 matches expDE CSV | got 0.6879, expected 0.6880 |
| PASS | blocking | Universal slow feature count matches expDE CSV | universal_slow_count=52 |
| PASS | blocking | Generated bound uses equal-weight feature-set FSI | fsi_definition=equal_weight_feature_set |
| PASS | blocking | Generated bound has 300 observations | points=300, summary_n=300 |
| PASS | blocking | Generated bound R2 matches paper claim | got 0.7239, expected 0.7240 |
| PASS | blocking | Generated bound lambda matches paper claim | got 0.3925, expected 0.3920 |
| PASS | blocking | Generated per-kappa lambda mean matches paper claim | got 0.3344, expected 0.3340 |
| PASS | blocking | Generated per-kappa lambda is stable | max-min lambda spread=0.0016 |
| PASS | blocking | Generated frontier includes required paper points | missing=[] |
| PASS | blocking | Generated frontier turnover units are fractional before plotting | max raw turnover=0.3553 |
| PASS | blocking | Frontier Ridge_Slow net SR matches generated point | got 0.4720, expected 0.4720 |
| PASS | blocking | Frontier Ridge_Slow_Q net SR matches generated point | got 0.4881, expected 0.4880 |
| PASS | blocking | Frontier WF-EMA net SR matches generated point | got 0.4513, expected 0.4510 |
| PASS | blocking | Pooled bootstrap one-sided p matches paper (0.10) | got 0.0991, expected 0.0990 |
| PASS | blocking | Pooled bootstrap CI low matches paper (-0.08) | got -0.0825, expected -0.0800 |
| PASS | blocking | Pooled bootstrap CI high matches paper (0.51) | got 0.5122, expected 0.5100 |
| PASS | blocking | Bootstrap fraction positive matches paper (90%) | got 0.9009, expected 0.9000 |
| PASS | blocking | Ridge_Slow market wins match paper (11/15) | market_wins=11 |
| PASS | blocking | Pooled t-stat matches paper (0.77) | got 0.7673, expected 0.7700 |
| PASS | blocking | Effect size d matches paper (0.08) | got 0.0796, expected 0.0800 |
| PASS | blocking | Short-only net SRs match expAY CSV | MomentumSort=-0.436, Ridge=-0.764, XGBoost=-0.975 |
| PASS | blocking | Short-only net SR negative for every strategy | all short-only strategies negative at 3x + borrow |
| PASS | blocking | Feature-speed decomposition summary matches Table 3 (lag=1 rerun) | Ridge 43.0%/XGB 46.8% FAST importance, 63.4% pred changes |
| PASS | blocking | Cross-sectional R2 of degradation~FSI at kappa_1 (paper: 64-65%) | got 0.6381, expected 0.6380 |
| PASS | blocking | Cross-sectional R2 of degradation~FSI at kappa_1.5 (paper: 64-65%) | got 0.6397, expected 0.6400 |
| PASS | blocking | Cross-sectional R2 of degradation~FSI at kappa_2 (paper: 64-65%) | got 0.6443, expected 0.6440 |
| PASS | blocking | Cross-sectional R2 of degradation~FSI at kappa_3 (paper: 64-65%) | got 0.6537, expected 0.6540 |
| PASS | blocking | Per-kappa slope/kappa at kappa_1 matches lambda=0.33 convention | got 0.3338, expected 0.3340 |
| PASS | blocking | Per-kappa slope/kappa at kappa_1.5 matches lambda=0.33 convention | got 0.3345, expected 0.3340 |
| PASS | blocking | Per-kappa slope/kappa at kappa_2 matches lambda=0.33 convention | got 0.3353, expected 0.3340 |
| PASS | blocking | Per-kappa slope/kappa at kappa_3 matches lambda=0.33 convention | got 0.3338, expected 0.3340 |
| PASS | blocking | Pooled R2 on kappa alone (paper: 37%) | got 0.3715, expected 0.3720 |
| PASS | blocking | Pooled R2 on kappa*FSI (paper: 0.72) | got 0.7239, expected 0.7240 |
| PASS | blocking | R2 of realized turnover on FSI (paper: 0.73) | got 0.7268, expected 0.7270 |
| PASS | blocking | Partial corr(FSI, degradation | turnover) at kappa=3 (paper: ~0.25) | got 0.2628, expected 0.2630 |
| PASS | blocking | Within-strategy R2 at kappa=3 (paper: 0.07 -- strategy-level resolution) | got 0.0648, expected 0.0650 |
| PASS | blocking | Table 3 / Sec 4.1 class rank AC matches expDA diagnostic (0.97/0.83/0.68) | SLOW=0.965, MEDIUM=0.830, FAST=0.675 |
| PASS | blocking | FSI coefficient with spread control at kappa=3 (paper: unchanged) | got 1.0017, expected 1.0020 |
| PASS | blocking | Two-channel pooled R2 with non-smoothness indicator (paper: 0.73) | got 0.7340, expected 0.7340 |
| PASS | blocking | Two-channel non-smoothness coefficient (paper: +0.22 SR) | got 0.2233, expected 0.2230 |
| PASS | blocking | Lambda cluster-market CI low (paper: 0.27) | got 0.2656, expected 0.2660 |
| PASS | blocking | Lambda cluster-market CI high (paper: 0.40) | got 0.4020, expected 0.4020 |
| PASS | blocking | Lambda cluster-strategy CI low (paper: 0.26) | got 0.2564, expected 0.2560 |
| PASS | blocking | Lambda cluster-strategy CI high (paper: 0.41) | got 0.4111, expected 0.4110 |
| PASS | blocking | Joint sign-flip p across eight interventions (paper: 0.008) | got 0.0079, expected 0.0079 |
| PASS | blocking | Joint mean SR advantage over MomSort (paper: +0.11) | got 0.1119, expected 0.1120 |
| PASS | blocking | Eight interventions enter the joint test | 8 interventions in expDJ CSV |
| PASS | blocking | Every intervention beats MomSort on mean SR | all eight mean SR diffs positive |
| PASS | blocking | Ex-GRC Ridge_Slow net SR (paper: 0.45) | got 0.4526, expected 0.4530 |
| PASS | blocking | Ex-GRC MomentumSort net SR (paper: 0.36) | got 0.3546, expected 0.3550 |
| PASS | blocking | Ex-GRC frontier band matches paper (0.42-0.47) | band [0.424, 0.470] |
| PASS | blocking | Full-OOS XGBoost net SR at 3x (paper: 0.24) | got 0.2368, expected 0.2370 |
| PASS | blocking | Full-OOS XGB walk-forward EMA net SR at 3x (paper: 0.35) | got 0.3505, expected 0.3510 |
| PASS | blocking | Full-OOS best post-hoc fixed alpha=0.2 net SR (paper: 0.45) | got 0.4463, expected 0.4460 |
| PASS | blocking | Full-OOS XGB turnover reduction 36%->25% (WF EMA) | got 0.2462, expected 0.2460 |
| PASS | blocking | Best fixed EMA alpha on full OOS is 0.2 (post hoc) | best fixed grid point: XGB_EMA_0.2 (0.446) |
| PASS | blocking | expDM MomentumSort exactly reproduces expAX (pipeline fidelity) | expDM 0.3596 vs expAX 0.3596 |
| PASS | blocking | Paper discloses post-2020 tree-EMA 0.53 does not replicate on full OOS | non-replication disclosed and stale 0.53 claim removed |
| PASS | blocking | Value-weighted Ridge_Slow net SR at 3x (paper: 0.40) | got 0.3999, expected 0.4000 |
| PASS | blocking | Value-weighted MomentumSort net SR at 3x (paper: 0.31) | got 0.3051, expected 0.3050 |
| PASS | blocking | Value-weighted Ridge_All net SR at 3x (paper: 0.30) | got 0.2982, expected 0.2980 |
| PASS | blocking | Value-weighted Ridge_Slow wins 10/15, positive 14/15 | wins 10/15, positive 14/15 |
| PASS | blocking | Value-weighted hierarchy Ridge_Slow > MomSort > Ridge_All holds | Ridge_Slow 0.400 > MomSort 0.305 > Ridge_All 0.298 |
| PASS | blocking | US validation table net SRs match expCM CSV | MomentumSort@0x=0.775, MomentumSort@1x=0.693, MomentumSort@2x=0.612, MomentumSort@3x=0.530, Ridge_All@0x=0.580, Ridge_All@1x=0.445, Ridge_All@2x=0.310, Ridge_All@3x=0.175, Ridge_Slow@0x=0.638, Ridge_Slow@1x=0.596, Ridge_Slow@2x=0.553, Ridge_Slow@3x=0.510, Ridge_Fast@0x=0.404, Ridge_Fast@1x=0.206, Ridge_Fast@2x=0.008, Ridge_Fast@3x=-0.189 |
| PASS | blocking | expAX_full_comparison.py excludes market cap from predictors | EXCLUDE_COLS contains me and market_equity |
| PASS | blocking | expAX_full_comparison.py lags market cap before universe filter | market cap is shifted one month before top-N filtering |
| PASS | blocking | expAX_full_comparison.py uses past-only walk-forward training dates | train_dates are strictly before the current test date |
| PASS | blocking | expAX_full_comparison.py fits preprocessing inside the walk-forward loop | imputer is fit on training data inside walk-forward execution |
| PASS | blocking | expAH_feature_speed_test.py excludes market cap from predictors | EXCLUDE_COLS contains me and market_equity |
| PASS | blocking | expAH_feature_speed_test.py lags market cap before universe filter | market cap is shifted one month before top-N filtering |
| PASS | blocking | expAH_feature_speed_test.py uses past-only walk-forward training dates | train_dates are strictly before the current test date |
| PASS | blocking | expAH_feature_speed_test.py fits preprocessing inside the walk-forward loop | imputer is fit on training data inside walk-forward execution |
| PASS | blocking | Bound figure FSI formula matches paper text | script and paper both use equal-weight feature-set FSI |
| PASS | advisory | Bound figure header is current | header does not contain stale R2=0.95 claim |
| PASS | blocking | Frontier figure does not double-scale tree turnover | turnover is stored as fractions and scaled only at plotting time |
| PASS | blocking | Frontier figure is generated from CSVs rather than hard-coded paper values | frontier script reads paper points from canonical CSVs |
| PASS | advisory | Tree EMA post-2020 OOS window is labeled in paper | paper labels XGB_EMA as a separate post-2020 validation sample |
| PASS | advisory | WF-EMA alpha-selection wording matches implementation | paper states no future OOS months and script notes this is not nested OOF validation |
| PASS | advisory | EMA script lag header matches implementation | header documents lag=1 on pre-lagged JKP/GFD accounting fields |
| PASS | blocking | Stale paper results summary is removed from evidence path | paper_icaifv2/results_summary.md points to the public verification outputs |
| PASS | blocking | Statistical-test summary matches current v6 outputs | statistical summary was regenerated from current expAX CSV |

## Computed Claim Tables

### expAX_oos_3x

| Item | Gross_SR | Net_SR | Turnover |
|---|---|---|---|
| CompositeSort | 0.5565 | 0.3238 | 0.1673 |
| LightGBM | 0.6375 | 0.1269 | 0.3824 |
| LowVolSort | 0.4400 | -0.0758 | 0.4428 |
| MomentumML_Blend | 0.6334 | 0.3829 | 0.1900 |
| MomentumSort | 0.5816 | 0.3596 | 0.1699 |
| Ridge | 0.6824 | 0.3527 | 0.2332 |
| XGBoost | 0.6571 | 0.1691 | 0.3632 |


### expAH_oos_3x

| Item | Net_SR | Turnover |
|---|---|---|
| MomentumSort | 0.3596 | 0.1699 |
| MomentumSort_Q | 0.3734 | 0.1200 |
| Ridge_All | 0.3527 | 0.2332 |
| Ridge_All_Q | 0.4815 | 0.1129 |
| Ridge_Fast | 0.2082 | 0.3006 |
| Ridge_Slow | 0.4720 | 0.1016 |
| Ridge_SlowMed | 0.3929 | 0.1451 |
| Ridge_Slow_Q | 0.4881 | 0.0734 |


### expAN_oos_3x_net_sr

- `MomentumSort`: `0.3596`
- `Ridge_AdhocSlow`: `0.472`
- `Ridge_All`: `0.3527`
- `Ridge_EmpSlow_FS`: `0.4695`
- `Ridge_PersFiltered`: `0.4723`


### expAR_oos_3x

| Item | Net_SR | Turnover |
|---|---|---|
| MomentumSort | 0.3596 | 0.1699 |
| Ridge_EMA_0.05 | 0.3542 | 0.0520 |
| Ridge_EMA_0.1 | 0.4113 | 0.0591 |
| Ridge_EMA_0.15 | 0.4140 | 0.0654 |
| Ridge_EMA_0.2 | 0.4336 | 0.0731 |
| Ridge_EMA_0.25 | 0.4626 | 0.0801 |
| Ridge_EMA_0.3 | 0.4637 | 0.0887 |
| Ridge_EMA_0.35 | 0.4741 | 0.0971 |
| Ridge_EMA_0.4 | 0.4696 | 0.1064 |
| Ridge_EMA_0.45 | 0.4669 | 0.1150 |
| Ridge_EMA_0.5 | 0.4738 | 0.1231 |
| Ridge_EMA_0.6 | 0.4516 | 0.1423 |
| Ridge_EMA_0.7 | 0.4471 | 0.1624 |
| Ridge_EMA_0.8 | 0.4382 | 0.1858 |
| Ridge_EMA_0.9 | 0.3939 | 0.2122 |
| Ridge_EMA_1.0 | 0.3527 | 0.2332 |
| Ridge_WF_EMA | 0.4513 | 0.1028 |


### expAI_oos_3x_turnover

- `CompositeSort_10`: `0.0576`
- `CompositeSort_151`: `0.1395`
- `CompositeSort_20`: `0.0698`
- `CompositeSort_5`: `0.0556`
- `CompositeSort_50`: `0.0681`
- `MomentumSort`: `0.1699`
- `Ridge_10`: `0.0558`
- `Ridge_20`: `0.0761`
- `Ridge_5`: `0.0507`
- `Ridge_50`: `0.0817`
- `Ridge_All`: `0.2332`
- `Sort_be_me`: `0.0453`
- `Sort_f_score`: `0.125`
- `Sort_gp_at`: `0.0372`
- `Sort_ivol_capm_21d`: `0.4428`
- `Sort_oaccruals_at`: `0.0895`
- `Sort_ret_12_1`: `0.1699`
- `Sort_ret_3_1`: `0.4898`
- `Sort_ret_6_1`: `0.2752`
- `Sort_rvol_21d`: `0.4299`


### expAK_ic

| Item | Mean_IC | Pred_Stability |
|---|---|---|
| CompositeSort | 0.0104 | 0.8452 |
| MomentumSort | 0.0311 | 0.8941 |
| Ridge_All | 0.0221 | 0.6653 |
| Ridge_Fast | 0.0126 | 0.5385 |
| Ridge_Slow | 0.0314 | 0.9394 |


### expDG_ic_decay

| Item | FAST | MEDIUM | SLOW |
|---|---|---|---|
| h1_ic | 0.0248 | 0.0129 | 0.0268 |
| retention_h12 | 0.5659 | 0.6447 | 0.6933 |


### expCP_3x

| Item | Gross_SR | Net_SR | Turnover |
|---|---|---|---|
| LGBM_All | 0.7475 | 0.2652 | 0.3553 |
| LGBM_AllSmoothed | 0.7057 | 0.3079 | 0.2866 |
| LGBM_Slow | 0.6569 | 0.2412 | 0.3086 |
| MomentumSort | 0.6057 | 0.3731 | 0.1721 |
| Ridge_All | 0.6836 | 0.3575 | 0.2301 |
| Ridge_AllSmoothed | 0.6162 | 0.4679 | 0.1036 |
| Ridge_Slow | 0.6159 | 0.4793 | 0.0991 |
| XGB_All | 0.7670 | 0.3003 | 0.3455 |
| XGB_AllSmoothed | 0.7394 | 0.3478 | 0.2788 |
| XGB_Slow | 0.6475 | 0.2629 | 0.2865 |


### expDF_oos_3x

| Item | Gross_SR | Net_SR | Turnover |
|---|---|---|---|
| LGBM_EMA_0.2 | 0.5758 | 0.2083 | 0.3446 |
| LGBM_EMA_0.35 | 0.7859 | 0.4123 | 0.3353 |
| LGBM_EMA_0.5 | 0.8095 | 0.3184 | 0.3846 |
| LGBM_EMA_0.7 | 0.7819 | 0.1640 | 0.4529 |
| LightGBM | 0.7161 | -0.0917 | 0.5407 |
| Ridge_Slow | 0.6283 | 0.4791 | 0.0884 |
| XGB_EMA_0.2 | 0.6923 | 0.4430 | 0.1807 |
| XGB_EMA_0.35 | 0.8029 | 0.4974 | 0.2172 |
| XGB_EMA_0.5 | 0.9106 | 0.5265 | 0.2607 |
| XGB_EMA_0.7 | 0.8590 | 0.3686 | 0.3356 |
| XGBoost | 0.7430 | 0.0875 | 0.4446 |


### expDE_transfer

- `boot_lambda_hi`: `0.44760305898094455`
- `boot_lambda_lo`: `0.3395443645038035`
- `boot_r2_hi`: `0.7981768034585894`
- `boot_r2_lo`: `0.6646768279370578`
- `in_sample_lambda`: `0.39245769604624603`
- `in_sample_r2`: `0.7238699193991565`
- `l5o_5pct_r2`: `0.5407595299735881`
- `l5o_95pct_r2`: `0.786325773835365`
- `l5o_mean_r2`: `0.6879266578378853`
- `l5o_median_r2`: `0.7001578830010777`
- `loo_lambda_std`: `0.007913868912094583`
- `loo_mean_r2`: `0.5955322223541323`
- `slow_jaccard_mean`: `0.971086446274416`
- `universal_slow_count`: `52`


### generated_bound_summary

- `fsi_definition`: `equal_weight_feature_set`
- `lambda`: `0.3925`
- `n`: `300`
- `per_kappa_lambda_mean`: `0.3344`
- `per_kappa_lambda_std`: `0.0007`
- `r2`: `0.7239`


### generated_frontier_points

| Item | Net_SR | Source | Turnover |
|---|---|---|---|
| Blend | 0.3829 | expAX | 0.1900 |
| CostAware_Linear | 0.3665 | expDB | 0.0714 |
| CostAware_Lip | 0.3422 | expDB | 0.1259 |
| LGBM_All | 0.2652 | expCP | 0.3553 |
| LGBM_Slow | 0.2412 | expCP | 0.3086 |
| Lipschitz_NN | 0.0663 | expDB | 0.3337 |
| MomentumSort | 0.3596 | expAH | 0.1699 |
| PersistFilter | 0.4723 | expAN | 0.1333 |
| Ridge_All | 0.3527 | expAH | 0.2332 |
| Ridge_All_Q | 0.4815 | expAH | 0.1129 |
| Ridge_Fast | 0.2082 | expAH | 0.3006 |
| Ridge_Slow | 0.4720 | expAH | 0.1016 |
| Ridge_SlowMed | 0.3929 | expAH | 0.1451 |
| Ridge_Slow_Q | 0.4881 | expAH | 0.0734 |
| SpeedTercile | 0.4695 | expAN | 0.1337 |
| WF-EMA | 0.4513 | expAR | 0.1028 |
| XGB_All | 0.3003 | expCP | 0.3455 |
| XGB_Slow | 0.2629 | expCP | 0.2865 |


### expDH_pooled_bootstrap

- `block`: `6.0`
- `boot_ci_hi`: `0.5121772823425597`
- `boot_ci_lo`: `-0.0824885965096536`
- `boot_frac_positive`: `0.9009`
- `boot_p_onesided`: `0.0991`
- `effect_size_d`: `0.07956174061962806`
- `market_wins`: `11.0`
- `months_for_80pct_power`: `976.6964148718323`
- `n_markets`: `15.0`
- `obs_sr_diff`: `0.1690097465614379`
- `p_t_onesided`: `0.22244410792849534`
- `pooled_T`: `93.0`
- `replications`: `10000.0`
- `seed`: `42.0`
- `t_stat`: `0.7672656404724003`


### expAY_short_only_3x

- `CompositeSort`: `-0.5602`
- `LightGBM`: `-0.9717`
- `LowVolSort`: `-0.9659`
- `MomentumML_Blend`: `-0.4395`
- `MomentumSort`: `-0.4355`
- `Ridge`: `-0.7642`
- `XGBoost`: `-0.9754`


### expDI_frontier_diagnostics

- `cross_sectional_kappa_1`: `0.6381`
- `cross_sectional_kappa_1.5`: `0.6397`
- `cross_sectional_kappa_2`: `0.6443`
- `cross_sectional_kappa_3`: `0.6537`
- `fsi_coef_spread_controlled_kappa_3`: `1.0017`
- `lambda_ci_hi_cluster_market`: `0.402`
- `lambda_ci_hi_cluster_strategy`: `0.4111`
- `lambda_ci_lo_cluster_market`: `0.2656`
- `lambda_ci_lo_cluster_strategy`: `0.2564`
- `partial_corr_fsi_given_turnover_kappa_1`: `0.2442`
- `partial_corr_fsi_given_turnover_kappa_3`: `0.2628`
- `pooled_2ch_fsi_coef`: `1.008`
- `pooled_2ch_fsi_only_r2`: `0.3359`
- `pooled_2ch_full_r2`: `0.734`
- `pooled_2ch_nonsmooth_coef`: `0.2233`
- `pooled_kappa_fsi_r2`: `0.7239`
- `pooled_kappa_only_r2`: `0.3715`
- `r2_fsi_plus_spread_kappa_3`: `0.6543`
- `turnover_on_fsi_r2`: `0.7268`
- `within_strategy_r2_kappa_3`: `0.0648`


### expDA_class_rank_ac

- `FAST`: `0.675`
- `MEDIUM`: `0.8295`
- `SLOW`: `0.965`


### expDJ_joint_interventions

| Item | mean_sr_diff | signflip_p |
|---|---|---|
| EMA_0.35 | 0.1144 | 0.0219 |
| JOINT_all_eight | 0.1119 | 0.0079 |
| PersistenceFilter | 0.1126 | 0.0108 |
| RT_Slow | 0.1042 | 0.0132 |
| Ridge_All_Q | 0.1219 | 0.0131 |
| Ridge_Slow | 0.1124 | 0.0083 |
| Ridge_Slow_Q | 0.1285 | 0.0029 |
| SpeedTercile | 0.1099 | 0.0179 |
| WF_EMA | 0.0916 | 0.0355 |


### expDK_grc_exclusion

- `Blend`: `0.375`
- `CompositeSort`: `0.2812`
- `EMA_0.35`: `0.4541`
- `LightGBM`: `0.1004`
- `MomentumSort`: `0.3546`
- `PersistenceFilter`: `0.4327`
- `RT_Slow`: `0.4244`
- `Ridge_All`: `0.3261`
- `Ridge_All_Q`: `0.4639`
- `Ridge_Fast`: `0.2073`
- `Ridge_Slow`: `0.4526`
- `Ridge_Slow_Q`: `0.4699`
- `SpeedTercile`: `0.4405`
- `WF_EMA`: `0.4317`
- `XGBoost`: `0.1352`


### expDM_full_oos_tree_ema

- `MomentumSort`: `0.3596`
- `XGB_EMA_0.2`: `0.4463`
- `XGB_EMA_0.35`: `0.4363`
- `XGB_EMA_0.5`: `0.4166`
- `XGB_EMA_0.7`: `0.3434`
- `XGB_WF_EMA`: `0.3505`
- `XGBoost`: `0.2368`


### expDN_value_weighted

- `MomentumSort`: `0.3051`
- `Ridge_All`: `0.2982`
- `Ridge_Slow`: `0.3999`


### expCM_us_oos

| Item | Net_SR |
|---|---|
| MomentumSort@0x | 0.7747 |
| MomentumSort@1x | 0.6930 |
| MomentumSort@2x | 0.6115 |
| MomentumSort@3x | 0.5303 |
| Ridge_All@0x | 0.5802 |
| Ridge_All@1x | 0.4449 |
| Ridge_All@2x | 0.3097 |
| Ridge_All@3x | 0.1746 |
| Ridge_Fast@0x | 0.4041 |
| Ridge_Fast@1x | 0.2062 |
| Ridge_Fast@2x | 0.0084 |
| Ridge_Fast@3x | -0.1892 |
| Ridge_Slow@0x | 0.6385 |
| Ridge_Slow@1x | 0.5958 |
| Ridge_Slow@2x | 0.5531 |
| Ridge_Slow@3x | 0.5103 |
