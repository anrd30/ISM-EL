# Learned controllers against the baselines (simulated)

Simulator configuration `v1,eval-fleet,sigma=0.25`; latency model uncalibrated (sim/LIMITATIONS.md). Workload `data_pipeline/profiles/profile_diurnal_long_holdout.csv` (sha256 `94fe6bd6f8a5cb163cba7a9fafac1274d4ec6511113920a63d4993f03295a53d` (Corrected: this is the LF hash of the committed file; an earlier note quoted the CRLF hash of the same content from a Windows test run that temporarily rewrote it before the LF version was restored. The file content is unchanged.)). 20 paired scenario seeds (0..19), identical for every policy. Learned policies act deterministically. Reward is the contract reward with contract coefficients.

## Checkpoints

| checkpoint | policy | iteration | trained_on | hazard | train_fleet | sigma | val_gap | val_gap_sem | val_seeds | params |
|---|---|---|---|---|---|---|---|---|---|---|
| runs/search/seasonal/seed0/search.json | seasonal_search seed0 | 7 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | 3.783 | 0.83 | 5000..5039 | progress_scale=0.715, spread=0.823, mix=0.905 |
| runs/search/seasonal/seed1/search.json | seasonal_search seed1 | 2 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | 3.07 | 0.816 | 5000..5039 | progress_scale=0.799, spread=0.980, mix=0.820 |
| runs/search/seasonal/seed2/search.json | seasonal_search seed2 | 2 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | 4.656 | 0.934 | 5000..5039 | progress_scale=0.488, spread=0.897, mix=0.977 |
| runs/search/rule/seed0/search.json | rule_progress_search seed0 | 2 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | 1.359 | 0.573 | 5000..5039 | progress_scale=0.779 |
| runs/search/rule/seed1/search.json | rule_progress_search seed1 | 2 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | 1.367 | 0.572 | 5000..5039 | progress_scale=0.797 |
| runs/search/rule/seed2/search.json | rule_progress_search seed2 | 4 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | 2.341 | 0.708 | 5000..5039 | progress_scale=0.488 |

## Summary

| profile | policy | workload_synthetic | n | completed | horizon_expired | terminal_failure | completion_h | lat_deg_mean | lat_deg_p95 | viol_frac | lat_budget_met | p_fail_mean | p_fail_max | risk_viol | clip_frac | deliv_mbps | stall_frac | reward | reward_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_holdout | fixed_20 | false | 20 | 20 | 0 | 0 | 144.8 | 0.1431 | 0.85 | 0.02772 | 0.85 | 0.001984 | 0.003874 | 0.1 | 0.123 | 26.9 | 0 | -446.4 | 149.8 |
| diurnal_long_holdout | fixed_50 | false | 20 | 20 | 0 | 0 | 63.08 | 0.1814 | 0.85 | 0.0352 | 0.85 | 0.00123 | 0.002381 | 0.1 | 0.08194 | 54.12 | 0 | -348.3 | 157.2 |
| diurnal_long_holdout | fixed_80 | false | 20 | 20 | 0 | 0 | 40.82 | 0.2702 | 1.142 | 0.07297 | 0.85 | 0.0009776 | 0.001919 | 0.1 | 0.05825 | 82 | 0 | -498.6 | 147.4 |
| diurnal_long_holdout | fixed_120 | false | 20 | 20 | 0 | 0 | 29.27 | 3.498 | 5.667 | 0.6726 | 0 | 0.0008022 | 0.0016 | 0.1 | 0 | 114 | 0 | -3630 | 90.84 |
| diurnal_long_holdout | workload_only_throttle | false | 20 | 20 | 0 | 0 | 43.6 | 0.2315 | 0.85 | 0.04439 | 0.85 | 0.001079 | 0.002298 | 0.1 | 0.06453 | 77.33 | 0 | -330.2 | 157.9 |
| diurnal_long_holdout | risk_aware_rule_based | false | 20 | 20 | 0 | 0 | 43.6 | 0.2315 | 0.85 | 0.04439 | 0.85 | 0.001079 | 0.002298 | 0.1 | 0.02963 | 77.33 | 0 | -330.2 | 157.9 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | false | 20 | 20 | 0 | 0 | 36.86 | 0.2462 | 0.8837 | 0.04682 | 0.85 | 0.0009537 | 0.002293 | 0.1 | 0.02572 | 90.95 | 0 | -318.3 | 158.8 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | false | 20 | 20 | 0 | 0 | 40.59 | 0.2409 | 0.8532 | 0.0462 | 0.85 | 0.001079 | 0.002298 | 0.1 | 0.03069 | 82.88 | 0 | -326.5 | 158.1 |
| diurnal_long_holdout | seasonal_search seed0 | false | 20 | 20 | 0 | 0 | 41.61 | 0.2355 | 0.85 | 0.04547 | 0.85 | 0.001079 | 0.002299 | 0.1 | 0.03048 | 80.86 | 0 | -326.9 | 158.1 |
| diurnal_long_holdout | seasonal_search seed1 | false | 20 | 20 | 0 | 0 | 41.9 | 0.2353 | 0.85 | 0.04541 | 0.85 | 0.001076 | 0.002299 | 0.1 | 0.03045 | 80.36 | 0 | -327.3 | 158.1 |
| diurnal_long_holdout | seasonal_search seed2 | false | 20 | 20 | 0 | 0 | 41.36 | 0.2366 | 0.85 | 0.04573 | 0.85 | 0.001085 | 0.002378 | 0.1 | 0.03092 | 81.36 | 0 | -326.5 | 158.1 |
| diurnal_long_holdout | rule_progress_search seed0 | false | 20 | 20 | 0 | 0 | 43.87 | 0.2303 | 0.85 | 0.04424 | 0.85 | 0.001084 | 0.002299 | 0.1 | 0.0298 | 76.85 | 0 | -330.3 | 157.9 |
| diurnal_long_holdout | rule_progress_search seed1 | false | 20 | 20 | 0 | 0 | 43.85 | 0.2305 | 0.85 | 0.04427 | 0.85 | 0.001084 | 0.002299 | 0.1 | 0.02984 | 76.88 | 0 | -330.3 | 157.9 |
| diurnal_long_holdout | rule_progress_search seed2 | false | 20 | 20 | 0 | 0 | 44.17 | 0.2286 | 0.85 | 0.04403 | 0.85 | 0.001088 | 0.002378 | 0.1 | 0.02993 | 76.34 | 0 | -330.4 | 157.9 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Paired total-reward gap against risk_aware_rule_based, by stratum

`gap` is treatment minus rule, mean over paired scenarios; `sem` its standard error; `n_better` / `n_worse` count scenarios. Strata are defined in the module docstring.

| policy | stratum | n | gap | sem | n_better | n_worse |
|---|---|---|---|---|---|---|
| rule_seasonal_forecast (hand-built) | all | 20 | 3.685 | 0.2758 | 20 | 0 |
| rule_seasonal_forecast (hand-built) | initial_budget_infeasible=False (pre-registered) | 18 | 3.969 | 0.2161 | 18 | 0 |
| rule_seasonal_forecast (hand-built) | initial_budget_infeasible=True (pre-registered) | 2 | 1.125 | 0.1713 | 2 | 0 |
| rule_seasonal_forecast (hand-built) | typical, rule reward >= -500 (post hoc) | 17 | 4.09 | 0.1902 | 17 | 0 |
| rule_seasonal_forecast (hand-built) | extreme, rule reward < -500 (post hoc) | 3 | 1.39 | 0.2823 | 3 | 0 |
| seasonal_search seed0 | all | 20 | 3.273 | 0.2526 | 20 | 0 |
| seasonal_search seed0 | initial_budget_infeasible=False (pre-registered) | 18 | 3.521 | 0.2055 | 18 | 0 |
| seasonal_search seed0 | initial_budget_infeasible=True (pre-registered) | 2 | 1.045 | 0.3828 | 2 | 0 |
| seasonal_search seed0 | typical, rule reward >= -500 (post hoc) | 17 | 3.609 | 0.1968 | 17 | 0 |
| seasonal_search seed0 | extreme, rule reward < -500 (post hoc) | 3 | 1.37 | 0.3935 | 3 | 0 |
| seasonal_search seed1 | all | 20 | 2.862 | 0.238 | 20 | 0 |
| seasonal_search seed1 | initial_budget_infeasible=False (pre-registered) | 18 | 3.077 | 0.2057 | 18 | 0 |
| seasonal_search seed1 | initial_budget_infeasible=True (pre-registered) | 2 | 0.9285 | 0.3665 | 2 | 0 |
| seasonal_search seed1 | typical, rule reward >= -500 (post hoc) | 17 | 3.159 | 0.2 | 17 | 0 |
| seasonal_search seed1 | extreme, rule reward < -500 (post hoc) | 3 | 1.178 | 0.3275 | 3 | 0 |
| seasonal_search seed2 | all | 20 | 3.684 | 0.2488 | 20 | 0 |
| seasonal_search seed2 | initial_budget_infeasible=False (pre-registered) | 18 | 3.958 | 0.1799 | 18 | 0 |
| seasonal_search seed2 | initial_budget_infeasible=True (pre-registered) | 2 | 1.217 | 0.2108 | 2 | 0 |
| seasonal_search seed2 | typical, rule reward >= -500 (post hoc) | 17 | 4.038 | 0.1708 | 17 | 0 |
| seasonal_search seed2 | extreme, rule reward < -500 (post hoc) | 3 | 1.676 | 0.4749 | 3 | 0 |
| rule_progress_search seed0 | all | 20 | -0.1424 | 0.05408 | 6 | 14 |
| rule_progress_search seed0 | initial_budget_infeasible=False (pre-registered) | 18 | -0.1584 | 0.05901 | 4 | 14 |
| rule_progress_search seed0 | initial_budget_infeasible=True (pre-registered) | 2 | 0.001064 | 0 | 2 | 0 |
| rule_progress_search seed0 | typical, rule reward >= -500 (post hoc) | 17 | -0.1614 | 0.06251 | 4 | 13 |
| rule_progress_search seed0 | extreme, rule reward < -500 (post hoc) | 3 | -0.03487 | 0.03593 | 2 | 1 |
| rule_progress_search seed1 | all | 20 | -0.1368 | 0.05249 | 5 | 14 |
| rule_progress_search seed1 | initial_budget_infeasible=False (pre-registered) | 18 | -0.152 | 0.05731 | 4 | 14 |
| rule_progress_search seed1 | initial_budget_infeasible=True (pre-registered) | 2 | 0.000532 | 0.000532 | 1 | 0 |
| rule_progress_search seed1 | typical, rule reward >= -500 (post hoc) | 17 | -0.1547 | 0.06073 | 4 | 13 |
| rule_progress_search seed1 | extreme, rule reward < -500 (post hoc) | 3 | -0.03522 | 0.03576 | 1 | 1 |
| rule_progress_search seed2 | all | 20 | -0.1721 | 0.09542 | 7 | 13 |
| rule_progress_search seed2 | initial_budget_infeasible=False (pre-registered) | 18 | -0.1747 | 0.1056 | 6 | 12 |
| rule_progress_search seed2 | initial_budget_infeasible=True (pre-registered) | 2 | -0.1489 | 0.15 | 1 | 1 |
| rule_progress_search seed2 | typical, rule reward >= -500 (post hoc) | 17 | -0.1804 | 0.1119 | 6 | 11 |
| rule_progress_search seed2 | extreme, rule reward < -500 (post hoc) | 3 | -0.1252 | 0.0898 | 1 | 2 |


## Seasonal search against the progress-price search of the same seed

The comparator retunes only the rule's progress price on the same training days, so this gap is what the forecast adds beyond retuning the baseline.

| treatment | comparator | stratum | n | gap | sem |
|---|---|---|---|---|---|
| seasonal_search seed0 | rule_progress_search seed0 | all | 20 | 3.416 | 0.255 |
| seasonal_search seed1 | rule_progress_search seed1 | all | 20 | 2.999 | 0.2437 |
| seasonal_search seed2 | rule_progress_search seed2 | all | 20 | 3.856 | 0.2527 |

## Paired difference: rule_seasonal_forecast (hand-built) minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 3.685 | 0.2758 | 20 |
| diurnal_long_holdout | latency_degradation_mean | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 0.009383 | 0.005559 | 20 |
| diurnal_long_holdout | latency_violation_fraction | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 0.001804 | 0.001045 | 20 |
| diurnal_long_holdout | completion_hours | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | -3.008 | 0.2061 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 5.542 | 0.2749 | 20 |
| diurnal_long_holdout | p_failure_mean | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 7.435e-07 | 9.384e-06 | 20 |
| diurnal_long_holdout | clip_fraction | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 0.001064 | 0.0007475 | 20 |

## Paired difference: seasonal_search seed0 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | seasonal_search seed0 | risk_aware_rule_based | 3.273 | 0.2526 | 20 |
| diurnal_long_holdout | latency_degradation_mean | seasonal_search seed0 | risk_aware_rule_based | 0.004005 | 0.00378 | 20 |
| diurnal_long_holdout | latency_violation_fraction | seasonal_search seed0 | risk_aware_rule_based | 0.001078 | 0.000703 | 20 |
| diurnal_long_holdout | completion_hours | seasonal_search seed0 | risk_aware_rule_based | -1.992 | 0.1916 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | seasonal_search seed0 | risk_aware_rule_based | 3.525 | 0.2886 | 20 |
| diurnal_long_holdout | p_failure_mean | seasonal_search seed0 | risk_aware_rule_based | 3.448e-07 | 7.953e-06 | 20 |
| diurnal_long_holdout | clip_fraction | seasonal_search seed0 | risk_aware_rule_based | 0.0008556 | 0.0005909 | 20 |

## Paired difference: seasonal_search seed1 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | seasonal_search seed1 | risk_aware_rule_based | 2.862 | 0.238 | 20 |
| diurnal_long_holdout | latency_degradation_mean | seasonal_search seed1 | risk_aware_rule_based | 0.003862 | 0.003757 | 20 |
| diurnal_long_holdout | latency_violation_fraction | seasonal_search seed1 | risk_aware_rule_based | 0.001021 | 0.000695 | 20 |
| diurnal_long_holdout | completion_hours | seasonal_search seed1 | risk_aware_rule_based | -1.704 | 0.186 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | seasonal_search seed1 | risk_aware_rule_based | 3.028 | 0.2959 | 20 |
| diurnal_long_holdout | p_failure_mean | seasonal_search seed1 | risk_aware_rule_based | -2.533e-06 | 8.167e-06 | 20 |
| diurnal_long_holdout | clip_fraction | seasonal_search seed1 | risk_aware_rule_based | 0.0008197 | 0.0005691 | 20 |

## Paired difference: seasonal_search seed2 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | seasonal_search seed2 | risk_aware_rule_based | 3.684 | 0.2488 | 20 |
| diurnal_long_holdout | latency_degradation_mean | seasonal_search seed2 | risk_aware_rule_based | 0.005079 | 0.004127 | 20 |
| diurnal_long_holdout | latency_violation_fraction | seasonal_search seed2 | risk_aware_rule_based | 0.001338 | 0.0007696 | 20 |
| diurnal_long_holdout | completion_hours | seasonal_search seed2 | risk_aware_rule_based | -2.238 | 0.1724 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | seasonal_search seed2 | risk_aware_rule_based | 4.023 | 0.2432 | 20 |
| diurnal_long_holdout | p_failure_mean | seasonal_search seed2 | risk_aware_rule_based | 6.146e-06 | 7.61e-06 | 20 |
| diurnal_long_holdout | clip_fraction | seasonal_search seed2 | risk_aware_rule_based | 0.001295 | 0.0009315 | 20 |

## Paired difference: rule_progress_search seed0 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | rule_progress_search seed0 | risk_aware_rule_based | -0.1424 | 0.05408 | 20 |
| diurnal_long_holdout | latency_degradation_mean | rule_progress_search seed0 | risk_aware_rule_based | -0.00114 | 0.0004351 | 20 |
| diurnal_long_holdout | latency_violation_fraction | rule_progress_search seed0 | risk_aware_rule_based | -0.0001505 | 8.36e-05 | 20 |
| diurnal_long_holdout | completion_hours | rule_progress_search seed0 | risk_aware_rule_based | 0.2708 | 0.04599 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | rule_progress_search seed0 | risk_aware_rule_based | -0.4827 | 0.08183 | 20 |
| diurnal_long_holdout | p_failure_mean | rule_progress_search seed0 | risk_aware_rule_based | 5.261e-06 | 1.874e-06 | 20 |
| diurnal_long_holdout | clip_fraction | rule_progress_search seed0 | risk_aware_rule_based | 0.0001761 | 0.0002226 | 20 |

## Paired difference: rule_progress_search seed1 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | rule_progress_search seed1 | risk_aware_rule_based | -0.1368 | 0.05249 | 20 |
| diurnal_long_holdout | latency_degradation_mean | rule_progress_search seed1 | risk_aware_rule_based | -0.00096 | 0.0004285 | 20 |
| diurnal_long_holdout | latency_violation_fraction | rule_progress_search seed1 | risk_aware_rule_based | -0.0001215 | 8.082e-05 | 20 |
| diurnal_long_holdout | completion_hours | rule_progress_search seed1 | risk_aware_rule_based | 0.25 | 0.0476 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | rule_progress_search seed1 | risk_aware_rule_based | -0.4548 | 0.08106 | 20 |
| diurnal_long_holdout | p_failure_mean | rule_progress_search seed1 | risk_aware_rule_based | 5.337e-06 | 1.81e-06 | 20 |
| diurnal_long_holdout | clip_fraction | rule_progress_search seed1 | risk_aware_rule_based | 0.0002112 | 0.0002573 | 20 |

## Paired difference: rule_progress_search seed2 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | rule_progress_search seed2 | risk_aware_rule_based | -0.1721 | 0.09542 | 20 |
| diurnal_long_holdout | latency_degradation_mean | rule_progress_search seed2 | risk_aware_rule_based | -0.002907 | 0.001278 | 20 |
| diurnal_long_holdout | latency_violation_fraction | rule_progress_search seed2 | risk_aware_rule_based | -0.0003665 | 0.0002472 | 20 |
| diurnal_long_holdout | completion_hours | rule_progress_search seed2 | risk_aware_rule_based | 0.5708 | 0.07411 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | rule_progress_search seed2 | risk_aware_rule_based | -0.9956 | 0.1284 | 20 |
| diurnal_long_holdout | p_failure_mean | rule_progress_search seed2 | risk_aware_rule_based | 9.818e-06 | 2.943e-06 | 20 |
| diurnal_long_holdout | clip_fraction | rule_progress_search seed2 | risk_aware_rule_based | 0.0003066 | 0.000503 | 20 |
