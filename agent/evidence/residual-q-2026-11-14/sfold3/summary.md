# Learned controllers against the baselines (simulated)

Simulator configuration `v1,eval-fleet,sigma=0.25`; latency model uncalibrated (sim/LIMITATIONS.md). Workload `data_pipeline/profiles/profile_diurnal_long_sfold3_holdout.csv` (sha256 `94fe6bd6f8a5cb163cba7a9fafac1274d4ec6511113920a63d4993f03295a53d`). 20 paired scenario seeds (0..19), identical for every policy. Learned policies act by their mode (argmax). Reward is the contract reward with contract coefficients.

## Checkpoints

| checkpoint | policy | iteration | trained_on | hazard | train_fleet | sigma | val_gap | val_gap_sem | val_seeds | params |
|---|---|---|---|---|---|---|---|---|---|---|
| runs/residual_q/sfold3/seed0 | residual_q seed0 | 0 | data_pipeline/profiles/profile_diurnal_long_sfold3_train.csv | v1 | train-fleet | 0.25 | 6.5 | 1.583 | 5000..5039 | margin=0.1 (deterministic) |
| runs/residual_q/sfold3/seed1 | residual_q seed1 | 0 | data_pipeline/profiles/profile_diurnal_long_sfold3_train.csv | v1 | train-fleet | 0.25 | 7.727 | 1.386 | 5000..5039 | margin=0.1 (deterministic) |
| runs/residual_q/sfold3/seed2 | residual_q seed2 | 0 | data_pipeline/profiles/profile_diurnal_long_sfold3_train.csv | v1 | train-fleet | 0.25 | 8.67 | 1.248 | 5000..5039 | margin=0.25 (deterministic) |

## Summary

| profile | policy | workload_synthetic | n | completed | horizon_expired | terminal_failure | completion_h | lat_deg_mean | lat_deg_p95 | viol_frac | lat_budget_met | p_fail_mean | p_fail_max | risk_viol | clip_frac | deliv_mbps | stall_frac | reward | reward_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_sfold3_holdout | fixed_20 | false | 20 | 20 | 0 | 0 | 144.8 | 0.1431 | 0.85 | 0.02772 | 0.85 | 0.001984 | 0.003874 | 0.1 | 0.123 | 26.9 | 0 | -446.4 | 149.8 |
| diurnal_long_sfold3_holdout | fixed_50 | false | 20 | 20 | 0 | 0 | 63.08 | 0.1814 | 0.85 | 0.0352 | 0.85 | 0.00123 | 0.002381 | 0.1 | 0.08194 | 54.12 | 0 | -348.3 | 157.2 |
| diurnal_long_sfold3_holdout | fixed_80 | false | 20 | 20 | 0 | 0 | 40.82 | 0.2702 | 1.142 | 0.07297 | 0.85 | 0.0009776 | 0.001919 | 0.1 | 0.05825 | 82 | 0 | -498.6 | 147.4 |
| diurnal_long_sfold3_holdout | fixed_120 | false | 20 | 20 | 0 | 0 | 29.27 | 3.498 | 5.667 | 0.6726 | 0 | 0.0008022 | 0.0016 | 0.1 | 0 | 114 | 0 | -3630 | 90.84 |
| diurnal_long_sfold3_holdout | workload_only_throttle | false | 20 | 20 | 0 | 0 | 43.6 | 0.2315 | 0.85 | 0.04439 | 0.85 | 0.001079 | 0.002298 | 0.1 | 0.06453 | 77.33 | 0 | -330.2 | 157.9 |
| diurnal_long_sfold3_holdout | risk_aware_rule_based | false | 20 | 20 | 0 | 0 | 43.6 | 0.2315 | 0.85 | 0.04439 | 0.85 | 0.001079 | 0.002298 | 0.1 | 0.02963 | 77.33 | 0 | -330.2 | 157.9 |
| diurnal_long_sfold3_holdout | ORACLE_lookahead (not deployable) | false | 20 | 20 | 0 | 0 | 36.86 | 0.2462 | 0.8837 | 0.04682 | 0.85 | 0.0009537 | 0.002293 | 0.1 | 0.02572 | 90.95 | 0 | -318.3 | 158.8 |
| diurnal_long_sfold3_holdout | rule_seasonal_forecast (hand-built) | false | 20 | 20 | 0 | 0 | 40.59 | 0.2409 | 0.8532 | 0.0462 | 0.85 | 0.001079 | 0.002298 | 0.1 | 0.03069 | 82.88 | 0 | -326.5 | 158.1 |
| diurnal_long_sfold3_holdout | residual_q seed0 | false | 20 | 20 | 0 | 0 | 39.05 | 0.2447 | 0.8696 | 0.04621 | 0.85 | 0.001023 | 0.002718 | 0.1 | 0.02819 | 86.04 | 0 | -321.9 | 158.5 |
| diurnal_long_sfold3_holdout | residual_q seed1 | false | 20 | 20 | 0 | 0 | 39.16 | 0.245 | 0.863 | 0.04653 | 0.85 | 0.001027 | 0.002528 | 0.1 | 0.0295 | 85.79 | 0 | -322.9 | 158.5 |
| diurnal_long_sfold3_holdout | residual_q seed2 | false | 20 | 20 | 0 | 0 | 40.25 | 0.2407 | 0.8539 | 0.04582 | 0.85 | 0.001069 | 0.002449 | 0.1 | 0.03078 | 83.55 | 0 | -322.4 | 158.5 |

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
| residual_q seed0 | all | 20 | 8.314 | 1.277 | 20 | 0 |
| residual_q seed0 | initial_budget_infeasible=False (pre-registered) | 18 | 9.112 | 1.284 | 18 | 0 |
| residual_q seed0 | initial_budget_infeasible=True (pre-registered) | 2 | 1.133 | 0.4645 | 2 | 0 |
| residual_q seed0 | typical, rule reward >= -500 (post hoc) | 17 | 9.548 | 1.281 | 17 | 0 |
| residual_q seed0 | extreme, rule reward < -500 (post hoc) | 3 | 1.322 | 0.3277 | 3 | 0 |
| residual_q seed1 | all | 20 | 7.342 | 1.565 | 19 | 1 |
| residual_q seed1 | initial_budget_infeasible=False (pre-registered) | 18 | 8.019 | 1.664 | 17 | 1 |
| residual_q seed1 | initial_budget_infeasible=True (pre-registered) | 2 | 1.25 | 0.3942 | 2 | 0 |
| residual_q seed1 | typical, rule reward >= -500 (post hoc) | 17 | 8.434 | 1.71 | 16 | 1 |
| residual_q seed1 | extreme, rule reward < -500 (post hoc) | 3 | 1.153 | 0.2473 | 3 | 0 |
| residual_q seed2 | all | 20 | 7.801 | 1.319 | 20 | 0 |
| residual_q seed2 | initial_budget_infeasible=False (pre-registered) | 18 | 8.542 | 1.355 | 18 | 0 |
| residual_q seed2 | initial_budget_infeasible=True (pre-registered) | 2 | 1.129 | 0.1747 | 2 | 0 |
| residual_q seed2 | typical, rule reward >= -500 (post hoc) | 17 | 8.993 | 1.356 | 17 | 0 |
| residual_q seed2 | extreme, rule reward < -500 (post hoc) | 3 | 1.049 | 0.1284 | 3 | 0 |


## Paired difference: rule_seasonal_forecast (hand-built) minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_sfold3_holdout | total_reward | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 3.685 | 0.2758 | 20 |
| diurnal_long_sfold3_holdout | latency_degradation_mean | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 0.009383 | 0.005559 | 20 |
| diurnal_long_sfold3_holdout | latency_violation_fraction | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 0.001804 | 0.001045 | 20 |
| diurnal_long_sfold3_holdout | completion_hours | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | -3.008 | 0.2061 | 20 |
| diurnal_long_sfold3_holdout | mean_delivered_mbps | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 5.542 | 0.2749 | 20 |
| diurnal_long_sfold3_holdout | p_failure_mean | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 7.435e-07 | 9.384e-06 | 20 |
| diurnal_long_sfold3_holdout | clip_fraction | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 0.001064 | 0.0007475 | 20 |

## Paired difference: residual_q seed0 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_sfold3_holdout | total_reward | residual_q seed0 | risk_aware_rule_based | 8.314 | 1.277 | 20 |
| diurnal_long_sfold3_holdout | latency_degradation_mean | residual_q seed0 | risk_aware_rule_based | 0.01324 | 0.007638 | 20 |
| diurnal_long_sfold3_holdout | latency_violation_fraction | residual_q seed0 | risk_aware_rule_based | 0.001816 | 0.001488 | 20 |
| diurnal_long_sfold3_holdout | completion_hours | residual_q seed0 | risk_aware_rule_based | -4.546 | 0.3564 | 20 |
| diurnal_long_sfold3_holdout | mean_delivered_mbps | residual_q seed0 | risk_aware_rule_based | 8.702 | 0.5804 | 20 |
| diurnal_long_sfold3_holdout | p_failure_mean | residual_q seed0 | risk_aware_rule_based | -5.6e-05 | 1.851e-05 | 20 |
| diurnal_long_sfold3_holdout | clip_fraction | residual_q seed0 | risk_aware_rule_based | -0.001435 | 0.001639 | 20 |

## Paired difference: residual_q seed1 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_sfold3_holdout | total_reward | residual_q seed1 | risk_aware_rule_based | 7.342 | 1.565 | 20 |
| diurnal_long_sfold3_holdout | latency_degradation_mean | residual_q seed1 | risk_aware_rule_based | 0.01352 | 0.008251 | 20 |
| diurnal_long_sfold3_holdout | latency_violation_fraction | residual_q seed1 | risk_aware_rule_based | 0.002139 | 0.001571 | 20 |
| diurnal_long_sfold3_holdout | completion_hours | residual_q seed1 | risk_aware_rule_based | -4.442 | 0.3396 | 20 |
| diurnal_long_sfold3_holdout | mean_delivered_mbps | residual_q seed1 | risk_aware_rule_based | 8.455 | 0.543 | 20 |
| diurnal_long_sfold3_holdout | p_failure_mean | residual_q seed1 | risk_aware_rule_based | -5.133e-05 | 1.724e-05 | 20 |
| diurnal_long_sfold3_holdout | clip_fraction | residual_q seed1 | risk_aware_rule_based | -0.000125 | 0.001079 | 20 |

## Paired difference: residual_q seed2 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_sfold3_holdout | total_reward | residual_q seed2 | risk_aware_rule_based | 7.801 | 1.319 | 20 |
| diurnal_long_sfold3_holdout | latency_degradation_mean | residual_q seed2 | risk_aware_rule_based | 0.009186 | 0.006549 | 20 |
| diurnal_long_sfold3_holdout | latency_violation_fraction | residual_q seed2 | risk_aware_rule_based | 0.001428 | 0.001254 | 20 |
| diurnal_long_sfold3_holdout | completion_hours | residual_q seed2 | risk_aware_rule_based | -3.354 | 0.2403 | 20 |
| diurnal_long_sfold3_holdout | mean_delivered_mbps | residual_q seed2 | risk_aware_rule_based | 6.212 | 0.3407 | 20 |
| diurnal_long_sfold3_holdout | p_failure_mean | residual_q seed2 | risk_aware_rule_based | -9.114e-06 | 1.118e-05 | 20 |
| diurnal_long_sfold3_holdout | clip_fraction | residual_q seed2 | risk_aware_rule_based | 0.001152 | 0.0008218 | 20 |
