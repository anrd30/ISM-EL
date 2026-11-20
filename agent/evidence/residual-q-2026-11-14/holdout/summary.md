# Learned controllers against the baselines (simulated)

Simulator configuration `v1,eval-fleet,sigma=0.25`; latency model uncalibrated (sim/LIMITATIONS.md). Workload `data_pipeline/profiles/profile_diurnal_long_holdout.csv` (sha256 `94fe6bd6f8a5cb163cba7a9fafac1274d4ec6511113920a63d4993f03295a53d`). 20 paired scenario seeds (0..19), identical for every policy. Learned policies act by their mode (argmax). Reward is the contract reward with contract coefficients.

## Checkpoints

| checkpoint | policy | iteration | trained_on | hazard | train_fleet | sigma | val_gap | val_gap_sem | val_seeds | params |
|---|---|---|---|---|---|---|---|---|---|---|
| runs/residual_q/main/seed0 | residual_q seed0 | 0 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | 10.45 | 1.412 | 5000..5039 | margin=0.1 (deterministic) |
| runs/residual_q/main/seed1 | residual_q seed1 | 0 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | 6.936 | 1.224 | 5000..5039 | margin=0.25 (deterministic) |
| runs/residual_q/main/seed2 | residual_q seed2 | 0 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | 7.025 | 1.608 | 5000..5039 | margin=0.1 (deterministic) |

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
| diurnal_long_holdout | residual_q seed0 | false | 20 | 20 | 0 | 0 | 38.99 | 0.2452 | 0.8723 | 0.04639 | 0.85 | 0.001024 | 0.00272 | 0.1 | 0.02848 | 86.14 | 0 | -323.1 | 158.4 |
| diurnal_long_holdout | residual_q seed1 | false | 20 | 20 | 0 | 0 | 40.36 | 0.2414 | 0.8548 | 0.04599 | 0.85 | 0.001073 | 0.002322 | 0.1 | 0.03043 | 83.35 | 0 | -323.3 | 158.4 |
| diurnal_long_holdout | residual_q seed2 | false | 20 | 20 | 0 | 0 | 39.01 | 0.2442 | 0.8666 | 0.04619 | 0.85 | 0.001023 | 0.002445 | 0.1 | 0.02885 | 86.1 | 0 | -321.7 | 158.6 |

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
| residual_q seed0 | all | 20 | 7.06 | 1.112 | 20 | 0 |
| residual_q seed0 | initial_budget_infeasible=False (pre-registered) | 18 | 7.703 | 1.136 | 18 | 0 |
| residual_q seed0 | initial_budget_infeasible=True (pre-registered) | 2 | 1.278 | 0.4224 | 2 | 0 |
| residual_q seed0 | typical, rule reward >= -500 (post hoc) | 17 | 8.087 | 1.134 | 17 | 0 |
| residual_q seed0 | extreme, rule reward < -500 (post hoc) | 3 | 1.242 | 0.2466 | 3 | 0 |
| residual_q seed1 | all | 20 | 6.939 | 1.175 | 20 | 0 |
| residual_q seed1 | initial_budget_infeasible=False (pre-registered) | 18 | 7.595 | 1.208 | 18 | 0 |
| residual_q seed1 | initial_budget_infeasible=True (pre-registered) | 2 | 1.035 | 0.562 | 2 | 0 |
| residual_q seed1 | typical, rule reward >= -500 (post hoc) | 17 | 7.987 | 1.213 | 17 | 0 |
| residual_q seed1 | extreme, rule reward < -500 (post hoc) | 3 | 1.005 | 0.3258 | 3 | 0 |
| residual_q seed2 | all | 20 | 8.535 | 1.352 | 20 | 0 |
| residual_q seed2 | initial_budget_infeasible=False (pre-registered) | 18 | 9.358 | 1.367 | 18 | 0 |
| residual_q seed2 | initial_budget_infeasible=True (pre-registered) | 2 | 1.133 | 0.4645 | 2 | 0 |
| residual_q seed2 | typical, rule reward >= -500 (post hoc) | 17 | 9.785 | 1.378 | 17 | 0 |
| residual_q seed2 | extreme, rule reward < -500 (post hoc) | 3 | 1.455 | 0.4188 | 3 | 0 |


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

## Paired difference: residual_q seed0 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | residual_q seed0 | risk_aware_rule_based | 7.06 | 1.112 | 20 |
| diurnal_long_holdout | latency_degradation_mean | residual_q seed0 | risk_aware_rule_based | 0.01374 | 0.007444 | 20 |
| diurnal_long_holdout | latency_violation_fraction | residual_q seed0 | risk_aware_rule_based | 0.001992 | 0.001445 | 20 |
| diurnal_long_holdout | completion_hours | residual_q seed0 | risk_aware_rule_based | -4.608 | 0.3573 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | residual_q seed0 | risk_aware_rule_based | 8.81 | 0.5731 | 20 |
| diurnal_long_holdout | p_failure_mean | residual_q seed0 | risk_aware_rule_based | -5.429e-05 | 1.717e-05 | 20 |
| diurnal_long_holdout | clip_fraction | residual_q seed0 | risk_aware_rule_based | -0.001147 | 0.001199 | 20 |

## Paired difference: residual_q seed1 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | residual_q seed1 | risk_aware_rule_based | 6.939 | 1.175 | 20 |
| diurnal_long_holdout | latency_degradation_mean | residual_q seed1 | risk_aware_rule_based | 0.009972 | 0.006786 | 20 |
| diurnal_long_holdout | latency_violation_fraction | residual_q seed1 | risk_aware_rule_based | 0.001593 | 0.001297 | 20 |
| diurnal_long_holdout | completion_hours | residual_q seed1 | risk_aware_rule_based | -3.238 | 0.2257 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | residual_q seed1 | risk_aware_rule_based | 6.014 | 0.3121 | 20 |
| diurnal_long_holdout | p_failure_mean | residual_q seed1 | risk_aware_rule_based | -5.688e-06 | 1.093e-05 | 20 |
| diurnal_long_holdout | clip_fraction | residual_q seed1 | risk_aware_rule_based | 0.000803 | 0.0006481 | 20 |

## Paired difference: residual_q seed2 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | residual_q seed2 | risk_aware_rule_based | 8.535 | 1.352 | 20 |
| diurnal_long_holdout | latency_degradation_mean | residual_q seed2 | risk_aware_rule_based | 0.01277 | 0.007585 | 20 |
| diurnal_long_holdout | latency_violation_fraction | residual_q seed2 | risk_aware_rule_based | 0.001793 | 0.001478 | 20 |
| diurnal_long_holdout | completion_hours | residual_q seed2 | risk_aware_rule_based | -4.588 | 0.3593 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | residual_q seed2 | risk_aware_rule_based | 8.763 | 0.5773 | 20 |
| diurnal_long_holdout | p_failure_mean | residual_q seed2 | risk_aware_rule_based | -5.516e-05 | 1.846e-05 | 20 |
| diurnal_long_holdout | clip_fraction | residual_q seed2 | risk_aware_rule_based | -0.000779 | 0.001159 | 20 |
