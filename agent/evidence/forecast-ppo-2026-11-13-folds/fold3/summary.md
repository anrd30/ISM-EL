# Learned controllers against the baselines (simulated)

Simulator configuration `v1,eval-fleet,sigma=0.25`; latency model uncalibrated (sim/LIMITATIONS.md). Workload `data_pipeline/profiles/profile_diurnal_long_sfold3_holdout.csv` (sha256 `94fe6bd6f8a5cb163cba7a9fafac1274d4ec6511113920a63d4993f03295a53d`). 20 paired scenario seeds (0..19), identical for every policy. Learned policies act by sampling. Reward is the contract reward with contract coefficients.

## Checkpoints

| checkpoint | policy | iteration | trained_on | hazard | train_fleet | sigma | val_gap | val_gap_sem | val_seeds | params |
|---|---|---|---|---|---|---|---|---|---|---|
| runs/forecast_ppo_sfold3/bc/seed0/ckpt_best.pt | forecast_ppo_bc seed0 | 80 | data_pipeline/profiles/profile_diurnal_long_sfold3_train.csv | v1 | train-fleet | 0.25 | 6.158 | 1.725 | 5000..5039 | bc_accuracy=0.960 |
| runs/forecast_ppo_sfold3/bc/seed1/ckpt_best.pt | forecast_ppo_bc seed1 | 40 | data_pipeline/profiles/profile_diurnal_long_sfold3_train.csv | v1 | train-fleet | 0.25 | 3.778 | 1.822 | 5000..5039 | bc_accuracy=0.956 |
| runs/forecast_ppo_sfold3/bc/seed2/ckpt_best.pt | forecast_ppo_bc seed2 | 80 | data_pipeline/profiles/profile_diurnal_long_sfold3_train.csv | v1 | train-fleet | 0.25 | 6.478 | 1.654 | 5000..5039 | bc_accuracy=0.957 |

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
| diurnal_long_sfold3_holdout | forecast_ppo_bc seed0 | false | 20 | 20 | 0 | 0 | 41.6 | 0.24 | 0.8507 | 0.04713 | 0.85 | 0.00111 | 0.002349 | 0.1 | 0.03073 | 80.91 | 0 | -333.4 | 158.5 |
| diurnal_long_sfold3_holdout | forecast_ppo_bc seed1 | false | 20 | 20 | 0 | 0 | 40.36 | 0.2475 | 0.8763 | 0.05009 | 0.85 | 0.001076 | 0.002501 | 0.1 | 0.02972 | 83.32 | 0 | -347 | 157.6 |
| diurnal_long_sfold3_holdout | forecast_ppo_bc seed2 | false | 20 | 20 | 0 | 0 | 41.52 | 0.2387 | 0.8508 | 0.04653 | 0.85 | 0.001113 | 0.002329 | 0.1 | 0.02975 | 81.08 | 0 | -331.2 | 158.9 |

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
| forecast_ppo_bc seed0 | all | 20 | -3.174 | 2.177 | 11 | 9 |
| forecast_ppo_bc seed0 | initial_budget_infeasible=False (pre-registered) | 18 | -1.814 | 2.134 | 11 | 7 |
| forecast_ppo_bc seed0 | initial_budget_infeasible=True (pre-registered) | 2 | -15.42 | 6.129 | 0 | 2 |
| forecast_ppo_bc seed0 | typical, rule reward >= -500 (post hoc) | 17 | -2.037 | 2.251 | 10 | 7 |
| forecast_ppo_bc seed0 | extreme, rule reward < -500 (post hoc) | 3 | -9.617 | 6.797 | 1 | 2 |
| forecast_ppo_bc seed1 | all | 20 | -16.82 | 2.878 | 3 | 17 |
| forecast_ppo_bc seed1 | initial_budget_infeasible=False (pre-registered) | 18 | -16.9 | 3.173 | 3 | 15 |
| forecast_ppo_bc seed1 | initial_budget_infeasible=True (pre-registered) | 2 | -16.09 | 5.765 | 0 | 2 |
| forecast_ppo_bc seed1 | typical, rule reward >= -500 (post hoc) | 17 | -17.37 | 3.329 | 3 | 14 |
| forecast_ppo_bc seed1 | extreme, rule reward < -500 (post hoc) | 3 | -13.73 | 4.084 | 0 | 3 |
| forecast_ppo_bc seed2 | all | 20 | -1.029 | 2.247 | 12 | 8 |
| forecast_ppo_bc seed2 | initial_budget_infeasible=False (pre-registered) | 18 | 0.5737 | 2.125 | 12 | 6 |
| forecast_ppo_bc seed2 | initial_budget_infeasible=True (pre-registered) | 2 | -15.45 | 6.099 | 0 | 2 |
| forecast_ppo_bc seed2 | typical, rule reward >= -500 (post hoc) | 17 | 1.111 | 2.181 | 12 | 5 |
| forecast_ppo_bc seed2 | extreme, rule reward < -500 (post hoc) | 3 | -13.15 | 4.205 | 0 | 3 |


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

## Paired difference: forecast_ppo_bc seed0 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_sfold3_holdout | total_reward | forecast_ppo_bc seed0 | risk_aware_rule_based | -3.174 | 2.177 | 20 |
| diurnal_long_sfold3_holdout | latency_degradation_mean | forecast_ppo_bc seed0 | risk_aware_rule_based | 0.008504 | 0.005872 | 20 |
| diurnal_long_sfold3_holdout | latency_violation_fraction | forecast_ppo_bc seed0 | risk_aware_rule_based | 0.002734 | 0.001238 | 20 |
| diurnal_long_sfold3_holdout | completion_hours | forecast_ppo_bc seed0 | risk_aware_rule_based | -2.004 | 0.1882 | 20 |
| diurnal_long_sfold3_holdout | mean_delivered_mbps | forecast_ppo_bc seed0 | risk_aware_rule_based | 3.578 | 0.2923 | 20 |
| diurnal_long_sfold3_holdout | p_failure_mean | forecast_ppo_bc seed0 | risk_aware_rule_based | 3.152e-05 | 6.673e-06 | 20 |
| diurnal_long_sfold3_holdout | clip_fraction | forecast_ppo_bc seed0 | risk_aware_rule_based | 0.001108 | 0.001052 | 20 |

## Paired difference: forecast_ppo_bc seed1 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_sfold3_holdout | total_reward | forecast_ppo_bc seed1 | risk_aware_rule_based | -16.82 | 2.878 | 20 |
| diurnal_long_sfold3_holdout | latency_degradation_mean | forecast_ppo_bc seed1 | risk_aware_rule_based | 0.016 | 0.006606 | 20 |
| diurnal_long_sfold3_holdout | latency_violation_fraction | forecast_ppo_bc seed1 | risk_aware_rule_based | 0.005692 | 0.001297 | 20 |
| diurnal_long_sfold3_holdout | completion_hours | forecast_ppo_bc seed1 | risk_aware_rule_based | -3.242 | 0.2851 | 20 |
| diurnal_long_sfold3_holdout | mean_delivered_mbps | forecast_ppo_bc seed1 | risk_aware_rule_based | 5.985 | 0.4315 | 20 |
| diurnal_long_sfold3_holdout | p_failure_mean | forecast_ppo_bc seed1 | risk_aware_rule_based | -2.314e-06 | 1.155e-05 | 20 |
| diurnal_long_sfold3_holdout | clip_fraction | forecast_ppo_bc seed1 | risk_aware_rule_based | 9.629e-05 | 0.0002658 | 20 |

## Paired difference: forecast_ppo_bc seed2 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_sfold3_holdout | total_reward | forecast_ppo_bc seed2 | risk_aware_rule_based | -1.029 | 2.247 | 20 |
| diurnal_long_sfold3_holdout | latency_degradation_mean | forecast_ppo_bc seed2 | risk_aware_rule_based | 0.007234 | 0.00463 | 20 |
| diurnal_long_sfold3_holdout | latency_violation_fraction | forecast_ppo_bc seed2 | risk_aware_rule_based | 0.002137 | 0.001084 | 20 |
| diurnal_long_sfold3_holdout | completion_hours | forecast_ppo_bc seed2 | risk_aware_rule_based | -2.083 | 0.2064 | 20 |
| diurnal_long_sfold3_holdout | mean_delivered_mbps | forecast_ppo_bc seed2 | risk_aware_rule_based | 3.743 | 0.3162 | 20 |
| diurnal_long_sfold3_holdout | p_failure_mean | forecast_ppo_bc seed2 | risk_aware_rule_based | 3.433e-05 | 6.142e-06 | 20 |
| diurnal_long_sfold3_holdout | clip_fraction | forecast_ppo_bc seed2 | risk_aware_rule_based | 0.0001226 | 0.0006461 | 20 |
