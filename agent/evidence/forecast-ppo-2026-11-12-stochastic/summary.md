# Learned controllers against the baselines (simulated)

Simulator configuration `v1,eval-fleet,sigma=0.25`; latency model uncalibrated (sim/LIMITATIONS.md). Workload `data_pipeline/profiles/profile_diurnal_long_holdout.csv` (sha256 `94fe6bd6f8a5cb163cba7a9fafac1274d4ec6511113920a63d4993f03295a53d`). 20 paired scenario seeds (0..19), identical for every policy. Learned policies act by sampling. Reward is the contract reward with contract coefficients.

## Checkpoints

| checkpoint | policy | iteration | trained_on | hazard | train_fleet | sigma | val_gap | val_gap_sem | val_seeds | params |
|---|---|---|---|---|---|---|---|---|---|---|
| agent/evidence/forecast-ppo-2026-11-11/training/bc/seed0/ckpt_best.pt | forecast_ppo_bc seed0 | 40 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | 4.406 | 1.694 | 5000..5039 | bc_accuracy=0.959 |
| agent/evidence/forecast-ppo-2026-11-11/training/bc/seed1/ckpt_best.pt | forecast_ppo_bc seed1 | 80 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | 6.472 | 1.651 | 5000..5039 | bc_accuracy=0.963 |
| agent/evidence/forecast-ppo-2026-11-11/training/bc/seed2/ckpt_best.pt | forecast_ppo_bc seed2 | 100 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | 6.782 | 1.602 | 5000..5039 | bc_accuracy=0.958 |

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
| diurnal_long_holdout | forecast_ppo_bc seed0 | false | 20 | 20 | 0 | 0 | 40.61 | 0.2453 | 0.8615 | 0.04928 | 0.85 | 0.001083 | 0.002322 | 0.1 | 0.02891 | 82.8 | 0 | -343 | 157.9 |
| diurnal_long_holdout | forecast_ppo_bc seed1 | false | 20 | 20 | 0 | 0 | 41.1 | 0.241 | 0.854 | 0.04752 | 0.85 | 0.001082 | 0.002327 | 0.1 | 0.02815 | 81.86 | 0 | -335.4 | 158.6 |
| diurnal_long_holdout | forecast_ppo_bc seed2 | false | 20 | 20 | 0 | 0 | 41.08 | 0.2422 | 0.8557 | 0.04752 | 0.85 | 0.001104 | 0.002731 | 0.1 | 0.02957 | 81.9 | 0 | -335.1 | 158.4 |

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
| forecast_ppo_bc seed0 | all | 20 | -12.81 | 2.578 | 4 | 16 |
| forecast_ppo_bc seed0 | initial_budget_infeasible=False (pre-registered) | 18 | -12.45 | 2.822 | 4 | 14 |
| forecast_ppo_bc seed0 | initial_budget_infeasible=True (pre-registered) | 2 | -16.04 | 5.717 | 0 | 2 |
| forecast_ppo_bc seed0 | typical, rule reward >= -500 (post hoc) | 17 | -12.67 | 2.984 | 4 | 13 |
| forecast_ppo_bc seed0 | extreme, rule reward < -500 (post hoc) | 3 | -13.61 | 4.102 | 0 | 3 |
| forecast_ppo_bc seed1 | all | 20 | -5.208 | 2.59 | 8 | 12 |
| forecast_ppo_bc seed1 | initial_budget_infeasible=False (pre-registered) | 18 | -4.081 | 2.71 | 8 | 10 |
| forecast_ppo_bc seed1 | initial_budget_infeasible=True (pre-registered) | 2 | -15.35 | 6.099 | 0 | 2 |
| forecast_ppo_bc seed1 | typical, rule reward >= -500 (post hoc) | 17 | -3.829 | 2.862 | 8 | 9 |
| forecast_ppo_bc seed1 | extreme, rule reward < -500 (post hoc) | 3 | -13.02 | 4.221 | 0 | 3 |
| forecast_ppo_bc seed2 | all | 20 | -4.906 | 2.686 | 8 | 12 |
| forecast_ppo_bc seed2 | initial_budget_infeasible=False (pre-registered) | 18 | -3.721 | 2.811 | 8 | 10 |
| forecast_ppo_bc seed2 | initial_budget_infeasible=True (pre-registered) | 2 | -15.58 | 5.973 | 0 | 2 |
| forecast_ppo_bc seed2 | typical, rule reward >= -500 (post hoc) | 17 | -4.057 | 2.96 | 7 | 10 |
| forecast_ppo_bc seed2 | extreme, rule reward < -500 (post hoc) | 3 | -9.717 | 6.798 | 1 | 2 |


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

## Paired difference: forecast_ppo_bc seed0 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | forecast_ppo_bc seed0 | risk_aware_rule_based | -12.81 | 2.578 | 20 |
| diurnal_long_holdout | latency_degradation_mean | forecast_ppo_bc seed0 | risk_aware_rule_based | 0.01382 | 0.006591 | 20 |
| diurnal_long_holdout | latency_violation_fraction | forecast_ppo_bc seed0 | risk_aware_rule_based | 0.004887 | 0.001314 | 20 |
| diurnal_long_holdout | completion_hours | forecast_ppo_bc seed0 | risk_aware_rule_based | -2.988 | 0.2658 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | forecast_ppo_bc seed0 | risk_aware_rule_based | 5.468 | 0.4008 | 20 |
| diurnal_long_holdout | p_failure_mean | forecast_ppo_bc seed0 | risk_aware_rule_based | 4.329e-06 | 1.038e-05 | 20 |
| diurnal_long_holdout | clip_fraction | forecast_ppo_bc seed0 | risk_aware_rule_based | -0.0007129 | 0.0005911 | 20 |

## Paired difference: forecast_ppo_bc seed1 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | forecast_ppo_bc seed1 | risk_aware_rule_based | -5.208 | 2.59 | 20 |
| diurnal_long_holdout | latency_degradation_mean | forecast_ppo_bc seed1 | risk_aware_rule_based | 0.009571 | 0.004995 | 20 |
| diurnal_long_holdout | latency_violation_fraction | forecast_ppo_bc seed1 | risk_aware_rule_based | 0.003127 | 0.001149 | 20 |
| diurnal_long_holdout | completion_hours | forecast_ppo_bc seed1 | risk_aware_rule_based | -2.504 | 0.2304 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | forecast_ppo_bc seed1 | risk_aware_rule_based | 4.525 | 0.3394 | 20 |
| diurnal_long_holdout | p_failure_mean | forecast_ppo_bc seed1 | risk_aware_rule_based | 3.803e-06 | 1.134e-05 | 20 |
| diurnal_long_holdout | clip_fraction | forecast_ppo_bc seed1 | risk_aware_rule_based | -0.00147 | 0.001164 | 20 |

## Paired difference: forecast_ppo_bc seed2 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | forecast_ppo_bc seed2 | risk_aware_rule_based | -4.906 | 2.686 | 20 |
| diurnal_long_holdout | latency_degradation_mean | forecast_ppo_bc seed2 | risk_aware_rule_based | 0.01069 | 0.005876 | 20 |
| diurnal_long_holdout | latency_violation_fraction | forecast_ppo_bc seed2 | risk_aware_rule_based | 0.003129 | 0.001262 | 20 |
| diurnal_long_holdout | completion_hours | forecast_ppo_bc seed2 | risk_aware_rule_based | -2.517 | 0.2189 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | forecast_ppo_bc seed2 | risk_aware_rule_based | 4.561 | 0.3277 | 20 |
| diurnal_long_holdout | p_failure_mean | forecast_ppo_bc seed2 | risk_aware_rule_based | 2.524e-05 | 7.614e-06 | 20 |
| diurnal_long_holdout | clip_fraction | forecast_ppo_bc seed2 | risk_aware_rule_based | -6e-05 | 0.0002698 | 20 |
