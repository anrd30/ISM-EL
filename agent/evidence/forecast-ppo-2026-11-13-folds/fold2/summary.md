# Learned controllers against the baselines (simulated)

Simulator configuration `v1,eval-fleet,sigma=0.25`; latency model uncalibrated (sim/LIMITATIONS.md). Workload `data_pipeline/profiles/profile_diurnal_long_sfold2_holdout.csv` (sha256 `a6a9eeae6fd8c4e041b524ba3930cc9a86f9e8df079599c1c21ca2a1cb5cbf17`). 20 paired scenario seeds (0..19), identical for every policy. Learned policies act by sampling. Reward is the contract reward with contract coefficients.

## Checkpoints

| checkpoint | policy | iteration | trained_on | hazard | train_fleet | sigma | val_gap | val_gap_sem | val_seeds | params |
|---|---|---|---|---|---|---|---|---|---|---|
| runs/forecast_ppo_sfold2/bc/seed0/ckpt_best.pt | forecast_ppo_bc seed0 | 120 | data_pipeline/profiles/profile_diurnal_long_sfold2_train.csv | v1 | train-fleet | 0.25 | 8.567 | 1.683 | 5000..5039 | bc_accuracy=0.955 |
| runs/forecast_ppo_sfold2/bc/seed1/ckpt_best.pt | forecast_ppo_bc seed1 | 120 | data_pipeline/profiles/profile_diurnal_long_sfold2_train.csv | v1 | train-fleet | 0.25 | 5.204 | 2.169 | 5000..5039 | bc_accuracy=0.956 |
| runs/forecast_ppo_sfold2/bc/seed2/ckpt_best.pt | forecast_ppo_bc seed2 | 120 | data_pipeline/profiles/profile_diurnal_long_sfold2_train.csv | v1 | train-fleet | 0.25 | 7.042 | 1.954 | 5000..5039 | bc_accuracy=0.959 |

## Summary

| profile | policy | workload_synthetic | n | completed | horizon_expired | terminal_failure | completion_h | lat_deg_mean | lat_deg_p95 | viol_frac | lat_budget_met | p_fail_mean | p_fail_max | risk_viol | clip_frac | deliv_mbps | stall_frac | reward | reward_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_sfold2_holdout | fixed_20 | false | 20 | 20 | 0 | 0 | 144.8 | 0.1423 | 0.85 | 0.02773 | 0.85 | 0.001984 | 0.003874 | 0.1 | 0.123 | 26.9 | 0 | -446.3 | 149.3 |
| diurnal_long_sfold2_holdout | fixed_50 | false | 20 | 20 | 0 | 0 | 63.08 | 0.1807 | 0.85 | 0.03521 | 0.85 | 0.001231 | 0.002381 | 0.1 | 0.08202 | 54.12 | 0 | -348.2 | 156.7 |
| diurnal_long_sfold2_holdout | fixed_80 | false | 20 | 20 | 0 | 0 | 40.8 | 0.2619 | 1.098 | 0.06872 | 0.85 | 0.0009776 | 0.00192 | 0.1 | 0.05819 | 81.99 | 0 | -473.8 | 148.2 |
| diurnal_long_sfold2_holdout | fixed_120 | false | 20 | 20 | 0 | 0 | 29.2 | 3.449 | 5.667 | 0.6701 | 0 | 0.0008026 | 0.0016 | 0.1 | 0 | 114.3 | 0 | -3594 | 92.93 |
| diurnal_long_sfold2_holdout | workload_only_throttle | false | 20 | 20 | 0 | 0 | 43.13 | 0.2343 | 0.85 | 0.04537 | 0.85 | 0.001064 | 0.002298 | 0.1 | 0.06392 | 78.19 | 0 | -331.8 | 157.3 |
| diurnal_long_sfold2_holdout | risk_aware_rule_based | false | 20 | 20 | 0 | 0 | 43.13 | 0.2343 | 0.85 | 0.04537 | 0.85 | 0.001064 | 0.002298 | 0.1 | 0.02844 | 78.19 | 0 | -331.8 | 157.3 |
| diurnal_long_sfold2_holdout | ORACLE_lookahead (not deployable) | false | 20 | 20 | 0 | 0 | 36.74 | 0.2459 | 0.8815 | 0.04708 | 0.85 | 0.0009532 | 0.002289 | 0.1 | 0.02543 | 91.21 | 0 | -318.4 | 158.3 |
| diurnal_long_sfold2_holdout | rule_seasonal_forecast (hand-built) | false | 20 | 20 | 0 | 0 | 40.39 | 0.2399 | 0.8528 | 0.04655 | 0.85 | 0.001069 | 0.002298 | 0.1 | 0.02922 | 83.26 | 0 | -328.2 | 157.5 |
| diurnal_long_sfold2_holdout | forecast_ppo_bc seed0 | false | 20 | 20 | 0 | 0 | 41.71 | 0.2402 | 0.8501 | 0.04728 | 0.85 | 0.001106 | 0.002374 | 0.1 | 0.02789 | 80.73 | 0 | -335.9 | 157.8 |
| diurnal_long_sfold2_holdout | forecast_ppo_bc seed1 | false | 20 | 20 | 0 | 0 | 40.58 | 0.2442 | 0.8644 | 0.04904 | 0.85 | 0.001086 | 0.002969 | 0.1 | 0.02821 | 82.9 | 0 | -341.1 | 157.6 |
| diurnal_long_sfold2_holdout | forecast_ppo_bc seed2 | false | 20 | 20 | 0 | 0 | 40.95 | 0.2425 | 0.8529 | 0.04874 | 0.85 | 0.001087 | 0.002303 | 0.1 | 0.0307 | 82.13 | 0 | -340.7 | 157.9 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Paired total-reward gap against risk_aware_rule_based, by stratum

`gap` is treatment minus rule, mean over paired scenarios; `sem` its standard error; `n_better` / `n_worse` count scenarios. Strata are defined in the module docstring.

| policy | stratum | n | gap | sem | n_better | n_worse |
|---|---|---|---|---|---|---|
| rule_seasonal_forecast (hand-built) | all | 20 | 3.576 | 0.2591 | 20 | 0 |
| rule_seasonal_forecast (hand-built) | initial_budget_infeasible=False (pre-registered) | 18 | 3.847 | 0.2003 | 18 | 0 |
| rule_seasonal_forecast (hand-built) | initial_budget_infeasible=True (pre-registered) | 2 | 1.136 | 0.02243 | 2 | 0 |
| rule_seasonal_forecast (hand-built) | typical, rule reward >= -500 (post hoc) | 17 | 3.943 | 0.1864 | 17 | 0 |
| rule_seasonal_forecast (hand-built) | extreme, rule reward < -500 (post hoc) | 3 | 1.495 | 0.3588 | 3 | 0 |
| forecast_ppo_bc seed0 | all | 20 | -4.04 | 2.666 | 8 | 12 |
| forecast_ppo_bc seed0 | initial_budget_infeasible=False (pre-registered) | 18 | -2.76 | 2.765 | 8 | 10 |
| forecast_ppo_bc seed0 | initial_budget_infeasible=True (pre-registered) | 2 | -15.56 | 5.748 | 0 | 2 |
| forecast_ppo_bc seed0 | typical, rule reward >= -500 (post hoc) | 17 | -3.042 | 2.917 | 7 | 10 |
| forecast_ppo_bc seed0 | extreme, rule reward < -500 (post hoc) | 3 | -9.696 | 6.736 | 1 | 2 |
| forecast_ppo_bc seed1 | all | 20 | -9.27 | 3.536 | 8 | 12 |
| forecast_ppo_bc seed1 | initial_budget_infeasible=False (pre-registered) | 18 | -7.964 | 3.705 | 8 | 10 |
| forecast_ppo_bc seed1 | initial_budget_infeasible=True (pre-registered) | 2 | -21.02 | 11.05 | 0 | 2 |
| forecast_ppo_bc seed1 | typical, rule reward >= -500 (post hoc) | 17 | -8.538 | 3.883 | 7 | 10 |
| forecast_ppo_bc seed1 | extreme, rule reward < -500 (post hoc) | 3 | -13.42 | 9.927 | 1 | 2 |
| forecast_ppo_bc seed2 | all | 20 | -8.858 | 3.292 | 7 | 13 |
| forecast_ppo_bc seed2 | initial_budget_infeasible=False (pre-registered) | 18 | -7.503 | 3.518 | 7 | 11 |
| forecast_ppo_bc seed2 | initial_budget_infeasible=True (pre-registered) | 2 | -21.05 | 0.4423 | 0 | 2 |
| forecast_ppo_bc seed2 | typical, rule reward >= -500 (post hoc) | 17 | -8.064 | 3.683 | 6 | 11 |
| forecast_ppo_bc seed2 | extreme, rule reward < -500 (post hoc) | 3 | -13.36 | 7.701 | 1 | 2 |


## Paired difference: rule_seasonal_forecast (hand-built) minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_sfold2_holdout | total_reward | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 3.576 | 0.2591 | 20 |
| diurnal_long_sfold2_holdout | latency_degradation_mean | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 0.005567 | 0.003565 | 20 |
| diurnal_long_sfold2_holdout | latency_violation_fraction | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 0.001182 | 0.0006935 | 20 |
| diurnal_long_sfold2_holdout | completion_hours | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | -2.746 | 0.2596 | 20 |
| diurnal_long_sfold2_holdout | mean_delivered_mbps | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 5.07 | 0.4124 | 20 |
| diurnal_long_sfold2_holdout | p_failure_mean | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 5.657e-06 | 5.881e-06 | 20 |
| diurnal_long_sfold2_holdout | clip_fraction | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 0.0007839 | 0.000543 | 20 |

## Paired difference: forecast_ppo_bc seed0 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_sfold2_holdout | total_reward | forecast_ppo_bc seed0 | risk_aware_rule_based | -4.04 | 2.666 | 20 |
| diurnal_long_sfold2_holdout | latency_degradation_mean | forecast_ppo_bc seed0 | risk_aware_rule_based | 0.005926 | 0.003678 | 20 |
| diurnal_long_sfold2_holdout | latency_violation_fraction | forecast_ppo_bc seed0 | risk_aware_rule_based | 0.001911 | 0.000909 | 20 |
| diurnal_long_sfold2_holdout | completion_hours | forecast_ppo_bc seed0 | risk_aware_rule_based | -1.421 | 0.2094 | 20 |
| diurnal_long_sfold2_holdout | mean_delivered_mbps | forecast_ppo_bc seed0 | risk_aware_rule_based | 2.544 | 0.3435 | 20 |
| diurnal_long_sfold2_holdout | p_failure_mean | forecast_ppo_bc seed0 | risk_aware_rule_based | 4.173e-05 | 6.921e-06 | 20 |
| diurnal_long_sfold2_holdout | clip_fraction | forecast_ppo_bc seed0 | risk_aware_rule_based | -0.0005465 | 0.0006497 | 20 |

## Paired difference: forecast_ppo_bc seed1 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_sfold2_holdout | total_reward | forecast_ppo_bc seed1 | risk_aware_rule_based | -9.27 | 3.536 | 20 |
| diurnal_long_sfold2_holdout | latency_degradation_mean | forecast_ppo_bc seed1 | risk_aware_rule_based | 0.00986 | 0.004591 | 20 |
| diurnal_long_sfold2_holdout | latency_violation_fraction | forecast_ppo_bc seed1 | risk_aware_rule_based | 0.003676 | 0.001159 | 20 |
| diurnal_long_sfold2_holdout | completion_hours | forecast_ppo_bc seed1 | risk_aware_rule_based | -2.554 | 0.2813 | 20 |
| diurnal_long_sfold2_holdout | mean_delivered_mbps | forecast_ppo_bc seed1 | risk_aware_rule_based | 4.715 | 0.4415 | 20 |
| diurnal_long_sfold2_holdout | p_failure_mean | forecast_ppo_bc seed1 | risk_aware_rule_based | 2.217e-05 | 7.543e-06 | 20 |
| diurnal_long_sfold2_holdout | clip_fraction | forecast_ppo_bc seed1 | risk_aware_rule_based | -0.0002275 | 0.0005991 | 20 |

## Paired difference: forecast_ppo_bc seed2 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_sfold2_holdout | total_reward | forecast_ppo_bc seed2 | risk_aware_rule_based | -8.858 | 3.292 | 20 |
| diurnal_long_sfold2_holdout | latency_degradation_mean | forecast_ppo_bc seed2 | risk_aware_rule_based | 0.008143 | 0.004526 | 20 |
| diurnal_long_sfold2_holdout | latency_violation_fraction | forecast_ppo_bc seed2 | risk_aware_rule_based | 0.003373 | 0.001185 | 20 |
| diurnal_long_sfold2_holdout | completion_hours | forecast_ppo_bc seed2 | risk_aware_rule_based | -2.179 | 0.2855 | 20 |
| diurnal_long_sfold2_holdout | mean_delivered_mbps | forecast_ppo_bc seed2 | risk_aware_rule_based | 3.949 | 0.4739 | 20 |
| diurnal_long_sfold2_holdout | p_failure_mean | forecast_ppo_bc seed2 | risk_aware_rule_based | 2.329e-05 | 7.74e-06 | 20 |
| diurnal_long_sfold2_holdout | clip_fraction | forecast_ppo_bc seed2 | risk_aware_rule_based | 0.002261 | 0.002376 | 20 |
