# Learned controllers against the baselines (simulated)

Simulator configuration `v1,eval-fleet,sigma=0.25`; latency model uncalibrated (sim/LIMITATIONS.md). Workload `runs/selection_check/profile_bursty_long_holdout.csv` (sha256 `7525ddef5ab96687ba2057f3ac3a705eb6b16e24235c44ab44b59664f4676b09`). 20 paired scenario seeds (0..19), identical for every policy. Learned policies act deterministically. Reward is the contract reward with contract coefficients.

## Checkpoints

| checkpoint | policy | iteration | trained_on | hazard | train_fleet | sigma | val_gap | val_gap_sem | val_seeds | params |
|---|---|---|---|---|---|---|---|---|---|---|
| agent/evidence/forecast-ppo-2026-11-11/training/bc/seed0/ckpt_best.pt | forecast_ppo_bc seed0 | 40 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | 4.406 | 1.694 | 5000..5039 | bc_accuracy=0.959 |
| agent/evidence/forecast-ppo-2026-11-11/training/bc/seed1/ckpt_best.pt | forecast_ppo_bc seed1 | 80 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | 6.472 | 1.651 | 5000..5039 | bc_accuracy=0.963 |
| agent/evidence/forecast-ppo-2026-11-11/training/bc/seed2/ckpt_best.pt | forecast_ppo_bc seed2 | 100 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | 6.782 | 1.602 | 5000..5039 | bc_accuracy=0.958 |

## Summary

| profile | policy | workload_synthetic | n | completed | horizon_expired | terminal_failure | completion_h | lat_deg_mean | lat_deg_p95 | viol_frac | lat_budget_met | p_fail_mean | p_fail_max | risk_viol | clip_frac | deliv_mbps | stall_frac | reward | reward_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bursty_long_holdout | fixed_20 | false | 20 | 20 | 0 | 0 | 144.9 | 0.0498 | 0.2946 | 0.01396 | 0.95 | 0.001983 | 0.003874 | 0.1 | 0.1224 | 26.98 | 0.0009459 | -380.3 | 99.48 |
| bursty_long_holdout | fixed_50 | false | 20 | 20 | 0 | 0 | 63.23 | 0.1584 | 0.7664 | 0.06114 | 0.35 | 0.001227 | 0.002382 | 0.1 | 0.08113 | 54.12 | 0.001021 | -651.5 | 104.3 |
| bursty_long_holdout | fixed_80 | false | 20 | 20 | 0 | 0 | 42.01 | 1.399 | 4.635 | 0.3124 | 0.15 | 0.0009762 | 0.001919 | 0.1 | 0.05683 | 79.96 | 0.0009508 | -2413 | 372.9 |
| bursty_long_holdout | fixed_120 | false | 20 | 20 | 0 | 0 | 31.94 | 2.361 | 4.364 | 0.4757 | 0.1 | 0.0008077 | 0.0016 | 0.1 | 0 | 106.1 | 0.0006858 | -3005 | 558 |
| bursty_long_holdout | workload_only_throttle | false | 20 | 20 | 0 | 0 | 45.57 | 0.0936 | 0.4447 | 0.02754 | 0.9 | 0.001303 | 0.002952 | 0.1 | 0.04905 | 79.22 | 0.0005773 | -223.9 | 105.5 |
| bursty_long_holdout | risk_aware_rule_based | false | 20 | 20 | 0 | 0 | 45.57 | 0.0936 | 0.4447 | 0.02754 | 0.9 | 0.001303 | 0.002952 | 0.1 | 0.003108 | 79.22 | 0.0005773 | -223.9 | 105.5 |
| bursty_long_holdout | ORACLE_lookahead (not deployable) | false | 20 | 20 | 0 | 0 | 38.72 | 0.09834 | 0.496 | 0.02775 | 0.9 | 0.001078 | 0.002807 | 0.1 | 0.0001449 | 91.5 | 0.0006134 | -206.7 | 105.7 |
| bursty_long_holdout | rule_seasonal_forecast (hand-built) | false | 20 | 20 | 0 | 0 | 45.9 | 0.09508 | 0.4381 | 0.02831 | 0.9 | 0.001298 | 0.002953 | 0.1 | 0.003177 | 79.29 | 0.0005727 | -227.8 | 105.2 |
| bursty_long_holdout | forecast_ppo_bc seed0 | false | 20 | 20 | 0 | 0 | 40.62 | 0.2093 | 0.9989 | 0.07712 | 0.5 | 0.00105 | 0.002128 | 0.1 | 0.002419 | 85.79 | 0.0006995 | -573.4 | 134.1 |
| bursty_long_holdout | forecast_ppo_bc seed1 | false | 20 | 20 | 0 | 0 | 41.27 | 0.187 | 0.9041 | 0.06844 | 0.55 | 0.00105 | 0.002161 | 0.1 | 0.003026 | 84.91 | 0.0006834 | -516.4 | 122.8 |
| bursty_long_holdout | forecast_ppo_bc seed2 | false | 20 | 20 | 0 | 0 | 41.29 | 0.1951 | 0.8988 | 0.06921 | 0.55 | 0.001053 | 0.00219 | 0.1 | 0.002785 | 84.92 | 0.0006796 | -519.3 | 119.2 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Paired total-reward gap against risk_aware_rule_based, by stratum

`gap` is treatment minus rule, mean over paired scenarios; `sem` its standard error; `n_better` / `n_worse` count scenarios. Strata are defined in the module docstring.

| policy | stratum | n | gap | sem | n_better | n_worse |
|---|---|---|---|---|---|---|
| rule_seasonal_forecast (hand-built) | all | 20 | -3.935 | 2.219 | 5 | 14 |
| rule_seasonal_forecast (hand-built) | initial_budget_infeasible=False (pre-registered) | 18 | -3.836 | 2.438 | 4 | 13 |
| rule_seasonal_forecast (hand-built) | initial_budget_infeasible=True (pre-registered) | 2 | -4.83 | 5.048 | 1 | 1 |
| rule_seasonal_forecast (hand-built) | typical, rule reward >= -500 (post hoc) | 18 | -4.351 | 2.453 | 5 | 12 |
| rule_seasonal_forecast (hand-built) | extreme, rule reward < -500 (post hoc) | 2 | -0.1978 | 0.002172 | 0 | 2 |
| forecast_ppo_bc seed0 | all | 20 | -349.5 | 97.05 | 1 | 19 |
| forecast_ppo_bc seed0 | initial_budget_infeasible=False (pre-registered) | 18 | -382.8 | 105 | 1 | 17 |
| forecast_ppo_bc seed0 | initial_budget_infeasible=True (pre-registered) | 2 | -49.33 | 48.97 | 0 | 2 |
| forecast_ppo_bc seed0 | typical, rule reward >= -500 (post hoc) | 18 | -388.3 | 104 | 0 | 18 |
| forecast_ppo_bc seed0 | extreme, rule reward < -500 (post hoc) | 2 | -0.04317 | 0.2439 | 1 | 1 |
| forecast_ppo_bc seed1 | all | 20 | -292.6 | 78.35 | 1 | 18 |
| forecast_ppo_bc seed1 | initial_budget_infeasible=False (pre-registered) | 18 | -321.7 | 84.37 | 1 | 17 |
| forecast_ppo_bc seed1 | initial_budget_infeasible=True (pre-registered) | 2 | -30.7 | 30.7 | 0 | 1 |
| forecast_ppo_bc seed1 | typical, rule reward >= -500 (post hoc) | 18 | -325.1 | 83.68 | 0 | 17 |
| forecast_ppo_bc seed1 | extreme, rule reward < -500 (post hoc) | 2 | -0.1432 | 0.3439 | 1 | 1 |
| forecast_ppo_bc seed2 | all | 20 | -295.4 | 74.07 | 0 | 20 |
| forecast_ppo_bc seed2 | initial_budget_infeasible=False (pre-registered) | 18 | -320.7 | 80.04 | 0 | 18 |
| forecast_ppo_bc seed2 | initial_budget_infeasible=True (pre-registered) | 2 | -67.33 | 66.97 | 0 | 2 |
| forecast_ppo_bc seed2 | typical, rule reward >= -500 (post hoc) | 18 | -328.1 | 78.64 | 0 | 18 |
| forecast_ppo_bc seed2 | extreme, rule reward < -500 (post hoc) | 2 | -0.778 | 0.001211 | 0 | 2 |


## Paired difference: rule_seasonal_forecast (hand-built) minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| bursty_long_holdout | total_reward | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | -3.935 | 2.219 | 20 |
| bursty_long_holdout | latency_degradation_mean | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 0.001477 | 0.0009404 | 20 |
| bursty_long_holdout | latency_violation_fraction | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 0.0007708 | 0.0004967 | 20 |
| bursty_long_holdout | completion_hours | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 0.3208 | 0.2502 | 20 |
| bursty_long_holdout | mean_delivered_mbps | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 0.0753 | 0.3705 | 20 |
| bursty_long_holdout | p_failure_mean | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | -5.218e-06 | 6.753e-06 | 20 |
| bursty_long_holdout | clip_fraction | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 6.869e-05 | 6.869e-05 | 20 |

## Paired difference: forecast_ppo_bc seed0 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| bursty_long_holdout | total_reward | forecast_ppo_bc seed0 | risk_aware_rule_based | -349.5 | 97.05 | 20 |
| bursty_long_holdout | latency_degradation_mean | forecast_ppo_bc seed0 | risk_aware_rule_based | 0.1157 | 0.02999 | 20 |
| bursty_long_holdout | latency_violation_fraction | forecast_ppo_bc seed0 | risk_aware_rule_based | 0.04957 | 0.01239 | 20 |
| bursty_long_holdout | completion_hours | forecast_ppo_bc seed0 | risk_aware_rule_based | -4.95 | 1.088 | 20 |
| bursty_long_holdout | mean_delivered_mbps | forecast_ppo_bc seed0 | risk_aware_rule_based | 6.577 | 1.342 | 20 |
| bursty_long_holdout | p_failure_mean | forecast_ppo_bc seed0 | risk_aware_rule_based | -0.000253 | 7.494e-05 | 20 |
| bursty_long_holdout | clip_fraction | forecast_ppo_bc seed0 | risk_aware_rule_based | -0.0006888 | 0.0006888 | 20 |

## Paired difference: forecast_ppo_bc seed1 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| bursty_long_holdout | total_reward | forecast_ppo_bc seed1 | risk_aware_rule_based | -292.6 | 78.35 | 20 |
| bursty_long_holdout | latency_degradation_mean | forecast_ppo_bc seed1 | risk_aware_rule_based | 0.0934 | 0.02311 | 20 |
| bursty_long_holdout | latency_violation_fraction | forecast_ppo_bc seed1 | risk_aware_rule_based | 0.0409 | 0.009852 | 20 |
| bursty_long_holdout | completion_hours | forecast_ppo_bc seed1 | risk_aware_rule_based | -4.304 | 0.9598 | 20 |
| bursty_long_holdout | mean_delivered_mbps | forecast_ppo_bc seed1 | risk_aware_rule_based | 5.693 | 1.3 | 20 |
| bursty_long_holdout | p_failure_mean | forecast_ppo_bc seed1 | risk_aware_rule_based | -0.0002529 | 7.14e-05 | 20 |
| bursty_long_holdout | clip_fraction | forecast_ppo_bc seed1 | risk_aware_rule_based | -8.179e-05 | 8.179e-05 | 20 |

## Paired difference: forecast_ppo_bc seed2 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| bursty_long_holdout | total_reward | forecast_ppo_bc seed2 | risk_aware_rule_based | -295.4 | 74.07 | 20 |
| bursty_long_holdout | latency_degradation_mean | forecast_ppo_bc seed2 | risk_aware_rule_based | 0.1015 | 0.02195 | 20 |
| bursty_long_holdout | latency_violation_fraction | forecast_ppo_bc seed2 | risk_aware_rule_based | 0.04167 | 0.00927 | 20 |
| bursty_long_holdout | completion_hours | forecast_ppo_bc seed2 | risk_aware_rule_based | -4.283 | 0.9297 | 20 |
| bursty_long_holdout | mean_delivered_mbps | forecast_ppo_bc seed2 | risk_aware_rule_based | 5.705 | 1.236 | 20 |
| bursty_long_holdout | p_failure_mean | forecast_ppo_bc seed2 | risk_aware_rule_based | -0.0002501 | 7.118e-05 | 20 |
| bursty_long_holdout | clip_fraction | forecast_ppo_bc seed2 | risk_aware_rule_based | -0.000323 | 0.000323 | 20 |
