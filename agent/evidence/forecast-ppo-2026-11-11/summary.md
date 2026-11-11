# Learned controllers against the baselines (simulated)

Simulator configuration `v1,eval-fleet,sigma=0.25`; latency model uncalibrated (sim/LIMITATIONS.md). Workload `data_pipeline/profiles/profile_diurnal_long_holdout.csv` (sha256 `2c83b9db2e80b0a85afc996ea7df525692f89b331d07eec26c14011bd06a59b1`). 20 paired scenario seeds (0..19), identical for every policy. Learned policies act deterministically. Reward is the contract reward with contract coefficients.

## Checkpoints

| checkpoint | policy | iteration | trained_on | hazard | train_fleet | sigma | val_gap | val_gap_sem | val_seeds | params |
|---|---|---|---|---|---|---|---|---|---|---|
| runs/forecast_ppo/bc/seed0/ckpt_best.pt | forecast_ppo_bc seed0 | 40 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | 4.406 | 1.694 | 5000..5039 | bc_accuracy=0.959 |
| runs/forecast_ppo/bc/seed1/ckpt_best.pt | forecast_ppo_bc seed1 | 80 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | 6.472 | 1.651 | 5000..5039 | bc_accuracy=0.963 |
| runs/forecast_ppo/bc/seed2/ckpt_best.pt | forecast_ppo_bc seed2 | 100 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | 6.782 | 1.602 | 5000..5039 | bc_accuracy=0.958 |
| runs/forecast_ppo/bc/seed0/ckpt_init.pt | forecast_bc_only seed0 | 0 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | -3.64 | 2.178 | 5000..5039 | bc_accuracy=0.959 |
| runs/forecast_ppo/bc/seed1/ckpt_init.pt | forecast_bc_only seed1 | 0 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | 3.833 | 1.782 | 5000..5039 | bc_accuracy=0.963 |
| runs/forecast_ppo/bc/seed2/ckpt_init.pt | forecast_bc_only seed2 | 0 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | -0.63 | 2.095 | 5000..5039 | bc_accuracy=0.958 |
| runs/forecast_ppo/scratch/seed0/ckpt_best.pt | forecast_ppo_scratch seed0 | 60 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | -13.63 | 1.141 | 5000..5039 |  |
| runs/forecast_ppo/scratch/seed1/ckpt_best.pt | forecast_ppo_scratch seed1 | 40 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | -13.63 | 1.141 | 5000..5039 |  |
| runs/forecast_ppo/scratch/seed2/ckpt_best.pt | forecast_ppo_scratch seed2 | 10 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | -13.63 | 1.141 | 5000..5039 |  |

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
| diurnal_long_holdout | forecast_ppo_bc seed0 | false | 20 | 20 | 0 | 0 | 41.41 | 0.2384 | 0.8506 | 0.04564 | 0.85 | 0.001114 | 0.002344 | 0.1 | 0.02857 | 81.28 | 0 | -324 | 158.4 |
| diurnal_long_holdout | forecast_ppo_bc seed1 | false | 20 | 20 | 0 | 0 | 41.6 | 0.2368 | 0.8504 | 0.04545 | 0.85 | 0.001117 | 0.002345 | 0.1 | 0.0282 | 80.92 | 0 | -323.9 | 158.4 |
| diurnal_long_holdout | forecast_ppo_bc seed2 | false | 20 | 20 | 0 | 0 | 41.47 | 0.238 | 0.851 | 0.04548 | 0.85 | 0.001116 | 0.002319 | 0.1 | 0.02986 | 81.19 | 0 | -323.5 | 158.4 |
| diurnal_long_holdout | forecast_bc_only seed0 | false | 20 | 20 | 0 | 0 | 41.18 | 0.2409 | 0.8558 | 0.04669 | 0.85 | 0.001116 | 0.002344 | 0.1 | 0.02894 | 81.74 | 0 | -329.1 | 158.7 |
| diurnal_long_holdout | forecast_bc_only seed1 | false | 20 | 20 | 0 | 0 | 41.3 | 0.2395 | 0.8522 | 0.04597 | 0.85 | 0.001107 | 0.002324 | 0.1 | 0.0283 | 81.54 | 0 | -325.8 | 158.2 |
| diurnal_long_holdout | forecast_bc_only seed2 | false | 20 | 20 | 0 | 0 | 41.02 | 0.242 | 0.862 | 0.04738 | 0.85 | 0.001114 | 0.002317 | 0.1 | 0.02903 | 82.01 | 0 | -333 | 158.3 |
| diurnal_long_holdout | forecast_ppo_scratch seed0 | false | 20 | 20 | 0 | 0 | 63.09 | 0.1814 | 0.85 | 0.0352 | 0.85 | 0.00123 | 0.002451 | 0.1 | 0.03183 | 54.12 | 0 | -348.3 | 157.2 |
| diurnal_long_holdout | forecast_ppo_scratch seed1 | false | 20 | 20 | 0 | 0 | 63.08 | 0.1832 | 0.85 | 0.03551 | 0.85 | 0.001229 | 0.002381 | 0.1 | 0.02292 | 54.13 | 0 | -350.6 | 158 |
| diurnal_long_holdout | forecast_ppo_scratch seed2 | false | 20 | 20 | 0 | 0 | 63.08 | 0.1814 | 0.85 | 0.0352 | 0.85 | 0.00123 | 0.002381 | 0.1 | 0.03183 | 54.12 | 0 | -348.3 | 157.2 |

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
| forecast_ppo_bc seed0 | all | 20 | 6.241 | 1.419 | 19 | 1 |
| forecast_ppo_bc seed0 | initial_budget_infeasible=False (pre-registered) | 18 | 6.856 | 1.509 | 17 | 1 |
| forecast_ppo_bc seed0 | initial_budget_infeasible=True (pre-registered) | 2 | 0.6988 | 0.2772 | 2 | 0 |
| forecast_ppo_bc seed0 | typical, rule reward >= -500 (post hoc) | 17 | 7.15 | 1.57 | 16 | 1 |
| forecast_ppo_bc seed0 | extreme, rule reward < -500 (post hoc) | 3 | 1.088 | 0.4212 | 3 | 0 |
| forecast_ppo_bc seed1 | all | 20 | 6.324 | 1.423 | 19 | 1 |
| forecast_ppo_bc seed1 | initial_budget_infeasible=False (pre-registered) | 18 | 6.931 | 1.516 | 17 | 1 |
| forecast_ppo_bc seed1 | initial_budget_infeasible=True (pre-registered) | 2 | 0.865 | 0.2307 | 2 | 0 |
| forecast_ppo_bc seed1 | typical, rule reward >= -500 (post hoc) | 17 | 7.231 | 1.576 | 16 | 1 |
| forecast_ppo_bc seed1 | extreme, rule reward < -500 (post hoc) | 3 | 1.184 | 0.3455 | 3 | 0 |
| forecast_ppo_bc seed2 | all | 20 | 6.715 | 1.262 | 20 | 0 |
| forecast_ppo_bc seed2 | initial_budget_infeasible=False (pre-registered) | 18 | 7.365 | 1.315 | 18 | 0 |
| forecast_ppo_bc seed2 | initial_budget_infeasible=True (pre-registered) | 2 | 0.865 | 0.3307 | 2 | 0 |
| forecast_ppo_bc seed2 | typical, rule reward >= -500 (post hoc) | 17 | 7.661 | 1.359 | 17 | 0 |
| forecast_ppo_bc seed2 | extreme, rule reward < -500 (post hoc) | 3 | 1.353 | 0.5244 | 3 | 0 |
| forecast_bc_only seed0 | all | 20 | 1.07 | 2.123 | 13 | 7 |
| forecast_bc_only seed0 | initial_budget_infeasible=False (pre-registered) | 18 | 2.283 | 2.175 | 13 | 5 |
| forecast_bc_only seed0 | initial_budget_infeasible=True (pre-registered) | 2 | -9.845 | 0.2336 | 0 | 2 |
| forecast_bc_only seed0 | typical, rule reward >= -500 (post hoc) | 17 | 2.329 | 2.307 | 12 | 5 |
| forecast_bc_only seed0 | extreme, rule reward < -500 (post hoc) | 3 | -6.062 | 3.786 | 1 | 2 |
| forecast_bc_only seed1 | all | 20 | 4.443 | 1.811 | 18 | 2 |
| forecast_bc_only seed1 | initial_budget_infeasible=False (pre-registered) | 18 | 4.865 | 1.992 | 16 | 2 |
| forecast_bc_only seed1 | initial_budget_infeasible=True (pre-registered) | 2 | 0.6454 | 0.3307 | 2 | 0 |
| forecast_bc_only seed1 | typical, rule reward >= -500 (post hoc) | 17 | 5.034 | 2.105 | 15 | 2 |
| forecast_bc_only seed1 | extreme, rule reward < -500 (post hoc) | 3 | 1.089 | 0.4833 | 3 | 0 |
| forecast_bc_only seed2 | all | 20 | -2.828 | 2.62 | 11 | 9 |
| forecast_bc_only seed2 | initial_budget_infeasible=False (pre-registered) | 18 | -2.048 | 2.858 | 11 | 7 |
| forecast_bc_only seed2 | initial_budget_infeasible=True (pre-registered) | 2 | -9.845 | 0.2341 | 0 | 2 |
| forecast_bc_only seed2 | typical, rule reward >= -500 (post hoc) | 17 | -2.273 | 3.022 | 10 | 7 |
| forecast_bc_only seed2 | extreme, rule reward < -500 (post hoc) | 3 | -5.973 | 3.874 | 1 | 2 |
| forecast_ppo_scratch seed0 | all | 20 | -18.08 | 1.46 | 0 | 20 |
| forecast_ppo_scratch seed0 | initial_budget_infeasible=False (pre-registered) | 18 | -19.06 | 1.439 | 0 | 18 |
| forecast_ppo_scratch seed0 | initial_budget_infeasible=True (pre-registered) | 2 | -9.259 | 1.32 | 0 | 2 |
| forecast_ppo_scratch seed0 | typical, rule reward >= -500 (post hoc) | 17 | -19.44 | 1.471 | 0 | 17 |
| forecast_ppo_scratch seed0 | extreme, rule reward < -500 (post hoc) | 3 | -10.36 | 1.342 | 0 | 3 |
| forecast_ppo_scratch seed1 | all | 20 | -20.42 | 2.404 | 0 | 20 |
| forecast_ppo_scratch seed1 | initial_budget_infeasible=False (pre-registered) | 18 | -19.05 | 1.438 | 0 | 18 |
| forecast_ppo_scratch seed1 | initial_budget_infeasible=True (pre-registered) | 2 | -32.71 | 24.77 | 0 | 2 |
| forecast_ppo_scratch seed1 | typical, rule reward >= -500 (post hoc) | 17 | -19.44 | 1.471 | 0 | 17 |
| forecast_ppo_scratch seed1 | extreme, rule reward < -500 (post hoc) | 3 | -26 | 15.8 | 0 | 3 |
| forecast_ppo_scratch seed2 | all | 20 | -18.07 | 1.459 | 0 | 20 |
| forecast_ppo_scratch seed2 | initial_budget_infeasible=False (pre-registered) | 18 | -19.05 | 1.438 | 0 | 18 |
| forecast_ppo_scratch seed2 | initial_budget_infeasible=True (pre-registered) | 2 | -9.259 | 1.32 | 0 | 2 |
| forecast_ppo_scratch seed2 | typical, rule reward >= -500 (post hoc) | 17 | -19.44 | 1.471 | 0 | 17 |
| forecast_ppo_scratch seed2 | extreme, rule reward < -500 (post hoc) | 3 | -10.36 | 1.342 | 0 | 3 |


## PPO-trained policy against its own imitation-only starting point

Same network before and after PPO, so this gap is what PPO training added.

| treatment | comparator | n | gap | sem |
|---|---|---|---|---|
| forecast_ppo_bc seed0 | forecast_bc_only seed0 | 20 | 5.17 | 1.241 |
| forecast_ppo_bc seed1 | forecast_bc_only seed1 | 20 | 1.881 | 0.9317 |
| forecast_ppo_bc seed2 | forecast_bc_only seed2 | 20 | 9.542 | 2.135 |

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
| diurnal_long_holdout | total_reward | forecast_ppo_bc seed0 | risk_aware_rule_based | 6.241 | 1.419 | 20 |
| diurnal_long_holdout | latency_degradation_mean | forecast_ppo_bc seed0 | risk_aware_rule_based | 0.00696 | 0.005905 | 20 |
| diurnal_long_holdout | latency_violation_fraction | forecast_ppo_bc seed0 | risk_aware_rule_based | 0.001244 | 0.001127 | 20 |
| diurnal_long_holdout | completion_hours | forecast_ppo_bc seed0 | risk_aware_rule_based | -2.188 | 0.2058 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | forecast_ppo_bc seed0 | risk_aware_rule_based | 3.948 | 0.3135 | 20 |
| diurnal_long_holdout | p_failure_mean | forecast_ppo_bc seed0 | risk_aware_rule_based | 3.573e-05 | 8.177e-06 | 20 |
| diurnal_long_holdout | clip_fraction | forecast_ppo_bc seed0 | risk_aware_rule_based | -0.001059 | 0.0009388 | 20 |

## Paired difference: forecast_ppo_bc seed1 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | forecast_ppo_bc seed1 | risk_aware_rule_based | 6.324 | 1.423 | 20 |
| diurnal_long_holdout | latency_degradation_mean | forecast_ppo_bc seed1 | risk_aware_rule_based | 0.005354 | 0.005442 | 20 |
| diurnal_long_holdout | latency_violation_fraction | forecast_ppo_bc seed1 | risk_aware_rule_based | 0.001055 | 0.001039 | 20 |
| diurnal_long_holdout | completion_hours | forecast_ppo_bc seed1 | risk_aware_rule_based | -1.996 | 0.2253 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | forecast_ppo_bc seed1 | risk_aware_rule_based | 3.588 | 0.3533 | 20 |
| diurnal_long_holdout | p_failure_mean | forecast_ppo_bc seed1 | risk_aware_rule_based | 3.807e-05 | 7.08e-06 | 20 |
| diurnal_long_holdout | clip_fraction | forecast_ppo_bc seed1 | risk_aware_rule_based | -0.001429 | 0.001155 | 20 |

## Paired difference: forecast_ppo_bc seed2 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | forecast_ppo_bc seed2 | risk_aware_rule_based | 6.715 | 1.262 | 20 |
| diurnal_long_holdout | latency_degradation_mean | forecast_ppo_bc seed2 | risk_aware_rule_based | 0.006523 | 0.005758 | 20 |
| diurnal_long_holdout | latency_violation_fraction | forecast_ppo_bc seed2 | risk_aware_rule_based | 0.001082 | 0.001108 | 20 |
| diurnal_long_holdout | completion_hours | forecast_ppo_bc seed2 | risk_aware_rule_based | -2.133 | 0.2308 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | forecast_ppo_bc seed2 | risk_aware_rule_based | 3.856 | 0.3553 | 20 |
| diurnal_long_holdout | p_failure_mean | forecast_ppo_bc seed2 | risk_aware_rule_based | 3.772e-05 | 8.959e-06 | 20 |
| diurnal_long_holdout | clip_fraction | forecast_ppo_bc seed2 | risk_aware_rule_based | 0.0002372 | 0.0002252 | 20 |

## Paired difference: forecast_bc_only seed0 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | forecast_bc_only seed0 | risk_aware_rule_based | 1.07 | 2.123 | 20 |
| diurnal_long_holdout | latency_degradation_mean | forecast_bc_only seed0 | risk_aware_rule_based | 0.009379 | 0.006448 | 20 |
| diurnal_long_holdout | latency_violation_fraction | forecast_bc_only seed0 | risk_aware_rule_based | 0.002292 | 0.00135 | 20 |
| diurnal_long_holdout | completion_hours | forecast_bc_only seed0 | risk_aware_rule_based | -2.417 | 0.2168 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | forecast_bc_only seed0 | risk_aware_rule_based | 4.401 | 0.322 | 20 |
| diurnal_long_holdout | p_failure_mean | forecast_bc_only seed0 | risk_aware_rule_based | 3.731e-05 | 7.797e-06 | 20 |
| diurnal_long_holdout | clip_fraction | forecast_bc_only seed0 | risk_aware_rule_based | -0.0006823 | 0.0006442 | 20 |

## Paired difference: forecast_bc_only seed1 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | forecast_bc_only seed1 | risk_aware_rule_based | 4.443 | 1.811 | 20 |
| diurnal_long_holdout | latency_degradation_mean | forecast_bc_only seed1 | risk_aware_rule_based | 0.008051 | 0.005933 | 20 |
| diurnal_long_holdout | latency_violation_fraction | forecast_bc_only seed1 | risk_aware_rule_based | 0.001578 | 0.001138 | 20 |
| diurnal_long_holdout | completion_hours | forecast_bc_only seed1 | risk_aware_rule_based | -2.304 | 0.2298 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | forecast_bc_only seed1 | risk_aware_rule_based | 4.207 | 0.3493 | 20 |
| diurnal_long_holdout | p_failure_mean | forecast_bc_only seed1 | risk_aware_rule_based | 2.859e-05 | 1.012e-05 | 20 |
| diurnal_long_holdout | clip_fraction | forecast_bc_only seed1 | risk_aware_rule_based | -0.001322 | 0.001084 | 20 |

## Paired difference: forecast_bc_only seed2 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | forecast_bc_only seed2 | risk_aware_rule_based | -2.828 | 2.62 | 20 |
| diurnal_long_holdout | latency_degradation_mean | forecast_bc_only seed2 | risk_aware_rule_based | 0.01056 | 0.006132 | 20 |
| diurnal_long_holdout | latency_violation_fraction | forecast_bc_only seed2 | risk_aware_rule_based | 0.002983 | 0.001284 | 20 |
| diurnal_long_holdout | completion_hours | forecast_bc_only seed2 | risk_aware_rule_based | -2.583 | 0.2328 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | forecast_bc_only seed2 | risk_aware_rule_based | 4.68 | 0.3475 | 20 |
| diurnal_long_holdout | p_failure_mean | forecast_bc_only seed2 | risk_aware_rule_based | 3.501e-05 | 8.632e-06 | 20 |
| diurnal_long_holdout | clip_fraction | forecast_bc_only seed2 | risk_aware_rule_based | -0.0005953 | 0.0005175 | 20 |

## Paired difference: forecast_ppo_scratch seed0 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | forecast_ppo_scratch seed0 | risk_aware_rule_based | -18.08 | 1.46 | 20 |
| diurnal_long_holdout | latency_degradation_mean | forecast_ppo_scratch seed0 | risk_aware_rule_based | -0.05004 | 0.02554 | 20 |
| diurnal_long_holdout | latency_violation_fraction | forecast_ppo_scratch seed0 | risk_aware_rule_based | -0.009196 | 0.00493 | 20 |
| diurnal_long_holdout | completion_hours | forecast_ppo_scratch seed0 | risk_aware_rule_based | 19.49 | 1.063 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | forecast_ppo_scratch seed0 | risk_aware_rule_based | -23.21 | 0.5959 | 20 |
| diurnal_long_holdout | p_failure_mean | forecast_ppo_scratch seed0 | risk_aware_rule_based | 0.0001517 | 2.121e-05 | 20 |
| diurnal_long_holdout | clip_fraction | forecast_ppo_scratch seed0 | risk_aware_rule_based | 0.002201 | 0.003125 | 20 |

## Paired difference: forecast_ppo_scratch seed1 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | forecast_ppo_scratch seed1 | risk_aware_rule_based | -20.42 | 2.404 | 20 |
| diurnal_long_holdout | latency_degradation_mean | forecast_ppo_scratch seed1 | risk_aware_rule_based | -0.04827 | 0.0249 | 20 |
| diurnal_long_holdout | latency_violation_fraction | forecast_ppo_scratch seed1 | risk_aware_rule_based | -0.008881 | 0.004804 | 20 |
| diurnal_long_holdout | completion_hours | forecast_ppo_scratch seed1 | risk_aware_rule_based | 19.48 | 1.065 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | forecast_ppo_scratch seed1 | risk_aware_rule_based | -23.2 | 0.5976 | 20 |
| diurnal_long_holdout | p_failure_mean | forecast_ppo_scratch seed1 | risk_aware_rule_based | 0.0001508 | 2.177e-05 | 20 |
| diurnal_long_holdout | clip_fraction | forecast_ppo_scratch seed1 | risk_aware_rule_based | -0.006703 | 0.004652 | 20 |

## Paired difference: forecast_ppo_scratch seed2 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | forecast_ppo_scratch seed2 | risk_aware_rule_based | -18.07 | 1.459 | 20 |
| diurnal_long_holdout | latency_degradation_mean | forecast_ppo_scratch seed2 | risk_aware_rule_based | -0.05004 | 0.02554 | 20 |
| diurnal_long_holdout | latency_violation_fraction | forecast_ppo_scratch seed2 | risk_aware_rule_based | -0.009196 | 0.00493 | 20 |
| diurnal_long_holdout | completion_hours | forecast_ppo_scratch seed2 | risk_aware_rule_based | 19.48 | 1.063 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | forecast_ppo_scratch seed2 | risk_aware_rule_based | -23.21 | 0.5962 | 20 |
| diurnal_long_holdout | p_failure_mean | forecast_ppo_scratch seed2 | risk_aware_rule_based | 0.0001516 | 2.122e-05 | 20 |
| diurnal_long_holdout | clip_fraction | forecast_ppo_scratch seed2 | risk_aware_rule_based | 0.002201 | 0.003125 | 20 |
