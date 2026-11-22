# Learned controllers against the baselines (simulated)

Simulator configuration `v1,eval-fleet,sigma=0.25`; latency model uncalibrated (sim/LIMITATIONS.md). Workload `data_pipeline/profiles/profile_diurnal_long_sfold1_holdout.csv` (sha256 `d9d949023440eeebedb9ec89287ae5845aca4675a32efd019f5925d59f3bfdf1`). 20 paired scenario seeds (0..19), identical for every policy. Learned policies act by their mode (argmax). Reward is the contract reward with contract coefficients.

## Checkpoints

| checkpoint | policy | iteration | trained_on | hazard | train_fleet | sigma | val_gap | val_gap_sem | val_seeds | params |
|---|---|---|---|---|---|---|---|---|---|---|
| runs/residual_q/sfold1/seed0 | residual_q seed0 | 0 | data_pipeline/profiles/profile_diurnal_long_sfold1_train.csv | v1 | train-fleet | 0.25 | 6.368 | 1.363 | 5000..5039 | margin=0.25 (deterministic) |
| runs/residual_q/sfold1/seed1 | residual_q seed1 | 0 | data_pipeline/profiles/profile_diurnal_long_sfold1_train.csv | v1 | train-fleet | 0.25 | 6.228 | 1.268 | 5000..5039 | margin=0.5 (deterministic) |
| runs/residual_q/sfold1/seed2 | residual_q seed2 | 0 | data_pipeline/profiles/profile_diurnal_long_sfold1_train.csv | v1 | train-fleet | 0.25 | 6.047 | 1.132 | 5000..5039 | margin=0.1 (deterministic) |

## Summary

| profile | policy | workload_synthetic | n | completed | horizon_expired | terminal_failure | completion_h | lat_deg_mean | lat_deg_p95 | viol_frac | lat_budget_met | p_fail_mean | p_fail_max | risk_viol | clip_frac | deliv_mbps | stall_frac | reward | reward_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_sfold1_holdout | fixed_20 | false | 20 | 20 | 0 | 0 | 144.8 | 0.139 | 0.85 | 0.02749 | 0.85 | 0.001984 | 0.003874 | 0.1 | 0.1229 | 26.91 | 0 | -442.4 | 147 |
| diurnal_long_sfold1_holdout | fixed_50 | false | 20 | 20 | 0 | 0 | 63.08 | 0.1764 | 0.85 | 0.03492 | 0.85 | 0.00123 | 0.002381 | 0.1 | 0.08176 | 54.13 | 0 | -344.3 | 154.4 |
| diurnal_long_sfold1_holdout | fixed_80 | false | 20 | 20 | 0 | 0 | 40.79 | 0.251 | 1.077 | 0.06575 | 0.85 | 0.0009773 | 0.001919 | 0.1 | 0.05819 | 82.02 | 0 | -453.3 | 147.3 |
| diurnal_long_sfold1_holdout | fixed_120 | false | 20 | 20 | 0 | 0 | 29.13 | 3.427 | 5.667 | 0.6684 | 0 | 0.0008024 | 0.0016 | 0.1 | 0 | 114.6 | 0 | -3572 | 90.66 |
| diurnal_long_sfold1_holdout | workload_only_throttle | false | 20 | 20 | 0 | 0 | 42.91 | 0.2277 | 0.85 | 0.04448 | 0.85 | 0.001068 | 0.002299 | 0.1 | 0.06099 | 78.51 | 0 | -325.5 | 155.2 |
| diurnal_long_sfold1_holdout | risk_aware_rule_based | false | 20 | 20 | 0 | 0 | 42.91 | 0.2277 | 0.85 | 0.04448 | 0.85 | 0.001068 | 0.002299 | 0.1 | 0.02744 | 78.51 | 0 | -325.5 | 155.2 |
| diurnal_long_sfold1_holdout | ORACLE_lookahead (not deployable) | false | 20 | 20 | 0 | 0 | 36.67 | 0.2395 | 0.8724 | 0.04657 | 0.85 | 0.0009524 | 0.002294 | 0.1 | 0.02527 | 91.41 | 0 | -313.8 | 156 |
| diurnal_long_sfold1_holdout | rule_seasonal_forecast (hand-built) | false | 20 | 20 | 0 | 0 | 40.37 | 0.2343 | 0.8501 | 0.04589 | 0.85 | 0.001067 | 0.002299 | 0.1 | 0.02815 | 83.32 | 0 | -322.1 | 155.4 |
| diurnal_long_sfold1_holdout | residual_q seed0 | false | 20 | 20 | 0 | 0 | 40.11 | 0.2349 | 0.85 | 0.0456 | 0.85 | 0.001061 | 0.002314 | 0.1 | 0.02806 | 83.84 | 0 | -318.3 | 155.7 |
| diurnal_long_sfold1_holdout | residual_q seed1 | false | 20 | 20 | 0 | 0 | 40.42 | 0.2338 | 0.85 | 0.04553 | 0.85 | 0.00107 | 0.002307 | 0.1 | 0.02823 | 83.2 | 0 | -319.1 | 155.7 |
| diurnal_long_sfold1_holdout | residual_q seed2 | false | 20 | 20 | 0 | 0 | 38.68 | 0.2388 | 0.8606 | 0.04603 | 0.85 | 0.001013 | 0.002442 | 0.1 | 0.02704 | 86.84 | 0 | -317.3 | 155.8 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Paired total-reward gap against risk_aware_rule_based, by stratum

`gap` is treatment minus rule, mean over paired scenarios; `sem` its standard error; `n_better` / `n_worse` count scenarios. Strata are defined in the module docstring.

| policy | stratum | n | gap | sem | n_better | n_worse |
|---|---|---|---|---|---|---|
| rule_seasonal_forecast (hand-built) | all | 20 | 3.406 | 0.2471 | 20 | 0 |
| rule_seasonal_forecast (hand-built) | initial_budget_infeasible=False (pre-registered) | 18 | 3.66 | 0.1949 | 18 | 0 |
| rule_seasonal_forecast (hand-built) | initial_budget_infeasible=True (pre-registered) | 2 | 1.124 | 0.03032 | 2 | 0 |
| rule_seasonal_forecast (hand-built) | typical, rule reward >= -500 (post hoc) | 17 | 3.753 | 0.1814 | 17 | 0 |
| rule_seasonal_forecast (hand-built) | extreme, rule reward < -500 (post hoc) | 3 | 1.44 | 0.3163 | 3 | 0 |
| residual_q seed0 | all | 20 | 7.2 | 1.235 | 20 | 0 |
| residual_q seed0 | initial_budget_infeasible=False (pre-registered) | 18 | 7.855 | 1.281 | 18 | 0 |
| residual_q seed0 | initial_budget_infeasible=True (pre-registered) | 2 | 1.308 | 0.1532 | 2 | 0 |
| residual_q seed0 | typical, rule reward >= -500 (post hoc) | 17 | 8.213 | 1.305 | 17 | 0 |
| residual_q seed0 | extreme, rule reward < -500 (post hoc) | 3 | 1.461 | 0.1768 | 3 | 0 |
| residual_q seed1 | all | 20 | 6.418 | 1.13 | 20 | 0 |
| residual_q seed1 | initial_budget_infeasible=False (pre-registered) | 18 | 6.997 | 1.178 | 18 | 0 |
| residual_q seed1 | initial_budget_infeasible=True (pre-registered) | 2 | 1.208 | 0.05319 | 2 | 0 |
| residual_q seed1 | typical, rule reward >= -500 (post hoc) | 17 | 7.294 | 1.209 | 17 | 0 |
| residual_q seed1 | extreme, rule reward < -500 (post hoc) | 3 | 1.453 | 0.2471 | 3 | 0 |
| residual_q seed2 | all | 20 | 8.203 | 1.29 | 20 | 0 |
| residual_q seed2 | initial_budget_infeasible=False (pre-registered) | 18 | 8.989 | 1.305 | 18 | 0 |
| residual_q seed2 | initial_budget_infeasible=True (pre-registered) | 2 | 1.133 | 0.1788 | 2 | 0 |
| residual_q seed2 | typical, rule reward >= -500 (post hoc) | 17 | 9.43 | 1.302 | 17 | 0 |
| residual_q seed2 | extreme, rule reward < -500 (post hoc) | 3 | 1.254 | 0.159 | 3 | 0 |


## Paired difference: rule_seasonal_forecast (hand-built) minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_sfold1_holdout | total_reward | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 3.406 | 0.2471 | 20 |
| diurnal_long_sfold1_holdout | latency_degradation_mean | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 0.006661 | 0.004008 | 20 |
| diurnal_long_sfold1_holdout | latency_violation_fraction | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 0.001418 | 0.0007681 | 20 |
| diurnal_long_sfold1_holdout | completion_hours | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | -2.542 | 0.2205 | 20 |
| diurnal_long_sfold1_holdout | mean_delivered_mbps | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 4.802 | 0.3663 | 20 |
| diurnal_long_sfold1_holdout | p_failure_mean | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | -1.048e-06 | 7.14e-06 | 20 |
| diurnal_long_sfold1_holdout | clip_fraction | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 0.000711 | 0.0004922 | 20 |

## Paired difference: residual_q seed0 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_sfold1_holdout | total_reward | residual_q seed0 | risk_aware_rule_based | 7.2 | 1.235 | 20 |
| diurnal_long_sfold1_holdout | latency_degradation_mean | residual_q seed0 | risk_aware_rule_based | 0.007276 | 0.005226 | 20 |
| diurnal_long_sfold1_holdout | latency_violation_fraction | residual_q seed0 | risk_aware_rule_based | 0.001123 | 0.00105 | 20 |
| diurnal_long_sfold1_holdout | completion_hours | residual_q seed0 | risk_aware_rule_based | -2.8 | 0.2227 | 20 |
| diurnal_long_sfold1_holdout | mean_delivered_mbps | residual_q seed0 | risk_aware_rule_based | 5.327 | 0.3539 | 20 |
| diurnal_long_sfold1_holdout | p_failure_mean | residual_q seed0 | risk_aware_rule_based | -6.522e-06 | 8.652e-06 | 20 |
| diurnal_long_sfold1_holdout | clip_fraction | residual_q seed0 | risk_aware_rule_based | 0.0006202 | 0.0004271 | 20 |

## Paired difference: residual_q seed1 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_sfold1_holdout | total_reward | residual_q seed1 | risk_aware_rule_based | 6.418 | 1.13 | 20 |
| diurnal_long_sfold1_holdout | latency_degradation_mean | residual_q seed1 | risk_aware_rule_based | 0.006151 | 0.004595 | 20 |
| diurnal_long_sfold1_holdout | latency_violation_fraction | residual_q seed1 | risk_aware_rule_based | 0.001052 | 0.0009223 | 20 |
| diurnal_long_sfold1_holdout | completion_hours | residual_q seed1 | risk_aware_rule_based | -2.483 | 0.2137 | 20 |
| diurnal_long_sfold1_holdout | mean_delivered_mbps | residual_q seed1 | risk_aware_rule_based | 4.684 | 0.3467 | 20 |
| diurnal_long_sfold1_holdout | p_failure_mean | residual_q seed1 | risk_aware_rule_based | 2.003e-06 | 7.09e-06 | 20 |
| diurnal_long_sfold1_holdout | clip_fraction | residual_q seed1 | risk_aware_rule_based | 0.0007954 | 0.0005525 | 20 |

## Paired difference: residual_q seed2 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_sfold1_holdout | total_reward | residual_q seed2 | risk_aware_rule_based | 8.203 | 1.29 | 20 |
| diurnal_long_sfold1_holdout | latency_degradation_mean | residual_q seed2 | risk_aware_rule_based | 0.01112 | 0.006408 | 20 |
| diurnal_long_sfold1_holdout | latency_violation_fraction | residual_q seed2 | risk_aware_rule_based | 0.001553 | 0.001275 | 20 |
| diurnal_long_sfold1_holdout | completion_hours | residual_q seed2 | risk_aware_rule_based | -4.229 | 0.3328 | 20 |
| diurnal_long_sfold1_holdout | mean_delivered_mbps | residual_q seed2 | risk_aware_rule_based | 8.327 | 0.5692 | 20 |
| diurnal_long_sfold1_holdout | p_failure_mean | residual_q seed2 | risk_aware_rule_based | -5.442e-05 | 1.841e-05 | 20 |
| diurnal_long_sfold1_holdout | clip_fraction | residual_q seed2 | risk_aware_rule_based | -0.0003971 | 0.0002808 | 20 |
