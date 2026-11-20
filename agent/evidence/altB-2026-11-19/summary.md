# Learned controllers against the baselines (simulated)

Simulator configuration `v1,eval-fleet,sigma=0.25`; latency model uncalibrated (sim/LIMITATIONS.md). Workload `data_pipeline/profiles/profile_diurnal_long_altB.csv` (sha256 `bf19b148bdd47546d1029e8365dfc530dae6635f402ebfa814782ac7073acd7e`). 20 paired scenario seeds (0..19), identical for every policy. Learned policies act by their mode (argmax). Reward is the contract reward with contract coefficients.

## Checkpoints

| checkpoint | policy | iteration | trained_on | hazard | train_fleet | sigma | val_gap | val_gap_sem | val_seeds | params |
|---|---|---|---|---|---|---|---|---|---|---|
| agent/evidence/residual-q-2026-11-14/training/main/seed0 | residual_q seed0 | 0 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | 10.45 | 1.412 | 5000..5039 | margin=0.1 (deterministic) |
| agent/evidence/residual-q-2026-11-14/training/main/seed1 | residual_q seed1 | 0 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | 6.936 | 1.224 | 5000..5039 | margin=0.25 (deterministic) |
| agent/evidence/residual-q-2026-11-14/training/main/seed2 | residual_q seed2 | 0 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | 7.025 | 1.608 | 5000..5039 | margin=0.1 (deterministic) |

## Summary

| profile | policy | workload_synthetic | n | completed | horizon_expired | terminal_failure | completion_h | lat_deg_mean | lat_deg_p95 | viol_frac | lat_budget_met | p_fail_mean | p_fail_max | risk_viol | clip_frac | deliv_mbps | stall_frac | reward | reward_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_altB | fixed_20 | false | 20 | 20 | 0 | 0 | 144.8 | 0.04454 | 0.3256 | 0.01189 | 0.9 | 0.001984 | 0.003874 | 0.1 | 0.1223 | 26.98 | 0 | -272.8 | 60.88 |
| diurnal_long_altB | fixed_50 | false | 20 | 20 | 0 | 0 | 63.01 | 0.05561 | 0.3465 | 0.01494 | 0.9 | 0.00123 | 0.002381 | 0.1 | 0.08106 | 54.26 | 0 | -174.7 | 66.94 |
| diurnal_long_altB | fixed_80 | false | 20 | 20 | 0 | 0 | 40.75 | 0.1312 | 0.7013 | 0.05725 | 0.6 | 0.0009778 | 0.001919 | 0.1 | 0.0569 | 82.2 | 0 | -374.8 | 77.33 |
| diurnal_long_altB | fixed_120 | false | 20 | 20 | 0 | 0 | 29 | 2.38 | 5.667 | 0.5432 | 0 | 0.0007958 | 0.0016 | 0.1 | 0 | 115.2 | 0 | -2766 | 148.3 |
| diurnal_long_altB | workload_only_throttle | false | 20 | 20 | 0 | 0 | 39.68 | 0.0774 | 0.395 | 0.02261 | 0.9 | 0.001029 | 0.002224 | 0.1 | 0.04221 | 84.87 | 0 | -172.9 | 67.8 |
| diurnal_long_altB | risk_aware_rule_based | false | 20 | 20 | 0 | 0 | 39.68 | 0.0774 | 0.395 | 0.02261 | 0.9 | 0.001029 | 0.002224 | 0.1 | 0.02149 | 84.87 | 0 | -172.9 | 67.8 |
| diurnal_long_altB | ORACLE_lookahead (not deployable) | false | 20 | 20 | 0 | 0 | 35.5 | 0.07352 | 0.431 | 0.01891 | 0.9 | 0.0009611 | 0.002201 | 0.1 | 0.01189 | 94.38 | 0 | -143.7 | 68.58 |
| diurnal_long_altB | rule_seasonal_forecast (hand-built) | false | 20 | 20 | 0 | 0 | 39.36 | 0.0769 | 0.4033 | 0.02207 | 0.9 | 0.001028 | 0.002224 | 0.1 | 0.02161 | 85.51 | 0 | -168.5 | 67.55 |
| diurnal_long_altB | residual_q seed0 | false | 20 | 20 | 0 | 0 | 39.27 | 0.09205 | 0.4612 | 0.03192 | 0.9 | 0.001046 | 0.003152 | 0.1 | 0.02283 | 85.63 | 0 | -223.7 | 65.11 |
| diurnal_long_altB | residual_q seed1 | false | 20 | 20 | 0 | 0 | 38.85 | 0.08655 | 0.4332 | 0.02894 | 0.9 | 0.001027 | 0.002294 | 0.1 | 0.02153 | 86.52 | 0 | -204.8 | 66.43 |
| diurnal_long_altB | residual_q seed2 | false | 20 | 20 | 0 | 0 | 39.18 | 0.09248 | 0.4669 | 0.03264 | 0.9 | 0.001052 | 0.003191 | 0.1 | 0.02342 | 85.89 | 0 | -226.7 | 65.74 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Paired total-reward gap against risk_aware_rule_based, by stratum

`gap` is treatment minus rule, mean over paired scenarios; `sem` its standard error; `n_better` / `n_worse` count scenarios. Strata are defined in the module docstring.

| policy | stratum | n | gap | sem | n_better | n_worse |
|---|---|---|---|---|---|---|
| rule_seasonal_forecast (hand-built) | all | 20 | 4.459 | 2.008 | 15 | 5 |
| rule_seasonal_forecast (hand-built) | initial_budget_infeasible=False (pre-registered) | 18 | 4.328 | 2.188 | 14 | 4 |
| rule_seasonal_forecast (hand-built) | initial_budget_infeasible=True (pre-registered) | 2 | 5.64 | 5.672 | 1 | 1 |
| rule_seasonal_forecast (hand-built) | typical, rule reward >= -500 (post hoc) | 18 | 4.328 | 2.188 | 14 | 4 |
| rule_seasonal_forecast (hand-built) | extreme, rule reward < -500 (post hoc) | 2 | 5.64 | 5.672 | 1 | 1 |
| residual_q seed0 | all | 20 | -50.76 | 7.346 | 1 | 19 |
| residual_q seed0 | initial_budget_infeasible=False (pre-registered) | 18 | -55.79 | 7.203 | 0 | 18 |
| residual_q seed0 | initial_budget_infeasible=True (pre-registered) | 2 | -5.42 | 5.485 | 1 | 1 |
| residual_q seed0 | typical, rule reward >= -500 (post hoc) | 18 | -55.79 | 7.203 | 0 | 18 |
| residual_q seed0 | extreme, rule reward < -500 (post hoc) | 2 | -5.42 | 5.485 | 1 | 1 |
| residual_q seed1 | all | 20 | -31.92 | 7.567 | 2 | 18 |
| residual_q seed1 | initial_budget_infeasible=False (pre-registered) | 18 | -34.9 | 8.108 | 2 | 16 |
| residual_q seed1 | initial_budget_infeasible=True (pre-registered) | 2 | -5.17 | 5.164 | 0 | 2 |
| residual_q seed1 | typical, rule reward >= -500 (post hoc) | 18 | -34.9 | 8.108 | 2 | 16 |
| residual_q seed1 | extreme, rule reward < -500 (post hoc) | 2 | -5.17 | 5.164 | 0 | 2 |
| residual_q seed2 | all | 20 | -53.74 | 9.677 | 2 | 18 |
| residual_q seed2 | initial_budget_infeasible=False (pre-registered) | 18 | -58.52 | 10.1 | 1 | 17 |
| residual_q seed2 | initial_budget_infeasible=True (pre-registered) | 2 | -10.69 | 10.76 | 1 | 1 |
| residual_q seed2 | typical, rule reward >= -500 (post hoc) | 18 | -58.52 | 10.1 | 1 | 17 |
| residual_q seed2 | extreme, rule reward < -500 (post hoc) | 2 | -10.69 | 10.76 | 1 | 1 |


## Paired difference: rule_seasonal_forecast (hand-built) minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_altB | total_reward | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 4.459 | 2.008 | 20 |
| diurnal_long_altB | latency_degradation_mean | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | -0.000496 | 0.001031 | 20 |
| diurnal_long_altB | latency_violation_fraction | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | -0.0005413 | 0.0003327 | 20 |
| diurnal_long_altB | completion_hours | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | -0.3167 | 0.1844 | 20 |
| diurnal_long_altB | mean_delivered_mbps | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 0.6368 | 0.3544 | 20 |
| diurnal_long_altB | p_failure_mean | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | -1.608e-06 | 2.188e-06 | 20 |
| diurnal_long_altB | clip_fraction | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 0.0001157 | 0.0001157 | 20 |

## Paired difference: residual_q seed0 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_altB | total_reward | residual_q seed0 | risk_aware_rule_based | -50.76 | 7.346 | 20 |
| diurnal_long_altB | latency_degradation_mean | residual_q seed0 | risk_aware_rule_based | 0.01465 | 0.001967 | 20 |
| diurnal_long_altB | latency_violation_fraction | residual_q seed0 | risk_aware_rule_based | 0.009314 | 0.00122 | 20 |
| diurnal_long_altB | completion_hours | residual_q seed0 | risk_aware_rule_based | -0.4083 | 0.2582 | 20 |
| diurnal_long_altB | mean_delivered_mbps | residual_q seed0 | risk_aware_rule_based | 0.7583 | 0.4761 | 20 |
| diurnal_long_altB | p_failure_mean | residual_q seed0 | risk_aware_rule_based | 1.657e-05 | 6.139e-06 | 20 |
| diurnal_long_altB | clip_fraction | residual_q seed0 | risk_aware_rule_based | 0.001333 | 0.001896 | 20 |

## Paired difference: residual_q seed1 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_altB | total_reward | residual_q seed1 | risk_aware_rule_based | -31.92 | 7.567 | 20 |
| diurnal_long_altB | latency_degradation_mean | residual_q seed1 | risk_aware_rule_based | 0.009153 | 0.002975 | 20 |
| diurnal_long_altB | latency_violation_fraction | residual_q seed1 | risk_aware_rule_based | 0.006326 | 0.001268 | 20 |
| diurnal_long_altB | completion_hours | residual_q seed1 | risk_aware_rule_based | -0.8333 | 0.1945 | 20 |
| diurnal_long_altB | mean_delivered_mbps | residual_q seed1 | risk_aware_rule_based | 1.652 | 0.3388 | 20 |
| diurnal_long_altB | p_failure_mean | residual_q seed1 | risk_aware_rule_based | -2.699e-06 | 3.497e-06 | 20 |
| diurnal_long_altB | clip_fraction | residual_q seed1 | risk_aware_rule_based | 3.485e-05 | 0.00105 | 20 |

## Paired difference: residual_q seed2 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_altB | total_reward | residual_q seed2 | risk_aware_rule_based | -53.74 | 9.677 | 20 |
| diurnal_long_altB | latency_degradation_mean | residual_q seed2 | risk_aware_rule_based | 0.01508 | 0.003261 | 20 |
| diurnal_long_altB | latency_violation_fraction | residual_q seed2 | risk_aware_rule_based | 0.01004 | 0.001609 | 20 |
| diurnal_long_altB | completion_hours | residual_q seed2 | risk_aware_rule_based | -0.5 | 0.1966 | 20 |
| diurnal_long_altB | mean_delivered_mbps | residual_q seed2 | risk_aware_rule_based | 1.016 | 0.3686 | 20 |
| diurnal_long_altB | p_failure_mean | residual_q seed2 | risk_aware_rule_based | 2.289e-05 | 5.179e-06 | 20 |
| diurnal_long_altB | clip_fraction | residual_q seed2 | risk_aware_rule_based | 0.001929 | 0.001863 | 20 |
