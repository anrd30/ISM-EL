# Learned controllers against the baselines (simulated)

Simulator configuration `v1,eval-fleet,sigma=0.25`; latency model uncalibrated (sim/LIMITATIONS.md). Workload `data_pipeline/profiles/profile_diurnal_long_sfold2_holdout.csv` (sha256 `a6a9eeae6fd8c4e041b524ba3930cc9a86f9e8df079599c1c21ca2a1cb5cbf17`). 20 paired scenario seeds (0..19), identical for every policy. Learned policies act by their mode (argmax). Reward is the contract reward with contract coefficients.

## Checkpoints

| checkpoint | policy | iteration | trained_on | hazard | train_fleet | sigma | val_gap | val_gap_sem | val_seeds | params |
|---|---|---|---|---|---|---|---|---|---|---|
| runs/residual_q/sfold2/seed0 | residual_q seed0 | 0 | data_pipeline/profiles/profile_diurnal_long_sfold2_train.csv | v1 | train-fleet | 0.25 | 10.15 | 1.375 | 5000..5039 | margin=0.25 (deterministic) |
| runs/residual_q/sfold2/seed1 | residual_q seed1 | 0 | data_pipeline/profiles/profile_diurnal_long_sfold2_train.csv | v1 | train-fleet | 0.25 | 8.445 | 1.414 | 5000..5039 | margin=0.1 (deterministic) |
| runs/residual_q/sfold2/seed2 | residual_q seed2 | 0 | data_pipeline/profiles/profile_diurnal_long_sfold2_train.csv | v1 | train-fleet | 0.25 | 8.775 | 1.438 | 5000..5039 | margin=0.1 (deterministic) |

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
| diurnal_long_sfold2_holdout | residual_q seed0 | false | 20 | 20 | 0 | 0 | 40.15 | 0.2393 | 0.854 | 0.04577 | 0.85 | 0.001063 | 0.002308 | 0.1 | 0.029 | 83.72 | 0 | -322.8 | 157.9 |
| diurnal_long_sfold2_holdout | residual_q seed1 | false | 20 | 20 | 0 | 0 | 38.82 | 0.2435 | 0.8684 | 0.04634 | 0.85 | 0.001018 | 0.002472 | 0.1 | 0.02792 | 86.5 | 0 | -321.7 | 158 |
| diurnal_long_sfold2_holdout | residual_q seed2 | false | 20 | 20 | 0 | 0 | 38.96 | 0.2433 | 0.8629 | 0.04636 | 0.85 | 0.001022 | 0.002512 | 0.1 | 0.02814 | 86.19 | 0 | -322.6 | 158 |

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
| residual_q seed0 | all | 20 | 9.007 | 1.506 | 20 | 0 |
| residual_q seed0 | initial_budget_infeasible=False (pre-registered) | 18 | 9.876 | 1.541 | 18 | 0 |
| residual_q seed0 | initial_budget_infeasible=True (pre-registered) | 2 | 1.186 | 0.02757 | 2 | 0 |
| residual_q seed0 | typical, rule reward >= -500 (post hoc) | 17 | 10.33 | 1.561 | 17 | 0 |
| residual_q seed0 | extreme, rule reward < -500 (post hoc) | 3 | 1.506 | 0.3204 | 3 | 0 |
| residual_q seed1 | all | 20 | 10.08 | 1.584 | 20 | 0 |
| residual_q seed1 | initial_budget_infeasible=False (pre-registered) | 18 | 11.03 | 1.604 | 18 | 0 |
| residual_q seed1 | initial_budget_infeasible=True (pre-registered) | 2 | 1.437 | 0.09305 | 2 | 0 |
| residual_q seed1 | typical, rule reward >= -500 (post hoc) | 17 | 11.55 | 1.613 | 17 | 0 |
| residual_q seed1 | extreme, rule reward < -500 (post hoc) | 3 | 1.744 | 0.3118 | 3 | 0 |
| residual_q seed2 | all | 20 | 9.262 | 1.89 | 19 | 1 |
| residual_q seed2 | initial_budget_infeasible=False (pre-registered) | 18 | 10.13 | 1.999 | 17 | 1 |
| residual_q seed2 | initial_budget_infeasible=True (pre-registered) | 2 | 1.479 | 0.03464 | 2 | 0 |
| residual_q seed2 | typical, rule reward >= -500 (post hoc) | 17 | 10.6 | 2.059 | 16 | 1 |
| residual_q seed2 | extreme, rule reward < -500 (post hoc) | 3 | 1.661 | 0.1826 | 3 | 0 |


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

## Paired difference: residual_q seed0 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_sfold2_holdout | total_reward | residual_q seed0 | risk_aware_rule_based | 9.007 | 1.506 | 20 |
| diurnal_long_sfold2_holdout | latency_degradation_mean | residual_q seed0 | risk_aware_rule_based | 0.004994 | 0.00405 | 20 |
| diurnal_long_sfold2_holdout | latency_violation_fraction | residual_q seed0 | risk_aware_rule_based | 0.0004082 | 0.0008745 | 20 |
| diurnal_long_sfold2_holdout | completion_hours | residual_q seed0 | risk_aware_rule_based | -2.988 | 0.2802 | 20 |
| diurnal_long_sfold2_holdout | mean_delivered_mbps | residual_q seed0 | risk_aware_rule_based | 5.538 | 0.4456 | 20 |
| diurnal_long_sfold2_holdout | p_failure_mean | residual_q seed0 | risk_aware_rule_based | -9.844e-07 | 6.958e-06 | 20 |
| diurnal_long_sfold2_holdout | clip_fraction | residual_q seed0 | risk_aware_rule_based | 0.0005643 | 0.0003987 | 20 |

## Paired difference: residual_q seed1 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_sfold2_holdout | total_reward | residual_q seed1 | risk_aware_rule_based | 10.08 | 1.584 | 20 |
| diurnal_long_sfold2_holdout | latency_degradation_mean | residual_q seed1 | risk_aware_rule_based | 0.009224 | 0.005671 | 20 |
| diurnal_long_sfold2_holdout | latency_violation_fraction | residual_q seed1 | risk_aware_rule_based | 0.0009787 | 0.001183 | 20 |
| diurnal_long_sfold2_holdout | completion_hours | residual_q seed1 | risk_aware_rule_based | -4.313 | 0.3651 | 20 |
| diurnal_long_sfold2_holdout | mean_delivered_mbps | residual_q seed1 | risk_aware_rule_based | 8.319 | 0.6187 | 20 |
| diurnal_long_sfold2_holdout | p_failure_mean | residual_q seed1 | risk_aware_rule_based | -4.614e-05 | 1.41e-05 | 20 |
| diurnal_long_sfold2_holdout | clip_fraction | residual_q seed1 | risk_aware_rule_based | -0.0005123 | 0.0006453 | 20 |

## Paired difference: residual_q seed2 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_sfold2_holdout | total_reward | residual_q seed2 | risk_aware_rule_based | 9.262 | 1.89 | 20 |
| diurnal_long_sfold2_holdout | latency_degradation_mean | residual_q seed2 | risk_aware_rule_based | 0.009017 | 0.005353 | 20 |
| diurnal_long_sfold2_holdout | latency_violation_fraction | residual_q seed2 | risk_aware_rule_based | 0.0009909 | 0.001123 | 20 |
| diurnal_long_sfold2_holdout | completion_hours | residual_q seed2 | risk_aware_rule_based | -4.171 | 0.3632 | 20 |
| diurnal_long_sfold2_holdout | mean_delivered_mbps | residual_q seed2 | risk_aware_rule_based | 8.006 | 0.6029 | 20 |
| diurnal_long_sfold2_holdout | p_failure_mean | residual_q seed2 | risk_aware_rule_based | -4.198e-05 | 1.32e-05 | 20 |
| diurnal_long_sfold2_holdout | clip_fraction | residual_q seed2 | risk_aware_rule_based | -0.0003004 | 0.0002421 | 20 |
