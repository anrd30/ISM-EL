# Baseline comparison (simulated)

Simulator configuration `eval-fleet,sigma=0.25`; latency model uncalibrated (sim/LIMITATIONS.md); workload provenance is listed under Workloads. 20 paired scenario seeds (0..19), identical for every policy. `completion_h` is the mean over completed episodes only. `lat_budget_met` and `risk_viol` are fractions of episodes; `viol_frac`, `clip_frac` and `stall_frac` are fractions of intervals. Latency terms are the mean-latency model of the simulator; `p_fail_*` are the filter's exposure for the applied allocation (nominal rate).

## Workloads

- `diurnal_long_altB`: source `data_pipeline/profiles/profile_diurnal_long_altB.csv`, synthetic: false
  - sha256: `bf19b148bdd47546d1029e8365dfc530dae6635f402ebfa814782ac7073acd7e`
  - disclosure: selected extremes, not a representative sample

## Scenario blocks

`filter_forced`: the safety filter raises a `fixed_20` rollout's applied action on at least 10% of the intervals of that scenario. Controllable scenarios are the rest. Both are properties of the scenario, identical for every policy.

| profile | definition | n_forced | n | forced_seeds |
|---|---|---|---|---|
| diurnal_long_altB | primary | 4 | 20 | 3 9 17 19 |
| diurnal_long_altB | clip_0.01 | 7 | 20 | 2 3 9 16 17 18 19 |
| diurnal_long_altB | clip_0.25 | 4 | 20 | 3 9 17 19 |
| diurnal_long_altB | fixed20_cost | 3 | 20 | 9 17 19 |

## All scenarios (20 profile-seed scenarios)

| profile | policy | workload_synthetic | n | completed | horizon_expired | terminal_failure | completion_h | lat_deg_mean | lat_deg_p95 | viol_frac | lat_budget_met | p_fail_mean | p_fail_max | risk_viol | clip_frac | deliv_mbps | stall_frac | reward | reward_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_altB | fixed_20 | false | 20 | 20 | 0 | 0 | 144.8 | 0.04454 | 0.3256 | 0.01189 | 0.9 | 0.001984 | 0.003874 | 0.1 | 0.1223 | 26.98 | 0 | -272.8 | 60.88 |
| diurnal_long_altB | fixed_50 | false | 20 | 20 | 0 | 0 | 63.01 | 0.05561 | 0.3465 | 0.01494 | 0.9 | 0.00123 | 0.002381 | 0.1 | 0.08106 | 54.26 | 0 | -174.7 | 66.94 |
| diurnal_long_altB | fixed_80 | false | 20 | 20 | 0 | 0 | 40.75 | 0.1312 | 0.7013 | 0.05725 | 0.6 | 0.0009778 | 0.001919 | 0.1 | 0.0569 | 82.2 | 0 | -374.8 | 77.33 |
| diurnal_long_altB | fixed_120 | false | 20 | 20 | 0 | 0 | 29 | 2.38 | 5.667 | 0.5432 | 0 | 0.0007958 | 0.0016 | 0.1 | 0 | 115.2 | 0 | -2766 | 148.3 |
| diurnal_long_altB | workload_only_throttle | false | 20 | 20 | 0 | 0 | 39.68 | 0.0774 | 0.395 | 0.02261 | 0.9 | 0.001029 | 0.002224 | 0.1 | 0.04221 | 84.87 | 0 | -172.9 | 67.8 |
| diurnal_long_altB | risk_aware_rule_based | false | 20 | 20 | 0 | 0 | 39.68 | 0.0774 | 0.395 | 0.02261 | 0.9 | 0.001029 | 0.002224 | 0.1 | 0.02149 | 84.87 | 0 | -172.9 | 67.8 |
| diurnal_long_altB | ORACLE_lookahead (not deployable) | false | 20 | 20 | 0 | 0 | 35.5 | 0.07352 | 0.431 | 0.01891 | 0.9 | 0.0009611 | 0.002201 | 0.1 | 0.01189 | 94.38 | 0 | -143.7 | 68.58 |
| diurnal_long_altB | slowest_risk_feasible (literal spec) | false | 20 | 20 | 0 | 0 | 144.8 | 0.04454 | 0.3256 | 0.01189 | 0.9 | 0.001984 | 0.003874 | 0.1 | 0.03292 | 26.98 | 0 | -272.8 | 60.88 |
| diurnal_long_altB | risk_aware_ceiling_persistence (literal spec) | false | 20 | 20 | 0 | 0 | 34.29 | 0.1942 | 0.9763 | 0.08051 | 0.05 | 0.0009367 | 0.002137 | 0.1 | 0.00755 | 97.58 | 0 | -447.5 | 58.41 |
| diurnal_long_altB | risk_aware_rule_based k=1 | false | 20 | 20 | 0 | 0 | 39.2 | 0.168 | 1.029 | 0.0487 | 0.8 | 0.0009691 | 0.002073 | 0.1 | 0.0004938 | 86.21 | 0 | -309.5 | 108.9 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Controllable scenarios (16 profile-seed scenarios)

| profile | policy | workload_synthetic | n | completed | horizon_expired | terminal_failure | completion_h | lat_deg_mean | lat_deg_p95 | viol_frac | lat_budget_met | p_fail_mean | p_fail_max | risk_viol | clip_frac | deliv_mbps | stall_frac | reward | reward_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_altB | fixed_20 | false | 16 | 16 | 0 | 0 | 164.3 | 0 | 0 | 0 | 1 | 0.001601 | 0.003205 | 0 | 0.01049 | 20.31 | 0 | -197.1 | 1.706 |
| diurnal_long_altB | fixed_50 | false | 16 | 16 | 0 | 0 | 66.67 | 0 | 0 | 0 | 1 | 0.0006709 | 0.00134 | 0 | 0 | 50 | 0 | -80 | 0 |
| diurnal_long_altB | fixed_80 | false | 16 | 16 | 0 | 0 | 41.71 | 0.07535 | 0.4125 | 0.04493 | 0.625 | 0.0004193 | 0.0008376 | 0 | 0 | 80 | 0 | -312.8 | 47.84 |
| diurnal_long_altB | fixed_120 | false | 16 | 16 | 0 | 0 | 29.09 | 2.452 | 5.667 | 0.5526 | 0 | 0.0002792 | 0.0005585 | 0 | 0 | 114.8 | 0 | -2833 | 162.6 |
| diurnal_long_altB | workload_only_throttle | false | 16 | 16 | 0 | 0 | 40.86 | 0.01185 | 0 | 0.004845 | 1 | 0.0004553 | 0.001143 | 0 | 0 | 82.03 | 0 | -78.58 | 2.954 |
| diurnal_long_altB | risk_aware_rule_based | false | 16 | 16 | 0 | 0 | 40.86 | 0.01185 | 0 | 0.004845 | 1 | 0.0004553 | 0.001143 | 0 | 0 | 82.03 | 0 | -78.58 | 2.954 |
| diurnal_long_altB | ORACLE_lookahead (not deployable) | false | 16 | 16 | 0 | 0 | 36.05 | 0.004949 | 0.03858 | 0 | 1 | 0.0003907 | 0.001122 | 0 | 0 | 92.75 | 0 | -45.41 | 0.5742 |
| diurnal_long_altB | slowest_risk_feasible (literal spec) | false | 16 | 16 | 0 | 0 | 164.3 | 0 | 0 | 0 | 1 | 0.001601 | 0.003205 | 0 | 0 | 20.31 | 0 | -197.1 | 1.706 |
| diurnal_long_altB | risk_aware_ceiling_persistence (literal spec) | false | 16 | 16 | 0 | 0 | 34.7 | 0.1375 | 0.6462 | 0.06701 | 0.0625 | 0.0003663 | 0.001051 | 0 | 0 | 96.3 | 0 | -379 | 17.28 |
| diurnal_long_altB | risk_aware_rule_based k=1 | false | 16 | 16 | 0 | 0 | 40.86 | 0.01185 | 0 | 0.004845 | 1 | 0.0004553 | 0.001143 | 0 | 0 | 82.03 | 0 | -78.58 | 2.954 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Filter-forced scenarios (4 profile-seed scenarios)

| profile | policy | workload_synthetic | n | completed | horizon_expired | terminal_failure | completion_h | lat_deg_mean | lat_deg_p95 | viol_frac | lat_budget_met | p_fail_mean | p_fail_max | risk_viol | clip_frac | deliv_mbps | stall_frac | reward | reward_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_altB | fixed_20 | false | 4 | 4 | 0 | 0 | 66.67 | 0.2227 | 1.628 | 0.05947 | 0.5 | 0.003517 | 0.006549 | 0.5 | 0.5695 | 53.64 | 0 | -575.6 | 281.3 |
| diurnal_long_altB | fixed_50 | false | 4 | 4 | 0 | 0 | 48.4 | 0.278 | 1.733 | 0.07468 | 0.5 | 0.003466 | 0.006547 | 0.5 | 0.4053 | 71.29 | 0 | -553.7 | 286.4 |
| diurnal_long_altB | fixed_80 | false | 4 | 4 | 0 | 0 | 36.94 | 0.3547 | 1.857 | 0.1065 | 0.5 | 0.003212 | 0.006245 | 0.5 | 0.2845 | 91.03 | 0 | -622.8 | 343.4 |
| diurnal_long_altB | fixed_120 | false | 4 | 4 | 0 | 0 | 28.62 | 2.091 | 5.667 | 0.5055 | 0 | 0.002862 | 0.005763 | 0.5 | 0 | 116.6 | 0 | -2498 | 372.1 |
| diurnal_long_altB | workload_only_throttle | false | 4 | 4 | 0 | 0 | 34.94 | 0.3396 | 1.975 | 0.09367 | 0.5 | 0.003326 | 0.006547 | 0.5 | 0.2111 | 96.23 | 0 | -550.3 | 293.4 |
| diurnal_long_altB | risk_aware_rule_based | false | 4 | 4 | 0 | 0 | 34.94 | 0.3396 | 1.975 | 0.09367 | 0.5 | 0.003326 | 0.006547 | 0.5 | 0.1075 | 96.23 | 0 | -550.3 | 293.4 |
| diurnal_long_altB | ORACLE_lookahead (not deployable) | false | 4 | 4 | 0 | 0 | 33.27 | 0.3478 | 2.001 | 0.09455 | 0.5 | 0.003243 | 0.006517 | 0.5 | 0.05947 | 100.9 | 0 | -536.7 | 290.8 |
| diurnal_long_altB | slowest_risk_feasible (literal spec) | false | 4 | 4 | 0 | 0 | 66.67 | 0.2227 | 1.628 | 0.05947 | 0.5 | 0.003517 | 0.006549 | 0.5 | 0.1646 | 53.64 | 0 | -575.6 | 281.3 |
| diurnal_long_altB | risk_aware_ceiling_persistence (literal spec) | false | 4 | 4 | 0 | 0 | 32.65 | 0.4208 | 2.297 | 0.1345 | 0 | 0.003218 | 0.006479 | 0.5 | 0.03775 | 102.7 | 0 | -721.3 | 266.1 |
| diurnal_long_altB | risk_aware_rule_based k=1 | false | 4 | 4 | 0 | 0 | 32.52 | 0.7924 | 5.144 | 0.2241 | 0 | 0.003024 | 0.00579 | 0.5 | 0.002469 | 102.9 | 0 | -1233 | 142.2 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Pareto table: completion time, latency, exposure and simulated data-loss probability

`exog_viol` is the mean number of intervals per episode in which the smallest action the safety filter allows also violates latency, computed on the policy's own trajectory. `data_loss_prob` is the simulated data-loss probability: the probability of two or more further failures for a Poisson count with mean the true group hazard integrated over the realised episode.

| profile | policy | n | completion_mean_h | completion_p50_h | completion_p95_h | completion_p99_h | viol_frac | lat_budget_met | latency_cost | exog_viol | p_fail_mean | p_fail_delivered | data_loss_prob |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_altB | fixed_20 | 20 | 144.8 | 166.7 | 166.7 | 166.7 | 0.01189 | 0.9 | 99.13 | 4.35 | 0.001984 | 0.001988 | 2.906e-05 |
| diurnal_long_altB | fixed_50 | 20 | 63.01 | 66.67 | 66.67 | 66.67 | 0.01494 | 0.9 | 99.13 | 4.35 | 0.00123 | 0.001234 | 1.551e-05 |
| diurnal_long_altB | fixed_80 | 20 | 40.75 | 41.67 | 41.75 | 41.75 | 0.05725 | 0.6 | 325.9 | 4.35 | 0.0009778 | 0.0009828 | 1.025e-05 |
| diurnal_long_altB | fixed_120 | 20 | 29 | 29 | 30.27 | 30.59 | 0.5432 | 0 | 2731 | 4.35 | 0.0007958 | 0.0008224 | 7.08e-06 |
| diurnal_long_altB | workload_only_throttle | 20 | 39.68 | 39.5 | 45.85 | 46.17 | 0.02261 | 0.9 | 125.3 | 4.35 | 0.001029 | 0.001034 | 9.347e-06 |
| diurnal_long_altB | risk_aware_rule_based | 20 | 39.68 | 39.5 | 45.85 | 46.17 | 0.02261 | 0.9 | 125.3 | 4.35 | 0.001029 | 0.001034 | 9.347e-06 |
| diurnal_long_altB | ORACLE_lookahead (not deployable) | 20 | 35.5 | 35.83 | 38.44 | 38.75 | 0.01891 | 0.9 | 101.1 | 4.35 | 0.0009611 | 0.0009662 | 8.603e-06 |
| diurnal_long_altB | slowest_risk_feasible (literal spec) | 20 | 144.8 | 166.7 | 166.7 | 166.7 | 0.01189 | 0.9 | 99.13 | 4.35 | 0.001984 | 0.001988 | 2.906e-05 |
| diurnal_long_altB | risk_aware_ceiling_persistence (literal spec) | 20 | 34.29 | 34.21 | 36.73 | 37.61 | 0.08051 | 0.05 | 406.3 | 4.35 | 0.0009367 | 0.000942 | 8.423e-06 |
| diurnal_long_altB | risk_aware_rule_based k=1 | 20 | 39.2 | 39.29 | 45.85 | 46.17 | 0.0487 | 0.8 | 262.4 | 4.35 | 0.0009691 | 0.0009756 | 8.87e-06 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Recovery-time distribution (hours, completed episodes)

| profile | policy | n | n_completed | mean | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_altB | fixed_20 | 20 | 20 | 144.8 | 166.7 | 166.7 | 166.7 | 166.7 | 166.7 |
| diurnal_long_altB | fixed_50 | 20 | 20 | 63.01 | 66.67 | 66.67 | 66.67 | 66.67 | 66.67 |
| diurnal_long_altB | fixed_80 | 20 | 20 | 40.75 | 41.67 | 41.75 | 41.75 | 41.75 | 41.75 |
| diurnal_long_altB | fixed_120 | 20 | 20 | 29 | 29 | 30.1 | 30.27 | 30.59 | 30.67 |
| diurnal_long_altB | workload_only_throttle | 20 | 20 | 39.68 | 39.5 | 44.63 | 45.85 | 46.17 | 46.25 |
| diurnal_long_altB | risk_aware_rule_based | 20 | 20 | 39.68 | 39.5 | 44.63 | 45.85 | 46.17 | 46.25 |
| diurnal_long_altB | ORACLE_lookahead (not deployable) | 20 | 20 | 35.5 | 35.83 | 37.89 | 38.44 | 38.75 | 38.83 |
| diurnal_long_altB | slowest_risk_feasible (literal spec) | 20 | 20 | 144.8 | 166.7 | 166.7 | 166.7 | 166.7 | 166.7 |
| diurnal_long_altB | risk_aware_ceiling_persistence (literal spec) | 20 | 20 | 34.29 | 34.21 | 36.44 | 36.73 | 37.61 | 37.83 |
| diurnal_long_altB | risk_aware_rule_based k=1 | 20 | 20 | 39.2 | 39.29 | 44.63 | 45.85 | 46.17 | 46.25 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Paired difference: risk_aware_rule_based minus workload_only_throttle

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_altB | total_reward | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 20 |
| diurnal_long_altB | latency_degradation_mean | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 20 |
| diurnal_long_altB | latency_violation_fraction | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 20 |
| diurnal_long_altB | completion_hours | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 20 |
| diurnal_long_altB | mean_delivered_mbps | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 20 |
| diurnal_long_altB | p_failure_mean | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 20 |
| diurnal_long_altB | clip_fraction | risk_aware_rule_based | workload_only_throttle | -0.02072 | 0.01136 | 20 |

## Paired difference, controllable scenarios: risk_aware_rule_based minus workload_only_throttle

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_altB | total_reward | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 16 |
| diurnal_long_altB | latency_degradation_mean | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 16 |
| diurnal_long_altB | latency_violation_fraction | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 16 |
| diurnal_long_altB | completion_hours | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 16 |
| diurnal_long_altB | mean_delivered_mbps | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 16 |
| diurnal_long_altB | p_failure_mean | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 16 |
| diurnal_long_altB | clip_fraction | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 16 |

## Paired difference: ORACLE_lookahead (not deployable) minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_altB | total_reward | ORACLE_lookahead (not deployable) | risk_aware_rule_based | 29.25 | 2.924 | 20 |
| diurnal_long_altB | latency_degradation_mean | ORACLE_lookahead (not deployable) | risk_aware_rule_based | -0.003877 | 0.002313 | 20 |
| diurnal_long_altB | latency_violation_fraction | ORACLE_lookahead (not deployable) | risk_aware_rule_based | -0.003698 | 0.0006754 | 20 |
| diurnal_long_altB | completion_hours | ORACLE_lookahead (not deployable) | risk_aware_rule_based | -4.183 | 0.5112 | 20 |
| diurnal_long_altB | mean_delivered_mbps | ORACLE_lookahead (not deployable) | risk_aware_rule_based | 9.506 | 0.9548 | 20 |
| diurnal_long_altB | p_failure_mean | ORACLE_lookahead (not deployable) | risk_aware_rule_based | -6.833e-05 | 1.629e-05 | 20 |
| diurnal_long_altB | clip_fraction | ORACLE_lookahead (not deployable) | risk_aware_rule_based | -0.009599 | 0.006669 | 20 |

## Paired difference, controllable scenarios: ORACLE_lookahead (not deployable) minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_altB | total_reward | ORACLE_lookahead (not deployable) | risk_aware_rule_based | 33.16 | 2.716 | 16 |
| diurnal_long_altB | latency_degradation_mean | ORACLE_lookahead (not deployable) | risk_aware_rule_based | -0.006906 | 0.001545 | 16 |
| diurnal_long_altB | latency_violation_fraction | ORACLE_lookahead (not deployable) | risk_aware_rule_based | -0.004845 | 0.0004823 | 16 |
| diurnal_long_altB | completion_hours | ORACLE_lookahead (not deployable) | risk_aware_rule_based | -4.812 | 0.5248 | 16 |
| diurnal_long_altB | mean_delivered_mbps | ORACLE_lookahead (not deployable) | risk_aware_rule_based | 10.72 | 0.965 | 16 |
| diurnal_long_altB | p_failure_mean | ORACLE_lookahead (not deployable) | risk_aware_rule_based | -6.465e-05 | 9.744e-06 | 16 |
| diurnal_long_altB | clip_fraction | ORACLE_lookahead (not deployable) | risk_aware_rule_based | 0 | 0 | 16 |

## Paired difference: risk_aware_rule_based minus slowest_risk_feasible (literal spec)

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_altB | total_reward | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 99.92 | 9.342 | 20 |
| diurnal_long_altB | latency_degradation_mean | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 0.03286 | 0.01488 | 20 |
| diurnal_long_altB | latency_violation_fraction | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 0.01071 | 0.003921 | 20 |
| diurnal_long_altB | completion_hours | risk_aware_rule_based | slowest_risk_feasible (literal spec) | -105.1 | 8.66 | 20 |
| diurnal_long_altB | mean_delivered_mbps | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 57.89 | 2.238 | 20 |
| diurnal_long_altB | p_failure_mean | risk_aware_rule_based | slowest_risk_feasible (literal spec) | -0.0009548 | 0.0001259 | 20 |
| diurnal_long_altB | clip_fraction | risk_aware_rule_based | slowest_risk_feasible (literal spec) | -0.01142 | 0.01081 | 20 |

## Paired difference, controllable scenarios: risk_aware_rule_based minus slowest_risk_feasible (literal spec)

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_altB | total_reward | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 118.6 | 3.818 | 16 |
| diurnal_long_altB | latency_degradation_mean | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 0.01185 | 0.001462 | 16 |
| diurnal_long_altB | latency_violation_fraction | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 0.004845 | 0.0004823 | 16 |
| diurnal_long_altB | completion_hours | risk_aware_rule_based | slowest_risk_feasible (literal spec) | -123.4 | 1.602 | 16 |
| diurnal_long_altB | mean_delivered_mbps | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 61.72 | 1.42 | 16 |
| diurnal_long_altB | p_failure_mean | risk_aware_rule_based | slowest_risk_feasible (literal spec) | -0.001146 | 9.205e-05 | 16 |
| diurnal_long_altB | clip_fraction | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 0 | 0 | 16 |

## Paired difference: risk_aware_rule_based minus risk_aware_ceiling_persistence (literal spec)

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_altB | total_reward | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | 274.5 | 21.05 | 20 |
| diurnal_long_altB | latency_degradation_mean | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | -0.1168 | 0.008046 | 20 |
| diurnal_long_altB | latency_violation_fraction | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | -0.0579 | 0.003752 | 20 |
| diurnal_long_altB | completion_hours | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | 5.387 | 0.6034 | 20 |
| diurnal_long_altB | mean_delivered_mbps | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | -12.71 | 1.174 | 20 |
| diurnal_long_altB | p_failure_mean | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | 9.277e-05 | 2.029e-05 | 20 |
| diurnal_long_altB | clip_fraction | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | 0.01394 | 0.009668 | 20 |

## Paired difference, controllable scenarios: risk_aware_rule_based minus risk_aware_ceiling_persistence (literal spec)

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_altB | total_reward | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | 300.4 | 16.83 | 16 |
| diurnal_long_altB | latency_degradation_mean | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | -0.1256 | 0.00738 | 16 |
| diurnal_long_altB | latency_violation_fraction | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | -0.06216 | 0.00308 | 16 |
| diurnal_long_altB | completion_hours | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | 6.161 | 0.6002 | 16 |
| diurnal_long_altB | mean_delivered_mbps | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | -14.27 | 1.144 | 16 |
| diurnal_long_altB | p_failure_mean | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | 8.904e-05 | 1.469e-05 | 16 |
| diurnal_long_altB | clip_fraction | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | 0 | 0 | 16 |

## Paired difference: risk_aware_rule_based k=1 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_altB | total_reward | risk_aware_rule_based k=1 | risk_aware_rule_based | -136.5 | 85.22 | 20 |
| diurnal_long_altB | latency_degradation_mean | risk_aware_rule_based k=1 | risk_aware_rule_based | 0.09057 | 0.05605 | 20 |
| diurnal_long_altB | latency_violation_fraction | risk_aware_rule_based k=1 | risk_aware_rule_based | 0.02609 | 0.01579 | 20 |
| diurnal_long_altB | completion_hours | risk_aware_rule_based k=1 | risk_aware_rule_based | -0.4833 | 0.2996 | 20 |
| diurnal_long_altB | mean_delivered_mbps | risk_aware_rule_based k=1 | risk_aware_rule_based | 1.341 | 0.811 | 20 |
| diurnal_long_altB | p_failure_mean | risk_aware_rule_based k=1 | risk_aware_rule_based | -6.038e-05 | 3.956e-05 | 20 |
| diurnal_long_altB | clip_fraction | risk_aware_rule_based k=1 | risk_aware_rule_based | -0.021 | 0.01467 | 20 |

## Paired difference, controllable scenarios: risk_aware_rule_based k=1 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_altB | total_reward | risk_aware_rule_based k=1 | risk_aware_rule_based | 0 | 0 | 16 |
| diurnal_long_altB | latency_degradation_mean | risk_aware_rule_based k=1 | risk_aware_rule_based | 0 | 0 | 16 |
| diurnal_long_altB | latency_violation_fraction | risk_aware_rule_based k=1 | risk_aware_rule_based | 0 | 0 | 16 |
| diurnal_long_altB | completion_hours | risk_aware_rule_based k=1 | risk_aware_rule_based | 0 | 0 | 16 |
| diurnal_long_altB | mean_delivered_mbps | risk_aware_rule_based k=1 | risk_aware_rule_based | 0 | 0 | 16 |
| diurnal_long_altB | p_failure_mean | risk_aware_rule_based k=1 | risk_aware_rule_based | 0 | 0 | 16 |
| diurnal_long_altB | clip_fraction | risk_aware_rule_based k=1 | risk_aware_rule_based | 0 | 0 | 16 |
