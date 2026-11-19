# Baseline comparison (simulated)

Simulator configuration `eval-fleet,sigma=0.25`; latency model uncalibrated (sim/LIMITATIONS.md); workload provenance is listed under Workloads. 20 paired scenario seeds (0..19), identical for every policy. `completion_h` is the mean over completed episodes only. `lat_budget_met` and `risk_viol` are fractions of episodes; `viol_frac`, `clip_frac` and `stall_frac` are fractions of intervals. Latency terms are the mean-latency model of the simulator; `p_fail_*` are the filter's exposure for the applied allocation (nominal rate).

## Workloads

- `diurnal_long_holdout`: source `data_pipeline/profiles/profile_diurnal_long_holdout.csv`, synthetic: false
  - sha256: `94fe6bd6f8a5cb163cba7a9fafac1274d4ec6511113920a63d4993f03295a53d`
  - disclosure: selected extremes, not a representative sample

## Scenario blocks

`filter_forced`: the safety filter raises a `fixed_20` rollout's applied action on at least 10% of the intervals of that scenario. Controllable scenarios are the rest. Both are properties of the scenario, identical for every policy.

| profile | definition | n_forced | n | forced_seeds |
|---|---|---|---|---|
| diurnal_long_holdout | primary | 4 | 20 | 3 9 17 19 |
| diurnal_long_holdout | clip_0.01 | 7 | 20 | 2 3 9 16 17 18 19 |
| diurnal_long_holdout | clip_0.25 | 4 | 20 | 3 9 17 19 |
| diurnal_long_holdout | fixed20_cost | 3 | 20 | 9 17 19 |

## All scenarios (20 profile-seed scenarios)

| profile | policy | workload_synthetic | n | completed | horizon_expired | terminal_failure | completion_h | lat_deg_mean | lat_deg_p95 | viol_frac | lat_budget_met | p_fail_mean | p_fail_max | risk_viol | clip_frac | deliv_mbps | stall_frac | reward | reward_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_holdout | fixed_20 | false | 20 | 20 | 0 | 0 | 144.8 | 0.1431 | 0.85 | 0.02772 | 0.85 | 0.001984 | 0.003874 | 0.1 | 0.123 | 26.9 | 0 | -446.4 | 149.8 |
| diurnal_long_holdout | fixed_50 | false | 20 | 20 | 0 | 0 | 63.08 | 0.1814 | 0.85 | 0.0352 | 0.85 | 0.00123 | 0.002381 | 0.1 | 0.08194 | 54.12 | 0 | -348.3 | 157.2 |
| diurnal_long_holdout | fixed_80 | false | 20 | 20 | 0 | 0 | 40.82 | 0.2702 | 1.142 | 0.07297 | 0.85 | 0.0009776 | 0.001919 | 0.1 | 0.05825 | 82 | 0 | -498.6 | 147.4 |
| diurnal_long_holdout | fixed_120 | false | 20 | 20 | 0 | 0 | 29.27 | 3.498 | 5.667 | 0.6726 | 0 | 0.0008022 | 0.0016 | 0.1 | 0 | 114 | 0 | -3630 | 90.84 |
| diurnal_long_holdout | workload_only_throttle | false | 20 | 20 | 0 | 0 | 43.6 | 0.2315 | 0.85 | 0.04439 | 0.85 | 0.001079 | 0.002298 | 0.1 | 0.06453 | 77.33 | 0 | -330.2 | 157.9 |
| diurnal_long_holdout | risk_aware_rule_based | false | 20 | 20 | 0 | 0 | 43.6 | 0.2315 | 0.85 | 0.04439 | 0.85 | 0.001079 | 0.002298 | 0.1 | 0.02963 | 77.33 | 0 | -330.2 | 157.9 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | false | 20 | 20 | 0 | 0 | 36.86 | 0.2462 | 0.8837 | 0.04682 | 0.85 | 0.0009537 | 0.002293 | 0.1 | 0.02572 | 90.95 | 0 | -318.3 | 158.8 |
| diurnal_long_holdout | slowest_risk_feasible (literal spec) | false | 20 | 20 | 0 | 0 | 144.8 | 0.1431 | 0.85 | 0.02772 | 0.85 | 0.001984 | 0.003874 | 0.1 | 0.03376 | 26.9 | 0 | -446.4 | 149.8 |
| diurnal_long_holdout | risk_aware_ceiling_persistence (literal spec) | false | 20 | 20 | 0 | 0 | 35.9 | 0.365 | 1.286 | 0.09493 | 0.25 | 0.0009362 | 0.002233 | 0.1 | 0.02499 | 93.33 | 0 | -575.1 | 145.3 |
| diurnal_long_holdout | risk_aware_rule_based k=1 | false | 20 | 20 | 0 | 0 | 43.08 | 0.3227 | 1.133 | 0.0621 | 0.8 | 0.00101 | 0.002146 | 0.1 | 0.0002564 | 78.41 | 0 | -442.5 | 183.9 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Controllable scenarios (16 profile-seed scenarios)

| profile | policy | workload_synthetic | n | completed | horizon_expired | terminal_failure | completion_h | lat_deg_mean | lat_deg_p95 | viol_frac | lat_budget_met | p_fail_mean | p_fail_max | risk_viol | clip_frac | deliv_mbps | stall_frac | reward | reward_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_holdout | fixed_20 | false | 16 | 16 | 0 | 0 | 164.3 | 0 | 0 | 0 | 1 | 0.001601 | 0.003205 | 0 | 0.01049 | 20.31 | 0 | -197.1 | 1.706 |
| diurnal_long_holdout | fixed_50 | false | 16 | 16 | 0 | 0 | 66.67 | 0 | 0 | 0 | 1 | 0.0006709 | 0.00134 | 0 | 0 | 50 | 0 | -80 | 0 |
| diurnal_long_holdout | fixed_80 | false | 16 | 16 | 0 | 0 | 41.7 | 0.05874 | 0.3442 | 0.03434 | 1 | 0.0004194 | 0.0008376 | 0 | 0 | 80 | 0 | -251.3 | 8.265 |
| diurnal_long_holdout | fixed_120 | false | 16 | 16 | 0 | 0 | 29.32 | 3.549 | 5.667 | 0.6801 | 0 | 0.0002803 | 0.0005585 | 0 | 0 | 113.8 | 0 | -3679 | 99.7 |
| diurnal_long_holdout | workload_only_throttle | false | 16 | 16 | 0 | 0 | 45.07 | 0.006052 | 0 | 0.0006892 | 1 | 0.0005152 | 0.001235 | 0 | 0 | 74.1 | 0 | -61.1 | 1.456 |
| diurnal_long_holdout | risk_aware_rule_based | false | 16 | 16 | 0 | 0 | 45.07 | 0.006052 | 0 | 0.0006892 | 1 | 0.0005152 | 0.001235 | 0 | 0 | 74.1 | 0 | -61.1 | 1.456 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | false | 16 | 16 | 0 | 0 | 37.59 | 0.004987 | 0.03908 | 0 | 1 | 0.0004038 | 0.001236 | 0 | 0 | 88.91 | 0 | -47.35 | 0.4406 |
| diurnal_long_holdout | slowest_risk_feasible (literal spec) | false | 16 | 16 | 0 | 0 | 164.3 | 0 | 0 | 0 | 1 | 0.001601 | 0.003205 | 0 | 0 | 20.31 | 0 | -197.1 | 1.706 |
| diurnal_long_holdout | risk_aware_ceiling_persistence (literal spec) | false | 16 | 16 | 0 | 0 | 36.53 | 0.1313 | 0.5166 | 0.05251 | 0.25 | 0.0003866 | 0.001163 | 0 | 0 | 91.53 | 0 | -331.3 | 6.009 |
| diurnal_long_holdout | risk_aware_rule_based k=1 | false | 16 | 16 | 0 | 0 | 45.07 | 0.006052 | 0 | 0.0006892 | 1 | 0.0005152 | 0.001235 | 0 | 0 | 74.1 | 0 | -61.1 | 1.456 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Filter-forced scenarios (4 profile-seed scenarios)

| profile | policy | workload_synthetic | n | completed | horizon_expired | terminal_failure | completion_h | lat_deg_mean | lat_deg_p95 | viol_frac | lat_budget_met | p_fail_mean | p_fail_max | risk_viol | clip_frac | deliv_mbps | stall_frac | reward | reward_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_holdout | fixed_20 | false | 4 | 4 | 0 | 0 | 66.98 | 0.7154 | 4.25 | 0.1386 | 0.25 | 0.003517 | 0.006551 | 0.5 | 0.573 | 53.26 | 0 | -1443 | 544.2 |
| diurnal_long_holdout | fixed_50 | false | 4 | 4 | 0 | 0 | 48.75 | 0.9072 | 4.25 | 0.176 | 0.25 | 0.003467 | 0.006548 | 0.5 | 0.4097 | 70.61 | 0 | -1421 | 550.1 |
| diurnal_long_holdout | fixed_80 | false | 4 | 4 | 0 | 0 | 37.31 | 1.116 | 4.335 | 0.2275 | 0.25 | 0.00321 | 0.006246 | 0.5 | 0.2912 | 90.01 | 0 | -1488 | 528.2 |
| diurnal_long_holdout | fixed_120 | false | 4 | 4 | 0 | 0 | 29.08 | 3.295 | 5.667 | 0.6429 | 0 | 0.00289 | 0.005763 | 0.5 | 0 | 114.8 | 0 | -3431 | 214.4 |
| diurnal_long_holdout | workload_only_throttle | false | 4 | 4 | 0 | 0 | 37.71 | 1.133 | 4.25 | 0.2192 | 0.25 | 0.003332 | 0.006548 | 0.5 | 0.3226 | 90.27 | 0 | -1407 | 554.1 |
| diurnal_long_holdout | risk_aware_rule_based | false | 4 | 4 | 0 | 0 | 37.71 | 1.133 | 4.25 | 0.2192 | 0.25 | 0.003332 | 0.006548 | 0.5 | 0.1481 | 90.27 | 0 | -1407 | 554.1 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | false | 4 | 4 | 0 | 0 | 33.96 | 1.211 | 4.262 | 0.2341 | 0.25 | 0.003153 | 0.006524 | 0.5 | 0.1286 | 99.1 | 0 | -1402 | 555.7 |
| diurnal_long_holdout | slowest_risk_feasible (literal spec) | false | 4 | 4 | 0 | 0 | 66.98 | 0.7154 | 4.25 | 0.1386 | 0.25 | 0.003517 | 0.006551 | 0.5 | 0.1688 | 53.26 | 0 | -1443 | 544.2 |
| diurnal_long_holdout | risk_aware_ceiling_persistence (literal spec) | false | 4 | 4 | 0 | 0 | 33.42 | 1.3 | 4.364 | 0.2646 | 0.25 | 0.003134 | 0.006513 | 0.5 | 0.1249 | 100.6 | 0 | -1551 | 520.6 |
| diurnal_long_holdout | risk_aware_rule_based k=1 | false | 4 | 4 | 0 | 0 | 35.12 | 1.59 | 5.667 | 0.3077 | 0 | 0.002987 | 0.00579 | 0.5 | 0.001282 | 95.67 | 0 | -1968 | 317.8 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Pareto table: completion time, latency, exposure and simulated data-loss probability

`exog_viol` is the mean number of intervals per episode in which the smallest action the safety filter allows also violates latency, computed on the policy's own trajectory. `data_loss_prob` is the simulated data-loss probability: the probability of two or more further failures for a Poisson count with mean the true group hazard integrated over the realised episode.

| profile | policy | n | completion_mean_h | completion_p50_h | completion_p95_h | completion_p99_h | viol_frac | lat_budget_met | latency_cost | exog_viol | p_fail_mean | p_fail_delivered | data_loss_prob |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_holdout | fixed_20 | 20 | 144.8 | 166.7 | 166.7 | 166.7 | 0.02772 | 0.85 | 272.6 | 8.9 | 0.001984 | 0.001994 | 2.942e-05 |
| diurnal_long_holdout | fixed_50 | 20 | 63.08 | 66.67 | 66.67 | 66.67 | 0.0352 | 0.85 | 272.6 | 8.9 | 0.00123 | 0.001242 | 1.583e-05 |
| diurnal_long_holdout | fixed_80 | 20 | 40.82 | 41.67 | 41.75 | 41.75 | 0.07297 | 0.85 | 449.6 | 8.9 | 0.0009776 | 0.0009923 | 1.053e-05 |
| diurnal_long_holdout | fixed_120 | 20 | 29.27 | 29.33 | 29.58 | 29.58 | 0.6726 | 0 | 3594 | 8.8 | 0.0008022 | 0.000845 | 7.219e-06 |
| diurnal_long_holdout | workload_only_throttle | 20 | 43.6 | 44.83 | 46.75 | 46.82 | 0.04439 | 0.85 | 277.9 | 8.85 | 0.001079 | 0.001094 | 9.878e-06 |
| diurnal_long_holdout | risk_aware_rule_based | 20 | 43.6 | 44.83 | 46.75 | 46.82 | 0.04439 | 0.85 | 277.9 | 8.85 | 0.001079 | 0.001094 | 9.878e-06 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | 20 | 36.86 | 37.17 | 39.59 | 39.72 | 0.04682 | 0.85 | 274.1 | 8.85 | 0.0009537 | 0.00097 | 8.661e-06 |
| diurnal_long_holdout | slowest_risk_feasible (literal spec) | 20 | 144.8 | 166.7 | 166.7 | 166.7 | 0.02772 | 0.85 | 272.6 | 8.9 | 0.001984 | 0.001994 | 2.942e-05 |
| diurnal_long_holdout | risk_aware_ceiling_persistence (literal spec) | 20 | 35.9 | 36.25 | 38.19 | 38.5 | 0.09493 | 0.25 | 532.1 | 8.85 | 0.0009362 | 0.0009528 | 8.514e-06 |
| diurnal_long_holdout | risk_aware_rule_based k=1 | 20 | 43.08 | 44.62 | 46.75 | 46.82 | 0.0621 | 0.8 | 390.8 | 8.8 | 0.00101 | 0.001029 | 9.36e-06 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Recovery-time distribution (hours, completed episodes)

| profile | policy | n | n_completed | mean | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_holdout | fixed_20 | 20 | 20 | 144.8 | 166.7 | 166.7 | 166.7 | 166.7 | 166.7 |
| diurnal_long_holdout | fixed_50 | 20 | 20 | 63.08 | 66.67 | 66.67 | 66.67 | 66.67 | 66.67 |
| diurnal_long_holdout | fixed_80 | 20 | 20 | 40.82 | 41.67 | 41.75 | 41.75 | 41.75 | 41.75 |
| diurnal_long_holdout | fixed_120 | 20 | 20 | 29.27 | 29.33 | 29.51 | 29.58 | 29.58 | 29.58 |
| diurnal_long_holdout | workload_only_throttle | 20 | 20 | 43.6 | 44.83 | 46.6 | 46.75 | 46.82 | 46.83 |
| diurnal_long_holdout | risk_aware_rule_based | 20 | 20 | 43.6 | 44.83 | 46.6 | 46.75 | 46.82 | 46.83 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | 20 | 20 | 36.86 | 37.17 | 39.51 | 39.59 | 39.72 | 39.75 |
| diurnal_long_holdout | slowest_risk_feasible (literal spec) | 20 | 20 | 144.8 | 166.7 | 166.7 | 166.7 | 166.7 | 166.7 |
| diurnal_long_holdout | risk_aware_ceiling_persistence (literal spec) | 20 | 20 | 35.9 | 36.25 | 38.02 | 38.19 | 38.5 | 38.58 |
| diurnal_long_holdout | risk_aware_rule_based k=1 | 20 | 20 | 43.08 | 44.62 | 46.6 | 46.75 | 46.82 | 46.83 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Paired difference: risk_aware_rule_based minus workload_only_throttle

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 20 |
| diurnal_long_holdout | latency_degradation_mean | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 20 |
| diurnal_long_holdout | latency_violation_fraction | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 20 |
| diurnal_long_holdout | completion_hours | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 20 |
| diurnal_long_holdout | p_failure_mean | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 20 |
| diurnal_long_holdout | clip_fraction | risk_aware_rule_based | workload_only_throttle | -0.0349 | 0.01827 | 20 |

## Paired difference, controllable scenarios: risk_aware_rule_based minus workload_only_throttle

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 16 |
| diurnal_long_holdout | latency_degradation_mean | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 16 |
| diurnal_long_holdout | latency_violation_fraction | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 16 |
| diurnal_long_holdout | completion_hours | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 16 |
| diurnal_long_holdout | mean_delivered_mbps | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 16 |
| diurnal_long_holdout | p_failure_mean | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 16 |
| diurnal_long_holdout | clip_fraction | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 16 |

## Paired difference: ORACLE_lookahead (not deployable) minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | ORACLE_lookahead (not deployable) | risk_aware_rule_based | 11.85 | 1.496 | 20 |
| diurnal_long_holdout | latency_degradation_mean | ORACLE_lookahead (not deployable) | risk_aware_rule_based | 0.01476 | 0.009441 | 20 |
| diurnal_long_holdout | latency_violation_fraction | ORACLE_lookahead (not deployable) | risk_aware_rule_based | 0.002431 | 0.001822 | 20 |
| diurnal_long_holdout | completion_hours | ORACLE_lookahead (not deployable) | risk_aware_rule_based | -6.737 | 0.5412 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | ORACLE_lookahead (not deployable) | risk_aware_rule_based | 13.62 | 0.9661 | 20 |
| diurnal_long_holdout | p_failure_mean | ORACLE_lookahead (not deployable) | risk_aware_rule_based | -0.0001248 | 2.889e-05 | 20 |
| diurnal_long_holdout | clip_fraction | ORACLE_lookahead (not deployable) | risk_aware_rule_based | -0.003903 | 0.003328 | 20 |

## Paired difference, controllable scenarios: ORACLE_lookahead (not deployable) minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | ORACLE_lookahead (not deployable) | risk_aware_rule_based | 13.75 | 1.472 | 16 |
| diurnal_long_holdout | latency_degradation_mean | ORACLE_lookahead (not deployable) | risk_aware_rule_based | -0.001065 | 0.0002797 | 16 |
| diurnal_long_holdout | latency_violation_fraction | ORACLE_lookahead (not deployable) | risk_aware_rule_based | -0.0006892 | 0.0002298 | 16 |
| diurnal_long_holdout | completion_hours | ORACLE_lookahead (not deployable) | risk_aware_rule_based | -7.484 | 0.4105 | 16 |
| diurnal_long_holdout | mean_delivered_mbps | ORACLE_lookahead (not deployable) | risk_aware_rule_based | 14.81 | 0.8653 | 16 |
| diurnal_long_holdout | p_failure_mean | ORACLE_lookahead (not deployable) | risk_aware_rule_based | -0.0001114 | 1.785e-05 | 16 |
| diurnal_long_holdout | clip_fraction | ORACLE_lookahead (not deployable) | risk_aware_rule_based | 0 | 0 | 16 |

## Paired difference: risk_aware_rule_based minus slowest_risk_feasible (literal spec)

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 116.2 | 9.537 | 20 |
| diurnal_long_holdout | latency_degradation_mean | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 0.0884 | 0.04773 | 20 |
| diurnal_long_holdout | latency_violation_fraction | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 0.01667 | 0.009178 | 20 |
| diurnal_long_holdout | completion_hours | risk_aware_rule_based | slowest_risk_feasible (literal spec) | -101.2 | 8.438 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 50.43 | 1.614 | 20 |
| diurnal_long_holdout | p_failure_mean | risk_aware_rule_based | slowest_risk_feasible (literal spec) | -0.0009058 | 0.000112 | 20 |
| diurnal_long_holdout | clip_fraction | risk_aware_rule_based | slowest_risk_feasible (literal spec) | -0.004131 | 0.004789 | 20 |

## Paired difference, controllable scenarios: risk_aware_rule_based minus slowest_risk_feasible (literal spec)

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 136 | 2.53 | 16 |
| diurnal_long_holdout | latency_degradation_mean | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 0.006052 | 0.0002993 | 16 |
| diurnal_long_holdout | latency_violation_fraction | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 0.0006892 | 0.0002298 | 16 |
| diurnal_long_holdout | completion_hours | risk_aware_rule_based | slowest_risk_feasible (literal spec) | -119.2 | 1.469 | 16 |
| diurnal_long_holdout | mean_delivered_mbps | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 53.78 | 0.5622 | 16 |
| diurnal_long_holdout | p_failure_mean | risk_aware_rule_based | slowest_risk_feasible (literal spec) | -0.001086 | 8.373e-05 | 16 |
| diurnal_long_holdout | clip_fraction | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 0 | 0 | 16 |

## Paired difference: risk_aware_rule_based minus risk_aware_ceiling_persistence (literal spec)

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | 245 | 15.26 | 20 |
| diurnal_long_holdout | latency_degradation_mean | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | -0.1335 | 0.008444 | 20 |
| diurnal_long_holdout | latency_violation_fraction | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | -0.05053 | 0.001932 | 20 |
| diurnal_long_holdout | completion_hours | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | 7.696 | 0.6004 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | -16 | 1.104 | 20 |
| diurnal_long_holdout | p_failure_mean | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | 0.0001423 | 3.237e-05 | 20 |
| diurnal_long_holdout | clip_fraction | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | 0.004638 | 0.004081 | 20 |

## Paired difference, controllable scenarios: risk_aware_rule_based minus risk_aware_ceiling_persistence (literal spec)

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | 270.2 | 5.54 | 16 |
| diurnal_long_holdout | latency_degradation_mean | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | -0.1252 | 0.004124 | 16 |
| diurnal_long_holdout | latency_violation_fraction | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | -0.05183 | 0.001092 | 16 |
| diurnal_long_holdout | completion_hours | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | 8.547 | 0.4284 | 16 |
| diurnal_long_holdout | mean_delivered_mbps | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | -17.43 | 0.9383 | 16 |
| diurnal_long_holdout | p_failure_mean | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | 0.0001286 | 1.993e-05 | 16 |
| diurnal_long_holdout | clip_fraction | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | 0 | 0 | 16 |

## Paired difference: risk_aware_rule_based k=1 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | risk_aware_rule_based k=1 | risk_aware_rule_based | -112.3 | 67.41 | 20 |
| diurnal_long_holdout | latency_degradation_mean | risk_aware_rule_based k=1 | risk_aware_rule_based | 0.09127 | 0.05207 | 20 |
| diurnal_long_holdout | latency_violation_fraction | risk_aware_rule_based k=1 | risk_aware_rule_based | 0.01771 | 0.0103 | 20 |
| diurnal_long_holdout | completion_hours | risk_aware_rule_based k=1 | risk_aware_rule_based | -0.5167 | 0.3838 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | risk_aware_rule_based k=1 | risk_aware_rule_based | 1.08 | 0.7357 | 20 |
| diurnal_long_holdout | p_failure_mean | risk_aware_rule_based k=1 | risk_aware_rule_based | -6.895e-05 | 4.484e-05 | 20 |
| diurnal_long_holdout | clip_fraction | risk_aware_rule_based k=1 | risk_aware_rule_based | -0.02937 | 0.02026 | 20 |

## Paired difference, controllable scenarios: risk_aware_rule_based k=1 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | risk_aware_rule_based k=1 | risk_aware_rule_based | 0 | 0 | 16 |
| diurnal_long_holdout | latency_degradation_mean | risk_aware_rule_based k=1 | risk_aware_rule_based | 0 | 0 | 16 |
| diurnal_long_holdout | latency_violation_fraction | risk_aware_rule_based k=1 | risk_aware_rule_based | 0 | 0 | 16 |
| diurnal_long_holdout | completion_hours | risk_aware_rule_based k=1 | risk_aware_rule_based | 0 | 0 | 16 |
| diurnal_long_holdout | mean_delivered_mbps | risk_aware_rule_based k=1 | risk_aware_rule_based | 0 | 0 | 16 |
| diurnal_long_holdout | p_failure_mean | risk_aware_rule_based k=1 | risk_aware_rule_based | 0 | 0 | 16 |
| diurnal_long_holdout | clip_fraction | risk_aware_rule_based k=1 | risk_aware_rule_based | 0 | 0 | 16 |
