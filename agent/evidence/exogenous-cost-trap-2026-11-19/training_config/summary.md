# Baseline comparison (simulated)

Simulator configuration `train-fleet,sigma=0.25`; latency model uncalibrated (sim/LIMITATIONS.md); workload provenance is listed under Workloads. 20 paired scenario seeds (0..19), identical for every policy. `completion_h` is the mean over completed episodes only. `lat_budget_met` and `risk_viol` are fractions of episodes; `viol_frac`, `clip_frac` and `stall_frac` are fractions of intervals. Latency terms are the mean-latency model of the simulator; `p_fail_*` are the filter's exposure for the applied allocation (nominal rate).

## Workloads

- `diurnal_long_train`: source `data_pipeline/profiles/profile_diurnal_long_train.csv`, synthetic: false
  - sha256: `65680adfc9c5aa60f1dc30164ecbe66a0117859df9a773cacc3035fd8f1eea3d`
  - disclosure: selected extremes, not a representative sample

## Scenario blocks

`filter_forced`: the safety filter raises a `fixed_20` rollout's applied action on at least 10% of the intervals of that scenario. Controllable scenarios are the rest. Both are properties of the scenario, identical for every policy.

| profile | definition | n_forced | n | forced_seeds |
|---|---|---|---|---|
| diurnal_long_train | primary | 5 | 20 | 7 9 11 17 18 |
| diurnal_long_train | clip_0.01 | 8 | 20 | 3 4 7 9 11 14 17 18 |
| diurnal_long_train | clip_0.25 | 4 | 20 | 7 11 17 18 |
| diurnal_long_train | fixed20_cost | 3 | 20 | 7 11 18 |

## All scenarios (20 profile-seed scenarios)

| profile | policy | workload_synthetic | n | completed | horizon_expired | terminal_failure | completion_h | lat_deg_mean | lat_deg_p95 | viol_frac | lat_budget_met | p_fail_mean | p_fail_max | risk_viol | clip_frac | deliv_mbps | stall_frac | reward | reward_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_train | fixed_20 | false | 20 | 20 | 0 | 0 | 142.8 | 0.2044 | 0.5697 | 0.0405 | 0.9 | 0.002205 | 0.004547 | 0.1 | 0.1373 | 28.52 | 0 | -472.8 | 186 |
| diurnal_long_train | fixed_50 | false | 20 | 20 | 0 | 0 | 62.81 | 0.2357 | 0.5795 | 0.04683 | 0.9 | 0.001536 | 0.003129 | 0.1 | 0.09147 | 55.17 | 0 | -376.8 | 193.1 |
| diurnal_long_train | fixed_80 | false | 20 | 20 | 0 | 0 | 40.58 | 0.3029 | 0.8411 | 0.07556 | 0.85 | 0.001311 | 0.002649 | 0.1 | 0.07772 | 82.72 | 0 | -487.8 | 185.3 |
| diurnal_long_train | fixed_120 | false | 20 | 20 | 0 | 0 | 29.16 | 3.358 | 5.667 | 0.6577 | 0 | 0.001173 | 0.002331 | 0.1 | 0 | 114.5 | 0 | -3514 | 97.81 |
| diurnal_long_train | workload_only_throttle | false | 20 | 20 | 0 | 0 | 42.25 | 0.2521 | 0.586 | 0.05039 | 0.9 | 0.001341 | 0.003005 | 0.1 | 0.06298 | 79.54 | 0 | -361.4 | 194.1 |
| diurnal_long_train | risk_aware_rule_based | false | 20 | 20 | 0 | 0 | 42.25 | 0.2521 | 0.586 | 0.05039 | 0.9 | 0.001341 | 0.003005 | 0.1 | 0.02459 | 79.54 | 0 | -361.4 | 194.1 |
| diurnal_long_train | ORACLE_lookahead (not deployable) | false | 20 | 20 | 0 | 0 | 36.25 | 0.2662 | 0.6238 | 0.05231 | 0.85 | 0.001277 | 0.002997 | 0.1 | 0.02093 | 92.34 | 0 | -345.3 | 195.2 |
| diurnal_long_train | slowest_risk_feasible (literal spec) | false | 20 | 20 | 0 | 0 | 142.8 | 0.2044 | 0.5697 | 0.0405 | 0.9 | 0.002205 | 0.004547 | 0.1 | 0.05398 | 28.52 | 0 | -472.8 | 186 |
| diurnal_long_train | risk_aware_ceiling_persistence (literal spec) | false | 20 | 20 | 0 | 0 | 35.35 | 0.3774 | 0.9924 | 0.09594 | 0.7 | 0.001264 | 0.002938 | 0.1 | 0.01997 | 94.68 | 0 | -574.3 | 182.3 |
| diurnal_long_train | risk_aware_rule_based k=1 | false | 20 | 20 | 0 | 0 | 41.8 | 0.2962 | 0.852 | 0.05963 | 0.85 | 0.001303 | 0.002872 | 0.1 | 0.001359 | 80.56 | 0 | -415 | 200.1 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Controllable scenarios (15 profile-seed scenarios)

| profile | policy | workload_synthetic | n | completed | horizon_expired | terminal_failure | completion_h | lat_deg_mean | lat_deg_p95 | viol_frac | lat_budget_met | p_fail_mean | p_fail_max | risk_viol | clip_frac | deliv_mbps | stall_frac | reward | reward_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_train | fixed_20 | false | 15 | 15 | 0 | 0 | 163.2 | 0 | 0 | 0 | 1 | 0.001463 | 0.002931 | 0 | 0.01542 | 20.46 | 0 | -195.9 | 2.255 |
| diurnal_long_train | fixed_50 | false | 15 | 15 | 0 | 0 | 66.67 | 0 | 0 | 0 | 1 | 0.0006297 | 0.001258 | 0 | 0 | 50 | 0 | -80 | 3.798e-15 |
| diurnal_long_train | fixed_80 | false | 15 | 15 | 0 | 0 | 41.68 | 0.04701 | 0.278 | 0.02719 | 1 | 0.0003939 | 0.0007862 | 0 | 0 | 80 | 0 | -209.5 | 15.16 |
| diurnal_long_train | fixed_120 | false | 15 | 15 | 0 | 0 | 29.17 | 3.338 | 5.667 | 0.6522 | 0 | 0.0002619 | 0.0005242 | 0 | 0 | 114.5 | 0 | -3489 | 119.3 |
| diurnal_long_train | workload_only_throttle | false | 15 | 15 | 0 | 0 | 43.61 | 0.006461 | 0 | 0.00177 | 1 | 0.0004442 | 0.001141 | 0 | 0 | 76.65 | 0 | -65.02 | 2.817 |
| diurnal_long_train | risk_aware_rule_based | false | 15 | 15 | 0 | 0 | 43.61 | 0.006461 | 0 | 0.00177 | 1 | 0.0004442 | 0.001141 | 0 | 0 | 76.65 | 0 | -65.02 | 2.817 |
| diurnal_long_train | ORACLE_lookahead (not deployable) | false | 15 | 15 | 0 | 0 | 36.88 | 0.003948 | 0.03023 | 0 | 1 | 0.0003687 | 0.001138 | 0 | 0 | 90.59 | 0 | -46 | 0.4498 |
| diurnal_long_train | slowest_risk_feasible (literal spec) | false | 15 | 15 | 0 | 0 | 163.2 | 0 | 0 | 0 | 1 | 0.001463 | 0.002931 | 0 | 0 | 20.46 | 0 | -195.9 | 2.255 |
| diurnal_long_train | risk_aware_ceiling_persistence (literal spec) | false | 15 | 15 | 0 | 0 | 35.91 | 0.1219 | 0.4275 | 0.04715 | 0.8 | 0.0003542 | 0.00109 | 0 | 0 | 93.08 | 0 | -298.9 | 9.228 |
| diurnal_long_train | risk_aware_rule_based k=1 | false | 15 | 15 | 0 | 0 | 43.54 | 0.006718 | 0 | 0.001904 | 1 | 0.000438 | 0.001123 | 0 | 0 | 76.75 | 0 | -65.74 | 3.117 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Filter-forced scenarios (5 profile-seed scenarios)

| profile | policy | workload_synthetic | n | completed | horizon_expired | terminal_failure | completion_h | lat_deg_mean | lat_deg_p95 | viol_frac | lat_budget_met | p_fail_mean | p_fail_max | risk_viol | clip_frac | deliv_mbps | stall_frac | reward | reward_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_train | fixed_20 | false | 5 | 5 | 0 | 0 | 81.6 | 0.8174 | 2.279 | 0.162 | 0.6 | 0.004433 | 0.009395 | 0.4 | 0.503 | 52.71 | 0 | -1303 | 653.6 |
| diurnal_long_train | fixed_50 | false | 5 | 5 | 0 | 0 | 51.25 | 0.9427 | 2.318 | 0.1873 | 0.6 | 0.004257 | 0.008744 | 0.4 | 0.3659 | 70.7 | 0 | -1267 | 666.5 |
| diurnal_long_train | fixed_80 | false | 5 | 5 | 0 | 0 | 37.3 | 1.07 | 2.53 | 0.2207 | 0.4 | 0.004064 | 0.008239 | 0.4 | 0.3109 | 90.86 | 0 | -1322 | 646.2 |
| diurnal_long_train | fixed_120 | false | 5 | 5 | 0 | 0 | 29.15 | 3.418 | 5.667 | 0.6741 | 0 | 0.003905 | 0.00775 | 0.4 | 0 | 114.5 | 0 | -3589 | 174.9 |
| diurnal_long_train | workload_only_throttle | false | 5 | 5 | 0 | 0 | 38.18 | 0.9889 | 2.344 | 0.1962 | 0.6 | 0.00403 | 0.008598 | 0.4 | 0.2519 | 88.19 | 0 | -1251 | 672.5 |
| diurnal_long_train | risk_aware_rule_based | false | 5 | 5 | 0 | 0 | 38.18 | 0.9889 | 2.344 | 0.1962 | 0.6 | 0.00403 | 0.008598 | 0.4 | 0.09838 | 88.19 | 0 | -1251 | 672.5 |
| diurnal_long_train | ORACLE_lookahead (not deployable) | false | 5 | 5 | 0 | 0 | 34.35 | 1.053 | 2.404 | 0.2092 | 0.4 | 0.004003 | 0.008574 | 0.4 | 0.0837 | 97.57 | 0 | -1243 | 674.6 |
| diurnal_long_train | slowest_risk_feasible (literal spec) | false | 5 | 5 | 0 | 0 | 81.6 | 0.8174 | 2.279 | 0.162 | 0.6 | 0.004433 | 0.009395 | 0.4 | 0.2159 | 52.71 | 0 | -1303 | 653.6 |
| diurnal_long_train | risk_aware_ceiling_persistence (literal spec) | false | 5 | 5 | 0 | 0 | 33.67 | 1.144 | 2.687 | 0.2423 | 0.4 | 0.003991 | 0.008484 | 0.4 | 0.0799 | 99.48 | 0 | -1400 | 634.8 |
| diurnal_long_train | risk_aware_rule_based k=1 | false | 5 | 5 | 0 | 0 | 36.58 | 1.165 | 3.408 | 0.2328 | 0.4 | 0.003898 | 0.008122 | 0.4 | 0.005435 | 91.98 | 0 | -1463 | 628.4 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Pareto table: completion time, latency, exposure and simulated data-loss probability

`exog_viol` is the mean number of intervals per episode in which the smallest action the safety filter allows also violates latency, computed on the policy's own trajectory. `data_loss_prob` is the simulated data-loss probability: the probability of two or more further failures for a Poisson count with mean the true group hazard integrated over the realised episode.

| profile | policy | n | completion_mean_h | completion_p50_h | completion_p95_h | completion_p99_h | viol_frac | lat_budget_met | latency_cost | exog_viol | p_fail_mean | p_fail_delivered | data_loss_prob |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_train | fixed_20 | 20 | 142.8 | 166.7 | 166.7 | 166.7 | 0.0405 | 0.9 | 301.4 | 20.1 | 0.002205 | 0.002238 | 1.997e-05 |
| diurnal_long_train | fixed_50 | 20 | 62.81 | 66.67 | 66.67 | 66.67 | 0.04683 | 0.9 | 301.4 | 20.1 | 0.001536 | 0.001574 | 1.155e-05 |
| diurnal_long_train | fixed_80 | 20 | 40.58 | 41.67 | 41.75 | 41.75 | 0.07556 | 0.85 | 439.1 | 20.1 | 0.001311 | 0.001353 | 8.755e-06 |
| diurnal_long_train | fixed_120 | 20 | 29.16 | 29.08 | 29.68 | 29.8 | 0.6577 | 0 | 3479 | 19.9 | 0.001173 | 0.001238 | 7.055e-06 |
| diurnal_long_train | workload_only_throttle | 20 | 42.25 | 43.33 | 45.74 | 46.88 | 0.05039 | 0.9 | 310.7 | 20 | 0.001341 | 0.00138 | 9.821e-06 |
| diurnal_long_train | risk_aware_rule_based | 20 | 42.25 | 43.33 | 45.74 | 46.88 | 0.05039 | 0.9 | 310.7 | 20 | 0.001341 | 0.00138 | 9.821e-06 |
| diurnal_long_train | ORACLE_lookahead (not deployable) | 20 | 36.25 | 36.5 | 39.01 | 39.2 | 0.05231 | 0.85 | 301.8 | 20 | 0.001277 | 0.001319 | 8.497e-06 |
| diurnal_long_train | slowest_risk_feasible (literal spec) | 20 | 142.8 | 166.7 | 166.7 | 166.7 | 0.0405 | 0.9 | 301.4 | 20.1 | 0.002205 | 0.002238 | 1.997e-05 |
| diurnal_long_train | risk_aware_ceiling_persistence (literal spec) | 20 | 35.35 | 35.58 | 38.17 | 38.3 | 0.09594 | 0.7 | 531.9 | 19.95 | 0.001264 | 0.001306 | 8.279e-06 |
| diurnal_long_train | risk_aware_rule_based k=1 | 20 | 41.8 | 43.17 | 45.74 | 46.88 | 0.05963 | 0.85 | 364.8 | 19.9 | 0.001303 | 0.001344 | 9.649e-06 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Recovery-time distribution (hours, completed episodes)

| profile | policy | n | n_completed | mean | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_train | fixed_20 | 20 | 20 | 142.8 | 166.7 | 166.7 | 166.7 | 166.7 | 166.7 |
| diurnal_long_train | fixed_50 | 20 | 20 | 62.81 | 66.67 | 66.67 | 66.67 | 66.67 | 66.67 |
| diurnal_long_train | fixed_80 | 20 | 20 | 40.58 | 41.67 | 41.67 | 41.75 | 41.75 | 41.75 |
| diurnal_long_train | fixed_120 | 20 | 20 | 29.16 | 29.08 | 29.59 | 29.68 | 29.8 | 29.83 |
| diurnal_long_train | workload_only_throttle | 20 | 20 | 42.25 | 43.33 | 45.07 | 45.74 | 46.88 | 47.17 |
| diurnal_long_train | risk_aware_rule_based | 20 | 20 | 42.25 | 43.33 | 45.07 | 45.74 | 46.88 | 47.17 |
| diurnal_long_train | ORACLE_lookahead (not deployable) | 20 | 20 | 36.25 | 36.5 | 38.48 | 39.01 | 39.2 | 39.25 |
| diurnal_long_train | slowest_risk_feasible (literal spec) | 20 | 20 | 142.8 | 166.7 | 166.7 | 166.7 | 166.7 | 166.7 |
| diurnal_long_train | risk_aware_ceiling_persistence (literal spec) | 20 | 20 | 35.35 | 35.58 | 37.57 | 38.17 | 38.3 | 38.33 |
| diurnal_long_train | risk_aware_rule_based k=1 | 20 | 20 | 41.8 | 43.17 | 45.07 | 45.74 | 46.88 | 47.17 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Paired difference: risk_aware_rule_based minus workload_only_throttle

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_train | total_reward | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 20 |
| diurnal_long_train | latency_degradation_mean | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 20 |
| diurnal_long_train | latency_violation_fraction | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 20 |
| diurnal_long_train | completion_hours | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 20 |
| diurnal_long_train | mean_delivered_mbps | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 20 |
| diurnal_long_train | p_failure_mean | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 20 |
| diurnal_long_train | clip_fraction | risk_aware_rule_based | workload_only_throttle | -0.03839 | 0.02127 | 20 |

## Paired difference, controllable scenarios: risk_aware_rule_based minus workload_only_throttle

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_train | total_reward | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 15 |
| diurnal_long_train | latency_degradation_mean | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 15 |
| diurnal_long_train | latency_violation_fraction | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 15 |
| diurnal_long_train | completion_hours | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 15 |
| diurnal_long_train | mean_delivered_mbps | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 15 |
| diurnal_long_train | p_failure_mean | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 15 |
| diurnal_long_train | clip_fraction | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 15 |

## Total reward difference by definition of controllable: risk_aware_rule_based minus workload_only_throttle

| profile | definition | n_controllable | mean_diff | sem | t | better | worse |
|---|---|---|---|---|---|---|---|
| diurnal_long_train | primary | 15 | 0 | 0 | n/a | 0 | 0 |
| diurnal_long_train | clip_0.01 | 12 | 0 | 0 | n/a | 0 | 0 |
| diurnal_long_train | clip_0.25 | 16 | 0 | 0 | n/a | 0 | 0 |
| diurnal_long_train | fixed20_cost | 17 | 0 | 0 | n/a | 0 | 0 |

## Paired difference: ORACLE_lookahead (not deployable) minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_train | total_reward | ORACLE_lookahead (not deployable) | risk_aware_rule_based | 16.13 | 2.261 | 20 |
| diurnal_long_train | latency_degradation_mean | ORACLE_lookahead (not deployable) | risk_aware_rule_based | 0.01412 | 0.01091 | 20 |
| diurnal_long_train | latency_violation_fraction | ORACLE_lookahead (not deployable) | risk_aware_rule_based | 0.001925 | 0.002268 | 20 |
| diurnal_long_train | completion_hours | ORACLE_lookahead (not deployable) | risk_aware_rule_based | -6.004 | 0.49 | 20 |
| diurnal_long_train | mean_delivered_mbps | ORACLE_lookahead (not deployable) | risk_aware_rule_based | 12.8 | 0.8908 | 20 |
| diurnal_long_train | p_failure_mean | ORACLE_lookahead (not deployable) | risk_aware_rule_based | -6.339e-05 | 2.638e-05 | 20 |
| diurnal_long_train | clip_fraction | ORACLE_lookahead (not deployable) | risk_aware_rule_based | -0.003668 | 0.002535 | 20 |

## Paired difference, controllable scenarios: ORACLE_lookahead (not deployable) minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_train | total_reward | ORACLE_lookahead (not deployable) | risk_aware_rule_based | 19.02 | 2.454 | 15 |
| diurnal_long_train | latency_degradation_mean | ORACLE_lookahead (not deployable) | risk_aware_rule_based | -0.002512 | 0.0006388 | 15 |
| diurnal_long_train | latency_violation_fraction | ORACLE_lookahead (not deployable) | risk_aware_rule_based | -0.00177 | 0.0004659 | 15 |
| diurnal_long_train | completion_hours | ORACLE_lookahead (not deployable) | risk_aware_rule_based | -6.728 | 0.4741 | 15 |
| diurnal_long_train | mean_delivered_mbps | ORACLE_lookahead (not deployable) | risk_aware_rule_based | 13.94 | 0.9299 | 15 |
| diurnal_long_train | p_failure_mean | ORACLE_lookahead (not deployable) | risk_aware_rule_based | -7.552e-05 | 1.24e-05 | 15 |
| diurnal_long_train | clip_fraction | ORACLE_lookahead (not deployable) | risk_aware_rule_based | 0 | 0 | 15 |

## Total reward difference by definition of controllable: ORACLE_lookahead (not deployable) minus risk_aware_rule_based

| profile | definition | n_controllable | mean_diff | sem | t | better | worse |
|---|---|---|---|---|---|---|---|
| diurnal_long_train | primary | 15 | 19.02 | 2.454 | 7.752 | 15 | 0 |
| diurnal_long_train | clip_0.01 | 12 | 18.79 | 2.868 | 6.551 | 12 | 0 |
| diurnal_long_train | clip_0.25 | 16 | 19 | 2.295 | 8.279 | 16 | 0 |
| diurnal_long_train | fixed20_cost | 17 | 18.29 | 2.272 | 8.048 | 17 | 0 |

## Paired difference: risk_aware_rule_based minus slowest_risk_feasible (literal spec)

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_train | total_reward | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 111.3 | 9.339 | 20 |
| diurnal_long_train | latency_degradation_mean | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 0.04772 | 0.02695 | 20 |
| diurnal_long_train | latency_violation_fraction | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 0.009888 | 0.005201 | 20 |
| diurnal_long_train | completion_hours | risk_aware_rule_based | slowest_risk_feasible (literal spec) | -100.6 | 8.734 | 20 |
| diurnal_long_train | mean_delivered_mbps | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 51.01 | 2.931 | 20 |
| diurnal_long_train | p_failure_mean | risk_aware_rule_based | slowest_risk_feasible (literal spec) | -0.0008647 | 0.0001379 | 20 |
| diurnal_long_train | clip_fraction | risk_aware_rule_based | slowest_risk_feasible (literal spec) | -0.02939 | 0.02027 | 20 |

## Paired difference, controllable scenarios: risk_aware_rule_based minus slowest_risk_feasible (literal spec)

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_train | total_reward | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 130.9 | 3.591 | 15 |
| diurnal_long_train | latency_degradation_mean | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 0.006461 | 0.0005067 | 15 |
| diurnal_long_train | latency_violation_fraction | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 0.00177 | 0.0004659 | 15 |
| diurnal_long_train | completion_hours | risk_aware_rule_based | slowest_risk_feasible (literal spec) | -119.6 | 1.972 | 15 |
| diurnal_long_train | mean_delivered_mbps | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 56.19 | 0.9263 | 15 |
| diurnal_long_train | p_failure_mean | risk_aware_rule_based | slowest_risk_feasible (literal spec) | -0.001019 | 9.218e-05 | 15 |
| diurnal_long_train | clip_fraction | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 0 | 0 | 15 |

## Total reward difference by definition of controllable: risk_aware_rule_based minus slowest_risk_feasible (literal spec)

| profile | definition | n_controllable | mean_diff | sem | t | better | worse |
|---|---|---|---|---|---|---|---|
| diurnal_long_train | primary | 15 | 130.9 | 3.591 | 36.44 | 15 | 0 |
| diurnal_long_train | clip_0.01 | 12 | 135.2 | 3.271 | 41.33 | 12 | 0 |
| diurnal_long_train | clip_0.25 | 16 | 128.6 | 4.017 | 32.03 | 16 | 0 |
| diurnal_long_train | fixed20_cost | 17 | 126.5 | 4.343 | 29.13 | 17 | 0 |

## Paired difference: risk_aware_rule_based minus risk_aware_ceiling_persistence (literal spec)

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_train | total_reward | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | 212.8 | 13.41 | 20 |
| diurnal_long_train | latency_degradation_mean | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | -0.1253 | 0.008541 | 20 |
| diurnal_long_train | latency_violation_fraction | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | -0.04556 | 0.001348 | 20 |
| diurnal_long_train | completion_hours | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | 6.904 | 0.5291 | 20 |
| diurnal_long_train | mean_delivered_mbps | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | -15.14 | 0.9949 | 20 |
| diurnal_long_train | p_failure_mean | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | 7.705e-05 | 3.073e-05 | 20 |
| diurnal_long_train | clip_fraction | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | 0.00462 | 0.003197 | 20 |

## Paired difference, controllable scenarios: risk_aware_rule_based minus risk_aware_ceiling_persistence (literal spec)

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_train | total_reward | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | 233.9 | 7.502 | 15 |
| diurnal_long_train | latency_degradation_mean | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | -0.1155 | 0.004184 | 15 |
| diurnal_long_train | latency_violation_fraction | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | -0.04538 | 0.001424 | 15 |
| diurnal_long_train | completion_hours | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | 7.7 | 0.4923 | 15 |
| diurnal_long_train | mean_delivered_mbps | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | -16.42 | 1.005 | 15 |
| diurnal_long_train | p_failure_mean | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | 9e-05 | 1.461e-05 | 15 |
| diurnal_long_train | clip_fraction | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | 0 | 0 | 15 |

## Total reward difference by definition of controllable: risk_aware_rule_based minus risk_aware_ceiling_persistence (literal spec)

| profile | definition | n_controllable | mean_diff | sem | t | better | worse |
|---|---|---|---|---|---|---|---|
| diurnal_long_train | primary | 15 | 233.9 | 7.502 | 31.18 | 15 | 0 |
| diurnal_long_train | clip_0.01 | 12 | 233.1 | 9.391 | 24.82 | 12 | 0 |
| diurnal_long_train | clip_0.25 | 16 | 231.2 | 7.514 | 30.77 | 16 | 0 |
| diurnal_long_train | fixed20_cost | 17 | 230.3 | 7.114 | 32.37 | 17 | 0 |

## Paired difference: risk_aware_rule_based k=1 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_train | total_reward | risk_aware_rule_based k=1 | risk_aware_rule_based | -53.54 | 50.11 | 20 |
| diurnal_long_train | latency_degradation_mean | risk_aware_rule_based k=1 | risk_aware_rule_based | 0.04416 | 0.04309 | 20 |
| diurnal_long_train | latency_violation_fraction | risk_aware_rule_based k=1 | risk_aware_rule_based | 0.009245 | 0.008598 | 20 |
| diurnal_long_train | completion_hours | risk_aware_rule_based k=1 | risk_aware_rule_based | -0.4458 | 0.3266 | 20 |
| diurnal_long_train | mean_delivered_mbps | risk_aware_rule_based k=1 | risk_aware_rule_based | 1.018 | 0.7879 | 20 |
| diurnal_long_train | p_failure_mean | risk_aware_rule_based k=1 | risk_aware_rule_based | -3.755e-05 | 2.564e-05 | 20 |
| diurnal_long_train | clip_fraction | risk_aware_rule_based k=1 | risk_aware_rule_based | -0.02324 | 0.01607 | 20 |

## Paired difference, controllable scenarios: risk_aware_rule_based k=1 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_train | total_reward | risk_aware_rule_based k=1 | risk_aware_rule_based | -0.7209 | 0.7209 | 15 |
| diurnal_long_train | latency_degradation_mean | risk_aware_rule_based k=1 | risk_aware_rule_based | 0.000257 | 0.000257 | 15 |
| diurnal_long_train | latency_violation_fraction | risk_aware_rule_based k=1 | risk_aware_rule_based | 0.0001338 | 0.0001338 | 15 |
| diurnal_long_train | completion_hours | risk_aware_rule_based k=1 | risk_aware_rule_based | -0.06111 | 0.06111 | 15 |
| diurnal_long_train | mean_delivered_mbps | risk_aware_rule_based k=1 | risk_aware_rule_based | 0.09537 | 0.09537 | 15 |
| diurnal_long_train | p_failure_mean | risk_aware_rule_based k=1 | risk_aware_rule_based | -6.275e-06 | 6.275e-06 | 15 |
| diurnal_long_train | clip_fraction | risk_aware_rule_based k=1 | risk_aware_rule_based | 0 | 0 | 15 |

## Total reward difference by definition of controllable: risk_aware_rule_based k=1 minus risk_aware_rule_based

| profile | definition | n_controllable | mean_diff | sem | t | better | worse |
|---|---|---|---|---|---|---|---|
| diurnal_long_train | primary | 15 | -0.7209 | 0.7209 | -1 | 0 | 1 |
| diurnal_long_train | clip_0.01 | 12 | 0 | 0 | n/a | 0 | 0 |
| diurnal_long_train | clip_0.25 | 16 | -0.6758 | 0.6758 | -1 | 0 | 1 |
| diurnal_long_train | fixed20_cost | 17 | -3.919 | 3.305 | -1.186 | 0 | 2 |
