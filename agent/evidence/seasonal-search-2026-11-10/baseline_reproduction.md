# Baseline comparison (simulated)

Simulator configuration `eval-fleet,sigma=0.25`; latency model uncalibrated (sim/LIMITATIONS.md); workload provenance is listed under Workloads. 20 paired scenario seeds (0..19), identical for every policy. `completion_h` is the mean over completed episodes only. `lat_budget_met` and `risk_viol` are fractions of episodes; `viol_frac`, `clip_frac` and `stall_frac` are fractions of intervals. Latency terms are the mean-latency model of the simulator; `p_fail_*` are the filter's exposure for the applied allocation (nominal rate).

## Workloads

- `diurnal_long_holdout`: source `data_pipeline/profiles/profile_diurnal_long_holdout.csv`, synthetic: false
  - sha256: `2c83b9db2e80b0a85afc996ea7df525692f89b331d07eec26c14011bd06a59b1`
  - disclosure: selected extremes, not a representative sample

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
