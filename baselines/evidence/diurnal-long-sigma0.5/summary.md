# Baseline comparison (simulated)

Simulator configuration `eval-fleet,sigma=0.5`; latency model uncalibrated (sim/LIMITATIONS.md); workload provenance is listed under Workloads. 20 paired scenario seeds (0..19), identical for every policy. `completion_h` is the mean over completed episodes only. `lat_budget_met` and `risk_viol` are fractions of episodes; `viol_frac`, `clip_frac` and `stall_frac` are fractions of intervals. Latency terms are the mean-latency model of the simulator; `p_fail_*` are the filter's exposure for the applied allocation (nominal rate).

## Workloads

- `diurnal_long`: source `data_pipeline/profiles/profile_diurnal_long.csv`, synthetic: false
  - sha256: `4b0f6cf85fc4f88bf80ac23d44120c127c88e5ad41f46eb3cfc2fec576c74001`
  - disclosure: selected extremes, not a representative sample

| profile | policy | workload_synthetic | n | completed | horizon_expired | terminal_failure | completion_h | lat_deg_mean | lat_deg_p95 | viol_frac | lat_budget_met | p_fail_mean | p_fail_max | risk_viol | clip_frac | deliv_mbps | stall_frac | reward | reward_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long | fixed_20 | false | 20 | 20 | 0 | 0 | 144.8 | 0.1179 | 0.5974 | 0.02464 | 0.85 | 0.001985 | 0.003874 | 0.1 | 0.1227 | 26.92 | 0 | -406.1 | 129.5 |
| diurnal_long | fixed_50 | false | 20 | 20 | 0 | 0 | 63.06 | 0.1492 | 0.6705 | 0.03123 | 0.85 | 0.001231 | 0.002382 | 0.1 | 0.08149 | 54.16 | 0 | -308 | 136.8 |
| diurnal_long | fixed_80 | false | 20 | 20 | 0 | 0 | 40.8 | 0.2291 | 1.109 | 0.06666 | 0.85 | 0.0009785 | 0.001919 | 0.1 | 0.05767 | 82.07 | 0 | -450 | 127.9 |
| diurnal_long | fixed_120 | false | 20 | 20 | 0 | 0 | 29.27 | 3.488 | 5.667 | 0.6773 | 0 | 0.000801 | 0.0016 | 0.1 | 0 | 114.1 | 0 | -3642 | 90.63 |
| diurnal_long | workload_only_throttle | false | 20 | 20 | 0 | 0 | 42.93 | 0.1895 | 0.7894 | 0.03976 | 0.85 | 0.001053 | 0.002301 | 0.1 | 0.07161 | 78.4 | 0 | -293.3 | 137.7 |
| diurnal_long | risk_aware_rule_based | false | 20 | 20 | 0 | 0 | 42.93 | 0.1895 | 0.7894 | 0.03976 | 0.85 | 0.001053 | 0.002301 | 0.1 | 0.02605 | 78.4 | 0 | -293.3 | 137.7 |
| diurnal_long | ORACLE_lookahead (not deployable) | false | 20 | 20 | 0 | 0 | 36.6 | 0.1972 | 0.883 | 0.04052 | 0.85 | 0.0009633 | 0.002312 | 0.1 | 0.01945 | 91.45 | 0 | -278.1 | 138.5 |
| diurnal_long | slowest_risk_feasible (literal spec) | false | 20 | 20 | 0 | 0 | 144.8 | 0.1179 | 0.5974 | 0.02464 | 0.85 | 0.001985 | 0.003874 | 0.1 | 0.03306 | 26.92 | 0 | -406.1 | 129.5 |
| diurnal_long | risk_aware_ceiling_persistence (literal spec) | false | 20 | 20 | 0 | 0 | 35.75 | 0.3087 | 1.243 | 0.0856 | 0.5 | 0.0009479 | 0.002218 | 0.1 | 0.01817 | 93.61 | 0 | -517.9 | 126.4 |
| diurnal_long | risk_aware_rule_based k=1 | false | 20 | 20 | 0 | 0 | 42.49 | 0.3403 | 1.133 | 0.06732 | 0.8 | 0.0009915 | 0.002158 | 0.1 | 0.0002481 | 79.43 | 0 | -473.5 | 189.7 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Paired difference: risk_aware_rule_based minus workload_only_throttle

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long | total_reward | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 20 |
| diurnal_long | latency_degradation_mean | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 20 |
| diurnal_long | latency_violation_fraction | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 20 |
| diurnal_long | completion_hours | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 20 |
| diurnal_long | mean_delivered_mbps | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 20 |
| diurnal_long | p_failure_mean | risk_aware_rule_based | workload_only_throttle | 0 | 0 | 20 |
| diurnal_long | clip_fraction | risk_aware_rule_based | workload_only_throttle | -0.04556 | 0.023 | 20 |

## Paired difference: ORACLE_lookahead (not deployable) minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long | total_reward | ORACLE_lookahead (not deployable) | risk_aware_rule_based | 15.25 | 1.898 | 20 |
| diurnal_long | latency_degradation_mean | ORACLE_lookahead (not deployable) | risk_aware_rule_based | 0.007701 | 0.005431 | 20 |
| diurnal_long | latency_violation_fraction | ORACLE_lookahead (not deployable) | risk_aware_rule_based | 0.0007537 | 0.001196 | 20 |
| diurnal_long | completion_hours | ORACLE_lookahead (not deployable) | risk_aware_rule_based | -6.329 | 0.557 | 20 |
| diurnal_long | mean_delivered_mbps | ORACLE_lookahead (not deployable) | risk_aware_rule_based | 13.05 | 1.027 | 20 |
| diurnal_long | p_failure_mean | ORACLE_lookahead (not deployable) | risk_aware_rule_based | -8.981e-05 | 1.835e-05 | 20 |
| diurnal_long | clip_fraction | ORACLE_lookahead (not deployable) | risk_aware_rule_based | -0.0066 | 0.004655 | 20 |

## Paired difference: risk_aware_rule_based minus slowest_risk_feasible (literal spec)

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long | total_reward | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 112.8 | 9.808 | 20 |
| diurnal_long | latency_degradation_mean | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 0.07161 | 0.04041 | 20 |
| diurnal_long | latency_violation_fraction | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 0.01513 | 0.008307 | 20 |
| diurnal_long | completion_hours | risk_aware_rule_based | slowest_risk_feasible (literal spec) | -101.9 | 8.503 | 20 |
| diurnal_long | mean_delivered_mbps | risk_aware_rule_based | slowest_risk_feasible (literal spec) | 51.48 | 1.82 | 20 |
| diurnal_long | p_failure_mean | risk_aware_rule_based | slowest_risk_feasible (literal spec) | -0.0009318 | 0.0001203 | 20 |
| diurnal_long | clip_fraction | risk_aware_rule_based | slowest_risk_feasible (literal spec) | -0.007014 | 0.006349 | 20 |

## Paired difference: risk_aware_rule_based minus risk_aware_ceiling_persistence (literal spec)

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long | total_reward | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | 224.6 | 13.67 | 20 |
| diurnal_long | latency_degradation_mean | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | -0.1192 | 0.003578 | 20 |
| diurnal_long | latency_violation_fraction | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | -0.04583 | 0.001512 | 20 |
| diurnal_long | completion_hours | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | 7.188 | 0.6158 | 20 |
| diurnal_long | mean_delivered_mbps | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | -15.21 | 1.166 | 20 |
| diurnal_long | p_failure_mean | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | 0.0001052 | 2.152e-05 | 20 |
| diurnal_long | clip_fraction | risk_aware_rule_based | risk_aware_ceiling_persistence (literal spec) | 0.007876 | 0.005684 | 20 |

## Paired difference: risk_aware_rule_based k=1 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long | total_reward | risk_aware_rule_based k=1 | risk_aware_rule_based | -180.1 | 111.3 | 20 |
| diurnal_long | latency_degradation_mean | risk_aware_rule_based k=1 | risk_aware_rule_based | 0.1508 | 0.09051 | 20 |
| diurnal_long | latency_violation_fraction | risk_aware_rule_based k=1 | risk_aware_rule_based | 0.02756 | 0.01652 | 20 |
| diurnal_long | completion_hours | risk_aware_rule_based k=1 | risk_aware_rule_based | -0.4458 | 0.2809 | 20 |
| diurnal_long | mean_delivered_mbps | risk_aware_rule_based k=1 | risk_aware_rule_based | 1.028 | 0.636 | 20 |
| diurnal_long | p_failure_mean | risk_aware_rule_based k=1 | risk_aware_rule_based | -6.161e-05 | 3.938e-05 | 20 |
| diurnal_long | clip_fraction | risk_aware_rule_based k=1 | risk_aware_rule_based | -0.0258 | 0.01778 | 20 |
