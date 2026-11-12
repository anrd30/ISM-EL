# Trained policy against the baselines (simulated)

Simulator configuration `train-fleet,sigma=0.25`; latency model uncalibrated (sim/LIMITATIONS.md). 20 paired scenario seeds (0..19), identical for every policy. Learned policies act by their mode (argmax). `completion_h` is the mean over completed episodes only.

## Checkpoints

Evaluated under configuration `train-fleet,sigma=0.25`.

| checkpoint | seed | iterations | trained_on | eval_fleet | hazard_error_sigma | ablate_hazard |
|---|---|---|---|---|---|---|
| runs\l3_244\seed0\ckpt_final.pt | 0 | 244 | data_pipeline/profiles/profile_diurnal_long_train.csv | false | 0.25 | false |
| runs\l3_244\seed1\ckpt_final.pt | 1 | 244 | data_pipeline/profiles/profile_diurnal_long_train.csv | false | 0.25 | false |
| runs\l3_244\seed2\ckpt_final.pt | 2 | 244 | data_pipeline/profiles/profile_diurnal_long_train.csv | false | 0.25 | false |

## Workloads

- `diurnal_long_holdout`: source `data_pipeline/profiles/profile_diurnal_long_holdout.csv`, synthetic: false
  - sha256: `94fe6bd6f8a5cb163cba7a9fafac1274d4ec6511113920a63d4993f03295a53d`
  - disclosure: selected extremes, not a representative sample

| profile | policy | workload_synthetic | n | completed | horizon_expired | terminal_failure | completion_h | lat_deg_mean | lat_deg_p95 | viol_frac | lat_budget_met | p_fail_mean | p_fail_max | risk_viol | clip_frac | deliv_mbps | stall_frac | reward | reward_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_holdout | fixed_20 | false | 20 | 20 | 0 | 0 | 142.8 | 0.2217 | 0.6196 | 0.04356 | 0.85 | 0.002198 | 0.004547 | 0.1 | 0.1373 | 28.5 | 0 | -508.2 | 194.8 |
| diurnal_long_holdout | fixed_50 | false | 20 | 20 | 0 | 0 | 62.83 | 0.2578 | 0.85 | 0.0508 | 0.85 | 0.001528 | 0.003129 | 0.1 | 0.09167 | 55.14 | 0 | -412.2 | 202.3 |
| diurnal_long_holdout | fixed_80 | false | 20 | 20 | 0 | 0 | 40.62 | 0.3389 | 1.135 | 0.08692 | 0.85 | 0.001303 | 0.002649 | 0.1 | 0.07811 | 82.67 | 0 | -559 | 191.9 |
| diurnal_long_holdout | fixed_120 | false | 20 | 20 | 0 | 0 | 29.27 | 3.498 | 5.667 | 0.6726 | 0 | 0.00117 | 0.002331 | 0.1 | 0 | 114 | 0 | -3630 | 90.84 |
| diurnal_long_holdout | workload_only_throttle | false | 20 | 20 | 0 | 0 | 43.44 | 0.2986 | 0.85 | 0.05842 | 0.85 | 0.001371 | 0.002983 | 0.1 | 0.07804 | 77.83 | 0 | -394.2 | 203.4 |
| diurnal_long_holdout | risk_aware_rule_based | false | 20 | 20 | 0 | 0 | 43.44 | 0.2986 | 0.85 | 0.05842 | 0.85 | 0.001371 | 0.002983 | 0.1 | 0.03991 | 77.83 | 0 | -394.2 | 203.4 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | false | 20 | 20 | 0 | 0 | 36.66 | 0.3121 | 0.8833 | 0.06082 | 0.85 | 0.001292 | 0.002977 | 0.1 | 0.03813 | 91.55 | 0 | -382.8 | 204.2 |
| diurnal_long_holdout | slowest_risk_feasible (literal spec) | false | 20 | 20 | 0 | 0 | 142.8 | 0.2217 | 0.6196 | 0.04356 | 0.85 | 0.002198 | 0.004547 | 0.1 | 0.05467 | 28.5 | 0 | -508.2 | 194.8 |
| diurnal_long_holdout | risk_aware_ceiling_persistence (literal spec) | false | 20 | 20 | 0 | 0 | 35.73 | 0.4275 | 1.281 | 0.1073 | 0.35 | 0.001279 | 0.002928 | 0.1 | 0.03794 | 93.86 | 0 | -630.8 | 187.1 |
| diurnal_long_holdout | risk_aware_rule_based k=1 | false | 20 | 20 | 0 | 0 | 43.1 | 0.3819 | 0.85 | 0.07296 | 0.85 | 0.001341 | 0.002885 | 0.1 | 0.008974 | 78.54 | 0 | -490 | 233 |
| diurnal_long_holdout | ppo_lagrangian seed0 | false | 20 | 20 | 0 | 0 | 58.53 | 0.2945 | 0.85 | 0.059 | 0.85 | 0.001492 | 0.00312 | 0.1 | 0.04537 | 60 | 0 | -418.1 | 205 |
| diurnal_long_holdout | ppo_lagrangian seed1 | false | 20 | 20 | 0 | 0 | 65.25 | 0.2221 | 0.6314 | 0.04365 | 0.85 | 0.001715 | 0.004544 | 0.1 | 0.1662 | 52.51 | 0 | -415.1 | 202.8 |
| diurnal_long_holdout | ppo_lagrangian seed2 | false | 20 | 20 | 0 | 0 | 64.73 | 0.2848 | 0.85 | 0.05651 | 0.85 | 0.001557 | 0.003186 | 0.1 | 0.04385 | 54.77 | 0 | -416.9 | 203.3 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Paired difference: ppo_lagrangian seed0 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | ppo_lagrangian seed0 | risk_aware_rule_based | -23.96 | 8.102 | 20 |
| diurnal_long_holdout | latency_degradation_mean | ppo_lagrangian seed0 | risk_aware_rule_based | -0.004131 | 0.01234 | 20 |
| diurnal_long_holdout | latency_violation_fraction | ppo_lagrangian seed0 | risk_aware_rule_based | 0.0005848 | 0.002845 | 20 |
| diurnal_long_holdout | completion_hours | ppo_lagrangian seed0 | risk_aware_rule_based | 15.09 | 1.704 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | ppo_lagrangian seed0 | risk_aware_rule_based | -17.83 | 1.97 | 20 |
| diurnal_long_holdout | p_failure_mean | ppo_lagrangian seed0 | risk_aware_rule_based | 0.0001205 | 3.011e-05 | 20 |
| diurnal_long_holdout | clip_fraction | ppo_lagrangian seed0 | risk_aware_rule_based | 0.005458 | 0.003052 | 20 |

## Paired difference: ppo_lagrangian seed1 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | ppo_lagrangian seed1 | risk_aware_rule_based | -20.95 | 1.521 | 20 |
| diurnal_long_holdout | latency_degradation_mean | ppo_lagrangian seed1 | risk_aware_rule_based | -0.07652 | 0.04266 | 20 |
| diurnal_long_holdout | latency_violation_fraction | ppo_lagrangian seed1 | risk_aware_rule_based | -0.01477 | 0.00838 | 20 |
| diurnal_long_holdout | completion_hours | ppo_lagrangian seed1 | risk_aware_rule_based | 21.8 | 1.242 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | ppo_lagrangian seed1 | risk_aware_rule_based | -25.32 | 0.7003 | 20 |
| diurnal_long_holdout | p_failure_mean | ppo_lagrangian seed1 | risk_aware_rule_based | 0.0003436 | 0.0001176 | 20 |
| diurnal_long_holdout | clip_fraction | ppo_lagrangian seed1 | risk_aware_rule_based | 0.1263 | 0.04119 | 20 |

## Paired difference: ppo_lagrangian seed2 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | ppo_lagrangian seed2 | risk_aware_rule_based | -22.68 | 2.129 | 20 |
| diurnal_long_holdout | latency_degradation_mean | ppo_lagrangian seed2 | risk_aware_rule_based | -0.01383 | 0.009823 | 20 |
| diurnal_long_holdout | latency_violation_fraction | ppo_lagrangian seed2 | risk_aware_rule_based | -0.001911 | 0.002116 | 20 |
| diurnal_long_holdout | completion_hours | ppo_lagrangian seed2 | risk_aware_rule_based | 21.29 | 1.986 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | ppo_lagrangian seed2 | risk_aware_rule_based | -23.06 | 1.952 | 20 |
| diurnal_long_holdout | p_failure_mean | ppo_lagrangian seed2 | risk_aware_rule_based | 0.0001856 | 4.196e-05 | 20 |
| diurnal_long_holdout | clip_fraction | ppo_lagrangian seed2 | risk_aware_rule_based | 0.003938 | 0.02285 | 20 |
