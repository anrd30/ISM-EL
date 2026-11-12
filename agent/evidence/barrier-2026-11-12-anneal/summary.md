# Trained policy against the baselines (simulated)

Simulator configuration `train-fleet,sigma=0.25`; latency model uncalibrated (sim/LIMITATIONS.md). 20 paired scenario seeds (0..19), identical for every policy. Learned policies act by their mode (argmax). `completion_h` is the mean over completed episodes only.

## Checkpoints

Evaluated under configuration `train-fleet,sigma=0.25`.

| checkpoint | seed | iterations | trained_on | eval_fleet | hazard_error_sigma | ablate_hazard |
|---|---|---|---|---|---|---|
| runs\barrier_anneal_244\seed0\ckpt_final.pt | 0 | 244 | data_pipeline/profiles/profile_diurnal_long_train.csv | false | 0.25 | false |
| runs\barrier_anneal_244\seed1\ckpt_final.pt | 1 | 244 | data_pipeline/profiles/profile_diurnal_long_train.csv | false | 0.25 | false |
| runs\barrier_anneal_244\seed2\ckpt_final.pt | 2 | 244 | data_pipeline/profiles/profile_diurnal_long_train.csv | false | 0.25 | false |

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
| diurnal_long_holdout | ppo_lagrangian seed0 | false | 20 | 20 | 0 | 0 | 66 | 0.2719 | 0.85 | 0.05361 | 0.85 | 0.001534 | 0.003359 | 0.1 | 0.03754 | 53.45 | 0 | -419.9 | 202.3 |
| diurnal_long_holdout | ppo_lagrangian seed1 | false | 20 | 20 | 0 | 0 | 60.9 | 0.2428 | 0.85 | 0.04794 | 0.85 | 0.001786 | 0.00452 | 0.1 | 0.168 | 56.37 | 0 | -409.9 | 202.7 |
| diurnal_long_holdout | ppo_lagrangian seed2 | false | 20 | 20 | 0 | 0 | 51.48 | 0.2519 | 0.85 | 0.05082 | 0.85 | 0.001467 | 0.003128 | 0.1 | 0.1046 | 65.66 | 0 | -410.3 | 202.6 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Paired difference: ppo_lagrangian seed0 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | ppo_lagrangian seed0 | risk_aware_rule_based | -25.77 | 3.557 | 20 |
| diurnal_long_holdout | latency_degradation_mean | ppo_lagrangian seed0 | risk_aware_rule_based | -0.02667 | 0.01464 | 20 |
| diurnal_long_holdout | latency_violation_fraction | ppo_lagrangian seed0 | risk_aware_rule_based | -0.004809 | 0.002858 | 20 |
| diurnal_long_holdout | completion_hours | ppo_lagrangian seed0 | risk_aware_rule_based | 22.56 | 1.924 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | ppo_lagrangian seed0 | risk_aware_rule_based | -24.39 | 1.617 | 20 |
| diurnal_long_holdout | p_failure_mean | ppo_lagrangian seed0 | risk_aware_rule_based | 0.0001626 | 3.887e-05 | 20 |
| diurnal_long_holdout | clip_fraction | ppo_lagrangian seed0 | risk_aware_rule_based | -0.002377 | 0.02138 | 20 |

## Paired difference: ppo_lagrangian seed1 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | ppo_lagrangian seed1 | risk_aware_rule_based | -15.74 | 1.299 | 20 |
| diurnal_long_holdout | latency_degradation_mean | ppo_lagrangian seed1 | risk_aware_rule_based | -0.05583 | 0.03186 | 20 |
| diurnal_long_holdout | latency_violation_fraction | ppo_lagrangian seed1 | risk_aware_rule_based | -0.01048 | 0.006213 | 20 |
| diurnal_long_holdout | completion_hours | ppo_lagrangian seed1 | risk_aware_rule_based | 17.46 | 1.14 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | ppo_lagrangian seed1 | risk_aware_rule_based | -21.46 | 0.7717 | 20 |
| diurnal_long_holdout | p_failure_mean | ppo_lagrangian seed1 | risk_aware_rule_based | 0.0004148 | 9.534e-05 | 20 |
| diurnal_long_holdout | clip_fraction | ppo_lagrangian seed1 | risk_aware_rule_based | 0.128 | 0.04505 | 20 |

## Paired difference: ppo_lagrangian seed2 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | ppo_lagrangian seed2 | risk_aware_rule_based | -16.1 | 2.965 | 20 |
| diurnal_long_holdout | latency_degradation_mean | ppo_lagrangian seed2 | risk_aware_rule_based | -0.04674 | 0.0287 | 20 |
| diurnal_long_holdout | latency_violation_fraction | ppo_lagrangian seed2 | risk_aware_rule_based | -0.007596 | 0.005696 | 20 |
| diurnal_long_holdout | completion_hours | ppo_lagrangian seed2 | risk_aware_rule_based | 8.042 | 0.3456 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | ppo_lagrangian seed2 | risk_aware_rule_based | -12.18 | 0.5601 | 20 |
| diurnal_long_holdout | p_failure_mean | ppo_lagrangian seed2 | risk_aware_rule_based | 9.586e-05 | 6.766e-05 | 20 |
| diurnal_long_holdout | clip_fraction | ppo_lagrangian seed2 | risk_aware_rule_based | 0.0647 | 0.03777 | 20 |
