# Trained policy against the baselines (simulated)

Simulator configuration `train-fleet,sigma=0.25`; latency model uncalibrated (sim/LIMITATIONS.md). 20 paired scenario seeds (0..19), identical for every policy. Learned policies act by their mode (argmax). `completion_h` is the mean over completed episodes only.

## Checkpoints

Evaluated under configuration `train-fleet,sigma=0.25`.

| checkpoint | seed | iterations | trained_on | eval_fleet | hazard_error_sigma | ablate_hazard |
|---|---|---|---|---|---|---|
| runs\barrier_244\seed0\ckpt_final.pt | 0 | 244 | data_pipeline/profiles/profile_diurnal_long_train.csv | false | 0.25 | false |
| runs\barrier_244\seed1\ckpt_final.pt | 1 | 244 | data_pipeline/profiles/profile_diurnal_long_train.csv | false | 0.25 | false |
| runs\barrier_244\seed2\ckpt_final.pt | 2 | 244 | data_pipeline/profiles/profile_diurnal_long_train.csv | false | 0.25 | false |

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
| diurnal_long_holdout | ppo_lagrangian seed0 | false | 20 | 20 | 0 | 0 | 113.9 | 0.2327 | 0.7008 | 0.04577 | 0.85 | 0.002086 | 0.004546 | 0.1 | 0.1329 | 33.87 | 0 | -473.5 | 197.5 |
| diurnal_long_holdout | ppo_lagrangian seed1 | false | 20 | 20 | 0 | 0 | 54.58 | 0.2719 | 0.85 | 0.05367 | 0.85 | 0.001476 | 0.003117 | 0.1 | 0.08449 | 62.88 | 0 | -402.3 | 202.9 |
| diurnal_long_holdout | ppo_lagrangian seed2 | false | 20 | 20 | 0 | 0 | 81.15 | 0.2578 | 0.85 | 0.0508 | 0.85 | 0.002066 | 0.004542 | 0.1 | 0.1469 | 45.14 | 0 | -434.2 | 200.4 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Paired difference: ppo_lagrangian seed0 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | ppo_lagrangian seed0 | risk_aware_rule_based | -79.33 | 6.708 | 20 |
| diurnal_long_holdout | latency_degradation_mean | ppo_lagrangian seed0 | risk_aware_rule_based | -0.06587 | 0.03631 | 20 |
| diurnal_long_holdout | latency_violation_fraction | ppo_lagrangian seed0 | risk_aware_rule_based | -0.01265 | 0.007143 | 20 |
| diurnal_long_holdout | completion_hours | ppo_lagrangian seed0 | risk_aware_rule_based | 70.45 | 5.922 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | ppo_lagrangian seed0 | risk_aware_rule_based | -43.96 | 2.121 | 20 |
| diurnal_long_holdout | p_failure_mean | ppo_lagrangian seed0 | risk_aware_rule_based | 0.0007146 | 0.0001315 | 20 |
| diurnal_long_holdout | clip_fraction | ppo_lagrangian seed0 | risk_aware_rule_based | 0.09299 | 0.03468 | 20 |

## Paired difference: ppo_lagrangian seed1 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | ppo_lagrangian seed1 | risk_aware_rule_based | -8.146 | 1.16 | 20 |
| diurnal_long_holdout | latency_degradation_mean | ppo_lagrangian seed1 | risk_aware_rule_based | -0.02668 | 0.01304 | 20 |
| diurnal_long_holdout | latency_violation_fraction | ppo_lagrangian seed1 | risk_aware_rule_based | -0.004752 | 0.00257 | 20 |
| diurnal_long_holdout | completion_hours | ppo_lagrangian seed1 | risk_aware_rule_based | 11.14 | 0.9138 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | ppo_lagrangian seed1 | risk_aware_rule_based | -14.95 | 0.9458 | 20 |
| diurnal_long_holdout | p_failure_mean | ppo_lagrangian seed1 | risk_aware_rule_based | 0.000105 | 3.393e-05 | 20 |
| diurnal_long_holdout | clip_fraction | ppo_lagrangian seed1 | risk_aware_rule_based | 0.04458 | 0.02491 | 20 |

## Paired difference: ppo_lagrangian seed2 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | ppo_lagrangian seed2 | risk_aware_rule_based | -40.03 | 3.992 | 20 |
| diurnal_long_holdout | latency_degradation_mean | ppo_lagrangian seed2 | risk_aware_rule_based | -0.04079 | 0.02211 | 20 |
| diurnal_long_holdout | latency_violation_fraction | ppo_lagrangian seed2 | risk_aware_rule_based | -0.007621 | 0.004361 | 20 |
| diurnal_long_holdout | completion_hours | ppo_lagrangian seed2 | risk_aware_rule_based | 37.71 | 3.592 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | ppo_lagrangian seed2 | risk_aware_rule_based | -32.69 | 2.159 | 20 |
| diurnal_long_holdout | p_failure_mean | ppo_lagrangian seed2 | risk_aware_rule_based | 0.0006944 | 0.000101 | 20 |
| diurnal_long_holdout | clip_fraction | ppo_lagrangian seed2 | risk_aware_rule_based | 0.107 | 0.03395 | 20 |
