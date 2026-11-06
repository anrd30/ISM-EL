# Trained policy against the baselines (simulated)

Simulator configuration `eval-fleet,sigma=0.25`; latency model uncalibrated (sim/LIMITATIONS.md). 20 paired scenario seeds (0..19), identical for every policy. Learned policies act by their mode (argmax). `completion_h` is the mean over completed episodes only.

## Checkpoints

Evaluated under configuration `eval-fleet,sigma=0.25`.

| checkpoint | seed | iterations | trained_on | eval_fleet | hazard_error_sigma | ablate_hazard |
|---|---|---|---|---|---|---|
| runs\ppo-clean\seed0\ckpt_final.pt | 0 | 244 | data_pipeline/profiles/profile_diurnal_long_train.csv | true | 0.25 | false |
| runs\ppo-clean\seed1\ckpt_final.pt | 1 | 244 | data_pipeline/profiles/profile_diurnal_long_train.csv | true | 0.25 | false |
| runs\ppo-clean\seed2\ckpt_final.pt | 2 | 244 | data_pipeline/profiles/profile_diurnal_long_train.csv | true | 0.25 | false |

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
| diurnal_long_holdout | ppo_lagrangian seed0 | false | 20 | 20 | 0 | 0 | 64.05 | 0.1816 | 0.85 | 0.03524 | 0.85 | 0.001231 | 0.002469 | 0.1 | 0.08188 | 53.44 | 0 | -349.4 | 157.1 |
| diurnal_long_holdout | ppo_lagrangian seed1 | false | 20 | 20 | 0 | 0 | 63 | 0.1845 | 0.85 | 0.03578 | 0.85 | 0.001226 | 0.002381 | 0.1 | 0.07793 | 54.29 | 0 | -348.2 | 157.1 |
| diurnal_long_holdout | ppo_lagrangian seed2 | false | 20 | 20 | 0 | 0 | 64.46 | 0.1965 | 0.85 | 0.03812 | 0.85 | 0.001352 | 0.003823 | 0.1 | 0.05014 | 53.8 | 0 | -353.1 | 157.8 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Paired difference: ppo_lagrangian seed0 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | ppo_lagrangian seed0 | risk_aware_rule_based | -19.23 | 1.568 | 20 |
| diurnal_long_holdout | latency_degradation_mean | ppo_lagrangian seed0 | risk_aware_rule_based | -0.04983 | 0.02546 | 20 |
| diurnal_long_holdout | latency_violation_fraction | ppo_lagrangian seed0 | risk_aware_rule_based | -0.009155 | 0.004914 | 20 |
| diurnal_long_holdout | completion_hours | ppo_lagrangian seed0 | risk_aware_rule_based | 20.45 | 1.209 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | ppo_lagrangian seed0 | risk_aware_rule_based | -23.89 | 0.7167 | 20 |
| diurnal_long_holdout | p_failure_mean | ppo_lagrangian seed0 | risk_aware_rule_based | 0.0001521 | 2.148e-05 | 20 |
| diurnal_long_holdout | clip_fraction | ppo_lagrangian seed0 | risk_aware_rule_based | 0.05225 | 0.02577 | 20 |

## Paired difference: ppo_lagrangian seed1 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | ppo_lagrangian seed1 | risk_aware_rule_based | -17.97 | 1.492 | 20 |
| diurnal_long_holdout | latency_degradation_mean | ppo_lagrangian seed1 | risk_aware_rule_based | -0.04703 | 0.02351 | 20 |
| diurnal_long_holdout | latency_violation_fraction | ppo_lagrangian seed1 | risk_aware_rule_based | -0.008615 | 0.004546 | 20 |
| diurnal_long_holdout | completion_hours | ppo_lagrangian seed1 | risk_aware_rule_based | 19.4 | 1.11 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | ppo_lagrangian seed1 | risk_aware_rule_based | -23.05 | 0.6549 | 20 |
| diurnal_long_holdout | p_failure_mean | ppo_lagrangian seed1 | risk_aware_rule_based | 0.000147 | 2.308e-05 | 20 |
| diurnal_long_holdout | clip_fraction | ppo_lagrangian seed1 | risk_aware_rule_based | 0.04831 | 0.0237 | 20 |

## Paired difference: ppo_lagrangian seed2 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | ppo_lagrangian seed2 | risk_aware_rule_based | -22.86 | 2.201 | 20 |
| diurnal_long_holdout | latency_degradation_mean | ppo_lagrangian seed2 | risk_aware_rule_based | -0.03497 | 0.01784 | 20 |
| diurnal_long_holdout | latency_violation_fraction | ppo_lagrangian seed2 | risk_aware_rule_based | -0.006277 | 0.003426 | 20 |
| diurnal_long_holdout | completion_hours | ppo_lagrangian seed2 | risk_aware_rule_based | 20.86 | 1.635 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | ppo_lagrangian seed2 | risk_aware_rule_based | -23.54 | 1.185 | 20 |
| diurnal_long_holdout | p_failure_mean | ppo_lagrangian seed2 | risk_aware_rule_based | 0.0002739 | 4.733e-05 | 20 |
| diurnal_long_holdout | clip_fraction | ppo_lagrangian seed2 | risk_aware_rule_based | 0.02052 | 0.01935 | 20 |
