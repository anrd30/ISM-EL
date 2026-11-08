# Trained policy against the baselines (simulated)

Simulator configuration `eval-fleet,sigma=0.25`; latency model uncalibrated (sim/LIMITATIONS.md). 20 paired scenario seeds (0..19), identical for every policy. Learned policies act by their mode (argmax). `completion_h` is the mean over completed episodes only.

## Checkpoints

Evaluated under configuration `eval-fleet,sigma=0.25`.

| checkpoint | seed | iterations | trained_on | eval_fleet | hazard_error_sigma | ablate_hazard |
|---|---|---|---|---|---|---|
| runs\b2_244\seed0\ckpt_final.pt | 0 | 244 | data_pipeline/profiles/profile_diurnal_long_train.csv | true | 0.25 | false |
| runs\b2_244\seed1\ckpt_final.pt | 1 | 244 | data_pipeline/profiles/profile_diurnal_long_train.csv | true | 0.25 | false |
| runs\b2_244\seed2\ckpt_final.pt | 2 | 244 | data_pipeline/profiles/profile_diurnal_long_train.csv | true | 0.25 | false |

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
| diurnal_long_holdout | ppo_lagrangian seed0 | false | 20 | 20 | 0 | 0 | 57.29 | 0.1709 | 0.85 | 0.0332 | 0.85 | 0.001184 | 0.002871 | 0.1 | 0.1091 | 58.79 | 0 | -341.3 | 158 |
| diurnal_long_holdout | ppo_lagrangian seed1 | false | 20 | 20 | 0 | 0 | 54.97 | 0.2409 | 0.85 | 0.04565 | 0.85 | 0.001167 | 0.002378 | 0.1 | 0.03502 | 61.52 | 0 | -414.3 | 191.5 |
| diurnal_long_holdout | ppo_lagrangian seed2 | false | 20 | 20 | 0 | 0 | 61.46 | 0.1499 | 0.85 | 0.02909 | 0.85 | 0.001412 | 0.003857 | 0.1 | 0.1415 | 54.72 | 0 | -346.3 | 158.3 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Paired difference: ppo_lagrangian seed0 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | ppo_lagrangian seed0 | risk_aware_rule_based | -11.12 | 1.294 | 20 |
| diurnal_long_holdout | latency_degradation_mean | ppo_lagrangian seed0 | risk_aware_rule_based | -0.06061 | 0.03485 | 20 |
| diurnal_long_holdout | latency_violation_fraction | ppo_lagrangian seed0 | risk_aware_rule_based | -0.0112 | 0.006667 | 20 |
| diurnal_long_holdout | completion_hours | ppo_lagrangian seed0 | risk_aware_rule_based | 13.69 | 0.5007 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | ppo_lagrangian seed0 | risk_aware_rule_based | -18.55 | 0.7889 | 20 |
| diurnal_long_holdout | p_failure_mean | ppo_lagrangian seed0 | risk_aware_rule_based | 0.000105 | 2.563e-05 | 20 |
| diurnal_long_holdout | clip_fraction | ppo_lagrangian seed0 | risk_aware_rule_based | 0.07949 | 0.03741 | 20 |

## Paired difference: ppo_lagrangian seed1 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | ppo_lagrangian seed1 | risk_aware_rule_based | -84.07 | 48.23 | 20 |
| diurnal_long_holdout | latency_degradation_mean | ppo_lagrangian seed1 | risk_aware_rule_based | 0.009439 | 0.0287 | 20 |
| diurnal_long_holdout | latency_violation_fraction | ppo_lagrangian seed1 | risk_aware_rule_based | 0.00126 | 0.004739 | 20 |
| diurnal_long_holdout | completion_hours | ppo_lagrangian seed1 | risk_aware_rule_based | 11.37 | 0.5645 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | ppo_lagrangian seed1 | risk_aware_rule_based | -15.82 | 0.5448 | 20 |
| diurnal_long_holdout | p_failure_mean | ppo_lagrangian seed1 | risk_aware_rule_based | 8.853e-05 | 2.586e-05 | 20 |
| diurnal_long_holdout | clip_fraction | ppo_lagrangian seed1 | risk_aware_rule_based | 0.005397 | 0.0112 | 20 |

## Paired difference: ppo_lagrangian seed2 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_holdout | total_reward | ppo_lagrangian seed2 | risk_aware_rule_based | -16.13 | 1.74 | 20 |
| diurnal_long_holdout | latency_degradation_mean | ppo_lagrangian seed2 | risk_aware_rule_based | -0.08154 | 0.04466 | 20 |
| diurnal_long_holdout | latency_violation_fraction | ppo_lagrangian seed2 | risk_aware_rule_based | -0.01531 | 0.008583 | 20 |
| diurnal_long_holdout | completion_hours | ppo_lagrangian seed2 | risk_aware_rule_based | 17.86 | 0.8002 | 20 |
| diurnal_long_holdout | mean_delivered_mbps | ppo_lagrangian seed2 | risk_aware_rule_based | -22.62 | 1.234 | 20 |
| diurnal_long_holdout | p_failure_mean | ppo_lagrangian seed2 | risk_aware_rule_based | 0.0003335 | 5.896e-05 | 20 |
| diurnal_long_holdout | clip_fraction | ppo_lagrangian seed2 | risk_aware_rule_based | 0.1119 | 0.04388 | 20 |
