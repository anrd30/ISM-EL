# Learned controllers against the baselines (simulated)

Simulator configuration `v1,eval-fleet,sigma=0.25`; latency model uncalibrated (sim/LIMITATIONS.md). Workload `data_pipeline/profiles/profile_diurnal_long_sfold1_holdout.csv` (sha256 `d9d949023440eeebedb9ec89287ae5845aca4675a32efd019f5925d59f3bfdf1`). 20 paired scenario seeds (0..19), identical for every policy. Learned policies act by sampling. Reward is the contract reward with contract coefficients.

## Checkpoints

| checkpoint | policy | iteration | trained_on | hazard | train_fleet | sigma | val_gap | val_gap_sem | val_seeds | params |
|---|---|---|---|---|---|---|---|---|---|---|
| runs/forecast_ppo_sfold1/bc/seed0/ckpt_best.pt | forecast_ppo_bc seed0 | 60 | data_pipeline/profiles/profile_diurnal_long_sfold1_train.csv | v1 | train-fleet | 0.25 | 5.361 | 2.01 | 5000..5039 | bc_accuracy=0.959 |
| runs/forecast_ppo_sfold1/bc/seed1/ckpt_best.pt | forecast_ppo_bc seed1 | 120 | data_pipeline/profiles/profile_diurnal_long_sfold1_train.csv | v1 | train-fleet | 0.25 | 5.42 | 1.976 | 5000..5039 | bc_accuracy=0.959 |
| runs/forecast_ppo_sfold1/bc/seed2/ckpt_best.pt | forecast_ppo_bc seed2 | 70 | data_pipeline/profiles/profile_diurnal_long_sfold1_train.csv | v1 | train-fleet | 0.25 | 4.838 | 1.899 | 5000..5039 | bc_accuracy=0.961 |

## Summary

| profile | policy | workload_synthetic | n | completed | horizon_expired | terminal_failure | completion_h | lat_deg_mean | lat_deg_p95 | viol_frac | lat_budget_met | p_fail_mean | p_fail_max | risk_viol | clip_frac | deliv_mbps | stall_frac | reward | reward_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_sfold1_holdout | fixed_20 | false | 20 | 20 | 0 | 0 | 144.8 | 0.139 | 0.85 | 0.02749 | 0.85 | 0.001984 | 0.003874 | 0.1 | 0.1229 | 26.91 | 0 | -442.4 | 147 |
| diurnal_long_sfold1_holdout | fixed_50 | false | 20 | 20 | 0 | 0 | 63.08 | 0.1764 | 0.85 | 0.03492 | 0.85 | 0.00123 | 0.002381 | 0.1 | 0.08176 | 54.13 | 0 | -344.3 | 154.4 |
| diurnal_long_sfold1_holdout | fixed_80 | false | 20 | 20 | 0 | 0 | 40.79 | 0.251 | 1.077 | 0.06575 | 0.85 | 0.0009773 | 0.001919 | 0.1 | 0.05819 | 82.02 | 0 | -453.3 | 147.3 |
| diurnal_long_sfold1_holdout | fixed_120 | false | 20 | 20 | 0 | 0 | 29.13 | 3.427 | 5.667 | 0.6684 | 0 | 0.0008024 | 0.0016 | 0.1 | 0 | 114.6 | 0 | -3572 | 90.66 |
| diurnal_long_sfold1_holdout | workload_only_throttle | false | 20 | 20 | 0 | 0 | 42.91 | 0.2277 | 0.85 | 0.04448 | 0.85 | 0.001068 | 0.002299 | 0.1 | 0.06099 | 78.51 | 0 | -325.5 | 155.2 |
| diurnal_long_sfold1_holdout | risk_aware_rule_based | false | 20 | 20 | 0 | 0 | 42.91 | 0.2277 | 0.85 | 0.04448 | 0.85 | 0.001068 | 0.002299 | 0.1 | 0.02744 | 78.51 | 0 | -325.5 | 155.2 |
| diurnal_long_sfold1_holdout | ORACLE_lookahead (not deployable) | false | 20 | 20 | 0 | 0 | 36.67 | 0.2395 | 0.8724 | 0.04657 | 0.85 | 0.0009524 | 0.002294 | 0.1 | 0.02527 | 91.41 | 0 | -313.8 | 156 |
| diurnal_long_sfold1_holdout | rule_seasonal_forecast (hand-built) | false | 20 | 20 | 0 | 0 | 40.37 | 0.2343 | 0.8501 | 0.04589 | 0.85 | 0.001067 | 0.002299 | 0.1 | 0.02815 | 83.32 | 0 | -322.1 | 155.4 |
| diurnal_long_sfold1_holdout | forecast_ppo_bc seed0 | false | 20 | 20 | 0 | 0 | 40.79 | 0.2361 | 0.8521 | 0.04814 | 0.85 | 0.001089 | 0.002321 | 0.1 | 0.02709 | 82.45 | 0 | -336.8 | 155.1 |
| diurnal_long_sfold1_holdout | forecast_ppo_bc seed1 | false | 20 | 20 | 0 | 0 | 40.91 | 0.2374 | 0.8508 | 0.04838 | 0.85 | 0.001084 | 0.002321 | 0.1 | 0.02777 | 82.27 | 0 | -334.9 | 156 |
| diurnal_long_sfold1_holdout | forecast_ppo_bc seed2 | false | 20 | 20 | 0 | 0 | 40.74 | 0.2408 | 0.8523 | 0.04914 | 0.85 | 0.001089 | 0.002304 | 0.1 | 0.02813 | 82.6 | 0 | -340.5 | 155.5 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Paired total-reward gap against risk_aware_rule_based, by stratum

`gap` is treatment minus rule, mean over paired scenarios; `sem` its standard error; `n_better` / `n_worse` count scenarios. Strata are defined in the module docstring.

| policy | stratum | n | gap | sem | n_better | n_worse |
|---|---|---|---|---|---|---|
| rule_seasonal_forecast (hand-built) | all | 20 | 3.406 | 0.2471 | 20 | 0 |
| rule_seasonal_forecast (hand-built) | initial_budget_infeasible=False (pre-registered) | 18 | 3.66 | 0.1949 | 18 | 0 |
| rule_seasonal_forecast (hand-built) | initial_budget_infeasible=True (pre-registered) | 2 | 1.124 | 0.03032 | 2 | 0 |
| rule_seasonal_forecast (hand-built) | typical, rule reward >= -500 (post hoc) | 17 | 3.753 | 0.1814 | 17 | 0 |
| rule_seasonal_forecast (hand-built) | extreme, rule reward < -500 (post hoc) | 3 | 1.44 | 0.3163 | 3 | 0 |
| forecast_ppo_bc seed0 | all | 20 | -11.36 | 2.769 | 4 | 16 |
| forecast_ppo_bc seed0 | initial_budget_infeasible=False (pre-registered) | 18 | -10.9 | 3.03 | 4 | 14 |
| forecast_ppo_bc seed0 | initial_budget_infeasible=True (pre-registered) | 2 | -15.53 | 5.705 | 0 | 2 |
| forecast_ppo_bc seed0 | typical, rule reward >= -500 (post hoc) | 17 | -11.61 | 3.124 | 3 | 14 |
| forecast_ppo_bc seed0 | extreme, rule reward < -500 (post hoc) | 3 | -9.953 | 6.477 | 1 | 2 |
| forecast_ppo_bc seed1 | all | 20 | -9.412 | 3.289 | 8 | 12 |
| forecast_ppo_bc seed1 | initial_budget_infeasible=False (pre-registered) | 18 | -7.556 | 3.346 | 8 | 10 |
| forecast_ppo_bc seed1 | initial_budget_infeasible=True (pre-registered) | 2 | -26.11 | 5.762 | 0 | 2 |
| forecast_ppo_bc seed1 | typical, rule reward >= -500 (post hoc) | 17 | -8.121 | 3.498 | 7 | 10 |
| forecast_ppo_bc seed1 | extreme, rule reward < -500 (post hoc) | 3 | -16.73 | 9.96 | 1 | 2 |
| forecast_ppo_bc seed2 | all | 20 | -15 | 3.024 | 5 | 15 |
| forecast_ppo_bc seed2 | initial_budget_infeasible=False (pre-registered) | 18 | -13.76 | 3.197 | 5 | 13 |
| forecast_ppo_bc seed2 | initial_budget_infeasible=True (pre-registered) | 2 | -26.2 | 5.849 | 0 | 2 |
| forecast_ppo_bc seed2 | typical, rule reward >= -500 (post hoc) | 17 | -14.66 | 3.253 | 4 | 13 |
| forecast_ppo_bc seed2 | extreme, rule reward < -500 (post hoc) | 3 | -16.95 | 9.847 | 1 | 2 |


## Paired difference: rule_seasonal_forecast (hand-built) minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_sfold1_holdout | total_reward | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 3.406 | 0.2471 | 20 |
| diurnal_long_sfold1_holdout | latency_degradation_mean | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 0.006661 | 0.004008 | 20 |
| diurnal_long_sfold1_holdout | latency_violation_fraction | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 0.001418 | 0.0007681 | 20 |
| diurnal_long_sfold1_holdout | completion_hours | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | -2.542 | 0.2205 | 20 |
| diurnal_long_sfold1_holdout | mean_delivered_mbps | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 4.802 | 0.3663 | 20 |
| diurnal_long_sfold1_holdout | p_failure_mean | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | -1.048e-06 | 7.14e-06 | 20 |
| diurnal_long_sfold1_holdout | clip_fraction | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 0.000711 | 0.0004922 | 20 |

## Paired difference: forecast_ppo_bc seed0 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_sfold1_holdout | total_reward | forecast_ppo_bc seed0 | risk_aware_rule_based | -11.36 | 2.769 | 20 |
| diurnal_long_sfold1_holdout | latency_degradation_mean | forecast_ppo_bc seed0 | risk_aware_rule_based | 0.008389 | 0.003284 | 20 |
| diurnal_long_sfold1_holdout | latency_violation_fraction | forecast_ppo_bc seed0 | risk_aware_rule_based | 0.003662 | 0.0007972 | 20 |
| diurnal_long_sfold1_holdout | completion_hours | forecast_ppo_bc seed0 | risk_aware_rule_based | -2.117 | 0.2286 | 20 |
| diurnal_long_sfold1_holdout | mean_delivered_mbps | forecast_ppo_bc seed0 | risk_aware_rule_based | 3.934 | 0.3828 | 20 |
| diurnal_long_sfold1_holdout | p_failure_mean | forecast_ppo_bc seed0 | risk_aware_rule_based | 2.134e-05 | 6.711e-06 | 20 |
| diurnal_long_sfold1_holdout | clip_fraction | forecast_ppo_bc seed0 | risk_aware_rule_based | -0.0003514 | 0.0007717 | 20 |

## Paired difference: forecast_ppo_bc seed1 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_sfold1_holdout | total_reward | forecast_ppo_bc seed1 | risk_aware_rule_based | -9.412 | 3.289 | 20 |
| diurnal_long_sfold1_holdout | latency_degradation_mean | forecast_ppo_bc seed1 | risk_aware_rule_based | 0.009718 | 0.004972 | 20 |
| diurnal_long_sfold1_holdout | latency_violation_fraction | forecast_ppo_bc seed1 | risk_aware_rule_based | 0.003899 | 0.001256 | 20 |
| diurnal_long_sfold1_holdout | completion_hours | forecast_ppo_bc seed1 | risk_aware_rule_based | -2 | 0.1956 | 20 |
| diurnal_long_sfold1_holdout | mean_delivered_mbps | forecast_ppo_bc seed1 | risk_aware_rule_based | 3.758 | 0.316 | 20 |
| diurnal_long_sfold1_holdout | p_failure_mean | forecast_ppo_bc seed1 | risk_aware_rule_based | 1.672e-05 | 1.074e-05 | 20 |
| diurnal_long_sfold1_holdout | clip_fraction | forecast_ppo_bc seed1 | risk_aware_rule_based | 0.0003313 | 0.0008193 | 20 |

## Paired difference: forecast_ppo_bc seed2 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| diurnal_long_sfold1_holdout | total_reward | forecast_ppo_bc seed2 | risk_aware_rule_based | -15 | 3.024 | 20 |
| diurnal_long_sfold1_holdout | latency_degradation_mean | forecast_ppo_bc seed2 | risk_aware_rule_based | 0.01315 | 0.004701 | 20 |
| diurnal_long_sfold1_holdout | latency_violation_fraction | forecast_ppo_bc seed2 | risk_aware_rule_based | 0.004667 | 0.001165 | 20 |
| diurnal_long_sfold1_holdout | completion_hours | forecast_ppo_bc seed2 | risk_aware_rule_based | -2.171 | 0.1943 | 20 |
| diurnal_long_sfold1_holdout | mean_delivered_mbps | forecast_ppo_bc seed2 | risk_aware_rule_based | 4.084 | 0.301 | 20 |
| diurnal_long_sfold1_holdout | p_failure_mean | forecast_ppo_bc seed2 | risk_aware_rule_based | 2.08e-05 | 7.476e-06 | 20 |
| diurnal_long_sfold1_holdout | clip_fraction | forecast_ppo_bc seed2 | risk_aware_rule_based | 0.0006944 | 0.0009784 | 20 |
