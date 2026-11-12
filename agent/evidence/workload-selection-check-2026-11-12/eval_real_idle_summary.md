# Learned controllers against the baselines (simulated)

Simulator configuration `v1,eval-fleet,sigma=0.25`; latency model uncalibrated (sim/LIMITATIONS.md). Workload `runs/selection_check/profile_idle_long_holdout.csv` (sha256 `58c352924cb88df66861b8b64bf1ddd385ead50fa3449e688d2e0d6160dbd454`). 20 paired scenario seeds (0..19), identical for every policy. Learned policies act deterministically. Reward is the contract reward with contract coefficients.

## Checkpoints

| checkpoint | policy | iteration | trained_on | hazard | train_fleet | sigma | val_gap | val_gap_sem | val_seeds | params |
|---|---|---|---|---|---|---|---|---|---|---|
| agent/evidence/forecast-ppo-2026-11-11/training/bc/seed0/ckpt_best.pt | forecast_ppo_bc seed0 | 40 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | 4.406 | 1.694 | 5000..5039 | bc_accuracy=0.959 |
| agent/evidence/forecast-ppo-2026-11-11/training/bc/seed1/ckpt_best.pt | forecast_ppo_bc seed1 | 80 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | 6.472 | 1.651 | 5000..5039 | bc_accuracy=0.963 |
| agent/evidence/forecast-ppo-2026-11-11/training/bc/seed2/ckpt_best.pt | forecast_ppo_bc seed2 | 100 | data_pipeline/profiles/profile_diurnal_long_train.csv | v1 | train-fleet | 0.25 | 6.782 | 1.602 | 5000..5039 | bc_accuracy=0.958 |

## Summary

| profile | policy | workload_synthetic | n | completed | horizon_expired | terminal_failure | completion_h | lat_deg_mean | lat_deg_p95 | viol_frac | lat_budget_met | p_fail_mean | p_fail_max | risk_viol | clip_frac | deliv_mbps | stall_frac | reward | reward_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| idle_long_holdout | fixed_20 | false | 20 | 20 | 0 | 0 | 144.7 | 0 | 0 | 0 | 1 | 0.001983 | 0.003874 | 0.1 | 0.1223 | 27.02 | 0 | -173.7 | 11.09 |
| idle_long_holdout | fixed_50 | false | 20 | 20 | 0 | 0 | 62.98 | 0 | 0 | 0 | 1 | 0.001228 | 0.002381 | 0.1 | 0.08104 | 54.31 | 0 | -75.58 | 2.326 |
| idle_long_holdout | fixed_80 | false | 20 | 20 | 0 | 0 | 40.7 | 0 | 0 | 0 | 1 | 0.0009759 | 0.001919 | 0.1 | 0.05672 | 82.27 | 0 | -48.84 | 0.6831 |
| idle_long_holdout | fixed_120 | false | 20 | 20 | 0 | 0 | 27.83 | 0 | 0 | 0 | 1 | 0.0008012 | 0.0016 | 0.1 | 0 | 120 | 0 | -33.4 | 0 |
| idle_long_holdout | workload_only_throttle | false | 20 | 20 | 0 | 0 | 27.83 | 0 | 0 | 0 | 1 | 0.0008012 | 0.0016 | 0.1 | 0 | 120 | 0 | -33.4 | 0 |
| idle_long_holdout | risk_aware_rule_based | false | 20 | 20 | 0 | 0 | 27.83 | 0 | 0 | 0 | 1 | 0.0008012 | 0.0016 | 0.1 | 0 | 120 | 0 | -33.4 | 0 |
| idle_long_holdout | ORACLE_lookahead (not deployable) | false | 20 | 20 | 0 | 0 | 27.83 | 0 | 0 | 0 | 1 | 0.0008012 | 0.0016 | 0.1 | 0 | 120 | 0 | -33.4 | 0 |
| idle_long_holdout | rule_seasonal_forecast (hand-built) | false | 20 | 20 | 0 | 0 | 27.83 | 0 | 0 | 0 | 1 | 0.0008012 | 0.0016 | 0.1 | 0 | 120 | 0 | -33.4 | 0 |
| idle_long_holdout | forecast_ppo_bc seed0 | false | 20 | 20 | 0 | 0 | 27.83 | 0 | 0 | 0 | 1 | 0.0008012 | 0.0016 | 0.1 | 0 | 120 | 0 | -33.4 | 0 |
| idle_long_holdout | forecast_ppo_bc seed1 | false | 20 | 20 | 0 | 0 | 27.83 | 0 | 0 | 0 | 1 | 0.0008012 | 0.0016 | 0.1 | 0 | 120 | 0 | -33.4 | 0 |
| idle_long_holdout | forecast_ppo_bc seed2 | false | 20 | 20 | 0 | 0 | 27.83 | 0 | 0 | 0 | 1 | 0.0008012 | 0.0016 | 0.1 | 0 | 120 | 0 | -33.4 | 0 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Paired total-reward gap against risk_aware_rule_based, by stratum

`gap` is treatment minus rule, mean over paired scenarios; `sem` its standard error; `n_better` / `n_worse` count scenarios. Strata are defined in the module docstring.

| policy | stratum | n | gap | sem | n_better | n_worse |
|---|---|---|---|---|---|---|
| rule_seasonal_forecast (hand-built) | all | 20 | 0 | 0 | 0 | 0 |
| rule_seasonal_forecast (hand-built) | initial_budget_infeasible=False (pre-registered) | 18 | 0 | 0 | 0 | 0 |
| rule_seasonal_forecast (hand-built) | initial_budget_infeasible=True (pre-registered) | 2 | 0 | 0 | 0 | 0 |
| rule_seasonal_forecast (hand-built) | typical, rule reward >= -500 (post hoc) | 20 | 0 | 0 | 0 | 0 |
| forecast_ppo_bc seed0 | all | 20 | 0 | 0 | 0 | 0 |
| forecast_ppo_bc seed0 | initial_budget_infeasible=False (pre-registered) | 18 | 0 | 0 | 0 | 0 |
| forecast_ppo_bc seed0 | initial_budget_infeasible=True (pre-registered) | 2 | 0 | 0 | 0 | 0 |
| forecast_ppo_bc seed0 | typical, rule reward >= -500 (post hoc) | 20 | 0 | 0 | 0 | 0 |
| forecast_ppo_bc seed1 | all | 20 | 0 | 0 | 0 | 0 |
| forecast_ppo_bc seed1 | initial_budget_infeasible=False (pre-registered) | 18 | 0 | 0 | 0 | 0 |
| forecast_ppo_bc seed1 | initial_budget_infeasible=True (pre-registered) | 2 | 0 | 0 | 0 | 0 |
| forecast_ppo_bc seed1 | typical, rule reward >= -500 (post hoc) | 20 | 0 | 0 | 0 | 0 |
| forecast_ppo_bc seed2 | all | 20 | 0 | 0 | 0 | 0 |
| forecast_ppo_bc seed2 | initial_budget_infeasible=False (pre-registered) | 18 | 0 | 0 | 0 | 0 |
| forecast_ppo_bc seed2 | initial_budget_infeasible=True (pre-registered) | 2 | 0 | 0 | 0 | 0 |
| forecast_ppo_bc seed2 | typical, rule reward >= -500 (post hoc) | 20 | 0 | 0 | 0 | 0 |


## Paired difference: rule_seasonal_forecast (hand-built) minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| idle_long_holdout | total_reward | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 0 | 0 | 20 |
| idle_long_holdout | latency_degradation_mean | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 0 | 0 | 20 |
| idle_long_holdout | latency_violation_fraction | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 0 | 0 | 20 |
| idle_long_holdout | completion_hours | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 0 | 0 | 20 |
| idle_long_holdout | mean_delivered_mbps | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 0 | 0 | 20 |
| idle_long_holdout | p_failure_mean | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 0 | 0 | 20 |
| idle_long_holdout | clip_fraction | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | 0 | 0 | 20 |

## Paired difference: forecast_ppo_bc seed0 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| idle_long_holdout | total_reward | forecast_ppo_bc seed0 | risk_aware_rule_based | 0 | 0 | 20 |
| idle_long_holdout | latency_degradation_mean | forecast_ppo_bc seed0 | risk_aware_rule_based | 0 | 0 | 20 |
| idle_long_holdout | latency_violation_fraction | forecast_ppo_bc seed0 | risk_aware_rule_based | 0 | 0 | 20 |
| idle_long_holdout | completion_hours | forecast_ppo_bc seed0 | risk_aware_rule_based | 0 | 0 | 20 |
| idle_long_holdout | mean_delivered_mbps | forecast_ppo_bc seed0 | risk_aware_rule_based | 0 | 0 | 20 |
| idle_long_holdout | p_failure_mean | forecast_ppo_bc seed0 | risk_aware_rule_based | 0 | 0 | 20 |
| idle_long_holdout | clip_fraction | forecast_ppo_bc seed0 | risk_aware_rule_based | 0 | 0 | 20 |

## Paired difference: forecast_ppo_bc seed1 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| idle_long_holdout | total_reward | forecast_ppo_bc seed1 | risk_aware_rule_based | 0 | 0 | 20 |
| idle_long_holdout | latency_degradation_mean | forecast_ppo_bc seed1 | risk_aware_rule_based | 0 | 0 | 20 |
| idle_long_holdout | latency_violation_fraction | forecast_ppo_bc seed1 | risk_aware_rule_based | 0 | 0 | 20 |
| idle_long_holdout | completion_hours | forecast_ppo_bc seed1 | risk_aware_rule_based | 0 | 0 | 20 |
| idle_long_holdout | mean_delivered_mbps | forecast_ppo_bc seed1 | risk_aware_rule_based | 0 | 0 | 20 |
| idle_long_holdout | p_failure_mean | forecast_ppo_bc seed1 | risk_aware_rule_based | 0 | 0 | 20 |
| idle_long_holdout | clip_fraction | forecast_ppo_bc seed1 | risk_aware_rule_based | 0 | 0 | 20 |

## Paired difference: forecast_ppo_bc seed2 minus risk_aware_rule_based

| profile | metric | treatment | reference | mean_diff | sem | n |
|---|---|---|---|---|---|---|
| idle_long_holdout | total_reward | forecast_ppo_bc seed2 | risk_aware_rule_based | 0 | 0 | 20 |
| idle_long_holdout | latency_degradation_mean | forecast_ppo_bc seed2 | risk_aware_rule_based | 0 | 0 | 20 |
| idle_long_holdout | latency_violation_fraction | forecast_ppo_bc seed2 | risk_aware_rule_based | 0 | 0 | 20 |
| idle_long_holdout | completion_hours | forecast_ppo_bc seed2 | risk_aware_rule_based | 0 | 0 | 20 |
| idle_long_holdout | mean_delivered_mbps | forecast_ppo_bc seed2 | risk_aware_rule_based | 0 | 0 | 20 |
| idle_long_holdout | p_failure_mean | forecast_ppo_bc seed2 | risk_aware_rule_based | 0 | 0 | 20 |
| idle_long_holdout | clip_fraction | forecast_ppo_bc seed2 | risk_aware_rule_based | 0 | 0 | 20 |
