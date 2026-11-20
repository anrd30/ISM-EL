# Vendor-holdout evaluation, `vendor_holdout,sigma=0.25`, history_rows=576

Scenarios are identical for every arm. Gains are paired by scenario seed; `sem` is across scenarios. `primary` is the registered definition of controllable (the safety filter raises a `fixed_20` rollout on fewer than 10% of the intervals).

| profile | definition | n_forced | n | forced_seeds |
|---|---|---|---|---|
| diurnal_long_holdout | primary | 0 | 20 |  |
| diurnal_long_holdout | clip_0.01 | 0 | 20 |  |
| diurnal_long_holdout | clip_0.25 | 0 | 20 |  |
| diurnal_long_holdout | fixed20_cost | 0 | 20 |  |

## Summary, all scenarios

| profile | policy | workload_synthetic | n | completed | horizon_expired | terminal_failure | completion_h | lat_deg_mean | lat_deg_p95 | viol_frac | lat_budget_met | p_fail_mean | p_fail_max | risk_viol | clip_frac | deliv_mbps | stall_frac | reward | reward_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_holdout | fixed_20 | false | 20 | 20 | 0 | 0 | 166.7 | 0 | 0 | 0 | 1 | 0.000884 | 0.001767 | 0 | 0 | 20 | 0 | -200 | 6.52e-15 |
| diurnal_long_holdout | fixed_50 | false | 20 | 20 | 0 | 0 | 66.67 | 0 | 0 | 0 | 1 | 0.000354 | 0.000707 | 0 | 0 | 50 | 0 | -80 | 3.26e-15 |
| diurnal_long_holdout | fixed_80 | false | 20 | 20 | 0 | 0 | 41.72 | 0.05257 | 0.32 | 0.03125 | 1 | 0.0002211 | 0.0004419 | 0 | 0 | 80 | 0 | -232.9 | 6.751 |
| diurnal_long_holdout | fixed_120 | false | 20 | 20 | 0 | 0 | 29.26 | 3.483 | 5.667 | 0.6722 | 0 | 0.0001478 | 0.0002946 | 0 | 0 | 114.1 | 0 | -3621 | 90.47 |
| diurnal_long_holdout | risk_aware_rule_based | false | 20 | 20 | 0 | 0 | 44.77 | 0.006356 | 0 | 0.0008451 | 1 | 0.0002632 | 0.0006412 | 0 | 0 | 74.58 | 0 | -61.63 | 1.233 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | false | 20 | 20 | 0 | 0 | 41.55 | 0.0066 | 0.004419 | 0.0009088 | 1 | 0.0002636 | 0.0006412 | 0 | 0 | 80.37 | 0 | -57.64 | 1.272 |
| diurnal_long_holdout | rule+history | false | 20 | 20 | 0 | 0 | 45.24 | 0.004408 | 0 | 0 | 1 | 0.0002713 | 0.0006547 | 0 | 0 | 73.79 | 0 | -56.68 | 0.2382 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | false | 20 | 20 | 0 | 0 | 37.73 | 0.003838 | 0.0161 | 0 | 1 | 0.0002126 | 0.0006386 | 0 | 0 | 88.55 | 0 | -47.01 | 0.3651 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | false | 20 | 20 | 0 | 0 | 42.3 | 0.006091 | 0.0005466 | 0.0002989 | 1 | 0.0002742 | 0.001218 | 0 | 0 | 78.96 | 0 | -55.34 | 0.9151 |
| diurnal_long_holdout | predictive_controller (history) | false | 20 | 20 | 0 | 0 | 37.73 | 0.003838 | 0.0161 | 0 | 1 | 0.0002126 | 0.0006386 | 0 | 0 | 88.55 | 0 | -47.01 | 0.3651 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | false | 20 | 20 | 0 | 0 | 37.45 | 0.005082 | 0.03826 | 0 | 1 | 0.0002109 | 0.0006383 | 0 | 0 | 89.23 | 0 | -47.22 | 0.3666 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | false | 20 | 20 | 0 | 0 | 38.17 | 0.0006966 | 0.000131 | 0 | 1 | 0.0002165 | 0.0006408 | 0 | 0 | 87.43 | 0 | -46.12 | 0.3466 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Paired gains by block

| profile | treatment | reference | block | n | mean_gain | sem | t | better | worse | percent_of_optimum | viol_frac_diff | viol_frac_diff_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | all | 20 | 3.987 | 0.1791 | 22.26 | 20 | 0 | 25.71 | 6.371e-05 | 1.678e-05 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | primary | 20 | 3.987 | 0.1791 | 22.26 | 20 | 0 | 25.71 | 6.371e-05 | 1.678e-05 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | clip_0.01 | 20 | 3.987 | 0.1791 | 22.26 | 20 | 0 | 25.71 | 6.371e-05 | 1.678e-05 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | clip_0.25 | 20 | 3.987 | 0.1791 | 22.26 | 20 | 0 | 25.71 | 6.371e-05 | 1.678e-05 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | fixed20_cost | 20 | 3.987 | 0.1791 | 22.26 | 20 | 0 | 25.71 | 6.371e-05 | 1.678e-05 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | all | 20 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | primary | 20 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | clip_0.01 | 20 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | clip_0.25 | 20 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | fixed20_cost | 20 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_holdout | rule+history | risk_aware_rule_based | all | 20 | 4.952 | 1.264 | 3.917 | 14 | 4 | 31.93 | -0.0008451 | 0.0002145 |
| diurnal_long_holdout | rule+history | risk_aware_rule_based | primary | 20 | 4.952 | 1.264 | 3.917 | 14 | 4 | 31.93 | -0.0008451 | 0.0002145 |
| diurnal_long_holdout | rule+history | risk_aware_rule_based | clip_0.01 | 20 | 4.952 | 1.264 | 3.917 | 14 | 4 | 31.93 | -0.0008451 | 0.0002145 |
| diurnal_long_holdout | rule+history | risk_aware_rule_based | clip_0.25 | 20 | 4.952 | 1.264 | 3.917 | 14 | 4 | 31.93 | -0.0008451 | 0.0002145 |
| diurnal_long_holdout | rule+history | risk_aware_rule_based | fixed20_cost | 20 | 4.952 | 1.264 | 3.917 | 14 | 4 | 31.93 | -0.0008451 | 0.0002145 |
| diurnal_long_holdout | rule+history | rule_seasonal_forecast (hand-built) | all | 20 | 0.9644 | 1.314 | 0.7338 | 9 | 11 | 8.372 | -0.0009088 | 0.0002307 |
| diurnal_long_holdout | rule+history | rule_seasonal_forecast (hand-built) | primary | 20 | 0.9644 | 1.314 | 0.7338 | 9 | 11 | 8.372 | -0.0009088 | 0.0002307 |
| diurnal_long_holdout | rule+history | rule_seasonal_forecast (hand-built) | clip_0.01 | 20 | 0.9644 | 1.314 | 0.7338 | 9 | 11 | 8.372 | -0.0009088 | 0.0002307 |
| diurnal_long_holdout | rule+history | rule_seasonal_forecast (hand-built) | clip_0.25 | 20 | 0.9644 | 1.314 | 0.7338 | 9 | 11 | 8.372 | -0.0009088 | 0.0002307 |
| diurnal_long_holdout | rule+history | rule_seasonal_forecast (hand-built) | fixed20_cost | 20 | 0.9644 | 1.314 | 0.7338 | 9 | 11 | 8.372 | -0.0009088 | 0.0002307 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | risk_aware_rule_based | all | 20 | 14.62 | 1.259 | 11.61 | 20 | 0 | 94.28 | -0.0008451 | 0.0002145 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | risk_aware_rule_based | primary | 20 | 14.62 | 1.259 | 11.61 | 20 | 0 | 94.28 | -0.0008451 | 0.0002145 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | risk_aware_rule_based | clip_0.01 | 20 | 14.62 | 1.259 | 11.61 | 20 | 0 | 94.28 | -0.0008451 | 0.0002145 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | risk_aware_rule_based | clip_0.25 | 20 | 14.62 | 1.259 | 11.61 | 20 | 0 | 94.28 | -0.0008451 | 0.0002145 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | risk_aware_rule_based | fixed20_cost | 20 | 14.62 | 1.259 | 11.61 | 20 | 0 | 94.28 | -0.0008451 | 0.0002145 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | all | 20 | 10.63 | 1.268 | 8.382 | 20 | 0 | 92.3 | -0.0009088 | 0.0002307 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | primary | 20 | 10.63 | 1.268 | 8.382 | 20 | 0 | 92.3 | -0.0009088 | 0.0002307 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | clip_0.01 | 20 | 10.63 | 1.268 | 8.382 | 20 | 0 | 92.3 | -0.0009088 | 0.0002307 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | clip_0.25 | 20 | 10.63 | 1.268 | 8.382 | 20 | 0 | 92.3 | -0.0009088 | 0.0002307 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | fixed20_cost | 20 | 10.63 | 1.268 | 8.382 | 20 | 0 | 92.3 | -0.0009088 | 0.0002307 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | risk_aware_rule_based | all | 20 | 6.291 | 1.468 | 4.285 | 19 | 1 | 40.57 | -0.0005461 | 0.0002402 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | risk_aware_rule_based | primary | 20 | 6.291 | 1.468 | 4.285 | 19 | 1 | 40.57 | -0.0005461 | 0.0002402 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | risk_aware_rule_based | clip_0.01 | 20 | 6.291 | 1.468 | 4.285 | 19 | 1 | 40.57 | -0.0005461 | 0.0002402 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | risk_aware_rule_based | clip_0.25 | 20 | 6.291 | 1.468 | 4.285 | 19 | 1 | 40.57 | -0.0005461 | 0.0002402 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | risk_aware_rule_based | fixed20_cost | 20 | 6.291 | 1.468 | 4.285 | 19 | 1 | 40.57 | -0.0005461 | 0.0002402 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | all | 20 | 2.303 | 1.445 | 1.594 | 7 | 13 | 19.99 | -0.0006099 | 0.0002538 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | primary | 20 | 2.303 | 1.445 | 1.594 | 7 | 13 | 19.99 | -0.0006099 | 0.0002538 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | clip_0.01 | 20 | 2.303 | 1.445 | 1.594 | 7 | 13 | 19.99 | -0.0006099 | 0.0002538 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | clip_0.25 | 20 | 2.303 | 1.445 | 1.594 | 7 | 13 | 19.99 | -0.0006099 | 0.0002538 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | fixed20_cost | 20 | 2.303 | 1.445 | 1.594 | 7 | 13 | 19.99 | -0.0006099 | 0.0002538 |
| diurnal_long_holdout | predictive_controller (history) | risk_aware_rule_based | all | 20 | 14.62 | 1.259 | 11.61 | 20 | 0 | 94.28 | -0.0008451 | 0.0002145 |
| diurnal_long_holdout | predictive_controller (history) | risk_aware_rule_based | primary | 20 | 14.62 | 1.259 | 11.61 | 20 | 0 | 94.28 | -0.0008451 | 0.0002145 |
| diurnal_long_holdout | predictive_controller (history) | risk_aware_rule_based | clip_0.01 | 20 | 14.62 | 1.259 | 11.61 | 20 | 0 | 94.28 | -0.0008451 | 0.0002145 |
| diurnal_long_holdout | predictive_controller (history) | risk_aware_rule_based | clip_0.25 | 20 | 14.62 | 1.259 | 11.61 | 20 | 0 | 94.28 | -0.0008451 | 0.0002145 |
| diurnal_long_holdout | predictive_controller (history) | risk_aware_rule_based | fixed20_cost | 20 | 14.62 | 1.259 | 11.61 | 20 | 0 | 94.28 | -0.0008451 | 0.0002145 |
| diurnal_long_holdout | predictive_controller (history) | rule_seasonal_forecast (hand-built) | all | 20 | 10.63 | 1.268 | 8.382 | 20 | 0 | 92.3 | -0.0009088 | 0.0002307 |
| diurnal_long_holdout | predictive_controller (history) | rule_seasonal_forecast (hand-built) | primary | 20 | 10.63 | 1.268 | 8.382 | 20 | 0 | 92.3 | -0.0009088 | 0.0002307 |
| diurnal_long_holdout | predictive_controller (history) | rule_seasonal_forecast (hand-built) | clip_0.01 | 20 | 10.63 | 1.268 | 8.382 | 20 | 0 | 92.3 | -0.0009088 | 0.0002307 |
| diurnal_long_holdout | predictive_controller (history) | rule_seasonal_forecast (hand-built) | clip_0.25 | 20 | 10.63 | 1.268 | 8.382 | 20 | 0 | 92.3 | -0.0009088 | 0.0002307 |
| diurnal_long_holdout | predictive_controller (history) | rule_seasonal_forecast (hand-built) | fixed20_cost | 20 | 10.63 | 1.268 | 8.382 | 20 | 0 | 92.3 | -0.0009088 | 0.0002307 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | all | 20 | 14.41 | 1.26 | 11.44 | 20 | 0 | 92.91 | -0.0008451 | 0.0002145 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | primary | 20 | 14.41 | 1.26 | 11.44 | 20 | 0 | 92.91 | -0.0008451 | 0.0002145 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | clip_0.01 | 20 | 14.41 | 1.26 | 11.44 | 20 | 0 | 92.91 | -0.0008451 | 0.0002145 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | clip_0.25 | 20 | 14.41 | 1.26 | 11.44 | 20 | 0 | 92.91 | -0.0008451 | 0.0002145 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | fixed20_cost | 20 | 14.41 | 1.26 | 11.44 | 20 | 0 | 92.91 | -0.0008451 | 0.0002145 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | all | 20 | 10.42 | 1.268 | 8.216 | 20 | 0 | 90.46 | -0.0009088 | 0.0002307 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | primary | 20 | 10.42 | 1.268 | 8.216 | 20 | 0 | 90.46 | -0.0009088 | 0.0002307 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.01 | 20 | 10.42 | 1.268 | 8.216 | 20 | 0 | 90.46 | -0.0009088 | 0.0002307 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.25 | 20 | 10.42 | 1.268 | 8.216 | 20 | 0 | 90.46 | -0.0009088 | 0.0002307 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | fixed20_cost | 20 | 10.42 | 1.268 | 8.216 | 20 | 0 | 90.46 | -0.0009088 | 0.0002307 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | all | 20 | 15.51 | 1.257 | 12.34 | 20 | 0 | 100 | -0.0008451 | 0.0002145 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | primary | 20 | 15.51 | 1.257 | 12.34 | 20 | 0 | 100 | -0.0008451 | 0.0002145 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | clip_0.01 | 20 | 15.51 | 1.257 | 12.34 | 20 | 0 | 100 | -0.0008451 | 0.0002145 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | clip_0.25 | 20 | 15.51 | 1.257 | 12.34 | 20 | 0 | 100 | -0.0008451 | 0.0002145 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | fixed20_cost | 20 | 15.51 | 1.257 | 12.34 | 20 | 0 | 100 | -0.0008451 | 0.0002145 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | all | 20 | 11.52 | 1.273 | 9.052 | 20 | 0 | 100 | -0.0009088 | 0.0002307 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | primary | 20 | 11.52 | 1.273 | 9.052 | 20 | 0 | 100 | -0.0009088 | 0.0002307 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.01 | 20 | 11.52 | 1.273 | 9.052 | 20 | 0 | 100 | -0.0009088 | 0.0002307 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.25 | 20 | 11.52 | 1.273 | 9.052 | 20 | 0 | 100 | -0.0009088 | 0.0002307 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | fixed20_cost | 20 | 11.52 | 1.273 | 9.052 | 20 | 0 | 100 | -0.0009088 | 0.0002307 |
