# Predictive controller and comparators, `eval-fleet,sigma=0.25`, history_rows=576

Scenarios are identical for every arm. Gains are paired by scenario seed; learned models with several seeds are averaged within each scenario first; `sem` is across scenarios. `primary` is the registered definition of controllable (the safety filter raises a `fixed_20` rollout on fewer than 10% of the intervals); the other blocks are the registered sensitivity variants.

| profile | definition | n_forced | n | forced_seeds |
|---|---|---|---|---|
| diurnal_long_holdout | primary | 4 | 20 | 3 9 17 19 |
| diurnal_long_holdout | clip_0.01 | 7 | 20 | 2 3 9 16 17 18 19 |
| diurnal_long_holdout | clip_0.25 | 4 | 20 | 3 9 17 19 |
| diurnal_long_holdout | fixed20_cost | 3 | 20 | 9 17 19 |

## Summary, all scenarios

| profile | policy | workload_synthetic | n | completed | horizon_expired | terminal_failure | completion_h | lat_deg_mean | lat_deg_p95 | viol_frac | lat_budget_met | p_fail_mean | p_fail_max | risk_viol | clip_frac | deliv_mbps | stall_frac | reward | reward_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_holdout | fixed_20 | false | 20 | 20 | 0 | 0 | 144.8 | 0.1434 | 0.85 | 0.02806 | 0.85 | 0.001984 | 0.003874 | 0.1 | 0.1231 | 26.89 | 0 | -449.3 | 151.2 |
| diurnal_long_holdout | fixed_50 | false | 20 | 20 | 0 | 0 | 63.1 | 0.1817 | 0.85 | 0.03561 | 0.85 | 0.00123 | 0.002381 | 0.1 | 0.08196 | 54.1 | 0 | -351.2 | 158.7 |
| diurnal_long_holdout | fixed_80 | false | 20 | 20 | 0 | 0 | 40.85 | 0.2664 | 1.12 | 0.07125 | 0.85 | 0.0009774 | 0.001919 | 0.1 | 0.0583 | 81.97 | 0 | -488.5 | 150.3 |
| diurnal_long_holdout | fixed_120 | false | 20 | 20 | 0 | 0 | 29.26 | 3.483 | 5.667 | 0.6722 | 0 | 0.0008027 | 0.0016 | 0.1 | 0 | 114.1 | 0 | -3621 | 90.47 |
| diurnal_long_holdout | risk_aware_rule_based | false | 20 | 20 | 0 | 0 | 43.35 | 0.2333 | 0.85 | 0.04528 | 0.85 | 0.001075 | 0.002298 | 0.1 | 0.02914 | 77.73 | 0 | -334 | 159.3 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | false | 20 | 20 | 0 | 0 | 40.41 | 0.2417 | 0.854 | 0.04694 | 0.85 | 0.001075 | 0.002298 | 0.1 | 0.03006 | 83.19 | 0 | -330.4 | 159.5 |
| diurnal_long_holdout | rule+history | false | 20 | 20 | 0 | 0 | 43.71 | 0.2316 | 0.85 | 0.04454 | 0.85 | 0.001087 | 0.002302 | 0.1 | 0.02977 | 77.11 | 0 | -329.6 | 159.7 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | false | 20 | 20 | 0 | 0 | 37.05 | 0.2449 | 0.8636 | 0.04725 | 0.85 | 0.0009593 | 0.002291 | 0.1 | 0.02572 | 90.46 | 0 | -321 | 160.3 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | false | 20 | 20 | 0 | 0 | 41.02 | 0.2391 | 0.8509 | 0.04599 | 0.85 | 0.001087 | 0.003056 | 0.1 | 0.02904 | 81.97 | 0 | -328.5 | 159.7 |
| diurnal_long_holdout | predictive_controller (history) | false | 20 | 20 | 0 | 0 | 37.05 | 0.2449 | 0.8636 | 0.04725 | 0.85 | 0.0009593 | 0.002291 | 0.1 | 0.02572 | 90.46 | 0 | -321 | 160.3 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | false | 20 | 20 | 0 | 0 | 36.79 | 0.2467 | 0.8828 | 0.04736 | 0.85 | 0.0009544 | 0.00229 | 0.1 | 0.02576 | 91.1 | 0 | -321.2 | 160.3 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | false | 20 | 20 | 0 | 0 | 37.44 | 0.2405 | 0.8501 | 0.04694 | 0.85 | 0.0009675 | 0.002296 | 0.1 | 0.099 | 89.44 | 0 | -320.2 | 160.3 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Paired gains by block

| profile | treatment | reference | block | n | mean_gain | sem | t | better | worse | percent_of_optimum | viol_frac_diff | viol_frac_diff_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | all | 20 | 3.616 | 0.2694 | 13.42 | 20 | 0 | 26.15 | 0.001654 | 0.0009068 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | primary | 16 | 3.991 | 0.2026 | 19.69 | 16 | 0 | 24.88 | 7.231e-05 | 1.944e-05 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | clip_0.01 | 13 | 4.083 | 0.2387 | 17.1 | 13 | 0 | 25.84 | 6.928e-05 | 2.252e-05 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | clip_0.25 | 16 | 3.991 | 0.2026 | 19.69 | 16 | 0 | 24.88 | 7.231e-05 | 1.944e-05 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | fixed20_cost | 17 | 4.005 | 0.1909 | 20.98 | 17 | 0 | 25.48 | 7.66e-05 | 1.876e-05 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | all | 20 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | primary | 16 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_holdout | rule+history | risk_aware_rule_based | all | 20 | 4.441 | 1.253 | 3.545 | 13 | 3 | 32.11 | -0.0007473 | 0.0002101 |
| diurnal_long_holdout | rule+history | risk_aware_rule_based | primary | 16 | 5.546 | 1.441 | 3.848 | 12 | 3 | 34.58 | -0.0009343 | 0.0002414 |
| diurnal_long_holdout | rule+history | risk_aware_rule_based | clip_0.01 | 13 | 5.036 | 1.594 | 3.159 | 9 | 3 | 31.88 | -0.0008673 | 0.0002706 |
| diurnal_long_holdout | rule+history | risk_aware_rule_based | clip_0.25 | 16 | 5.546 | 1.441 | 3.848 | 12 | 3 | 34.58 | -0.0009343 | 0.0002414 |
| diurnal_long_holdout | rule+history | risk_aware_rule_based | fixed20_cost | 17 | 5.225 | 1.392 | 3.755 | 13 | 3 | 33.24 | -0.0008791 | 0.0002333 |
| diurnal_long_holdout | rule+history | rule_seasonal_forecast (hand-built) | all | 20 | 0.8255 | 1.223 | 0.6748 | 8 | 12 | 8.083 | -0.002402 | 0.0008647 |
| diurnal_long_holdout | rule+history | rule_seasonal_forecast (hand-built) | primary | 16 | 1.556 | 1.472 | 1.057 | 8 | 8 | 12.91 | -0.001007 | 0.0002601 |
| diurnal_long_holdout | rule+history | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 0.9532 | 1.614 | 0.5904 | 6 | 7 | 8.136 | -0.0009365 | 0.0002922 |
| diurnal_long_holdout | rule+history | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 1.556 | 1.472 | 1.057 | 8 | 8 | 12.91 | -0.001007 | 0.0002601 |
| diurnal_long_holdout | rule+history | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | 1.22 | 1.423 | 0.8573 | 8 | 9 | 10.41 | -0.0009557 | 0.0002496 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | risk_aware_rule_based | all | 20 | 13.01 | 1.576 | 8.256 | 20 | 0 | 94.09 | 0.001962 | 0.001637 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | risk_aware_rule_based | primary | 16 | 15.15 | 1.498 | 10.11 | 16 | 0 | 94.43 | -0.0009343 | 0.0002414 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | risk_aware_rule_based | clip_0.01 | 13 | 14.88 | 1.672 | 8.902 | 13 | 0 | 94.21 | -0.0008673 | 0.0002706 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | risk_aware_rule_based | clip_0.25 | 16 | 15.15 | 1.498 | 10.11 | 16 | 0 | 94.43 | -0.0009343 | 0.0002414 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | risk_aware_rule_based | fixed20_cost | 17 | 14.83 | 1.443 | 10.28 | 17 | 0 | 94.34 | -0.000858 | 0.0002392 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | all | 20 | 9.396 | 1.442 | 6.516 | 20 | 0 | 92 | 0.0003075 | 0.0007518 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | primary | 16 | 11.16 | 1.483 | 7.525 | 16 | 0 | 92.58 | -0.001007 | 0.0002601 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 10.8 | 1.633 | 6.614 | 13 | 0 | 92.19 | -0.0009365 | 0.0002922 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 11.16 | 1.483 | 7.525 | 16 | 0 | 92.58 | -0.001007 | 0.0002601 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | 10.82 | 1.432 | 7.558 | 17 | 0 | 92.41 | -0.0009346 | 0.0002547 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | risk_aware_rule_based | all | 20 | 5.546 | 1.645 | 3.371 | 18 | 2 | 40.11 | 0.0007049 | 0.0008093 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | risk_aware_rule_based | primary | 16 | 7.168 | 1.777 | 4.034 | 15 | 1 | 44.68 | -0.0006871 | 0.000291 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | risk_aware_rule_based | clip_0.01 | 13 | 7.175 | 1.62 | 4.428 | 13 | 0 | 45.42 | -0.0007119 | 0.0002631 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | risk_aware_rule_based | clip_0.25 | 16 | 7.168 | 1.777 | 4.034 | 15 | 1 | 44.68 | -0.0006871 | 0.000291 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | risk_aware_rule_based | fixed20_cost | 17 | 6.295 | 1.883 | 3.343 | 15 | 2 | 40.05 | -0.0005244 | 0.0003181 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | all | 20 | 1.931 | 1.592 | 1.213 | 8 | 12 | 18.91 | -0.0009495 | 0.0003628 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | primary | 16 | 3.177 | 1.749 | 1.817 | 7 | 9 | 26.36 | -0.0007594 | 0.0003074 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 3.093 | 1.56 | 1.982 | 5 | 8 | 26.4 | -0.0007811 | 0.0002831 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 3.177 | 1.749 | 1.817 | 7 | 9 | 26.36 | -0.0007594 | 0.0003074 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | 2.29 | 1.867 | 1.227 | 7 | 10 | 19.55 | -0.000601 | 0.0003294 |
| diurnal_long_holdout | predictive_controller (history) | risk_aware_rule_based | all | 20 | 13.01 | 1.576 | 8.256 | 20 | 0 | 94.09 | 0.001962 | 0.001637 |
| diurnal_long_holdout | predictive_controller (history) | risk_aware_rule_based | primary | 16 | 15.15 | 1.498 | 10.11 | 16 | 0 | 94.43 | -0.0009343 | 0.0002414 |
| diurnal_long_holdout | predictive_controller (history) | risk_aware_rule_based | clip_0.01 | 13 | 14.88 | 1.672 | 8.902 | 13 | 0 | 94.21 | -0.0008673 | 0.0002706 |
| diurnal_long_holdout | predictive_controller (history) | risk_aware_rule_based | clip_0.25 | 16 | 15.15 | 1.498 | 10.11 | 16 | 0 | 94.43 | -0.0009343 | 0.0002414 |
| diurnal_long_holdout | predictive_controller (history) | risk_aware_rule_based | fixed20_cost | 17 | 14.83 | 1.443 | 10.28 | 17 | 0 | 94.34 | -0.000858 | 0.0002392 |
| diurnal_long_holdout | predictive_controller (history) | rule_seasonal_forecast (hand-built) | all | 20 | 9.396 | 1.442 | 6.516 | 20 | 0 | 92 | 0.0003075 | 0.0007518 |
| diurnal_long_holdout | predictive_controller (history) | rule_seasonal_forecast (hand-built) | primary | 16 | 11.16 | 1.483 | 7.525 | 16 | 0 | 92.58 | -0.001007 | 0.0002601 |
| diurnal_long_holdout | predictive_controller (history) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 10.8 | 1.633 | 6.614 | 13 | 0 | 92.19 | -0.0009365 | 0.0002922 |
| diurnal_long_holdout | predictive_controller (history) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 11.16 | 1.483 | 7.525 | 16 | 0 | 92.58 | -0.001007 | 0.0002601 |
| diurnal_long_holdout | predictive_controller (history) | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | 10.82 | 1.432 | 7.558 | 17 | 0 | 92.41 | -0.0009346 | 0.0002547 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | all | 20 | 12.81 | 1.574 | 8.139 | 20 | 0 | 92.61 | 0.002077 | 0.001711 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | primary | 16 | 14.93 | 1.5 | 9.954 | 16 | 0 | 93.08 | -0.0009343 | 0.0002414 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | clip_0.01 | 13 | 14.68 | 1.679 | 8.745 | 13 | 0 | 92.93 | -0.0008673 | 0.0002706 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | clip_0.25 | 16 | 14.93 | 1.5 | 9.954 | 16 | 0 | 93.08 | -0.0009343 | 0.0002414 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | fixed20_cost | 17 | 14.62 | 1.444 | 10.12 | 17 | 0 | 92.99 | -0.0008569 | 0.0002396 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | all | 20 | 9.191 | 1.44 | 6.384 | 20 | 0 | 90 | 0.000423 | 0.0008249 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | primary | 16 | 10.94 | 1.484 | 7.373 | 16 | 0 | 90.79 | -0.001007 | 0.0002601 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 10.6 | 1.64 | 6.463 | 13 | 0 | 90.47 | -0.0009365 | 0.0002922 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 10.94 | 1.484 | 7.373 | 16 | 0 | 90.79 | -0.001007 | 0.0002601 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | 10.61 | 1.432 | 7.409 | 17 | 0 | 90.6 | -0.0009335 | 0.000255 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | all | 20 | 13.83 | 1.597 | 8.662 | 20 | 0 | 100 | 0.001659 | 0.001445 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | primary | 16 | 16.04 | 1.486 | 10.8 | 16 | 0 | 100 | -0.0009343 | 0.0002414 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | clip_0.01 | 13 | 15.8 | 1.655 | 9.547 | 13 | 0 | 100 | -0.0008673 | 0.0002706 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | clip_0.25 | 16 | 16.04 | 1.486 | 10.8 | 16 | 0 | 100 | -0.0009343 | 0.0002414 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | fixed20_cost | 17 | 15.72 | 1.433 | 10.97 | 17 | 0 | 100 | -0.0008597 | 0.0002387 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | all | 20 | 10.21 | 1.462 | 6.984 | 20 | 0 | 100 | 4.345e-06 | 0.0005732 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | primary | 16 | 12.05 | 1.477 | 8.16 | 16 | 0 | 100 | -0.001007 | 0.0002601 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 11.72 | 1.626 | 7.205 | 13 | 0 | 100 | -0.0009365 | 0.0002922 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 12.05 | 1.477 | 8.16 | 16 | 0 | 100 | -0.001007 | 0.0002601 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | 11.71 | 1.428 | 8.203 | 17 | 0 | 100 | -0.0009363 | 0.0002542 |
