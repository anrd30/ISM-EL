# Predictive controller and comparators, `eval-fleet,sigma=0.25`, history_rows=576

Scenarios are identical for every arm. Gains are paired by scenario seed; learned models with several seeds are averaged within each scenario first; `sem` is across scenarios. `primary` is the registered definition of controllable (the safety filter raises a `fixed_20` rollout on fewer than 10% of the intervals); the other blocks are the registered sensitivity variants.

| profile | definition | n_forced | n | forced_seeds |
|---|---|---|---|---|
| diurnal_long_popC | primary | 4 | 20 | 3 9 17 19 |
| diurnal_long_popC | clip_0.01 | 7 | 20 | 2 3 9 16 17 18 19 |
| diurnal_long_popC | clip_0.25 | 4 | 20 | 3 9 17 19 |
| diurnal_long_popC | fixed20_cost | 4 | 20 | 3 9 17 19 |

## Summary, all scenarios

| profile | policy | workload_synthetic | n | completed | horizon_expired | terminal_failure | completion_h | lat_deg_mean | lat_deg_p95 | viol_frac | lat_budget_met | p_fail_mean | p_fail_max | risk_viol | clip_frac | deliv_mbps | stall_frac | reward | reward_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_popC | fixed_20 | false | 20 | 20 | 0 | 0 | 144.8 | 0.1006 | 0.5951 | 0.02299 | 0.85 | 0.001987 | 0.003874 | 0.1 | 0.123 | 26.88 | 0 | -391.8 | 120.5 |
| diurnal_long_popC | fixed_50 | false | 20 | 20 | 0 | 0 | 63.1 | 0.127 | 0.6025 | 0.02941 | 0.85 | 0.001234 | 0.002381 | 0.1 | 0.08214 | 54.09 | 0 | -293.7 | 127.7 |
| diurnal_long_popC | fixed_80 | false | 20 | 20 | 0 | 0 | 40.81 | 0.1992 | 0.9245 | 0.06692 | 0.55 | 0.000982 | 0.001919 | 0.1 | 0.05844 | 81.96 | 0 | -443.6 | 133.1 |
| diurnal_long_popC | fixed_120 | false | 20 | 20 | 0 | 0 | 29.55 | 2.88 | 5.667 | 0.5822 | 0 | 0.0008029 | 0.0016 | 0.1 | 0 | 113 | 0 | -3125 | 94.81 |
| diurnal_long_popC | risk_aware_rule_based | false | 20 | 20 | 0 | 0 | 44.04 | 0.1462 | 0.6074 | 0.03361 | 0.85 | 0.001099 | 0.002715 | 0.1 | 0.02574 | 76.39 | 0 | -273.7 | 129 |
| diurnal_long_popC | rule_seasonal_forecast (hand-built) | false | 20 | 20 | 0 | 0 | 40.54 | 0.1612 | 0.6243 | 0.03685 | 0.85 | 0.001092 | 0.002715 | 0.1 | 0.02577 | 82.89 | 0 | -270.3 | 129 |
| diurnal_long_popC | rule+history | false | 20 | 20 | 0 | 0 | 45.42 | 0.1446 | 0.6075 | 0.03361 | 0.85 | 0.001111 | 0.002786 | 0.1 | 0.02795 | 74.06 | 0 | -273.2 | 129 |
| diurnal_long_popC | predictive_controller[seasonal]+history | false | 20 | 20 | 0 | 0 | 37.47 | 0.1623 | 0.6233 | 0.03757 | 0.85 | 0.0009741 | 0.002247 | 0.1 | 0.01541 | 89.31 | 0 | -264.9 | 129.3 |
| diurnal_long_popC | predictive_controller[adaptive] (no history) | false | 20 | 20 | 0 | 0 | 38.42 | 0.1638 | 0.6204 | 0.03835 | 0.85 | 0.001008 | 0.002373 | 0.1 | 0.01598 | 87.2 | 0 | -271.1 | 129 |
| diurnal_long_popC | predictive_controller (history) | false | 20 | 20 | 0 | 0 | 37.6 | 0.1607 | 0.6221 | 0.03731 | 0.85 | 0.0009749 | 0.002255 | 0.1 | 0.01531 | 89 | 0 | -265 | 129.3 |
| diurnal_long_popC | residual_q seed0 | false | 20 | 20 | 0 | 0 | 37.94 | 0.1767 | 0.7544 | 0.03957 | 0.85 | 0.0009941 | 0.002806 | 0.1 | 0.02187 | 88.22 | 0 | -281.5 | 128.2 |
| diurnal_long_popC | residual_q seed1 | false | 20 | 20 | 0 | 0 | 39.15 | 0.1708 | 0.6813 | 0.03855 | 0.85 | 0.001027 | 0.002593 | 0.1 | 0.02547 | 85.59 | 0 | -276.8 | 128.5 |
| diurnal_long_popC | residual_q seed2 | false | 20 | 20 | 0 | 0 | 38.62 | 0.1772 | 0.7527 | 0.04058 | 0.85 | 0.001017 | 0.002727 | 0.1 | 0.02018 | 86.76 | 0 | -287.9 | 128.2 |
| diurnal_long_popC | ORACLE_lookahead (not deployable) | false | 20 | 20 | 0 | 0 | 36.98 | 0.1653 | 0.6487 | 0.0377 | 0.85 | 0.000965 | 0.002196 | 0.1 | 0.01422 | 90.47 | 0 | -264.8 | 129.3 |
| diurnal_long_popC | DP_hindsight_optimum (not deployable) | false | 20 | 20 | 0 | 0 | 37.69 | 0.1593 | 0.6121 | 0.037 | 0.85 | 0.0009758 | 0.002249 | 0.1 | 0.1099 | 88.71 | 0 | -261.4 | 129.1 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Paired gains by block

| profile | treatment | reference | block | n | mean_gain | sem | t | better | worse | percent_of_optimum | viol_frac_diff | viol_frac_diff_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_popC | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | all | 20 | 3.457 | 0.3588 | 9.637 | 20 | 0 | 28.17 | 0.003233 | 0.002074 |
| diurnal_long_popC | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | primary | 16 | 3.454 | 0.361 | 9.566 | 16 | 0 | 30.04 | 2.112e-05 | 1.478e-05 |
| diurnal_long_popC | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | clip_0.01 | 13 | 3.438 | 0.3954 | 8.694 | 13 | 0 | 28.57 | 2.6e-05 | 1.804e-05 |
| diurnal_long_popC | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | clip_0.25 | 16 | 3.454 | 0.361 | 9.566 | 16 | 0 | 30.04 | 2.112e-05 | 1.478e-05 |
| diurnal_long_popC | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | fixed20_cost | 16 | 3.454 | 0.361 | 9.566 | 16 | 0 | 30.04 | 2.112e-05 | 1.478e-05 |
| diurnal_long_popC | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | all | 20 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_popC | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | primary | 16 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_popC | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_popC | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_popC | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | fixed20_cost | 16 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_popC | rule+history | risk_aware_rule_based | all | 20 | 0.5106 | 1.242 | 0.4111 | 10 | 9 | 4.16 | -2.473e-07 | 0.0001721 |
| diurnal_long_popC | rule+history | risk_aware_rule_based | primary | 16 | 0.5063 | 1.561 | 0.3243 | 7 | 9 | 4.404 | -0.0002165 | 0.0001479 |
| diurnal_long_popC | rule+history | risk_aware_rule_based | clip_0.01 | 13 | 0.9705 | 1.904 | 0.5096 | 6 | 7 | 8.068 | -0.0002665 | 0.0001804 |
| diurnal_long_popC | rule+history | risk_aware_rule_based | clip_0.25 | 16 | 0.5063 | 1.561 | 0.3243 | 7 | 9 | 4.404 | -0.0002165 | 0.0001479 |
| diurnal_long_popC | rule+history | risk_aware_rule_based | fixed20_cost | 16 | 0.5063 | 1.561 | 0.3243 | 7 | 9 | 4.404 | -0.0002165 | 0.0001479 |
| diurnal_long_popC | rule+history | rule_seasonal_forecast (hand-built) | all | 20 | -2.947 | 1.201 | -2.453 | 2 | 18 | -33.42 | -0.003233 | 0.001958 |
| diurnal_long_popC | rule+history | rule_seasonal_forecast (hand-built) | primary | 16 | -2.947 | 1.49 | -1.978 | 2 | 14 | -36.64 | -0.0002376 | 0.0001623 |
| diurnal_long_popC | rule+history | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | -2.467 | 1.818 | -1.357 | 2 | 11 | -28.71 | -0.0002925 | 0.000198 |
| diurnal_long_popC | rule+history | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | -2.947 | 1.49 | -1.978 | 2 | 14 | -36.64 | -0.0002376 | 0.0001623 |
| diurnal_long_popC | rule+history | rule_seasonal_forecast (hand-built) | fixed20_cost | 16 | -2.947 | 1.49 | -1.978 | 2 | 14 | -36.64 | -0.0002376 | 0.0001623 |
| diurnal_long_popC | predictive_controller[seasonal]+history | risk_aware_rule_based | all | 20 | 8.815 | 1.231 | 7.163 | 20 | 0 | 71.82 | 0.003958 | 0.002616 |
| diurnal_long_popC | predictive_controller[seasonal]+history | risk_aware_rule_based | primary | 16 | 9.692 | 1.423 | 6.81 | 16 | 0 | 84.29 | -8.494e-05 | 0.0001098 |
| diurnal_long_popC | predictive_controller[seasonal]+history | risk_aware_rule_based | clip_0.01 | 13 | 10.02 | 1.708 | 5.867 | 13 | 0 | 83.33 | -0.0001045 | 0.0001356 |
| diurnal_long_popC | predictive_controller[seasonal]+history | risk_aware_rule_based | clip_0.25 | 16 | 9.692 | 1.423 | 6.81 | 16 | 0 | 84.29 | -8.494e-05 | 0.0001098 |
| diurnal_long_popC | predictive_controller[seasonal]+history | risk_aware_rule_based | fixed20_cost | 16 | 9.692 | 1.423 | 6.81 | 16 | 0 | 84.29 | -8.494e-05 | 0.0001098 |
| diurnal_long_popC | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | all | 20 | 5.357 | 1.02 | 5.254 | 20 | 0 | 60.76 | 0.0007249 | 0.0005541 |
| diurnal_long_popC | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | primary | 16 | 6.239 | 1.173 | 5.318 | 16 | 0 | 77.55 | -0.0001061 | 0.0001206 |
| diurnal_long_popC | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 6.587 | 1.422 | 4.632 | 13 | 0 | 76.66 | -0.0001305 | 0.0001487 |
| diurnal_long_popC | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 6.239 | 1.173 | 5.318 | 16 | 0 | 77.55 | -0.0001061 | 0.0001206 |
| diurnal_long_popC | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | fixed20_cost | 16 | 6.239 | 1.173 | 5.318 | 16 | 0 | 77.55 | -0.0001061 | 0.0001206 |
| diurnal_long_popC | predictive_controller[adaptive] (no history) | risk_aware_rule_based | all | 20 | 2.654 | 2.636 | 1.007 | 13 | 7 | 21.62 | 0.004732 | 0.002517 |
| diurnal_long_popC | predictive_controller[adaptive] (no history) | risk_aware_rule_based | primary | 16 | 2.882 | 3.275 | 0.8802 | 10 | 6 | 25.07 | 0.000891 | 0.0004717 |
| diurnal_long_popC | predictive_controller[adaptive] (no history) | risk_aware_rule_based | clip_0.01 | 13 | 4.052 | 3.686 | 1.099 | 8 | 5 | 33.68 | 0.000747 | 0.0004947 |
| diurnal_long_popC | predictive_controller[adaptive] (no history) | risk_aware_rule_based | clip_0.25 | 16 | 2.882 | 3.275 | 0.8802 | 10 | 6 | 25.07 | 0.000891 | 0.0004717 |
| diurnal_long_popC | predictive_controller[adaptive] (no history) | risk_aware_rule_based | fixed20_cost | 16 | 2.882 | 3.275 | 0.8802 | 10 | 6 | 25.07 | 0.000891 | 0.0004717 |
| diurnal_long_popC | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | all | 20 | -0.8034 | 2.523 | -0.3184 | 13 | 7 | -9.113 | 0.001499 | 0.0005887 |
| diurnal_long_popC | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | primary | 16 | -0.5714 | 3.107 | -0.1839 | 10 | 6 | -7.103 | 0.0008699 | 0.0004796 |
| diurnal_long_popC | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 0.6145 | 3.386 | 0.1815 | 8 | 5 | 7.152 | 0.000721 | 0.0005057 |
| diurnal_long_popC | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | -0.5714 | 3.107 | -0.1839 | 10 | 6 | -7.103 | 0.0008699 | 0.0004796 |
| diurnal_long_popC | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | fixed20_cost | 16 | -0.5714 | 3.107 | -0.1839 | 10 | 6 | -7.103 | 0.0008699 | 0.0004796 |
| diurnal_long_popC | predictive_controller (history) | risk_aware_rule_based | all | 20 | 8.764 | 1.222 | 7.17 | 20 | 0 | 71.4 | 0.003692 | 0.002383 |
| diurnal_long_popC | predictive_controller (history) | risk_aware_rule_based | primary | 16 | 9.669 | 1.407 | 6.872 | 16 | 0 | 84.09 | -8.494e-05 | 0.0001098 |
| diurnal_long_popC | predictive_controller (history) | risk_aware_rule_based | clip_0.01 | 13 | 9.983 | 1.696 | 5.887 | 13 | 0 | 82.98 | -0.0001045 | 0.0001356 |
| diurnal_long_popC | predictive_controller (history) | risk_aware_rule_based | clip_0.25 | 16 | 9.669 | 1.407 | 6.872 | 16 | 0 | 84.09 | -8.494e-05 | 0.0001098 |
| diurnal_long_popC | predictive_controller (history) | risk_aware_rule_based | fixed20_cost | 16 | 9.669 | 1.407 | 6.872 | 16 | 0 | 84.09 | -8.494e-05 | 0.0001098 |
| diurnal_long_popC | predictive_controller (history) | rule_seasonal_forecast (hand-built) | all | 20 | 5.306 | 1.019 | 5.207 | 20 | 0 | 60.18 | 0.0004592 | 0.0003389 |
| diurnal_long_popC | predictive_controller (history) | rule_seasonal_forecast (hand-built) | primary | 16 | 6.215 | 1.165 | 5.336 | 16 | 0 | 77.26 | -0.0001061 | 0.0001206 |
| diurnal_long_popC | predictive_controller (history) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 6.545 | 1.418 | 4.617 | 13 | 0 | 76.17 | -0.0001305 | 0.0001487 |
| diurnal_long_popC | predictive_controller (history) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 6.215 | 1.165 | 5.336 | 16 | 0 | 77.26 | -0.0001061 | 0.0001206 |
| diurnal_long_popC | predictive_controller (history) | rule_seasonal_forecast (hand-built) | fixed20_cost | 16 | 6.215 | 1.165 | 5.336 | 16 | 0 | 77.26 | -0.0001061 | 0.0001206 |
| diurnal_long_popC | ORACLE_lookahead (not deployable) | risk_aware_rule_based | all | 20 | 8.956 | 1.421 | 6.302 | 20 | 0 | 72.97 | 0.004083 | 0.002738 |
| diurnal_long_popC | ORACLE_lookahead (not deployable) | risk_aware_rule_based | primary | 16 | 9.912 | 1.664 | 5.956 | 16 | 0 | 86.21 | -0.0002165 | 0.0001479 |
| diurnal_long_popC | ORACLE_lookahead (not deployable) | risk_aware_rule_based | clip_0.01 | 13 | 10.4 | 2.005 | 5.186 | 13 | 0 | 86.45 | -0.0002665 | 0.0001804 |
| diurnal_long_popC | ORACLE_lookahead (not deployable) | risk_aware_rule_based | clip_0.25 | 16 | 9.912 | 1.664 | 5.956 | 16 | 0 | 86.21 | -0.0002165 | 0.0001479 |
| diurnal_long_popC | ORACLE_lookahead (not deployable) | risk_aware_rule_based | fixed20_cost | 16 | 9.912 | 1.664 | 5.956 | 16 | 0 | 86.21 | -0.0002165 | 0.0001479 |
| diurnal_long_popC | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | all | 20 | 5.498 | 1.249 | 4.402 | 19 | 1 | 62.36 | 0.0008506 | 0.0006764 |
| diurnal_long_popC | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | primary | 16 | 6.458 | 1.465 | 4.407 | 15 | 1 | 80.28 | -0.0002376 | 0.0001623 |
| diurnal_long_popC | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 6.962 | 1.775 | 3.922 | 12 | 1 | 81.03 | -0.0002925 | 0.000198 |
| diurnal_long_popC | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 6.458 | 1.465 | 4.407 | 15 | 1 | 80.28 | -0.0002376 | 0.0001623 |
| diurnal_long_popC | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | fixed20_cost | 16 | 6.458 | 1.465 | 4.407 | 15 | 1 | 80.28 | -0.0002376 | 0.0001623 |
| diurnal_long_popC | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | all | 20 | 12.27 | 2.317 | 5.297 | 20 | 0 | 100 | 0.003385 | 0.002478 |
| diurnal_long_popC | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | primary | 16 | 11.5 | 1.663 | 6.916 | 16 | 0 | 100 | -0.0002165 | 0.0001479 |
| diurnal_long_popC | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | clip_0.01 | 13 | 12.03 | 2 | 6.016 | 13 | 0 | 100 | -0.0002665 | 0.0001804 |
| diurnal_long_popC | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | clip_0.25 | 16 | 11.5 | 1.663 | 6.916 | 16 | 0 | 100 | -0.0002165 | 0.0001479 |
| diurnal_long_popC | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | fixed20_cost | 16 | 11.5 | 1.663 | 6.916 | 16 | 0 | 100 | -0.0002165 | 0.0001479 |
| diurnal_long_popC | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | all | 20 | 8.817 | 2.095 | 4.209 | 20 | 0 | 100 | 0.0001517 | 0.0005354 |
| diurnal_long_popC | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | primary | 16 | 8.044 | 1.47 | 5.474 | 16 | 0 | 100 | -0.0002376 | 0.0001623 |
| diurnal_long_popC | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 8.592 | 1.779 | 4.83 | 13 | 0 | 100 | -0.0002925 | 0.000198 |
| diurnal_long_popC | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 8.044 | 1.47 | 5.474 | 16 | 0 | 100 | -0.0002376 | 0.0001623 |
| diurnal_long_popC | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | fixed20_cost | 16 | 8.044 | 1.47 | 5.474 | 16 | 0 | 100 | -0.0002376 | 0.0001623 |
| diurnal_long_popC | residual-Q (zero-shot transfer from A, seeds averaged) | risk_aware_rule_based | all | 20 | -8.339 | 2.139 | -3.899 | 5 | 15 | -67.94 | 0.005952 | 0.002564 |
| diurnal_long_popC | residual-Q (zero-shot transfer from A, seeds averaged) | risk_aware_rule_based | primary | 16 | -9.1 | 2.36 | -3.855 | 3 | 13 | -79.15 | 0.002138 | 0.0004819 |
| diurnal_long_popC | residual-Q (zero-shot transfer from A, seeds averaged) | risk_aware_rule_based | clip_0.01 | 13 | -8.989 | 2.654 | -3.387 | 2 | 11 | -74.72 | 0.002242 | 0.0005568 |
| diurnal_long_popC | residual-Q (zero-shot transfer from A, seeds averaged) | risk_aware_rule_based | clip_0.25 | 16 | -9.1 | 2.36 | -3.855 | 3 | 13 | -79.15 | 0.002138 | 0.0004819 |
| diurnal_long_popC | residual-Q (zero-shot transfer from A, seeds averaged) | risk_aware_rule_based | fixed20_cost | 16 | -9.1 | 2.36 | -3.855 | 3 | 13 | -79.15 | 0.002138 | 0.0004819 |
| diurnal_long_popC | residual-Q (zero-shot transfer from A, seeds averaged) | rule_seasonal_forecast (hand-built) | all | 20 | -11.8 | 2.2 | -5.363 | 0 | 20 | -133.8 | 0.002719 | 0.0006449 |
| diurnal_long_popC | residual-Q (zero-shot transfer from A, seeds averaged) | rule_seasonal_forecast (hand-built) | primary | 16 | -12.55 | 2.437 | -5.151 | 0 | 16 | -156.1 | 0.002117 | 0.0004817 |
| diurnal_long_popC | residual-Q (zero-shot transfer from A, seeds averaged) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | -12.43 | 2.825 | -4.399 | 0 | 13 | -144.6 | 0.002216 | 0.0005569 |
| diurnal_long_popC | residual-Q (zero-shot transfer from A, seeds averaged) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | -12.55 | 2.437 | -5.151 | 0 | 16 | -156.1 | 0.002117 | 0.0004817 |
| diurnal_long_popC | residual-Q (zero-shot transfer from A, seeds averaged) | rule_seasonal_forecast (hand-built) | fixed20_cost | 16 | -12.55 | 2.437 | -5.151 | 0 | 16 | -156.1 | 0.002117 | 0.0004817 |
