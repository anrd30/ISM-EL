# Predictive controller and comparators, `eval-fleet,sigma=0.25`, history_rows=576

Scenarios are identical for every arm. Gains are paired by scenario seed; learned models with several seeds are averaged within each scenario first; `sem` is across scenarios. `primary` is the registered definition of controllable (the safety filter raises a `fixed_20` rollout on fewer than 10% of the intervals); the other blocks are the registered sensitivity variants.

| profile | definition | n_forced | n | forced_seeds |
|---|---|---|---|---|
| diurnal_long_popD | primary | 4 | 20 | 3 9 17 19 |
| diurnal_long_popD | clip_0.01 | 7 | 20 | 2 3 9 16 17 18 19 |
| diurnal_long_popD | clip_0.25 | 4 | 20 | 3 9 17 19 |
| diurnal_long_popD | fixed20_cost | 4 | 20 | 3 9 17 19 |

## Summary, all scenarios

| profile | policy | workload_synthetic | n | completed | horizon_expired | terminal_failure | completion_h | lat_deg_mean | lat_deg_p95 | viol_frac | lat_budget_met | p_fail_mean | p_fail_max | risk_viol | clip_frac | deliv_mbps | stall_frac | reward | reward_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_popD | fixed_20 | false | 20 | 20 | 0 | 0 | 144.8 | 0.1007 | 0.6067 | 0.03105 | 0.85 | 0.001984 | 0.003874 | 0.1 | 0.1225 | 26.97 | 0 | -437.2 | 144.8 |
| diurnal_long_popD | fixed_50 | false | 20 | 20 | 0 | 0 | 63.04 | 0.1349 | 0.6204 | 0.04146 | 0.85 | 0.00123 | 0.002382 | 0.1 | 0.08131 | 54.24 | 0 | -359.6 | 150.5 |
| diurnal_long_popD | fixed_80 | false | 20 | 20 | 0 | 0 | 40.83 | 0.257 | 1.055 | 0.08687 | 0.65 | 0.0009771 | 0.001919 | 0.1 | 0.0572 | 82.07 | 0 | -557.6 | 142.7 |
| diurnal_long_popD | fixed_120 | false | 20 | 20 | 0 | 0 | 29.06 | 3.139 | 5.667 | 0.8475 | 0 | 0.0007973 | 0.0016 | 0.1 | 0 | 114.9 | 0 | -4088 | 91.86 |
| diurnal_long_popD | risk_aware_rule_based | false | 20 | 20 | 0 | 0 | 47.08 | 0.1556 | 0.6415 | 0.04877 | 0.85 | 0.001081 | 0.002744 | 0.1 | 0.02346 | 71.89 | 0 | -348.4 | 152.5 |
| diurnal_long_popD | rule_seasonal_forecast (hand-built) | false | 20 | 20 | 0 | 0 | 49.66 | 0.1537 | 0.6413 | 0.04867 | 0.85 | 0.00109 | 0.00275 | 0.1 | 0.02281 | 68.51 | 0 | -363.9 | 150.3 |
| diurnal_long_popD | rule+history | false | 20 | 20 | 0 | 0 | 46.86 | 0.1547 | 0.6415 | 0.04846 | 0.85 | 0.00107 | 0.002765 | 0.1 | 0.02358 | 72.14 | 0 | -345.5 | 152.8 |
| diurnal_long_popD | predictive_controller[seasonal]+history | false | 20 | 20 | 0 | 0 | 49.7 | 0.1495 | 0.6409 | 0.04647 | 0.85 | 0.001137 | 0.003359 | 0.1 | 0.02562 | 68.52 | 0 | -332.1 | 152.9 |
| diurnal_long_popD | predictive_controller[adaptive] (no history) | false | 20 | 20 | 0 | 0 | 46.11 | 0.1534 | 0.6446 | 0.04771 | 0.85 | 0.001073 | 0.003142 | 0.1 | 0.02211 | 73.27 | 0 | -334.6 | 152.7 |
| diurnal_long_popD | predictive_controller (history) | false | 20 | 20 | 0 | 0 | 46.16 | 0.1539 | 0.6416 | 0.04803 | 0.85 | 0.001072 | 0.003028 | 0.1 | 0.02357 | 73.25 | 0 | -333.1 | 153.2 |
| diurnal_long_popD | residual_q seed0 | false | 20 | 20 | 0 | 0 | 45.1 | 0.1761 | 0.6692 | 0.05802 | 0.85 | 0.001044 | 0.003071 | 0.1 | 0.02077 | 74.65 | 0 | -402.2 | 148.9 |
| diurnal_long_popD | residual_q seed1 | false | 20 | 20 | 0 | 0 | 46.94 | 0.1668 | 0.6421 | 0.05367 | 0.85 | 0.001062 | 0.002862 | 0.1 | 0.02206 | 72.01 | 0 | -384.6 | 148.9 |
| diurnal_long_popD | residual_q seed2 | false | 20 | 20 | 0 | 0 | 45.4 | 0.1774 | 0.6705 | 0.05882 | 0.85 | 0.001052 | 0.00295 | 0.1 | 0.02096 | 74.2 | 0 | -410.5 | 148.1 |
| diurnal_long_popD | ORACLE_lookahead (not deployable) | false | 20 | 20 | 0 | 0 | 41.15 | 0.1562 | 0.6658 | 0.04737 | 0.85 | 0.0009906 | 0.002605 | 0.1 | 0.01851 | 81.44 | 0 | -314 | 153.7 |
| diurnal_long_popD | DP_hindsight_optimum (not deployable) | false | 20 | 20 | 0 | 0 | 41.83 | 0.1518 | 0.6429 | 0.04711 | 0.85 | 0.001001 | 0.002679 | 0.1 | 0.1055 | 80.1 | 0 | -313.1 | 153.7 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Paired gains by block

| profile | treatment | reference | block | n | mean_gain | sem | t | better | worse | percent_of_optimum | viol_frac_diff | viol_frac_diff_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_popD | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | all | 20 | -15.52 | 5.549 | -2.796 | 5 | 15 | -43.99 | -9.955e-05 | 0.001556 |
| diurnal_long_popD | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | primary | 16 | -21.43 | 6.049 | -3.542 | 2 | 14 | -56.54 | 0.002096 | 0.0008402 |
| diurnal_long_popD | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | clip_0.01 | 13 | -18.18 | 6.31 | -2.88 | 2 | 11 | -48.52 | 0.001651 | 0.0009133 |
| diurnal_long_popD | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | clip_0.25 | 16 | -21.43 | 6.049 | -3.542 | 2 | 14 | -56.54 | 0.002096 | 0.0008402 |
| diurnal_long_popD | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | fixed20_cost | 16 | -21.43 | 6.049 | -3.542 | 2 | 14 | -56.54 | 0.002096 | 0.0008402 |
| diurnal_long_popD | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | all | 20 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_popD | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | primary | 16 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_popD | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_popD | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_popD | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | fixed20_cost | 16 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_popD | rule+history | risk_aware_rule_based | all | 20 | 2.867 | 1.476 | 1.942 | 12 | 3 | 8.128 | -0.000314 | 0.0002598 |
| diurnal_long_popD | rule+history | risk_aware_rule_based | primary | 16 | 3.509 | 1.819 | 1.929 | 11 | 3 | 9.257 | -0.0004087 | 0.0003219 |
| diurnal_long_popD | rule+history | risk_aware_rule_based | clip_0.01 | 13 | 3.57 | 2.113 | 1.689 | 10 | 2 | 9.53 | -0.0003543 | 0.0003811 |
| diurnal_long_popD | rule+history | risk_aware_rule_based | clip_0.25 | 16 | 3.509 | 1.819 | 1.929 | 11 | 3 | 9.257 | -0.0004087 | 0.0003219 |
| diurnal_long_popD | rule+history | risk_aware_rule_based | fixed20_cost | 16 | 3.509 | 1.819 | 1.929 | 11 | 3 | 9.257 | -0.0004087 | 0.0003219 |
| diurnal_long_popD | rule+history | rule_seasonal_forecast (hand-built) | all | 20 | 18.38 | 5.692 | 3.229 | 14 | 6 | 36.19 | -0.0002144 | 0.001592 |
| diurnal_long_popD | rule+history | rule_seasonal_forecast (hand-built) | primary | 16 | 24.94 | 6.041 | 4.128 | 13 | 3 | 42.03 | -0.002505 | 0.0008689 |
| diurnal_long_popD | rule+history | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 21.74 | 6.074 | 3.58 | 11 | 2 | 39.09 | -0.002005 | 0.0009047 |
| diurnal_long_popD | rule+history | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 24.94 | 6.041 | 4.128 | 13 | 3 | 42.03 | -0.002505 | 0.0008689 |
| diurnal_long_popD | rule+history | rule_seasonal_forecast (hand-built) | fixed20_cost | 16 | 24.94 | 6.041 | 4.128 | 13 | 3 | 42.03 | -0.002505 | 0.0008689 |
| diurnal_long_popD | predictive_controller[seasonal]+history | risk_aware_rule_based | all | 20 | 16.21 | 3.898 | 4.159 | 16 | 4 | 45.95 | -0.002303 | 0.0007235 |
| diurnal_long_popD | predictive_controller[seasonal]+history | risk_aware_rule_based | primary | 16 | 17.27 | 4.748 | 3.638 | 13 | 3 | 45.57 | -0.002974 | 0.0006436 |
| diurnal_long_popD | predictive_controller[seasonal]+history | risk_aware_rule_based | clip_0.01 | 13 | 18.77 | 5.118 | 3.667 | 11 | 2 | 50.11 | -0.003253 | 0.0006834 |
| diurnal_long_popD | predictive_controller[seasonal]+history | risk_aware_rule_based | clip_0.25 | 16 | 17.27 | 4.748 | 3.638 | 13 | 3 | 45.57 | -0.002974 | 0.0006436 |
| diurnal_long_popD | predictive_controller[seasonal]+history | risk_aware_rule_based | fixed20_cost | 16 | 17.27 | 4.748 | 3.638 | 13 | 3 | 45.57 | -0.002974 | 0.0006436 |
| diurnal_long_popD | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | all | 20 | 31.73 | 5.619 | 5.646 | 18 | 2 | 62.46 | -0.002203 | 0.002 |
| diurnal_long_popD | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | primary | 16 | 38.7 | 5.776 | 6.7 | 15 | 1 | 65.23 | -0.005071 | 0.0008155 |
| diurnal_long_popD | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 36.94 | 6.107 | 6.05 | 12 | 1 | 66.41 | -0.004904 | 0.0007911 |
| diurnal_long_popD | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 38.7 | 5.776 | 6.7 | 15 | 1 | 65.23 | -0.005071 | 0.0008155 |
| diurnal_long_popD | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | fixed20_cost | 16 | 38.7 | 5.776 | 6.7 | 15 | 1 | 65.23 | -0.005071 | 0.0008155 |
| diurnal_long_popD | predictive_controller[adaptive] (no history) | risk_aware_rule_based | all | 20 | 13.74 | 2.692 | 5.105 | 17 | 3 | 38.95 | -0.001065 | 0.0006219 |
| diurnal_long_popD | predictive_controller[adaptive] (no history) | risk_aware_rule_based | primary | 16 | 15.45 | 3.099 | 4.986 | 14 | 2 | 40.76 | -0.00204 | 0.0005083 |
| diurnal_long_popD | predictive_controller[adaptive] (no history) | risk_aware_rule_based | clip_0.01 | 13 | 16.52 | 3.1 | 5.328 | 12 | 1 | 44.09 | -0.002269 | 0.0005424 |
| diurnal_long_popD | predictive_controller[adaptive] (no history) | risk_aware_rule_based | clip_0.25 | 16 | 15.45 | 3.099 | 4.986 | 14 | 2 | 40.76 | -0.00204 | 0.0005083 |
| diurnal_long_popD | predictive_controller[adaptive] (no history) | risk_aware_rule_based | fixed20_cost | 16 | 15.45 | 3.099 | 4.986 | 14 | 2 | 40.76 | -0.00204 | 0.0005083 |
| diurnal_long_popD | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | all | 20 | 29.26 | 5.283 | 5.537 | 19 | 1 | 57.6 | -0.0009655 | 0.001891 |
| diurnal_long_popD | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | primary | 16 | 36.88 | 4.935 | 7.474 | 16 | 0 | 62.16 | -0.004136 | 0.0007302 |
| diurnal_long_popD | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 34.69 | 5.173 | 6.707 | 13 | 0 | 62.36 | -0.00392 | 0.0007263 |
| diurnal_long_popD | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 36.88 | 4.935 | 7.474 | 16 | 0 | 62.16 | -0.004136 | 0.0007302 |
| diurnal_long_popD | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | fixed20_cost | 16 | 36.88 | 4.935 | 7.474 | 16 | 0 | 62.16 | -0.004136 | 0.0007302 |
| diurnal_long_popD | predictive_controller (history) | risk_aware_rule_based | all | 20 | 15.24 | 3.282 | 4.642 | 20 | 0 | 43.2 | -0.0007456 | 0.0009047 |
| diurnal_long_popD | predictive_controller (history) | risk_aware_rule_based | primary | 16 | 17.29 | 3.86 | 4.48 | 16 | 0 | 45.62 | -0.002194 | 0.0005661 |
| diurnal_long_popD | predictive_controller (history) | risk_aware_rule_based | clip_0.01 | 13 | 17.54 | 4.254 | 4.124 | 13 | 0 | 46.84 | -0.002342 | 0.0005959 |
| diurnal_long_popD | predictive_controller (history) | risk_aware_rule_based | clip_0.25 | 16 | 17.29 | 3.86 | 4.48 | 16 | 0 | 45.62 | -0.002194 | 0.0005661 |
| diurnal_long_popD | predictive_controller (history) | risk_aware_rule_based | fixed20_cost | 16 | 17.29 | 3.86 | 4.48 | 16 | 0 | 45.62 | -0.002194 | 0.0005661 |
| diurnal_long_popD | predictive_controller (history) | rule_seasonal_forecast (hand-built) | all | 20 | 30.75 | 6.488 | 4.74 | 18 | 2 | 60.55 | -0.0006461 | 0.002292 |
| diurnal_long_popD | predictive_controller (history) | rule_seasonal_forecast (hand-built) | primary | 16 | 38.72 | 6.728 | 5.755 | 15 | 1 | 65.26 | -0.00429 | 0.0009272 |
| diurnal_long_popD | predictive_controller (history) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 35.72 | 6.719 | 5.316 | 12 | 1 | 64.2 | -0.003992 | 0.0008855 |
| diurnal_long_popD | predictive_controller (history) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 38.72 | 6.728 | 5.755 | 15 | 1 | 65.26 | -0.00429 | 0.0009272 |
| diurnal_long_popD | predictive_controller (history) | rule_seasonal_forecast (hand-built) | fixed20_cost | 16 | 38.72 | 6.728 | 5.755 | 15 | 1 | 65.26 | -0.00429 | 0.0009272 |
| diurnal_long_popD | ORACLE_lookahead (not deployable) | risk_aware_rule_based | all | 20 | 34.32 | 3.884 | 8.837 | 20 | 0 | 97.3 | -0.0014 | 0.00165 |
| diurnal_long_popD | ORACLE_lookahead (not deployable) | risk_aware_rule_based | primary | 16 | 36.83 | 4.046 | 9.103 | 16 | 0 | 97.17 | -0.004181 | 0.0004789 |
| diurnal_long_popD | ORACLE_lookahead (not deployable) | risk_aware_rule_based | clip_0.01 | 13 | 36.4 | 4.754 | 7.657 | 13 | 0 | 97.18 | -0.004269 | 0.0005482 |
| diurnal_long_popD | ORACLE_lookahead (not deployable) | risk_aware_rule_based | clip_0.25 | 16 | 36.83 | 4.046 | 9.103 | 16 | 0 | 97.17 | -0.004181 | 0.0004789 |
| diurnal_long_popD | ORACLE_lookahead (not deployable) | risk_aware_rule_based | fixed20_cost | 16 | 36.83 | 4.046 | 9.103 | 16 | 0 | 97.17 | -0.004181 | 0.0004789 |
| diurnal_long_popD | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | all | 20 | 49.84 | 6.042 | 8.248 | 20 | 0 | 98.12 | -0.0013 | 0.002983 |
| diurnal_long_popD | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | primary | 16 | 58.26 | 5.468 | 10.65 | 16 | 0 | 98.19 | -0.006277 | 0.0006626 |
| diurnal_long_popD | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 54.58 | 5.838 | 9.349 | 13 | 0 | 98.1 | -0.005919 | 0.0006631 |
| diurnal_long_popD | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 58.26 | 5.468 | 10.65 | 16 | 0 | 98.19 | -0.006277 | 0.0006626 |
| diurnal_long_popD | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | fixed20_cost | 16 | 58.26 | 5.468 | 10.65 | 16 | 0 | 98.19 | -0.006277 | 0.0006626 |
| diurnal_long_popD | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | all | 20 | 35.27 | 3.924 | 8.99 | 20 | 0 | 100 | -0.001663 | 0.001517 |
| diurnal_long_popD | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | primary | 16 | 37.9 | 4.074 | 9.305 | 16 | 0 | 100 | -0.004181 | 0.0004789 |
| diurnal_long_popD | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | clip_0.01 | 13 | 37.46 | 4.793 | 7.816 | 13 | 0 | 100 | -0.004269 | 0.0005482 |
| diurnal_long_popD | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | clip_0.25 | 16 | 37.9 | 4.074 | 9.305 | 16 | 0 | 100 | -0.004181 | 0.0004789 |
| diurnal_long_popD | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | fixed20_cost | 16 | 37.9 | 4.074 | 9.305 | 16 | 0 | 100 | -0.004181 | 0.0004789 |
| diurnal_long_popD | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | all | 20 | 50.79 | 6.102 | 8.324 | 20 | 0 | 100 | -0.001564 | 0.002853 |
| diurnal_long_popD | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | primary | 16 | 59.33 | 5.504 | 10.78 | 16 | 0 | 100 | -0.006277 | 0.0006626 |
| diurnal_long_popD | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 55.63 | 5.875 | 9.47 | 13 | 0 | 100 | -0.005919 | 0.0006631 |
| diurnal_long_popD | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 59.33 | 5.504 | 10.78 | 16 | 0 | 100 | -0.006277 | 0.0006626 |
| diurnal_long_popD | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | fixed20_cost | 16 | 59.33 | 5.504 | 10.78 | 16 | 0 | 100 | -0.006277 | 0.0006626 |
| diurnal_long_popD | residual-Q (zero-shot transfer from A, seeds averaged) | risk_aware_rule_based | all | 20 | -50.72 | 10.39 | -4.884 | 0 | 20 | -143.8 | 0.008063 | 0.0014 |
| diurnal_long_popD | residual-Q (zero-shot transfer from A, seeds averaged) | risk_aware_rule_based | primary | 16 | -62.57 | 11.13 | -5.624 | 0 | 16 | -165.1 | 0.009241 | 0.001591 |
| diurnal_long_popD | residual-Q (zero-shot transfer from A, seeds averaged) | risk_aware_rule_based | clip_0.01 | 13 | -53.89 | 10.52 | -5.125 | 0 | 13 | -143.9 | 0.007982 | 0.001538 |
| diurnal_long_popD | residual-Q (zero-shot transfer from A, seeds averaged) | risk_aware_rule_based | clip_0.25 | 16 | -62.57 | 11.13 | -5.624 | 0 | 16 | -165.1 | 0.009241 | 0.001591 |
| diurnal_long_popD | residual-Q (zero-shot transfer from A, seeds averaged) | risk_aware_rule_based | fixed20_cost | 16 | -62.57 | 11.13 | -5.624 | 0 | 16 | -165.1 | 0.009241 | 0.001591 |
| diurnal_long_popD | residual-Q (zero-shot transfer from A, seeds averaged) | rule_seasonal_forecast (hand-built) | all | 20 | -35.2 | 6.717 | -5.241 | 2 | 18 | -69.31 | 0.008163 | 0.001437 |
| diurnal_long_popD | residual-Q (zero-shot transfer from A, seeds averaged) | rule_seasonal_forecast (hand-built) | primary | 16 | -41.15 | 7.65 | -5.379 | 1 | 15 | -69.35 | 0.007145 | 0.001122 |
| diurnal_long_popD | residual-Q (zero-shot transfer from A, seeds averaged) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | -35.71 | 8.059 | -4.432 | 1 | 12 | -64.19 | 0.006331 | 0.001178 |
| diurnal_long_popD | residual-Q (zero-shot transfer from A, seeds averaged) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | -41.15 | 7.65 | -5.379 | 1 | 15 | -69.35 | 0.007145 | 0.001122 |
| diurnal_long_popD | residual-Q (zero-shot transfer from A, seeds averaged) | rule_seasonal_forecast (hand-built) | fixed20_cost | 16 | -41.15 | 7.65 | -5.379 | 1 | 15 | -69.35 | 0.007145 | 0.001122 |
