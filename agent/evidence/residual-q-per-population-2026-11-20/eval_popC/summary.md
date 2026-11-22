# Post-registration: residual Q retrained per population, `profile_diurnal_long_popC_holdout.csv`

Run after the pre-registered evaluation; it does not replace it. Gains are paired by scenario seed; learned models with several seeds are averaged within each scenario first; `sem` is across scenarios.

| profile | definition | n_forced | n | forced_seeds |
|---|---|---|---|---|
| diurnal_long_popC_holdout | primary | 4 | 20 | 3 9 17 19 |
| diurnal_long_popC_holdout | clip_0.01 | 7 | 20 | 2 3 9 16 17 18 19 |
| diurnal_long_popC_holdout | clip_0.25 | 4 | 20 | 3 9 17 19 |
| diurnal_long_popC_holdout | fixed20_cost | 3 | 20 | 9 17 19 |

| profile | treatment | reference | block | n | mean_gain | sem | t | better | worse | percent_of_optimum | viol_frac_diff | viol_frac_diff_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_popC_holdout | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | all | 20 | 3.429 | 0.305 | 11.24 | 20 | 0 | 35.95 | 0.001809 | 0.001044 |
| diurnal_long_popC_holdout | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | primary | 16 | 3.772 | 0.3123 | 12.08 | 16 | 0 | 35.15 | 0 | 0 |
| diurnal_long_popC_holdout | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | clip_0.01 | 13 | 3.877 | 0.3671 | 10.56 | 13 | 0 | 35.33 | 0 | 0 |
| diurnal_long_popC_holdout | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | clip_0.25 | 16 | 3.772 | 0.3123 | 12.08 | 16 | 0 | 35.15 | 0 | 0 |
| diurnal_long_popC_holdout | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | fixed20_cost | 17 | 3.746 | 0.2945 | 12.72 | 17 | 0 | 35.05 | 0 | 0 |
| diurnal_long_popC_holdout | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | all | 20 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_popC_holdout | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | primary | 16 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_popC_holdout | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_popC_holdout | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_popC_holdout | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_popC_holdout | predictive_controller (history) | risk_aware_rule_based | all | 20 | 8.74 | 0.8201 | 10.66 | 20 | 0 | 91.63 | 0.002605 | 0.001529 |
| diurnal_long_popC_holdout | predictive_controller (history) | risk_aware_rule_based | primary | 16 | 9.782 | 0.7464 | 13.11 | 16 | 0 | 91.14 | 0 | 0 |
| diurnal_long_popC_holdout | predictive_controller (history) | risk_aware_rule_based | clip_0.01 | 13 | 10.07 | 0.8442 | 11.93 | 13 | 0 | 91.74 | 0 | 0 |
| diurnal_long_popC_holdout | predictive_controller (history) | risk_aware_rule_based | clip_0.25 | 16 | 9.782 | 0.7464 | 13.11 | 16 | 0 | 91.14 | 0 | 0 |
| diurnal_long_popC_holdout | predictive_controller (history) | risk_aware_rule_based | fixed20_cost | 17 | 9.772 | 0.7012 | 13.94 | 17 | 0 | 91.43 | 0 | 0 |
| diurnal_long_popC_holdout | predictive_controller (history) | rule_seasonal_forecast (hand-built) | all | 20 | 5.31 | 0.6092 | 8.718 | 20 | 0 | 86.93 | 0.0007966 | 0.0005754 |
| diurnal_long_popC_holdout | predictive_controller (history) | rule_seasonal_forecast (hand-built) | primary | 16 | 6.01 | 0.5863 | 10.25 | 16 | 0 | 86.34 | 0 | 0 |
| diurnal_long_popC_holdout | predictive_controller (history) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 6.191 | 0.6414 | 9.652 | 13 | 0 | 87.23 | 0 | 0 |
| diurnal_long_popC_holdout | predictive_controller (history) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 6.01 | 0.5863 | 10.25 | 16 | 0 | 86.34 | 0 | 0 |
| diurnal_long_popC_holdout | predictive_controller (history) | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | 6.026 | 0.551 | 10.94 | 17 | 0 | 86.81 | 0 | 0 |
| diurnal_long_popC_holdout | residual-Q (retrained on its own training split) | risk_aware_rule_based | all | 20 | 2.264 | 0.3442 | 6.578 | 20 | 0 | 23.74 | 0.001926 | 0.001077 |
| diurnal_long_popC_holdout | residual-Q (retrained on its own training split) | risk_aware_rule_based | primary | 16 | 2.425 | 0.4146 | 5.849 | 16 | 0 | 22.59 | 8.012e-05 | 5.476e-05 |
| diurnal_long_popC_holdout | residual-Q (retrained on its own training split) | risk_aware_rule_based | clip_0.01 | 13 | 2.736 | 0.4634 | 5.905 | 13 | 0 | 24.93 | 5.077e-05 | 5.077e-05 |
| diurnal_long_popC_holdout | residual-Q (retrained on its own training split) | risk_aware_rule_based | clip_0.25 | 16 | 2.425 | 0.4146 | 5.849 | 16 | 0 | 22.59 | 8.012e-05 | 5.476e-05 |
| diurnal_long_popC_holdout | residual-Q (retrained on its own training split) | risk_aware_rule_based | fixed20_cost | 17 | 2.436 | 0.3896 | 6.253 | 17 | 0 | 22.79 | 7.541e-05 | 5.166e-05 |
| diurnal_long_popC_holdout | residual-Q (retrained on its own training split) | rule_seasonal_forecast (hand-built) | all | 20 | -1.165 | 0.2526 | -4.613 | 0 | 20 | -19.07 | 0.0001173 | 5.952e-05 |
| diurnal_long_popC_holdout | residual-Q (retrained on its own training split) | rule_seasonal_forecast (hand-built) | primary | 16 | -1.347 | 0.2965 | -4.544 | 0 | 16 | -19.36 | 8.012e-05 | 5.476e-05 |
| diurnal_long_popC_holdout | residual-Q (retrained on its own training split) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | -1.141 | 0.2888 | -3.951 | 0 | 13 | -16.08 | 5.077e-05 | 5.077e-05 |
| diurnal_long_popC_holdout | residual-Q (retrained on its own training split) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | -1.347 | 0.2965 | -4.544 | 0 | 16 | -19.36 | 8.012e-05 | 5.476e-05 |
| diurnal_long_popC_holdout | residual-Q (retrained on its own training split) | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | -1.31 | 0.281 | -4.662 | 0 | 17 | -18.87 | 7.541e-05 | 5.166e-05 |
| diurnal_long_popC_holdout | residual-Q (zero-shot transfer from A) | risk_aware_rule_based | all | 20 | -11.29 | 1.648 | -6.853 | 2 | 18 | -118.4 | 0.004936 | 0.001271 |
| diurnal_long_popC_holdout | residual-Q (zero-shot transfer from A) | risk_aware_rule_based | primary | 16 | -12.43 | 1.473 | -8.438 | 0 | 16 | -115.8 | 0.002726 | 0.0002865 |
| diurnal_long_popC_holdout | residual-Q (zero-shot transfer from A) | risk_aware_rule_based | clip_0.01 | 13 | -11.88 | 1.685 | -7.048 | 0 | 13 | -108.2 | 0.00269 | 0.0003362 |
| diurnal_long_popC_holdout | residual-Q (zero-shot transfer from A) | risk_aware_rule_based | clip_0.25 | 16 | -12.43 | 1.473 | -8.438 | 0 | 16 | -115.8 | 0.002726 | 0.0002865 |
| diurnal_long_popC_holdout | residual-Q (zero-shot transfer from A) | risk_aware_rule_based | fixed20_cost | 17 | -13.08 | 1.532 | -8.541 | 0 | 17 | -122.4 | 0.002807 | 0.000281 |
| diurnal_long_popC_holdout | residual-Q (zero-shot transfer from A) | rule_seasonal_forecast (hand-built) | all | 20 | -14.72 | 1.776 | -8.287 | 0 | 20 | -241 | 0.003127 | 0.0003498 |
| diurnal_long_popC_holdout | residual-Q (zero-shot transfer from A) | rule_seasonal_forecast (hand-built) | primary | 16 | -16.2 | 1.52 | -10.66 | 0 | 16 | -232.7 | 0.002726 | 0.0002865 |
| diurnal_long_popC_holdout | residual-Q (zero-shot transfer from A) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | -15.75 | 1.732 | -9.098 | 0 | 13 | -222 | 0.00269 | 0.0003362 |
| diurnal_long_popC_holdout | residual-Q (zero-shot transfer from A) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | -16.2 | 1.52 | -10.66 | 0 | 16 | -232.7 | 0.002726 | 0.0002865 |
| diurnal_long_popC_holdout | residual-Q (zero-shot transfer from A) | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | -16.83 | 1.562 | -10.78 | 0 | 17 | -242.5 | 0.002807 | 0.000281 |
| diurnal_long_popC_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | all | 20 | 8.344 | 0.8465 | 9.857 | 20 | 0 | 87.48 | 0.002793 | 0.001628 |
| diurnal_long_popC_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | primary | 16 | 9.393 | 0.7902 | 11.89 | 16 | 0 | 87.52 | 0 | 0 |
| diurnal_long_popC_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | clip_0.01 | 13 | 9.685 | 0.8979 | 10.79 | 13 | 0 | 88.25 | 0 | 0 |
| diurnal_long_popC_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | clip_0.25 | 16 | 9.393 | 0.7902 | 11.89 | 16 | 0 | 87.52 | 0 | 0 |
| diurnal_long_popC_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | fixed20_cost | 17 | 9.381 | 0.7423 | 12.64 | 17 | 0 | 87.77 | 0 | 0 |
| diurnal_long_popC_holdout | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | all | 20 | 4.915 | 0.6476 | 7.589 | 20 | 0 | 80.45 | 0.0009844 | 0.0006413 |
| diurnal_long_popC_holdout | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | primary | 16 | 5.621 | 0.647 | 8.687 | 16 | 0 | 80.75 | 0 | 0 |
| diurnal_long_popC_holdout | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 5.808 | 0.7114 | 8.164 | 13 | 0 | 81.84 | 0 | 0 |
| diurnal_long_popC_holdout | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 5.621 | 0.647 | 8.687 | 16 | 0 | 80.75 | 0 | 0 |
| diurnal_long_popC_holdout | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | 5.635 | 0.6079 | 9.269 | 17 | 0 | 81.17 | 0 | 0 |
| diurnal_long_popC_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | all | 20 | 9.538 | 0.8743 | 10.91 | 20 | 0 | 100 | 0.002605 | 0.001529 |
| diurnal_long_popC_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | primary | 16 | 10.73 | 0.7604 | 14.11 | 16 | 0 | 100 | 0 | 0 |
| diurnal_long_popC_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | clip_0.01 | 13 | 10.97 | 0.8773 | 12.51 | 13 | 0 | 100 | 0 | 0 |
| diurnal_long_popC_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | clip_0.25 | 16 | 10.73 | 0.7604 | 14.11 | 16 | 0 | 100 | 0 | 0 |
| diurnal_long_popC_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | fixed20_cost | 17 | 10.69 | 0.7157 | 14.93 | 17 | 0 | 100 | 0 | 0 |
| diurnal_long_popC_holdout | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | all | 20 | 6.109 | 0.6635 | 9.207 | 20 | 0 | 100 | 0.0007966 | 0.0005754 |
| diurnal_long_popC_holdout | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | primary | 16 | 6.961 | 0.6052 | 11.5 | 16 | 0 | 100 | 0 | 0 |
| diurnal_long_popC_holdout | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 7.097 | 0.6766 | 10.49 | 13 | 0 | 100 | 0 | 0 |
| diurnal_long_popC_holdout | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 6.961 | 0.6052 | 11.5 | 16 | 0 | 100 | 0 | 0 |
| diurnal_long_popC_holdout | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | 6.942 | 0.5688 | 12.2 | 17 | 0 | 100 | 0 | 0 |
