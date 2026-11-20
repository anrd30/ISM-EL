# Post-registration: residual Q retrained per population, `profile_diurnal_long_popD_holdout.csv`

Run after the pre-registered evaluation; it does not replace it. Gains are paired by scenario seed; learned models with several seeds are averaged within each scenario first; `sem` is across scenarios.

| profile | definition | n_forced | n | forced_seeds |
|---|---|---|---|---|
| diurnal_long_popD_holdout | primary | 4 | 20 | 3 9 17 19 |
| diurnal_long_popD_holdout | clip_0.01 | 7 | 20 | 2 3 9 16 17 18 19 |
| diurnal_long_popD_holdout | clip_0.25 | 4 | 20 | 3 9 17 19 |
| diurnal_long_popD_holdout | fixed20_cost | 3 | 20 | 9 17 19 |

| profile | treatment | reference | block | n | mean_gain | sem | t | better | worse | percent_of_optimum | viol_frac_diff | viol_frac_diff_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_popD_holdout | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | all | 20 | -5.023 | 3.227 | -1.556 | 3 | 17 | -24.9 | 0.0006819 | 0.000574 |
| diurnal_long_popD_holdout | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | primary | 16 | -5.929 | 4.025 | -1.473 | 3 | 13 | -27.04 | 0.0006406 | 0.0006693 |
| diurnal_long_popD_holdout | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | clip_0.01 | 13 | -8.367 | 4.484 | -1.866 | 2 | 11 | -39.05 | 0.001098 | 0.0007358 |
| diurnal_long_popD_holdout | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | clip_0.25 | 16 | -5.929 | 4.025 | -1.473 | 3 | 13 | -27.04 | 0.0006406 | 0.0006693 |
| diurnal_long_popD_holdout | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | fixed20_cost | 17 | -5.774 | 3.784 | -1.526 | 3 | 14 | -24.61 | 0.0005825 | 0.0006314 |
| diurnal_long_popD_holdout | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | all | 20 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_popD_holdout | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | primary | 16 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_popD_holdout | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_popD_holdout | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_popD_holdout | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_popD_holdout | predictive_controller (history) | risk_aware_rule_based | all | 20 | 2.912 | 2.506 | 1.162 | 14 | 6 | 14.44 | 0.0002236 | 0.0006466 |
| diurnal_long_popD_holdout | predictive_controller (history) | risk_aware_rule_based | primary | 16 | 1.476 | 2.411 | 0.6121 | 10 | 6 | 6.729 | -0.0001837 | 0.0003888 |
| diurnal_long_popD_holdout | predictive_controller (history) | risk_aware_rule_based | clip_0.01 | 13 | 1.039 | 2.86 | 0.3634 | 8 | 5 | 4.85 | -9.54e-05 | 0.0004621 |
| diurnal_long_popD_holdout | predictive_controller (history) | risk_aware_rule_based | clip_0.25 | 16 | 1.476 | 2.411 | 0.6121 | 10 | 6 | 6.729 | -0.0001837 | 0.0003888 |
| diurnal_long_popD_holdout | predictive_controller (history) | risk_aware_rule_based | fixed20_cost | 17 | 3.363 | 2.948 | 1.141 | 11 | 6 | 14.33 | -0.0005086 | 0.0004888 |
| diurnal_long_popD_holdout | predictive_controller (history) | rule_seasonal_forecast (hand-built) | all | 20 | 7.935 | 3.928 | 2.02 | 15 | 5 | 31.5 | -0.0004582 | 0.0007141 |
| diurnal_long_popD_holdout | predictive_controller (history) | rule_seasonal_forecast (hand-built) | primary | 16 | 7.405 | 4.51 | 1.642 | 11 | 5 | 26.58 | -0.0008243 | 0.0007166 |
| diurnal_long_popD_holdout | predictive_controller (history) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 9.406 | 5.379 | 1.749 | 10 | 3 | 31.57 | -0.001194 | 0.000847 |
| diurnal_long_popD_holdout | predictive_controller (history) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 7.405 | 4.51 | 1.642 | 11 | 5 | 26.58 | -0.0008243 | 0.0007166 |
| diurnal_long_popD_holdout | predictive_controller (history) | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | 9.137 | 4.577 | 1.996 | 12 | 5 | 31.25 | -0.001091 | 0.0007241 |
| diurnal_long_popD_holdout | residual-Q (retrained on its own training split) | risk_aware_rule_based | all | 20 | 10.57 | 2.314 | 4.569 | 14 | 6 | 52.41 | -0.002243 | 0.0003447 |
| diurnal_long_popD_holdout | residual-Q (retrained on its own training split) | risk_aware_rule_based | primary | 16 | 11.86 | 2.474 | 4.794 | 13 | 3 | 54.08 | -0.002183 | 0.0003909 |
| diurnal_long_popD_holdout | residual-Q (retrained on its own training split) | risk_aware_rule_based | clip_0.01 | 13 | 11.09 | 2.962 | 3.744 | 10 | 3 | 51.76 | -0.002017 | 0.0004586 |
| diurnal_long_popD_holdout | residual-Q (retrained on its own training split) | risk_aware_rule_based | clip_0.25 | 16 | 11.86 | 2.474 | 4.794 | 13 | 3 | 54.08 | -0.002183 | 0.0003909 |
| diurnal_long_popD_holdout | residual-Q (retrained on its own training split) | risk_aware_rule_based | fixed20_cost | 17 | 12.54 | 2.421 | 5.179 | 14 | 3 | 53.44 | -0.002327 | 0.0003943 |
| diurnal_long_popD_holdout | residual-Q (retrained on its own training split) | rule_seasonal_forecast (hand-built) | all | 20 | 15.59 | 3.766 | 4.141 | 18 | 2 | 61.9 | -0.002925 | 0.0005768 |
| diurnal_long_popD_holdout | residual-Q (retrained on its own training split) | rule_seasonal_forecast (hand-built) | primary | 16 | 17.79 | 4.318 | 4.119 | 15 | 1 | 63.85 | -0.002824 | 0.0006696 |
| diurnal_long_popD_holdout | residual-Q (retrained on its own training split) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 19.46 | 5.101 | 3.815 | 12 | 1 | 65.3 | -0.003115 | 0.0007903 |
| diurnal_long_popD_holdout | residual-Q (retrained on its own training split) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 17.79 | 4.318 | 4.119 | 15 | 1 | 63.85 | -0.002824 | 0.0006696 |
| diurnal_long_popD_holdout | residual-Q (retrained on its own training split) | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | 18.31 | 4.09 | 4.478 | 16 | 1 | 62.64 | -0.002909 | 0.0006348 |
| diurnal_long_popD_holdout | residual-Q (zero-shot transfer from A) | risk_aware_rule_based | all | 20 | -17.25 | 2.882 | -5.987 | 0 | 20 | -85.54 | 0.00401 | 0.000823 |
| diurnal_long_popD_holdout | residual-Q (zero-shot transfer from A) | risk_aware_rule_based | primary | 16 | -19.53 | 3.335 | -5.858 | 0 | 16 | -89.08 | 0.003354 | 0.0005729 |
| diurnal_long_popD_holdout | residual-Q (zero-shot transfer from A) | risk_aware_rule_based | clip_0.01 | 13 | -21.25 | 3.92 | -5.422 | 0 | 13 | -99.19 | 0.003694 | 0.0006611 |
| diurnal_long_popD_holdout | residual-Q (zero-shot transfer from A) | risk_aware_rule_based | clip_0.25 | 16 | -19.53 | 3.335 | -5.858 | 0 | 16 | -89.08 | 0.003354 | 0.0005729 |
| diurnal_long_popD_holdout | residual-Q (zero-shot transfer from A) | risk_aware_rule_based | fixed20_cost | 17 | -18.55 | 3.283 | -5.649 | 0 | 17 | -79.06 | 0.003152 | 0.0005748 |
| diurnal_long_popD_holdout | residual-Q (zero-shot transfer from A) | rule_seasonal_forecast (hand-built) | all | 20 | -12.23 | 2.843 | -4.302 | 2 | 18 | -48.55 | 0.003328 | 0.0006592 |
| diurnal_long_popD_holdout | residual-Q (zero-shot transfer from A) | rule_seasonal_forecast (hand-built) | primary | 16 | -13.6 | 3.427 | -3.969 | 1 | 15 | -48.83 | 0.002713 | 0.0005225 |
| diurnal_long_popD_holdout | residual-Q (zero-shot transfer from A) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | -12.89 | 3.98 | -3.238 | 1 | 12 | -43.25 | 0.002596 | 0.000613 |
| diurnal_long_popD_holdout | residual-Q (zero-shot transfer from A) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | -13.6 | 3.427 | -3.969 | 1 | 15 | -48.83 | 0.002713 | 0.0005225 |
| diurnal_long_popD_holdout | residual-Q (zero-shot transfer from A) | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | -12.77 | 3.324 | -3.842 | 2 | 15 | -43.69 | 0.002569 | 0.0005115 |
| diurnal_long_popD_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | all | 20 | 18.93 | 2.867 | 6.602 | 19 | 1 | 93.85 | 0.0006692 | 0.002275 |
| diurnal_long_popD_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | primary | 16 | 20.66 | 2.319 | 8.91 | 16 | 0 | 94.23 | -0.002797 | 0.0003438 |
| diurnal_long_popD_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | clip_0.01 | 13 | 20.11 | 2.738 | 7.343 | 13 | 0 | 93.85 | -0.002728 | 0.0004046 |
| diurnal_long_popD_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | clip_0.25 | 16 | 20.66 | 2.319 | 8.91 | 16 | 0 | 94.23 | -0.002797 | 0.0003438 |
| diurnal_long_popD_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | fixed20_cost | 17 | 22.19 | 2.658 | 8.348 | 17 | 0 | 94.55 | -0.003081 | 0.0004298 |
| diurnal_long_popD_holdout | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | all | 20 | 23.95 | 4.457 | 5.374 | 20 | 0 | 95.07 | -1.267e-05 | 0.002209 |
| diurnal_long_popD_holdout | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | primary | 16 | 26.59 | 4.663 | 5.703 | 16 | 0 | 95.46 | -0.003437 | 0.000728 |
| diurnal_long_popD_holdout | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 28.48 | 5.402 | 5.272 | 13 | 0 | 95.57 | -0.003826 | 0.0008423 |
| diurnal_long_popD_holdout | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 26.59 | 4.663 | 5.703 | 16 | 0 | 95.46 | -0.003437 | 0.000728 |
| diurnal_long_popD_holdout | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | 27.96 | 4.588 | 6.094 | 17 | 0 | 95.63 | -0.003663 | 0.0007201 |
| diurnal_long_popD_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | all | 20 | 20.17 | 2.872 | 7.022 | 20 | 0 | 100 | -0.0004431 | 0.001633 |
| diurnal_long_popD_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | primary | 16 | 21.93 | 2.297 | 9.547 | 16 | 0 | 100 | -0.002797 | 0.0003438 |
| diurnal_long_popD_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | clip_0.01 | 13 | 21.43 | 2.716 | 7.889 | 13 | 0 | 100 | -0.002728 | 0.0004046 |
| diurnal_long_popD_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | clip_0.25 | 16 | 21.93 | 2.297 | 9.547 | 16 | 0 | 100 | -0.002797 | 0.0003438 |
| diurnal_long_popD_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | fixed20_cost | 17 | 23.46 | 2.647 | 8.862 | 17 | 0 | 100 | -0.003081 | 0.0004298 |
| diurnal_long_popD_holdout | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | all | 20 | 25.19 | 4.474 | 5.631 | 20 | 0 | 100 | -0.001125 | 0.001591 |
| diurnal_long_popD_holdout | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | primary | 16 | 27.86 | 4.67 | 5.965 | 16 | 0 | 100 | -0.003437 | 0.000728 |
| diurnal_long_popD_holdout | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 29.79 | 5.414 | 5.503 | 13 | 0 | 100 | -0.003826 | 0.0008423 |
| diurnal_long_popD_holdout | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 27.86 | 4.67 | 5.965 | 16 | 0 | 100 | -0.003437 | 0.000728 |
| diurnal_long_popD_holdout | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | 29.24 | 4.599 | 6.357 | 17 | 0 | 100 | -0.003663 | 0.0007201 |
