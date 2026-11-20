# Vendor-holdout evaluation, `pooled_vendor_fleet,sigma=0.25`, history_rows=576

Scenarios are identical for every arm. Gains are paired by scenario seed; `sem` is across scenarios. `primary` is the registered definition of controllable (the safety filter raises a `fixed_20` rollout on fewer than 10% of the intervals).

| profile | definition | n_forced | n | forced_seeds |
|---|---|---|---|---|
| diurnal_long_holdout | primary | 8 | 20 | 3 4 7 9 11 13 15 19 |
| diurnal_long_holdout | clip_0.01 | 9 | 20 | 3 4 7 9 11 13 15 17 19 |
| diurnal_long_holdout | clip_0.25 | 5 | 20 | 7 9 11 15 19 |
| diurnal_long_holdout | fixed20_cost | 3 | 20 | 7 9 19 |

## Summary, all scenarios

| profile | policy | workload_synthetic | n | completed | horizon_expired | terminal_failure | completion_h | lat_deg_mean | lat_deg_p95 | viol_frac | lat_budget_met | p_fail_mean | p_fail_max | risk_viol | clip_frac | deliv_mbps | stall_frac | reward | reward_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_holdout | fixed_20 | false | 20 | 20 | 0 | 0 | 138.9 | 0.0922 | 0.68 | 0.01845 | 0.85 | 0.00192 | 0.003539 | 0.05 | 0.1537 | 27.1 | 0 | -383.3 | 116.3 |
| diurnal_long_holdout | fixed_50 | false | 20 | 20 | 0 | 0 | 63.32 | 0.1243 | 0.85 | 0.02488 | 0.85 | 0.001272 | 0.00232 | 0.05 | 0.07758 | 53.39 | 0 | -292.6 | 122.5 |
| diurnal_long_holdout | fixed_80 | false | 20 | 20 | 0 | 0 | 41.07 | 0.2057 | 1.119 | 0.05958 | 0.85 | 0.000971 | 0.001835 | 0.05 | 0.03981 | 81.36 | 0 | -427.1 | 114.5 |
| diurnal_long_holdout | fixed_120 | false | 20 | 20 | 0 | 0 | 29.26 | 3.483 | 5.667 | 0.6722 | 0 | 0.0007323 | 0.001458 | 0.05 | 0 | 114.1 | 0 | -3621 | 90.47 |
| diurnal_long_holdout | risk_aware_rule_based | false | 20 | 20 | 0 | 0 | 43.67 | 0.162 | 0.8516 | 0.03201 | 0.85 | 0.00103 | 0.002182 | 0.05 | 0.001991 | 76.69 | 0 | -275.4 | 123.1 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | false | 20 | 20 | 0 | 0 | 40.53 | 0.1732 | 0.8577 | 0.03426 | 0.85 | 0.001042 | 0.002182 | 0.05 | 0.002143 | 82.63 | 0 | -271.6 | 123.2 |
| diurnal_long_holdout | rule+history | false | 20 | 20 | 0 | 0 | 44 | 0.1606 | 0.8515 | 0.03135 | 0.85 | 0.001041 | 0.00222 | 0.05 | 0.001991 | 76.1 | 0 | -271.6 | 123.5 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | false | 20 | 20 | 0 | 0 | 37.15 | 0.1766 | 0.8663 | 0.03457 | 0.85 | 0.0009189 | 0.002175 | 0.05 | 0.002211 | 89.98 | 0 | -262.3 | 123.9 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | false | 20 | 20 | 0 | 0 | 41.14 | 0.1722 | 0.8535 | 0.03365 | 0.85 | 0.00106 | 0.002832 | 0.05 | 0.002138 | 81.43 | 0 | -269.6 | 123.4 |
| diurnal_long_holdout | predictive_controller (history) | false | 20 | 20 | 0 | 0 | 37.15 | 0.1766 | 0.8663 | 0.03457 | 0.85 | 0.0009189 | 0.002175 | 0.05 | 0.002211 | 89.98 | 0 | -262.3 | 123.9 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | false | 20 | 20 | 0 | 0 | 36.91 | 0.1785 | 0.8846 | 0.03472 | 0.85 | 0.000914 | 0.002174 | 0.05 | 0.002222 | 90.6 | 0 | -262.4 | 123.9 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | false | 20 | 20 | 0 | 0 | 37.56 | 0.1735 | 0.8518 | 0.03422 | 0.85 | 0.0009282 | 0.00218 | 0.05 | 0.09864 | 88.93 | 0 | -261.3 | 123.9 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Paired gains by block

| profile | treatment | reference | block | n | mean_gain | sem | t | better | worse | percent_of_optimum | viol_frac_diff | viol_frac_diff_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | all | 20 | 3.835 | 0.1912 | 20.06 | 20 | 0 | 27.28 | 0.002254 | 0.001268 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | primary | 12 | 3.879 | 0.2247 | 17.26 | 12 | 0 | 24.5 | 7.423e-05 | 2.344e-05 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | clip_0.01 | 11 | 3.832 | 0.2407 | 15.92 | 11 | 0 | 23.57 | 8.098e-05 | 2.458e-05 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | clip_0.25 | 15 | 3.914 | 0.1928 | 20.3 | 15 | 0 | 25.07 | 6.911e-05 | 2.05e-05 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | fixed20_cost | 17 | 3.968 | 0.1936 | 20.5 | 17 | 0 | 25.56 | 9.694e-05 | 3.707e-05 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | all | 20 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | primary | 12 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | clip_0.01 | 11 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | clip_0.25 | 15 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_holdout | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_holdout | rule+history | risk_aware_rule_based | all | 20 | 3.848 | 1.207 | 3.188 | 12 | 4 | 27.36 | -0.000658 | 0.000205 |
| diurnal_long_holdout | rule+history | risk_aware_rule_based | primary | 12 | 5.423 | 1.648 | 3.29 | 9 | 3 | 34.25 | -0.0009407 | 0.0002838 |
| diurnal_long_holdout | rule+history | risk_aware_rule_based | clip_0.01 | 11 | 5.927 | 1.719 | 3.448 | 9 | 2 | 36.46 | -0.001026 | 0.0002965 |
| diurnal_long_holdout | rule+history | risk_aware_rule_based | clip_0.25 | 15 | 5.144 | 1.468 | 3.503 | 12 | 3 | 32.95 | -0.0008751 | 0.0002502 |
| diurnal_long_holdout | rule+history | risk_aware_rule_based | fixed20_cost | 17 | 4.527 | 1.358 | 3.335 | 12 | 4 | 29.15 | -0.0007741 | 0.0002304 |
| diurnal_long_holdout | rule+history | rule_seasonal_forecast (hand-built) | all | 20 | 0.01238 | 1.203 | 0.01029 | 7 | 13 | 0.121 | -0.002912 | 0.001226 |
| diurnal_long_holdout | rule+history | rule_seasonal_forecast (hand-built) | primary | 12 | 1.544 | 1.688 | 0.9146 | 6 | 6 | 12.91 | -0.001015 | 0.0003062 |
| diurnal_long_holdout | rule+history | rule_seasonal_forecast (hand-built) | clip_0.01 | 11 | 2.095 | 1.747 | 1.199 | 6 | 5 | 16.86 | -0.001107 | 0.0003199 |
| diurnal_long_holdout | rule+history | rule_seasonal_forecast (hand-built) | clip_0.25 | 15 | 1.23 | 1.472 | 0.8355 | 7 | 8 | 10.52 | -0.0009442 | 0.00027 |
| diurnal_long_holdout | rule+history | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | 0.5584 | 1.375 | 0.4061 | 7 | 10 | 4.831 | -0.0008711 | 0.000244 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | risk_aware_rule_based | all | 20 | 13.15 | 1.445 | 9.099 | 20 | 0 | 93.49 | 0.002564 | 0.001983 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | risk_aware_rule_based | primary | 12 | 14.92 | 1.707 | 8.739 | 12 | 0 | 94.21 | -0.0009407 | 0.0002838 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | risk_aware_rule_based | clip_0.01 | 11 | 15.33 | 1.815 | 8.445 | 11 | 0 | 94.3 | -0.001026 | 0.0002965 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | risk_aware_rule_based | clip_0.25 | 15 | 14.69 | 1.576 | 9.321 | 15 | 0 | 94.1 | -0.0008751 | 0.0002502 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | risk_aware_rule_based | fixed20_cost | 17 | 14.65 | 1.405 | 10.43 | 17 | 0 | 94.35 | -0.0008391 | 0.0002265 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | all | 20 | 9.311 | 1.397 | 6.667 | 20 | 0 | 91.05 | 0.0003101 | 0.0007543 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | primary | 12 | 11.04 | 1.701 | 6.49 | 12 | 0 | 92.34 | -0.001015 | 0.0003062 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | clip_0.01 | 11 | 11.5 | 1.794 | 6.407 | 11 | 0 | 92.54 | -0.001107 | 0.0003199 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | clip_0.25 | 15 | 10.78 | 1.54 | 6.996 | 15 | 0 | 92.13 | -0.0009442 | 0.00027 |
| diurnal_long_holdout | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | 10.68 | 1.392 | 7.671 | 17 | 0 | 92.4 | -0.000936 | 0.0002488 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | risk_aware_rule_based | all | 20 | 5.801 | 1.409 | 4.117 | 19 | 1 | 41.26 | 0.001644 | 0.001284 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | risk_aware_rule_based | primary | 12 | 6.578 | 2.132 | 3.085 | 11 | 1 | 41.54 | -0.0006112 | 0.0003557 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | risk_aware_rule_based | clip_0.01 | 11 | 6.903 | 2.308 | 2.991 | 10 | 1 | 42.47 | -0.0006668 | 0.0003849 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | risk_aware_rule_based | clip_0.25 | 15 | 6.612 | 1.841 | 3.591 | 14 | 1 | 42.36 | -0.0006115 | 0.0003004 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | risk_aware_rule_based | fixed20_cost | 17 | 6.279 | 1.635 | 3.841 | 16 | 1 | 40.44 | -0.0005081 | 0.0002743 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | all | 20 | 1.966 | 1.367 | 1.438 | 8 | 12 | 19.22 | -0.0006093 | 0.0002488 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | primary | 12 | 2.699 | 2.102 | 1.284 | 5 | 7 | 22.58 | -0.0006854 | 0.000375 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | clip_0.01 | 11 | 3.071 | 2.266 | 1.355 | 5 | 6 | 24.72 | -0.0007477 | 0.0004051 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | clip_0.25 | 15 | 2.699 | 1.795 | 1.503 | 6 | 9 | 23.07 | -0.0006806 | 0.0003176 |
| diurnal_long_holdout | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | 2.311 | 1.6 | 1.444 | 6 | 11 | 19.99 | -0.000605 | 0.0002839 |
| diurnal_long_holdout | predictive_controller (history) | risk_aware_rule_based | all | 20 | 13.15 | 1.445 | 9.099 | 20 | 0 | 93.49 | 0.002564 | 0.001983 |
| diurnal_long_holdout | predictive_controller (history) | risk_aware_rule_based | primary | 12 | 14.92 | 1.707 | 8.739 | 12 | 0 | 94.21 | -0.0009407 | 0.0002838 |
| diurnal_long_holdout | predictive_controller (history) | risk_aware_rule_based | clip_0.01 | 11 | 15.33 | 1.815 | 8.445 | 11 | 0 | 94.3 | -0.001026 | 0.0002965 |
| diurnal_long_holdout | predictive_controller (history) | risk_aware_rule_based | clip_0.25 | 15 | 14.69 | 1.576 | 9.321 | 15 | 0 | 94.1 | -0.0008751 | 0.0002502 |
| diurnal_long_holdout | predictive_controller (history) | risk_aware_rule_based | fixed20_cost | 17 | 14.65 | 1.405 | 10.43 | 17 | 0 | 94.35 | -0.0008391 | 0.0002265 |
| diurnal_long_holdout | predictive_controller (history) | rule_seasonal_forecast (hand-built) | all | 20 | 9.311 | 1.397 | 6.667 | 20 | 0 | 91.05 | 0.0003101 | 0.0007543 |
| diurnal_long_holdout | predictive_controller (history) | rule_seasonal_forecast (hand-built) | primary | 12 | 11.04 | 1.701 | 6.49 | 12 | 0 | 92.34 | -0.001015 | 0.0003062 |
| diurnal_long_holdout | predictive_controller (history) | rule_seasonal_forecast (hand-built) | clip_0.01 | 11 | 11.5 | 1.794 | 6.407 | 11 | 0 | 92.54 | -0.001107 | 0.0003199 |
| diurnal_long_holdout | predictive_controller (history) | rule_seasonal_forecast (hand-built) | clip_0.25 | 15 | 10.78 | 1.54 | 6.996 | 15 | 0 | 92.13 | -0.0009442 | 0.00027 |
| diurnal_long_holdout | predictive_controller (history) | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | 10.68 | 1.392 | 7.671 | 17 | 0 | 92.4 | -0.000936 | 0.0002488 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | all | 20 | 12.96 | 1.436 | 9.021 | 20 | 0 | 92.13 | 0.002715 | 0.002074 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | primary | 12 | 14.69 | 1.698 | 8.655 | 12 | 0 | 92.79 | -0.0009407 | 0.0002838 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | clip_0.01 | 11 | 15.09 | 1.808 | 8.345 | 11 | 0 | 92.82 | -0.001026 | 0.0002965 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | clip_0.25 | 15 | 14.47 | 1.57 | 9.217 | 15 | 0 | 92.7 | -0.0008751 | 0.0002502 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | fixed20_cost | 17 | 14.44 | 1.401 | 10.31 | 17 | 0 | 92.99 | -0.0008362 | 0.0002263 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | all | 20 | 9.12 | 1.387 | 6.574 | 20 | 0 | 89.18 | 0.0004614 | 0.0008424 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | primary | 12 | 10.81 | 1.69 | 6.398 | 12 | 0 | 90.45 | -0.001015 | 0.0003062 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.01 | 11 | 11.26 | 1.787 | 6.301 | 11 | 0 | 90.61 | -0.001107 | 0.0003199 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.25 | 15 | 10.56 | 1.533 | 6.886 | 15 | 0 | 90.25 | -0.0009442 | 0.00027 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | 10.47 | 1.388 | 7.543 | 17 | 0 | 90.58 | -0.0009331 | 0.0002483 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | all | 20 | 14.06 | 1.426 | 9.863 | 20 | 0 | 100 | 0.00221 | 0.001829 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | primary | 12 | 15.83 | 1.694 | 9.345 | 12 | 0 | 100 | -0.0009407 | 0.0002838 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | clip_0.01 | 11 | 16.26 | 1.798 | 9.042 | 11 | 0 | 100 | -0.001026 | 0.0002965 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | clip_0.25 | 15 | 15.61 | 1.558 | 10.02 | 15 | 0 | 100 | -0.0008751 | 0.0002502 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | fixed20_cost | 17 | 15.53 | 1.392 | 11.16 | 17 | 0 | 100 | -0.000842 | 0.0002268 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | all | 20 | 10.23 | 1.381 | 7.405 | 20 | 0 | 100 | -4.334e-05 | 0.000618 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | primary | 12 | 11.96 | 1.694 | 7.056 | 12 | 0 | 100 | -0.001015 | 0.0003062 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.01 | 11 | 12.42 | 1.784 | 6.965 | 11 | 0 | 100 | -0.001107 | 0.0003199 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.25 | 15 | 11.7 | 1.528 | 7.657 | 15 | 0 | 100 | -0.0009442 | 0.00027 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | 11.56 | 1.385 | 8.346 | 17 | 0 | 100 | -0.0009389 | 0.0002494 |
