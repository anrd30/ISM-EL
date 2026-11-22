# Predictive controller and comparators, `eval-fleet,sigma=0.25`, history_rows=576

Scenarios are identical for every arm. Gains are paired by scenario seed; learned models with several seeds are averaged within each scenario first; `sem` is across scenarios. `primary` is the registered definition of controllable (the safety filter raises a `fixed_20` rollout on fewer than 10% of the intervals); the other blocks are the registered sensitivity variants.

| profile | definition | n_forced | n | forced_seeds |
|---|---|---|---|---|
| diurnal_long_altB | primary | 4 | 20 | 3 9 17 19 |
| diurnal_long_altB | clip_0.01 | 7 | 20 | 2 3 9 16 17 18 19 |
| diurnal_long_altB | clip_0.25 | 4 | 20 | 3 9 17 19 |
| diurnal_long_altB | fixed20_cost | 3 | 20 | 9 17 19 |

## Summary, all scenarios

| profile | policy | workload_synthetic | n | completed | horizon_expired | terminal_failure | completion_h | lat_deg_mean | lat_deg_p95 | viol_frac | lat_budget_met | p_fail_mean | p_fail_max | risk_viol | clip_frac | deliv_mbps | stall_frac | reward | reward_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_altB | fixed_20 | false | 20 | 20 | 0 | 0 | 144.8 | 0.06816 | 0.5667 | 0.01807 | 0.9 | 0.001985 | 0.003874 | 0.1 | 0.1224 | 26.96 | 0 | -328.1 | 93.01 |
| diurnal_long_altB | fixed_50 | false | 20 | 20 | 0 | 0 | 63.02 | 0.08581 | 0.5672 | 0.02283 | 0.9 | 0.00123 | 0.002381 | 0.1 | 0.0812 | 54.23 | 0 | -230 | 99.57 |
| diurnal_long_altB | fixed_80 | false | 20 | 20 | 0 | 0 | 40.77 | 0.168 | 0.9012 | 0.06683 | 0.6 | 0.0009779 | 0.001919 | 0.1 | 0.05723 | 82.17 | 0 | -430.1 | 101 |
| diurnal_long_altB | fixed_120 | false | 20 | 20 | 0 | 0 | 29 | 2.394 | 5.667 | 0.5511 | 0 | 0.0007964 | 0.0016 | 0.1 | 0 | 115.2 | 0 | -2797 | 138.8 |
| diurnal_long_altB | risk_aware_rule_based | false | 20 | 20 | 0 | 0 | 39.68 | 0.1179 | 0.5721 | 0.03311 | 0.9 | 0.001016 | 0.002224 | 0.1 | 0.01843 | 84.87 | 0 | -227.7 | 99.74 |
| diurnal_long_altB | rule_seasonal_forecast (hand-built) | false | 20 | 20 | 0 | 0 | 39.36 | 0.1177 | 0.5805 | 0.03275 | 0.9 | 0.001015 | 0.002224 | 0.1 | 0.01858 | 85.52 | 0 | -223.7 | 100.1 |
| diurnal_long_altB | rule+history | false | 20 | 20 | 0 | 0 | 39.66 | 0.118 | 0.5721 | 0.03308 | 0.9 | 0.001025 | 0.002245 | 0.1 | 0.01848 | 84.89 | 0 | -227.1 | 99.78 |
| diurnal_long_altB | predictive_controller[seasonal]+history | false | 20 | 20 | 0 | 0 | 41.33 | 0.1078 | 0.5714 | 0.03038 | 0.9 | 0.001045 | 0.002983 | 0.1 | 0.02263 | 81.41 | 0 | -219.1 | 101.2 |
| diurnal_long_altB | predictive_controller[adaptive] (no history) | false | 20 | 20 | 0 | 0 | 40.96 | 0.1117 | 0.5754 | 0.03148 | 0.9 | 0.001034 | 0.002842 | 0.1 | 0.02028 | 82.25 | 0 | -220.1 | 100.7 |
| diurnal_long_altB | predictive_controller (history) | false | 20 | 20 | 0 | 0 | 40.58 | 0.1108 | 0.5714 | 0.03103 | 0.9 | 0.001029 | 0.002677 | 0.1 | 0.02015 | 82.95 | 0 | -216.6 | 101.3 |
| diurnal_long_altB | ORACLE_lookahead (not deployable) | false | 20 | 20 | 0 | 0 | 35.51 | 0.1155 | 0.6065 | 0.02988 | 0.9 | 0.0009437 | 0.002201 | 0.1 | 0.0139 | 94.33 | 0 | -199 | 101.2 |
| diurnal_long_altB | DP_hindsight_optimum (not deployable) | false | 20 | 20 | 0 | 0 | 36.08 | 0.1104 | 0.5714 | 0.02964 | 0.9 | 0.0009572 | 0.002235 | 0.1 | 0.113 | 92.76 | 0 | -197.8 | 101.2 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## Paired gains by block

| profile | treatment | reference | block | n | mean_gain | sem | t | better | worse | percent_of_optimum | viol_frac_diff | viol_frac_diff_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_altB | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | all | 20 | 3.931 | 2.136 | 1.84 | 15 | 5 | 13.18 | -0.0003647 | 0.0003898 |
| diurnal_long_altB | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | primary | 16 | 6.441 | 1.513 | 4.256 | 14 | 2 | 18.73 | -0.0009773 | 0.0002548 |
| diurnal_long_altB | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | clip_0.01 | 13 | 6.139 | 1.578 | 3.89 | 12 | 1 | 18.71 | -0.0008655 | 0.0002835 |
| diurnal_long_altB | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | clip_0.25 | 16 | 6.441 | 1.513 | 4.256 | 14 | 2 | 18.73 | -0.0009773 | 0.0002548 |
| diurnal_long_altB | rule_seasonal_forecast (hand-built) | risk_aware_rule_based | fixed20_cost | 17 | 4.636 | 2.298 | 2.017 | 14 | 3 | 13.83 | -0.0006788 | 0.0003826 |
| diurnal_long_altB | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | all | 20 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_altB | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | primary | 16 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_altB | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_altB | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_altB | rule_seasonal_forecast (hand-built) | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | 0 | 0 | n/a | 0 | 0 | 0 | 0 | 0 |
| diurnal_long_altB | rule+history | risk_aware_rule_based | all | 20 | 0.6098 | 1.285 | 0.4745 | 11 | 8 | 2.045 | -3.573e-05 | 0.0002637 |
| diurnal_long_altB | rule+history | risk_aware_rule_based | primary | 16 | 0.7796 | 1.614 | 0.4829 | 10 | 6 | 2.266 | -0.0001483 | 0.0003101 |
| diurnal_long_altB | rule+history | risk_aware_rule_based | clip_0.01 | 13 | 0.02968 | 1.839 | 0.01614 | 8 | 5 | 0.09048 | -1.586e-05 | 0.0003391 |
| diurnal_long_altB | rule+history | risk_aware_rule_based | clip_0.25 | 16 | 0.7796 | 1.614 | 0.4829 | 10 | 6 | 2.266 | -0.0001483 | 0.0003101 |
| diurnal_long_altB | rule+history | risk_aware_rule_based | fixed20_cost | 17 | 0.722 | 1.517 | 0.4758 | 10 | 7 | 2.153 | -0.0001401 | 0.0002914 |
| diurnal_long_altB | rule+history | rule_seasonal_forecast (hand-built) | all | 20 | -3.321 | 2.263 | -1.468 | 5 | 15 | -12.83 | 0.0003289 | 0.0003758 |
| diurnal_long_altB | rule+history | rule_seasonal_forecast (hand-built) | primary | 16 | -5.662 | 1.864 | -3.038 | 2 | 14 | -20.25 | 0.000829 | 0.0003192 |
| diurnal_long_altB | rule+history | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | -6.109 | 2.054 | -2.974 | 1 | 12 | -22.91 | 0.0008496 | 0.0003827 |
| diurnal_long_altB | rule+history | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | -5.662 | 1.864 | -3.038 | 2 | 14 | -20.25 | 0.000829 | 0.0003192 |
| diurnal_long_altB | rule+history | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | -3.914 | 2.474 | -1.582 | 3 | 14 | -13.55 | 0.0005387 | 0.0004173 |
| diurnal_long_altB | predictive_controller[seasonal]+history | risk_aware_rule_based | all | 20 | 8.608 | 3.562 | 2.416 | 13 | 7 | 28.87 | -0.002738 | 0.0009913 |
| diurnal_long_altB | predictive_controller[seasonal]+history | risk_aware_rule_based | primary | 16 | 10.98 | 4.006 | 2.74 | 10 | 6 | 31.91 | -0.002059 | 0.0005778 |
| diurnal_long_altB | predictive_controller[seasonal]+history | risk_aware_rule_based | clip_0.01 | 13 | 9.283 | 4.241 | 2.189 | 8 | 5 | 28.29 | -0.001663 | 0.0005665 |
| diurnal_long_altB | predictive_controller[seasonal]+history | risk_aware_rule_based | clip_0.25 | 16 | 10.98 | 4.006 | 2.74 | 10 | 6 | 31.91 | -0.002059 | 0.0005778 |
| diurnal_long_altB | predictive_controller[seasonal]+history | risk_aware_rule_based | fixed20_cost | 17 | 11.16 | 3.768 | 2.963 | 11 | 6 | 33.29 | -0.002061 | 0.0005428 |
| diurnal_long_altB | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | all | 20 | 4.677 | 3.211 | 1.456 | 11 | 9 | 18.07 | -0.002374 | 0.001161 |
| diurnal_long_altB | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | primary | 16 | 4.537 | 3.149 | 1.441 | 9 | 7 | 16.23 | -0.001082 | 0.0004475 |
| diurnal_long_altB | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 3.144 | 3.524 | 0.8921 | 7 | 6 | 11.79 | -0.0007974 | 0.0004636 |
| diurnal_long_altB | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 4.537 | 3.149 | 1.441 | 9 | 7 | 16.23 | -0.001082 | 0.0004475 |
| diurnal_long_altB | predictive_controller[seasonal]+history | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | 6.527 | 3.565 | 1.831 | 10 | 7 | 22.59 | -0.001383 | 0.000517 |
| diurnal_long_altB | predictive_controller[adaptive] (no history) | risk_aware_rule_based | all | 20 | 7.512 | 3.429 | 2.191 | 11 | 9 | 25.19 | -0.00163 | 0.0006316 |
| diurnal_long_altB | predictive_controller[adaptive] (no history) | risk_aware_rule_based | primary | 16 | 9.078 | 4.045 | 2.245 | 10 | 6 | 26.39 | -0.001762 | 0.0006531 |
| diurnal_long_altB | predictive_controller[adaptive] (no history) | risk_aware_rule_based | clip_0.01 | 13 | 6.074 | 4.29 | 1.416 | 7 | 6 | 18.51 | -0.001158 | 0.0006557 |
| diurnal_long_altB | predictive_controller[adaptive] (no history) | risk_aware_rule_based | clip_0.25 | 16 | 9.078 | 4.045 | 2.245 | 10 | 6 | 26.39 | -0.001762 | 0.0006531 |
| diurnal_long_altB | predictive_controller[adaptive] (no history) | risk_aware_rule_based | fixed20_cost | 17 | 9.478 | 3.82 | 2.481 | 11 | 6 | 28.27 | -0.001782 | 0.0006138 |
| diurnal_long_altB | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | all | 20 | 3.581 | 3.469 | 1.032 | 13 | 7 | 13.83 | -0.001266 | 0.0007069 |
| diurnal_long_altB | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | primary | 16 | 2.637 | 3.262 | 0.8084 | 10 | 6 | 9.433 | -0.0007848 | 0.0005532 |
| diurnal_long_altB | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | -0.06497 | 3.514 | -0.01849 | 7 | 6 | -0.2436 | -0.000293 | 0.0005812 |
| diurnal_long_altB | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 2.637 | 3.262 | 0.8084 | 10 | 6 | 9.433 | -0.0007848 | 0.0005532 |
| diurnal_long_altB | predictive_controller[adaptive] (no history) | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | 4.842 | 3.775 | 1.283 | 11 | 6 | 16.76 | -0.001103 | 0.0006095 |
| diurnal_long_altB | predictive_controller (history) | risk_aware_rule_based | all | 20 | 11.02 | 3.441 | 3.204 | 14 | 6 | 36.98 | -0.002081 | 0.0004806 |
| diurnal_long_altB | predictive_controller (history) | risk_aware_rule_based | primary | 16 | 13.84 | 3.764 | 3.676 | 12 | 4 | 40.22 | -0.002338 | 0.0005758 |
| diurnal_long_altB | predictive_controller (history) | risk_aware_rule_based | clip_0.01 | 13 | 12.43 | 4.067 | 3.056 | 10 | 3 | 37.88 | -0.002044 | 0.0005723 |
| diurnal_long_altB | predictive_controller (history) | risk_aware_rule_based | clip_0.25 | 16 | 13.84 | 3.764 | 3.676 | 12 | 4 | 40.22 | -0.002338 | 0.0005758 |
| diurnal_long_altB | predictive_controller (history) | risk_aware_rule_based | fixed20_cost | 17 | 13.89 | 3.536 | 3.927 | 13 | 4 | 41.41 | -0.002324 | 0.000541 |
| diurnal_long_altB | predictive_controller (history) | rule_seasonal_forecast (hand-built) | all | 20 | 7.094 | 3.114 | 2.278 | 13 | 7 | 27.4 | -0.001716 | 0.0004784 |
| diurnal_long_altB | predictive_controller (history) | rule_seasonal_forecast (hand-built) | primary | 16 | 7.395 | 2.995 | 2.469 | 11 | 5 | 26.45 | -0.001361 | 0.0004579 |
| diurnal_long_altB | predictive_controller (history) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 6.29 | 3.437 | 1.83 | 8 | 5 | 23.59 | -0.001178 | 0.0004868 |
| diurnal_long_altB | predictive_controller (history) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 7.395 | 2.995 | 2.469 | 11 | 5 | 26.45 | -0.001361 | 0.0004579 |
| diurnal_long_altB | predictive_controller (history) | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | 9.25 | 3.37 | 2.745 | 12 | 5 | 32.01 | -0.001645 | 0.0005157 |
| diurnal_long_altB | ORACLE_lookahead (not deployable) | risk_aware_rule_based | all | 20 | 28.68 | 3.143 | 9.125 | 20 | 0 | 96.2 | -0.003232 | 0.0009835 |
| diurnal_long_altB | ORACLE_lookahead (not deployable) | risk_aware_rule_based | primary | 16 | 33.16 | 2.716 | 12.21 | 16 | 0 | 96.41 | -0.004845 | 0.0004823 |
| diurnal_long_altB | ORACLE_lookahead (not deployable) | risk_aware_rule_based | clip_0.01 | 13 | 31.59 | 3.195 | 9.887 | 13 | 0 | 96.27 | -0.004525 | 0.0005557 |
| diurnal_long_altB | ORACLE_lookahead (not deployable) | risk_aware_rule_based | clip_0.25 | 16 | 33.16 | 2.716 | 12.21 | 16 | 0 | 96.41 | -0.004845 | 0.0004823 |
| diurnal_long_altB | ORACLE_lookahead (not deployable) | risk_aware_rule_based | fixed20_cost | 17 | 32.3 | 2.695 | 11.98 | 17 | 0 | 96.33 | -0.004683 | 0.000481 |
| diurnal_long_altB | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | all | 20 | 24.75 | 2.381 | 10.4 | 20 | 0 | 95.62 | -0.002868 | 0.000752 |
| diurnal_long_altB | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | primary | 16 | 26.72 | 1.939 | 13.78 | 16 | 0 | 95.59 | -0.003867 | 0.0003688 |
| diurnal_long_altB | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 25.45 | 2.083 | 12.22 | 13 | 0 | 95.42 | -0.00366 | 0.000422 |
| diurnal_long_altB | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 26.72 | 1.939 | 13.78 | 16 | 0 | 95.59 | -0.003867 | 0.0003688 |
| diurnal_long_altB | ORACLE_lookahead (not deployable) | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | 27.66 | 2.048 | 13.5 | 17 | 0 | 95.74 | -0.004004 | 0.0003726 |
| diurnal_long_altB | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | all | 20 | 29.82 | 3.192 | 9.342 | 20 | 0 | 100 | -0.003476 | 0.0008489 |
| diurnal_long_altB | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | primary | 16 | 34.4 | 2.743 | 12.54 | 16 | 0 | 100 | -0.004845 | 0.0004823 |
| diurnal_long_altB | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | clip_0.01 | 13 | 32.81 | 3.226 | 10.17 | 13 | 0 | 100 | -0.004525 | 0.0005557 |
| diurnal_long_altB | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | clip_0.25 | 16 | 34.4 | 2.743 | 12.54 | 16 | 0 | 100 | -0.004845 | 0.0004823 |
| diurnal_long_altB | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | fixed20_cost | 17 | 33.53 | 2.72 | 12.33 | 17 | 0 | 100 | -0.004683 | 0.000481 |
| diurnal_long_altB | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | all | 20 | 25.88 | 2.423 | 10.68 | 20 | 0 | 100 | -0.003112 | 0.0006193 |
| diurnal_long_altB | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | primary | 16 | 27.96 | 1.956 | 14.29 | 16 | 0 | 100 | -0.003867 | 0.0003688 |
| diurnal_long_altB | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.01 | 13 | 26.67 | 2.122 | 12.57 | 13 | 0 | 100 | -0.00366 | 0.000422 |
| diurnal_long_altB | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | clip_0.25 | 16 | 27.96 | 1.956 | 14.29 | 16 | 0 | 100 | -0.003867 | 0.0003688 |
| diurnal_long_altB | DP_hindsight_optimum (not deployable) | rule_seasonal_forecast (hand-built) | fixed20_cost | 17 | 28.89 | 2.062 | 14.01 | 17 | 0 | 100 | -0.004004 | 0.0003726 |
