# PPO with and without the marginal-cost dual update, against the predictive controller and the optimum, `train-fleet,sigma=0.25`

| profile | treatment | reference | block | n | mean_gain | sem | t | better | worse | percent_of_optimum | viol_frac_diff | viol_frac_diff_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_holdout | predictive_controller (history) | risk_aware_rule_based | all | 20 | 12.44 | 1.481 | 8.402 | 20 | 0 | 93.88 | 0.001933 | 0.001766 |
| diurnal_long_holdout | predictive_controller (history) | risk_aware_rule_based | primary | 15 | 14.55 | 1.51 | 9.635 | 15 | 0 | 93.92 | -0.0008848 | 0.0002529 |
| diurnal_long_holdout | predictive_controller (history) | risk_aware_rule_based | clip_0.01 | 12 | 13.74 | 1.632 | 8.419 | 12 | 0 | 93.5 | -0.0007953 | 0.0002838 |
| diurnal_long_holdout | predictive_controller (history) | risk_aware_rule_based | clip_0.25 | 16 | 14.34 | 1.428 | 10.05 | 16 | 0 | 93.99 | -0.0008295 | 0.000243 |
| diurnal_long_holdout | predictive_controller (history) | risk_aware_rule_based | fixed20_cost | 17 | 14.11 | 1.361 | 10.37 | 17 | 0 | 93.94 | -0.0007807 | 0.0002334 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | all | 20 | 12.25 | 1.471 | 8.329 | 20 | 0 | 92.45 | 0.002017 | 0.001833 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | primary | 15 | 14.33 | 1.509 | 9.495 | 15 | 0 | 92.49 | -0.0008848 | 0.0002529 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | clip_0.01 | 12 | 13.51 | 1.629 | 8.294 | 12 | 0 | 91.89 | -0.0007953 | 0.0002838 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | clip_0.25 | 16 | 14.11 | 1.428 | 9.885 | 16 | 0 | 92.48 | -0.0008295 | 0.000243 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | risk_aware_rule_based | fixed20_cost | 17 | 13.89 | 1.359 | 10.22 | 17 | 0 | 92.47 | -0.0007807 | 0.0002334 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | all | 20 | 13.26 | 1.518 | 8.731 | 20 | 0 | 100 | 0.00181 | 0.001685 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | primary | 15 | 15.49 | 1.505 | 10.29 | 15 | 0 | 100 | -0.0008848 | 0.0002529 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | clip_0.01 | 12 | 14.7 | 1.626 | 9.039 | 12 | 0 | 100 | -0.0007953 | 0.0002838 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | clip_0.25 | 16 | 15.26 | 1.427 | 10.7 | 16 | 0 | 100 | -0.0008295 | 0.000243 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | risk_aware_rule_based | fixed20_cost | 17 | 15.02 | 1.361 | 11.04 | 17 | 0 | 100 | -0.0007807 | 0.0002334 |
| diurnal_long_holdout | ppo_clean (deterministic, 3 seeds averaged) | risk_aware_rule_based | all | 20 | -24.61 | 2.055 | -11.98 | 0 | 20 | -185.6 | -0.007851 | 0.004372 |
| diurnal_long_holdout | ppo_clean (deterministic, 3 seeds averaged) | risk_aware_rule_based | primary | 15 | -27.75 | 1.502 | -18.47 | 0 | 15 | -179.1 | -0.0008848 | 0.0002529 |
| diurnal_long_holdout | ppo_clean (deterministic, 3 seeds averaged) | risk_aware_rule_based | clip_0.01 | 12 | -29.23 | 1.442 | -20.27 | 0 | 12 | -198.9 | -0.0007953 | 0.0002838 |
| diurnal_long_holdout | ppo_clean (deterministic, 3 seeds averaged) | risk_aware_rule_based | clip_0.25 | 16 | -27.68 | 1.407 | -19.67 | 0 | 16 | -181.4 | -0.0008295 | 0.000243 |
| diurnal_long_holdout | ppo_clean (deterministic, 3 seeds averaged) | risk_aware_rule_based | fixed20_cost | 17 | -27.7 | 1.322 | -20.95 | 0 | 17 | -184.4 | -0.0007807 | 0.0002334 |
| diurnal_long_holdout | ppo_clean (sampled, 3 seeds averaged) | risk_aware_rule_based | all | 20 | -65.19 | 4.479 | -14.55 | 0 | 20 | -491.8 | -0.004249 | 0.004255 |
| diurnal_long_holdout | ppo_clean (sampled, 3 seeds averaged) | risk_aware_rule_based | primary | 15 | -72.79 | 2.921 | -24.92 | 0 | 15 | -469.9 | 0.002617 | 0.0003386 |
| diurnal_long_holdout | ppo_clean (sampled, 3 seeds averaged) | risk_aware_rule_based | clip_0.01 | 12 | -74.26 | 3.333 | -22.28 | 0 | 12 | -505.2 | 0.002689 | 0.0004007 |
| diurnal_long_holdout | ppo_clean (sampled, 3 seeds averaged) | risk_aware_rule_based | clip_0.25 | 16 | -71.46 | 3.037 | -23.53 | 0 | 16 | -468.3 | 0.002549 | 0.0003241 |
| diurnal_long_holdout | ppo_clean (sampled, 3 seeds averaged) | risk_aware_rule_based | fixed20_cost | 17 | -71.54 | 2.853 | -25.07 | 0 | 17 | -476.3 | 0.00263 | 0.0003152 |
| diurnal_long_holdout | ppo_c_prime (deterministic, 3 seeds averaged) | risk_aware_rule_based | all | 20 | -21.09 | 2.855 | -7.387 | 0 | 20 | -159.1 | -0.005681 | 0.003555 |
| diurnal_long_holdout | ppo_c_prime (deterministic, 3 seeds averaged) | risk_aware_rule_based | primary | 15 | -18.93 | 1.409 | -13.43 | 0 | 15 | -122.2 | -0.0008848 | 0.0002529 |
| diurnal_long_holdout | ppo_c_prime (deterministic, 3 seeds averaged) | risk_aware_rule_based | clip_0.01 | 12 | -19.82 | 1.565 | -12.66 | 0 | 12 | -134.8 | -0.0007953 | 0.0002838 |
| diurnal_long_holdout | ppo_c_prime (deterministic, 3 seeds averaged) | risk_aware_rule_based | clip_0.25 | 16 | -18.89 | 1.319 | -14.32 | 0 | 16 | -123.8 | -0.0008295 | 0.000243 |
| diurnal_long_holdout | ppo_c_prime (deterministic, 3 seeds averaged) | risk_aware_rule_based | fixed20_cost | 17 | -18.95 | 1.24 | -15.28 | 0 | 17 | -126.2 | -0.0007807 | 0.0002334 |
| diurnal_long_holdout | ppo_c_prime (sampled, 3 seeds averaged) | risk_aware_rule_based | all | 20 | -73.95 | 5.376 | -13.76 | 0 | 20 | -557.9 | -0.001317 | 0.004129 |
| diurnal_long_holdout | ppo_c_prime (sampled, 3 seeds averaged) | risk_aware_rule_based | primary | 15 | -83.33 | 3.456 | -24.11 | 0 | 15 | -537.9 | 0.005204 | 0.0004532 |
| diurnal_long_holdout | ppo_c_prime (sampled, 3 seeds averaged) | risk_aware_rule_based | clip_0.01 | 12 | -82.24 | 4.033 | -20.39 | 0 | 12 | -559.6 | 0.005029 | 0.0005138 |
| diurnal_long_holdout | ppo_c_prime (sampled, 3 seeds averaged) | risk_aware_rule_based | clip_0.25 | 16 | -81.73 | 3.604 | -22.68 | 0 | 16 | -535.6 | 0.005088 | 0.0004394 |
| diurnal_long_holdout | ppo_c_prime (sampled, 3 seeds averaged) | risk_aware_rule_based | fixed20_cost | 17 | -81.61 | 3.387 | -24.1 | 0 | 17 | -543.3 | 0.005145 | 0.0004166 |
