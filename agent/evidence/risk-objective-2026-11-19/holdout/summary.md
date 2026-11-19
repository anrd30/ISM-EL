# Exposure-weighted objective, `profile_diurnal_long_holdout.csv`, hazard_ref = 0.000102537 per hour

## Summary

| environment | class | policy | n | total_reward | viol_frac | data_loss_probability | completion_h | delivered_mbps |
|---|---|---|---|---|---|---|---|---|
| contract | all | risk_aware_rule_based | 20 | -334 | 0.04528 | 9.883e-06 | 43.35 | 77.73 |
| contract | all | predictive_controller | 20 | -321 | 0.04725 | 8.754e-06 | 37.05 | 90.46 |
| contract | all | DP_hindsight_optimum (not deployable) | 20 | -320.2 | 0.04694 | 8.852e-06 | 37.44 | 89.44 |
| contract | primary_controllable | risk_aware_rule_based | 16 | -62.19 | 0.0009343 | 4.709e-07 | 44.77 | 74.59 |
| contract | primary_controllable | predictive_controller | 16 | -47.04 | 0 | 3.427e-07 | 37.77 | 88.47 |
| contract | primary_controllable | DP_hindsight_optimum (not deployable) | 16 | -46.15 | 0 | 3.505e-07 | 38.19 | 87.39 |
| contract | filter_forced | risk_aware_rule_based | 4 | -1421 | 0.2227 | 4.753e-05 | 37.71 | 90.27 |
| contract | filter_forced | predictive_controller | 4 | -1417 | 0.2362 | 4.24e-05 | 34.17 | 98.42 |
| contract | filter_forced | DP_hindsight_optimum (not deployable) | 4 | -1416 | 0.2347 | 4.286e-05 | 34.44 | 97.64 |
| objective | all | risk_aware_rule_based | 20 | -311.2 | 0.04528 | 9.883e-06 | 43.35 | 77.73 |
| objective | all | workload_only_throttle | 20 | -311.2 | 0.04528 | 9.883e-06 | 43.35 | 77.73 |
| objective | all | predictive_controller | 20 | -303.1 | 0.04725 | 8.754e-06 | 37.05 | 90.46 |
| objective | all | hazard_weighted_rule | 20 | -310.6 | 0.04554 | 9.818e-06 | 44.23 | 76.35 |
| objective | all | hazard_weighted_controller | 20 | -302.7 | 0.04736 | 8.779e-06 | 40.31 | 83.57 |
| objective | all | DP_hindsight_optimum (not deployable) | 20 | -301.8 | 0.04724 | 8.765e-06 | 37.78 | 88.7 |
| objective | primary_controllable | risk_aware_rule_based | 16 | -18.91 | 0.0009343 | 4.709e-07 | 44.77 | 74.59 |
| objective | primary_controllable | workload_only_throttle | 16 | -18.91 | 0.0009343 | 4.709e-07 | 44.77 | 74.59 |
| objective | primary_controllable | predictive_controller | 16 | -10.57 | 0 | 3.427e-07 | 37.77 | 88.47 |
| objective | primary_controllable | hazard_weighted_rule | 16 | -18.34 | 0.0009129 | 4.903e-07 | 45.89 | 72.75 |
| objective | primary_controllable | hazard_weighted_controller | 16 | -10.08 | 0 | 4.024e-07 | 41.85 | 79.8 |
| objective | primary_controllable | DP_hindsight_optimum (not deployable) | 16 | -9.068 | 0 | 3.572e-07 | 38.64 | 86.38 |
| objective | filter_forced | risk_aware_rule_based | 4 | -1480 | 0.2227 | 4.753e-05 | 37.71 | 90.27 |
| objective | filter_forced | workload_only_throttle | 4 | -1480 | 0.2227 | 4.753e-05 | 37.71 | 90.27 |
| objective | filter_forced | predictive_controller | 4 | -1473 | 0.2362 | 4.24e-05 | 34.17 | 98.42 |
| objective | filter_forced | hazard_weighted_rule | 4 | -1480 | 0.224 | 4.713e-05 | 37.56 | 90.73 |
| objective | filter_forced | hazard_weighted_controller | 4 | -1473 | 0.2368 | 4.228e-05 | 34.15 | 98.63 |
| objective | filter_forced | DP_hindsight_optimum (not deployable) | 4 | -1473 | 0.2362 | 4.24e-05 | 34.33 | 98 |
| objective_unforced | all | risk_aware_rule_based | 20 | -274.9 | 0.03819 | 1.042e-05 | 43.47 | 77.38 |
| objective_unforced | all | predictive_controller | 20 | -266.6 | 0.04014 | 9.075e-06 | 37.12 | 90.21 |
| objective_unforced | all | hazard_weighted_rule | 20 | -274.3 | 0.03844 | 1.033e-05 | 44.34 | 76 |
| objective_unforced | all | hazard_weighted_controller | 20 | -266.2 | 0.04023 | 9.099e-06 | 40.39 | 83.31 |
| objective_unforced | all | DP_hindsight_optimum (not deployable) | 20 | -102.5 | 0.01049 | 1.742e-05 | 39.18 | 85.48 |
| objective_unforced | primary_controllable | risk_aware_rule_based | 16 | -18.91 | 0.0009343 | 4.709e-07 | 44.77 | 74.59 |
| objective_unforced | primary_controllable | predictive_controller | 16 | -10.57 | 0 | 3.427e-07 | 37.77 | 88.47 |
| objective_unforced | primary_controllable | hazard_weighted_rule | 16 | -18.34 | 0.0009129 | 4.903e-07 | 45.89 | 72.75 |
| objective_unforced | primary_controllable | hazard_weighted_controller | 16 | -10.08 | 0 | 4.024e-07 | 41.85 | 79.8 |
| objective_unforced | primary_controllable | DP_hindsight_optimum (not deployable) | 16 | -9.068 | 0 | 3.572e-07 | 38.64 | 86.38 |
| objective_unforced | filter_forced | risk_aware_rule_based | 4 | -1299 | 0.1872 | 5.019e-05 | 38.29 | 88.56 |
| objective_unforced | filter_forced | predictive_controller | 4 | -1291 | 0.2007 | 4.4e-05 | 34.54 | 97.16 |
| objective_unforced | filter_forced | hazard_weighted_rule | 4 | -1298 | 0.1885 | 4.968e-05 | 38.12 | 89.03 |
| objective_unforced | filter_forced | hazard_weighted_controller | 4 | -1291 | 0.2012 | 4.389e-05 | 34.52 | 97.34 |
| objective_unforced | filter_forced | DP_hindsight_optimum (not deployable) | 4 | -476.4 | 0.05247 | 8.567e-05 | 41.35 | 81.89 |

## Gain over the rule

| environment | class | policy | n | gain_over_rule | sem | t | scenarios_differing_from_rule |
|---|---|---|---|---|---|---|---|
| contract | all | predictive_controller | 20 | 13.01 | 1.576 | 8.256 | 20 |
| contract | primary_controllable | predictive_controller | 16 | 15.15 | 1.498 | 10.11 | 20 |
| contract | all | DP_hindsight_optimum (not deployable) | 20 | 13.83 | 1.597 | 8.662 | 20 |
| contract | primary_controllable | DP_hindsight_optimum (not deployable) | 16 | 16.04 | 1.486 | 10.8 | 20 |
| objective | all | workload_only_throttle | 20 | 0 | 0 | n/a | 0 |
| objective | primary_controllable | workload_only_throttle | 16 | 0 | 0 | n/a | 0 |
| objective | all | predictive_controller | 20 | 8.114 | 1.17 | 6.937 | 20 |
| objective | primary_controllable | predictive_controller | 16 | 8.34 | 1.464 | 5.697 | 20 |
| objective | all | hazard_weighted_rule | 20 | 0.5236 | 0.07247 | 7.226 | 20 |
| objective | primary_controllable | hazard_weighted_rule | 16 | 0.5771 | 0.07784 | 7.414 | 20 |
| objective | all | hazard_weighted_controller | 20 | 8.465 | 1.205 | 7.022 | 20 |
| objective | primary_controllable | hazard_weighted_controller | 16 | 8.838 | 1.497 | 5.903 | 20 |
| objective | all | DP_hindsight_optimum (not deployable) | 20 | 9.409 | 1.185 | 7.941 | 20 |
| objective | primary_controllable | DP_hindsight_optimum (not deployable) | 16 | 9.845 | 1.466 | 6.713 | 20 |
| objective_unforced | all | predictive_controller | 20 | 8.295 | 1.178 | 7.044 | 20 |
| objective_unforced | primary_controllable | predictive_controller | 16 | 8.34 | 1.464 | 5.697 | 20 |
| objective_unforced | all | hazard_weighted_rule | 20 | 0.5418 | 0.07722 | 7.016 | 20 |
| objective_unforced | primary_controllable | hazard_weighted_rule | 16 | 0.5771 | 0.07784 | 7.414 | 20 |
| objective_unforced | all | hazard_weighted_controller | 20 | 8.651 | 1.21 | 7.148 | 20 |
| objective_unforced | primary_controllable | hazard_weighted_controller | 16 | 8.838 | 1.497 | 5.903 | 20 |
| objective_unforced | all | DP_hindsight_optimum (not deployable) | 20 | 172.3 | 115.1 | 1.498 | 20 |
| objective_unforced | primary_controllable | DP_hindsight_optimum (not deployable) | 16 | 9.845 | 1.466 | 6.713 | 20 |

## Filter-forced seeds [9, 17, 19]: violation fraction against data-loss probability

| environment | policy | seed | viol_frac | data_loss_probability | total_reward | completion_h | delivered_mbps | risk_budget_violated |
|---|---|---|---|---|---|---|---|---|
| contract | risk_aware_rule_based | 9 | 0.4665 | 7.154e-05 | -2802 | 32.33 | 103.1 | true |
| contract | risk_aware_rule_based | 17 | 0.2356 | 9.273e-05 | -1429 | 33.25 | 100.4 | true |
| contract | risk_aware_rule_based | 19 | 0.1868 | 1.86e-05 | -1385 | 39.25 | 84.93 | false |
| contract | predictive_controller | 9 | 0.4918 | 6.438e-05 | -2800 | 30.67 | 108.7 | true |
| contract | predictive_controller | 17 | 0.2467 | 8.459e-05 | -1428 | 31.75 | 105 | true |
| contract | predictive_controller | 19 | 0.2042 | 1.558e-05 | -1381 | 35.92 | 92.95 | false |
| contract | DP_hindsight_optimum (not deployable) | 9 | 0.4879 | 6.543e-05 | -2800 | 30.92 | 107.8 | true |
| contract | DP_hindsight_optimum (not deployable) | 17 | 0.2461 | 8.503e-05 | -1427 | 31.83 | 104.7 | true |
| contract | DP_hindsight_optimum (not deployable) | 19 | 0.2028 | 1.58e-05 | -1380 | 36.17 | 92.17 | false |
| objective | risk_aware_rule_based | 9 | 0.4665 | 7.154e-05 | -2904 | 32.33 | 103.1 | true |
| objective | risk_aware_rule_based | 17 | 0.2356 | 9.273e-05 | -1550 | 33.25 | 100.4 | true |
| objective | risk_aware_rule_based | 19 | 0.1868 | 1.86e-05 | -1410 | 39.25 | 84.93 | false |
| objective | workload_only_throttle | 9 | 0.4665 | 7.154e-05 | -2904 | 32.33 | 103.1 | true |
| objective | workload_only_throttle | 17 | 0.2356 | 9.273e-05 | -1550 | 33.25 | 100.4 | true |
| objective | workload_only_throttle | 19 | 0.1868 | 1.86e-05 | -1410 | 39.25 | 84.93 | false |
| objective | predictive_controller | 9 | 0.4918 | 6.438e-05 | -2896 | 30.67 | 108.7 | true |
| objective | predictive_controller | 17 | 0.2467 | 8.459e-05 | -1542 | 31.75 | 105 | true |
| objective | predictive_controller | 19 | 0.2042 | 1.558e-05 | -1403 | 35.92 | 92.95 | false |
| objective | hazard_weighted_rule | 9 | 0.4689 | 7.081e-05 | -2903 | 32.17 | 103.8 | true |
| objective | hazard_weighted_rule | 17 | 0.2362 | 9.227e-05 | -1549 | 33.17 | 100.7 | true |
| objective | hazard_weighted_rule | 19 | 0.1892 | 1.813e-05 | -1409 | 38.75 | 86.2 | false |
| objective | hazard_weighted_controller | 9 | 0.4932 | 6.403e-05 | -2896 | 30.58 | 109.1 | true |
| objective | hazard_weighted_controller | 17 | 0.2467 | 8.459e-05 | -1543 | 31.75 | 105.3 | true |
| objective | hazard_weighted_controller | 19 | 0.2051 | 1.544e-05 | -1404 | 35.75 | 93.48 | false |
| objective | DP_hindsight_optimum (not deployable) | 9 | 0.4932 | 6.403e-05 | -2896 | 30.58 | 109 | true |
| objective | DP_hindsight_optimum (not deployable) | 17 | 0.2467 | 8.459e-05 | -1542 | 31.75 | 105 | true |
| objective | DP_hindsight_optimum (not deployable) | 19 | 0.2028 | 1.58e-05 | -1403 | 36.17 | 92.17 | false |
| objective_unforced | risk_aware_rule_based | 9 | 0.3245 | 8.219e-05 | -2178 | 34.67 | 96.25 | true |
| objective_unforced | risk_aware_rule_based | 17 | 0.2356 | 9.273e-05 | -1550 | 33.25 | 100.4 | true |
| objective_unforced | risk_aware_rule_based | 19 | 0.1868 | 1.86e-05 | -1410 | 39.25 | 84.93 | false |
| objective_unforced | predictive_controller | 9 | 0.3497 | 7.081e-05 | -2167 | 32.17 | 103.7 | true |
| objective_unforced | predictive_controller | 17 | 0.2467 | 8.459e-05 | -1542 | 31.75 | 105 | true |
| objective_unforced | predictive_controller | 19 | 0.2042 | 1.558e-05 | -1403 | 35.92 | 92.95 | false |
| objective_unforced | hazard_weighted_rule | 9 | 0.3269 | 8.101e-05 | -2177 | 34.42 | 96.93 | true |
| objective_unforced | hazard_weighted_rule | 17 | 0.2362 | 9.227e-05 | -1549 | 33.17 | 100.7 | true |
| objective_unforced | hazard_weighted_rule | 19 | 0.1892 | 1.813e-05 | -1409 | 38.75 | 86.2 | false |
| objective_unforced | hazard_weighted_controller | 9 | 0.3506 | 7.044e-05 | -2167 | 32.08 | 104 | true |
| objective_unforced | hazard_weighted_controller | 17 | 0.2467 | 8.459e-05 | -1543 | 31.75 | 105.3 | true |
| objective_unforced | hazard_weighted_controller | 19 | 0.2051 | 1.544e-05 | -1404 | 35.75 | 93.48 | false |
| objective_unforced | DP_hindsight_optimum (not deployable) | 9 | 0 | 0.0001088 | -175.7 | 39.92 | 83.53 | true |
| objective_unforced | DP_hindsight_optimum (not deployable) | 17 | 0.00495 | 0.0002129 | -278.2 | 50.5 | 66.01 | true |
| objective_unforced | DP_hindsight_optimum (not deployable) | 19 | 0.2028 | 1.58e-05 | -1403 | 36.17 | 92.17 | false |
