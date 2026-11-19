# Exposure-weighted objective, `profile_diurnal_long_altB.csv`, hazard_ref = 0.000102537 per hour

## Summary

| environment | class | policy | n | total_reward | viol_frac | data_loss_probability | completion_h | delivered_mbps |
|---|---|---|---|---|---|---|---|---|
| contract | all | risk_aware_rule_based | 20 | -227.7 | 0.03311 | 9.347e-06 | 39.68 | 84.87 |
| contract | all | predictive_controller | 20 | -216.6 | 0.03103 | 9.525e-06 | 40.58 | 82.95 |
| contract | all | DP_hindsight_optimum (not deployable) | 20 | -197.8 | 0.02964 | 8.817e-06 | 36.08 | 92.76 |
| contract | primary_controllable | risk_aware_rule_based | 16 | -78.58 | 0.004845 | 4.018e-07 | 40.86 | 82.03 |
| contract | primary_controllable | predictive_controller | 16 | -64.74 | 0.002507 | 4.247e-07 | 41.82 | 80.04 |
| contract | primary_controllable | DP_hindsight_optimum (not deployable) | 16 | -44.18 | 0 | 3.235e-07 | 36.68 | 91.06 |
| contract | filter_forced | risk_aware_rule_based | 4 | -824 | 0.1462 | 4.513e-05 | 34.94 | 96.23 |
| contract | filter_forced | predictive_controller | 4 | -824.2 | 0.1451 | 4.593e-05 | 35.65 | 94.59 |
| contract | filter_forced | DP_hindsight_optimum (not deployable) | 4 | -812.5 | 0.1482 | 4.279e-05 | 33.67 | 99.57 |
| objective | all | risk_aware_rule_based | 20 | -207.7 | 0.03311 | 9.347e-06 | 39.68 | 84.87 |
| objective | all | workload_only_throttle | 20 | -207.7 | 0.03311 | 9.347e-06 | 39.68 | 84.87 |
| objective | all | predictive_controller | 20 | -196 | 0.03103 | 9.525e-06 | 40.58 | 82.95 |
| objective | all | hazard_weighted_rule | 20 | -196.7 | 0.03115 | 9.293e-06 | 44.92 | 75.86 |
| objective | all | hazard_weighted_controller | 20 | -193 | 0.03061 | 9.444e-06 | 45.3 | 75.07 |
| objective | all | DP_hindsight_optimum (not deployable) | 20 | -180.7 | 0.02992 | 8.672e-06 | 36.22 | 92.47 |
| objective | primary_controllable | risk_aware_rule_based | 16 | -39.11 | 0.004845 | 4.018e-07 | 40.86 | 82.03 |
| objective | primary_controllable | workload_only_throttle | 16 | -39.11 | 0.004845 | 4.018e-07 | 40.86 | 82.03 |
| objective | primary_controllable | predictive_controller | 16 | -24.38 | 0.002507 | 4.247e-07 | 41.82 | 80.04 |
| objective | primary_controllable | hazard_weighted_rule | 16 | -25.54 | 0.002051 | 5.218e-07 | 47.47 | 70.58 |
| objective | primary_controllable | hazard_weighted_controller | 16 | -20.14 | 0.001425 | 5.374e-07 | 47.75 | 70.09 |
| objective | primary_controllable | DP_hindsight_optimum (not deployable) | 16 | -8.653 | 0 | 3.269e-07 | 36.9 | 90.54 |
| objective | filter_forced | risk_aware_rule_based | 4 | -882 | 0.1462 | 4.513e-05 | 34.94 | 96.23 |
| objective | filter_forced | workload_only_throttle | 4 | -882 | 0.1462 | 4.513e-05 | 34.94 | 96.23 |
| objective | filter_forced | predictive_controller | 4 | -882.6 | 0.1451 | 4.593e-05 | 35.65 | 94.59 |
| objective | filter_forced | hazard_weighted_rule | 4 | -881.5 | 0.1475 | 4.438e-05 | 34.71 | 96.98 |
| objective | filter_forced | hazard_weighted_controller | 4 | -884.5 | 0.1474 | 4.507e-05 | 35.5 | 95 |
| objective | filter_forced | DP_hindsight_optimum (not deployable) | 4 | -868.9 | 0.1496 | 4.205e-05 | 33.48 | 100.2 |
| objective_unforced | all | risk_aware_rule_based | 20 | -209.8 | 0.03355 | 9.385e-06 | 39.69 | 84.85 |
| objective_unforced | all | predictive_controller | 20 | -202.2 | 0.03214 | 9.602e-06 | 40.6 | 82.9 |
| objective_unforced | all | hazard_weighted_rule | 20 | -197.8 | 0.0314 | 9.293e-06 | 44.92 | 75.84 |
| objective_unforced | all | hazard_weighted_controller | 20 | -196.5 | 0.03126 | 9.502e-06 | 45.31 | 75.04 |
| objective_unforced | all | DP_hindsight_optimum (not deployable) | 20 | -40.21 | 0.001664 | 1.546e-05 | 37.43 | 89.56 |
| objective_unforced | primary_controllable | risk_aware_rule_based | 16 | -39.11 | 0.004845 | 4.018e-07 | 40.86 | 82.03 |
| objective_unforced | primary_controllable | predictive_controller | 16 | -24.38 | 0.002507 | 4.247e-07 | 41.82 | 80.04 |
| objective_unforced | primary_controllable | hazard_weighted_rule | 16 | -25.54 | 0.002051 | 5.218e-07 | 47.47 | 70.58 |
| objective_unforced | primary_controllable | hazard_weighted_controller | 16 | -20.14 | 0.001425 | 5.374e-07 | 47.75 | 70.09 |
| objective_unforced | primary_controllable | DP_hindsight_optimum (not deployable) | 16 | -8.653 | 0 | 3.269e-07 | 36.9 | 90.54 |
| objective_unforced | filter_forced | risk_aware_rule_based | 4 | -892.7 | 0.1484 | 4.532e-05 | 34.98 | 96.13 |
| objective_unforced | filter_forced | predictive_controller | 4 | -913.3 | 0.1507 | 4.631e-05 | 35.73 | 94.33 |
| objective_unforced | filter_forced | hazard_weighted_rule | 4 | -886.6 | 0.1488 | 4.438e-05 | 34.71 | 96.91 |
| objective_unforced | filter_forced | hazard_weighted_controller | 4 | -902.2 | 0.1506 | 4.536e-05 | 35.56 | 94.85 |
| objective_unforced | filter_forced | DP_hindsight_optimum (not deployable) | 4 | -166.5 | 0.008318 | 7.599e-05 | 39.54 | 85.68 |

## Gain over the rule

| environment | class | policy | n | gain_over_rule | sem | t | scenarios_differing_from_rule |
|---|---|---|---|---|---|---|---|
| contract | all | predictive_controller | 20 | 11.02 | 3.441 | 3.204 | 20 |
| contract | primary_controllable | predictive_controller | 16 | 13.84 | 3.764 | 3.676 | 20 |
| contract | all | DP_hindsight_optimum (not deployable) | 20 | 29.82 | 3.192 | 9.342 | 20 |
| contract | primary_controllable | DP_hindsight_optimum (not deployable) | 16 | 34.4 | 2.743 | 12.54 | 20 |
| objective | all | workload_only_throttle | 20 | 0 | 0 | n/a | 0 |
| objective | primary_controllable | workload_only_throttle | 16 | 0 | 0 | n/a | 0 |
| objective | all | predictive_controller | 20 | 11.67 | 3.462 | 3.37 | 20 |
| objective | primary_controllable | predictive_controller | 16 | 14.73 | 3.693 | 3.987 | 20 |
| objective | all | hazard_weighted_rule | 20 | 10.97 | 2.316 | 4.738 | 20 |
| objective | primary_controllable | hazard_weighted_rule | 16 | 13.57 | 2.497 | 5.435 | 20 |
| objective | all | hazard_weighted_controller | 20 | 14.69 | 3.182 | 4.617 | 20 |
| objective | primary_controllable | hazard_weighted_controller | 16 | 18.97 | 2.79 | 6.801 | 20 |
| objective | all | DP_hindsight_optimum (not deployable) | 20 | 27 | 2.868 | 9.413 | 20 |
| objective | primary_controllable | DP_hindsight_optimum (not deployable) | 16 | 30.46 | 2.745 | 11.1 | 20 |
| objective_unforced | all | predictive_controller | 20 | 7.676 | 5.712 | 1.344 | 20 |
| objective_unforced | primary_controllable | predictive_controller | 16 | 14.73 | 3.693 | 3.987 | 20 |
| objective_unforced | all | hazard_weighted_rule | 20 | 12.07 | 2.313 | 5.221 | 20 |
| objective_unforced | primary_controllable | hazard_weighted_rule | 16 | 13.57 | 2.497 | 5.435 | 20 |
| objective_unforced | all | hazard_weighted_controller | 20 | 13.29 | 3.956 | 3.36 | 20 |
| objective_unforced | primary_controllable | hazard_weighted_controller | 16 | 18.97 | 2.79 | 6.801 | 20 |
| objective_unforced | all | DP_hindsight_optimum (not deployable) | 20 | 169.6 | 97.96 | 1.732 | 20 |
| objective_unforced | primary_controllable | DP_hindsight_optimum (not deployable) | 16 | 30.46 | 2.745 | 11.1 | 20 |

## Filter-forced seeds [9, 17]: violation fraction against data-loss probability

| environment | policy | seed | viol_frac | data_loss_probability | total_reward | completion_h | delivered_mbps | risk_budget_violated |
|---|---|---|---|---|---|---|---|---|
| contract | risk_aware_rule_based | 9 | 0.2296 | 7.792e-05 | -1366 | 33.75 | 98.86 | true |
| contract | risk_aware_rule_based | 17 | 0.3245 | 8.239e-05 | -1676 | 31.33 | 106.6 | true |
| contract | predictive_controller | 9 | 0.2296 | 7.792e-05 | -1366 | 33.75 | 98.96 | true |
| contract | predictive_controller | 17 | 0.3228 | 8.459e-05 | -1692 | 31.75 | 105.2 | true |
| contract | DP_hindsight_optimum (not deployable) | 9 | 0.2298 | 7.451e-05 | -1343 | 33 | 101 | true |
| contract | DP_hindsight_optimum (not deployable) | 17 | 0.3333 | 7.808e-05 | -1675 | 30.5 | 109.3 | true |
| objective | risk_aware_rule_based | 9 | 0.2296 | 7.792e-05 | -1472 | 33.75 | 98.86 | true |
| objective | risk_aware_rule_based | 17 | 0.3245 | 8.239e-05 | -1789 | 31.33 | 106.6 | true |
| objective | workload_only_throttle | 9 | 0.2296 | 7.792e-05 | -1472 | 33.75 | 98.86 | true |
| objective | workload_only_throttle | 17 | 0.3245 | 8.239e-05 | -1789 | 31.33 | 106.6 | true |
| objective | predictive_controller | 9 | 0.2296 | 7.792e-05 | -1472 | 33.75 | 98.96 | true |
| objective | predictive_controller | 17 | 0.3228 | 8.459e-05 | -1807 | 31.75 | 105.2 | true |
| objective | hazard_weighted_rule | 9 | 0.2302 | 7.754e-05 | -1472 | 33.67 | 99.28 | true |
| objective | hazard_weighted_rule | 17 | 0.3288 | 8.022e-05 | -1788 | 30.92 | 107.9 | true |
| objective | hazard_weighted_controller | 9 | 0.2333 | 7.715e-05 | -1482 | 33.58 | 99.27 | true |
| objective | hazard_weighted_controller | 17 | 0.328 | 8.196e-05 | -1805 | 31.25 | 106.7 | true |
| objective | DP_hindsight_optimum (not deployable) | 9 | 0.2316 | 7.339e-05 | -1447 | 32.75 | 101.8 | true |
| objective | DP_hindsight_optimum (not deployable) | 17 | 0.337 | 7.639e-05 | -1785 | 30.17 | 110.6 | true |
| objective_unforced | risk_aware_rule_based | 9 | 0.2383 | 7.869e-05 | -1515 | 33.92 | 98.47 | true |
| objective_unforced | risk_aware_rule_based | 17 | 0.3245 | 8.239e-05 | -1789 | 31.33 | 106.6 | true |
| objective_unforced | predictive_controller | 9 | 0.2518 | 7.946e-05 | -1594 | 34.08 | 97.91 | true |
| objective_unforced | predictive_controller | 17 | 0.3228 | 8.459e-05 | -1807 | 31.75 | 105.2 | true |
| objective_unforced | hazard_weighted_rule | 9 | 0.2351 | 7.754e-05 | -1493 | 33.67 | 99.03 | true |
| objective_unforced | hazard_weighted_rule | 17 | 0.3288 | 8.022e-05 | -1788 | 30.92 | 107.9 | true |
| objective_unforced | hazard_weighted_controller | 9 | 0.2463 | 7.83e-05 | -1552 | 33.83 | 98.68 | true |
| objective_unforced | hazard_weighted_controller | 17 | 0.328 | 8.196e-05 | -1805 | 31.25 | 106.7 | true |
| objective_unforced | DP_hindsight_optimum (not deployable) | 9 | 0.003497 | 0.0001549 | -230.2 | 47.67 | 69.93 | true |
| objective_unforced | DP_hindsight_optimum (not deployable) | 17 | 0 | 0.0001306 | -191.4 | 39.5 | 84.39 | true |
