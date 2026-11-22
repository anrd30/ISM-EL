# Value of information, `eval-fleet,sigma=0.25`

Workload `data_pipeline/profiles/profile_diurnal_long_holdout.csv`, 20 scenarios. Gain is `NOISY_ORACLE` minus `risk_aware_rule_based`; percent of optimum is that gain over `DP_hindsight_optimum` minus the rule, on the scenarios controllable under each definition.

| definition | sigma_mbps | n_controllable | gain_over_rule | sem | optimum_gain | percent_of_optimum |
|---|---|---|---|---|---|---|
| primary | 0 | 16 | 13.75 | 1.472 | 14.84 | 92.7 |
| primary | 2 | 16 | 13.5 | 1.488 | 14.84 | 90.98 |
| primary | 5 | 16 | 12.6 | 1.447 | 14.84 | 84.91 |
| primary | 7 | 16 | 9.708 | 1.667 | 14.84 | 65.43 |
| primary | 10 | 16 | 3.648 | 2.139 | 14.84 | 24.58 |
| primary | 15 | 16 | -8.717 | 2.658 | 14.84 | -58.75 |
| clip_0.01 | 0 | 13 | 13.25 | 1.618 | 14.34 | 92.36 |
| clip_0.01 | 2 | 13 | 13 | 1.625 | 14.34 | 90.63 |
| clip_0.01 | 5 | 13 | 12.16 | 1.571 | 14.34 | 84.74 |
| clip_0.01 | 7 | 13 | 9.01 | 1.859 | 14.34 | 62.81 |
| clip_0.01 | 10 | 13 | 2.326 | 2.388 | 14.34 | 16.21 |
| clip_0.01 | 15 | 13 | -9.741 | 3.171 | 14.34 | -67.91 |
| clip_0.25 | 0 | 16 | 13.75 | 1.472 | 14.84 | 92.7 |
| clip_0.25 | 2 | 16 | 13.5 | 1.488 | 14.84 | 90.98 |
| clip_0.25 | 5 | 16 | 12.6 | 1.447 | 14.84 | 84.91 |
| clip_0.25 | 7 | 16 | 9.708 | 1.667 | 14.84 | 65.43 |
| clip_0.25 | 10 | 16 | 3.648 | 2.139 | 14.84 | 24.58 |
| clip_0.25 | 15 | 16 | -8.717 | 2.658 | 14.84 | -58.75 |
| fixed20_cost | 0 | 17 | 13.51 | 1.404 | 14.58 | 92.62 |
| fixed20_cost | 2 | 17 | 13.25 | 1.421 | 14.58 | 90.82 |
| fixed20_cost | 5 | 17 | 12.33 | 1.384 | 14.58 | 84.58 |
| fixed20_cost | 7 | 17 | 9.477 | 1.582 | 14.58 | 64.98 |
| fixed20_cost | 10 | 17 | 2.957 | 2.125 | 14.58 | 20.27 |
| fixed20_cost | 15 | 17 | -8.935 | 2.506 | 14.58 | -61.27 |
| excluded_seeds | 0 | 17 | 13.51 | 1.404 | 14.58 | 92.62 |
| excluded_seeds | 2 | 17 | 13.25 | 1.421 | 14.58 | 90.82 |
| excluded_seeds | 5 | 17 | 12.33 | 1.384 | 14.58 | 84.58 |
| excluded_seeds | 7 | 17 | 9.477 | 1.582 | 14.58 | 64.98 |
| excluded_seeds | 10 | 17 | 2.957 | 2.125 | 14.58 | 20.27 |
| excluded_seeds | 15 | 17 | -8.935 | 2.506 | 14.58 | -61.27 |
