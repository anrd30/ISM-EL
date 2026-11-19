# Value of information, `eval-fleet,sigma=0.25`

Workload `data_pipeline/profiles/profile_diurnal_long_altB.csv`, 20 scenarios. Gain is `NOISY_ORACLE` minus `risk_aware_rule_based`; percent of optimum is that gain over `DP_hindsight_optimum` minus the rule, on the scenarios controllable under each definition.

| definition | sigma_mbps | n_controllable | gain_over_rule | sem | optimum_gain | percent_of_optimum |
|---|---|---|---|---|---|---|
| primary | 0 | 16 | 33.16 | 2.716 | 34.4 | 96.41 |
| primary | 2 | 16 | 32.8 | 2.708 | 34.4 | 95.36 |
| primary | 5 | 16 | 31.61 | 2.856 | 34.4 | 91.89 |
| primary | 7 | 16 | 29.64 | 2.892 | 34.4 | 86.15 |
| primary | 10 | 16 | 24.65 | 2.845 | 34.4 | 71.66 |
| primary | 15 | 16 | 10.79 | 3.463 | 34.4 | 31.38 |
| clip_0.01 | 0 | 13 | 31.59 | 3.195 | 32.81 | 96.27 |
| clip_0.01 | 2 | 13 | 31.21 | 3.178 | 32.81 | 95.13 |
| clip_0.01 | 5 | 13 | 29.8 | 3.321 | 32.81 | 90.82 |
| clip_0.01 | 7 | 13 | 27.76 | 3.35 | 32.81 | 84.62 |
| clip_0.01 | 10 | 13 | 22.85 | 3.29 | 32.81 | 69.66 |
| clip_0.01 | 15 | 13 | 7.771 | 3.768 | 32.81 | 23.69 |
| clip_0.25 | 0 | 16 | 33.16 | 2.716 | 34.4 | 96.41 |
| clip_0.25 | 2 | 16 | 32.8 | 2.708 | 34.4 | 95.36 |
| clip_0.25 | 5 | 16 | 31.61 | 2.856 | 34.4 | 91.89 |
| clip_0.25 | 7 | 16 | 29.64 | 2.892 | 34.4 | 86.15 |
| clip_0.25 | 10 | 16 | 24.65 | 2.845 | 34.4 | 71.66 |
| clip_0.25 | 15 | 16 | 10.79 | 3.463 | 34.4 | 31.38 |
| fixed20_cost | 0 | 17 | 32.3 | 2.695 | 33.53 | 96.33 |
| fixed20_cost | 2 | 17 | 31.91 | 2.695 | 33.53 | 95.18 |
| fixed20_cost | 5 | 17 | 30.81 | 2.798 | 33.53 | 91.9 |
| fixed20_cost | 7 | 17 | 28.92 | 2.809 | 33.53 | 86.25 |
| fixed20_cost | 10 | 17 | 24.08 | 2.733 | 33.53 | 71.82 |
| fixed20_cost | 15 | 17 | 10.78 | 3.253 | 33.53 | 32.15 |
| excluded_seeds | 0 | 18 | 30.58 | 3.067 | 31.79 | 96.19 |
| excluded_seeds | 2 | 18 | 30.22 | 3.051 | 31.79 | 95.08 |
| excluded_seeds | 5 | 18 | 29.18 | 3.102 | 31.79 | 91.79 |
| excluded_seeds | 7 | 18 | 27.34 | 3.082 | 31.79 | 86.01 |
| excluded_seeds | 10 | 18 | 22.66 | 2.94 | 31.79 | 71.28 |
| excluded_seeds | 15 | 18 | 9.914 | 3.186 | 31.79 | 31.19 |
