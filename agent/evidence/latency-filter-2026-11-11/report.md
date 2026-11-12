# Latency ceiling: every condition, no filter / oracle-fg / causal-fg

`none` is `latency_filter=False`. `oracle-fg` (`ceiling_oracle_fg`, `latency_filter=True`) reads the foreground load of the interval being decided. `causal-fg` (`ceiling_causal_fg`, `latency_filter_causal=True`) reads the previous interval's load, the value the policy's own observation carries. Same scenario seeds (0..19), same policies and checkpoints in every column. `gap` is the paired total-reward difference against `risk_aware_rule_based` under the same variant (policy minus rule; standard error in brackets); `better` counts scenarios with a positive gap. `viol` is the mean fraction of intervals violating latency; `lat_ok` the fraction of episodes within the latency budget.

## Harness reproduction (no filter, home, against the committed forecast-ppo evidence)

| policy | max abs per-scenario difference | mean reward here | mean reward committed |
|---|---|---|---|
| risk_aware_rule_based | 0 | -330.20 | -330.20 |
| rule_seasonal_forecast (hand-built) | 0 | -326.51 | -326.51 |
| forecast_ppo_bc seed0 | 0 | -323.96 | -323.96 |
| forecast_ppo_bc seed1 | 0 | -323.87 | -323.87 |
| forecast_ppo_bc seed2 | 0 | -323.48 | -323.48 |

## Condition `home`

| policy | reward none | reward oracle-fg | reward causal-fg | gap none | gap oracle-fg | gap causal-fg | better none | better oracle-fg | better causal-fg |
|---|---|---|---|---|---|---|---|---|---|
| risk_aware_rule_based | -330.2 | -326.8 | -330.2 | - | - | - | - | - | - |
| rule_seasonal_forecast (hand-built) | -326.5 | -323.1 | -326.5 | +3.68 (0.28) | +3.69 (0.27) | +3.68 (0.27) | 20/20 | 20/20 | 20/20 |
| forecast_ppo_bc seed0 | -324.0 | -323.4 | -323.9 | +6.24 (1.42) | +3.35 (0.27) | +6.31 (1.42) | 19/20 | 20/20 | 19/20 |
| forecast_ppo_bc seed1 | -323.9 | -323.4 | -323.9 | +6.32 (1.42) | +3.43 (0.27) | +6.33 (1.42) | 19/20 | 20/20 | 19/20 |
| forecast_ppo_bc seed2 | -323.5 | -323.5 | -323.4 | +6.71 (1.26) | +3.30 (0.24) | +6.83 (1.26) | 20/20 | 20/20 | 20/20 |

| policy | viol none | viol oracle-fg | viol causal-fg | lat_ok none / oracle-fg / causal-fg | completion h none / oracle-fg / causal-fg |
|---|---|---|---|---|---|
| risk_aware_rule_based | 0.0444 | 0.0438 | 0.0444 | 0.85 / 0.85 / 0.85 | 43.6 / 43.6 / 43.6 |
| rule_seasonal_forecast (hand-built) | 0.0462 | 0.0456 | 0.0461 | 0.85 / 0.85 / 0.85 | 40.6 / 40.6 / 40.7 |
| forecast_ppo_bc seed0 | 0.0456 | 0.0455 | 0.0452 | 0.85 / 0.85 / 0.85 | 41.4 / 41.4 / 41.8 |
| forecast_ppo_bc seed1 | 0.0454 | 0.0454 | 0.0451 | 0.85 / 0.85 / 0.85 | 41.6 / 41.6 / 41.9 |
| forecast_ppo_bc seed2 | 0.0455 | 0.0455 | 0.0451 | 0.85 / 0.85 / 0.85 | 41.5 / 41.5 / 41.8 |

Worst single-episode reward over these policies: none -2782.7, oracle-fg -2782.7, causal-fg -2782.7.

## Condition `peak`

| policy | reward none | reward oracle-fg | reward causal-fg | gap none | gap oracle-fg | gap causal-fg | better none | better oracle-fg | better causal-fg |
|---|---|---|---|---|---|---|---|---|---|
| risk_aware_rule_based | -1808.2 | -1808.2 | -1808.2 | - | - | - | - | - | - |
| rule_seasonal_forecast (hand-built) | -1804.5 | -1804.5 | -1804.5 | +3.69 (0.42) | +3.69 (0.42) | +3.69 (0.42) | 20/20 | 20/20 | 20/20 |
| forecast_ppo_bc seed0 | -7441.2 | -1828.7 | -2120.6 | -5633.07 (491.90) | -20.49 (1.86) | -312.44 (29.09) | 0/20 | 0/20 | 0/20 |
| forecast_ppo_bc seed1 | -7441.2 | -1828.7 | -2120.6 | -5633.07 (491.90) | -20.49 (1.86) | -312.44 (29.09) | 0/20 | 0/20 | 0/20 |
| forecast_ppo_bc seed2 | -7441.2 | -1828.7 | -2120.6 | -5633.07 (491.90) | -20.49 (1.86) | -312.44 (29.09) | 0/20 | 0/20 | 0/20 |

| policy | viol none | viol oracle-fg | viol causal-fg | lat_ok none / oracle-fg / causal-fg | completion h none / oracle-fg / causal-fg |
|---|---|---|---|---|---|
| risk_aware_rule_based | 0.1319 | 0.1319 | 0.1319 | 0.70 / 0.70 / 0.70 | 124.3 / 124.3 / 124.3 |
| rule_seasonal_forecast (hand-built) | 0.1336 | 0.1336 | 0.1336 | 0.70 / 0.70 / 0.70 | 121.1 / 121.1 / 121.1 |
| forecast_ppo_bc seed0 | 0.7607 | 0.1384 | 0.1588 | 0.00 / 0.70 / 0.65 | 65.5 / 107.5 / 107.5 |
| forecast_ppo_bc seed1 | 0.7607 | 0.1384 | 0.1588 | 0.00 / 0.70 / 0.65 | 65.5 / 107.5 / 107.5 |
| forecast_ppo_bc seed2 | 0.7607 | 0.1384 | 0.1588 | 0.00 / 0.70 / 0.65 | 65.5 / 107.5 / 107.5 |

Worst single-episode reward over these policies: none -10042.3, oracle-fg -9344.8, causal-fg -9364.5.

## Condition `bursty`

| policy | reward none | reward oracle-fg | reward causal-fg | gap none | gap oracle-fg | gap causal-fg | better none | better oracle-fg | better causal-fg |
|---|---|---|---|---|---|---|---|---|---|
| risk_aware_rule_based | -263.6 | -219.9 | -263.6 | - | - | - | - | - | - |
| rule_seasonal_forecast (hand-built) | -249.3 | -216.7 | -249.5 | +14.30 (3.07) | +3.17 (0.42) | +14.05 (3.03) | 20/20 | 20/20 | 20/20 |
| forecast_ppo_bc seed0 | -322.9 | -213.5 | -309.9 | -59.31 (3.97) | +6.42 (1.73) | -46.34 (4.01) | 0/20 | 20/20 | 0/20 |
| forecast_ppo_bc seed1 | -324.5 | -213.4 | -313.6 | -60.89 (5.09) | +6.52 (1.72) | -50.05 (4.54) | 0/20 | 20/20 | 0/20 |
| forecast_ppo_bc seed2 | -334.1 | -213.4 | -312.4 | -70.58 (6.19) | +6.54 (1.73) | -48.83 (4.55) | 0/20 | 20/20 | 0/20 |

| policy | viol none | viol oracle-fg | viol causal-fg | lat_ok none / oracle-fg / causal-fg | completion h none / oracle-fg / causal-fg |
|---|---|---|---|---|---|
| risk_aware_rule_based | 0.0333 | 0.0266 | 0.0333 | 0.85 / 0.85 / 0.85 | 40.1 / 40.2 / 40.1 |
| rule_seasonal_forecast (hand-built) | 0.0334 | 0.0280 | 0.0332 | 0.85 / 0.85 / 0.85 | 37.5 / 37.6 / 37.8 |
| forecast_ppo_bc seed0 | 0.0499 | 0.0277 | 0.0464 | 0.85 / 0.85 / 0.85 | 34.3 / 35.3 / 35.3 |
| forecast_ppo_bc seed1 | 0.0499 | 0.0277 | 0.0472 | 0.85 / 0.85 / 0.85 | 34.3 / 35.1 / 35.2 |
| forecast_ppo_bc seed2 | 0.0515 | 0.0278 | 0.0470 | 0.85 / 0.85 / 0.85 | 34.2 / 35.2 / 35.2 |

Worst single-episode reward over these policies: none -1439.4, oracle-fg -1370.8, causal-fg -1396.2.

## Condition `diurnal_generic`

| policy | reward none | reward oracle-fg | reward causal-fg | gap none | gap oracle-fg | gap causal-fg | better none | better oracle-fg | better causal-fg |
|---|---|---|---|---|---|---|---|---|---|
| risk_aware_rule_based | -310.0 | -291.3 | -310.0 | - | - | - | - | - | - |
| rule_seasonal_forecast (hand-built) | -300.4 | -289.1 | -300.5 | +9.69 (1.08) | +2.27 (0.15) | +9.56 (1.06) | 20/20 | 20/20 | 20/20 |
| forecast_ppo_bc seed0 | -300.6 | -291.7 | -300.5 | +9.42 (1.46) | -0.31 (0.17) | +9.59 (1.45) | 16/20 | 6/20 | 16/20 |
| forecast_ppo_bc seed1 | -300.7 | -291.7 | -300.6 | +9.38 (1.47) | -0.35 (0.19) | +9.45 (1.47) | 16/20 | 6/20 | 16/20 |
| forecast_ppo_bc seed2 | -301.4 | -291.9 | -299.2 | +8.63 (2.25) | -0.55 (0.19) | +10.89 (1.73) | 14/20 | 4/20 | 16/20 |

| policy | viol none | viol oracle-fg | viol causal-fg | lat_ok none / oracle-fg / causal-fg | completion h none / oracle-fg / causal-fg |
|---|---|---|---|---|---|
| risk_aware_rule_based | 0.0413 | 0.0378 | 0.0413 | 0.85 / 0.85 / 0.85 | 40.8 / 40.9 / 40.8 |
| rule_seasonal_forecast (hand-built) | 0.0412 | 0.0390 | 0.0411 | 0.85 / 0.85 / 0.85 | 39.0 / 39.1 / 39.2 |
| forecast_ppo_bc seed0 | 0.0406 | 0.0389 | 0.0404 | 0.85 / 0.85 / 0.85 | 40.4 / 40.4 / 40.7 |
| forecast_ppo_bc seed1 | 0.0406 | 0.0389 | 0.0404 | 0.85 / 0.85 / 0.85 | 40.7 / 40.7 / 40.9 |
| forecast_ppo_bc seed2 | 0.0409 | 0.0390 | 0.0401 | 0.85 / 0.85 / 0.85 | 40.5 / 40.6 / 40.8 |

Worst single-episode reward over these policies: none -2075.7, oracle-fg -2055.2, causal-fg -2065.8.

## Condition `idle`

| policy | reward none | reward oracle-fg | reward causal-fg | gap none | gap oracle-fg | gap causal-fg | better none | better oracle-fg | better causal-fg |
|---|---|---|---|---|---|---|---|---|---|
| risk_aware_rule_based | -33.4 | -33.4 | -33.4 | - | - | - | - | - | - |
| rule_seasonal_forecast (hand-built) | -33.4 | -33.4 | -33.4 | +0.00 (0.00) | +0.00 (0.00) | +0.00 (0.00) | 0/20 | 0/20 | 0/20 |
| forecast_ppo_bc seed0 | -33.4 | -33.4 | -33.4 | +0.00 (0.00) | +0.00 (0.00) | +0.00 (0.00) | 0/20 | 0/20 | 0/20 |
| forecast_ppo_bc seed1 | -33.4 | -33.4 | -33.4 | +0.00 (0.00) | +0.00 (0.00) | +0.00 (0.00) | 0/20 | 0/20 | 0/20 |
| forecast_ppo_bc seed2 | -33.4 | -33.4 | -33.4 | +0.00 (0.00) | +0.00 (0.00) | +0.00 (0.00) | 0/20 | 0/20 | 0/20 |

| policy | viol none | viol oracle-fg | viol causal-fg | lat_ok none / oracle-fg / causal-fg | completion h none / oracle-fg / causal-fg |
|---|---|---|---|---|---|
| risk_aware_rule_based | 0.0000 | 0.0000 | 0.0000 | 1.00 / 1.00 / 1.00 | 27.8 / 27.8 / 27.8 |
| rule_seasonal_forecast (hand-built) | 0.0000 | 0.0000 | 0.0000 | 1.00 / 1.00 / 1.00 | 27.8 / 27.8 / 27.8 |
| forecast_ppo_bc seed0 | 0.0000 | 0.0000 | 0.0000 | 1.00 / 1.00 / 1.00 | 27.8 / 27.8 / 27.8 |
| forecast_ppo_bc seed1 | 0.0000 | 0.0000 | 0.0000 | 1.00 / 1.00 / 1.00 | 27.8 / 27.8 / 27.8 |
| forecast_ppo_bc seed2 | 0.0000 | 0.0000 | 0.0000 | 1.00 / 1.00 / 1.00 | 27.8 / 27.8 / 27.8 |

Worst single-episode reward over these policies: none -33.4, oracle-fg -33.4, causal-fg -33.4.

## Condition `sigma_0`

| policy | reward none | reward oracle-fg | reward causal-fg | gap none | gap oracle-fg | gap causal-fg | better none | better oracle-fg | better causal-fg |
|---|---|---|---|---|---|---|---|---|---|
| risk_aware_rule_based | -330.2 | -326.8 | -330.2 | - | - | - | - | - | - |
| rule_seasonal_forecast (hand-built) | -326.5 | -323.1 | -326.5 | +3.68 (0.28) | +3.69 (0.27) | +3.68 (0.27) | 20/20 | 20/20 | 20/20 |
| forecast_ppo_bc seed0 | -324.0 | -323.4 | -323.9 | +6.24 (1.42) | +3.35 (0.27) | +6.31 (1.42) | 19/20 | 20/20 | 19/20 |
| forecast_ppo_bc seed1 | -323.9 | -323.4 | -323.9 | +6.32 (1.42) | +3.43 (0.27) | +6.33 (1.42) | 19/20 | 20/20 | 19/20 |
| forecast_ppo_bc seed2 | -323.5 | -323.5 | -323.4 | +6.71 (1.26) | +3.30 (0.24) | +6.83 (1.26) | 20/20 | 20/20 | 20/20 |

| policy | viol none | viol oracle-fg | viol causal-fg | lat_ok none / oracle-fg / causal-fg | completion h none / oracle-fg / causal-fg |
|---|---|---|---|---|---|
| risk_aware_rule_based | 0.0444 | 0.0438 | 0.0444 | 0.85 / 0.85 / 0.85 | 43.6 / 43.6 / 43.6 |
| rule_seasonal_forecast (hand-built) | 0.0462 | 0.0456 | 0.0461 | 0.85 / 0.85 / 0.85 | 40.6 / 40.6 / 40.7 |
| forecast_ppo_bc seed0 | 0.0456 | 0.0455 | 0.0452 | 0.85 / 0.85 / 0.85 | 41.4 / 41.4 / 41.8 |
| forecast_ppo_bc seed1 | 0.0454 | 0.0454 | 0.0451 | 0.85 / 0.85 / 0.85 | 41.6 / 41.6 / 41.9 |
| forecast_ppo_bc seed2 | 0.0455 | 0.0455 | 0.0451 | 0.85 / 0.85 / 0.85 | 41.5 / 41.5 / 41.8 |

Worst single-episode reward over these policies: none -2782.7, oracle-fg -2782.7, causal-fg -2782.7.

## Condition `sigma_0.5`

| policy | reward none | reward oracle-fg | reward causal-fg | gap none | gap oracle-fg | gap causal-fg | better none | better oracle-fg | better causal-fg |
|---|---|---|---|---|---|---|---|---|---|
| risk_aware_rule_based | -330.2 | -326.8 | -330.2 | - | - | - | - | - | - |
| rule_seasonal_forecast (hand-built) | -326.5 | -323.1 | -326.5 | +3.68 (0.28) | +3.69 (0.27) | +3.68 (0.27) | 20/20 | 20/20 | 20/20 |
| forecast_ppo_bc seed0 | -324.0 | -323.4 | -323.9 | +6.24 (1.42) | +3.35 (0.27) | +6.31 (1.42) | 19/20 | 20/20 | 19/20 |
| forecast_ppo_bc seed1 | -323.9 | -323.4 | -323.9 | +6.32 (1.42) | +3.43 (0.27) | +6.33 (1.42) | 19/20 | 20/20 | 19/20 |
| forecast_ppo_bc seed2 | -323.5 | -323.5 | -323.4 | +6.71 (1.26) | +3.30 (0.24) | +6.83 (1.26) | 20/20 | 20/20 | 20/20 |

| policy | viol none | viol oracle-fg | viol causal-fg | lat_ok none / oracle-fg / causal-fg | completion h none / oracle-fg / causal-fg |
|---|---|---|---|---|---|
| risk_aware_rule_based | 0.0444 | 0.0438 | 0.0444 | 0.85 / 0.85 / 0.85 | 43.6 / 43.6 / 43.6 |
| rule_seasonal_forecast (hand-built) | 0.0462 | 0.0456 | 0.0461 | 0.85 / 0.85 / 0.85 | 40.6 / 40.6 / 40.7 |
| forecast_ppo_bc seed0 | 0.0456 | 0.0455 | 0.0452 | 0.85 / 0.85 / 0.85 | 41.4 / 41.4 / 41.8 |
| forecast_ppo_bc seed1 | 0.0454 | 0.0454 | 0.0451 | 0.85 / 0.85 / 0.85 | 41.6 / 41.6 / 41.9 |
| forecast_ppo_bc seed2 | 0.0455 | 0.0455 | 0.0451 | 0.85 / 0.85 / 0.85 | 41.5 / 41.5 / 41.8 |

Worst single-episode reward over these policies: none -2782.7, oracle-fg -2782.7, causal-fg -2782.7.

## Condition `degraded_fleet`

| policy | reward none | reward oracle-fg | reward causal-fg | gap none | gap oracle-fg | gap causal-fg | better none | better oracle-fg | better causal-fg |
|---|---|---|---|---|---|---|---|---|---|
| risk_aware_rule_based | -683.7 | -682.0 | -683.7 | - | - | - | - | - | - |
| rule_seasonal_forecast (hand-built) | -680.3 | -678.6 | -680.2 | +3.45 (0.29) | +3.46 (0.29) | +3.46 (0.28) | 20/20 | 20/20 | 20/20 |
| forecast_ppo_bc seed0 | -679.6 | -679.1 | -679.4 | +4.08 (1.15) | +2.96 (0.26) | +4.34 (1.15) | 19/20 | 20/20 | 19/20 |
| forecast_ppo_bc seed1 | -680.0 | -678.4 | -679.9 | +3.67 (1.29) | +3.65 (0.57) | +3.85 (1.26) | 18/20 | 20/20 | 18/20 |
| forecast_ppo_bc seed2 | -678.9 | -678.4 | -678.8 | +4.76 (0.95) | +3.60 (0.59) | +4.95 (0.96) | 20/20 | 20/20 | 20/20 |

| policy | viol none | viol oracle-fg | viol causal-fg | lat_ok none / oracle-fg / causal-fg | completion h none / oracle-fg / causal-fg |
|---|---|---|---|---|---|
| risk_aware_rule_based | 0.0979 | 0.0976 | 0.0979 | 0.70 / 0.70 / 0.70 | 42.1 / 42.1 / 42.1 |
| rule_seasonal_forecast (hand-built) | 0.1027 | 0.1024 | 0.1025 | 0.65 / 0.65 / 0.65 | 39.2 / 39.2 / 39.4 |
| forecast_ppo_bc seed0 | 0.1026 | 0.1025 | 0.1021 | 0.65 / 0.65 / 0.65 | 39.8 / 39.8 / 40.2 |
| forecast_ppo_bc seed1 | 0.1024 | 0.1020 | 0.1019 | 0.65 / 0.70 / 0.70 | 40.0 / 40.0 / 40.3 |
| forecast_ppo_bc seed2 | 0.1024 | 0.1023 | 0.1019 | 0.65 / 0.70 / 0.70 | 39.9 / 40.0 / 40.3 |

Worst single-episode reward over these policies: none -2782.7, oracle-fg -2782.7, causal-fg -2782.7.

