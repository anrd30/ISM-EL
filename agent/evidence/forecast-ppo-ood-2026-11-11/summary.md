# Forecast PPO agent outside its training distribution (simulated)

Fitted risk model, held-out fleet table, 20 paired scenario seeds per condition. `gap` is total reward minus `risk_aware_rule_based`, mean over paired scenarios; conditions are defined in `agent/evaluate_ood.py`.

| condition | policy | n | rule_reward | gap | sem | n_better | viol_frac | rule_viol_frac |
|---|---|---|---|---|---|---|---|---|
| synthetic_idle_4day | ORACLE_lookahead (not deployable) | 20 | -33.4 | 0 | 0 | 0 | 0 | 0 |
| synthetic_idle_4day | rule_seasonal_forecast (hand-built) | 20 | -33.4 | 0 | 0 | 0 | 0 | 0 |
| synthetic_idle_4day | forecast_ppo_bc seed0 | 20 | -33.4 | 0 | 0 | 0 | 0 | 0 |
| synthetic_idle_4day | forecast_ppo_bc seed1 | 20 | -33.4 | 0 | 0 | 0 | 0 | 0 |
| synthetic_idle_4day | forecast_ppo_bc seed2 | 20 | -33.4 | 0 | 0 | 0 | 0 | 0 |
| synthetic_diurnal_4day | ORACLE_lookahead (not deployable) | 20 | -186.4 | 1.053 | 0.5583 | 16 | 0.02511 | 0.02484 |
| synthetic_diurnal_4day | rule_seasonal_forecast (hand-built) | 20 | -186.4 | -4.515 | 1.482 | 5 | 0.02595 | 0.02484 |
| synthetic_diurnal_4day | forecast_ppo_bc seed0 | 20 | -186.4 | -23.43 | 7.47 | 2 | 0.02898 | 0.02484 |
| synthetic_diurnal_4day | forecast_ppo_bc seed1 | 20 | -186.4 | -2.535 | 1.091 | 2 | 0.02459 | 0.02484 |
| synthetic_diurnal_4day | forecast_ppo_bc seed2 | 20 | -186.4 | -13.86 | 5.276 | 1 | 0.02674 | 0.02484 |
| synthetic_bursty_4day | ORACLE_lookahead (not deployable) | 20 | -311.3 | 186.8 | 11.19 | 20 | 0.0171 | 0.01808 |
| synthetic_bursty_4day | rule_seasonal_forecast (hand-built) | 20 | -311.3 | -37.82 | 6.561 | 0 | 0.02052 | 0.01808 |
| synthetic_bursty_4day | forecast_ppo_bc seed0 | 20 | -311.3 | -196.4 | 20.44 | 0 | 0.08419 | 0.01808 |
| synthetic_bursty_4day | forecast_ppo_bc seed1 | 20 | -311.3 | -174.3 | 16.98 | 0 | 0.06891 | 0.01808 |
| synthetic_bursty_4day | forecast_ppo_bc seed2 | 20 | -311.3 | -190.3 | 19.72 | 0 | 0.06766 | 0.01808 |
| synthetic_peak_4day | ORACLE_lookahead (not deployable) | 20 | -2952 | 5.405 | 1.728 | 19 | 0.1617 | 0.1613 |
| synthetic_peak_4day | rule_seasonal_forecast (hand-built) | 20 | -2952 | -0.7783 | 0.7783 | 0 | 0.1613 | 0.1613 |
| synthetic_peak_4day | forecast_ppo_bc seed0 | 20 | -2952 | -9283 | 847.4 | 0 | 0.9766 | 0.1613 |
| synthetic_peak_4day | forecast_ppo_bc seed1 | 20 | -2952 | -9283 | 847.4 | 0 | 0.9766 | 0.1613 |
| synthetic_peak_4day | forecast_ppo_bc seed2 | 20 | -2952 | -9283 | 847.4 | 0 | 0.9766 | 0.1613 |
| holdout_phase_shift | ORACLE_lookahead (not deployable) | 20 | -285.8 | 19.41 | 2.377 | 20 | 0.03749 | 0.0355 |
| holdout_phase_shift | rule_seasonal_forecast (hand-built) | 20 | -285.8 | -3.182 | 2.624 | 15 | 0.03808 | 0.0355 |
| holdout_phase_shift | forecast_ppo_bc seed0 | 20 | -285.8 | -35.4 | 8.071 | 5 | 0.04415 | 0.0355 |
| holdout_phase_shift | forecast_ppo_bc seed1 | 20 | -285.8 | -10.47 | 4.386 | 7 | 0.03864 | 0.0355 |
| holdout_phase_shift | forecast_ppo_bc seed2 | 20 | -285.8 | -17.71 | 5.377 | 6 | 0.03974 | 0.0355 |
| holdout_sigma_0 | ORACLE_lookahead (not deployable) | 20 | -330.2 | 11.85 | 1.496 | 20 | 0.04682 | 0.04439 |
| holdout_sigma_0 | rule_seasonal_forecast (hand-built) | 20 | -330.2 | 3.685 | 0.2758 | 20 | 0.0462 | 0.04439 |
| holdout_sigma_0 | forecast_ppo_bc seed0 | 20 | -330.2 | 6.241 | 1.419 | 19 | 0.04564 | 0.04439 |
| holdout_sigma_0 | forecast_ppo_bc seed1 | 20 | -330.2 | 6.324 | 1.423 | 19 | 0.04545 | 0.04439 |
| holdout_sigma_0 | forecast_ppo_bc seed2 | 20 | -330.2 | 6.715 | 1.262 | 20 | 0.04548 | 0.04439 |
| holdout_sigma_0.5 | ORACLE_lookahead (not deployable) | 20 | -330.2 | 11.85 | 1.496 | 20 | 0.04682 | 0.04439 |
| holdout_sigma_0.5 | rule_seasonal_forecast (hand-built) | 20 | -330.2 | 3.685 | 0.2758 | 20 | 0.0462 | 0.04439 |
| holdout_sigma_0.5 | forecast_ppo_bc seed0 | 20 | -330.2 | 6.241 | 1.419 | 19 | 0.04564 | 0.04439 |
| holdout_sigma_0.5 | forecast_ppo_bc seed1 | 20 | -330.2 | 6.324 | 1.423 | 19 | 0.04545 | 0.04439 |
| holdout_sigma_0.5 | forecast_ppo_bc seed2 | 20 | -330.2 | 6.715 | 1.262 | 20 | 0.04548 | 0.04439 |
| holdout_degraded_fleet | ORACLE_lookahead (not deployable) | 20 | -634.8 | 9.492 | 1.429 | 20 | 0.09801 | 0.0915 |
| holdout_degraded_fleet | rule_seasonal_forecast (hand-built) | 20 | -634.8 | 3.503 | 0.3049 | 20 | 0.09581 | 0.0915 |
| holdout_degraded_fleet | forecast_ppo_bc seed0 | 20 | -634.8 | 4.16 | 1.139 | 19 | 0.09574 | 0.0915 |
| holdout_degraded_fleet | forecast_ppo_bc seed1 | 20 | -634.8 | 4.277 | 1.146 | 19 | 0.09536 | 0.0915 |
| holdout_degraded_fleet | forecast_ppo_bc seed2 | 20 | -634.8 | 4.806 | 0.9459 | 20 | 0.09541 | 0.0915 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.
