# Forecast PPO agent outside its training distribution (simulated)

Fitted risk model, held-out fleet table, 20 paired scenario seeds per condition. `gap` is total reward minus `risk_aware_rule_based`, mean over paired scenarios; conditions are defined in `agent/evaluate_ood.py`.

| condition | policy | n | rule_reward | gap | sem | n_better | viol_frac | rule_viol_frac |
|---|---|---|---|---|---|---|---|---|
| synthetic_idle_4day | ORACLE_lookahead (not deployable) | 20 | -33.4 | 0 | 0 | 0 | 0 | 0 |
| synthetic_idle_4day | rule_seasonal_forecast (hand-built) | 20 | -33.4 | 0 | 0 | 0 | 0 | 0 |
| synthetic_idle_4day | residual_q seed0 | 20 | -33.4 | 0 | 0 | 0 | 0 | 0 |
| synthetic_idle_4day | residual_q seed1 | 20 | -33.4 | -0.005 | 0.005 | 0 | 0 | 0 |
| synthetic_idle_4day | residual_q seed2 | 20 | -33.4 | -0.045 | 0.01352 | 0 | 0 | 0 |
| synthetic_diurnal_4day | ORACLE_lookahead (not deployable) | 20 | -186.4 | 1.053 | 0.5583 | 16 | 0.02511 | 0.02484 |
| synthetic_diurnal_4day | rule_seasonal_forecast (hand-built) | 20 | -186.4 | -4.515 | 1.482 | 5 | 0.02595 | 0.02484 |
| synthetic_diurnal_4day | residual_q seed0 | 20 | -186.4 | -18.05 | 2.997 | 0 | 0.02845 | 0.02484 |
| synthetic_diurnal_4day | residual_q seed1 | 20 | -186.4 | -8.758 | 2.861 | 1 | 0.02675 | 0.02484 |
| synthetic_diurnal_4day | residual_q seed2 | 20 | -186.4 | -15.79 | 3.578 | 0 | 0.02794 | 0.02484 |
| synthetic_bursty_4day | ORACLE_lookahead (not deployable) | 20 | -311.3 | 186.8 | 11.19 | 20 | 0.0171 | 0.01808 |
| synthetic_bursty_4day | rule_seasonal_forecast (hand-built) | 20 | -311.3 | -37.82 | 6.561 | 0 | 0.02052 | 0.01808 |
| synthetic_bursty_4day | residual_q seed0 | 20 | -311.3 | -72.28 | 11.28 | 0 | 0.03655 | 0.01808 |
| synthetic_bursty_4day | residual_q seed1 | 20 | -311.3 | -42.99 | 8.815 | 3 | 0.02818 | 0.01808 |
| synthetic_bursty_4day | residual_q seed2 | 20 | -311.3 | -51.09 | 7.981 | 0 | 0.03 | 0.01808 |
| synthetic_peak_4day | ORACLE_lookahead (not deployable) | 20 | -2952 | 5.405 | 1.728 | 19 | 0.1617 | 0.1613 |
| synthetic_peak_4day | rule_seasonal_forecast (hand-built) | 20 | -2952 | -0.7783 | 0.7783 | 0 | 0.1613 | 0.1613 |
| synthetic_peak_4day | residual_q seed0 | 20 | -2952 | -33.51 | 3.558 | 1 | 0.1635 | 0.1613 |
| synthetic_peak_4day | residual_q seed1 | 20 | -2952 | -0.7783 | 0.7783 | 0 | 0.1613 | 0.1613 |
| synthetic_peak_4day | residual_q seed2 | 20 | -2952 | -1.297 | 0.9123 | 0 | 0.1613 | 0.1613 |
| holdout_phase_shift | ORACLE_lookahead (not deployable) | 20 | -285.8 | 19.41 | 2.377 | 20 | 0.03749 | 0.0355 |
| holdout_phase_shift | rule_seasonal_forecast (hand-built) | 20 | -285.8 | -3.182 | 2.624 | 15 | 0.03808 | 0.0355 |
| holdout_phase_shift | residual_q seed0 | 20 | -285.8 | 6.33 | 2.344 | 17 | 0.03779 | 0.0355 |
| holdout_phase_shift | residual_q seed1 | 20 | -285.8 | 3.279 | 2.524 | 15 | 0.03765 | 0.0355 |
| holdout_phase_shift | residual_q seed2 | 20 | -285.8 | 5.045 | 2.834 | 16 | 0.03801 | 0.0355 |
| holdout_degraded_fleet | ORACLE_lookahead (not deployable) | 20 | -634.8 | 9.492 | 1.429 | 20 | 0.09801 | 0.0915 |
| holdout_degraded_fleet | rule_seasonal_forecast (hand-built) | 20 | -634.8 | 3.503 | 0.3049 | 20 | 0.09581 | 0.0915 |
| holdout_degraded_fleet | residual_q seed0 | 20 | -634.8 | 6.45 | 1.226 | 20 | 0.09666 | 0.0915 |
| holdout_degraded_fleet | residual_q seed1 | 20 | -634.8 | 5.791 | 1.142 | 20 | 0.09588 | 0.0915 |
| holdout_degraded_fleet | residual_q seed2 | 20 | -634.8 | 6.745 | 1.211 | 20 | 0.09667 | 0.0915 |
| profile_bursty_long_holdout | ORACLE_lookahead (not deployable) | 20 | -223.9 | 17.22 | 3.32 | 20 | 0.02775 | 0.02754 |
| profile_bursty_long_holdout | rule_seasonal_forecast (hand-built) | 20 | -223.9 | -3.935 | 2.219 | 5 | 0.02831 | 0.02754 |
| profile_bursty_long_holdout | residual_q seed0 | 20 | -223.9 | -49.98 | 8.931 | 2 | 0.0357 | 0.02754 |
| profile_bursty_long_holdout | residual_q seed1 | 20 | -223.9 | -17.65 | 4.039 | 3 | 0.03097 | 0.02754 |
| profile_bursty_long_holdout | residual_q seed2 | 20 | -223.9 | -14.04 | 4.863 | 7 | 0.03023 | 0.02754 |
| profile_idle_long_holdout | ORACLE_lookahead (not deployable) | 20 | -33.4 | 0 | 0 | 0 | 0 | 0 |
| profile_idle_long_holdout | rule_seasonal_forecast (hand-built) | 20 | -33.4 | 0 | 0 | 0 | 0 | 0 |
| profile_idle_long_holdout | residual_q seed0 | 20 | -33.4 | 0 | 0 | 0 | 0 | 0 |
| profile_idle_long_holdout | residual_q seed1 | 20 | -33.4 | -0.005 | 0.005 | 0 | 0 | 0 |
| profile_idle_long_holdout | residual_q seed2 | 20 | -33.4 | -0.085 | 0.015 | 0 | 0 | 0 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.
