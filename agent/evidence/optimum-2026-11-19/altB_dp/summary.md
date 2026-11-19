# Hindsight optimum against the rule (simulated), `eval-fleet,sigma=0.25`

| profile | policy | workload_synthetic | n | completed | horizon_expired | terminal_failure | completion_h | lat_deg_mean | lat_deg_p95 | viol_frac | lat_budget_met | p_fail_mean | p_fail_max | risk_viol | clip_frac | deliv_mbps | stall_frac | reward | reward_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_altB | risk_aware_rule_based | false | 20 | 20 | 0 | 0 | 39.68 | 0.0774 | 0.395 | 0.02261 | 0.9 | 0.001029 | 0.002224 | 0.1 | 0.02149 | 84.87 | 0 | -172.9 | 67.8 |
| diurnal_long_altB | ORACLE_lookahead (not deployable) | false | 20 | 20 | 0 | 0 | 35.5 | 0.07352 | 0.431 | 0.01891 | 0.9 | 0.0009611 | 0.002201 | 0.1 | 0.01189 | 94.38 | 0 | -143.7 | 68.58 |
| diurnal_long_altB | DP_hindsight_optimum (not deployable) | false | 20 | 20 | 0 | 0 | 36.07 | 0.06875 | 0.3955 | 0.01876 | 0.9 | 0.0009743 | 0.002235 | 0.1 | 0.1222 | 92.81 | 0 | -142.5 | 68.63 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## ORACLE_lookahead (not deployable) minus risk_aware_rule_based, by definition of controllable

| profile | definition | n_controllable | mean_diff | sem | t | better | worse |
|---|---|---|---|---|---|---|---|
| diurnal_long_altB | primary | 16 | 33.16 | 2.716 | 12.21 | 16 | 0 |
| diurnal_long_altB | clip_0.01 | 13 | 31.59 | 3.195 | 9.887 | 13 | 0 |
| diurnal_long_altB | clip_0.25 | 16 | 33.16 | 2.716 | 12.21 | 16 | 0 |
| diurnal_long_altB | fixed20_cost | 17 | 32.3 | 2.695 | 11.98 | 17 | 0 |

## DP_hindsight_optimum (not deployable) minus risk_aware_rule_based, by definition of controllable

| profile | definition | n_controllable | mean_diff | sem | t | better | worse |
|---|---|---|---|---|---|---|---|
| diurnal_long_altB | primary | 16 | 34.4 | 2.743 | 12.54 | 16 | 0 |
| diurnal_long_altB | clip_0.01 | 13 | 32.81 | 3.226 | 10.17 | 13 | 0 |
| diurnal_long_altB | clip_0.25 | 16 | 34.4 | 2.743 | 12.54 | 16 | 0 |
| diurnal_long_altB | fixed20_cost | 17 | 33.53 | 2.72 | 12.33 | 17 | 0 |

## Share of the hindsight optimum captured by `ORACLE_lookahead`

| definition | n_controllable | dp_minus_rule | oracle_minus_rule | oracle_share_of_optimum |
|---|---|---|---|---|
| primary | 16 | 34.4 | 33.16 | 0.9641 |
| clip_0.01 | 13 | 32.81 | 31.59 | 0.9627 |
| clip_0.25 | 16 | 34.4 | 33.16 | 0.9641 |
| fixed20_cost | 17 | 33.53 | 32.3 | 0.9633 |
