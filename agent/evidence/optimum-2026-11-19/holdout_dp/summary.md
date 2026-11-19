# Hindsight optimum against the rule (simulated), `eval-fleet,sigma=0.25`

| profile | policy | workload_synthetic | n | completed | horizon_expired | terminal_failure | completion_h | lat_deg_mean | lat_deg_p95 | viol_frac | lat_budget_met | p_fail_mean | p_fail_max | risk_viol | clip_frac | deliv_mbps | stall_frac | reward | reward_sem |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| diurnal_long_holdout | risk_aware_rule_based | false | 20 | 20 | 0 | 0 | 43.6 | 0.2315 | 0.85 | 0.04439 | 0.85 | 0.001079 | 0.002298 | 0.1 | 0.02963 | 77.33 | 0 | -330.2 | 157.9 |
| diurnal_long_holdout | ORACLE_lookahead (not deployable) | false | 20 | 20 | 0 | 0 | 36.86 | 0.2462 | 0.8837 | 0.04682 | 0.85 | 0.0009537 | 0.002293 | 0.1 | 0.02572 | 90.95 | 0 | -318.3 | 158.8 |
| diurnal_long_holdout | DP_hindsight_optimum (not deployable) | false | 20 | 20 | 0 | 0 | 37.52 | 0.2399 | 0.8501 | 0.04638 | 0.85 | 0.0009667 | 0.002296 | 0.1 | 0.09878 | 89.28 | 0 | -317.3 | 158.8 |

ORACLE rows read the next interval's foreground load from the trace; no deployable controller can.

## ORACLE_lookahead (not deployable) minus risk_aware_rule_based, by definition of controllable

| profile | definition | n_controllable | mean_diff | sem | t | better | worse |
|---|---|---|---|---|---|---|---|
| diurnal_long_holdout | primary | 16 | 13.75 | 1.472 | 9.345 | 16 | 0 |
| diurnal_long_holdout | clip_0.01 | 13 | 13.25 | 1.618 | 8.188 | 13 | 0 |
| diurnal_long_holdout | clip_0.25 | 16 | 13.75 | 1.472 | 9.345 | 16 | 0 |
| diurnal_long_holdout | fixed20_cost | 17 | 13.51 | 1.404 | 9.621 | 17 | 0 |

## DP_hindsight_optimum (not deployable) minus risk_aware_rule_based, by definition of controllable

| profile | definition | n_controllable | mean_diff | sem | t | better | worse |
|---|---|---|---|---|---|---|---|
| diurnal_long_holdout | primary | 16 | 14.84 | 1.453 | 10.21 | 16 | 0 |
| diurnal_long_holdout | clip_0.01 | 13 | 14.34 | 1.591 | 9.017 | 13 | 0 |
| diurnal_long_holdout | clip_0.25 | 16 | 14.84 | 1.453 | 10.21 | 16 | 0 |
| diurnal_long_holdout | fixed20_cost | 17 | 14.58 | 1.388 | 10.51 | 17 | 0 |

## Share of the hindsight optimum captured by `ORACLE_lookahead`

| definition | n_controllable | dp_minus_rule | oracle_minus_rule | oracle_share_of_optimum |
|---|---|---|---|---|
| primary | 16 | 14.84 | 13.75 | 0.927 |
| clip_0.01 | 13 | 14.34 | 13.25 | 0.9236 |
| clip_0.25 | 16 | 14.84 | 13.75 | 0.927 |
| fixed20_cost | 17 | 14.58 | 13.51 | 0.9262 |
