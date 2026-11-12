# Exogenous violation fraction

- evaluation directory: agent/evidence/b2-244-clean-2026-11-12
- workload: data_pipeline/profiles/profile_diurnal_long_holdout.csv
- fixed_20 validation, max |exogenous_fraction - latency_violation_fraction|: 4.212e-01

| policy | exogenous_fraction | policy_violation_fraction | marginal_fraction |
|---|---|---|---|
| fixed_20 | 0.000000 | 0.043559 | 0.043559 |
| fixed_50 | 0.000000 | 0.050797 | 0.050797 |
| fixed_80 | 0.000000 | 0.086925 | 0.086925 |
| fixed_120 | 0.000000 | 0.672634 | 0.672634 |
| workload_only_throttle | 0.000000 | 0.058419 | 0.058419 |
| risk_aware_rule_based | 0.000000 | 0.058419 | 0.058419 |
| ORACLE_lookahead (not deployable) | 0.000000 | 0.060824 | 0.060824 |
| slowest_risk_feasible (literal spec) | 0.000000 | 0.043559 | 0.043559 |
| risk_aware_ceiling_persistence (literal spec) | 0.000000 | 0.107260 | 0.107260 |
| risk_aware_rule_based k=1 | 0.000000 | 0.072963 | 0.072963 |
| ppo_lagrangian seed0 | 0.000000 | 0.050797 | 0.050797 |
| ppo_lagrangian seed1 | 0.000000 | 0.051074 | 0.051074 |
| ppo_lagrangian seed2 | 0.000000 | 0.050322 | 0.050322 |
