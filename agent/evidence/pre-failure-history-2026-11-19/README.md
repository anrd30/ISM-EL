# History on versus off (holdout and altB)

`history_on_vs_off.csv`: per profile, policy (`risk_aware_rule_based`, `fixed_50`) and seed 0 to 19, the start row and the
total reward with `SimConfig.history_rows` 0 and 576 (evaluation configuration `eval-fleet,sigma=0.25`). `summary.txt`: the
seeds whose start row changes and the reward change on them. Seeds whose start row is unchanged have identical rewards.
Profiles: holdout (1152 rows) and `profile_diurnal_long_altB.csv` of `agent/residual-q-altb-check-2026-11-19`
(4032 rows, sha256 `bf19b148bdd47546d1029e8365dfc530dae6635f402ebfa814782ac7073acd7e`). Reading: `agent/PRE_FAILURE_HISTORY_DECISION.md`.
