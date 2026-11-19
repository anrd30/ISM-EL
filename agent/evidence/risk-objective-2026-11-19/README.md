# Exposure-weighted objective: holdout and altB

    python -m agent.evaluate_risk_objective --workload-csv data_pipeline/profiles/profile_diurnal_long_holdout.csv --seeds 20 --forced-seeds 9,17,19 --out holdout
    python -m agent.evaluate_risk_objective --workload-csv data_pipeline/profiles/profile_diurnal_long_altB.csv --seeds 20 --forced-seeds 9,17 --out altB

Per profile: `episodes.csv` (environments `contract`, `objective`, `objective_unforced`), `summary.csv`, `gains.csv`,
`forced_trade.csv`, `summary.md`. Evaluation fleet, sigma 0.25, history rows 576; `hazard_ref` = 1.0253723e-4 per hour
(mean initial true group hazard, seeds 1000 to 1399). Profiles: `profile_diurnal_long_holdout.csv` (sha256
`94fe6bd6f8a5cb163cba7a9fafac1274d4ec6511113920a63d4993f03295a53d`), `profile_diurnal_long_altB.csv`
(sha256 `bf19b148bdd47546d1029e8365dfc530dae6635f402ebfa814782ac7073acd7e`). Reading: `agent/RISK_OBJECTIVE_DECISION.md`.
