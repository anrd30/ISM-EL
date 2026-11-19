# Hindsight optimum and value-of-information sweep (holdout and altB)

`<profile>_dp/`: `python -m baselines.dp_optimum --workload-csv <csv> --seeds 20 --eval-fleet --hazard-error-sigma 0.25 --out <dir>`
(rule, `ORACLE_lookahead`, `DP_hindsight_optimum (not deployable)`; `episodes.csv`, `summary.md`).

`<profile>_voi/`: `python -m agent.voi_sweep --workload-csv <csv> --sigmas 0,2,5,7,10,15 --seeds 20 --eval-fleet --hazard-error-sigma 0.25 --exclude-seeds <list> --out <dir>`
with `--exclude-seeds 9,17,19` (holdout) and `9,17` (altB); `voi_sweep.csv`, `episodes.csv`, `summary.md`.

Profiles: `data_pipeline/profiles/profile_diurnal_long_holdout.csv` (sha256 `94fe6bd6f8a5cb163cba7a9fafac1274d4ec6511113920a63d4993f03295a53d`)
and `profile_diurnal_long_altB.csv` of `agent/residual-q-altb-check-2026-11-19`
(sha256 `bf19b148bdd47546d1029e8365dfc530dae6635f402ebfa814782ac7073acd7e`). Tables and reading: `agent/OPTIMUM_DECISION.md`.
