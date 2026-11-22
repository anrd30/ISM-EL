# Exogenous cost trap: measurements

- `l3_stats/seed{0,1,2}/l3_stats.csv`: copies of `runs/l3_244/seed*/l3_stats.csv` (counters of `agent.train --marginal-violation-cost`).
- `holdout_altB/`: `python -m agent.exogenous_trap tables --episodes agent/evidence/scenario-split-2026-11-19/{holdout,altB}/episodes.csv --l3-stats l3_stats/seed*/l3_stats.csv --out holdout_altB`
  (`x_summary.csv`, `x_by_scenario.csv`, `eq3_check.csv`, `training_x.csv`).
- `training_config/`: `python -m baselines.evaluate --workload-csv data_pipeline/profiles/profile_diurnal_long_train.csv --seeds 20 --hazard-error-sigma 0.25 --diagnostics --out training_config`
  (the PPO training workload and configuration: train-split fleet, sigma 0.25); `training_config_tables/`: the same `tables` command on its `episodes.csv`.
- `compare_holdout/`: `python -m agent.exogenous_trap compare --workload-csv data_pipeline/profiles/profile_diurnal_long_holdout.csv --ppo-label ppo_clean=runs/b2_244_clean ppo_c_prime=runs/l3_244 --seeds 20 --hazard-error-sigma 0.25 --out compare_holdout`.
  Produced on the local merge of #34, #35, #36, #37 and #38 plus this branch (the controller and the optimum are in those branches); the PPO checkpoints are
  `runs/{b2_244_clean,l3_244}/seed*/ckpt_final.pt` (gitignored). Reading: `agent/EXOGENOUS_COST_TRAP_DECISION.md`.
- Integration commit the comparison was produced on: `b197c7dc099ed0e4061dba91839a7c528193adca`, pushed as the branch `integration/phase6-evidence-2026-11-19` (the fresh-populations branch, #38, with this branch merged in).
