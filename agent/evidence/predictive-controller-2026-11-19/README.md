# Predictive controller: reproduction and protocol runs

- `reproduce_prototype_scenarios.py`, `reproduce_holdout.{txt,csv}`, `reproduce_altB.{txt,csv}`: the controller and its
  arms on the original scenarios (history_rows = 0, start rows unchanged), history built by wrap-around from the
  private start row (the script says so). Run on the merge of #34, #35, #36 and this branch.
- `holdout/`, `altB/`: `python -m agent.evaluate_predictive --workload-csv <csv> --seeds 20 --eval-fleet --hazard-error-sigma 0.25 --out <dir>`
  (history_rows = 576; `episodes.csv`, `gaps.csv`, `summary.md`). `agent/evaluate_predictive.py` is introduced with the
  fresh-population evaluation; the files here were produced on the local merge of #34, #35, #36 and this branch.
- Profiles: `data_pipeline/profiles/profile_diurnal_long_holdout.csv` (sha256 `94fe6bd6f8a5cb163cba7a9fafac1274d4ec6511113920a63d4993f03295a53d`) and
  `profile_diurnal_long_altB.csv` of `agent/residual-q-altb-check-2026-11-19`
  (sha256 `bf19b148bdd47546d1029e8365dfc530dae6635f402ebfa814782ac7073acd7e`). Reading: `agent/PREDICTIVE_CONTROLLER_DECISION.md`.
- Integration commit the evidence was produced on: `8c13dbf6217b35324e8a5ae12e2aefc951fb7502`, pushed as the branch `integration/phase4-evidence-2026-11-19` (merge of #34, #35, #36 and this branch). `agent/evaluate_predictive.py` was not in that commit; the run used the file of commit `a524c43` (#38), added to the working tree.
