# Pre-registration: the predictive controller on two fresh populations

Written and committed before any policy is evaluated on population C or D, and before C or D is built. Nothing
below changes after this commit. The criterion was written after the controller's results on populations A (the
44-disk holdout) and B (altB) were known; C and D have not been evaluated by any policy.

## What is tested

`predictive_controller (history)` (`agent/predictive_controller.py`, parameters as committed, not tuned) against
`risk_aware_rule_based`, on populations the controller, its parameters and every earlier selection never touched.

## Populations (14 days, 4032 rows each, built with the `build_profile_altB.py` construction; recorded in `PREPROCESSING.md` section B5c)

- **A and B** (already evaluated; reported for reference, not part of the criterion): `profile_diurnal_long_holdout.csv`
  and `profile_diurnal_long_altB.csv`.
- **C.** The diurnal rule of `build_profiles.py` (highest `daily_strength`, the median-bytes-per-I/O eligibility rule,
  N the largest whose aggregate p95 is at most 100 MB/s x 1.005, six-column replay schema, `t_seconds` from 0,
  `validate_replay_frame`). Window: the best-covered 14-day window (most volumes with a gap-free run of 4032 samples,
  earliest start on ties) that shares no sample time with A or B. Volumes: those of A (44) and B (150) are removed
  from the pool before ranking.
- **D.** Same eligibility rule and window rule, with the window disjoint in time from A, B and C and the volumes
  disjoint from A, B and C. The volumes are not ranked by the diurnal rule: the eligible volumes are put in a random
  order with `numpy.random.default_rng(20261119)` (a permutation of the sorted `disk_uid` list), and N is the largest
  number of volumes of that order whose aggregate p95 is at most 100 MB/s x 1.005 (the `tune_n` rule `not_above`
  applied over every prefix, as for the other profiles). No disk is skipped or swapped.
- Each profile gets a `.meta.json` sidecar with the seed, N, window, volume list, p95, mean and sha256, a `B5c`
  section in `PREPROCESSING.md`, and manifest entries. Only the small CSVs and sidecars are committed.

## Evaluation protocol (once)

- Configuration `v1,eval-fleet,sigma=0.25`, `SimConfig.history_rows = 576` (contract 1.5.0), seeds 0 to 19, the same
  scenarios for every policy.
- Policies: `fixed_20`, `fixed_50`, `fixed_80`, `fixed_120`, `risk_aware_rule_based`, `rule_seasonal_forecast
  (hand-built)`, the residual Q-policy models of `agent/evidence/residual-q-2026-11-14/training/main/seed{0,1,2}`
  zero-shot (comparator, seeds averaged within each scenario), `rule+history`, `predictive_controller[seasonal]+history`,
  `predictive_controller[adaptive] (no history)`, `predictive_controller (history)`, `ORACLE_lookahead (not
  deployable)`, `DP_hindsight_optimum (not deployable)`. Run by `python -m agent.evaluate_predictive`.
- One evaluation per population. If a run fails for a technical reason (a crash), the identical configuration is
  re-run after the fault is fixed, and the fault and the fix are recorded; no setting, seed, policy or definition is
  changed.

## Scenario classes

- **Controllable (primary definition).** A scenario is filter-forced when the safety filter raises a `fixed_20`
  rollout's applied action on at least 10% of the intervals (`FILTER_FORCED_CLIP_FRACTION`); controllable is the rest.
- **Sensitivity variants**, reported beside every primary figure: clip fraction at least 1%, at least 25%, and
  `fixed_20` latency cost above 50.

## Success criterion

On each of C and D, on the scenarios controllable under the primary definition, with `gain_s` the total reward of
`predictive_controller (history)` minus that of `risk_aware_rule_based` in scenario `s`, `n` the number of
controllable scenarios, `mean` and `sem` = sample standard deviation / sqrt(n) of `gain_s`:

1. `t = mean / sem` is greater than 2; and
2. the mean over the same scenarios of (`latency_violation_fraction` of the controller minus that of the rule) is
   not greater than 0.

The criterion is met only if both conditions hold on both C and D. If fewer than 8 scenarios of a population are
controllable, that population does not meet the criterion. The verdict reports, per population, `n`, the number
better and worse, `mean`, `sem`, `t` and the violation difference with its standard error, and states PASS or FAIL
against the sentence above, verbatim.

## Secondary and reported alongside

- The share of the optimum captured: `mean gain / mean(DP_hindsight_optimum minus rule)` on the same scenarios.
- The same figures under each sensitivity variant, and on all scenarios (the all-scenario verdict is reported; it is
  not the criterion).
- The residual Q-policy comparator and the fairness arms (`rule+history`, seasonal-only with history, controller
  without history), the gain of every policy over the rule, and the oracle and optimum references.

## Known limits recorded in advance

- Populations C and D are 14-day windows of the same Tencent trace as A and B; they share the data source and the
  sampling interval. D is the least selected population, a random sample of eligible volumes, whose aggregate may have
  a weak daily cycle.
- The simulator's latency model is uncalibrated (`sim/LIMITATIONS.md`); the figures are properties of the model.
