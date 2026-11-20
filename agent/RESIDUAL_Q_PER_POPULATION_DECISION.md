# Decision record: residual Q retrained on each population's own training split (post-registration comparator)

Status: proposed. Not yet reviewed by the other lanes. **Post-registration:** this evaluation was run after the pre-registered
evaluation of `agent/PREREGISTRATION_2026-11-19.md` (commit `e32dcfd2a7f0be7a2994d14b68d62e632f62d2a9`) and does not alter its verdict
(PASS on populations C and D). The comparator of that evaluation is now named "residual-Q (zero-shot transfer from A)".

Code: `agent/evaluate_residual_q_per_population.py`; `data_pipeline/tencent/split_long_profile.py` (the splitter of `diurnal_long`,
output names taken from the source's stem); tests in `data_pipeline/tests/test_pop_splits.py`. Evidence:
`agent/evidence/residual-q-per-population-2026-11-20/` (`training/`, `eval_popC/`, `eval_popD/`). Splits: `PREPROCESSING.md` section B5d,
`data_pipeline/manifest.json`.

## Decision

- **Splits.** Populations C and D are cut as `diurnal_long` is: the first 10 of 14 days (rows 0 to 2879) are the train half, the last 4
  days (rows 2880 to 4031) the holdout half, each rebased to `t_seconds` 0 and validated; hash-pinned.
- **Retraining.** `python -m agent.residual_q --seeds 0 1 2 --workload-csv profile_diurnal_long_<pop>_train.csv --hazard-error-sigma 0.25`,
  exactly the configuration of the main run on A (80 episodes on the first 7 training days, exploration 0.1, 3 inner validation days,
  40 validation scenarios, margin chosen from {0.1, 0.25, 0.5} on the inner time-forward validation days, final refit on the whole
  training split, train-split fleets).
- **Evaluation.** On each holdout half: retrained residual-Q (3 models), residual-Q zero-shot transfer from A (the 3 models of the main
  run), `risk_aware_rule_based`, `rule_seasonal_forecast (hand-built)`, `predictive_controller (history)`, `ORACLE_lookahead`,
  `DP_hindsight_optimum`; evaluation fleet, sigma 0.25, history rows 576 (the holdout half has 1152 rows, so 576 start rows remain, as
  on population A's holdout), seeds 0 to 19, same scenarios for every arm; models averaged within each scenario first.

## Evidence

Selected margin and inner validation gap over `SeasonalForecastRule` (`training/*/seed*/meta.json`): C seeds 0, 1, 2: margin 0.5 for all,
gap +0.59, -6.30, -2.08; D: margin 0.5 for all, gap +16.04, +11.76, +12.22.

Gain over `risk_aware_rule_based` on the 16 scenarios controllable under the primary definition (filter-forced seeds {3, 9, 17, 19} on both
holdouts), mean +/- sem across scenarios, `t`, better / worse; and the share of the optimum's gain:

| Policy | C holdout | D holdout |
| --- | --- | --- |
| `rule_seasonal_forecast` | +3.772 +/- 0.312 (12.08; 16/0) | -5.929 +/- 4.025 (-1.47; 3/13) |
| `predictive_controller (history)` | +9.782 +/- 0.746 (13.11; 16/0), 91.1% | +1.476 +/- 2.411 (0.61; 10/6), 6.7% |
| residual-Q retrained on its own training split | +2.425 +/- 0.415 (5.85; 16/0), 22.6% | +11.858 +/- 2.474 (4.79; 13/3), 54.1% |
| residual-Q zero-shot transfer from A | -12.426 +/- 1.473 (-8.44; 0/16) | -19.533 +/- 3.335 (-5.86; 0/16) |
| `ORACLE_lookahead` (not deployable) | +9.393 +/- 0.790 | +20.663 +/- 2.319 |
| `DP_hindsight_optimum` (not deployable) | +10.733 +/- 0.760 | +21.929 +/- 2.297 |

All 20 scenarios: C retrained +2.264 +/- 0.344 (t 6.58), zero-shot -11.291 +/- 1.648, controller +8.740 +/- 0.820; D retrained +10.571 +/- 2.314
(t 4.57), zero-shot -17.254 +/- 2.882, controller +2.912 +/- 2.506 (t 1.16).

- **Retraining removes the failure, not the gap.** The zero-shot models lose to the rule on both populations in every controllable scenario
  (0 of 16 better); retrained on the population's own training days they beat the rule on both (16 of 16 on C, 13 of 16 on D).
- **Retrained residual-Q against the seasonal rule.** On C it loses to the hand-built seasonal rule by 1.347 +/- 0.297 (0 of 16 better);
  on D it beats it by 17.787 +/- 4.318 (15 of 16), the seasonal rule being worse than the rule there.
- **Retrained residual-Q against the controller.** On C the controller is ahead by 7.4; on the D holdout half the retrained residual-Q is
  ahead by 10.4 and the controller's gain over the rule is not significant (t 0.61), unlike the registered evaluation on the full 14 days of
  D (`agent/FRESH_POPULATIONS_DECISION.md`, +17.293, t 4.48). The two evaluations differ in window (14 days against the last 4), in
  start rows (576 history rows remove half of the start rows of the 4-day file) and in the policy set; they were not designed to be
  compared.
- **Validation was not a guide on C.** The inner validation gaps on C were negative for two of three seeds even at the largest margin.

## Decisions made during execution

- **The splitter was extended, not duplicated:** the target file names come from the source's stem; the default call regenerates the
  committed `diurnal_long` halves byte for byte (tested).
- **Training output copied, not pointed to.** `model.pkl`, `meta.json` and `validation.csv` of each seed are committed under
  `training/<population>/seed<k>/`; the transient split directories are not.
- **Same evaluation protocol as the registered evaluation** (evaluation fleet, sigma 0.25, history rows 576), including the start-row
  remap, so that all arms share scenarios.
- **Comparator renamed** to "residual-Q (zero-shot transfer from A)" in `FRESH_POPULATIONS_DECISION.md`, the evidence summaries of the
  registered evaluation and `agent/evaluate_predictive.py`'s label; `PREREGISTRATION_2026-11-19.md` is unchanged.

## Limits

- **Within-population.** The retrained models are trained and tested on days of the same volumes (14 for C, 336 for D) of one 14-day window; this is
  the setting in which the residual-Q result on A looked good before the independent-population test (`agent/RESIDUAL_Q_DECISION.md`).
- **A 4-day holdout.** 20 scenarios on 1152 rows; the controllable class has 16 scenarios.
- **Post-registration, unregistered criterion.** No success criterion was fixed for this comparison; it is reported as measured.
- **Margins at the top of the grid.** Every selected margin is the largest of the grid (0.5), so the grid may be too narrow for these
  populations; it was not widened (the configuration is the main run's).
