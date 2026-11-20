# Decision record: hazard model fitted without one drive vendor (vendor holdout)

Status: proposed. Not yet reviewed by the other lanes. Post-registration, unregistered criterion: no success threshold was fixed before the
run; results are reported as measured. No A1 to A5 artifact or record is changed (`hazard/artifacts/v1/`, `backblaze_drive_split.parquet`,
`hazard/A4_DECISION.md`, the committed fleet tables are untouched; the options added to `hazard/fit_v1.py` and
`hazard/build_fleet_features.py` are off by default and tested to reproduce the previous behaviour).

Code: `data_pipeline/backblaze/vendor_holdout_split.py`, `hazard/fit_v1.py` (`--drive-split --out --manifest`),
`hazard/evaluate_vendor_holdout.py`, `hazard/build_fleet_features.py` (`--drive-split --bucket --out`), `sim/config.py`
(`vendor_holdout_config`), `sim/scenario.py` (`v1_vendor_holdout_drive_features`), `agent/evaluate_vendor_holdout.py`. Tests:
`data_pipeline/tests/test_vendor_holdout_split.py`, `hazard/test_fit_v1_options.py`, `sim/test_vendor_holdout_config.py`. Evidence:
`agent/evidence/hazard-vendor-holdout-2026-11-20/`.

## Decision

- **Split.** Every TOSHIBA MG07ACA14TA drive is in a new bucket `vendor_holdout`. The three Seagate models are split by `labels.assign_splits`
  on their own drives (A2 seed and fractions), which reproduces A2's assignment of those drives (tested against the committed split).
- **Refit.** `python -m hazard.fit_v1 --drive-split ...vendor_holdout.parquet --out hazard/artifacts/v1_seagate_only --manifest hazard/manifest_v1_seagate_only.json`:
  25 boosters, same method and hyperparameters as A3 (no retuning), `model` categories ST12000NM0008, ST4000DM000, ST8000NM0055.
- **Evaluation.** Calibration and ranking on the Toshiba drives (`hazard.evaluate_vendor_holdout`); policy comparison under
  `vendor_holdout_config()` (Seagate-only hazard model, fleets drawn from a table built from the Toshiba drives) on the population-A holdout
  workload, with `agent.evaluate_vendor_holdout` (same arms, 20 seeds, sigma 0.25, history rows 576, pairing and statistics as
  `agent.evaluate_predictive`).

## Evidence

Split (drives / failed drives / positive label rows): train 37,763 / 970 / 28,902; val 8,092 / 208 / 6,008; test 8,091 / 207 / 6,045;
vendor_holdout 38,579 / 385 / 11,432 (`PREPROCESSING.md` A2v). Training set size is 970 failed drives, against 1,385 Seagate failed drives in A1.
Fit: 3,498 s on cuda; every booster used the 2,000-round cap; ensemble validation log-loss 0.012288, Brier 0.002048.

Hazard model on label rows, drive-bootstrap 95% intervals (200 resamples):

| Cell | Predictor | PR-AUC | Brier | Recall at 1% FPR | ROC-AUC |
| --- | --- | --- | --- | --- | --- |
| Toshiba, all 38,579 drives (385 failed; vendor holdout) | Seagate-only v1 | 0.0007 [0.0007, 0.0008] | 8.944e-4 [8.262e-4, 9.878e-4] | 0.011 | 0.424 |
| | `hazard/v0.py` | 0.0753 [0.0559, 0.0981] | 8.942e-4 | 0.377 | 0.758 |
| | constant base rate | 0.0009 | 8.940e-4 | none | |
| Toshiba, A2 test drives (58 failed) | Seagate-only v1 | 0.0007 [0.0006, 0.0010] | 8.909e-4 | 0.009 | 0.410 |
| | pooled v1 (A3/A4) | 0.0682 [0.0243, 0.1347] | 8.566e-4 [6.337e-4, 1.103e-3] | 0.251 | 0.744 |
| | `hazard/v0.py` | 0.0897 [0.0304, 0.1767] | 8.905e-4 | 0.421 | 0.762 |
| Seagate test drives, 207 failed (in-vendor control) | Seagate-only v1 | 0.1528 [0.1206, 0.2031] | 2.078e-3 | 0.420 | 0.814 |
| | pooled v1 | 0.1606 [0.1204, 0.2026] | 2.075e-3 | 0.446 | |
| | `hazard/v0.py` | 0.0687 | 2.257e-3 | 0.350 | 0.856 |

On the Toshiba drives the Seagate-only model's mean score (0.000891) equals the base rate (0.000895) but its ranking is inverted (ROC-AUC 0.42, below
0.5; PR-AUC below the base rate): its Brier score equals that of the constant base rate and the calibration advantage of the pooled model
(Brier 8.566e-4 against 8.909e-4 on the same 58-failed-drive cell; paired difference Seagate-only minus pooled +3.4e-5, interval [-6.7e-6, +7.2e-5], includes 0; PR-AUC difference -0.0675, interval [-0.1348, -0.0229]) does not appear. The pooled
model's ranking power on Toshiba (PR-AUC 0.068, ROC-AUC 0.744) is therefore due to Toshiba drives being in its training data, not to transfer
from the Seagate models. Not a bug (checks below): it is a model that does not generalise across vendors.

Policy comparison, gain in total reward over `risk_aware_rule_based` on the scenarios controllable under the primary definition (mean +/- sem, t,
better/worse, share of the DP optimum's gain):

| Policy | Population A, pooled hazard (existing; 16 scenarios) | Pooled hazard, Toshiba fleets (control; 12) | Seagate-only hazard, Toshiba fleets (20) |
| --- | --- | --- | --- |
| `rule_seasonal_forecast` | +3.991 +/- 0.203 (19.69; 16/0), 24.9% | +3.879 +/- 0.225 (17.26; 12/0), 24.5% | +3.987 +/- 0.179 (22.26; 20/0), 25.7% |
| `predictive_controller (history)` | +15.147 +/- 1.498 (10.11; 16/0), 94.4% | +14.918 +/- 1.707 (8.74; 12/0), 94.2% | +14.620 +/- 1.259 (11.61; 20/0), 94.3% |
| `ORACLE_lookahead` | +14.931 +/- 1.500, 93.1% | +14.692 +/- 1.698, 92.8% | +14.409 +/- 1.260, 92.9% |
| `DP_hindsight_optimum` | +16.041 +/- 1.486 | +15.835 +/- 1.694 | +15.508 +/- 1.257 |

The ordering is the same in all three columns: rule below seasonal rule below controller, the controller within 6% of the DP optimum, ahead of the
rule in every controllable scenario. Under the Seagate-only model there are no filter-forced scenarios, so all 20 are controllable (4 in the existing
population-A evaluation, 8 in the control). Latency violation fraction differences are within 0.001 in all three.

- **The environment changes.** Under the Seagate-only model the simulated per-interval failure probability is lower: `risk_aware_rule_based` mean
  `p_failure_mean` 0.00026 (existing: 0.00108; control: 0.00103), `p_failure_max` 0.00064 (0.0023; 0.0022); no episode violates the risk budget and the
  safety filter never clips a `fixed_20` rollout (clip fraction 0.000; 0.123 existing). Mean latency cost of `risk_aware_rule_based` is 7.9
  (existing 282.0, control 223.0) and of the DP optimum 0.32 (275.2; 216.3); `fixed_50` has zero latency cost. The gains remain of the same size because
  they are driven by the time cost of staying degraded and by the remaining latency cost, not by the risk constraint.
- **What survives and what does not.** The policy ordering and the controller's share of the optimum survive. The test of the risk input as a
  constraint does not: with the risk budget never binding, this evaluation cannot tell whether the controller's gain depends on the hazard input.
- **Fleet effect (control).** The pooled model with Toshiba fleets changes the controllable set (12 scenarios) and the gain by under 0.3; the
  differences in the last column are due to the hazard model.

Sanity checks: batch scoring path against `FittedHazardV1` per row on 200 Toshiba holdout rows, maximum absolute difference 0.0; scores are not constant
(2,159,160 distinct values on the holdout, range 0.0008 to 0.067); the same code gives ROC-AUC 0.814 on the Seagate test drives and agrees with the
pooled model there (PR-AUC 0.153 against 0.161).

## Decisions made during execution

- **Bug or finding.** The hard-stop condition (AUC at or below chance) was met on Toshiba; judged a finding on the three checks above and the run continued.
- **Options, not copies.** `fit_v1.fit`, `build_fleet_features.build` gained default-off arguments. Equivalence with the previous behaviour: a
  synthetic fit reproduces pre-edit numbers exactly (`hazard/test_fit_v1_options.py`); the committed train and test fleet tables are rebuilt
  byte for byte through the new options. `hazard/artifacts/v1/` was not regenerated.
- **Manifest for the new artifacts** is `hazard/manifest_v1_seagate_only.json`; `hazard/manifest.json` is unchanged (a scratch run that rewrote it
  was reverted before commit).
- **Threshold rows.** Thresholds at fixed FPR for the Toshiba cells are taken from the Toshiba drives of A2's validation split; for the Seagate cell from
  the validation bucket of the new split.
- **Control arm.** A pooled-hazard run with the Toshiba fleet table separates the effect of the fleet from that of the hazard model.
- **Not run:** the optional repeat on `altB`.

## Limits

- The simulator's true hazard is derived from the hazard model's own output, so the policy comparison measures behaviour in an environment built
  from the model under test; it cannot validate calibration, and an inverted ranking is invisible to it. The calibration result stands on the
  Backblaze rows alone.
- One held-out vendor and one split seed; 385 failed Toshiba drives. Vendor-to-vendor generalisation of other pairs (Toshiba to Seagate, or a
  Seagate model alone held out) is not tested.
- Toshiba lacks `smart_187_raw` and `smart_188_raw`, which the Seagate models have (A1); the Seagate-only model scores Toshiba with those columns
  missing and `model` as an unseen category. Which of the two causes the ranking inversion was not isolated.
- 20 scenarios on one 4-day holdout workload; the existing comparison and the control use scenario classes of different size (16, 12, 20).
- The evaluation compares total reward, not the registered risk-constrained objective; the risk budget is not active in the Seagate-only column.
