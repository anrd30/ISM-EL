# A4 decision record: v1 against v0 on the held-out split

Status: proposed. Not yet reviewed by the other lanes.

The Evidence and Verdict blocks are written by `python -m hazard.evaluate_v1 --write-record`; they hold the printed numbers of that run and are replaced on every run.

## Decision

- Script: `hazard/evaluate_v1.py` (run from the repository root). Nothing is fitted or recalibrated; `hazard/v0.py`, `hazard/v1.py`, `hazard/fit_v1.py` and the v1 artifacts are used as they stand.
- Rows: label != -1 rows of the drives of the held-out split (A2). The held-out rows are read only by `load_test_split`. Validation rows are read only to choose the two decision thresholds.
- Predictors: `hazard_v0` (`HeuristicHazardV0`, default AFR), `hazard_v1` (`FittedHazardV1`), and the diagnostic `constant_base_rate (diagnostic)`, which outputs the base rate of the cell's held-out rows and is not a comparator. The name follows the diagnostic naming of `baselines/policies.py` (snake_case with a parenthesised qualifier).
- Bulk path: `predict_v1_batch` loads the boosters and calibrator thresholds once, builds one frame of many rows in training column order, predicts every booster on the whole batch, calibrates each output with its isotonic map and averages (same arithmetic as `FittedHazardV1.__call__`). `cross_check` calls the per-row `FittedHazardV1` interface on a random sample of 200 held-out rows and compares both outputs with the batch path; the run aborts if the largest absolute difference exceeds 1e-4. `hazard/test_evaluate_v1.py` repeats the comparison on the CPU boosters with a tolerance of 1e-7. v0 is called through its own interface, once per distinct input combination.
- PR-AUC: average precision, scores that are equal form one group.
- Recall at a false-positive-rate (FPR) target of 1% and of 0.1%: the threshold is chosen on the validation rows of the same cell and frozen. The brief's rule "largest threshold that does not exceed the target FPR" is implemented as the one-sided rule of the load-profile selection (the largest N whose p95 does not exceed the target): the most permissive operating point whose validation FPR does not exceed the target. With `k = floor(target x N_neg)` validation negatives allowed, the threshold is the (k + 1)-th largest validation negative score and a row is flagged when its score is strictly above it. A literal "largest score" would flag nothing. If no validation row is above that threshold, the target is reported as unattainable and no recall is given.
- Calibration: Brier score and an equal-frequency reliability table. Bin edges are quantiles of the scores; equal scores always share a bin, so tied scores can give fewer bins than requested. The requested count is 20, reduced to keep at least 10 failure rows per bin on average (`positives // 10`). With about 7,750 held-out positive rows pooled, 20 bins give about 390 positive rows per bin on average, and every cell has at least 1,400 positive rows, so 20 bins is used everywhere.
- Cells: pooled, vendor (Seagate: the three ST models; Toshiba) and each of the four drive models. Thresholds, metrics and intervals are computed within each cell. A cell with fewer than 50 failed held-out drives is marked `FEW_FAILED_DRIVES`: at 50 drives the 95% half-width of a proportion near 0.5 is about 14 percentage points.
- Intervals: 95% percentile intervals from 200 drive-bootstrap resamples per cell (whole drives drawn with replacement within the cell, every row of a drawn drive carried with it; seed 20220401). The v1 minus v0 differences of PR-AUC and Brier use the same resamples for both predictors.
- Verdict rule: v1 is better when the interval of the PR-AUC difference is above 0 and the interval of the Brier difference is below 0; v0 is better in the reverse case; otherwise the verdict names which metric each leads on, or no difference.

## Evidence

<!-- a4:evidence -->
- Device: cuda; per-row cross-check on 200 held-out rows, largest absolute difference 4.337e-19.
- Rows scored: held-out 4,564,332, validation 4,567,066. Drive bootstrap resamples per cell: 200, seed 20220401.

| Cell | Failed drives | Predictor | PR-AUC [95%] | Brier [95%] | Recall @1% FPR [95%] | Recall @0.1% FPR [95%] |
|---|---|---|---|---|---|---|
| pooled | 265 | constant_base_rate (diagnostic) | 0.0017 [0.0015, 0.0019] | 1.696e-03 [1.503e-03, 1.886e-03] | unattainable | unattainable |
| pooled | 265 | hazard_v0 | 0.0569 [0.0442, 0.0762] | 1.683e-03 [1.491e-03, 1.871e-03] | 0.368 [0.313, 0.422] | 0.124 [0.091, 0.164] |
| pooled | 265 | hazard_v1 | 0.1361 [0.0985, 0.1801] | 1.563e-03 [1.391e-03, 1.728e-03] | 0.409 [0.360, 0.457] | 0.227 [0.185, 0.274] |
| vendor=Seagate | 207 | constant_base_rate (diagnostic) | 0.0023 [0.0020, 0.0026] | 2.279e-03 [1.978e-03, 2.618e-03] | unattainable | unattainable |
| vendor=Seagate | 207 | hazard_v0 | 0.0687 [0.0470, 0.1009] | 2.257e-03 [1.954e-03, 2.596e-03] | 0.350 [0.282, 0.403] | 0.153 [0.107, 0.200] |
| vendor=Seagate | 207 | hazard_v1 | 0.1606 [0.1211, 0.2028] | 2.075e-03 [1.804e-03, 2.401e-03] | 0.446 [0.390, 0.504] | 0.232 [0.182, 0.278] |
| vendor=Toshiba | 58 | constant_base_rate (diagnostic) | 0.0009 [0.0007, 0.0011] | 8.904e-04 [6.656e-04, 1.138e-03] | unattainable | unattainable |
| vendor=Toshiba | 58 | hazard_v0 | 0.0897 [0.0318, 0.1727] | 8.905e-04 [6.663e-04, 1.137e-03] | 0.421 [0.313, 0.550] | 0.177 [0.090, 0.263] |
| vendor=Toshiba | 58 | hazard_v1 | 0.0682 [0.0243, 0.1347] | 8.566e-04 [6.337e-04, 1.103e-03] | 0.251 [0.165, 0.362] | 0.154 [0.085, 0.244] |
| model=ST12000NM0008 | 60 | constant_base_rate (diagnostic) | 0.0017 [0.0013, 0.0023] | 1.745e-03 [1.323e-03, 2.251e-03] | unattainable | unattainable |
| model=ST12000NM0008 | 60 | hazard_v0 | 0.1378 [0.0923, 0.2121] | 1.733e-03 [1.314e-03, 2.235e-03] | 0.494 [0.398, 0.609] | 0.115 [0.054, 0.200] |
| model=ST12000NM0008 | 60 | hazard_v1 | 0.2102 [0.1500, 0.2845] | 1.543e-03 [1.155e-03, 1.968e-03] | 0.448 [0.353, 0.543] | 0.245 [0.169, 0.330] |
| model=ST4000DM000 | 95 | constant_base_rate (diagnostic) | 0.0031 [0.0025, 0.0036] | 3.064e-03 [2.487e-03, 3.617e-03] | unattainable | unattainable |
| model=ST4000DM000 | 95 | hazard_v0 | 0.1211 [0.0893, 0.1730] | 3.026e-03 [2.452e-03, 3.575e-03] | 0.501 [0.416, 0.601] | 0.205 [0.133, 0.281] |
| model=ST4000DM000 | 95 | hazard_v1 | 0.2031 [0.1399, 0.2759] | 2.730e-03 [2.220e-03, 3.212e-03] | 0.485 [0.389, 0.574] | 0.258 [0.187, 0.331] |
| model=ST8000NM0055 | 52 | constant_base_rate (diagnostic) | 0.0020 [0.0014, 0.0026] | 2.016e-03 [1.383e-03, 2.556e-03] | unattainable | unattainable |
| model=ST8000NM0055 | 52 | hazard_v0 | 0.0229 [0.0133, 0.0508] | 2.004e-03 [1.377e-03, 2.538e-03] | 0.140 [0.065, 0.251] | 0.044 [0.000, 0.101] |
| model=ST8000NM0055 | 52 | hazard_v1 | 0.0495 [0.0209, 0.1198] | 1.977e-03 [1.367e-03, 2.498e-03] | 0.333 [0.232, 0.475] | 0.149 [0.071, 0.251] |
| model=TOSHIBA MG07ACA14TA | 58 | constant_base_rate (diagnostic) | 0.0009 [0.0007, 0.0011] | 8.904e-04 [6.803e-04, 1.130e-03] | unattainable | unattainable |
| model=TOSHIBA MG07ACA14TA | 58 | hazard_v0 | 0.0897 [0.0294, 0.2020] | 8.905e-04 [6.807e-04, 1.129e-03] | 0.421 [0.293, 0.551] | 0.177 [0.090, 0.274] |
| model=TOSHIBA MG07ACA14TA | 58 | hazard_v1 | 0.0682 [0.0267, 0.1290] | 8.566e-04 [6.531e-04, 1.083e-03] | 0.251 [0.157, 0.343] | 0.154 [0.089, 0.240] |

Frozen thresholds (flag when score is strictly above), chosen on validation rows of each cell:

| Cell | Predictor | Target FPR | Threshold | Val FPR | Test FPR |
|---|---|---|---|---|---|
| pooled | constant_base_rate (diagnostic) | 1.0% | unattainable | - | - |
| pooled | constant_base_rate (diagnostic) | 0.1% | unattainable | - | - |
| pooled | hazard_v0 | 1.0% | 6.4678e-03 | 0.9765% | 0.9587% |
| pooled | hazard_v0 | 0.1% | 1.5312e-02 | 0.1000% | 0.1217% |
| pooled | hazard_v1 | 1.0% | 9.9615e-03 | 1.0000% | 0.9053% |
| pooled | hazard_v1 | 0.1% | 1.2089e-01 | 0.1000% | 0.1188% |
| vendor=Seagate | constant_base_rate (diagnostic) | 1.0% | unattainable | - | - |
| vendor=Seagate | constant_base_rate (diagnostic) | 0.1% | unattainable | - | - |
| vendor=Seagate | hazard_v0 | 1.0% | 8.7242e-03 | 0.9900% | 0.9887% |
| vendor=Seagate | hazard_v0 | 0.1% | 1.6025e-02 | 0.1000% | 0.1696% |
| vendor=Seagate | hazard_v1 | 1.0% | 1.1283e-02 | 1.0000% | 0.9657% |
| vendor=Seagate | hazard_v1 | 0.1% | 1.3402e-01 | 0.0999% | 0.1457% |
| vendor=Toshiba | constant_base_rate (diagnostic) | 1.0% | unattainable | - | - |
| vendor=Toshiba | constant_base_rate (diagnostic) | 0.1% | unattainable | - | - |
| vendor=Toshiba | hazard_v0 | 1.0% | 1.9428e-03 | 1.0000% | 0.9352% |
| vendor=Toshiba | hazard_v0 | 0.1% | 2.5332e-03 | 0.1000% | 0.1312% |
| vendor=Toshiba | hazard_v1 | 1.0% | 8.6299e-03 | 1.0000% | 0.8175% |
| vendor=Toshiba | hazard_v1 | 0.1% | 8.7635e-02 | 0.0999% | 0.0642% |
| model=ST12000NM0008 | constant_base_rate (diagnostic) | 1.0% | unattainable | - | - |
| model=ST12000NM0008 | constant_base_rate (diagnostic) | 0.1% | unattainable | - | - |
| model=ST12000NM0008 | hazard_v0 | 1.0% | 4.0716e-03 | 1.0000% | 0.7560% |
| model=ST12000NM0008 | hazard_v0 | 0.1% | 1.3643e-02 | 0.0999% | 0.0774% |
| model=ST12000NM0008 | hazard_v1 | 1.0% | 7.5274e-03 | 1.0000% | 0.8297% |
| model=ST12000NM0008 | hazard_v1 | 0.1% | 1.4567e-01 | 0.0999% | 0.0763% |
| model=ST4000DM000 | constant_base_rate (diagnostic) | 1.0% | unattainable | - | - |
| model=ST4000DM000 | constant_base_rate (diagnostic) | 0.1% | unattainable | - | - |
| model=ST4000DM000 | hazard_v0 | 1.0% | 9.0749e-03 | 0.8589% | 1.0866% |
| model=ST4000DM000 | hazard_v0 | 0.1% | 1.4971e-02 | 0.0968% | 0.2312% |
| model=ST4000DM000 | hazard_v1 | 1.0% | 1.7540e-02 | 0.9999% | 1.0714% |
| model=ST4000DM000 | hazard_v1 | 0.1% | 1.4232e-01 | 0.1000% | 0.1845% |
| model=ST8000NM0055 | constant_base_rate (diagnostic) | 1.0% | unattainable | - | - |
| model=ST8000NM0055 | constant_base_rate (diagnostic) | 0.1% | unattainable | - | - |
| model=ST8000NM0055 | hazard_v0 | 1.0% | 1.2182e-02 | 0.9999% | 0.7544% |
| model=ST8000NM0055 | hazard_v0 | 0.1% | 1.7122e-02 | 0.0999% | 0.1224% |
| model=ST8000NM0055 | hazard_v1 | 1.0% | 1.0563e-02 | 0.9999% | 0.9515% |
| model=ST8000NM0055 | hazard_v1 | 0.1% | 9.4068e-02 | 0.0999% | 0.2250% |
| model=TOSHIBA MG07ACA14TA | constant_base_rate (diagnostic) | 1.0% | unattainable | - | - |
| model=TOSHIBA MG07ACA14TA | constant_base_rate (diagnostic) | 0.1% | unattainable | - | - |
| model=TOSHIBA MG07ACA14TA | hazard_v0 | 1.0% | 1.9428e-03 | 1.0000% | 0.9352% |
| model=TOSHIBA MG07ACA14TA | hazard_v0 | 0.1% | 2.5332e-03 | 0.1000% | 0.1312% |
| model=TOSHIBA MG07ACA14TA | hazard_v1 | 1.0% | 8.6299e-03 | 1.0000% | 0.8175% |
| model=TOSHIBA MG07ACA14TA | hazard_v1 | 0.1% | 8.7635e-02 | 0.0999% | 0.0642% |

Pooled reliability, hazard_v0 (17 equal-frequency bins):

| Bin | Rows | Positives | Mean prediction | Observed rate |
|---|---|---|---|---|
| 0 | 228,254 | 78 | 0.001758 | 0.000342 |
| 1 | 228,297 | 62 | 0.001772 | 0.000272 |
| 2 | 228,225 | 180 | 0.001781 | 0.000789 |
| 3 | 228,228 | 118 | 0.001789 | 0.000517 |
| 4 | 228,158 | 54 | 0.001796 | 0.000237 |
| 5 | 228,218 | 79 | 0.001802 | 0.000346 |
| 6 | 228,187 | 102 | 0.001809 | 0.000447 |
| 7 | 228,198 | 34 | 0.001815 | 0.000149 |
| 8 | 228,333 | 194 | 0.001821 | 0.000850 |
| 9 | 228,088 | 114 | 0.001827 | 0.000500 |
| 10 | 228,320 | 257 | 0.001834 | 0.001126 |
| 11 | 228,119 | 221 | 0.001843 | 0.000969 |
| 12 | 228,191 | 63 | 0.001881 | 0.000276 |
| 13 | 228,296 | 191 | 0.001934 | 0.000837 |
| 14 | 1,140,344 | 774 | 0.001944 | 0.000679 |
| 15 | 659 | 0 | 0.001948 | 0.000000 |
| 16 | 228,217 | 5,233 | 0.004650 | 0.022930 |

Pooled reliability, hazard_v1 (16 equal-frequency bins):

| Bin | Rows | Positives | Mean prediction | Observed rate |
|---|---|---|---|---|
| 0 | 1,245,404 | 649 | 0.000716 | 0.000521 |
| 1 | 127,781 | 71 | 0.000717 | 0.000556 |
| 2 | 224,338 | 187 | 0.000720 | 0.000834 |
| 3 | 228,210 | 123 | 0.000727 | 0.000539 |
| 4 | 228,216 | 130 | 0.000740 | 0.000570 |
| 5 | 228,217 | 99 | 0.000758 | 0.000434 |
| 6 | 228,217 | 139 | 0.000786 | 0.000609 |
| 7 | 228,216 | 228 | 0.000828 | 0.000999 |
| 8 | 228,217 | 237 | 0.000893 | 0.001038 |
| 9 | 228,216 | 206 | 0.000987 | 0.000903 |
| 10 | 228,217 | 278 | 0.001117 | 0.001218 |
| 11 | 228,216 | 268 | 0.001286 | 0.001174 |
| 12 | 228,217 | 287 | 0.001497 | 0.001258 |
| 13 | 228,216 | 331 | 0.001800 | 0.001450 |
| 14 | 228,217 | 470 | 0.002409 | 0.002059 |
| 15 | 228,217 | 4,051 | 0.015493 | 0.017751 |
<!-- /a4:evidence -->

## Verdict

Verdict (from the paired drive-bootstrap differences, v1 minus v0):

<!-- a4:verdict -->
- Pooled: v1 better than v0 on PR-AUC and Brier (PR-AUC v1-v0 +0.07918 [+0.04876, +0.11860]; Brier v1-v0 -1.201e-04 [-1.578e-04, -8.341e-05]).
- vendor=Seagate: v1 better than v0 on PR-AUC and Brier (PR-AUC v1-v0 +0.09193 [+0.04216, +0.12945]; Brier v1-v0 -1.825e-04 [-2.325e-04, -1.233e-04]).
- vendor=Toshiba: v1 ahead on Brier, no difference at 95% on the other (PR-AUC v1-v0 -0.02149 [-0.09196, +0.03546]; Brier v1-v0 -3.390e-05 [-7.553e-05, -8.563e-07]).
- model=ST12000NM0008: v1 ahead on Brier, no difference at 95% on the other (PR-AUC v1-v0 +0.07232 [-0.00998, +0.13887]; Brier v1-v0 -1.892e-04 [-2.929e-04, -1.111e-04]).
- model=ST4000DM000: v1 better than v0 on PR-AUC and Brier (PR-AUC v1-v0 +0.08203 [+0.02892, +0.13762]; Brier v1-v0 -2.961e-04 [-4.055e-04, -1.846e-04]).
- model=ST8000NM0055: no difference at 95% on PR-AUC or Brier (PR-AUC v1-v0 +0.02666 [-0.00149, +0.09075]; Brier v1-v0 -2.748e-05 [-1.181e-04, +4.532e-05]).
- model=TOSHIBA MG07ACA14TA: no difference at 95% on PR-AUC or Brier (PR-AUC v1-v0 -0.02149 [-0.12248, +0.04076]; Brier v1-v0 -3.390e-05 [-6.815e-05, +5.067e-06]).
<!-- /a4:verdict -->

## Limits

- v0 reads `smart_187_raw`, `smart_197_raw`, `smart_5_raw`, `smart_9_raw` and `smart_188_raw`. The timelines hold no `smart_188_raw` level column, so that input is absent for every row, and `smart_187_raw` is absent for Toshiba rows. Absent inputs add no severity.
- v0's AFR (2.11%) comes from Q1 2022 data of three Seagate models that includes held-out drives; the pooled AFR of the 2022 extract is 1.95%. v0 is scored as it stands.
- Every booster of v1 reached the 2000-round cap (A3 limit c), so v1 is a first-pass fit, not a tuned one.
- Thresholds are chosen on the validation rows of each cell, so the realised held-out FPR differs from the target; the table lists both. The intervals reflect resampling of held-out drives only; the variability of the validation-chosen thresholds is not in them.
- Failed held-out drives per drive model are 52 to 95 and positive rows come from these drives (up to 31 rows per failed drive). Metrics are computed per row, not per drive. Cells marked `FEW_FAILED_DRIVES` are noisy.
- The Brier score is dominated by the negative rows (a positive-row share below 0.2%); a small Brier difference is small relative to the base-rate variance, and the constant predictor is the reference for it.
- The split is one draw of one seed and is not time-forward (A2 limits apply).
- The held-out rows are read by every run of the script; no choice made here (thresholds, bins, cells) was changed after looking at held-out results.
- vendor=Toshiba and model=TOSHIBA MG07ACA14TA cover identical rows (Toshiba has one model in this dataset) and have identical Brier point estimates (-3.390e-05), but their bootstrap confidence intervals differ because resamples are drawn independently per cell rather than reused for duplicate cells -- one crosses zero, the other doesn't. Treat the Toshiba Brier result as borderline either way, not as two different findings.
