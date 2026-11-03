# A3 decision record: fitted hazard model (v1)

Status: proposed. Not yet reviewed by the other lanes.

## Decision

- Model family: XGBoost gradient-boosted trees (`binary:logistic`, `hist`), device `cuda` when available, otherwise `cpu`. Learning rate 0.05, depth 6, 256 bins, at most 2000 rounds, early stopping after 50 rounds without a log-loss improvement on the val split.
- Target: the 30-day label of A2 (`backblaze_labels.parquet`), so `p_horizon` is over 720 h, equal to `contracts.HORIZON_HOURS`; any other `horizon_hours` raises `ContractViolation`.
- Rows: label != -1; fit on drives of the train split (21,291,333 rows, 36,871 positive); early stopping and calibration on the val split (4,567,066 rows, 7,762 positive). The test split is not read (`fit_v1.ALLOWED_SPLITS`).
- Features (57): the 28 value columns of B2, their 28 `_missing` flags and the drive model as a native categorical feature. NaN is passed through; no imputation.
- Imbalance: `scale_pos_weight` = negative rows / positive rows of the train split; no resampling by class.
- Uncertainty: 25 boosters, each fitted on a bootstrap resample of the train rows (same row count, drawn with replacement; carried as draw-count sample weights). `p_horizon` is the mean and `uncertainty` the population standard deviation of the 25 calibrated outputs, clipped to 0.5.
- Calibration: one isotonic map per booster, fitted on that booster's raw val predictions (`out_of_bounds="clip"`), applied before averaging. Stored as threshold arrays and applied with linear interpolation.
- Files: `hazard/artifacts/v1/` (`booster_00..24.ubj`, `calibrators.npz`, `meta.json` with seeds, feature order, parameters and the metrics below). Wrapper: `hazard/v1.py` (`FittedHazardV1`). Fit: `python -m hazard.fit_v1`.
- At inference a feature that is absent or unusable is NaN with its flag set to 1; the model name is read from the key `model`, and an absent or unknown name is a missing category.

## Evidence (last run of `hazard/fit_v1.py`)

- Device: cuda; xgboost 3.2.0; wall-clock 5668 s.
- scale_pos_weight: 576.45.
- Seeds (25): 20220301, 20220302, 20220303, 20220304, 20220305, 20220306, 20220307, 20220308, 20220309, 20220310, 20220311, 20220312, 20220313, 20220314, 20220315, 20220316, 20220317, 20220318, 20220319, 20220320, 20220321, 20220322, 20220323, 20220324, 20220325.
- Boosting rounds kept per booster: 2000 (the cap of 2000 was reached by every booster).
- Val log-loss after calibration, per booster: min 0.010156, max 0.010435; val Brier: min 0.001582, max 0.001607. Per-booster values are in `meta.json`.
- Ensemble (mean of the 25 calibrated outputs) on the val split: log-loss 0.010024, Brier 0.001585.
- Val base rate for reference: 0.001700 positive share.
- Raw (uncalibrated) val log-loss per booster is 0.104 to 0.112, because `scale_pos_weight` inflates the raw probabilities; the isotonic maps remove this.
- Artifact size: 228 MB for 27 files.

## Limits

- (a) `sim/scenario.py` generates only the 5 synthetic features of `hazard.v0.FEATURE_NAMES` for the simulator's fleet, so the 57 inputs of this model cannot be fed by the simulator yet. Mapping simulator state to these features is part of A5 and is not done here.
- (b) GPU and CPU training can differ in the last decimal places of the fitted trees.
- (c) First pass with default-style hyperparameters; no grid or random search was run. Every booster reached the 2000-round cap, so early stopping did not act and the log-loss on val was still decreasing at the cap.
- (d) The val numbers are sanity checks. The val split was used for early stopping and for fitting the isotonic maps, so the calibrated val log-loss and Brier values are in-sample for the calibration step. The A4 comparison against `hazard/v0.py` on PR-AUC, recall and calibration uses the test split and is not done by this task.
- The bootstrap is carried as sample weights on one quantised matrix, so the bin edges are computed once from the full train rows and not per resample.
- The early-stopping metric is unweighted val log-loss of the raw outputs, which are inflated by `scale_pos_weight`.
- The artifacts total about 218 MB, far above the few MB expected; committing them adds that size to the repository history.
- Calibration has a hard floor, not graceful degradation: isotonic calibrator 0's lowest step covers every raw booster score under about 0.0215 and maps all of them to the same output, 0.000716. An all-zero input and inputs with one feature set to 500, 5000 or 50000 (the other 55 features absent) all calibrate to that identical p_horizon, because all their raw scores stay under the floor — the model has effectively no resolution across most of the low-risk input space.
- Within the resolved region, single-feature-present inputs are not monotone: smart_5_raw=100 with everything else absent gives p=0.0082; smart_5_raw=10000 with everything else absent gives p=0.0021, lower for the nominally worse reading. One feature present and the other 56 absent is a missingness pattern essentially unseen in training, where a drive's SMART fields are missing together or present together, not one at a time.
- The all-missing input is a separate, more extreme case: its raw per-booster scores sit at 0.80-0.92, well above the calibration floor, because it repeatedly takes each tree's default missing-value direction into leaves training data rarely populates. Its calibrated output is 0.044.
- The split is not time-forward and the training rows of one drive are correlated (A2 limits apply).
