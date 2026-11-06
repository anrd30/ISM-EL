# A2 decision record: horizon labels and drive-level split

Status: accepted. Reviewed by T-lane (see agent/T_DECISION.md) and B-lane.

## Decision

- Labels: the 30-day rule of B2 (`timelines.row_labels`), stored per timelines row in `data/processed/backblaze_labels.parquet` (`serial_number`, `date`, `model`, `label`; 1 positive, 0 negative, -1 censored). Censored rows are excluded from training and scoring.
- Split unit: the drive (`serial_number`). All rows of a drive are in one split. File: `data/processed/backblaze_drive_split.parquet` (`serial_number`, `model`, `ever_failed`, `split`), one row per drive of the drive index, including the 122 fully censored drives.
- Fractions: train 70%, val 15%, test 15%.
- Strata: (`model`, `ever_failed`), `ever_failed` = `failed_in_window`.
- Procedure: drives sorted by `serial_number`; per stratum a seeded permutation (`numpy.random.default_rng([20220101, stratum number])`, strata in sorted key order); sizes by largest-remainder rounding, so each split size is within one drive of its target share.
- Seed: 20220101.
- Why drive-level: rows of one drive are strongly correlated (the SMART values and the label windows of consecutive days overlap), so a row-level split would put near-copies of a drive in train and test. Stratifying by model and outcome keeps the model mix and the rare failed drives in every split.

## Counts (computed by `backblaze/labels.py`)

| Split | Model | Drives | Failed drives | Positive rows | Negative rows | Censored rows |
|---|---|---|---|---|---|---|
| train | ST4000DM000 | 13,258 | 443 | 13,232 | 4,289,970 | 370,514 |
| val | ST4000DM000 | 2,841 | 95 | 2,806 | 923,164 | 79,479 |
| test | ST4000DM000 | 2,841 | 95 | 2,835 | 919,546 | 79,523 |
| train | ST8000NM0055 | 10,340 | 244 | 7,148 | 3,351,946 | 292,354 |
| val | ST8000NM0055 | 2,215 | 52 | 1,477 | 718,482 | 62,608 |
| test | ST8000NM0055 | 2,215 | 52 | 1,452 | 717,292 | 62,933 |
| train | ST12000NM0008 | 14,165 | 283 | 8,522 | 4,679,379 | 402,427 |
| val | ST12000NM0008 | 3,036 | 61 | 1,725 | 1,001,902 | 86,245 |
| test | ST12000NM0008 | 3,035 | 60 | 1,758 | 1,003,854 | 86,262 |
| train | TOSHIBA MG07ACA14TA | 27,005 | 269 | 7,969 | 8,933,167 | 775,008 |
| val | TOSHIBA MG07ACA14TA | 5,787 | 58 | 1,754 | 1,915,756 | 166,103 |
| test | TOSHIBA MG07ACA14TA | 5,787 | 58 | 1,709 | 1,915,886 | 166,097 |
| train | all | 64,768 | 1,239 | 36,871 | 21,254,462 | 1,840,303 |
| val | all | 13,879 | 266 | 7,762 | 4,559,304 | 394,435 |
| test | all | 13,878 | 265 | 7,754 | 4,556,578 | 394,815 |

Totals: 92,525 drives, 1,770 failed drives, 52,387 positive, 30,370,344 negative and 2,629,553 censored rows (33,052,284 rows).

## Limits

- Small failed-drive counts per split and model: 52 to 95 failed drives in each val or test split for the Seagate models and 58 for the Toshiba model; positive rows come from these drives and are correlated within a drive. Metrics per model and per split are noisy.
- The split is not time-forward. Drives span the whole window, so the three splits overlap in dates (`a2_split_dates` lines in `PREPROCESSING.md` give the date range and the positive rows by month of each split). A late-window evaluation needs a separate definition.
- One split, one seed. No cross-validation folds are stored.
- Fully censored drives (122) carry no labelled rows and are in the split table only to keep one row per drive.
- Positive rows for a drive are the rows in the 30 days up to its failure date; a failed drive has up to 31 positive rows.

## Reproduction

`python -I backblaze/labels.py` from `data_pipeline/`. The run aborts if the label counts differ from the values recorded in `backblaze/labels.py`.
