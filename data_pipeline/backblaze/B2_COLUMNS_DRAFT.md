# B2 per-drive timeline — draft column list

**DRAFT for review. Not final.** Window sizes and the SMART subset are
judgment calls (CLAUDE.md §4) and need confirmation before `timelines.py`
is written. Input is `data/processed/backblaze_filtered.parquet` (B1,
provisional three-model Seagate subset, one quarter).

## Keys and structural columns

| Column | Definition | Why |
|---|---|---|
| `serial_number`, `date` | Row key; sorted by `serial_number`, then `date` | One row per drive per observed day |
| `model`, `capacity_bytes`, `failure` | Carried from B1 | Label source and stratification |
| `first_seen` | Min `date` for the drive in the extract | Left-censoring: drive already running at quarter start if equal to quarter start |
| `last_seen` | Max `date` for the drive in the extract | Separates a failed drive (`failure=1` on `last_seen`) from one that left observation (right-censored) |
| `days_observed` | Count of distinct observed dates for the drive | Distinguishes a full 90-day history from a short one |
| `days_since_first_seen` | `date - first_seen` in days, per row | Trailing-only age within the observation window; not the drive's true age |
| `failed_in_window` | `True` for every row of a drive with any `failure=1` | Drive-level outcome flag, no horizon label (the horizon label is built downstream) |
| `gap_days_before` | Calendar days since the previous observed row for this drive (`NaN` on the first row) | Exposes missing days so windows are not silently computed over gaps |

## Missingness columns

One `<col>_missing` boolean per SMART input column used as a feature
(`True` where the raw value is NaN on that row). Values are never imputed
across drives. If forward-fill is used, it is within one drive from past
rows only, and the filled row keeps `<col>_missing = True`.

## Proposed SMART inputs

Selection follows attributes commonly reported as failure-predictive in
public SMART studies and Backblaze's own drive-stats write-ups. Final
choice depends on which columns are non-null for the kept models
(`smart_columns_all_null` in the B1 log).

| Attribute | Name | Proposed use |
|---|---|---|
| `smart_5_raw` | Reallocated sectors count | Level + trailing 7 d and 30 d difference |
| `smart_187_raw` | Reported uncorrectable errors | Level + trailing 7 d and 30 d difference |
| `smart_188_raw` | Command timeout | Level + trailing 7 d difference. Seagate packs several counters into this raw value, so the level is only usable after checking its range |
| `smart_197_raw` | Current pending sector count | Level + trailing 7 d and 30 d difference |
| `smart_198_raw` | Offline uncorrectable sector count | Level + trailing 7 d and 30 d difference |
| `smart_9_raw` | Power-on hours | Level only (drive age proxy) |
| `smart_12_raw` | Power cycle count | Trailing 30 d difference |
| `smart_194_raw` | Temperature | Trailing 7 d mean and max |
| `smart_193_raw` | Load/unload cycle count | Trailing 30 d difference |

Not proposed as raw features: `smart_1_raw`, `smart_7_raw`, `smart_240_raw`
to `smart_242_raw` on Seagate drives. Their raw values are vendor-packed or
workload-dependent and are not comparable across models without decoding.

## Window rules

- **Trailing only.** Every windowed feature uses rows with `date <= row date`
  and nothing later. Verified by `tests/test_no_leakage.py`.
- **Calendar windows, not row windows.** A 7 d window means dates in
  `(date - 7 d, date]`, so a missing day shortens the number of observations
  instead of reaching further back.
- **Minimum observations.** A windowed value is `NaN` (with its missing flag
  set) when the window holds fewer than 2 observed rows for a difference or
  fewer than 3 for a mean.
- **Proposed window sizes:** 7 d and 30 d. 30 d matches the hazard horizon
  `HORIZON_DAYS = 30`. The first 30 days of a one-quarter extract therefore
  carry partial 30 d windows; the missing flags mark them.

## Open points for review

1. Window sizes (7 d / 30 d) and the minimum-observation thresholds.
2. SMART subset above versus the columns actually populated for the three models.
3. Whether one quarter is enough: rows in the last 30 days of the quarter have
   an incomplete 30-day outcome window, so the horizon label for those rows is
   censored. Handling is a label-construction decision outside this stage.
4. `smart_188_raw` handling (packed counter).
