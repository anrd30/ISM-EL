# B2 per-drive timeline — column list

**Draft for review.** `backblaze/timelines.py` implements this list. Window
sizes and the SMART subset are judgment calls (CLAUDE.md §4) and need
confirmation. Input is `data/processed/backblaze_filtered.parquet` (B1,
provisional three-model Seagate subset, Q1 2022 only).

## Two tables, split at the leakage boundary

A value is a **feature** only if it is a function of that drive's rows with
`date <=` the row date. Whole-drive facts such as `last_seen` or
`failed_in_window` describe the future of every earlier row, so they live in
a separate table and are never feature inputs.

### 1. `data/processed/backblaze_timelines.parquet` — one row per drive-day

Backward-looking columns only. `tests/test_no_leakage.py` cuts the input at a
date, recomputes, and requires all earlier rows to be identical.

| Column | Definition |
|---|---|
| `serial_number`, `date` | Row key, sorted by `serial_number` then `date` |
| `model`, `capacity_bytes` | Carried from B1 |
| `failure` | The day's own B1 flag (`1` on the last operational day). This is the label source, not a feature |
| `gap_days_before` | Calendar days since the drive's previous observed row; null on its first row |
| `days_since_first_seen` | `date - first_seen`. **Position in the observation window, not drive age** |
| `first_seen` | Date of the drive's first row in the extract (equals the minimum over rows up to the current one) |
| `<smart>` | Current raw SMART value (all nine columns except `smart_188_raw`) |
| `<smart>_diff7d`, `<smart>_diff30d` | Trailing calendar-window difference (rule below). `smart_188_raw` has `_diff7d` only |
| `<feature>_missing` | Boolean, one per feature column above; `True` where the feature is NaN |

Columns: 8 identifiers/structure + 25 features + 25 flags = 58. No value is
imputed; a flag marks a genuine gap, an absent SMART reading, or too few
observations in the window.

### 2. `data/processed/backblaze_drive_index.parquet` — one row per drive

| Column | Definition |
|---|---|
| `serial_number`, `model` | Drive key |
| `first_seen`, `last_seen` | Min and max `date` in the extract |
| `days_observed` | Count of observed dates |
| `failed_in_window` | `True` if any row has `failure = 1` |
| `failure_date` | Date of that row; null if none |

Joined to the timelines only at label construction, downstream of the
leakage boundary. `last_seen` with `failed_in_window = False` marks a drive
that left observation without a recorded failure (censored, fate unknown).

## Age versus observation position

`smart_9_raw` (power-on hours) is the age feature. It is kept as a raw level
in the timelines table. `days_since_first_seen` is only the distance from the
start of this extract: a drive first seen on 2022-03-01 has value 0 whether it
is new or six years old. It must not be read as age.

## SMART inputs

`smart_5_raw`, `smart_9_raw`, `smart_12_raw`, `smart_187_raw`,
`smart_188_raw`, `smart_193_raw`, `smart_194_raw`, `smart_197_raw`,
`smart_198_raw`. All nine are populated for the three kept models in Q1 2022.

- **`smart_188_raw` (command timeout): no level column.** Seagate packs
  several counters into this raw value, so its level is not comparable across
  drives. The 7-day difference is kept because an increase still means a
  counter incremented.
- **`smart_9_raw`:** level plus both differences. The difference is close to
  24 per day of continuous operation and is mostly a coverage check.

## Window rule

For a window of N days, `<smart>_diffNd` is the value on the current row minus
the value on the earliest observed row with `date >= today - N`.

- Calendar-based: a missing day shortens the window's observation count and
  does not reach further back.
- NaN when fewer than 2 observed rows lie in `[today - N, today]`, or when
  either endpoint is NaN.
- Raw counters can reset or decrease (drive replacement, firmware); negative
  differences are kept as they are.
- Windows are 7 and 30 days. 30 days matches `HORIZON_DAYS`. Rows in the first
  30 days of the extract have partial 30-day windows; the flags mark them.
- Differences are not divided by the span between the two endpoints. With a
  gap inside the window the span can be shorter than N.

## Changes from the earlier draft

- Whole-drive columns (`last_seen`, `days_observed`, `failed_in_window`) moved
  from the per-day table to the drive index.
- One uniform scheme replaces the per-attribute subset: level plus 7 d and
  30 d differences for every column, except `smart_188_raw` (7 d only, no
  level).
- Temperature mean/max windows were dropped; temperature uses the same
  level and difference columns as the rest.
- The window is `[today - N, today]` (closed on the left) so that a full
  7-day window spans 7 days.

## Label events in Q1 2022 (measured from the drive index)

- 276 drives have `failed_in_window = True`, with failure dates from
  2022-01-01 to 2022-03-28.
- With a 30-day outcome horizon, the last date with a complete horizon inside
  the quarter is 2022-03-01. 168 of the 276 failures fall on or before that
  date and 108 fall after it.
- The figure of about 184 usable labeled events was not reproduced from this
  data. The split depends on the labeling rule: a failure after 2022-03-01 is
  still a known positive for the rows up to 30 days before it, and what is
  censored in the final month is the negative label for non-failing rows.
  The label construction decides which count applies.
- A second quarter (the estimate of about 460 events for the full set) has not
  been downloaded or measured. It is not needed for this version.

## Open points for review

1. Window sizes (7 d / 30 d) and the 2-observation minimum.
2. Whether one quarter is enough events for the risk model.
3. `smart_194_raw` and `smart_12_raw` differences are less standard than the
   error counters; keep or drop after a first fit.
