# A1 decision record: Backblaze data window, drive models, SMART columns

Status: accepted. Reviewed by T-lane (see agent/T_DECISION.md) and B-lane.

## Decision

- Window: calendar year 2022 (Q1, Q2, Q3, Q4). 364 daily files are used; `backblaze_2022Q4/2022-12-10.csv` is excluded (see Limits).
- Drive models (4): ST4000DM000, ST8000NM0055, ST12000NM0008, TOSHIBA MG07ACA14TA.
- Excluded: TOSHIBA MG08ACA16TE (93 failures, below the other four models; see below).
- Vendor/model identity is a model input. Calibration is reported per model and per vendor.
- SMART raw columns:
  - Shared by all four models: smart_5, 9, 193, 194, 197, 198, 199.
  - Seagate only (NaN for Toshiba): smart_187, 188.
  - smart_12 (power cycles) is in the current timelines.py list and was not coverage-checked for Toshiba; kept only if it is non-null for all four models.
  - Dropped: smart_189 (0% populated for ST12000NM0008 and both Toshiba models), smart_190 (duplicates 194; absent for Toshiba), smart_241 and smart_242 (absent for Toshiba).

## Evidence (computed from data/raw/backblaze_2022Q1..Q4, 364 files)

| Model | Failures | Drive-days | AFR |
|---|---|---|---|
| ST4000DM000 | 633 | 6,681,069 | 3.46% |
| ST8000NM0055 | 348 | 5,215,692 | 2.44% |
| ST12000NM0008 | 404 | 7,272,074 | 2.03% |
| TOSHIBA MG07ACA14TA | 385 | 13,883,449 | 1.01% |
| Pooled (4 models) | 1,770 | 33,052,284 | 1.95% |
| TOSHIBA MG08ACA16TE (excluded) | 93 | 2,167,676 | 1.57% |

Per-quarter Seagate failures (pooled): Q1 276, Q2 343, Q3 433, Q4 333.
Q1 pooled Seagate failures (276) match the previously extracted value.

Column coverage (share of rows non-null), identical in all four quarters; header count 179 in every checked file:
- smart_5, 9, 193, 194, 197, 198, 199: 1.00 for all four models.
- smart_187, 188, 190, 241, 242: 1.00 for the three Seagate models, 0.00 for Toshiba.
- smart_189: 1.00 for ST4000DM000 and ST8000NM0055, 0.00 for ST12000NM0008 and Toshiba.

Signal check: share of rows with raw value > 0, on the failure-day row versus all healthy rows (same window):
- smart_5: 27% vs 0.8% (MG07ACA14TA); 49-80% vs 1.7-12% (Seagate).
- smart_197: 48% vs 0.3% (MG07ACA14TA); 26-68% vs 0.3-2.9% (Seagate).
- smart_198: 12% vs 0.1% (MG07ACA14TA).
- smart_9, 193, 194: no separation (non-zero for nearly all rows).
- smart_199: ratio 1.5-1.9, weak.

## Limits

- The signal check uses the failure-day row only. It shows the columns are informative, not how much lead time they give.
- Most Toshiba failure days show no positive smart_5 or smart_198 (73% and 88% zero). Recall at a 30-day horizon is expected to be lower for Toshiba.
- `backblaze_2022Q4/2022-12-10.csv` is excluded: every row is dated 2022-12-11 and the file is a copy of the 2022-12-11 file (`smart_9_raw` differs for 411 of 235,587 shared drives). It holds 90,706 rows of the four models and no failures: ST4000DM000 18,284, ST8000NM0055 14,405, ST12000NM0008 19,831, TOSHIBA MG07ACA14TA 38,186. The drive-day figures above exclude them (they were 6,699,353 / 5,230,097 / 7,291,905 / 13,921,635, pooled 33,142,990 with the file). 2022-12-10 has no row for any drive.
- One calendar year; two vendors; no vendor held out for a transfer test.
- Models have different base rates (1.01% to 3.46%); a pooled calibration curve alone is not sufficient.
- hazard/v0.py uses smart_187 and smart_188, which are absent for Toshiba; v0 applies no multiplier from those columns to Toshiba rows.
- MG08ACA16TE is excluded for low failure count (93); it can be added later without changing the schema.

## Reproduction

Per-model failure and drive-day counts: `python backblaze/inspect_schema.py data/raw/<quarter> --failures` for each quarter.
