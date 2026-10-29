# PREPROCESSING.md — Data Pipeline (B1–B6)

Every filter, dropped row, and imputation applied to the Backblaze and Tencent
data, with row counts before and after. Updated incrementally in every PR —
never written only at the end. This file is the B6 deliverable and is what
the B-lane acceptance check (`check_outputs.py`) verifies counts against.

Status: **scaffold only — no data processed yet.** Sections below are filled
in as each task (B1–B5) is implemented.

---

## B1 — Backblaze extract

- Quarter: `backblaze_2022Q1` (provisional)
- Models kept: `ST4000DM000, ST8000NM0055, ST12000NM0008` (provisional)
- SMART columns kept: every `smart_*_raw` column present in any daily file (provisional)
- Source file digests: `manifest.json` key `sources`
- Counts from the last run of `backblaze/extract.py`:

- `quarter_dir=backblaze_2022Q1 daily_files=90`
- `models=ST4000DM000,ST8000NM0055,ST12000NM0008`
- `columns_out=92 smart_columns=87`
- `smart_columns_all_null=60`
- `files_missing_some_smart_column=0`
- `rows_in=18845260`
- `rows_after_model_subset=4775365 dropped_other_models=14069895`
- `duplicate_serial_date_dropped=0`
- `rows_out=4775365`
- `rows_date_not_iso=53144`
- `rows_date_differs_from_filename=0`
- `rows_null_capacity_bytes=0`
- `model=ST4000DM000 drive_days=1670395 drives=18614 failures=117`
- `model=ST8000NM0055 drive_days=1289659 drives=14409 failures=86`
- `model=ST12000NM0008 drive_days=1815311 drives=20212 failures=73`
- `pooled drive_days=4775365 failures=276 afr=2.1096%`
- `output=data/processed/backblaze_filtered.parquet sha256=0cce95ff31b2fb6b3d4da14badefe8a46d42714ca2e16ca1b815d55de8471c9c`

- `smart_columns_all_null` counts aligned SMART columns that hold no value for the kept models; they are retained so the column list stays fixed.

## B2 — Per-drive timelines

- Input: `data/processed/backblaze_filtered.parquet` (provisional B1 subset)
- SMART columns: `smart_5_raw`, `smart_9_raw`, `smart_12_raw`, `smart_187_raw`, `smart_188_raw`, `smart_193_raw`, `smart_194_raw`, `smart_197_raw`, `smart_198_raw`
- Windows (calendar days, trailing): 7, 30; `smart_188_raw` uses 7 only and has no level column
- Minimum observations for a difference: 2
- Feature columns: `smart_5_raw`, `smart_5_raw_diff7d`, `smart_5_raw_diff30d`, `smart_9_raw`, `smart_9_raw_diff7d`, `smart_9_raw_diff30d`, `smart_12_raw`, `smart_12_raw_diff7d`, `smart_12_raw_diff30d`, `smart_187_raw`, `smart_187_raw_diff7d`, `smart_187_raw_diff30d`, `smart_188_raw_diff7d`, `smart_193_raw`, `smart_193_raw_diff7d`, `smart_193_raw_diff30d`, `smart_194_raw`, `smart_194_raw_diff7d`, `smart_194_raw_diff30d`, `smart_197_raw`, `smart_197_raw_diff7d`, `smart_197_raw_diff30d`, `smart_198_raw`, `smart_198_raw_diff7d`, `smart_198_raw_diff30d`
- Missingness: one `<feature>_missing` boolean per feature column; no value is imputed
- Timelines table holds backward-looking columns only; `last_seen`, `days_observed`, `failed_in_window` and `failure_date` are in the drive index
- Counts from the last run of `backblaze/timelines.py`:

- `rows_in=4775365 drives=53235`
- `timelines_rows=4775365 timelines_columns=58`
- `drive_index_rows=53235 failed_in_window=276`
- `drives_with_multiple_failure_rows=0`
- `rows_with_gap_days_before_gt_1=420`
- `rows_first_observation=53235`
- `smart_5_raw_missing=528`
- `smart_5_raw_diff7d_missing=54357`
- `smart_5_raw_diff30d_missing=54732`
- `smart_9_raw_missing=528`
- `smart_9_raw_diff7d_missing=54357`
- `smart_9_raw_diff30d_missing=54732`
- `smart_12_raw_missing=528`
- `smart_12_raw_diff7d_missing=54357`
- `smart_12_raw_diff30d_missing=54732`
- `smart_187_raw_missing=528`
- `smart_187_raw_diff7d_missing=54357`
- `smart_187_raw_diff30d_missing=54732`
- `smart_188_raw_diff7d_missing=54357`
- `smart_193_raw_missing=528`
- `smart_193_raw_diff7d_missing=54357`
- `smart_193_raw_diff30d_missing=54732`
- `smart_194_raw_missing=528`
- `smart_194_raw_diff7d_missing=54357`
- `smart_194_raw_diff30d_missing=54732`
- `smart_197_raw_missing=528`
- `smart_197_raw_diff7d_missing=54357`
- `smart_197_raw_diff30d_missing=54732`
- `smart_198_raw_missing=528`
- `smart_198_raw_diff7d_missing=54357`
- `smart_198_raw_diff30d_missing=54732`
- `output=data/processed/backblaze_timelines.parquet rows=4775365 sha256=e69a9f721555a375967570cbff4aca3e22f6d54eef78670e0c18e34b39f241cb`
- `output=data/processed/backblaze_drive_index.parquet rows=53235 sha256=8164b58225818faf6cb56cbe490ee81c6f1e4eb1df60cea8641d3fc84b44f594`

## B3 — Tencent extract

- Disk(s) / warehouse(s) selected and why: `TODO`
- Column mapping applied (source → canonical `t_seconds, read_iops, write_iops,
  read_bw_kbps, write_bw_kbps, disk_usage_mb`): `TODO`
- Units: bandwidth stored as **KB/s** (source unit, not pre-converted — see
  CLAUDE.md §6 Open Question 4). Conversion to MB/s happens in the
  simulator-side adapter, not here.
- Gap / duplicate / negative-value counts per disk: `TODO`

## B4 — Metadata join

- Metadata columns attached (`disk_attr, disk_type, vm_cpu, vm_memory,
  disk_capacity`) by `disk_uid`: `TODO`
- Unmatched `disk_uid` count (subscription side / load side): `TODO`
- Stored as a separate disk-index file, not merged into the B3 replay files
  (see CLAUDE.md §4, B3 vs B4 note).

## B5 — Four intensity profiles

- Selection thresholds used for idle / diurnal / bursty / peak: `TODO — written
  down here before any window is chosen, per CLAUDE.md §4`
- Disk(s) and original time ranges chosen for each profile: `TODO`
- Row count per profile file: target 288 rows (24h at 300s). Verified by
  `tests/test_profile_grid.py`.

## B6 — This file / acceptance

- `manifest.json` lists every output file with row count and sha256.
- `check_outputs.py` re-verifies every count in this file against the actual
  output files, plus column names, dtypes, unique keys, and the 288-row /
  300-second grid for each profile.
