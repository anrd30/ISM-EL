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

- Source field order (dataset README): `timestamp, read_iops, read_bw_kbps, write_iops, write_bw_kbps, disk_usage_mb`; files are headerless
- Clean table `data/processed/tencent_load_clean.parquet`: columns `disk_uid, timestamp, read_iops, write_iops, read_bw_kbps, write_bw_kbps, disk_usage_mb`, all int64, one row group per disk, rows sorted by timestamp
- `timestamp` is source epoch seconds; the rebase to `t_seconds` starting at 0 happens when a window is cut (`to_replay_frame`)
- Units: bandwidth stored as **KB/s** (source unit, not pre-converted — see CLAUDE.md §6 Open Question 4)
- Cleaning order: sort by timestamp; drop exact duplicate rows; drop every row of a timestamp that still repeats with differing values; drop rows with a negative value. Gaps are reported, never filled
- Subscription table: first row kept per `disk_uid`; `conflicting_uid` lines below are uids whose repeated rows disagree
- Counts from the last run of `tencent/extract_load.py`:

- `load_files=16296`
- `rows_raw=271868975`
- `out_of_order_rows=7812 disks=5646`
- `exact_duplicate_rows_dropped=1592593 disks=5168`
- `conflict_timestamps=674543 conflict_rows_dropped=1349086 disks=3294`
- `negative_rows_dropped=0 disks=0`
- `rows_clean=268927296`
- `off_grid_rows=0 disks=0`
- `disks_with_breaks=7298 missing_slots=4693480`
- `disks_with_17280_clean_rows=9132`
- `subscription_rows=16311 unique_uids=16296 repeated_uids=15 conflicting_uids=2`
- `subscription_only_uids=0 load_only_uids=0`
- `conflicting_uid=3fb7e888-6abd-4b89-8252-72bfa4a15dc1`
- `conflicting_uid=c94098d8-c464-4396-af24-87e7a701d377`
- `output=data/processed/tencent_load_clean.parquet rows=268927296 sha256=afb5c4823f5eb657b455b1116b0e80053d7816ecf76245cea4b90d379a1c3de3`
- `output=data/processed/tencent_disk_summary.parquet rows=16296 sha256=d8dc097d58232a4f12302dfb8fc7c23b5c5235c44236e4efbf6c1fd6858fe337`
- `output=data/processed/tencent_disk_runs.parquet rows=32032 sha256=454a6369571498dafc87fd95777a4d7b3f87ab7051c5281279aecf8652326029`
- `conflict_diagnostic timestamps=674543`
- `conflict_load_identical_usage_differs=167824`
- `conflict_load_rounding_level(abs<=1)=1568`
- `conflict_load_small(rel<=0.1)=0`
- `conflict_load_large(rel>0.1)=505151`
- `conflict_load_abs_spread_where_differs median=47 p90=298 p99=9376 max=1.514e+05`
- `conflict_load_rel_spread_where_differs median=1 p90=1 p99=1 max=1`
- `conflict_usage_spread_mb median=0 p90=2048 p99=1.843e+04 max=5.56e+05`
- `conflict_rows_per_timestamp pairs=674543 more_than_two=0 max=2`
- `conflict_blocks=5520 block_rows median=136 p90=288 p99=288 max=1440 multiple_of_288=895`
- `disks_with_conflicts_and_out_of_order=3294`
- `agg_disks=16296 disks_with_clean_rows=16296`
- `agg_mean_total_bw_kbps median=85.37 p90=495.3 p99=5559 max=4.372e+04`
- `agg_peak_total_bw_kbps median=4458 p90=2.435e+04 p99=6.972e+04 max=1.581e+05`
- `agg_burstiness_p99_over_median disks_defined=14408 disks_median_zero=1888 median=2.733 p90=45.54 p99=299.9 max=4539`
- `agg_disks_with_24h_window=16296 sampling_offset_classes=1`
- `agg_disks_sharing_a_24h_window=16296`
- `agg_largest_24h_group=6817 window_start_ts=1596686400 window_start_utc=2020-08-06 04:00`

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
