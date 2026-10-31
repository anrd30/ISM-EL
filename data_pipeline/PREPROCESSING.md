# PREPROCESSING.md — Data Pipeline (B1–B6)

Every filter, dropped row, and imputation applied to the Backblaze and Tencent
data, with row counts before and after. Updated incrementally in every PR —
never written only at the end. This file is the B6 deliverable and is what
the B-lane acceptance check (`check_outputs.py`) verifies counts against.

Status: B1–B5 are implemented and have been run on the full data; the counts in
sections B1–B5 below are written by the scripts named in each section on their
last run. B6 is in progress: `check_outputs.py` re-verifies those outputs
against `manifest.json`, this file and the contract validators (see the B6
section). B1 and B2 use one provisional quarter and a provisional three-model
subset; B5 profiles are aggregates of selected extreme volumes over one shared
24 h window.

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
- `timestamp` is source epoch seconds; the rebase to `t_seconds` starting at 0 happens when a window is cut (`contracts.contracts.to_replay_frame`)
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

- Output `data/processed/tencent_disk_index.parquet`: one row per unique `disk_uid`, columns `disk_uid, disk_capacity, disk_attr, disk_type, user_type, vm_cpu, vm_memory` (`DISK_INDEX_COLUMNS`, the dataset README field order); `disk_uid` is a string, the other columns int64
- Stored as a separate disk-index file, not merged into the B3 replay files (see CLAUDE.md §4, B3 vs B4 note); join on `disk_uid`
- Duplicate `disk_uid` rule: the first row in file order is kept. `conflicting_uid` lines below are ids whose repeated rows disagree; they are compared with the `conflicting_uid` lines of the B3 section
- Ids present on one side only (subscription vs `tencent_load_clean.parquet`) are counted and listed (first 10 per side); no row or disk is dropped for a mismatch
- Counts from the last run of `tencent/build_disk_index.py`:

- `subscription_rows_in=16311`
- `duplicate_rows_dropped=15 repeated_uids=15`
- `disk_index_rows=16296`
- `conflicting_uids=2 documented_in_b3=2 match=True`
- `conflicting_uid=3fb7e888-6abd-4b89-8252-72bfa4a15dc1`
- `conflicting_uid=c94098d8-c464-4396-af24-87e7a701d377`
- `subscription_only_uids=0 load_only_uids=0`
- `output=data/processed/tencent_disk_index.parquet rows=16296 sha256=43521cb615bc73c7f205d5d69a14441b53012131e302307b748ec8db11f6be7b`

## B5 — Four intensity profiles

- Each profile is the sum of N real volumes over one shared 24 h window (288 samples, 300 s), in the six-column replay schema, bandwidth in KB/s. Each is an aggregate of N volumes chosen by character: selected extremes, not a representative sample of volumes or of a storage group
- Bursty: selected extremes, not a representative sample; volumes are ranked by window-local p99/median and the same statistic is then reported for the aggregate (selection on the outcome), so the reported value is high partly by construction
- Window: the start timestamp shared by the most volumes with a gap-free 24 h run; only those volumes are candidates
- Selection rules: peak = highest mean total bandwidth; diurnal = highest `daily_strength` (share of total-bandwidth variance explained by the 288-bin time-of-day mean, at least 14 days of clean rows); bursty = highest window-local p99/median of total bandwidth (the 288 window samples of the volume, median > 0); idle = lowest mean total bandwidth. Ties are broken by `disk_uid`
- N: every N from 1 to the number of ranked candidates is evaluated. All four profiles use the same one-sided rule: the largest N whose aggregate p95 does not exceed target x 1.005 (targets: peak 160, diurnal 100, bursty 120, idle 20 MB/s; the 0.5% tolerance keeps a p95 a hair above the literal target from being counted as an overshoot)
- "Closest to target" was dropped for all four profiles in favour of "largest N at or below target x 1.005": a symmetric distance is the wrong fit for a one-sided feasibility ceiling. Overshoot uses up the headroom the simulator's latency limit (`sim/config.py`) leaves for rebuild traffic, while undershoot only makes the profile lighter. No disk is skipped or swapped to move p95 or p99/median toward a target
- p99/median of a profile is computed over its 288 window samples of total bandwidth (read + write); the synthetic generator's bursty profile gives 8.5-9.8 on the same definition
- Files: `data/processed/tencent_profile_<name>.parquet` and `profiles/profile_<name>.csv` (identical rows); selected disks in `tencent_profile_disks.parquet`; candidate statistics (daily-cycle strength, window-local p99/median) in `tencent_profile_candidates.parquet`
- Visual check: `profiles/profile_comparison.png`, one panel per profile on a shared bandwidth axis and a shared 24 h time axis
- Counts and measurements from the last run of `tencent/build_profiles.py`:

- `window_start_ts=1596686400 window_start_utc=2020-08-06 04:00`
- `candidate_disks=6817 diurnal_candidates=6759`
- `profile=idle n_disks=748 label=aggregate_of_748_lowest-mean-bandwidth_volumes rule="lowest mean total bandwidth" tuned=True target_p95_mbps=20 p95_mbps=20.09 p99_over_median=1.224 mean_mbps=14.13 max_mbps=22.86`
- `profile=idle mean_total_bw_kbps_range=[0,45.18] zero_bandwidth_disks=26`
- `profile=idle user_type_counts={0: 463, 1: 285} disk_type_counts={0: 317, 1: 431}`
- `profile=diurnal n_disks=39 label=aggregate_of_39_strongest-daily-cycle_volumes rule="highest daily_strength (variance share of the time-of-day mean profile)" tuned=True target_p95_mbps=100 p95_mbps=96.83 p99_over_median=1.395 mean_mbps=69.13 max_mbps=111.8`
- `profile=diurnal daily_strength_range=[0.8678,0.9965] zero_bandwidth_disks=0`
- `profile=diurnal user_type_counts={0: 19, 1: 20} disk_type_counts={0: 13, 1: 26}`
- `profile=bursty n_disks=23 label=aggregate_of_23_highest-window-p99/median_volumes rule="highest window-local p99/median of total bandwidth" tuned=True target_p95_mbps=120 p95_mbps=116 p99_over_median=5.909 mean_mbps=38.92 max_mbps=153.5`
- `profile=bursty burstiness_range=[130.7,2398] zero_bandwidth_disks=0`
- `profile=bursty user_type_counts={0: 15, 1: 8} disk_type_counts={0: 7, 1: 16}`
- `profile=peak n_disks=4 label=aggregate_of_4_highest-mean-bandwidth_volumes rule="highest mean total bandwidth" tuned=True target_p95_mbps=160 p95_mbps=144.4 p99_over_median=1.064 mean_mbps=135.9 max_mbps=148.9`
- `profile=peak start_n=64 p95_mbps_at_start_n=752.1`
- `profile=peak mean_total_bw_kbps_range=[2.135e+04,4.372e+04] zero_bandwidth_disks=0`
- `profile=peak user_type_counts={0: 3, 1: 1} disk_type_counts={0: 1, 1: 3}`
- `bursty_p99_over_median=5.909 synthetic_reference_range=8.5-9.8 below_reference=True ranking_statistic=window_p99_over_median selected_on_reported_statistic=True`
- `overlap_idle_diurnal=3`
- `overlap_idle_bursty=0`
- `overlap_idle_peak=0`
- `overlap_diurnal_bursty=1`
- `overlap_diurnal_peak=0`
- `overlap_bursty_peak=0`
- `output=data/processed/tencent_profile_idle.parquet rows=288 sha256=4c2d82aee4a29cb056d5f4178ea788bf97ef1887f499ca851e651f3296afef65`
- `output=profiles/profile_idle.csv rows=288 sha256=a8977c722067997f248eac271ccc44211b7eb6801168a0eb1a5b55b60258d6fd`
- `output=data/processed/tencent_profile_diurnal.parquet rows=288 sha256=7794fd7754e81dac2ab082a24cf0216f530fa9f7872d9e08eb144b3082c6b02d`
- `output=profiles/profile_diurnal.csv rows=288 sha256=83643f1fb9717441966b3532548104787750e08d9e70079c2061dc96916c6ca9`
- `output=data/processed/tencent_profile_bursty.parquet rows=288 sha256=6e3b44e69d59ad7215701299b8b67f4918fb6b069d1962bb589c2f553c1acf48`
- `output=profiles/profile_bursty.csv rows=288 sha256=6432c2bac432380944a2a190bf13921579efe0e1330f323eade4ee8c988a7985`
- `output=data/processed/tencent_profile_peak.parquet rows=288 sha256=da71bb6d1774a859a5b0d6c8a6f1b835d436096dc5e0b861391fef94d62b8574`
- `output=profiles/profile_peak.csv rows=288 sha256=b57927ff6fbb3e465d391c5ac63e22a8d255cbb6dab664f11938b71f34e50f4d`
- `output=profiles/profile_comparison.png sha256=2b2e11be223e9f8e2311b3ec1498817204ffc44ce0c155a9af6567b22d35cf3d`
- `output=data/processed/tencent_profile_disks.parquet rows=814 sha256=60c4a9946a40167da73e00c0f77c727b44a8c18521d1908555c1a492257c8e4c`
- `output=data/processed/tencent_profile_candidates.parquet rows=6817 sha256=7064a68540bec367c26d6bc8b8dbd8098090187733df6c7665b5324709e3c696`

## B6 — This file / acceptance

- `manifest.json` lists every output file with its sha256 and, for tables, its
  row count (the plot has no row count).
- `check_outputs.py` prints one PASS / FAIL / MISSING line per check and exits
  with status 1 on any FAIL or MISSING:
  - `manifest.json` and `PREPROCESSING.md` exist, and the manifest lists every
    output file;
  - sha256 of every file in the manifest, and its row count where one is
    recorded;
  - row counts of the Backblaze extract, timelines and drive index, the Tencent
    clean table, summary, runs and disk index, and the profile files, against
    the counts recorded in this file;
  - key uniqueness: `(serial_number, date)` in the extract and the timelines,
    `serial_number` in the drive index, `disk_uid` in the disk summary;
  - `validate_disk_index_frame` (contract) on the disk index;
  - `validate_replay_frame` and `validate_replay_grid` (contract) on each of
    the four profiles as CSV and as Parquet, the 288-row length, and equality
    of the CSV and Parquet versions;
  - `profile_comparison.png`, `tencent_profile_disks.parquet` and
    `tencent_profile_candidates.parquet` are present, listed in the manifest
    and named in this file, and the per-profile disk counts in the manifest
    agree with `tencent_profile_disks.parquet`.
- `check_outputs.py` does not re-run the pipeline scripts and does not judge
  the plot; whether the four profiles are visibly different is read from
  `profiles/profile_comparison.png`.
