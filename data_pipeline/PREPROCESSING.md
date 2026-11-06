# PREPROCESSING.md — Data Pipeline (B1–B6)

Every filter, dropped row, and imputation applied to the Backblaze and Tencent
data, with row counts before and after. Updated incrementally in every PR —
never written only at the end. This file is the B6 deliverable and is what
the B-lane acceptance check (`check_outputs.py`) verifies counts against.

Status: B1–B6 are complete. The counts in sections B1–B5 below are written by the
scripts named in each section on their last run, and `check_outputs.py` (B6)
re-verifies those outputs against `manifest.json`, this file and the contract
validators. `check_outputs.py --committed-only` runs only the checks whose inputs
are tracked in git (manifest, this file, `profiles/`) and reports every check on
a `data/processed` file as SKIP, for a checkout without the generated data. B1
and B2 use the 2022 window, four models and the SMART columns in `A1_DECISION.md`
(status: proposed); B5 profiles are aggregates of selected extreme volumes over one
shared 24 h window.

---

## B1 — Backblaze extract

- Window, models and SMART columns: `data_pipeline/A1_DECISION.md`
- Quarter directories: `backblaze_2022Q1`, `backblaze_2022Q2`, `backblaze_2022Q3`, `backblaze_2022Q4`, read in date order; the file count must equal the number of calendar days from the first to the last date, no date may appear in two files, and every kept row's date must equal its file name
- Excluded files: `backblaze_2022Q4/2022-12-10.csv` (all rows dated 2022-12-11; copy of the 2022-12-11 file); 90706 rows of the kept models are not in the output and the date is a permitted gap in the calendar-span check
- Models kept: `ST4000DM000, ST8000NM0055, ST12000NM0008, TOSHIBA MG07ACA14TA`
- SMART columns kept: `smart_5_raw`, `smart_9_raw`, `smart_12_raw`, `smart_187_raw`, `smart_188_raw`, `smart_193_raw`, `smart_194_raw`, `smart_197_raw`, `smart_198_raw`, `smart_199_raw`
- A column that a model does not report (`smart_187_raw`, `smart_188_raw` for the Toshiba model) is NaN for that model; it is not dropped and not filled
- Source file digests: `manifest.json` key `sources`
- A run over the four default directories compares the per-model failures and drive-days, the pooled totals, the file count, the date range and the per-quarter Seagate failures with the values in `A1_DECISION.md` and aborts on any difference
- Counts from the last run of `backblaze/extract.py`:

- `quarter_dirs=backblaze_2022Q1,backblaze_2022Q2,backblaze_2022Q3,backblaze_2022Q4`
- `daily_files=364 date_first=2022-01-01 date_last=2022-12-31`
- `quarter=backblaze_2022Q1 files=90 failures=364 seagate_failures=276`
- `quarter=backblaze_2022Q2 files=91 failures=440 seagate_failures=343`
- `quarter=backblaze_2022Q3 files=92 failures=550 seagate_failures=433`
- `quarter=backblaze_2022Q4 files=91 failures=416 seagate_failures=333`
- `excluded_files=backblaze_2022Q4/2022-12-10.csv rows_excluded=90706`
- `models=ST4000DM000,ST8000NM0055,ST12000NM0008,TOSHIBA MG07ACA14TA`
- `columns_out=15 smart_columns=10`
- `smart_columns_all_null=0`
- `files_missing_some_smart_column=0`
- `rows_in=80122139`
- `rows_after_model_subset=33052284 dropped_other_models=47069855`
- `duplicate_serial_date_dropped=0`
- `rows_out=33052284`
- `rows_date_not_iso=91355`
- `rows_date_differs_from_filename=0`
- `rows_null_capacity_bytes=0`
- `model=ST4000DM000 drive_days=6681069 drives=18940 failures=633 smart_columns_all_null=none`
- `model=ST8000NM0055 drive_days=5215692 drives=14770 failures=348 smart_columns_all_null=none`
- `model=ST12000NM0008 drive_days=7272074 drives=20236 failures=404 smart_columns_all_null=none`
- `model=TOSHIBA MG07ACA14TA drive_days=13883449 drives=38579 failures=385 smart_columns_all_null=smart_187_raw,smart_188_raw`
- `pooled drive_days=33052284 failures=1770 afr=1.9546%`
- `verification_against_a1_decision=PASS`
- `output=data/processed/backblaze_filtered.parquet sha256=6055ce133a24063c6b09de4e36877e744ae53ec29fa567901dad2dcbd9c25e86`

- `smart_columns_all_null` counts aligned SMART columns that hold no value for any kept model; the per-model entry lists the columns with no value for that model.

## B2 — Per-drive timelines

- Input: `data/processed/backblaze_filtered.parquet` (B1; window, models and columns in `data_pipeline/A1_DECISION.md`)
- SMART columns: `smart_5_raw`, `smart_9_raw`, `smart_12_raw`, `smart_187_raw`, `smart_188_raw`, `smart_193_raw`, `smart_194_raw`, `smart_197_raw`, `smart_198_raw`, `smart_199_raw`
- `smart_187_raw` and `smart_188_raw` have no value for the Toshiba model; their level and difference features and the matching `_missing` flags are NaN and True for that model. `smart_12_raw` is kept because it is non-null in more than 99% of rows for every kept model
- Windows (calendar days, trailing): 7, 30; `smart_188_raw` uses 7 only and has no level column
- Minimum observations for a difference: 2
- Feature columns: `smart_5_raw`, `smart_5_raw_diff7d`, `smart_5_raw_diff30d`, `smart_9_raw`, `smart_9_raw_diff7d`, `smart_9_raw_diff30d`, `smart_12_raw`, `smart_12_raw_diff7d`, `smart_12_raw_diff30d`, `smart_187_raw`, `smart_187_raw_diff7d`, `smart_187_raw_diff30d`, `smart_188_raw_diff7d`, `smart_193_raw`, `smart_193_raw_diff7d`, `smart_193_raw_diff30d`, `smart_194_raw`, `smart_194_raw_diff7d`, `smart_194_raw_diff30d`, `smart_197_raw`, `smart_197_raw_diff7d`, `smart_197_raw_diff30d`, `smart_198_raw`, `smart_198_raw_diff7d`, `smart_198_raw_diff30d`, `smart_199_raw`, `smart_199_raw_diff7d`, `smart_199_raw_diff30d`
- Missingness: one `<feature>_missing` boolean per feature column; no value is imputed
- Timelines table holds backward-looking columns only; `last_seen`, `days_observed`, `failed_in_window` and `failure_date` are in the drive index, which also carries `model`
- Horizon labels, counted per drive-day, not stored: horizon 30 days. Positive: the drive's recorded failure date lies in [d, d + 30]. Negative: no failure within 30 days and the drive has a row on or after d + 30. Censored: every other row, including rows whose 30-day horizon extends past the last date of the input and rows of drives that left the data without a recorded failure. The first and last date are taken from the input
- Eligible drives: drives with at least one positive or negative row; `all_censored` drives have none
- Counts from the last run of `backblaze/timelines.py`:

- `rows_in=33052284 drives=92525`
- `window_first=2022-01-01 window_last=2022-12-31 horizon_days=30`
- `timelines_rows=33052284 timelines_columns=64`
- `drive_index_rows=92525 failed_in_window=1770`
- `model=ST12000NM0008 timelines_rows=7272074 drives=20236 failed_in_window=404`
- `model=ST4000DM000 timelines_rows=6681069 drives=18940 failed_in_window=633`
- `model=ST8000NM0055 timelines_rows=5215692 drives=14770 failed_in_window=348`
- `model=TOSHIBA MG07ACA14TA timelines_rows=13883449 drives=38579 failed_in_window=385`
- `labels_h30 model=all positive=52387 negative=30370344 censored=2629553`
- `labels_h30 model=ST12000NM0008 positive=12005 negative=6685135 censored=574934`
- `labels_h30 model=ST4000DM000 positive=18873 negative=6132680 censored=529516`
- `labels_h30 model=ST8000NM0055 positive=10077 negative=4787720 censored=417895`
- `labels_h30 model=TOSHIBA MG07ACA14TA positive=11432 negative=12764809 censored=1107208`
- `label_drives model=all drives=92525 eligible=92403 with_positive=1770 negative_only=90633 all_censored=122`
- `label_drives model=ST12000NM0008 drives=20236 eligible=20219 with_positive=404 negative_only=19815 all_censored=17`
- `label_drives model=ST4000DM000 drives=18940 eligible=18894 with_positive=633 negative_only=18261 all_censored=46`
- `label_drives model=ST8000NM0055 drives=14770 eligible=14730 with_positive=348 negative_only=14382 all_censored=40`
- `label_drives model=TOSHIBA MG07ACA14TA drives=38579 eligible=38560 with_positive=385 negative_only=38175 all_censored=19`
- `drives_with_multiple_failure_rows=0`
- `rows_with_gap_days_before_gt_1=154660`
- `rows_first_observation=92525`
- `smart_5_raw_missing=3167`
- `smart_5_raw_diff7d_missing=99040`
- `smart_5_raw_diff30d_missing=99412`
- `smart_9_raw_missing=3167`
- `smart_9_raw_diff7d_missing=99040`
- `smart_9_raw_diff30d_missing=99412`
- `smart_12_raw_missing=3167`
- `smart_12_raw_diff7d_missing=99040`
- `smart_12_raw_diff30d_missing=99412`
- `smart_187_raw_missing=13885319`
- `smart_187_raw_diff7d_missing=13941287`
- `smart_187_raw_diff30d_missing=13941836`
- `smart_188_raw_diff7d_missing=13941287`
- `smart_193_raw_missing=3167`
- `smart_193_raw_diff7d_missing=99040`
- `smart_193_raw_diff30d_missing=99412`
- `smart_194_raw_missing=3167`
- `smart_194_raw_diff7d_missing=99040`
- `smart_194_raw_diff30d_missing=99412`
- `smart_197_raw_missing=3167`
- `smart_197_raw_diff7d_missing=99040`
- `smart_197_raw_diff30d_missing=99412`
- `smart_198_raw_missing=3167`
- `smart_198_raw_diff7d_missing=99040`
- `smart_198_raw_diff30d_missing=99412`
- `smart_199_raw_missing=3167`
- `smart_199_raw_diff7d_missing=99040`
- `smart_199_raw_diff30d_missing=99412`
- `output=data/processed/backblaze_timelines.parquet rows=33052284 sha256=210636de2eff526982e1d1d91aab67e248b8a28d36c10b59c5064a71a58511fe`
- `output=data/processed/backblaze_drive_index.parquet rows=92525 sha256=c85a2bab4d3d79e7f43bdff6b33a2d0eb5190b8cda29558ab1ac41762ddb57d0`

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
- Eligibility rule: a candidate is excluded from every profile when its median bytes per I/O is below 512 (one disk sector) for reads or for writes. The median is bandwidth in KB/s x 1000 / IOPS over the direction's rows with IOPS > 0, across the volume's whole clean trace; a direction with no such row has no median and does not exclude. A median below one 512-byte sector per I/O is physically implausible and indicates an unreliable IOPS counter. The rule is applied to all candidates before any ranking and does not depend on any profile target
- Selection rules: peak = highest mean total bandwidth; diurnal = highest `daily_strength` (share of total-bandwidth variance explained by the 288-bin time-of-day mean, at least 14 days of clean rows); bursty = highest window-local p99/median of total bandwidth (the 288 window samples of the volume, median > 0); idle = lowest mean total bandwidth. Ties are broken by `disk_uid`
- N: every N from 1 to the number of ranked candidates is evaluated. All four profiles use the same one-sided rule: the largest N whose aggregate p95 does not exceed target x 1.005 (targets: peak 160, diurnal 100, bursty 120, idle 20 MB/s; the 0.5% tolerance keeps a p95 a hair above the literal target from being counted as an overshoot)
- "Closest to target" was dropped for all four profiles in favour of "largest N at or below target x 1.005": a symmetric distance is the wrong fit for a one-sided feasibility ceiling. Overshoot uses up the headroom the simulator's latency limit (`sim/config.py`) leaves for rebuild traffic, while undershoot can add a safe action, so the ladder is checked separately by `latency_ladder`. No disk is skipped or swapped to move p95 or p99/median toward a target
- Latency ladder: for each action of `ACTIONS_MBPS` and each of the 288 intervals, the foreground load is the replay bandwidth in MB/s (`sim.scenario.trace_from_frame`), the applied allocation is min(action, max(0, `rho_max` x `BW_TOTAL_MBPS` - load)), the utilisation is min(`rho_max`, (load + applied) / `BW_TOTAL_MBPS`), and the interval violates when the contract's `latency_violation` flags the simulator's mean latency (`sim.config.latency_mean_ms`, default `SimConfig`). `violation_fraction_by_action_mbps` is the violating share per action; `safe_actions` counts the actions whose share is at most `LATENCY_VIOLATION_FRACTION_MAX`
- IOPS facts (recorded only; no selection depends on them): per profile, the maximum read and write IOPS, the share of rows above the observation scales `fg_read_iops` and `fg_write_iops`, and the median bytes per I/O (bandwidth in KB/s x 1000 / IOPS over rows with IOPS > 0)
- Sidecars: `profiles/profile_<name>.meta.json` (`profile`, `synthetic` = false, `source`, `n_disks`, window start, `label`, `disclosure`, `selection_rule`), the provenance record `sim.scenario.load_replay_csv` reads next to each CSV
- p99/median of a profile is computed over its 288 window samples of total bandwidth (read + write); the synthetic generator's bursty profile gives 8.5-9.8 on the same definition
- Files: `data/processed/tencent_profile_<name>.parquet` and `profiles/profile_<name>.csv` (identical rows); selected disks in `tencent_profile_disks.parquet`; candidate statistics (daily-cycle strength, window-local p99/median) in `tencent_profile_candidates.parquet`
- Long window `diurnal_long` (additive; the four profiles are unchanged): the diurnal rule applied to a 14-day window (4032 samples). The window start is the timestamp covered by a gap-free run of that length in the largest number of volumes; those volumes are ranked, filtered and counted by the diurnal rule and N is tuned against the diurnal target p95, as for the 24 h profiles; the sum of the N volumes is the replay frame (six-column schema, `t_seconds` from 0, bandwidth in KB/s). Like the 24 h profiles it is an aggregate of selected volumes, a modelling assumption, not a sample of a typical storage group. No scaling factor is applied (`scaling_factor` = 1). Files: `data/processed/tencent_profile_diurnal_long.parquet`, `profiles/profile_diurnal_long.csv` (identical rows) and `profiles/profile_diurnal_long.meta.json`, which records the volume list, window start and end, row counts, aggregate p95 and mean and a suggested time-forward split (first 10 of 14 days of the rows, rounded down; metadata only, nothing reads it)
- Visual check: `profiles/profile_comparison.png`, one panel per profile on a shared bandwidth axis and a shared 24 h time axis
- Counts and measurements from the last run of `tencent/build_profiles.py`:

- `window_start_ts=1596686400 window_start_utc=2020-08-06 04:00`
- `candidate_disks=6817 diurnal_candidates=6377`
- `min_bytes_per_io=512 candidates_excluded_by_rule=414 candidates_after_rule=6403`
- `profile=idle n_disks=612 label=aggregate_of_612_lowest-mean-bandwidth_volumes rule="lowest mean total bandwidth" tuned=True target_p95_mbps=20 p95_mbps=20.08 p99_over_median=1.223 mean_mbps=14.34 max_mbps=23.28`
- `profile=idle ranked_candidates=6403`
- `profile=idle mean_total_bw_kbps_range=[0,46.3] zero_bandwidth_disks=26`
- `profile=idle user_type_counts={0: 355, 1: 257} disk_type_counts={0: 233, 1: 379}`
- `profile=idle safe_actions=4 violation_fraction_by_action_mbps={20: 0.0, 50: 0.0, 80: 0.0, 120: 0.0}`
- `profile=idle max_read_iops=6027 max_write_iops=23236 fraction_rows_read_iops_above_obs_scale=0 fraction_rows_write_iops_above_obs_scale=0.003472 median_bytes_per_read_io=2.468e+04 median_bytes_per_write_io=7632`
- `profile=diurnal n_disks=37 label=aggregate_of_37_strongest-daily-cycle_volumes rule="highest daily_strength (variance share of the time-of-day mean profile)" tuned=True target_p95_mbps=100 p95_mbps=96.77 p99_over_median=1.396 mean_mbps=69.08 max_mbps=111.7`
- `profile=diurnal ranked_candidates=6377`
- `profile=diurnal daily_strength_range=[0.8678,0.9965] zero_bandwidth_disks=0`
- `profile=diurnal user_type_counts={0: 19, 1: 18} disk_type_counts={0: 11, 1: 26}`
- `profile=diurnal safe_actions=3 violation_fraction_by_action_mbps={20: 0.0, 50: 0.0, 80: 0.0243, 120: 0.6979}`
- `profile=diurnal max_read_iops=9778 max_write_iops=39768 fraction_rows_read_iops_above_obs_scale=0 fraction_rows_write_iops_above_obs_scale=0.01389 median_bytes_per_read_io=2742 median_bytes_per_write_io=1.729e+04`
- `profile=bursty n_disks=36 label=aggregate_of_36_highest-window-p99/median_volumes rule="highest window-local p99/median of total bandwidth" tuned=True target_p95_mbps=120 p95_mbps=118.8 p99_over_median=3.929 mean_mbps=46.1 max_mbps=142.2`
- `profile=bursty ranked_candidates=6190`
- `profile=bursty burstiness_range=[93.73,2398] zero_bandwidth_disks=0`
- `profile=bursty user_type_counts={0: 23, 1: 13} disk_type_counts={0: 11, 1: 25}`
- `profile=bursty safe_actions=2 violation_fraction_by_action_mbps={20: 0.0, 50: 0.0069, 80: 0.1771, 120: 0.2951}`
- `profile=bursty max_read_iops=8405 max_write_iops=11416 fraction_rows_read_iops_above_obs_scale=0 fraction_rows_write_iops_above_obs_scale=0 median_bytes_per_read_io=1.05e+04 median_bytes_per_write_io=1.632e+04`
- `profile=peak n_disks=4 label=aggregate_of_4_highest-mean-bandwidth_volumes rule="highest mean total bandwidth" tuned=True target_p95_mbps=160 p95_mbps=144.4 p99_over_median=1.064 mean_mbps=135.9 max_mbps=148.9`
- `profile=peak ranked_candidates=6403`
- `profile=peak start_n=64 p95_mbps_at_start_n=741.6`
- `profile=peak mean_total_bw_kbps_range=[2.135e+04,4.372e+04] zero_bandwidth_disks=0`
- `profile=peak user_type_counts={0: 3, 1: 1} disk_type_counts={0: 1, 1: 3}`
- `profile=peak safe_actions=1 violation_fraction_by_action_mbps={20: 0.0, 50: 0.7292, 80: 1.0, 120: 1.0}`
- `profile=peak max_read_iops=2348 max_write_iops=3859 fraction_rows_read_iops_above_obs_scale=0 fraction_rows_write_iops_above_obs_scale=0 median_bytes_per_read_io=1.285e+05 median_bytes_per_write_io=1.369e+05`
- `bursty_p99_over_median=3.929 synthetic_reference_range=8.5-9.8 below_reference=True ranking_statistic=window_p99_over_median selected_on_reported_statistic=True`
- `overlap_idle_diurnal=1`
- `overlap_idle_bursty=0`
- `overlap_idle_peak=0`
- `overlap_diurnal_bursty=1`
- `overlap_diurnal_peak=0`
- `overlap_bursty_peak=0`
- `long_window=diurnal_long n_disks=44 start_ts=1594094400 start_utc=2020-07-07T04:00:00Z end_ts=1595303700 end_utc=2020-07-21T03:55:00Z rows=4032 requested_rows=4032`
- `long_window=diurnal_long volumes_with_window=6469 ranked_candidates=6054 scaling_factor=1 target_p95_mbps=100 p95_mbps=99.79 mean_mbps=69.92 max_mbps=116.8`
- `long_window=diurnal_long diurnal_24h_n_disks=37 diurnal_24h_mean_mbps=69.08 overlap_with_diurnal_24h_disks=36`
- `long_window=diurnal_long suggested_split first_holdout_row=2880 first_holdout_t_seconds=864000 first_holdout_ts=1594958400 first_holdout_utc=2020-07-17T04:00:00Z train_rows=2880 holdout_rows=1152 nominal_train_days=10 nominal_holdout_days=4`
- `output=data/processed/tencent_profile_idle.parquet rows=288 sha256=7cdc40d214a19bbfdaf981186fc1e7c21e34141c0aaf60910b697fb70cc302d1`
- `output=profiles/profile_idle.csv rows=288 sha256=d27f918486aedc10b3a3f057f48f55e993a0e1bc1aa87aa59bf2fe7b18bfb163`
- `output=profiles/profile_idle.meta.json sha256=bcb4e06ae536d116302d442450dab88fe6c5f7d8b007c188e2c62110f292fd00`
- `output=data/processed/tencent_profile_diurnal.parquet rows=288 sha256=4aa85e419716e36278340c35c21ea41e5e0ba1319b6c2056c295dfe9f21eeb6b`
- `output=profiles/profile_diurnal.csv rows=288 sha256=3948433f91b6d97ede2e5c163f9da99ab4d460c84f7412519b020d9e8953c3bc`
- `output=profiles/profile_diurnal.meta.json sha256=bc5a6497b9972970a027f185b66582eb7ae3812ad9ad7d141b3bce2ade5d8e03`
- `output=data/processed/tencent_profile_bursty.parquet rows=288 sha256=4f43c98f384cc0e03b0ca67eb6a8916b0f461718ea9de865430e6728216f1578`
- `output=profiles/profile_bursty.csv rows=288 sha256=fee15ef6ab400e35b0971ac4079c08f050cf2eeabea25048b950cd0fa89b3a0e`
- `output=profiles/profile_bursty.meta.json sha256=9aa8d2027c1c7bdef2878908ea42e6f3cd6bd2d863a4f7790819720a928c423c`
- `output=data/processed/tencent_profile_peak.parquet rows=288 sha256=da71bb6d1774a859a5b0d6c8a6f1b835d436096dc5e0b861391fef94d62b8574`
- `output=profiles/profile_peak.csv rows=288 sha256=b57927ff6fbb3e465d391c5ac63e22a8d255cbb6dab664f11938b71f34e50f4d`
- `output=profiles/profile_peak.meta.json sha256=82796977f59a3398620d5394cb936f484e453300019023d899a9ea4dc2f20ce2`
- `output=profiles/profile_comparison.png sha256=698913d665d6b472daf88d3d9f92d71dff4c73283d8aa4ea07ac9c78d1659d09`
- `output=data/processed/tencent_profile_disks.parquet rows=689 sha256=fe8a5e5b26a683cd4893e980498aa436d663e74df6f8b8d7393d1abb87b12565`
- `output=data/processed/tencent_profile_candidates.parquet rows=6817 sha256=811c04efd4c475ddafffb106b5b72e52dd8b1a52815fe0c22b9ab748ac433e86`
- `output=data/processed/tencent_profile_diurnal_long.parquet rows=4032 sha256=24320474634b758922fadb3c833b6db27690298125ecbf98a885a231468a9e5a`
- `output=profiles/profile_diurnal_long.csv rows=4032 sha256=4b0f6cf85fc4f88bf80ac23d44120c127c88e5ad41f46eb3cfc2fec576c74001`
- `output=profiles/profile_diurnal_long.meta.json sha256=04be33b748daea85580b8dfe16620651807e0983956fd62a67d265bafca12ae9`

## B6 — This file / acceptance

- `manifest.json` lists every output file with its sha256 and, for tables, its
  row count (the plot and the sidecars have no row count).
- `check_outputs.py` prints one PASS / FAIL / MISSING / SKIP line per check. The
  last line is `RESULT: PASS`, `RESULT: FAIL` (any FAIL) or `RESULT: INCOMPLETE`
  (no FAIL, some MISSING); the exit status is non-zero for FAIL and INCOMPLETE:
  - `manifest.json` and `PREPROCESSING.md` exist, and the manifest lists every
    output file;
  - sha256 of every file in the manifest, and its row count where one is
    recorded;
  - row counts of the Backblaze extract, timelines and drive index, the Tencent
    clean table, summary, runs and disk index, and the profile files, against
    the counts recorded in this file;
  - key uniqueness: `(serial_number, date)` in the extract and the timelines,
    `serial_number` in the drive index, `disk_uid` in the disk summary (rows
    with a null key value are compared as `DataFrame.duplicated` compares them);
  - the Backblaze extract against the values recorded in `backblaze/extract.py`
    (`A1_DECISION.md`): the model set, failures and drive-days per model and pooled,
    the date range and 364 distinct dates, the NaN pattern of `smart_187_raw` and
    `smart_188_raw` (all NaN for the Toshiba model, populated for the Seagate models),
    the populated share of `smart_12_raw` (at least 0.99 for every model), and the 364
    source files in `manifest.json`;
  - the 30-day label counts and the eligible-drive counts recomputed from the
    timelines and the drive index against the lines in B2 above;
  - the A2 label table (one row per timelines row in the timelines order with the same
    `serial_number`, `date` and `model` values, unique `(serial_number, date)`, label values in
    {-1, 0, 1}, totals equal to the B2 lines) and the A2 drive split (the required columns, one
    row per drive of the drive index, `model` and `ever_failed` equal to the drive index, split
    values, no drive in two splits, a failed drive of every model in every split, stratum sizes
    within one drive of the target shares, per-split counts equal to the A2 lines);
  - `tests/test_no_leakage.py` exists, and a cut-and-recompute of the stored
    timelines on a sample of drives at the midpoint of the date range is identical to
    the stored rows and to the truncated recomputation;
  - `validate_disk_index_frame` (contract) on the disk index;
  - `validate_replay_frame` and `validate_replay_grid` (contract) on each of
    the four profiles as CSV and as Parquet, the 288-row length, and equality
    of the CSV and Parquet versions;
  - the four provenance sidecars `profiles/profile_<name>.meta.json` parse,
    `synthetic` is false, `profile` equals the profile name and `n_disks`
    equals the manifest;
  - the latency ladder of each profile CSV (`latency_ladder`) gives the expected
    number of safe rebuild allocations: idle 4, diurnal 3, bursty 2, peak 1;
  - `profile_comparison.png`, `tencent_profile_disks.parquet` and
    `tencent_profile_candidates.parquet` are present, listed in the manifest
    and named in this file, and the per-profile disk counts in the manifest
    agree with `tencent_profile_disks.parquet`.
- `check_outputs.py --committed-only` runs the checks whose inputs are tracked
  in git: the two existence checks, the manifest listing, and, for the files
  outside `data/processed` (the profile CSVs, sidecars and plot), sha256, row
  counts, the replay contract and grid on the CSVs, the sidecars and the ladder.
  Each check on a `data/processed` file is printed as `[SKIP] ... not tracked in
  git`.
- `check_outputs.py` does not re-run the pipeline scripts and does not judge
  the plot; whether the four profiles are visibly different is read from
  `profiles/profile_comparison.png`.

## A2 — Horizon labels and drive split

- Inputs: `data/processed/backblaze_timelines.parquet`, `data/processed/backblaze_drive_index.parquet`; decision record: `data_pipeline/A2_DECISION.md`
- Label rule: the 30-day rule of section B2 (`timelines.row_labels`), imported unchanged. `backblaze_labels.parquet` holds `serial_number`, `date`, `model` and `label` (1 positive, 0 negative, -1 censored) for every timelines row, in the timelines row order
- Split rule: unit = drive; 70/15/15 train/val/test; drives sorted by `serial_number` and shuffled with a seeded generator (seed 20220101) inside each stratum (`model`, `ever_failed`); largest-remainder rounding per stratum. `backblaze_drive_split.parquet` has one row per drive of the drive index, including the fully censored drives
- The split is not time-forward: drives span the whole window, so the three splits overlap in dates (`a2_split_dates` lines give the date range and the positive rows by month of each split)
- Censored rows (label -1) are unknown outcomes and are excluded from training and scoring
- Counts from the last run of `backblaze/labels.py`:

- `a2_labels rows=33052284 positive=52387 negative=30370344 censored=2629553`
- `a2_params seed=20220101 fractions=train:0.7,val:0.15,test:0.15 strata=model,ever_failed`
- `a2_split split=train model=ST12000NM0008 drives=14165 failed_drives=283 positive=8522 negative=4679379 censored=402427`
- `a2_split split=val model=ST12000NM0008 drives=3036 failed_drives=61 positive=1725 negative=1001902 censored=86245`
- `a2_split split=test model=ST12000NM0008 drives=3035 failed_drives=60 positive=1758 negative=1003854 censored=86262`
- `a2_split split=train model=ST4000DM000 drives=13258 failed_drives=443 positive=13232 negative=4289970 censored=370514`
- `a2_split split=val model=ST4000DM000 drives=2841 failed_drives=95 positive=2806 negative=923164 censored=79479`
- `a2_split split=test model=ST4000DM000 drives=2841 failed_drives=95 positive=2835 negative=919546 censored=79523`
- `a2_split split=train model=ST8000NM0055 drives=10340 failed_drives=244 positive=7148 negative=3351946 censored=292354`
- `a2_split split=val model=ST8000NM0055 drives=2215 failed_drives=52 positive=1477 negative=718482 censored=62608`
- `a2_split split=test model=ST8000NM0055 drives=2215 failed_drives=52 positive=1452 negative=717292 censored=62933`
- `a2_split split=train model=TOSHIBA MG07ACA14TA drives=27005 failed_drives=269 positive=7969 negative=8933167 censored=775008`
- `a2_split split=val model=TOSHIBA MG07ACA14TA drives=5787 failed_drives=58 positive=1754 negative=1915756 censored=166103`
- `a2_split split=test model=TOSHIBA MG07ACA14TA drives=5787 failed_drives=58 positive=1709 negative=1915886 censored=166097`
- `a2_split split=train model=all drives=64768 failed_drives=1239 positive=36871 negative=21254462 censored=1840303`
- `a2_split split=val model=all drives=13879 failed_drives=266 positive=7762 negative=4559304 censored=394435`
- `a2_split split=test model=all drives=13878 failed_drives=265 positive=7754 negative=4556578 censored=394815`
- `a2_split_dates split=train date_min=2022-01-01 date_max=2022-12-31 positive_by_month=2022-01:2215,2022-02:2906,2022-03:3182,2022-04:3367,2022-05:3370,2022-06:3231,2022-07:3703,2022-08:3951,2022-09:3672,2022-10:3478,2022-11:2740,2022-12:1056`
- `a2_split_dates split=val date_min=2022-01-01 date_max=2022-12-31 positive_by_month=2022-01:630,2022-02:487,2022-03:530,2022-04:581,2022-05:575,2022-06:828,2022-07:779,2022-08:923,2022-09:909,2022-10:779,2022-11:518,2022-12:223`
- `a2_split_dates split=test date_min=2022-01-01 date_max=2022-12-31 positive_by_month=2022-01:516,2022-02:485,2022-03:510,2022-04:696,2022-05:860,2022-06:857,2022-07:1019,2022-08:825,2022-09:588,2022-10:677,2022-11:530,2022-12:191`
- `output=data/processed/backblaze_labels.parquet rows=33052284 sha256=57067c5c8c3f3666cf94b21507d8af43162b70c6a0fd05e0b36b5ea8c1482263`
- `output=data/processed/backblaze_drive_split.parquet rows=92525 sha256=537b52c54ebf8744532f3af1311efbc13b02f70d83196d850246d2354afd36a1`
