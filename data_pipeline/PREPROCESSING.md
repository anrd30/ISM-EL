# PREPROCESSING.md — Data Pipeline (B1–B6)

Every filter, dropped row, and imputation applied to the Backblaze and Tencent
data, with row counts before and after. Updated incrementally in every PR —
never written only at the end. This file is the B6 deliverable and is what
the B-lane acceptance check (`check_outputs.py`) verifies counts against.

Status: **scaffold only — no data processed yet.** Sections below are filled
in as each task (B1–B5) is implemented.

---

## B1 — Backblaze extract

- Quarter used: `TODO — confirm with Aakarsh (A1)`
- Drive model(s) kept: `TODO — confirm with Aakarsh (A1)`
- SMART columns kept: `TODO — confirm with Aakarsh (A1)`
- Source files and sha256: `TODO`
- Rows before filtering → after column projection → after model subset → after dedup: `TODO`
- Duplicate `(serial_number, date)` rows dropped: `TODO`

## B2 — Per-drive timelines

- Rolling-window sizes used (e.g. `_d1`, `_d7`) and why: `TODO — confirm with Aakarsh`
- Missingness columns added: `TODO`
- Censoring fields added (`first_seen`, `last_seen`, `days_observed`, `failed_in_window`): `TODO`
- Rows before → after: `TODO`

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
