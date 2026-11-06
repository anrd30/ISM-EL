"""
B5: four load-replay profiles (idle, diurnal, bursty, peak) from the cleaned
Tencent traces.

Each profile is the sum of N real volumes over one shared 24 h window
(288 samples at 300 s), rebased to the six-column replay schema and checked
with the contract's `validate_replay_frame`. The volumes are chosen by
character, not at random, so each profile is an aggregate of N selected
extreme volumes (highest mean, strongest daily cycle, highest p99/median,
lowest mean). It is not a sample of a typical storage group.

Window: the start timestamp shared by the largest number of volumes that
have a gap-free 24 h run; volumes without one are not candidates.

Eligibility: a candidate is excluded from every profile when its median bytes
per I/O is below MIN_BYTES_PER_IO (512, one disk sector) for reads or for writes.
The median is bandwidth in KB/s x 1000 / IOPS over the direction's rows with
IOPS > 0, across the volume's whole clean trace; a direction with no such row has
no median and does not exclude. A median below one sector per I/O is physically
implausible and indicates an unreliable IOPS counter. The rule is applied to all
candidates before any ranking and does not depend on any profile target.

Selection per profile (ties broken by `disk_uid`):
  peak     highest mean total bandwidth; N is the largest whose aggregate p95
           does not exceed PEAK_TARGET_MBPS
  diurnal  highest daily_strength (share of total-bandwidth variance explained
           by the 288-bin time-of-day mean profile; at least MIN_DAYS of clean
           rows); N is the largest whose aggregate p95 does not exceed
           DIURNAL_TARGET_MBPS
  bursty   highest window-local p99/median of total bandwidth (the 288 window
           samples only, median > 0); N is the largest whose aggregate p95
           does not exceed BURSTY_TARGET_MBPS
  idle     lowest mean total bandwidth; N is the largest whose aggregate p95
           does not exceed IDLE_TARGET_MBPS

Tuning evaluates every N from 1 to the number of ranked candidates. Mode
"not_above" (all four profiles) keeps the largest N whose aggregate p95 is at
or below target x (1 + CEILING_TOLERANCE). The aggregate p95 is non-decreasing
in N, and foreground load above the target uses up the headroom the simulator's
latency limit leaves for rebuild traffic, while load below it can add a safe
action, so the ladder is checked separately by `latency_ladder`; a symmetric
distance to the target would treat the two alike. `tune_n` raises ValueError
when no N satisfies the ceiling. Mode "closest" (nearest p95, smaller N on a
tie) is implemented in `tune_n` and not used by PROFILE_SPECS. No disk is
skipped or swapped to move p95 or p99/median toward a target. The bursty ranking statistic and the
p99/median reported for its aggregate are the same statistic, so that value is
high partly by construction. Total bandwidth is read plus write in KB/s; MB/s
divides by 1000.

Long window (`diurnal_long`, additive; the four profiles above are unchanged):
the diurnal rule applied to a 14-day window (LONG_WINDOW_ROWS samples) instead of
24 h. The start is the timestamp covered by a gap-free run of that length in the
largest number of volumes; candidates are those volumes, ranked, filtered and
counted with the diurnal rule (`collect`, `build_profile`), N tuned against
DIURNAL_TARGET_MBPS, and the sum of the N volumes is the replay frame, six-column
schema, `t_seconds` from 0. No scaling factor is applied (recorded as 1). The
sidecar records the volumes, window, row counts, aggregate p95 and mean, and a
time-forward split point (first LONG_TRAIN_DAYS of LONG_WINDOW_DAYS of the rows,
rounded down); no code reads the split.

`latency_ladder` gives, per rebuild allocation in ACTIONS_MBPS, the share of the
288 intervals that violate the latency limit under the simulator's own queue
model, and `safe_actions`, the number of allocations whose share is at most
LATENCY_VIOLATION_FRACTION_MAX. `iops_facts` records IOPS maxima, the share of
rows above the observation scales and the median bytes per I/O of each profile;
no selection depends on them.

Usage:
    python -I build_profiles.py

Outputs:
    data/processed/tencent_profile_<name>.parquet
    profiles/profile_<name>.csv
    profiles/profile_<name>.meta.json
    data/processed/tencent_profile_diurnal_long.parquet
    profiles/profile_diurnal_long.csv
    profiles/profile_diurnal_long.meta.json
    data/processed/tencent_profile_disks.parquet
    data/processed/tencent_profile_candidates.parquet
    profiles/profile_comparison.png
    manifest.json (entries and `profiles` upserted)
    PREPROCESSING.md (B5 section rewritten)
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from contracts import contracts
    from data_pipeline.tencent import extract_load as el
except ImportError:  # pragma: no cover - script mode, repo root not on sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from contracts import contracts
    from data_pipeline.tencent import extract_load as el

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
IN_CLEAN = PROCESSED / "tencent_load_clean.parquet"
IN_SUMMARY = PROCESSED / "tencent_disk_summary.parquet"
IN_RUNS = PROCESSED / "tencent_disk_runs.parquet"
IN_INDEX = PROCESSED / "tencent_disk_index.parquet"
OUT_DISKS = PROCESSED / "tencent_profile_disks.parquet"
OUT_CANDIDATES = PROCESSED / "tencent_profile_candidates.parquet"
PROFILES_DIR = ROOT / "profiles"
MANIFEST = ROOT / "manifest.json"
PREPROCESSING = ROOT / "PREPROCESSING.md"

PROFILE_NAMES = ("idle", "diurnal", "bursty", "peak")
STEP_S = el.STEP_S
DAY_ROWS = el.DAY_ROWS
VALUE_COLUMNS = el.VALUE_COLUMNS
READ_BW, WRITE_BW = VALUE_COLUMNS.index("read_bw_kbps"), VALUE_COLUMNS.index("write_bw_kbps")
KBPS_PER_MBPS = 1000.0
SECONDS_PER_DAY = 86400
MIN_DAYS = 14
LONG_NAME = "diurnal_long"
LONG_WINDOW_DAYS = 14
LONG_WINDOW_ROWS = LONG_WINDOW_DAYS * DAY_ROWS
LONG_TRAIN_DAYS = 10
PEAK_TARGET_MBPS = 160.0    # 0.8 x BW_TOTAL_MBPS, contract section 10 decision 5
DIURNAL_TARGET_MBPS = 100.0
BURSTY_TARGET_MBPS = 120.0   # 0.6 x BW_TOTAL_MBPS
IDLE_TARGET_MBPS = 20.0
PEAK_START_N = 64
CEILING_TOLERANCE = 0.005    # p95 may exceed a "not_above" target by this fraction
MIN_BYTES_PER_IO = 512       # one disk sector
SOURCE_NAME = "Tencent Cloud Volume Workload Traces"
BYTES_PER_KB = 1000.0        # decimal, as the contract's kbps_to_mbps
CEILING_EPS = 1e-9           # relative slack so a p95 exactly on the ceiling passes in float64
DISCLOSURE = "selected extremes, not a representative sample"
BURSTY_DISCLOSURE = (
    DISCLOSURE + "; volumes are ranked by window-local p99/median and the same statistic is "
    "then reported for the aggregate (selection on the outcome), so the reported value is "
    "high partly by construction"
)
# p99/median range of the synthetic generator's bursty profile, kept for comparison only.
SYNTHETIC_BURSTY_RANGE = (8.5, 9.8)

PROFILE_SPECS = {
    "peak": {"key": "mean_total_bw_kbps", "ascending": False, "target_mbps": PEAK_TARGET_MBPS,
             "mode": "not_above", "start_n": PEAK_START_N, "fixed_n": None,
             "rule": "highest mean total bandwidth", "label": "highest-mean-bandwidth",
             "disclosure": DISCLOSURE},
    "diurnal": {"key": "daily_strength", "ascending": False, "target_mbps": DIURNAL_TARGET_MBPS,
                "mode": "not_above", "start_n": None, "fixed_n": None,
                "rule": "highest daily_strength (variance share of the time-of-day mean profile)",
                "label": "strongest-daily-cycle", "disclosure": DISCLOSURE},
    "bursty": {"key": "burstiness", "ascending": False, "target_mbps": BURSTY_TARGET_MBPS,
               "mode": "not_above", "start_n": None, "fixed_n": None,
               "rule": "highest window-local p99/median of total bandwidth",
               "label": "highest-window-p99/median", "disclosure": BURSTY_DISCLOSURE},
    "idle": {"key": "mean_total_bw_kbps", "ascending": True, "target_mbps": IDLE_TARGET_MBPS,
             "mode": "not_above", "start_n": None, "fixed_n": None,
             "rule": "lowest mean total bandwidth", "label": "lowest-mean-bandwidth",
             "disclosure": DISCLOSURE},
}


def sha256_of(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def eligible_disks(runs: pd.DataFrame, window_start: int, rows: int = DAY_ROWS) -> list[str]:
    """Disks with one gap-free run covering all ``rows`` samples from window_start."""
    window_end = window_start + (rows - 1) * STEP_S
    run_end = runs["run_start_ts"] + (runs["run_rows"] - 1) * STEP_S
    covering = runs[(runs["run_start_ts"] <= window_start) & (run_end >= window_end)]
    return sorted(covering["disk_uid"].unique())


def row_group_map(path: Path) -> tuple[pq.ParquetFile, dict[str, int]]:
    """Map each disk_uid to its row group, from row-group statistics."""
    pf = pq.ParquetFile(path)
    col = pf.schema_arrow.get_field_index("disk_uid")
    groups: dict[str, int] = {}
    for i in range(pf.num_row_groups):
        stats = pf.metadata.row_group(i).column(col).statistics
        if stats is not None and stats.has_min_max and stats.min == stats.max:
            groups[stats.min] = i
        else:
            for uid in pf.read_row_group(i, columns=["disk_uid"]).column(0).unique().to_pylist():
                groups[uid] = i
    return pf, groups


def read_disk(pf: pq.ParquetFile, group: int, uid: str) -> pd.DataFrame:
    df = pf.read_row_group(group).to_pandas()
    return df[df["disk_uid"] == uid].reset_index(drop=True)


def daily_stats(ts: np.ndarray, total_bw: np.ndarray) -> dict:
    """Daily-cycle strength of a trace.

    daily_strength is the share of the variance of total bandwidth explained
    by the mean per time-of-day bin (288 bins). daily_amplitude_kbps is the
    peak-to-trough range of that mean profile. Both are NaN-free except
    daily_strength, which is NaN for a constant trace.
    """
    bins = ((ts % SECONDS_PER_DAY) // STEP_S).astype("int64")
    counts = np.bincount(bins, minlength=DAY_ROWS)
    sums = np.bincount(bins, weights=total_bw, minlength=DAY_ROWS)
    present = counts > 0
    means = sums[present] / counts[present]
    grand = float(total_bw.mean())
    ss_total = float(((total_bw - grand) ** 2).sum())
    ss_between = float((counts[present] * (means - grand) ** 2).sum())
    return {
        "days": len(ts) / DAY_ROWS,
        "daily_strength": ss_between / ss_total if ss_total > 0 else float("nan"),
        "daily_amplitude_kbps": float(means.max() - means.min()),
    }


def collect(pf: pq.ParquetFile, groups: dict[str, int], uids: list[str],
            window_start: int, window_rows: int = DAY_ROWS) -> tuple[np.ndarray, pd.DataFrame]:
    """Window samples (n, window_rows, 5), daily-cycle stats, the window-local p99/median of
    total bandwidth (NaN when the window median is 0) and the median read and write
    bytes per I/O of the whole clean trace (NaN when a direction has no IOPS > 0 row)
    for each candidate disk."""
    grid = window_start + np.arange(window_rows, dtype="int64") * STEP_S
    windows = np.zeros((len(uids), window_rows, len(VALUE_COLUMNS)), dtype="int64")
    rows = []
    for k, uid in enumerate(uids):
        d = read_disk(pf, groups[uid], uid)
        ts = d["timestamp"].to_numpy()
        lo = int(np.searchsorted(ts, window_start))
        seg = d.iloc[lo:lo + window_rows]
        if not np.array_equal(seg["timestamp"].to_numpy(), grid):
            raise ValueError(f"{uid}: no complete window at {window_start}")
        windows[k] = seg[VALUE_COLUMNS].to_numpy()
        total = (d["read_bw_kbps"] + d["write_bw_kbps"]).to_numpy(dtype="float64")
        in_window = windows[k, :, READ_BW] + windows[k, :, WRITE_BW]
        rows.append({
            "disk_uid": uid, **daily_stats(ts, total),
            "window_p99_over_median": total_bw_stats(in_window)["p99_over_median"],
            "median_read_bytes_per_io": median_bytes_per_io(
                d["read_bw_kbps"].to_numpy(dtype="float64"), d["read_iops"].to_numpy(dtype="float64")),
            "median_write_bytes_per_io": median_bytes_per_io(
                d["write_bw_kbps"].to_numpy(dtype="float64"), d["write_iops"].to_numpy(dtype="float64")),
        })
    return windows, pd.DataFrame(rows, columns=[
        "disk_uid", "days", "daily_strength", "daily_amplitude_kbps", "window_p99_over_median",
        "median_read_bytes_per_io", "median_write_bytes_per_io"])


def candidate_frame(summary: pd.DataFrame, periodicity: pd.DataFrame) -> pd.DataFrame:
    """Whole-trace summary columns, daily-cycle stats, window-local burstiness and the
    eligibility flag per candidate.

    `implausible_io` is True when the median read or write bytes per I/O is below
    MIN_BYTES_PER_IO; an undefined median (NaN) never sets it.
    """
    cols = ["disk_uid", "mean_total_bw_kbps", "zero_row_fraction"]
    cand = periodicity.merge(summary[cols].astype({c: "float64" for c in cols[1:]}),
                             on="disk_uid", how="left")
    cand["burstiness"] = cand["window_p99_over_median"]
    cand["implausible_io"] = (cand["median_read_bytes_per_io"].lt(MIN_BYTES_PER_IO)
                              | cand["median_write_bytes_per_io"].lt(MIN_BYTES_PER_IO))
    cand.loc[cand["days"] < MIN_DAYS, "daily_strength"] = np.nan
    return cand


def rank_uids(cand: pd.DataFrame, key: str, ascending: bool) -> list[str]:
    ranked = cand.dropna(subset=[key]).sort_values([key, "disk_uid"], ascending=[ascending, True],
                                                    kind="stable")
    return ranked["disk_uid"].tolist()


def tune_n(p95_mbps_by_n: np.ndarray, target_mbps: float, mode: str = "closest") -> int:
    """N (1-based) from the aggregate p95 curve.

    "closest": p95 nearest the target, smaller N on ties. "not_above": largest N with
    p95 at or below target x (1 + CEILING_TOLERANCE); ValueError when no N qualifies.
    """
    if mode == "not_above":
        ceiling = target_mbps * (1.0 + CEILING_TOLERANCE) * (1.0 + CEILING_EPS)
        within = np.flatnonzero(p95_mbps_by_n <= ceiling)
        if not len(within):
            raise ValueError(
                f"no N satisfies the ceiling: target {target_mbps:g} MB/s, ceiling "
                f"{ceiling:g} MB/s, p95 at N=1 {float(p95_mbps_by_n[0]):g} MB/s")
        return int(within[-1]) + 1
    return int(np.argmin(np.abs(p95_mbps_by_n - target_mbps))) + 1


def total_bw_stats(total_kbps: np.ndarray) -> dict:
    median = float(np.median(total_kbps))
    p99 = float(np.percentile(total_kbps, 99))
    return {
        "p95_mbps": float(np.percentile(total_kbps, 95)) / KBPS_PER_MBPS,
        "mean_mbps": float(total_kbps.mean()) / KBPS_PER_MBPS,
        "max_mbps": float(total_kbps.max()) / KBPS_PER_MBPS,
        "p99_over_median": p99 / median if median > 0 else float("nan"),
    }


def build_profile(name: str, spec: dict, cand: pd.DataFrame, windows: np.ndarray,
                  position: dict[str, int], window_start: int, rows: int = DAY_ROWS) -> dict:
    order = rank_uids(cand[~cand["implausible_io"]], spec["key"], spec["ascending"])
    if not order:
        raise ValueError(f"{name}: no ranked candidates")
    idx = [position[u] for u in order]
    totals = windows[idx][:, :, READ_BW] + windows[idx][:, :, WRITE_BW]
    curve = np.percentile(np.cumsum(totals, axis=0), 95, axis=1) / KBPS_PER_MBPS

    if spec["fixed_n"] is not None:
        if len(order) < spec["fixed_n"]:
            raise ValueError(f"{name}: {len(order)} candidates, fixed N is {spec['fixed_n']}")
        n, tuned = spec["fixed_n"], False
    else:
        n, tuned = tune_n(curve, spec["target_mbps"], spec["mode"]), True
    start_n = spec["start_n"]

    agg = pd.DataFrame(windows[idx[:n]].sum(axis=0), columns=VALUE_COLUMNS)
    agg.insert(0, "timestamp", window_start + np.arange(rows, dtype="int64") * STEP_S)
    replay = contracts.to_replay_frame(agg)
    contracts.validate_replay_frame(replay)
    total = (replay["read_bw_kbps"] + replay["write_bw_kbps"]).to_numpy(dtype="float64")
    key_values = cand.set_index("disk_uid").loc[order[:n], spec["key"]]
    return {
        "name": name, "n": n, "tuned": tuned, "frame": replay, "uids": order[:n],
        "key_values": key_values.to_numpy(), "target_mbps": spec["target_mbps"],
        "start_n": start_n,
        "p95_at_start_n": float(curve[start_n - 1]) if start_n and start_n <= len(curve) else None,
        "candidates": len(order), **total_bw_stats(total),
    }


def split_row(rows: int) -> int:
    """Index of the first held-out row: LONG_TRAIN_DAYS of LONG_WINDOW_DAYS of ``rows``, rounded
    down, kept strictly inside [1, rows - 1]."""
    if rows < 2:
        raise ValueError(f"a split needs at least 2 rows, got {rows}")
    return min(max(rows * LONG_TRAIN_DAYS // LONG_WINDOW_DAYS, 1), rows - 1)


def build_long_window(pf: pq.ParquetFile, groups: dict[str, int], runs: pd.DataFrame,
                      summary: pd.DataFrame, diurnal: dict) -> dict:
    """Diurnal aggregate over the 14-day window shared by the most volumes.

    The window start is the timestamp covered by a gap-free run of LONG_WINDOW_ROWS
    samples in the largest number of volumes (`extract_load.shared_window_stats`).
    Volumes are ranked, filtered and counted with the diurnal rule of PROFILE_SPECS
    through `collect` and `build_profile`; the sum of the N selected volumes is the
    replay frame. ``diurnal`` is the 24 h diurnal profile, compared by volume overlap.
    """
    shared = el.shared_window_stats(runs, LONG_WINDOW_ROWS)
    start = shared["largest_group_start_ts"]
    if start is None:
        raise ValueError(f"no gap-free window of {LONG_WINDOW_ROWS} samples shared by any volume")
    uids = eligible_disks(runs, start, LONG_WINDOW_ROWS)
    windows, periodicity = collect(pf, groups, uids, start, LONG_WINDOW_ROWS)
    cand = candidate_frame(summary, periodicity)
    p = build_profile("diurnal", PROFILE_SPECS["diurnal"], cand, windows,
                      {u: i for i, u in enumerate(uids)}, start, LONG_WINDOW_ROWS)
    del windows
    rows = len(p["frame"])
    split = split_row(rows)
    return {
        "name": LONG_NAME, "uids": p["uids"], "n": p["n"], "frame": p["frame"], "start_ts": start,
        "end_ts": start + (rows - 1) * STEP_S, "rows": rows, "requested_rows": LONG_WINDOW_ROWS,
        "volumes_with_window": len(uids), "ranked_candidates": p["candidates"],
        "p95_mbps": p["p95_mbps"], "mean_mbps": p["mean_mbps"], "max_mbps": p["max_mbps"],
        "target_mbps": p["target_mbps"], "overlap_with_diurnal_24h": len(set(p["uids"]) & set(diurnal["uids"])),
        "diurnal_24h_n": diurnal["n"], "diurnal_24h_mean_mbps": diurnal["mean_mbps"],
        "split_row": split, "split_t_seconds": split * STEP_S, "split_ts": start + split * STEP_S,
    }


def utc(ts: int) -> str:
    return f"{pd.Timestamp(ts, unit='s', tz='UTC'):%Y-%m-%dT%H:%M:%SZ}"


def long_sidecar(w: dict) -> dict:
    """Provenance record written next to the long-window CSV."""
    return {
        "profile": w["name"],
        "synthetic": False,
        "source": SOURCE_NAME,
        "n_disks": w["n"],
        "disk_uids": w["uids"],
        "window_start_ts": w["start_ts"],
        "window_start_utc": utc(w["start_ts"]),
        "window_end_ts": w["end_ts"],
        "window_end_utc": utc(w["end_ts"]),
        "rows": w["rows"],
        "requested_rows": w["requested_rows"],
        "volumes_with_window": w["volumes_with_window"],
        "ranked_candidates": w["ranked_candidates"],
        "target_p95_mbps": w["target_mbps"],
        "p95_mbps": w["p95_mbps"],
        "mean_mbps": w["mean_mbps"],
        "max_mbps": w["max_mbps"],
        "scaling_factor": 1.0,
        "label": f"aggregate of {w['n']} {PROFILE_SPECS['diurnal']['label']} volumes over a shared "
                 f"{LONG_WINDOW_DAYS}-day window ({DISCLOSURE})",
        "disclosure": DISCLOSURE,
        "selection_rule": PROFILE_SPECS["diurnal"]["rule"],
        "suggested_split": {
            "train_days_nominal": LONG_TRAIN_DAYS,
            "holdout_days_nominal": LONG_WINDOW_DAYS - LONG_TRAIN_DAYS,
            "train_rows": w["split_row"],
            "holdout_rows": w["rows"] - w["split_row"],
            "first_holdout_row": w["split_row"],
            "first_holdout_t_seconds": w["split_t_seconds"],
            "first_holdout_ts": w["split_ts"],
            "first_holdout_utc": utc(w["split_ts"]),
        },
    }


def write_long_outputs(w: dict) -> list[dict]:
    """Write the long-window parquet, CSV and sidecar; return their manifest entries."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    entries = []
    parquet = PROCESSED / f"tencent_profile_{w['name']}.parquet"
    tmp = parquet.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pandas(w["frame"], preserve_index=False), tmp)
    tmp.replace(parquet)
    entries.append({"file": parquet.relative_to(ROOT).as_posix(), "rows": len(w["frame"]),
                    "sha256": sha256_of(parquet)})
    csv = PROFILES_DIR / f"profile_{w['name']}.csv"
    w["frame"].to_csv(csv, index=False, lineterminator="\n")
    entries.append({"file": csv.relative_to(ROOT).as_posix(), "rows": len(w["frame"]), "sha256": sha256_of(csv)})
    sidecar = PROFILES_DIR / f"profile_{w['name']}.meta.json"
    write_json(sidecar, long_sidecar(w))
    entries.append({"file": sidecar.relative_to(ROOT).as_posix(), "rows": None, "sha256": sha256_of(sidecar)})
    return entries


def long_report_lines(w: dict) -> list[str]:
    return [
        f"long_window={w['name']} n_disks={w['n']} start_ts={w['start_ts']} "
        f"start_utc={utc(w['start_ts'])} end_ts={w['end_ts']} end_utc={utc(w['end_ts'])} rows={w['rows']} "
        f"requested_rows={w['requested_rows']}",
        f"long_window={w['name']} volumes_with_window={w['volumes_with_window']} "
        f"ranked_candidates={w['ranked_candidates']} scaling_factor=1 "
        f"target_p95_mbps={fmt(w['target_mbps'])} p95_mbps={fmt(w['p95_mbps'])} "
        f"mean_mbps={fmt(w['mean_mbps'])} max_mbps={fmt(w['max_mbps'])}",
        f"long_window={w['name']} diurnal_24h_n_disks={w['diurnal_24h_n']} "
        f"diurnal_24h_mean_mbps={fmt(w['diurnal_24h_mean_mbps'])} "
        f"overlap_with_diurnal_24h_disks={w['overlap_with_diurnal_24h']}",
        f"long_window={w['name']} suggested_split first_holdout_row={w['split_row']} "
        f"first_holdout_t_seconds={w['split_t_seconds']} first_holdout_ts={w['split_ts']} "
        f"first_holdout_utc={utc(w['split_ts'])} train_rows={w['split_row']} "
        f"holdout_rows={w['rows'] - w['split_row']} nominal_train_days={LONG_TRAIN_DAYS} "
        f"nominal_holdout_days={LONG_WINDOW_DAYS - LONG_TRAIN_DAYS}",
    ]


def latency_ladder(frame: pd.DataFrame, name: str = "profile") -> dict:
    """Share of intervals that violate latency under each rebuild allocation.

    For each action of ACTIONS_MBPS and each interval: the foreground load is the
    replay bandwidth in MB/s (`sim.scenario.trace_from_frame`); the allocation the
    group can apply is min(action, max(0, rho_max x BW_TOTAL_MBPS - fg)); the
    utilisation is min(rho_max, (fg + applied) / BW_TOTAL_MBPS); the interval
    violates when the contract's `latency_violation` says the simulator's mean
    latency exceeds its limit at the configured target. `safe_actions` counts the
    allocations whose violating share is at most LATENCY_VIOLATION_FRACTION_MAX.
    """
    from sim import scenario
    from sim.config import latency_mean_ms, v0_config

    fg = scenario.trace_from_frame(frame, name, False).fg_bw_mbps
    cfg = v0_config()
    headroom = np.maximum(0.0, cfg.rho_max * contracts.BW_TOTAL_MBPS - fg)
    fractions = {}
    for action in contracts.ACTIONS_MBPS:
        applied = np.minimum(float(action), headroom)
        rho = np.minimum(cfg.rho_max, (fg + applied) / contracts.BW_TOTAL_MBPS)
        violations = [contracts.latency_violation(latency_mean_ms(float(r), cfg), cfg.latency_target_ms)
                      for r in rho]
        fractions[int(action)] = float(np.mean(violations))
    safe = sum(1 for f in fractions.values() if f <= contracts.LATENCY_VIOLATION_FRACTION_MAX)
    return {"violation_fraction_by_action_mbps": fractions, "safe_actions": safe}


def median_bytes_per_io(bw_kbps: np.ndarray, iops: np.ndarray) -> float:
    """Median of bw_kbps x BYTES_PER_KB / iops over the rows with iops > 0 (NaN when none)."""
    active = iops > 0
    if not active.any():
        return float("nan")
    return float(np.median(bw_kbps[active] * BYTES_PER_KB / iops[active]))


def iops_facts(frame: pd.DataFrame) -> dict:
    """IOPS maxima, share of rows above the observation scale, and median bytes per I/O.

    Bytes per I/O is bandwidth in KB/s x BYTES_PER_KB / IOPS over the rows with
    IOPS > 0 (None when no row has IOPS > 0). Recorded only; no selection uses it.
    """
    facts = {}
    for kind in ("read", "write"):
        iops = frame[f"{kind}_iops"].to_numpy(dtype="float64")
        bw_kbps = frame[f"{kind}_bw_kbps"].to_numpy(dtype="float64")
        median = median_bytes_per_io(bw_kbps, iops)
        facts[f"max_{kind}_iops"] = int(iops.max())
        facts[f"fraction_rows_{kind}_iops_above_obs_scale"] = float(
            (iops > contracts.OBS_SCALES[f"fg_{kind}_iops"]).mean())
        facts[f"median_bytes_per_{kind}_io"] = None if median != median else median
    return facts


def profile_label(p: dict) -> str:
    return f"aggregate of {p['n']} {PROFILE_SPECS[p['name']]['label']} volumes ({DISCLOSURE})"


def profile_sidecar(p: dict, window_start: int) -> dict:
    """Provenance record written next to a profile CSV (read by `sim.scenario.load_replay_csv`)."""
    spec = PROFILE_SPECS[p["name"]]
    start = pd.Timestamp(window_start, unit="s", tz="UTC")
    return {
        "profile": p["name"],
        "synthetic": False,
        "source": SOURCE_NAME,
        "n_disks": p["n"],
        "window_start_ts": window_start,
        "window_start_utc": f"{start:%Y-%m-%dT%H:%M:%SZ}",
        "label": profile_label(p),
        "disclosure": spec["disclosure"],
        "selection_rule": spec["rule"],
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def composition(index: pd.DataFrame, uids: list[str]) -> dict:
    sel = index[index["disk_uid"].isin(uids)]
    return {c: {int(k): int(v) for k, v in sel[c].value_counts().sort_index().items()}
            for c in ("user_type", "disk_type")}


def profile_table(profiles: list[dict]) -> pa.Table:
    rows = [{"profile": p["name"], "rank": i + 1, "disk_uid": uid, "key_value": float(v)}
            for p in profiles for i, (uid, v) in enumerate(zip(p["uids"], p["key_values"]))]
    return pa.Table.from_pandas(pd.DataFrame(rows, columns=["profile", "rank", "disk_uid",
                                                            "key_value"]), preserve_index=False)


PLOT_SURFACE, PLOT_INK, PLOT_INK_2, PLOT_GRID, PLOT_SERIES = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1", "#2a78d6")


def plot_profiles(profiles: list[dict], window_start: int, path: Path) -> None:
    """One PNG with a panel per profile; all panels share the bandwidth and time axes.

    Each panel shows total bandwidth (read + write, MB/s) over the 288 window
    samples, with a dashed line at the profile's p95.
    """
    from matplotlib.figure import Figure

    totals = [(p["frame"]["read_bw_kbps"] + p["frame"]["write_bw_kbps"]).to_numpy(dtype="float64")
              / KBPS_PER_MBPS for p in profiles]
    fig = Figure(figsize=(9.0, 10.5), facecolor=PLOT_SURFACE, layout="constrained")
    fig.get_layout_engine().set(rect=(0.0, 0.03, 1.0, 0.97))   # bottom margin for the footnote
    axes = fig.subplots(len(profiles), 1, sharex=True, sharey=True)
    for ax, p, total in zip(axes, profiles, totals):
        hours = p["frame"]["t_seconds"].to_numpy() / 3600.0
        ax.set_facecolor(PLOT_SURFACE)
        ax.plot(hours, total, color=PLOT_SERIES, linewidth=1.6)
        ax.axhline(p["p95_mbps"], color=PLOT_INK_2, linewidth=0.9, linestyle=(0, (4, 3)))
        ax.annotate(f"p95 {p['p95_mbps']:.1f} MB/s", xy=(24, p["p95_mbps"]),
                    xytext=(-4, 4), textcoords="offset points", ha="right", va="bottom",
                    fontsize=8, color=PLOT_INK_2)
        ax.set_title(f"{p['name']}: aggregate of {p['n']} {PROFILE_SPECS[p['name']]['label']} volumes",
                     loc="left", fontsize=10, color=PLOT_INK)
        ax.set_title(f"p99/median {p['p99_over_median']:.2f}", loc="right", fontsize=9,
                     color=PLOT_INK_2)
        ax.grid(axis="y", color=PLOT_GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(PLOT_GRID)
        ax.tick_params(colors=PLOT_INK_2, labelsize=8)
    axes[0].set_xlim(0, 24)
    axes[0].set_xticks(range(0, 25, 4))
    axes[0].set_ylim(0, max(float(t.max()) for t in totals) * 1.12)
    start = pd.Timestamp(window_start, unit="s", tz="UTC")
    axes[-1].set_xlabel(f"Hours from window start ({start:%Y-%m-%d %H:%M} UTC)", fontsize=9,
                        color=PLOT_INK_2)
    fig.supylabel("Total bandwidth, read + write (MB/s)", fontsize=9, color=PLOT_INK_2)
    fig.text(0.01, 0.005, f"Each panel: sum of N real volumes; {DISCLOSURE}. "
             "All panels share one y scale.", fontsize=8, color=PLOT_INK_2)
    fig.savefig(path, dpi=150, facecolor=PLOT_SURFACE, metadata={"Software": None})


def write_outputs(profiles: list[dict], cand: pd.DataFrame, window_start: int) -> list[dict]:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    entries = []

    def put(path: Path, table: pa.Table) -> None:
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp)
        tmp.replace(path)
        entries.append({"file": path.relative_to(ROOT).as_posix(), "rows": table.num_rows,
                        "sha256": sha256_of(path)})

    for p in profiles:
        put(PROCESSED / f"tencent_profile_{p['name']}.parquet",
            pa.Table.from_pandas(p["frame"], preserve_index=False))
        csv = PROFILES_DIR / f"profile_{p['name']}.csv"
        p["frame"].to_csv(csv, index=False, lineterminator="\n")
        entries.append({"file": csv.relative_to(ROOT).as_posix(), "rows": len(p["frame"]),
                        "sha256": sha256_of(csv)})
        sidecar = PROFILES_DIR / f"profile_{p['name']}.meta.json"
        write_json(sidecar, profile_sidecar(p, window_start))
        entries.append({"file": sidecar.relative_to(ROOT).as_posix(), "rows": None,
                        "sha256": sha256_of(sidecar)})
    png = PROFILES_DIR / "profile_comparison.png"
    plot_profiles(profiles, window_start, png)
    entries.append({"file": png.relative_to(ROOT).as_posix(), "rows": None,
                    "sha256": sha256_of(png)})
    put(OUT_DISKS, profile_table(profiles))
    put(OUT_CANDIDATES, pa.Table.from_pandas(
        cand[["disk_uid", "days", "daily_strength", "daily_amplitude_kbps",
              "window_p99_over_median", "median_read_bytes_per_io", "median_write_bytes_per_io",
              "implausible_io"]], preserve_index=False))
    return entries


def overlaps(profiles: list[dict]) -> list[str]:
    out = []
    for i, a in enumerate(profiles):
        for b in profiles[i + 1:]:
            out.append(f"overlap_{a['name']}_{b['name']}={len(set(a['uids']) & set(b['uids']))}")
    return out


def fmt(x) -> str:
    return "nan" if x is None or x != x else f"{x:.4g}"


def report_lines(r: dict) -> list[str]:
    lines = [
        f"window_start_ts={r['window_start']} window_start_utc="
        f"{pd.Timestamp(r['window_start'], unit='s', tz='UTC'):%Y-%m-%d %H:%M}",
        f"candidate_disks={r['candidates']} diurnal_candidates={r['diurnal_candidates']}",
        f"min_bytes_per_io={MIN_BYTES_PER_IO} candidates_excluded_by_rule={r['excluded']} "
        f"candidates_after_rule={r['candidates'] - r['excluded']}",
    ]
    for p in r["profiles"]:
        spec = PROFILE_SPECS[p["name"]]
        target = "none" if p["target_mbps"] is None else fmt(p["target_mbps"])
        lines.append(
            f"profile={p['name']} n_disks={p['n']} label=aggregate_of_{p['n']}_{spec['label']}_volumes "
            f"rule=\"{spec['rule']}\" tuned={p['tuned']} target_p95_mbps={target} "
            f"p95_mbps={fmt(p['p95_mbps'])} p99_over_median={fmt(p['p99_over_median'])} "
            f"mean_mbps={fmt(p['mean_mbps'])} max_mbps={fmt(p['max_mbps'])}")
        lines.append(f"profile={p['name']} ranked_candidates={p['candidates']}")
        if p["start_n"] is not None:
            lines.append(f"profile={p['name']} start_n={p['start_n']} "
                         f"p95_mbps_at_start_n={fmt(p['p95_at_start_n'])}")
        keys = p["key_values"]
        lines.append(f"profile={p['name']} {spec['key']}_range=[{fmt(keys.min())},{fmt(keys.max())}] "
                     f"zero_bandwidth_disks={p['zero_bw_disks']}")
        comp = p["composition"]
        lines.append(f"profile={p['name']} user_type_counts={comp['user_type']} "
                     f"disk_type_counts={comp['disk_type']}")
        fractions = {a: round(f, 4) for a, f in p["ladder"]["violation_fraction_by_action_mbps"].items()}
        lines.append(f"profile={p['name']} safe_actions={p['ladder']['safe_actions']} "
                     f"violation_fraction_by_action_mbps={fractions}")
        f = p["iops"]
        lines.append(
            f"profile={p['name']} max_read_iops={f['max_read_iops']} max_write_iops={f['max_write_iops']} "
            f"fraction_rows_read_iops_above_obs_scale={fmt(f['fraction_rows_read_iops_above_obs_scale'])} "
            f"fraction_rows_write_iops_above_obs_scale={fmt(f['fraction_rows_write_iops_above_obs_scale'])} "
            f"median_bytes_per_read_io={fmt(f['median_bytes_per_read_io'])} "
            f"median_bytes_per_write_io={fmt(f['median_bytes_per_write_io'])}")
    bursty = next(p for p in r["profiles"] if p["name"] == "bursty")
    lo, hi = SYNTHETIC_BURSTY_RANGE
    ratio = bursty["p99_over_median"]
    below = "undefined" if ratio != ratio else bool(ratio < lo)
    lines.append(f"bursty_p99_over_median={fmt(ratio)} synthetic_reference_range={lo}-{hi} "
                 f"below_reference={below} ranking_statistic=window_p99_over_median "
                 f"selected_on_reported_statistic=True")
    lines += overlaps(r["profiles"])
    lines += long_report_lines(r["long_window"])
    for e in r["entries"]:
        rows = "" if e["rows"] is None else f" rows={e['rows']}"
        lines.append(f"output={e['file']}{rows} sha256={e['sha256']}")
    return lines


def update_preprocessing(lines: list[str]) -> None:
    body = "\n".join(f"- `{ln}`" for ln in lines)
    section = (
        "## B5 — Four intensity profiles\n\n"
        "- Each profile is the sum of N real volumes over one shared 24 h window "
        "(288 samples, 300 s), in the six-column replay schema, bandwidth in KB/s. "
        "Each is an aggregate of N volumes chosen by character: " + DISCLOSURE
        + " of volumes or of a storage group\n"
        "- Bursty: " + BURSTY_DISCLOSURE + "\n"
        "- Window: the start timestamp shared by the most volumes with a gap-free 24 h run; "
        "only those volumes are candidates\n"
        f"- Eligibility rule: a candidate is excluded from every profile when its median bytes per "
        f"I/O is below {MIN_BYTES_PER_IO} (one disk sector) for reads or for writes. The median is "
        "bandwidth in KB/s x 1000 / IOPS over the direction's rows with IOPS > 0, across the "
        "volume's whole clean trace; a direction with no such row has no median and does not "
        "exclude. A median below one 512-byte sector per I/O is physically implausible and "
        "indicates an unreliable IOPS counter. The rule is applied to all candidates before any "
        "ranking and does not depend on any profile target\n"
        "- Selection rules: peak = highest mean total bandwidth; diurnal = highest "
        "`daily_strength` (share of total-bandwidth variance explained by the 288-bin "
        f"time-of-day mean, at least {MIN_DAYS} days of clean rows); bursty = highest "
        "window-local p99/median of total bandwidth (the 288 window samples of the volume, "
        "median > 0); idle = lowest mean total bandwidth. Ties are broken by `disk_uid`\n"
        "- N: every N from 1 to the number of ranked candidates is evaluated. All four profiles "
        "use the same one-sided rule: the largest N whose aggregate p95 does not exceed target x "
        f"{1 + CEILING_TOLERANCE:g} (targets: peak {PEAK_TARGET_MBPS:g}, diurnal "
        f"{DIURNAL_TARGET_MBPS:g}, bursty {BURSTY_TARGET_MBPS:g}, idle {IDLE_TARGET_MBPS:g} MB/s; "
        f"the {CEILING_TOLERANCE:.1%} tolerance keeps a p95 a hair above the literal target from "
        "being counted as an overshoot)\n"
        "- \"Closest to target\" was dropped for all four profiles in favour of \"largest N at "
        "or below target x 1.005\": a symmetric distance is the wrong fit for a one-sided "
        "feasibility ceiling. Overshoot uses up the headroom the simulator's latency limit "
        "(`sim/config.py`) leaves for rebuild traffic, while undershoot can add a safe action, "
        "so the ladder is checked separately by `latency_ladder`. No disk is skipped or swapped "
        "to move p95 or p99/median toward a target\n"
        "- Latency ladder: for each action of `ACTIONS_MBPS` and each of the 288 intervals, the "
        "foreground load is the replay bandwidth in MB/s (`sim.scenario.trace_from_frame`), the "
        "applied allocation is min(action, max(0, `rho_max` x `BW_TOTAL_MBPS` - load)), the "
        "utilisation is min(`rho_max`, (load + applied) / `BW_TOTAL_MBPS`), and the interval "
        "violates when the contract's `latency_violation` flags the simulator's mean latency "
        "(`sim.config.latency_mean_ms`, default `SimConfig`). `violation_fraction_by_action_mbps` "
        "is the violating share per action; `safe_actions` counts the actions whose share is at "
        "most `LATENCY_VIOLATION_FRACTION_MAX`\n"
        "- IOPS facts (recorded only; no selection depends on them): per profile, the maximum "
        "read and write IOPS, the share of rows above the observation scales `fg_read_iops` and "
        "`fg_write_iops`, and the median bytes per I/O (bandwidth in KB/s x 1000 / IOPS over rows "
        "with IOPS > 0)\n"
        "- Sidecars: `profiles/profile_<name>.meta.json` (`profile`, `synthetic` = false, `source`, "
        "`n_disks`, window start, `label`, `disclosure`, `selection_rule`), the provenance record "
        "`sim.scenario.load_replay_csv` reads next to each CSV\n"
        "- p99/median of a profile is computed over its 288 window samples of total bandwidth "
        f"(read + write); the synthetic generator's bursty profile gives "
        f"{SYNTHETIC_BURSTY_RANGE[0]}-{SYNTHETIC_BURSTY_RANGE[1]} on the same definition\n"
        "- Files: `data/processed/tencent_profile_<name>.parquet` and "
        "`profiles/profile_<name>.csv` (identical rows); selected disks in "
        "`tencent_profile_disks.parquet`; candidate statistics (daily-cycle strength, "
        "window-local p99/median) in `tencent_profile_candidates.parquet`\n"
        f"- Long window `{LONG_NAME}` (additive; the four profiles are unchanged): the diurnal "
        f"rule applied to a {LONG_WINDOW_DAYS}-day window ({LONG_WINDOW_ROWS} samples). The window "
        "start is the timestamp covered by a gap-free run of that length in the largest number of "
        "volumes; those volumes are ranked, filtered and counted by the diurnal rule and N is tuned "
        "against the diurnal target p95, as for the 24 h profiles; the sum of the N volumes is the "
        "replay frame (six-column schema, `t_seconds` from 0, bandwidth in KB/s). Like the 24 h "
        "profiles it is an aggregate of selected volumes, a modelling assumption, not a sample of "
        "a typical storage group. No scaling factor is applied (`scaling_factor` = 1). Files: "
        f"`data/processed/tencent_profile_{LONG_NAME}.parquet`, `profiles/profile_{LONG_NAME}.csv` "
        f"(identical rows) and `profiles/profile_{LONG_NAME}.meta.json`, which records the "
        "volume list, window start and end, row counts, aggregate p95 and mean and a suggested "
        f"time-forward split (first {LONG_TRAIN_DAYS} of {LONG_WINDOW_DAYS} days of the rows, "
        "rounded down; metadata only, nothing reads it)\n"
        "- Visual check: `profiles/profile_comparison.png`, one panel per profile on a shared "
        "bandwidth axis and a shared 24 h time axis\n"
        "- Counts and measurements from the last run of `tencent/build_profiles.py`:\n\n"
        f"{body}\n\n"
    )
    text = PREPROCESSING.read_text(encoding="utf-8")
    new, n = re.subn(r"## B5 — Four intensity profiles\n.*?(?=\n## B6 )",
                     section.rstrip("\n") + "\n", text, count=1, flags=re.S)
    if n != 1:
        raise SystemExit("B5 section not found in PREPROCESSING.md")
    PREPROCESSING.write_text(new, encoding="utf-8")


def update_manifest(r: dict) -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    names = {e["file"] for e in r["entries"]}
    data["files"] = [e for e in data.get("files", []) if e.get("file") not in names] + r["entries"]
    data["profiles"] = {
        p["name"]: {
            "n_disks": p["n"],
            "label": profile_label(p),
            "disclosure": PROFILE_SPECS[p["name"]]["disclosure"],
            "selection_rule": PROFILE_SPECS[p["name"]]["rule"],
            "n_selection_mode": PROFILE_SPECS[p["name"]]["mode"],
            "window_start_ts": r["window_start"],
            "tuned": p["tuned"],
            "target_p95_mbps": p["target_mbps"],
            "p95_mbps": p["p95_mbps"],
            "p99_over_median": None if p["p99_over_median"] != p["p99_over_median"]
            else p["p99_over_median"],
            "violation_fraction_by_action_mbps": p["ladder"]["violation_fraction_by_action_mbps"],
            "safe_actions": p["ladder"]["safe_actions"],
            "iops_facts": p["iops"],
            "ranked_candidates": p["candidates"],
        }
        for p in r["profiles"]
    }
    data["profile_eligibility"] = {
        "min_bytes_per_io": MIN_BYTES_PER_IO,
        "candidates": r["candidates"],
        "excluded_implausible_io": r["excluded"],
    }
    MANIFEST.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def build() -> dict:
    runs = pd.read_parquet(IN_RUNS)
    window_start = el.shared_window_stats(runs)["largest_group_start_ts"]
    if window_start is None:
        raise SystemExit("no complete 24 h window shared by any volume")
    uids = eligible_disks(runs, window_start)
    pf, groups = row_group_map(IN_CLEAN)
    windows, periodicity = collect(pf, groups, uids, window_start)
    summary = pd.read_parquet(IN_SUMMARY)
    index = pd.read_parquet(IN_INDEX)
    cand = candidate_frame(summary, periodicity)
    position = {u: i for i, u in enumerate(uids)}

    profiles = []
    for name in PROFILE_NAMES:
        p = build_profile(name, PROFILE_SPECS[name], cand, windows, position, window_start)
        p["composition"] = composition(index, p["uids"])
        p["ladder"] = latency_ladder(p["frame"], name)
        p["iops"] = iops_facts(p["frame"])
        zero = cand.set_index("disk_uid").loc[p["uids"], "zero_row_fraction"]
        p["zero_bw_disks"] = int((zero == 1.0).sum())
        profiles.append(p)
    entries = write_outputs(profiles, cand, window_start)
    diurnal = next(p for p in profiles if p["name"] == "diurnal")
    del windows
    long_window = build_long_window(pf, groups, runs, summary, diurnal)
    entries += write_long_outputs(long_window)
    eligible = cand[~cand["implausible_io"]]
    return {"window_start": window_start, "candidates": len(uids), "long_window": long_window,
            "excluded": int(cand["implausible_io"].sum()),
            "diurnal_candidates": int(eligible["daily_strength"].notna().sum()),
            "profiles": profiles, "entries": entries}


def main(argv: list[str]) -> int:
    r = build()
    lines = report_lines(r)
    print("\n".join(lines))
    update_preprocessing(lines)
    update_manifest(r)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
