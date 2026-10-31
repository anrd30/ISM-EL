"""
B3: Tencent Cloud Virtual Disk load-trace extract.

Reads every headerless per-disk load file, cleans it, and writes
  - one long Parquet with one row group per disk, and
  - one summary row per disk (coverage, gaps, load level).

Per-disk cleaning, in order (nothing is imputed, no gap is filled):
  1. count rows where the timestamp decreases (out of order), then sort by timestamp;
  2. drop exact duplicate rows;
  3. drop every row of a timestamp that still occurs more than once (all
     copies are dropped because the repeats disagree, so no value is chosen);
  4. drop rows with any negative value.

The size of the disagreement at each dropped timestamp, the contiguous runs of
each disk, and cross-disk load and shared-window statistics are reported;
see `conflict_lines` and `aggregation_lines`.

Timestamps stay as source epoch seconds and bandwidth stays in source KB/s.
A gap-free 300 s window of the clean table is rebased to the six-column int64
replay schema (`t_seconds` starting at 0) by `contracts.contracts.to_replay_frame`.

Usage:
    python -I extract_load.py [dataset_root]
    python -I extract_load.py --aggregation-only

<dataset_root> holds `disk_subscript_info` and `disk_load_data/`.

Outputs:
    data/processed/tencent_load_clean.parquet
    data/processed/tencent_disk_summary.parquet
    data/processed/tencent_disk_runs.parquet
    manifest.json (entries upserted)
    PREPROCESSING.md (B3 section rewritten)
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "raw" / "tencent" / "Cloud_Disk_dataset"
PROCESSED = ROOT / "data" / "processed"
OUT_CLEAN = PROCESSED / "tencent_load_clean.parquet"
OUT_SUMMARY = PROCESSED / "tencent_disk_summary.parquet"
OUT_RUNS = PROCESSED / "tencent_disk_runs.parquet"
MANIFEST = ROOT / "manifest.json"
PREPROCESSING = ROOT / "PREPROCESSING.md"

STEP_S = 300
DAY_ROWS = 288  # 24 h at STEP_S
LOAD_COLUMNS = ["read_iops", "write_iops", "read_bw_kbps", "write_bw_kbps"]
ROUNDING_UNITS = 1      # a load difference of at most this many units is rounding-level
SMALL_REL = 0.10        # relative load difference up to this is "small"
SUB_COLUMNS = ["disk_uid", "disk_capacity", "disk_attr", "disk_type",
               "user_type", "vm_cpu", "vm_memory"]
# Source field order (dataset README).
SOURCE_COLUMNS = ["timestamp", "read_iops", "read_bw_kbps",
                  "write_iops", "write_bw_kbps", "disk_usage_mb"]
VALUE_COLUMNS = ["read_iops", "write_iops", "read_bw_kbps", "write_bw_kbps", "disk_usage_mb"]
CLEAN_COLUMNS = ["timestamp"] + VALUE_COLUMNS

CLEAN_SCHEMA = pa.schema(
    [pa.field("disk_uid", pa.string())] + [pa.field(c, pa.int64()) for c in CLEAN_COLUMNS]
)
SUMMARY_FIELDS = [
    "disk_uid", "rows_raw", "rows_clean", "first_ts", "last_ts", "out_of_order_rows",
    "exact_duplicate_rows", "conflict_timestamps", "conflict_rows_dropped",
    "negative_rows_dropped", "off_grid_rows", "breaks", "missing_slots",
    "longest_run_rows", "mean_total_bw_kbps", "median_total_bw_kbps",
    "p95_total_bw_kbps", "p99_total_bw_kbps", "max_total_bw_kbps", "zero_row_fraction",
]
FLOAT_FIELDS = ("mean_total_bw_kbps", "median_total_bw_kbps", "p95_total_bw_kbps",
                "p99_total_bw_kbps", "zero_row_fraction")


def read_headerless(source, columns: list[str], label: str) -> pd.DataFrame:
    """Read a headerless CSV with exactly len(columns) fields in every row.

    The columns are read without names so that a row with more fields than the
    first, or a first row with extra fields, cannot be absorbed into an implicit
    index or shift the columns. Raises SystemExit naming `label` when the field
    count of the first row differs from len(columns), when any later row has
    more fields than the first (parser error), or when any field is missing.
    """
    try:
        df = pd.read_csv(source, header=None)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as err:
        raise SystemExit(f"{label}: {err}") from err
    if df.shape[1] != len(columns):
        raise SystemExit(f"{label}: {df.shape[1]} columns, expected {len(columns)}")
    df.columns = columns
    if df.isna().any().any():
        raise SystemExit(f"{label}: rows with missing fields")
    return df


def parse_trace(data: bytes, name: str) -> pd.DataFrame:
    df = read_headerless(io.BytesIO(data), SOURCE_COLUMNS, name)
    bad = [c for c in SOURCE_COLUMNS if not pd.api.types.is_integer_dtype(df[c])]
    if bad:
        raise SystemExit(f"{name}: non-integer columns {bad}")
    return df[CLEAN_COLUMNS].astype("int64")


def conflict_detail(rep: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per conflicting timestamp: how far apart the repeated rows are.

    `rep` holds all rows whose timestamp occurs more than once. Spread of a
    column is max - min over the copies; relative spread divides by
    max(column max, 1). `load_*` use the IOPS and bandwidth columns only.
    """
    g = rep.groupby("timestamp")
    hi, lo = g[VALUE_COLUMNS].max(), g[VALUE_COLUMNS].min()
    spread = hi - lo
    rel = spread / hi.clip(lower=1)
    ts = hi.index.to_numpy()
    edges = np.concatenate(([-1], np.flatnonzero(np.diff(ts) != STEP_S), [len(ts) - 1]))
    return {
        "load_abs": spread[LOAD_COLUMNS].max(axis=1).to_numpy(),
        "load_rel": rel[LOAD_COLUMNS].max(axis=1).to_numpy(),
        "usage_spread": spread["disk_usage_mb"].to_numpy(),
        "rows_per_ts": g.size().to_numpy(),
        "block_lengths": np.diff(edges),
    }


def clean_trace(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict | None]:
    ts = df["timestamp"].to_numpy()
    stats = {
        "rows_raw": len(df),
        "out_of_order_rows": int((np.diff(ts) < 0).sum()) if len(ts) else 0,
    }
    df = df.sort_values("timestamp", kind="stable")
    before = len(df)
    df = df.drop_duplicates()
    stats["exact_duplicate_rows"] = before - len(df)

    repeated = df["timestamp"].duplicated(keep=False)
    stats["conflict_timestamps"] = int(df.loc[repeated, "timestamp"].nunique())
    stats["conflict_rows_dropped"] = int(repeated.sum())
    conflicts = conflict_detail(df[repeated]) if repeated.any() else None
    df = df[~repeated]

    negative = (df < 0).any(axis=1)
    stats["negative_rows_dropped"] = int(negative.sum())
    df = df[~negative].reset_index(drop=True)
    return df, stats, conflicts


def trace_summary(df: pd.DataFrame) -> dict:
    n = len(df)
    out = {"rows_clean": n, "off_grid_rows": 0, "breaks": 0, "missing_slots": 0,
           "longest_run_rows": 0, "first_ts": None, "last_ts": None,
           "mean_total_bw_kbps": None, "median_total_bw_kbps": None,
           "p95_total_bw_kbps": None, "p99_total_bw_kbps": None,
           "max_total_bw_kbps": None, "zero_row_fraction": None}
    if n == 0:
        return out
    ts = df["timestamp"].to_numpy()
    diffs = np.diff(ts)
    total_bw = (df["read_bw_kbps"] + df["write_bw_kbps"]).to_numpy()
    zero = (df[["read_iops", "write_iops"]].sum(axis=1).to_numpy() == 0) & (total_bw == 0)

    break_idx = np.flatnonzero(diffs != STEP_S)
    longest = int(np.diff(np.concatenate(([-1], break_idx, [n - 1]))).max())
    over = diffs[diffs > STEP_S]
    out.update({
        "first_ts": int(ts[0]), "last_ts": int(ts[-1]),
        "off_grid_rows": int(((ts - ts[0]) % STEP_S != 0).sum()),
        "breaks": int((diffs != STEP_S).sum()),
        "missing_slots": int((over // STEP_S - 1).sum()),
        "longest_run_rows": longest,
        "mean_total_bw_kbps": float(total_bw.mean()),
        "median_total_bw_kbps": float(np.median(total_bw)),
        "p95_total_bw_kbps": float(np.percentile(total_bw, 95)),
        "p99_total_bw_kbps": float(np.percentile(total_bw, 99)),
        "max_total_bw_kbps": int(total_bw.max()),
        "zero_row_fraction": float(zero.mean()),
    })
    return out


def trace_runs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Start timestamp and row count of each gap-free 300 s run of a clean trace."""
    ts = df["timestamp"].to_numpy()
    if len(ts) == 0:
        return np.empty(0, dtype="int64"), np.empty(0, dtype="int64")
    break_idx = np.flatnonzero(np.diff(ts) != STEP_S)
    starts = np.concatenate(([0], break_idx + 1))
    lengths = np.diff(np.concatenate(([-1], break_idx, [len(ts) - 1])))
    return ts[starts].astype("int64"), lengths.astype("int64")


def read_subscription(path: Path | io.BytesIO) -> pd.DataFrame:
    return read_headerless(path, SUB_COLUMNS, "subscription")


def dedupe_subscription(sub: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep the first row per disk_uid; report which uids disagree between rows."""
    attrs = [c for c in SUB_COLUMNS if c != "disk_uid"]
    repeated = sub[sub["disk_uid"].duplicated(keep=False)]
    conflicting = sorted(
        uid for uid, g in repeated.groupby("disk_uid") if len(g[attrs].drop_duplicates()) > 1
    )
    info = {
        "rows": len(sub),
        "unique_uids": int(sub["disk_uid"].nunique()),
        "repeated_uids": int(repeated["disk_uid"].nunique()),
        "conflicting_uids": conflicting,
    }
    return sub.drop_duplicates(subset="disk_uid", keep="first").reset_index(drop=True), info


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_of(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def clean_table(uid: str, df: pd.DataFrame) -> pa.Table:
    cols = {"disk_uid": pa.array([uid] * len(df), type=pa.string())}
    cols.update({c: pa.array(df[c].to_numpy(dtype="int64")) for c in CLEAN_COLUMNS})
    return pa.table(cols, schema=CLEAN_SCHEMA)


def summary_table(rows: list[dict]) -> pa.Table:
    frame = pd.DataFrame(rows, columns=SUMMARY_FIELDS)
    ints = [c for c in SUMMARY_FIELDS if c != "disk_uid" and c not in FLOAT_FIELDS]
    for c in ints:
        frame[c] = frame[c].astype("Int64")
    return pa.Table.from_pandas(frame, preserve_index=False)


def runs_table(uids: list[str], starts: list[np.ndarray], lengths: list[np.ndarray]) -> pa.Table:
    return pa.table({
        "disk_uid": pa.array(np.repeat(uids, [len(a) for a in starts]).tolist(), type=pa.string()),
        "run_start_ts": pa.array(np.concatenate(starts) if starts else [], type=pa.int64()),
        "run_rows": pa.array(np.concatenate(lengths) if lengths else [], type=pa.int64()),
    })


def extract(root: Path) -> dict:
    sub_path = root / "disk_subscript_info"
    load_dir = root / "disk_load_data"
    files = sorted(p for p in load_dir.iterdir() if p.is_file() and not p.name.startswith("."))
    if not files:
        raise SystemExit(f"no load files in {load_dir}")

    sub_raw = sub_path.read_bytes()
    sub, sub_info = dedupe_subscription(read_subscription(io.BytesIO(sub_raw)))

    PROCESSED.mkdir(parents=True, exist_ok=True)
    tmp_clean = OUT_CLEAN.with_suffix(".parquet.tmp")
    rows, digests = [], []
    run_uids, run_starts, run_lengths = [], [], []
    conflict_parts: dict[str, list[np.ndarray]] = {}
    totals = dict.fromkeys(
        ["rows_raw", "rows_clean", "out_of_order_rows", "exact_duplicate_rows",
         "conflict_timestamps", "conflict_rows_dropped", "negative_rows_dropped",
         "off_grid_rows", "missing_slots"], 0)
    disks_with = dict.fromkeys(
        ["out_of_order", "exact_duplicates", "conflicts", "negatives", "off_grid", "breaks"], 0)

    with pq.ParquetWriter(tmp_clean, CLEAN_SCHEMA, compression="zstd") as writer:
        for p in files:
            data = p.read_bytes()
            digests.append(f"{p.name}:{sha256_bytes(data)}")
            clean, stats, conflicts = clean_trace(parse_trace(data, p.name))
            row = {"disk_uid": p.name, **stats, **trace_summary(clean)}
            rows.append(row)
            starts, lengths = trace_runs(clean)
            if len(starts):
                run_uids.append(p.name)
                run_starts.append(starts)
                run_lengths.append(lengths)
            for k, v in (conflicts or {}).items():
                conflict_parts.setdefault(k, []).append(v)
            for k in totals:
                totals[k] += row[k]
            disks_with["out_of_order"] += row["out_of_order_rows"] > 0
            disks_with["exact_duplicates"] += row["exact_duplicate_rows"] > 0
            disks_with["conflicts"] += row["conflict_timestamps"] > 0
            disks_with["negatives"] += row["negative_rows_dropped"] > 0
            disks_with["off_grid"] += row["off_grid_rows"] > 0
            disks_with["breaks"] += row["breaks"] > 0
            if len(clean):
                writer.write_table(clean_table(p.name, clean), row_group_size=len(clean))
    tmp_clean.replace(OUT_CLEAN)

    tmp_sum = OUT_SUMMARY.with_suffix(".parquet.tmp")
    pq.write_table(summary_table(rows), tmp_sum)
    tmp_sum.replace(OUT_SUMMARY)

    runs = runs_table(run_uids, run_starts, run_lengths)
    tmp_runs = OUT_RUNS.with_suffix(".parquet.tmp")
    pq.write_table(runs, tmp_runs)
    tmp_runs.replace(OUT_RUNS)

    uids_load = {p.name for p in files}
    uids_sub = set(sub["disk_uid"])
    return {
        "root": root, "files": len(files), "rows": rows, "totals": totals,
        "disks_with": {k: int(v) for k, v in disks_with.items()},
        "subscription": {**sub_info, "in_subscription_only": len(uids_sub - uids_load),
                         "in_load_only": len(uids_load - uids_sub)},
        "full_length_disks": sum(1 for r in rows if r["rows_clean"] == 17280),
        "source_sha256": {
            "subscription": sha256_bytes(sub_raw),
            "load_files": len(files),
            "combined": sha256_bytes("\n".join(digests).encode()),
        },
        "clean_sha256": sha256_of(OUT_CLEAN),
        "summary_sha256": sha256_of(OUT_SUMMARY),
        "runs_sha256": sha256_of(OUT_RUNS),
        "runs_rows": runs.num_rows,
        "clean_rows": totals["rows_clean"],
        "conflicts": {k: np.concatenate(v) for k, v in conflict_parts.items()} or None,
        "disks_conflict_and_out_of_order": sum(
            1 for x in rows if x["conflict_timestamps"] > 0 and x["out_of_order_rows"] > 0),
    }


def report_lines(r: dict) -> list[str]:
    t, d, s = r["totals"], r["disks_with"], r["subscription"]
    lines = [
        f"load_files={r['files']}",
        f"rows_raw={t['rows_raw']}",
        f"out_of_order_rows={t['out_of_order_rows']} disks={d['out_of_order']}",
        f"exact_duplicate_rows_dropped={t['exact_duplicate_rows']} disks={d['exact_duplicates']}",
        f"conflict_timestamps={t['conflict_timestamps']} "
        f"conflict_rows_dropped={t['conflict_rows_dropped']} disks={d['conflicts']}",
        f"negative_rows_dropped={t['negative_rows_dropped']} disks={d['negatives']}",
        f"rows_clean={t['rows_clean']}",
        f"off_grid_rows={t['off_grid_rows']} disks={d['off_grid']}",
        f"disks_with_breaks={d['breaks']} missing_slots={t['missing_slots']}",
        f"disks_with_17280_clean_rows={r['full_length_disks']}",
        f"subscription_rows={s['rows']} unique_uids={s['unique_uids']} "
        f"repeated_uids={s['repeated_uids']} conflicting_uids={len(s['conflicting_uids'])}",
        f"subscription_only_uids={s['in_subscription_only']} load_only_uids={s['in_load_only']}",
    ]
    for uid in s["conflicting_uids"]:
        lines.append(f"conflicting_uid={uid}")
    lines.append(f"output={OUT_CLEAN.relative_to(ROOT).as_posix()} "
                 f"rows={r['clean_rows']} sha256={r['clean_sha256']}")
    lines.append(f"output={OUT_SUMMARY.relative_to(ROOT).as_posix()} "
                 f"rows={r['files']} sha256={r['summary_sha256']}")
    lines.append(f"output={OUT_RUNS.relative_to(ROOT).as_posix()} "
                 f"rows={r['runs_rows']} sha256={r['runs_sha256']}")
    return lines


def fmt_quantiles(values) -> str:
    v = np.asarray(values, dtype="float64")
    if len(v) == 0:
        return "median=nan p90=nan p99=nan max=nan"
    q = np.quantile(v, [0.5, 0.9, 0.99])
    return f"median={q[0]:.4g} p90={q[1]:.4g} p99={q[2]:.4g} max={v.max():.4g}"


def conflict_lines(c: dict | None, disks_with_out_of_order: int = 0) -> list[str]:
    """Size of the disagreement at each dropped timestamp (load columns only)."""
    if c is None:
        return ["conflict_diagnostic timestamps=0"]
    abs_, rel = c["load_abs"], c["load_rel"]
    differs = abs_ > 0
    blocks = c["block_lengths"]
    return [
        f"conflict_diagnostic timestamps={len(abs_)}",
        f"conflict_load_identical_usage_differs={int((~differs).sum())}",
        f"conflict_load_rounding_level(abs<={ROUNDING_UNITS})="
        f"{int((differs & (abs_ <= ROUNDING_UNITS)).sum())}",
        f"conflict_load_small(rel<={SMALL_REL})="
        f"{int(((abs_ > ROUNDING_UNITS) & (rel <= SMALL_REL)).sum())}",
        f"conflict_load_large(rel>{SMALL_REL})="
        f"{int(((abs_ > ROUNDING_UNITS) & (rel > SMALL_REL)).sum())}",
        f"conflict_load_abs_spread_where_differs {fmt_quantiles(abs_[differs])}",
        f"conflict_load_rel_spread_where_differs {fmt_quantiles(rel[differs])}",
        f"conflict_usage_spread_mb {fmt_quantiles(c['usage_spread'])}",
        f"conflict_rows_per_timestamp pairs={int((c['rows_per_ts'] == 2).sum())} "
        f"more_than_two={int((c['rows_per_ts'] > 2).sum())} max={int(c['rows_per_ts'].max())}",
        f"conflict_blocks={len(blocks)} block_rows {fmt_quantiles(blocks)} "
        f"multiple_of_{DAY_ROWS}={int((blocks % DAY_ROWS == 0).sum())}",
        f"disks_with_conflicts_and_out_of_order={disks_with_out_of_order}",
    ]


def shared_window_stats(runs: pd.DataFrame) -> dict:
    """Disks that share a complete 24 h window (288 consecutive samples).

    A disk offers a window starting at t when one of its runs covers
    [t, t + 287 steps]. Windows are compared per sampling offset
    (start_ts mod 300), since only identical timestamps can be summed.
    """
    full = runs[runs["run_rows"] >= DAY_ROWS]
    out = {"disks_with_window": int(full["disk_uid"].nunique()), "offset_classes": 0,
           "shared_disks": 0, "largest_group": 0, "largest_group_start_ts": None}
    if full.empty:
        return out
    offset = (full["run_start_ts"] % STEP_S).to_numpy()
    out["offset_classes"] = int(len(np.unique(offset)))
    shared: set[str] = set()
    for cls in np.unique(offset):
        part = full[offset == cls]
        lo = ((part["run_start_ts"].to_numpy() - cls) // STEP_S).astype("int64")
        hi = lo + (part["run_rows"].to_numpy() - DAY_ROWS)
        points = np.unique(np.concatenate((lo, hi + 1)))
        delta = np.zeros(len(points), dtype="int64")
        np.add.at(delta, np.searchsorted(points, lo), 1)
        np.add.at(delta, np.searchsorted(points, hi + 1), -1)
        count = np.cumsum(delta)
        peak = int(count.max())
        if peak > out["largest_group"]:
            out["largest_group"] = peak
            out["largest_group_start_ts"] = int(points[int(count.argmax())] * STEP_S + cls)
        uids = part["disk_uid"].to_numpy()
        for i in range(len(lo)):
            a, b = np.searchsorted(points, lo[i]), np.searchsorted(points, hi[i] + 1)
            if count[a:b].max() >= 2:
                shared.add(uids[i])
    out["shared_disks"] = len(shared)
    return out


def aggregation_lines(summary_path: Path, runs_path: Path) -> list[str]:
    """Per-disk load level, burstiness, and shared-window figures for N-disk aggregation."""
    s = pd.read_parquet(summary_path)
    with_rows = s[s["rows_clean"].astype("int64") > 0]
    med = with_rows["median_total_bw_kbps"].astype("float64")
    p99 = with_rows["p99_total_bw_kbps"].astype("float64")
    defined = med > 0
    w = shared_window_stats(pd.read_parquet(runs_path))
    lines = [
        f"agg_disks={len(s)} disks_with_clean_rows={len(with_rows)}",
        "agg_mean_total_bw_kbps "
        + fmt_quantiles(with_rows["mean_total_bw_kbps"].astype("float64")),
        "agg_peak_total_bw_kbps "
        + fmt_quantiles(with_rows["max_total_bw_kbps"].astype("float64")),
        f"agg_burstiness_p99_over_median disks_defined={int(defined.sum())} "
        f"disks_median_zero={int((~defined).sum())} "
        + fmt_quantiles(p99[defined] / med[defined]),
        f"agg_disks_with_24h_window={w['disks_with_window']} "
        f"sampling_offset_classes={w['offset_classes']}",
        f"agg_disks_sharing_a_24h_window={w['shared_disks']}",
    ]
    if w["largest_group_start_ts"] is None:
        lines.append("agg_largest_24h_group=0")
    else:
        start = pd.Timestamp(w["largest_group_start_ts"], unit="s", tz="UTC")
        lines.append(f"agg_largest_24h_group={w['largest_group']} "
                     f"window_start_ts={w['largest_group_start_ts']} "
                     f"window_start_utc={start:%Y-%m-%d %H:%M}")
    return lines


def update_preprocessing(lines: list[str]) -> None:
    body = "\n".join(f"- `{ln}`" for ln in lines)
    section = (
        "## B3 — Tencent extract\n\n"
        "- Source field order (dataset README): `timestamp, read_iops, read_bw_kbps, "
        "write_iops, write_bw_kbps, disk_usage_mb`; files are headerless\n"
        "- Clean table `data/processed/tencent_load_clean.parquet`: columns "
        "`disk_uid, timestamp, read_iops, write_iops, read_bw_kbps, write_bw_kbps, "
        "disk_usage_mb`, all int64, one row group per disk, rows sorted by timestamp\n"
        "- `timestamp` is source epoch seconds; the rebase to `t_seconds` starting at 0 "
        "happens when a window is cut (`contracts.contracts.to_replay_frame`)\n"
        "- Units: bandwidth stored as **KB/s** (source unit, not pre-converted — see "
        "CLAUDE.md §6 Open Question 4)\n"
        "- Cleaning order: sort by timestamp; drop exact duplicate rows; drop every row of a "
        "timestamp that still repeats with differing values; drop rows with a negative value. "
        "Gaps are reported, never filled\n"
        "- Subscription table: first row kept per `disk_uid`; `conflicting_uid` lines below "
        "are uids whose repeated rows disagree\n"
        "- Counts from the last run of `tencent/extract_load.py`:\n\n"
        f"{body}\n\n"
    )
    text = PREPROCESSING.read_text(encoding="utf-8")
    new, n = re.subn(r"## B3 — Tencent extract\n.*?(?=\n## B4 )", section.rstrip("\n") + "\n",
                     text, count=1, flags=re.S)
    if n != 1:
        raise SystemExit("B3 section not found in PREPROCESSING.md")
    PREPROCESSING.write_text(new, encoding="utf-8")


def update_manifest(r: dict) -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = [
        {"file": OUT_CLEAN.relative_to(ROOT).as_posix(), "rows": r["clean_rows"],
         "sha256": r["clean_sha256"]},
        {"file": OUT_SUMMARY.relative_to(ROOT).as_posix(), "rows": r["files"],
         "sha256": r["summary_sha256"]},
        {"file": OUT_RUNS.relative_to(ROOT).as_posix(), "rows": r["runs_rows"],
         "sha256": r["runs_sha256"]},
    ]
    names = {e["file"] for e in entries}
    data["files"] = [e for e in data.get("files", []) if e.get("file") not in names] + entries
    data.setdefault("sources", {})["tencent"] = r["source_sha256"]
    MANIFEST.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    if len(argv) > 1 and argv[1] == "--aggregation-only":
        print("\n".join(aggregation_lines(OUT_SUMMARY, OUT_RUNS)))
        return 0
    root = Path(argv[1]) if len(argv) > 1 else DEFAULT_DATASET
    r = extract(root)
    lines = (report_lines(r)
             + conflict_lines(r["conflicts"], r["disks_conflict_and_out_of_order"])
             + aggregation_lines(OUT_SUMMARY, OUT_RUNS))
    print("\n".join(lines))
    update_preprocessing(lines)
    update_manifest(r)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
