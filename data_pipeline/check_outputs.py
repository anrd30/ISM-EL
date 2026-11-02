#!/usr/bin/env python3
"""
B6 acceptance check for the data pipeline outputs.

Checks, each reported as one PASS / FAIL / MISSING / SKIP line:
  - manifest.json and PREPROCESSING.md exist; the manifest lists every output file;
  - every file in the manifest: sha256 (and row count when one is recorded);
  - row counts of the Backblaze and Tencent tables against PREPROCESSING.md;
  - key uniqueness: (serial_number, date) in the extract and the timelines,
    serial_number in the drive index, disk_uid in the disk summary;
  - the Backblaze extract: the model set, per-model and pooled failures and drive-days,
    the date range and distinct-date count (EXCLUDED_FILES dates are the only gaps), the
    NaN pattern of smart_187_raw and smart_188_raw (all NaN for non-Seagate models, populated
    for Seagate models) and the populated share of smart_12_raw for every model, against
    the values recorded in backblaze/extract.py; the source file count in the manifest;
  - the 30-day label counts and the eligible-drive split recomputed from the timelines and
    the drive index against the lines in PREPROCESSING.md;
  - the label table (A2): one row per timelines row in the timelines order, key
    (serial_number, date) unique, label values in {-1, 0, 1}, totals overall and per model
    equal to the B2 lines of PREPROCESSING.md;
  - the drive split (A2): one row per drive of the drive index, split values train / val /
    test, no drive in two splits, a failed drive of every model in every split, stratum
    sizes within one drive of the target shares, per-split counts equal to the A2 lines of
    PREPROCESSING.md;
  - the leakage test file is present, and a cut-and-recompute of the stored timelines on a
    sample of drives is identical for rows up to the cut date;
  - the disk index against the contract's `validate_disk_index_frame`;
  - the four profiles, as CSV and as Parquet, against the contract's
    `validate_replay_frame` and `validate_replay_grid`, the 288-row length, and
    CSV/Parquet equality;
  - the four provenance sidecars: they parse, `synthetic` is false, `profile` equals
    the profile name and `n_disks` equals the manifest;
  - the latency ladder of each profile CSV (`build_profiles.latency_ladder`) against
    the expected number of safe rebuild allocations;
  - the profile comparison plot, the selected-disk table and the candidate table:
    present, listed in the manifest and named in PREPROCESSING.md, with the per-profile
    disk counts agreeing between the manifest and the selected-disk table.

With --committed-only, only checks whose inputs are tracked in git run: every check
on a file under data/processed is reported as SKIP, and the rest (manifest,
PREPROCESSING.md, profiles/ CSVs, sidecars and plot) run as usual. Files under
data/processed are not tracked; everything else this script reads is.

Final line: RESULT: PASS (no FAIL, no MISSING), RESULT: FAIL (any FAIL), or
RESULT: INCOMPLETE (no FAIL, some MISSING). Exit status 0 only for PASS.

Usage:
    python check_outputs.py [--committed-only]
"""
from __future__ import annotations

import datetime
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))  # repo root: `contracts`, `sim` and `data_pipeline` packages

import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from contracts import contracts  # noqa: E402
from contracts.contracts import CONTROL_INTERVAL_S, REPLAY_COLUMNS  # noqa: E402
from data_pipeline.backblaze import extract as backblaze_extract  # noqa: E402
from data_pipeline.backblaze import labels as backblaze_labels  # noqa: E402
from data_pipeline.backblaze import timelines as backblaze_timelines  # noqa: E402
from data_pipeline.tencent import build_profiles  # noqa: E402

MANIFEST = ROOT / "manifest.json"
PREPROCESSING = ROOT / "PREPROCESSING.md"

PROCESSED = "data/processed"
PROFILE_NAMES = ("idle", "diurnal", "bursty", "peak")
PROFILE_ROWS = 86400 // CONTROL_INTERVAL_S  # 24 h of samples
LEAKAGE_TEST = "tests/test_no_leakage.py"
LEAKAGE_SAMPLE_PER_MODEL = 100
NAN_PATTERN_COLUMNS = ("smart_187_raw", "smart_188_raw")
SHARED_COLUMNS = ("smart_12_raw",)
POPULATED_SHARE_MIN = 0.99
EXPECTED_SAFE_ACTIONS = {"idle": 4, "diurnal": 3, "bursty": 2, "peak": 1}

BACKBLAZE_FILTERED = f"{PROCESSED}/backblaze_filtered.parquet"
BACKBLAZE_TIMELINES = f"{PROCESSED}/backblaze_timelines.parquet"
BACKBLAZE_DRIVE_INDEX = f"{PROCESSED}/backblaze_drive_index.parquet"
BACKBLAZE_LABELS = f"{PROCESSED}/backblaze_labels.parquet"
BACKBLAZE_SPLIT = f"{PROCESSED}/backblaze_drive_split.parquet"
TENCENT_CLEAN = f"{PROCESSED}/tencent_load_clean.parquet"
TENCENT_SUMMARY = f"{PROCESSED}/tencent_disk_summary.parquet"
TENCENT_RUNS = f"{PROCESSED}/tencent_disk_runs.parquet"
TENCENT_DISK_INDEX = f"{PROCESSED}/tencent_disk_index.parquet"
PROFILE_PARQUET = {n: f"{PROCESSED}/tencent_profile_{n}.parquet" for n in PROFILE_NAMES}
PROFILE_CSV = {n: f"profiles/profile_{n}.csv" for n in PROFILE_NAMES}
PROFILE_SIDECAR = {n: f"profiles/profile_{n}.meta.json" for n in PROFILE_NAMES}
PROFILE_PLOT = "profiles/profile_comparison.png"
PROFILE_DISKS = f"{PROCESSED}/tencent_profile_disks.parquet"
PROFILE_CANDIDATES = f"{PROCESSED}/tencent_profile_candidates.parquet"

BACKBLAZE_FILES = [BACKBLAZE_FILTERED, BACKBLAZE_TIMELINES, BACKBLAZE_DRIVE_INDEX,
                   BACKBLAZE_LABELS, BACKBLAZE_SPLIT]
TENCENT_LOAD_FILES = [TENCENT_CLEAN, TENCENT_SUMMARY, TENCENT_RUNS]
PROFILE_TABLE_FILES = (
    list(PROFILE_PARQUET.values()) + list(PROFILE_CSV.values())
    + [PROFILE_DISKS, PROFILE_CANDIDATES]
)
ALL_OUTPUTS = (
    BACKBLAZE_FILES + TENCENT_LOAD_FILES + [TENCENT_DISK_INDEX] + PROFILE_TABLE_FILES
    + list(PROFILE_SIDECAR.values()) + [PROFILE_PLOT]
)

_STATUSES: list[str] = []
COMMITTED_ONLY = False


def report(status: str, label: str, detail: str = "") -> bool:
    _STATUSES.append(status)
    print(f"[{status}] {label}" + (f": {detail}" if detail else ""))
    return status == "PASS"


def skip(label: str) -> bool:
    return report("SKIP", label, "not tracked in git")


def tracked(rel: str) -> bool:
    """Whether the file is tracked in git: everything except data/processed."""
    return not COMMITTED_ONLY or not rel.startswith(f"{PROCESSED}/")


def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def sha256_of(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def actual_rows(path: Path) -> int | None:
    if path.suffix == ".parquet":
        return pq.ParquetFile(path).metadata.num_rows
    if path.suffix == ".csv":
        return len(pd.read_csv(path))
    return None


def safe_rows(path: Path) -> int | None:
    """Row count, or None when the file cannot be read."""
    try:
        return actual_rows(path)
    except Exception:
        return None


def recorded_rows(text: str) -> dict[str, int]:
    """Row counts written into PREPROCESSING.md: `output=<file> rows=<n>` lines, plus the
    B1 `rows_out=<n>` line, which belongs to the filtered Backblaze extract."""
    rows = {m.group(1): int(m.group(2)) for m in re.finditer(r"output=(\S+) rows=(\d+)", text)}
    b1 = re.search(r"## B1 — Backblaze extract\n.*?(?=\n## B2 )", text, flags=re.S)
    if b1:
        m = re.search(r"rows_out=(\d+)", b1.group(0))
        if m:
            rows.setdefault(BACKBLAZE_FILTERED, int(m.group(1)))
    return rows


def duplicate_key_rows(df: pd.DataFrame, columns: list[str]) -> int:
    return int(df.duplicated(subset=columns).sum())


def check_manifest_exists() -> bool:
    if MANIFEST.exists():
        return report("PASS", "manifest.json exists")
    return report("MISSING", "manifest.json exists")


def check_preprocessing_exists() -> bool:
    if PREPROCESSING.exists():
        return report("PASS", "PREPROCESSING.md exists")
    return report("MISSING", "PREPROCESSING.md exists")


def check_manifest_lists_outputs(manifest: dict) -> bool:
    listed = {e["file"] for e in manifest.get("files", [])}
    absent = [f for f in ALL_OUTPUTS if f not in listed]
    return report(verdict(not absent), "manifest lists every output file",
                  f"missing_entries={absent}" if absent else f"files={len(ALL_OUTPUTS)}")


def check_manifest_hashes(manifest: dict) -> None:
    for entry in manifest.get("files", []):
        rel = entry["file"]
        label = f"sha256 {rel}"
        if not tracked(rel):
            skip(label)
            continue
        path = ROOT / rel
        if not path.exists():
            report("MISSING", label)
            continue
        problems = []
        if sha256_of(path) != entry["sha256"]:
            problems.append("sha256 differs from manifest")
        if entry.get("rows") is not None:
            rows = safe_rows(path)
            if rows != entry["rows"]:
                problems.append(f"rows {'unreadable' if rows is None else rows} != manifest {entry['rows']}")
        report(verdict(not problems), label, "; ".join(problems))


def check_row_counts(files: list[str], recorded: dict[str, int]) -> None:
    for rel in files:
        label = f"rows vs PREPROCESSING.md {rel}"
        path = ROOT / rel
        if not tracked(rel):
            skip(label)
        elif not path.exists():
            report("MISSING", label)
        elif rel not in recorded:
            report("FAIL", label, "no row count recorded in PREPROCESSING.md")
        else:
            rows = safe_rows(path)
            if rows is None:
                report("FAIL", label, "file unreadable")
            else:
                report(verdict(rows == recorded[rel]), label, f"file={rows} recorded={recorded[rel]}")


def check_unique_key(rel: str, columns: list[str]) -> bool:
    path = ROOT / rel
    label = f"unique key ({', '.join(columns)}) {rel}"
    if not tracked(rel):
        return skip(label)
    if not path.exists():
        return report("MISSING", label)
    try:
        frame = pq.read_table(path, columns=columns).to_pandas()
    except Exception as err:
        return report("FAIL", label, f"file unreadable: {err}")
    duplicates = duplicate_key_rows(frame, columns)
    return report(verdict(duplicates == 0), label, f"duplicate_rows={duplicates}")


def read_backblaze_extract_summary(path: Path) -> pd.DataFrame:
    """Per model: rows, failures, rows with a value in each NAN_PATTERN_COLUMNS and SHARED_COLUMNS column."""
    counted = [*NAN_PATTERN_COLUMNS, *SHARED_COLUMNS]
    table = pq.read_table(path, columns=["model", "failure", *counted])
    aggregates = [("failure", "sum"), ("failure", "count")]
    aggregates += [(c, "count") for c in counted]
    grouped = table.group_by("model").aggregate(aggregates).to_pandas().set_index("model")
    return grouped.rename(columns={"failure_sum": "failures", "failure_count": "rows",
                                   **{f"{c}_count": c for c in counted}})


def check_backblaze_extract() -> None:
    rel = BACKBLAZE_FILTERED
    path = ROOT / rel
    labels = ("backblaze model set", "backblaze counts", "backblaze date range",
              "backblaze NaN pattern", "backblaze shared column coverage")
    if not tracked(rel):
        for label in labels:
            skip(f"{label} {rel}")
        return
    if not path.exists():
        for label in labels:
            report("MISSING", f"{label} {rel}")
        return
    try:
        summary = read_backblaze_extract_summary(path)
        dates = pq.read_table(path, columns=["date"]).column("date")
        bounds = pc.min_max(dates).as_py()
        distinct = len(pc.unique(dates))
    except Exception as err:
        for label in labels:
            report("FAIL", f"{label} {rel}", f"file unreadable: {err}")
        return

    expected_models = list(backblaze_extract.MODELS)
    measured_models = sorted(summary.index)
    report(verdict(measured_models == sorted(expected_models)), f"backblaze model set {rel}",
           f"models={measured_models} expected={sorted(expected_models)}")

    pooled_failures = pooled_days = 0
    for model, (failures, drive_days) in backblaze_extract.EXPECTED_PER_MODEL.items():
        label = f"backblaze counts {model}"
        if model not in summary.index:
            report("FAIL", label, "model absent from the extract")
            continue
        row = summary.loc[model]
        pooled_failures += int(row["failures"])
        pooled_days += int(row["rows"])
        report(verdict(int(row["failures"]) == failures and int(row["rows"]) == drive_days), label,
               f"failures={int(row['failures'])} expected={failures} "
               f"drive_days={int(row['rows'])} expected={drive_days}")
    exp_failures, exp_days = backblaze_extract.EXPECTED_POOLED
    report(verdict(pooled_failures == exp_failures and pooled_days == exp_days), "backblaze counts pooled",
           f"failures={pooled_failures} expected={exp_failures} "
           f"drive_days={pooled_days} expected={exp_days}")

    first = bounds["min"].isoformat() if bounds["min"] is not None else "none"
    last = bounds["max"].isoformat() if bounds["max"] is not None else "none"
    exp_first, exp_last = backblaze_extract.EXPECTED_DATE_RANGE
    ok = (first, last) == (exp_first, exp_last) and distinct == backblaze_extract.EXPECTED_FILES
    report(verdict(ok), f"backblaze date range {rel}",
           f"first={first} last={last} distinct_dates={distinct} expected={exp_first}..{exp_last} "
           f"in {backblaze_extract.EXPECTED_FILES} dates")

    problems = []
    for model in summary.index:
        seagate = model in backblaze_extract.SEAGATE_MODELS
        for column in NAN_PATTERN_COLUMNS:
            share = int(summary.loc[model, column]) / int(summary.loc[model, "rows"])
            if seagate and share < POPULATED_SHARE_MIN:
                problems.append(f"{model} {column} populated share {share:.4f}")
            if not seagate and share != 0:
                problems.append(f"{model} {column} populated share {share:.4f}, expected 0")
    report(verdict(not problems), f"backblaze NaN pattern {rel}",
           "; ".join(problems) if problems else
           f"{', '.join(NAN_PATTERN_COLUMNS)} populated for {', '.join(backblaze_extract.SEAGATE_MODELS)}, "
           "all NaN for the other models")

    shared_problems = []
    for model in summary.index:
        for column in SHARED_COLUMNS:
            share = int(summary.loc[model, column]) / int(summary.loc[model, "rows"])
            if share < POPULATED_SHARE_MIN:
                shared_problems.append(f"{model} {column} populated share {share:.4f}")
    report(verdict(not shared_problems), f"backblaze shared column coverage {rel}",
           "; ".join(shared_problems) if shared_problems else
           f"{', '.join(SHARED_COLUMNS)} populated share >= {POPULATED_SHARE_MIN} for every model")


def check_backblaze_sources(manifest: dict) -> None:
    label = "backblaze source files in manifest"
    names = sorted(manifest.get("sources", {}).get("backblaze", {}))
    first = datetime.date.fromisoformat(backblaze_extract.EXPECTED_DATE_RANGE[0])
    last = datetime.date.fromisoformat(backblaze_extract.EXPECTED_DATE_RANGE[1])
    excluded = {key.split("/")[-1] for key in backblaze_extract.EXCLUDED_FILES}
    expected = {(first + datetime.timedelta(days=i)).isoformat() + ".csv"
                for i in range((last - first).days + 1)} - excluded
    problems = []
    if len(names) != backblaze_extract.EXPECTED_FILES:
        problems.append(f"files={len(names)} expected={backblaze_extract.EXPECTED_FILES}")
    if set(names) != expected:
        problems.append(f"missing={len(expected - set(names))} unexpected={len(set(names) - expected)}")
    report(verdict(not problems), label, "; ".join(problems) if problems else f"files={len(names)}")


def parse_label_lines(preprocessing: str) -> dict[str, dict[str, tuple[int, ...]]]:
    """The B2 label-count and drive-split lines of PREPROCESSING.md, keyed by line kind and model."""
    match = re.search(r"## B2 — Per-drive timelines\n.*?(?=\n## |\Z)", preprocessing, flags=re.S)
    section = match.group(0) if match else ""
    labels = {m.group(1): tuple(int(g) for g in m.groups()[1:]) for m in re.finditer(
        r"labels_h\d+ model=(.+?) positive=(\d+) negative=(\d+) censored=(\d+)", section)}
    drives = {m.group(1): tuple(int(g) for g in m.groups()[1:]) for m in re.finditer(
        r"label_drives model=(.+?) drives=(\d+) eligible=(\d+) with_positive=(\d+) "
        r"negative_only=(\d+) all_censored=(\d+)", section)}
    return {"labels": labels, "drives": drives}


def recompute_label_lines(timelines_path: Path, index_path: Path) -> list[str]:
    index = pd.read_parquet(index_path)
    pf = pq.ParquetFile(timelines_path)
    totals: dict[str, dict[str, int]] = {}
    flag_parts = []
    for i in range(pf.num_row_groups):
        frame = pf.read_row_group(i, columns=["serial_number", "model", "date"]).to_pandas()
        labels = backblaze_timelines.row_labels(frame, index)
        for model, g in frame.groupby(frame["model"].astype(object)):
            counts = backblaze_timelines.label_counts(labels[g.index.to_numpy()])
            bucket = totals.setdefault(str(model), dict.fromkeys(counts, 0))
            for key, value in counts.items():
                bucket[key] += value
        flag_parts.append(backblaze_timelines.drive_label_flags(frame, labels))
    if flag_parts:
        flags = pd.concat(flag_parts).groupby(level=0).any()
    else:
        flags = pd.DataFrame({"has_positive": [], "has_negative": []})
    return backblaze_timelines.label_lines(totals, index, flags)


def check_backblaze_labels(preprocessing: str) -> None:
    label = "backblaze label counts vs PREPROCESSING.md"
    if not tracked(BACKBLAZE_TIMELINES):
        skip(label)
        return
    timelines_path, index_path = ROOT / BACKBLAZE_TIMELINES, ROOT / BACKBLAZE_DRIVE_INDEX
    if not (timelines_path.exists() and index_path.exists()):
        report("MISSING", label)
        return
    recorded = parse_label_lines(preprocessing)
    if not recorded["labels"] or not recorded["drives"]:
        report("FAIL", label, "no label lines recorded in PREPROCESSING.md")
        return
    try:
        measured = parse_label_lines(
            "## B2 — Per-drive timelines\n" + "\n".join(recompute_label_lines(timelines_path, index_path)))
    except Exception as err:
        report("FAIL", label, f"recomputation failed: {err}")
        return
    problems = [f"{kind} {model}: recorded={recorded[kind].get(model)} recomputed={measured[kind].get(model)}"
                for kind in ("labels", "drives")
                for model in sorted(set(recorded[kind]) | set(measured[kind]))
                if recorded[kind].get(model) != measured[kind].get(model)]
    all_row = measured["labels"].get("all", (0, 0, 0))
    report(verdict(not problems), label,
           "; ".join(problems) if problems else
           f"horizon_days={backblaze_timelines.HORIZON_DAYS} positive={all_row[0]} "
           f"negative={all_row[1]} censored={all_row[2]} models={len(measured['labels']) - 1}")


def parse_split_lines(preprocessing: str) -> dict[tuple[str, str], tuple[int, ...]]:
    """The A2 per-split, per-model count lines of PREPROCESSING.md."""
    match = re.search(r"## A2 — Horizon labels and drive split\n.*?(?=\n## |\Z)", preprocessing, flags=re.S)
    section = match.group(0) if match else ""
    return {(m.group(1), m.group(2)): tuple(int(g) for g in m.groups()[2:]) for m in re.finditer(
        r"a2_split split=(\w+) model=(.+?) drives=(\d+) failed_drives=(\d+) positive=(\d+) "
        r"negative=(\d+) censored=(\d+)", section)}


MODEL_COMPARE_ROWS = 1_000_000


def model_columns_equal(first: Path, second: Path) -> bool:
    """Elementwise equality of the model column of two Parquet files, read dictionary-encoded and
    decoded MODEL_COMPARE_ROWS rows at a time."""
    a = pq.read_table(first, columns=["model"], read_dictionary=["model"]).column("model").combine_chunks()
    b = pq.read_table(second, columns=["model"], read_dictionary=["model"]).column("model").combine_chunks()
    if len(a) != len(b):
        return False
    for start in range(0, len(a), MODEL_COMPARE_ROWS):
        left = a.slice(start, MODEL_COMPARE_ROWS).dictionary_decode().cast(pa.string())
        right = b.slice(start, MODEL_COMPARE_ROWS).dictionary_decode().cast(pa.string())
        if not left.equals(right):
            return False
    return True


LABEL_CHECKS = ("backblaze labels rows and key order", "backblaze labels values",
                "backblaze labels totals")


def check_labels_file(preprocessing: str) -> None:
    rel = BACKBLAZE_LABELS
    if not tracked(rel):
        for label in LABEL_CHECKS:
            skip(f"{label} {rel}")
        return
    labels_path, timelines_path = ROOT / rel, ROOT / BACKBLAZE_TIMELINES
    if not (labels_path.exists() and timelines_path.exists()):
        for label in LABEL_CHECKS:
            report("MISSING", f"{label} {rel}")
        return
    try:
        table = pq.read_table(labels_path, columns=["serial_number", "date", "model", "label"])
        keys = pq.read_table(timelines_path, columns=["serial_number", "date"])
    except Exception as err:
        for label in LABEL_CHECKS:
            report("FAIL", f"{label} {rel}", f"file unreadable: {err}")
        return
    same = (table.num_rows == keys.num_rows
            and table.column("serial_number").cast(pa.string()).equals(
                keys.column("serial_number").cast(pa.string()))
            and table.column("date").equals(keys.column("date")))
    try:
        models_equal = same and model_columns_equal(labels_path, timelines_path)
    except Exception as err:
        report("FAIL", f"backblaze labels rows and key order {rel}", f"model column not compared: {err}")
        models_equal = None
    if models_equal is not None:
        report(verdict(models_equal), f"backblaze labels rows and key order {rel}",
               f"labels={table.num_rows} timelines={keys.num_rows} keys_equal={same} "
               f"models_equal={models_equal}")

    present = {v for v in pc.unique(table.column("label")).to_pylist()}
    allowed = set(backblaze_labels.LABEL_VALUES)
    report(verdict(present <= allowed), f"backblaze labels values {rel}",
           f"values={sorted(v for v in present if v is not None)} "
           f"nulls={'yes' if None in present else 'no'}")

    grouped = table.group_by(["model", "label"]).aggregate([("serial_number", "count")]).to_pandas()
    measured: dict[str, tuple[int, int, int]] = {}
    for model, part in grouped.groupby("model"):
        by_label = dict(zip(part["label"], part["serial_number_count"]))
        measured[str(model)] = (int(by_label.get(1, 0)), int(by_label.get(0, 0)), int(by_label.get(-1, 0)))
    measured["all"] = tuple(sum(v[k] for v in measured.values()) for k in range(3))
    recorded = parse_label_lines(preprocessing)["labels"]
    problems = [f"{model}: recorded={recorded.get(model)} file={measured.get(model)}"
                for model in sorted(set(recorded) | set(measured))
                if recorded.get(model) != measured.get(model)]
    report(verdict(not problems and bool(recorded)), f"backblaze labels totals {rel}",
           "; ".join(problems) if problems else
           f"positive={measured['all'][0]} negative={measured['all'][1]} censored={measured['all'][2]}")


SPLIT_CHECKS = ("backblaze split one row per drive", "backblaze split model and failure metadata",
                "backblaze split values", "backblaze split drives unique",
                "backblaze split failed drives per model", "backblaze split stratum sizes",
                "backblaze split counts vs PREPROCESSING.md")


def check_drive_split(preprocessing: str) -> None:
    rel = BACKBLAZE_SPLIT
    if not tracked(rel):
        for label in SPLIT_CHECKS:
            skip(f"{label} {rel}")
        return
    split_path, index_path, labels_path = ROOT / rel, ROOT / BACKBLAZE_DRIVE_INDEX, ROOT / BACKBLAZE_LABELS
    if not (split_path.exists() and index_path.exists()):
        for label in SPLIT_CHECKS:
            report("MISSING", f"{label} {rel}")
        return
    try:
        split = pd.read_parquet(split_path)
        index = pd.read_parquet(index_path, columns=["serial_number", "model", "failed_in_window"])
        absent_columns = [c for c in backblaze_labels.SPLIT_COLUMNS if c not in split.columns]
        if absent_columns:
            for label in SPLIT_CHECKS:
                report("FAIL", f"{label} {rel}", f"missing columns={absent_columns}")
            return
        split_names = list(backblaze_labels.SPLITS)
        split["serial_number"] = split["serial_number"].astype(object)
        index["serial_number"] = index["serial_number"].astype(object)
    except Exception as err:
        for label in SPLIT_CHECKS:
            report("FAIL", f"{label} {rel}", f"file unreadable: {err}")
        return

    missing = set(index["serial_number"]) - set(split["serial_number"])
    extra = set(split["serial_number"]) - set(index["serial_number"])
    report(verdict(len(split) == len(index) and not missing and not extra and
                   list(split.columns) == backblaze_labels.SPLIT_COLUMNS),
           f"backblaze split one row per drive {rel}",
           f"split_rows={len(split)} index_rows={len(index)} missing={len(missing)} extra={len(extra)}")
    meta = index.merge(split[["serial_number", "model", "ever_failed"]].drop_duplicates("serial_number"),
                       on="serial_number", how="left", suffixes=("", "_split"))
    differing = (meta["model_split"].astype(object).ne(meta["model"].astype(object))
                 | meta["ever_failed"].astype(object).ne(meta["failed_in_window"].astype(bool)))
    report(verdict(not differing.any()), f"backblaze split model and failure metadata {rel}",
           f"mismatching_drives={int(differing.sum())}")
    unknown = sorted(set(split["split"].astype(object)) - set(split_names))
    report(verdict(not unknown), f"backblaze split values {rel}", f"unknown_values={unknown}")
    duplicates = int(split["serial_number"].duplicated().sum())
    report(verdict(duplicates == 0), f"backblaze split drives unique {rel}", f"duplicate_drives={duplicates}")

    joined = index.merge(split[["serial_number", "split"]].drop_duplicates("serial_number"),
                         on="serial_number", how="left")
    joined["split"] = joined["split"].astype(object)
    failed = joined[joined["failed_in_window"].astype(bool)]
    lacking = [f"{model}/{name}" for model in sorted(set(index["model"].astype(object)))
               for name in split_names
               if not ((failed["model"].astype(object) == model) & (failed["split"] == name)).any()]
    report(verdict(not lacking), f"backblaze split failed drives per model {rel}",
           f"splits_without_a_failed_drive={lacking}" if lacking else "every model has a failed drive in every split")

    problems = []
    for (model, ever), group in joined.groupby([joined["model"].astype(object),
                                                joined["failed_in_window"].astype(bool)]):
        for name, fraction in zip(split_names, backblaze_labels.FRACTIONS):
            target = len(group) * fraction
            actual = int((group["split"] == name).sum())
            if not abs(actual - target) < 1:
                problems.append(f"{model} failed={ever} {name}: {actual} vs {target:.2f}")
    report(verdict(not problems), f"backblaze split stratum sizes {rel}",
           "; ".join(problems) if problems else "every stratum within one drive of the target shares")

    label = f"backblaze split counts vs PREPROCESSING.md {rel}"
    if not labels_path.exists():
        report("MISSING", label)
        return
    recorded = parse_split_lines(preprocessing)
    try:
        stats = backblaze_labels.split_statistics(labels_path, split.assign(
            ever_failed=split["ever_failed"].astype(bool)))
    except Exception as err:
        report("FAIL", label, f"recomputation failed: {err}")
        return
    measured = {key: (c["drives"], c["failed_drives"], c["positive"], c["negative"], c["censored"])
                for key, c in stats["counts"].items()}
    differing = [f"{key[0]}/{key[1]}: recorded={recorded.get(key)} recomputed={measured.get(key)}"
                 for key in sorted(set(recorded) | set(measured)) if recorded.get(key) != measured.get(key)]
    report(verdict(not differing and bool(recorded)), label,
           "; ".join(differing[:3]) if differing else f"groups={len(measured)}")


def same_frames(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True),
                                      check_exact=True, check_dtype=False)
    except AssertionError:
        return False
    return True


def check_backblaze_leakage() -> None:
    report(verdict((ROOT / LEAKAGE_TEST).exists()), f"leakage test present {LEAKAGE_TEST}")
    label = "leakage cut-and-recompute backblaze_timelines"
    if not tracked(BACKBLAZE_TIMELINES):
        skip(label)
        return
    paths = [ROOT / p for p in (BACKBLAZE_FILTERED, BACKBLAZE_TIMELINES, BACKBLAZE_DRIVE_INDEX)]
    if not all(p.exists() for p in paths):
        report("MISSING", label)
        return
    filtered_path, timelines_path, index_path = paths
    try:
        index = pd.read_parquet(index_path).sort_values("serial_number")
        first, last = pd.Timestamp(index["first_seen"].min()), pd.Timestamp(index["last_seen"].max())
        cut = (first + (last - first) // 2).normalize()
        chosen = pd.concat([
            index.groupby("model").head(LEAKAGE_SAMPLE_PER_MODEL),
            index[index["failed_in_window"]].groupby("model").head(LEAKAGE_SAMPLE_PER_MODEL),
        ])
        serials = sorted(set(chosen["serial_number"].astype(str)))
        filters = [("serial_number", "in", serials)]
        columns = backblaze_timelines.CORE_IN + list(backblaze_timelines.SMART_COLUMNS)
        sample = pq.read_table(filtered_path, columns=columns, filters=filters).to_pandas()
        stored = pq.read_table(timelines_path, filters=filters).to_pandas()
        stored = stored.assign(date=pd.to_datetime(stored["date"]),
                               first_seen=pd.to_datetime(stored["first_seen"]))
        stored = stored.sort_values(backblaze_timelines.KEY, kind="stable")
        full = backblaze_timelines.build_timelines(sample)
        kept = sample[pd.to_datetime(sample["date"]) <= cut]
        truncated = backblaze_timelines.build_timelines(kept)
    except Exception as err:
        report("FAIL", label, f"not computed: {err}")
        return
    problems = []
    if not same_frames(full, stored):
        problems.append("stored timelines differ from a recomputation")
    if not same_frames(truncated, full[full["date"] <= cut]):
        problems.append("rows up to the cut differ when the input is truncated at the cut")
    report(verdict(not problems), label, "; ".join(problems) if problems else
           f"cut={cut.date().isoformat()} drives={len(serials)} rows={len(full)}")


def check_disk_index() -> bool:
    path = ROOT / TENCENT_DISK_INDEX
    label = f"validate_disk_index_frame {TENCENT_DISK_INDEX}"
    if not tracked(TENCENT_DISK_INDEX):
        return skip(label)
    if not path.exists():
        return report("MISSING", label)
    try:
        contracts.validate_disk_index_frame(pd.read_parquet(path))
    except contracts.ContractViolation as err:
        return report("FAIL", label, str(err))
    except Exception as err:
        return report("FAIL", label, f"file unreadable: {err}")
    return report("PASS", label)


def check_profile_frame(rel: str, frame: pd.DataFrame) -> bool:
    label = f"replay contract {rel}"
    if list(frame.columns) != list(REPLAY_COLUMNS):
        return report("FAIL", label, f"columns={list(frame.columns)} expected={list(REPLAY_COLUMNS)}")
    try:
        contracts.validate_replay_frame(frame)
        contracts.validate_replay_grid(frame["t_seconds"].to_numpy())
    except contracts.ContractViolation as err:
        return report("FAIL", label, str(err))
    if len(frame) != PROFILE_ROWS:
        return report("FAIL", label, f"rows={len(frame)} expected={PROFILE_ROWS}")
    return report("PASS", label, f"rows={len(frame)} step_s={CONTROL_INTERVAL_S}")


def check_profiles() -> None:
    for name in PROFILE_NAMES:
        frames = {}
        for rel, read in ((PROFILE_CSV[name], pd.read_csv), (PROFILE_PARQUET[name], pd.read_parquet)):
            label = f"replay contract {rel}"
            if not tracked(rel):
                skip(label)
                continue
            path = ROOT / rel
            if not path.exists():
                report("MISSING", label)
                continue
            try:
                frames[rel] = read(path)
            except Exception as err:
                report("FAIL", label, f"file unreadable: {err}")
                continue
            check_profile_frame(rel, frames[rel])
        label = f"CSV equals Parquet profile_{name}"
        if not tracked(PROFILE_PARQUET[name]):
            skip(label)
        elif len(frames) == 2:
            csv_frame, parquet_frame = frames.values()
            same = list(csv_frame.columns) == list(parquet_frame.columns) and csv_frame.equals(parquet_frame)
            report(verdict(same), label)


def check_sidecars(manifest: dict) -> None:
    for name in PROFILE_NAMES:
        rel = PROFILE_SIDECAR[name]
        label = f"sidecar {rel}"
        path = ROOT / rel
        if not path.exists():
            report("MISSING", label)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as err:
            report("FAIL", label, f"not parseable: {err}")
            continue
        problems = []
        if payload.get("synthetic") is not False:
            problems.append(f"synthetic={payload.get('synthetic')!r}, expected false")
        if payload.get("profile") != name:
            problems.append(f"profile={payload.get('profile')!r}, expected {name!r}")
        expected_n = manifest.get("profiles", {}).get(name, {}).get("n_disks")
        if payload.get("n_disks") != expected_n:
            problems.append(f"n_disks={payload.get('n_disks')!r}, manifest {expected_n!r}")
        report(verdict(not problems), label, "; ".join(problems))


def check_ladder() -> None:
    for name in PROFILE_NAMES:
        rel = PROFILE_CSV[name]
        label = f"latency ladder {rel}"
        path = ROOT / rel
        if not path.exists():
            report("MISSING", label)
            continue
        try:
            ladder = build_profiles.latency_ladder(pd.read_csv(path), name)
        except Exception as err:
            report("FAIL", label, f"ladder not computed: {err}")
            continue
        safe = ladder["safe_actions"]
        fractions = {a: round(f, 4) for a, f in ladder["violation_fraction_by_action_mbps"].items()}
        report(verdict(safe == EXPECTED_SAFE_ACTIONS[name]), label,
               f"safe_actions={safe} expected={EXPECTED_SAFE_ACTIONS[name]} "
               f"violation_fraction_by_action_mbps={fractions}")


def check_profile_references(manifest: dict, preprocessing: str) -> None:
    listed = {e["file"] for e in manifest.get("files", [])}
    for rel in (PROFILE_PLOT, PROFILE_DISKS, PROFILE_CANDIDATES):
        label = f"referenced {rel}"
        if not tracked(rel):
            skip(label)
        elif not (ROOT / rel).exists():
            report("MISSING", label)
        else:
            problems = []
            if rel not in listed:
                problems.append("not listed in manifest.json")
            if Path(rel).name not in preprocessing:
                problems.append("not named in PREPROCESSING.md")
            report(verdict(not problems), label, "; ".join(problems))

    label = "manifest profile disk counts vs tencent_profile_disks.parquet"
    if not tracked(PROFILE_DISKS):
        skip(label)
    elif not (ROOT / PROFILE_DISKS).exists():
        report("MISSING", label)
    else:
        try:
            counts = pd.read_parquet(ROOT / PROFILE_DISKS)["profile"].value_counts().to_dict()
        except Exception as err:
            report("FAIL", label, f"file unreadable: {err}")
            return
        profiles = manifest.get("profiles", {})
        expected = {n: profiles.get(n, {}).get("n_disks") for n in PROFILE_NAMES}
        actual = {n: counts.get(n) for n in PROFILE_NAMES}
        report(verdict(expected == actual), label, f"manifest={expected} table={actual}")


def main(argv: list[str] | None = None) -> int:
    global COMMITTED_ONLY
    args = sys.argv[1:] if argv is None else argv
    if any(a != "--committed-only" for a in args):
        print(__doc__)
        return 2
    COMMITTED_ONLY = "--committed-only" in args
    _STATUSES.clear()

    check_manifest_exists()
    check_preprocessing_exists()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    preprocessing = PREPROCESSING.read_text(encoding="utf-8") if PREPROCESSING.exists() else ""
    recorded = recorded_rows(preprocessing)

    check_manifest_lists_outputs(manifest)
    check_manifest_hashes(manifest)
    check_row_counts(BACKBLAZE_FILES, recorded)
    check_unique_key(BACKBLAZE_FILTERED, backblaze_extract.KEY)
    check_unique_key(BACKBLAZE_TIMELINES, backblaze_timelines.KEY)
    check_unique_key(BACKBLAZE_DRIVE_INDEX, ["serial_number"])
    check_backblaze_extract()
    check_backblaze_sources(manifest)
    check_backblaze_labels(preprocessing)
    check_backblaze_leakage()
    check_unique_key(BACKBLAZE_LABELS, backblaze_labels.LABEL_KEY)
    check_labels_file(preprocessing)
    check_drive_split(preprocessing)
    check_row_counts(TENCENT_LOAD_FILES, recorded)
    check_unique_key(TENCENT_SUMMARY, ["disk_uid"])
    check_disk_index()
    check_row_counts([TENCENT_DISK_INDEX], recorded)
    check_profiles()
    check_row_counts(PROFILE_TABLE_FILES, recorded)
    check_sidecars(manifest)
    check_ladder()
    check_profile_references(manifest, preprocessing)

    if "FAIL" in _STATUSES:
        result = "FAIL"
    elif "MISSING" in _STATUSES:
        result = "INCOMPLETE"
    else:
        result = "PASS"
    print("\nRESULT:", result)
    return 0 if result == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
