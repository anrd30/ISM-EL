#!/usr/bin/env python3
"""
B6 acceptance check for the data pipeline outputs.

Checks, each reported as one PASS / FAIL / MISSING / SKIP line:
  - manifest.json and PREPROCESSING.md exist; the manifest lists every output file;
  - every file in the manifest: sha256 (and row count when one is recorded);
  - row counts of the Backblaze and Tencent tables against PREPROCESSING.md;
  - key uniqueness: (serial_number, date) in the extract and the timelines,
    serial_number in the drive index, disk_uid in the disk summary;
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

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))  # repo root: `contracts`, `sim` and `data_pipeline` packages

import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from contracts import contracts  # noqa: E402
from contracts.contracts import CONTROL_INTERVAL_S, REPLAY_COLUMNS  # noqa: E402
from data_pipeline.backblaze import extract as backblaze_extract  # noqa: E402
from data_pipeline.backblaze import timelines as backblaze_timelines  # noqa: E402
from data_pipeline.tencent import build_profiles  # noqa: E402

MANIFEST = ROOT / "manifest.json"
PREPROCESSING = ROOT / "PREPROCESSING.md"

PROCESSED = "data/processed"
PROFILE_NAMES = ("idle", "diurnal", "bursty", "peak")
PROFILE_ROWS = 86400 // CONTROL_INTERVAL_S  # 24 h of samples
EXPECTED_SAFE_ACTIONS = {"idle": 4, "diurnal": 3, "bursty": 2, "peak": 1}

BACKBLAZE_FILTERED = f"{PROCESSED}/backblaze_filtered.parquet"
BACKBLAZE_TIMELINES = f"{PROCESSED}/backblaze_timelines.parquet"
BACKBLAZE_DRIVE_INDEX = f"{PROCESSED}/backblaze_drive_index.parquet"
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

BACKBLAZE_FILES = [BACKBLAZE_FILTERED, BACKBLAZE_TIMELINES, BACKBLAZE_DRIVE_INDEX]
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
