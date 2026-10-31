#!/usr/bin/env python3
"""
B6 acceptance check for the data pipeline outputs.

Checks, each reported as one PASS / FAIL / MISSING line:
  - manifest.json and PREPROCESSING.md exist; the manifest lists every output file;
  - every file in the manifest: sha256 (and row count when one is recorded);
  - row counts of the Backblaze and Tencent tables against PREPROCESSING.md;
  - key uniqueness: (serial_number, date) in the extract and the timelines,
    serial_number in the drive index, disk_uid in the disk summary;
  - the disk index against the contract's `validate_disk_index_frame`;
  - the four profiles, as CSV and as Parquet, against the contract's
    `validate_replay_frame` and `validate_replay_grid`, the 288-row length, and
    CSV/Parquet equality;
  - the profile comparison plot, the selected-disk table and the candidate table:
    present, listed in the manifest and named in PREPROCESSING.md, with the per-profile
    disk counts agreeing between the manifest and the selected-disk table.

Exit status 0 when every check passes, 1 otherwise.

Usage:
    python check_outputs.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))  # repo root: `contracts` and `data_pipeline` packages

import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from contracts import contracts  # noqa: E402
from contracts.contracts import CONTROL_INTERVAL_S, REPLAY_COLUMNS  # noqa: E402
from data_pipeline.backblaze import extract as backblaze_extract  # noqa: E402
from data_pipeline.backblaze import timelines as backblaze_timelines  # noqa: E402

MANIFEST = ROOT / "manifest.json"
PREPROCESSING = ROOT / "PREPROCESSING.md"

PROCESSED = "data/processed"
PROFILE_NAMES = ("idle", "diurnal", "bursty", "peak")
PROFILE_ROWS = 86400 // CONTROL_INTERVAL_S  # 24 h of samples

BACKBLAZE_FILTERED = f"{PROCESSED}/backblaze_filtered.parquet"
BACKBLAZE_TIMELINES = f"{PROCESSED}/backblaze_timelines.parquet"
BACKBLAZE_DRIVE_INDEX = f"{PROCESSED}/backblaze_drive_index.parquet"
TENCENT_CLEAN = f"{PROCESSED}/tencent_load_clean.parquet"
TENCENT_SUMMARY = f"{PROCESSED}/tencent_disk_summary.parquet"
TENCENT_RUNS = f"{PROCESSED}/tencent_disk_runs.parquet"
TENCENT_DISK_INDEX = f"{PROCESSED}/tencent_disk_index.parquet"
PROFILE_PARQUET = {n: f"{PROCESSED}/tencent_profile_{n}.parquet" for n in PROFILE_NAMES}
PROFILE_CSV = {n: f"profiles/profile_{n}.csv" for n in PROFILE_NAMES}
PROFILE_PLOT = "profiles/profile_comparison.png"
PROFILE_DISKS = f"{PROCESSED}/tencent_profile_disks.parquet"
PROFILE_CANDIDATES = f"{PROCESSED}/tencent_profile_candidates.parquet"

BACKBLAZE_FILES = [BACKBLAZE_FILTERED, BACKBLAZE_TIMELINES, BACKBLAZE_DRIVE_INDEX]
TENCENT_LOAD_FILES = [TENCENT_CLEAN, TENCENT_SUMMARY, TENCENT_RUNS]
PROFILE_FILES = (
    list(PROFILE_PARQUET.values()) + list(PROFILE_CSV.values())
    + [PROFILE_PLOT, PROFILE_DISKS, PROFILE_CANDIDATES]
)
ALL_OUTPUTS = BACKBLAZE_FILES + TENCENT_LOAD_FILES + [TENCENT_DISK_INDEX] + PROFILE_FILES


def report(status: str, label: str, detail: str = "") -> bool:
    print(f"[{status}] {label}" + (f": {detail}" if detail else ""))
    return status == "PASS"


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
    combined = pd.factorize(df[columns[0]])[0].astype("int64")
    for column in columns[1:]:
        codes, uniques = pd.factorize(df[column])
        combined = combined * len(uniques) + codes
    return int(pd.Series(combined).duplicated().sum())


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


def check_manifest_hashes(manifest: dict) -> bool:
    ok = True
    for entry in manifest.get("files", []):
        path = ROOT / entry["file"]
        label = f"sha256 {entry['file']}"
        if not path.exists():
            ok = report("MISSING", label) and ok
            continue
        problems = []
        if sha256_of(path) != entry["sha256"]:
            problems.append("sha256 differs from manifest")
        if entry.get("rows") is not None and actual_rows(path) != entry["rows"]:
            problems.append(f"rows {actual_rows(path)} != manifest {entry['rows']}")
        ok = report(verdict(not problems), label, "; ".join(problems)) and ok
    return ok


def check_row_counts(files: list[str], recorded: dict[str, int]) -> bool:
    ok = True
    for rel in files:
        path = ROOT / rel
        label = f"rows vs PREPROCESSING.md {rel}"
        if not path.exists():
            ok = report("MISSING", label) and ok
        elif rel not in recorded:
            ok = report("FAIL", label, "no row count recorded in PREPROCESSING.md") and ok
        else:
            rows = actual_rows(path)
            ok = report(verdict(rows == recorded[rel]), label,
                        f"file={rows} recorded={recorded[rel]}") and ok
    return ok


def check_unique_key(rel: str, columns: list[str]) -> bool:
    path = ROOT / rel
    label = f"unique key ({', '.join(columns)}) {rel}"
    if not path.exists():
        return report("MISSING", label)
    duplicates = duplicate_key_rows(pq.read_table(path, columns=columns).to_pandas(), columns)
    return report(verdict(duplicates == 0), label, f"duplicate_rows={duplicates}")


def check_disk_index() -> bool:
    path = ROOT / TENCENT_DISK_INDEX
    label = f"validate_disk_index_frame {TENCENT_DISK_INDEX}"
    if not path.exists():
        return report("MISSING", label)
    try:
        contracts.validate_disk_index_frame(pd.read_parquet(path))
    except contracts.ContractViolation as err:
        return report("FAIL", label, str(err))
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


def check_profiles() -> bool:
    ok = True
    for name in PROFILE_NAMES:
        frames = {}
        for rel, read in ((PROFILE_CSV[name], pd.read_csv), (PROFILE_PARQUET[name], pd.read_parquet)):
            path = ROOT / rel
            if not path.exists():
                ok = report("MISSING", f"replay contract {rel}") and ok
                continue
            frames[rel] = read(path)
            ok = check_profile_frame(rel, frames[rel]) and ok
        if len(frames) == 2:
            csv_frame, parquet_frame = frames.values()
            same = list(csv_frame.columns) == list(parquet_frame.columns) and csv_frame.equals(parquet_frame)
            ok = report(verdict(same), f"CSV equals Parquet profile_{name}") and ok
    return ok


def check_profile_references(manifest: dict, preprocessing: str) -> bool:
    ok = True
    listed = {e["file"] for e in manifest.get("files", [])}
    for rel in (PROFILE_PLOT, PROFILE_DISKS, PROFILE_CANDIDATES):
        label = f"referenced {rel}"
        if not (ROOT / rel).exists():
            ok = report("MISSING", label) and ok
            continue
        problems = []
        if rel not in listed:
            problems.append("not listed in manifest.json")
        if Path(rel).name not in preprocessing:
            problems.append("not named in PREPROCESSING.md")
        ok = report(verdict(not problems), label, "; ".join(problems)) and ok

    disks_path = ROOT / PROFILE_DISKS
    profiles = manifest.get("profiles", {})
    label = "manifest profile disk counts vs tencent_profile_disks.parquet"
    if not disks_path.exists():
        return report("MISSING", label) and ok
    counts = pd.read_parquet(disks_path)["profile"].value_counts().to_dict()
    expected = {n: profiles.get(n, {}).get("n_disks") for n in PROFILE_NAMES}
    actual = {n: counts.get(n) for n in PROFILE_NAMES}
    return report(verdict(expected == actual), label, f"manifest={expected} table={actual}") and ok


def main() -> int:
    results = [check_manifest_exists(), check_preprocessing_exists()]
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    preprocessing = PREPROCESSING.read_text(encoding="utf-8") if PREPROCESSING.exists() else ""
    recorded = recorded_rows(preprocessing)

    results += [
        check_manifest_lists_outputs(manifest),
        check_manifest_hashes(manifest),
        check_row_counts(BACKBLAZE_FILES, recorded),
        check_unique_key(BACKBLAZE_FILTERED, backblaze_extract.KEY),
        check_unique_key(BACKBLAZE_TIMELINES, backblaze_timelines.KEY),
        check_unique_key(BACKBLAZE_DRIVE_INDEX, ["serial_number"]),
        check_row_counts(TENCENT_LOAD_FILES, recorded),
        check_unique_key(TENCENT_SUMMARY, ["disk_uid"]),
        check_disk_index(),
        check_row_counts([TENCENT_DISK_INDEX], recorded),
        check_profiles(),
        check_row_counts([f for f in PROFILE_FILES if f != PROFILE_PLOT], recorded),
        check_profile_references(manifest, preprocessing),
    ]
    ok = all(results)
    print("\nRESULT:", "PASS" if ok else "INCOMPLETE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
