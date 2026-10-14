#!/usr/bin/env python3
"""
B6 lane-acceptance check: "the files load with the exact column names,
row counts match PREPROCESSING.md, and the four profiles are visibly
different when plotted. Checkable in under a minute." (CLAUDE.md §4)

Run from data_pipeline/:
    python check_outputs.py

Currently a scaffold — most checks are stubs that print what they will
verify once B1-B5 produce real output files. Wire each one up as its
task is implemented; keep this file the single command the team runs to
confirm the lane is done.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROFILES_DIR = ROOT / "profiles"
MANIFEST = ROOT / "manifest.json"
PREPROCESSING = ROOT / "PREPROCESSING.md"

EXPECTED_PROFILE_COLUMNS = [
    "t_seconds", "read_iops", "write_iops", "read_bw_kbps", "write_bw_kbps", "disk_usage_mb",
]


def check_manifest_exists() -> bool:
    ok = MANIFEST.exists()
    print(f"[{'OK' if ok else 'MISSING'}] manifest.json exists")
    return ok


def check_preprocessing_exists() -> bool:
    ok = PREPROCESSING.exists()
    print(f"[{'OK' if ok else 'MISSING'}] PREPROCESSING.md exists")
    return ok


def check_profiles() -> bool:
    if not PROFILES_DIR.exists():
        print("[PENDING] profiles/ does not exist yet — B5 not implemented")
        return True  # not a failure yet, just not started
    import pandas as pd

    all_ok = True
    for name in ["idle", "diurnal", "bursty", "peak"]:
        f = PROFILES_DIR / f"profile_{name}.csv"
        if not f.exists():
            print(f"[PENDING] {f.name} not generated yet")
            continue
        df = pd.read_csv(f)
        cols_ok = list(df.columns) == EXPECTED_PROFILE_COLUMNS
        rows_ok = len(df) == 288
        print(f"[{'OK' if cols_ok and rows_ok else 'FAIL'}] {f.name}: "
              f"columns={'ok' if cols_ok else df.columns.tolist()}, rows={len(df)} (expect 288)")
        all_ok = all_ok and cols_ok and rows_ok
    return all_ok


def main() -> int:
    results = [
        check_manifest_exists(),
        check_preprocessing_exists(),
        check_profiles(),
    ]
    ok = all(results)
    print("\nRESULT:", "PASS" if ok else "INCOMPLETE (expected while B1-B6 are still in progress)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
