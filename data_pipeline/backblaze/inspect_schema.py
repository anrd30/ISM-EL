"""
Inspect the column set of each daily Backblaze file in a downloaded quarter.

Read-only. Does not filter, convert, or write anything.

Usage:
    python inspect_schema.py <path-to-quarter-dir> [--failures]

--failures reads model+failure from every file in the quarter (not a
5-file sample) and reports total failure rows per model.

Expects a directory of daily CSV files as distributed by Backblaze
(https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def count_failures(files: list[Path]) -> None:
    drive_days: dict[str, int] = {}
    failures: dict[str, int] = {}
    for f in files:
        df = pd.read_csv(f, usecols=["model", "failure"])
        for m, n in df["model"].value_counts().items():
            drive_days[m] = drive_days.get(m, 0) + int(n)
        fdf = df[df["failure"] == 1]
        for m, n in fdf["model"].value_counts().items():
            failures[m] = failures.get(m, 0) + int(n)

    print(f"\nFailure counts across all {len(files)} files, by model:")
    models_with_failures = sorted(failures.items(), key=lambda kv: -kv[1])
    for m, n in models_with_failures:
        print(f"  {m}: {n} failures / {drive_days.get(m, 0)} drive-days")
    zero_failure_models = sorted(set(drive_days) - set(failures))
    print(f"\nModels with 0 failures in this quarter ({len(zero_failure_models)}):")
    for m in zero_failure_models:
        print(f"  {m}: {drive_days[m]} drive-days")


def inspect_quarter(quarter_dir: Path, do_failures: bool) -> None:
    files = sorted(quarter_dir.glob("*.csv"))
    if not files:
        print(f"No CSV files found in {quarter_dir}")
        return

    print(f"{len(files)} daily files found in {quarter_dir}\n")

    all_columns: set[str] = set()
    per_file_columns: dict[str, set[str]] = {}

    # Only read headers — these files can be large.
    for f in files:
        cols = set(pd.read_csv(f, nrows=0).columns)
        per_file_columns[f.name] = cols
        all_columns |= cols

    stable = set.intersection(*per_file_columns.values()) if per_file_columns else set()
    varying = all_columns - stable

    print(f"Columns present in EVERY file ({len(stable)}):")
    for c in sorted(stable):
        print(f"  {c}")

    print(f"\nColumns present in SOME but not all files ({len(varying)}):")
    for c in sorted(varying):
        present_in = sum(1 for cols in per_file_columns.values() if c in cols)
        print(f"  {c}  (in {present_in}/{len(files)} files)")

    if "model" in stable:
        models: dict[str, int] = {}
        for f in files[:5]:  # sample of 5 files only, not exhaustive
            df = pd.read_csv(f, usecols=["model"])
            for m, n in df["model"].value_counts().items():
                models[m] = models.get(m, 0) + int(n)
        print(f"\nDrive-model counts (sample: {min(5, len(files))}/{len(files)} files):")
        for m, n in sorted(models.items(), key=lambda kv: -kv[1]):
            print(f"  {m}: {n}")

    if do_failures and "failure" in stable:
        count_failures(files)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    inspect_quarter(Path(sys.argv[1]), do_failures="--failures" in sys.argv)
