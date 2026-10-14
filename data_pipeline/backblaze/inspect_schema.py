"""
B0 helper (not a B1-6 deliverable): inspect the column set of each daily
Backblaze file in a downloaded quarter, so we know what we're aligning
before A1 (quarter + drive models + SMART columns) is confirmed.

Read-only. Does not filter, convert, or write anything.

Usage:
    python inspect_schema.py <path-to-quarter-dir>

Expects a directory of daily CSV files as distributed by Backblaze
(https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def inspect_quarter(quarter_dir: Path) -> None:
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
        for f in files[:5]:  # sample, not exhaustive — this is a quick look
            df = pd.read_csv(f, usecols=["model"])
            for m, n in df["model"].value_counts().items():
                models[m] = models.get(m, 0) + int(n)
        print("\nSample drive-model counts (first 5 files only, not exhaustive):")
        for m, n in sorted(models.items(), key=lambda kv: -kv[1]):
            print(f"  {m}: {n}")

    print(
        "\nNext: confirm with Aakarsh (A1) which quarter, which 1-2 models, "
        "and which SMART columns to keep, then fill in PREPROCESSING.md "
        "and backblaze/extract.py's constants."
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    inspect_quarter(Path(sys.argv[1]))
