"""
B0 helper (not a B3 deliverable): inspect a sample Tencent subscription file
and a sample load-trace file to confirm the column names and units on
PPT slide 16 before writing the real B3/B4 parser.

Read-only. Does not filter, convert, or write anything.

Usage:
    python inspect_schema.py <subscription_csv> <one_load_trace_csv>
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def inspect(sub_path: Path, load_path: Path) -> None:
    sub = pd.read_csv(sub_path, nrows=20)
    print(f"Subscription file: {sub_path.name}")
    print(f"  columns: {list(sub.columns)}")
    print(f"  sample row:\n{sub.iloc[0].to_dict()}\n")

    load = pd.read_csv(load_path, nrows=20)
    print(f"Load trace file: {load_path.name}")
    print(f"  columns: {list(load.columns)}")
    print(f"  sample row:\n{load.iloc[0].to_dict()}\n")

    if "timestamp" in load.columns and len(load) > 1:
        deltas = load["timestamp"].diff().dropna().unique()
        print(f"  timestamp deltas seen (expect 300s): {sorted(deltas)[:10]}")

    print(
        "\nUNRESOLVED before B3/B4 can be finalized (see CLAUDE.md Open Questions):\n"
        "  - bandwidth columns appear to be KB/s, not MB/s — confirm, and confirm\n"
        "    who performs the KB/s -> MB/s conversion (proposal: the simulator-side\n"
        "    adapter, not this pipeline).\n"
        "  - t_seconds origin: epoch timestamp, or seconds since window start?\n"
        "  - how does one virtual disk's load map onto the simulated group's scale?\n"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    inspect(Path(sys.argv[1]), Path(sys.argv[2]))
