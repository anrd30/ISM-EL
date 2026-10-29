"""
Read-only schema check for the Tencent Cloud Virtual Disk dataset.

Both source files are headerless CSV; column names are assigned from the
dataset README field order. Nothing is filtered, converted, or written.

Usage:
    python -I inspect_schema.py <dataset_root> [n_sample_files]

<dataset_root> contains `disk_subscript_info` and `disk_load_data/`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SUB_COLUMNS = [
    "disk_uid", "disk_capacity", "disk_attr", "disk_type",
    "user_type", "vm_cpu", "vm_memory",
]
LOAD_COLUMNS = [
    "timestamp", "read_iops", "read_bw_kbps",
    "write_iops", "write_bw_kbps", "disk_usage_mb",
]
SAMPLE_STEP_S = 300


def inspect(root: Path, n_sample: int) -> None:
    sub = pd.read_csv(root / "disk_subscript_info", header=None, names=SUB_COLUMNS)
    print(f"subscription rows={len(sub)} unique_disk_uid={sub['disk_uid'].nunique()}")
    print(sub.dtypes.to_string())
    print(sub.drop(columns="disk_uid").describe().T.to_string())

    load_dir = root / "disk_load_data"
    files = sorted(p for p in load_dir.iterdir() if p.is_file() and not p.name.startswith("."))
    uids_sub = set(sub["disk_uid"])
    uids_load = {p.name for p in files}
    print(f"\nload files={len(files)}")
    print(f"in_subscription_only={len(uids_sub - uids_load)} in_load_only={len(uids_load - uids_sub)}")

    step = max(1, len(files) // n_sample)
    for p in files[::step][:n_sample]:
        df = pd.read_csv(p, header=None, names=LOAD_COLUMNS)
        deltas = sorted(int(d) for d in df["timestamp"].diff().dropna().unique())
        print(
            f"{p.name} rows={len(df)} ts=[{df['timestamp'].min()},{df['timestamp'].max()}] "
            f"deltas={deltas[:5]} nulls={int(df.isna().sum().sum())} "
            f"read_bw_kbps_max={df['read_bw_kbps'].max()} write_bw_kbps_max={df['write_bw_kbps'].max()}"
        )


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    inspect(Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) == 3 else 5)
