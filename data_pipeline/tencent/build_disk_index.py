"""
B4: per-disk metadata index for the Tencent virtual disks.

Reads the headerless subscription table, keeps the first row of every
repeated `disk_uid`, and writes one row per unique disk with the columns of
the contract's DISK_INDEX_COLUMNS (source field order). The table is kept
separate from the load traces and is joined on `disk_uid` only where needed.

Cross-checks against the cleaned load table: ids present on only one side are
counted and listed, never dropped. Ids whose repeated rows disagree are
compared with the ones recorded in the B3 section of PREPROCESSING.md.

Usage:
    python -I build_disk_index.py [dataset_root]

<dataset_root> holds `disk_subscript_info`.

Outputs:
    data/processed/tencent_disk_index.parquet
    manifest.json (entry upserted)
    PREPROCESSING.md (B4 section rewritten)
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "raw" / "tencent" / "Cloud_Disk_dataset"
PROCESSED = ROOT / "data" / "processed"
CLEAN = PROCESSED / "tencent_load_clean.parquet"
OUT_INDEX = PROCESSED / "tencent_disk_index.parquet"
MANIFEST = ROOT / "manifest.json"
PREPROCESSING = ROOT / "PREPROCESSING.md"

# Equal to contracts.contracts.DISK_INDEX_COLUMNS and to the source field order.
DISK_INDEX_COLUMNS = ("disk_uid", "disk_capacity", "disk_attr", "disk_type",
                      "user_type", "vm_cpu", "vm_memory")
VALUE_COLUMNS = DISK_INDEX_COLUMNS[1:]
LISTED_MISMATCHES = 10


def read_subscription(source) -> pd.DataFrame:
    sub = pd.read_csv(source, header=None, names=list(DISK_INDEX_COLUMNS))
    bad = [c for c in VALUE_COLUMNS if not pd.api.types.is_integer_dtype(sub[c])]
    if bad:
        raise SystemExit(f"subscription: non-integer columns {bad}")
    sub["disk_uid"] = sub["disk_uid"].astype("string")
    return sub.astype({c: "int64" for c in VALUE_COLUMNS})


def dedupe_subscription(sub: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep the first row per disk_uid; report repeated and disagreeing ids."""
    repeated = sub[sub["disk_uid"].duplicated(keep=False)]
    conflicting = sorted(
        uid for uid, g in repeated.groupby("disk_uid")
        if len(g[list(VALUE_COLUMNS)].drop_duplicates()) > 1
    )
    index = sub.drop_duplicates(subset="disk_uid", keep="first").reset_index(drop=True)
    info = {
        "rows_in": len(sub),
        "rows_out": len(index),
        "duplicate_rows_dropped": len(sub) - len(index),
        "repeated_uids": int(repeated["disk_uid"].nunique()),
        "conflicting_uids": conflicting,
    }
    return index, info


def validate_index(index: pd.DataFrame) -> None:
    if tuple(index.columns) != DISK_INDEX_COLUMNS:
        raise ValueError(f"columns {tuple(index.columns)} != {DISK_INDEX_COLUMNS}")
    if index.isna().any().any():
        raise ValueError("null values in disk index")
    if index["disk_uid"].duplicated().any():
        raise ValueError("duplicate disk_uid in disk index")
    bad = [c for c in VALUE_COLUMNS if str(index[c].dtype) != "int64"]
    if bad:
        raise ValueError(f"columns not int64: {bad}")


def documented_conflicting_uids(preprocessing_text: str) -> list[str]:
    """Conflicting ids recorded as `conflicting_uid=` lines in the B3 section."""
    m = re.search(r"## B3 — Tencent extract\n.*?(?=\n## B4 )", preprocessing_text, flags=re.S)
    return sorted(re.findall(r"conflicting_uid=([0-9A-Za-z-]+)", m.group(0))) if m else []


def clean_disk_uids(path: Path) -> set[str]:
    """Distinct disk_uid values of the clean load table, read from row-group statistics."""
    pf = pq.ParquetFile(path)
    col = pf.schema_arrow.get_field_index("disk_uid")
    uids: set[str] = set()
    for i in range(pf.num_row_groups):
        stats = pf.metadata.row_group(i).column(col).statistics
        if stats is not None and stats.has_min_max and stats.min == stats.max:
            uids.add(stats.min)
        else:
            uids.update(pf.read_row_group(i, columns=["disk_uid"]).column(0).unique().to_pylist())
    return uids


def compare_ids(subscription: set[str], load: set[str]) -> tuple[list[str], list[str]]:
    return sorted(subscription - load), sorted(load - subscription)


def sha256_of(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def index_table(index: pd.DataFrame) -> pa.Table:
    schema = pa.schema([pa.field("disk_uid", pa.string())]
                       + [pa.field(c, pa.int64()) for c in VALUE_COLUMNS])
    return pa.Table.from_pandas(index.astype({"disk_uid": "object"}), schema=schema,
                                preserve_index=False)


def build(root: Path) -> dict:
    if not CLEAN.exists():
        raise SystemExit(f"missing {CLEAN.name}")
    sub_raw = (root / "disk_subscript_info").read_bytes()
    index, info = dedupe_subscription(read_subscription(io.BytesIO(sub_raw)))
    validate_index(index)

    sub_only, load_only = compare_ids(set(index["disk_uid"]), clean_disk_uids(CLEAN))
    documented = documented_conflicting_uids(PREPROCESSING.read_text(encoding="utf-8"))

    PROCESSED.mkdir(parents=True, exist_ok=True)
    tmp = OUT_INDEX.with_suffix(".parquet.tmp")
    pq.write_table(index_table(index), tmp)
    tmp.replace(OUT_INDEX)
    return {
        **info,
        "subscription_only": sub_only,
        "load_only": load_only,
        "documented_conflicting_uids": documented,
        "output_sha256": sha256_of(OUT_INDEX),
    }


def report_lines(r: dict) -> list[str]:
    observed, documented = r["conflicting_uids"], r["documented_conflicting_uids"]
    lines = [
        f"subscription_rows_in={r['rows_in']}",
        f"duplicate_rows_dropped={r['duplicate_rows_dropped']} repeated_uids={r['repeated_uids']}",
        f"disk_index_rows={r['rows_out']}",
        f"conflicting_uids={len(observed)} documented_in_b3={len(documented)} "
        f"match={observed == documented}",
    ]
    lines += [f"conflicting_uid={uid}" for uid in observed]
    lines.append(f"subscription_only_uids={len(r['subscription_only'])} "
                 f"load_only_uids={len(r['load_only'])}")
    lines += [f"subscription_only_uid={u}" for u in r["subscription_only"][:LISTED_MISMATCHES]]
    lines += [f"load_only_uid={u}" for u in r["load_only"][:LISTED_MISMATCHES]]
    lines.append(f"output={OUT_INDEX.relative_to(ROOT).as_posix()} "
                 f"rows={r['rows_out']} sha256={r['output_sha256']}")
    return lines


def update_preprocessing(lines: list[str]) -> None:
    body = "\n".join(f"- `{ln}`" for ln in lines)
    section = (
        "## B4 — Metadata join\n\n"
        "- Output `data/processed/tencent_disk_index.parquet`: one row per unique `disk_uid`, "
        "columns `disk_uid, disk_capacity, disk_attr, disk_type, user_type, vm_cpu, vm_memory` "
        "(`DISK_INDEX_COLUMNS`, the dataset README field order); `disk_uid` is a string, "
        "the other columns int64\n"
        "- Stored as a separate disk-index file, not merged into the B3 replay files "
        "(see CLAUDE.md §4, B3 vs B4 note); join on `disk_uid`\n"
        "- Duplicate `disk_uid` rule: the first row in file order is kept. "
        "`conflicting_uid` lines below are ids whose repeated rows disagree; they are "
        "compared with the `conflicting_uid` lines of the B3 section\n"
        "- Ids present on one side only (subscription vs `tencent_load_clean.parquet`) are "
        "counted and listed (first "
        f"{LISTED_MISMATCHES} per side); no row or disk is dropped for a mismatch\n"
        "- Counts from the last run of `tencent/build_disk_index.py`:\n\n"
        f"{body}\n\n"
    )
    text = PREPROCESSING.read_text(encoding="utf-8")
    new, n = re.subn(r"## B4 — Metadata join\n.*?(?=\n## B5 )", section.rstrip("\n") + "\n",
                     text, count=1, flags=re.S)
    if n != 1:
        raise SystemExit("B4 section not found in PREPROCESSING.md")
    PREPROCESSING.write_text(new, encoding="utf-8")


def update_manifest(r: dict) -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rel = OUT_INDEX.relative_to(ROOT).as_posix()
    entry = {"file": rel, "rows": r["rows_out"], "sha256": r["output_sha256"]}
    data["files"] = [e for e in data.get("files", []) if e.get("file") != rel] + [entry]
    MANIFEST.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else DEFAULT_DATASET
    r = build(root)
    lines = report_lines(r)
    print("\n".join(lines))
    update_preprocessing(lines)
    update_manifest(r)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
