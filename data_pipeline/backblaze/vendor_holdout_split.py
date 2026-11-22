"""
A2v: a drive-level split with one drive model held out entirely (vendor holdout).

Parallel to A2 (`labels.py`), which is not changed and whose outputs are not read for writing.
One row per drive of the drive index: serial_number, model, ever_failed, split with split in
{train, val, test, vendor_holdout}.

- Every drive of `HOLDOUT_MODEL` (TOSHIBA MG07ACA14TA) is in `vendor_holdout`: none of its drives
  is in train, val or test.
- The other drive models are split by `labels.assign_splits` called on their drives only, with the
  A2 seed and fractions (70 / 15 / 15). The strata are (model, ever_failed) and the holdout model's
  strata sort after the others, so the assignment of the other drives equals A2's (checked by a test
  against the committed A2 split).
- All rows of a drive belong to one bucket; the split is by drive, never by row or date.

The timelines and labels tables are the A2 ones, unchanged: the trailing-window features of
`timelines.py` are computed per drive from its own past, so the holdout does not change any feature
of another drive (`test_vendor_holdout_split.py`).

Bucket statistics (drives, failed drives and label rows per bucket and model) are written to
PREPROCESSING.md (section A2v) and the output is registered in manifest.json.

Usage:
    python -m data_pipeline.backblaze.vendor_holdout_split

Output:
    data/processed/backblaze_drive_split_vendor_holdout.parquet
    manifest.json (one entry upserted)
    PREPROCESSING.md (A2v section rewritten or appended)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

from data_pipeline.backblaze import labels as a2  # noqa: E402
from data_pipeline.backblaze import timelines  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
IN_INDEX = PROCESSED / "backblaze_drive_index.parquet"
IN_LABELS = PROCESSED / "backblaze_labels.parquet"
OUT_SPLIT = PROCESSED / "backblaze_drive_split_vendor_holdout.parquet"
MANIFEST = ROOT / "manifest.json"
PREPROCESSING = ROOT / "PREPROCESSING.md"

HOLDOUT_MODEL = "TOSHIBA MG07ACA14TA"
HOLDOUT_BUCKET = "vendor_holdout"
BUCKETS = (*a2.SPLITS, HOLDOUT_BUCKET)
SECTION = "## A2v — Vendor-holdout drive split"


def assign_vendor_holdout(index: pd.DataFrame, holdout_model: str = HOLDOUT_MODEL) -> pd.DataFrame:
    """One row per drive: the drives of ``holdout_model`` in ``vendor_holdout``, the others split as A2 does."""
    models = index["model"].astype(object)
    if holdout_model not in set(models):
        raise ValueError(f"holdout model {holdout_model!r} is not in the drive index")
    fitted = a2.assign_splits(index[models != holdout_model])
    held = index[models == holdout_model]
    held = pd.DataFrame({
        "serial_number": held["serial_number"].astype(object).to_numpy(),
        "model": held["model"].astype(object).to_numpy(),
        "ever_failed": held["failed_in_window"].astype(bool).to_numpy(),
        "split": HOLDOUT_BUCKET,
    })
    out = pd.concat([fitted, held], ignore_index=True)
    return out.sort_values("serial_number", kind="stable").reset_index(drop=True)[a2.SPLIT_COLUMNS]


def bucket_statistics(labels_path: Path, splits: pd.DataFrame) -> dict[tuple[str, str], dict[str, int]]:
    """Drives, failed drives and label rows (positive, negative, censored) per (bucket, model), plus a model "all" row."""
    lookup = pd.Index(splits["serial_number"].astype(object))
    bucket_of = splits["split"].astype(object).to_numpy()
    names = {timelines.LABEL_POSITIVE: "positive", timelines.LABEL_NEGATIVE: "negative",
             timelines.LABEL_CENSORED: "censored"}
    rows: dict[tuple[str, str], dict[str, int]] = {}
    pf = pq.ParquetFile(labels_path)
    for i in range(pf.num_row_groups):
        frame = pf.read_row_group(i, columns=["serial_number", "model", "label"]).to_pandas()
        where = lookup.get_indexer(frame["serial_number"].astype(object))
        if (where < 0).any():
            raise ValueError("label rows for drives missing from the split table")
        frame["bucket"] = bucket_of[where]
        frame["model"] = frame["model"].astype(object)
        for (bucket, model, label), n in frame.groupby(["bucket", "model", "label"]).size().items():
            cell = rows.setdefault((bucket, model), dict.fromkeys(names.values(), 0))
            cell[names[int(label)]] += int(n)
    models = sorted(set(splits["model"].astype(object)))
    out: dict[tuple[str, str], dict[str, int]] = {}
    for bucket in BUCKETS:
        for model in models:
            part = splits[(splits["split"] == bucket) & (splits["model"].astype(object) == model)]
            out[(bucket, model)] = {"drives": len(part), "failed_drives": int(part["ever_failed"].sum()),
                                    **rows.get((bucket, model), dict.fromkeys(names.values(), 0))}
        out[(bucket, "all")] = {k: sum(out[(bucket, m)][k] for m in models)
                                for k in ("drives", "failed_drives", "positive", "negative", "censored")}
    return out


def report_lines(stats: dict[tuple[str, str], dict[str, int]]) -> list[str]:
    lines = []
    for (bucket, model), c in stats.items():
        lines.append(f"split={bucket} model={model} drives={c['drives']} failed_drives={c['failed_drives']} "
                     f"positive_rows={c['positive']} negative_rows={c['negative']} censored_rows={c['censored']}")
    return lines


def update_manifest(path: Path, splits: pd.DataFrame) -> dict:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entry = {"file": path.relative_to(ROOT).as_posix(), "rows": len(splits), "sha256": a2.sha256_of(path)}
    data["files"] = [e for e in data.get("files", []) if e.get("file") != entry["file"]] + [entry]
    MANIFEST.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    return entry


def update_preprocessing(lines: list[str], entry: dict) -> None:
    body = "\n".join(f"- `{ln}`" for ln in lines)
    section = (
        f"{SECTION}\n\n"
        f"- A drive-level split parallel to A2 with one drive model held out entirely: every drive of `{HOLDOUT_MODEL}` is in "
        f"`{HOLDOUT_BUCKET}`; the other models are split by `labels.assign_splits` (A2 seed and fractions) on their drives only, "
        "which reproduces A2's assignment of those drives. A2's outputs are unchanged. Script `backblaze/vendor_holdout_split.py`; "
        "tests `tests/test_vendor_holdout_split.py`\n"
        f"- Output `{entry['file']}` rows={entry['rows']} sha256={entry['sha256']}\n"
        "- Counts per bucket and model (label rows are the rows of `backblaze_labels.parquet`; censored rows are not used for fitting "
        "or evaluation):\n\n" + body + "\n\n"
    )
    text = PREPROCESSING.read_text(encoding="utf-8")
    pattern = re.escape(SECTION) + r"\n.*?(?=\n## |\Z)"
    if re.search(pattern, text, flags=re.S):
        new = re.sub(pattern, lambda _m: section.rstrip("\n"), text, count=1, flags=re.S)
    else:
        new = text.rstrip("\n") + "\n\n" + section.rstrip("\n") + "\n"
    PREPROCESSING.write_text(new, encoding="utf-8", newline="\n")


def main() -> int:
    index = pd.read_parquet(IN_INDEX)
    splits = assign_vendor_holdout(index)
    stats = bucket_statistics(IN_LABELS, splits)
    lines = report_lines(stats)
    print("\n".join(lines))
    staging = a2.tmp_path(OUT_SPLIT)
    pq.write_table(pa.Table.from_pandas(splits, preserve_index=False), staging)
    staging.replace(OUT_SPLIT)
    entry = update_manifest(OUT_SPLIT, splits)
    update_preprocessing(lines, entry)
    print(f"output={OUT_SPLIT} rows={len(splits)} sha256={entry['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
