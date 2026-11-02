"""
A2: stored 30-day horizon labels and a drive-level train/validation/test split.

labels (one row per timelines row, same order and (serial_number, date) key):
serial_number, date, model, label with label 1 positive, 0 negative, -1 censored.
The label rule is the one in timelines.py (row_labels / classify_rows); it is
imported, not reimplemented. The timelines table is read in row groups and only
serial_number and date are loaded.

split (one row per drive of the drive index): serial_number, model,
ever_failed (failed_in_window), split in {train, val, test}. All rows of a drive
belong to one split. Drives are sorted by serial_number and shuffled with a seeded
generator inside each stratum (model, ever_failed); each stratum is divided with
largest-remainder rounding so every split size is within one drive of its target
share. The split is not time-forward: drives span the whole window, so the splits
overlap in dates.

The label counts are compared with the recorded values and a difference aborts the
run before either output replaces a previous file. Both outputs are written
to <name>.tmp files and renamed after every step has succeeded.

Usage:
    python -I labels.py

Outputs:
    data/processed/backblaze_labels.parquet
    data/processed/backblaze_drive_split.parquet
    manifest.json (two entries upserted)
    PREPROCESSING.md (A2 section rewritten or appended)
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))  # repo root: `data_pipeline` package, also under `python -I`

from data_pipeline.backblaze import timelines  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
IN_TIMELINES = PROCESSED / "backblaze_timelines.parquet"
IN_INDEX = PROCESSED / "backblaze_drive_index.parquet"
OUT_LABELS = PROCESSED / "backblaze_labels.parquet"
OUT_SPLIT = PROCESSED / "backblaze_drive_split.parquet"
MANIFEST = ROOT / "manifest.json"
PREPROCESSING = ROOT / "PREPROCESSING.md"

SPLITS = ("train", "val", "test")
FRACTIONS = (0.70, 0.15, 0.15)
SPLIT_SEED = 20220101
LABEL_VALUES = (timelines.LABEL_CENSORED, timelines.LABEL_NEGATIVE, timelines.LABEL_POSITIVE)

# Counts produced by timelines.py on the 2022 extract (PREPROCESSING.md, B2).
EXPECTED_ROWS = 33_052_284
EXPECTED_TOTALS = {"positive": 52_387, "negative": 30_370_344, "censored": 2_629_553}
EXPECTED_BY_MODEL = {
    "ST4000DM000": {"positive": 18_873, "negative": 6_132_680, "censored": 529_516},
    "ST8000NM0055": {"positive": 10_077, "negative": 4_787_720, "censored": 417_895},
    "ST12000NM0008": {"positive": 12_005, "negative": 6_685_135, "censored": 574_934},
    "TOSHIBA MG07ACA14TA": {"positive": 11_432, "negative": 12_764_809, "censored": 1_107_208},
}

LABEL_SCHEMA = pa.schema([
    pa.field("serial_number", pa.string()),
    pa.field("date", pa.date32()),
    pa.field("model", pa.string()),
    pa.field("label", pa.int8()),
])
LABEL_KEY = ["serial_number", "date"]
SPLIT_COLUMNS = ["serial_number", "model", "ever_failed", "split"]


def sha256_of(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def tmp_path(path: Path) -> Path:
    return path.with_name(path.name + ".tmp")


def build_labels(timelines_path: Path, index_path: Path, out_path: Path,
                 check: Callable[[dict], None] | None = None) -> dict:
    """Write the label table to `out_path` (a temporary file path chosen by the caller).

    Returns {rows, totals, by_model}. `check(result)` runs after the last row group
    and aborts by raising."""
    index = pd.read_parquet(index_path)
    lookup = pd.Index(index["serial_number"].astype(object))
    model_of = index["model"].astype(object).to_numpy()
    pf = pq.ParquetFile(timelines_path)
    rows = 0
    by_model: dict[str, dict[str, int]] = {}
    with pq.ParquetWriter(out_path, LABEL_SCHEMA) as writer:
        for i in range(pf.num_row_groups):
            batch = pf.read_row_group(i, columns=["serial_number", "date"])
            frame = batch.to_pandas()
            labels = timelines.row_labels(frame, index)
            where = lookup.get_indexer(frame["serial_number"].astype(object))
            models = model_of[where]
            table = pa.table({
                "serial_number": batch.column("serial_number").cast(pa.string()),
                "date": batch.column("date").cast(pa.date32()),
                "model": pa.array(models, type=pa.string()),
                "label": pa.array(labels, type=pa.int8()),
            }, schema=LABEL_SCHEMA)
            writer.write_table(table)
            rows += len(table)
            for model in np.unique(models):
                counts = timelines.label_counts(labels[models == model])
                bucket = by_model.setdefault(str(model), dict.fromkeys(counts, 0))
                for key, value in counts.items():
                    bucket[key] += value
    totals = {k: sum(m[k] for m in by_model.values()) for k in ("positive", "negative", "censored")}
    result = {"rows": rows, "totals": totals, "by_model": by_model}
    if check is not None:
        check(result)
    return result


def verify_expected(result: dict) -> None:
    """Compare label counts with the recorded values; SystemExit on any difference."""
    problems = []

    def compare(name: str, expected, measured) -> None:
        if expected != measured:
            problems.append(f"{name}: expected {expected}, measured {measured}")

    compare("rows", EXPECTED_ROWS, result["rows"])
    compare("totals", EXPECTED_TOTALS, result["totals"])
    for model, counts in EXPECTED_BY_MODEL.items():
        compare(model, counts, result["by_model"].get(model))
    extra = sorted(set(result["by_model"]) - set(EXPECTED_BY_MODEL))
    if extra:
        problems.append(f"models not in the recorded values: {extra}")
    if problems:
        raise SystemExit("label verification failed:\n" + "\n".join(problems))


def largest_remainder(n: int, fractions: tuple[float, ...] = FRACTIONS) -> list[int]:
    """Integer shares of n that sum to n, each within one of n * fraction."""
    quotas = [n * f for f in fractions]
    shares = [int(np.floor(q + 1e-9)) for q in quotas]
    order = sorted(range(len(fractions)), key=lambda k: (-(quotas[k] - shares[k]), k))
    for k in order[: n - sum(shares)]:
        shares[k] += 1
    return shares


def assign_splits(index: pd.DataFrame, seed: int = SPLIT_SEED,
                  fractions: tuple[float, ...] = FRACTIONS) -> pd.DataFrame:
    """One row per drive of the drive index with its split."""
    drives = pd.DataFrame({
        "serial_number": index["serial_number"].astype(object).to_numpy(),
        "model": index["model"].astype(object).to_numpy(),
        "ever_failed": index["failed_in_window"].astype(bool).to_numpy(),
    }).sort_values("serial_number", kind="stable").reset_index(drop=True)
    drives["split"] = ""
    for k, (_, group) in enumerate(drives.groupby(["model", "ever_failed"], sort=True)):
        rng = np.random.default_rng([seed, k])
        position = group.index.to_numpy()[rng.permutation(len(group))]
        start = 0
        for name, share in zip(SPLITS, largest_remainder(len(group), fractions)):
            drives.loc[position[start:start + share], "split"] = name
            start += share
    return drives[SPLIT_COLUMNS]


def split_statistics(labels_path: Path, splits: pd.DataFrame) -> dict:
    """Per (split, model): drives, failed drives and label rows; per split: date range and
    positive rows by month. `counts` also holds a model "all" row for every split."""
    lookup = pd.Index(splits["serial_number"].astype(object))
    split_of = splits["split"].astype(object).to_numpy()
    label_rows: dict[tuple[str, str], dict[str, int]] = {}
    date_range: dict[str, list] = {}
    positive_months: dict[str, dict[str, int]] = {s: {} for s in SPLITS}
    pf = pq.ParquetFile(labels_path)
    names = {timelines.LABEL_POSITIVE: "positive", timelines.LABEL_NEGATIVE: "negative",
             timelines.LABEL_CENSORED: "censored"}
    for i in range(pf.num_row_groups):
        frame = pf.read_row_group(i, columns=["serial_number", "date", "model", "label"]).to_pandas()
        where = lookup.get_indexer(frame["serial_number"].astype(object))
        if (where < 0).any():
            raise ValueError("label rows for drives missing from the split table")
        frame["split"] = split_of[where]
        frame["model"] = frame["model"].astype(object)
        for (split, model, label), n in frame.groupby(["split", "model", "label"]).size().items():
            bucket = label_rows.setdefault((split, model), dict.fromkeys(names.values(), 0))
            bucket[names[int(label)]] += int(n)
        dates = pd.to_datetime(frame["date"])
        for split, bounds in dates.groupby(frame["split"]).agg(["min", "max"]).iterrows():
            current = date_range.setdefault(split, [bounds["min"], bounds["max"]])
            current[0], current[1] = min(current[0], bounds["min"]), max(current[1], bounds["max"])
        positive = frame["label"] == timelines.LABEL_POSITIVE
        months = dates[positive].dt.strftime("%Y-%m")
        for (split, month), n in months.groupby(frame.loc[positive, "split"]).value_counts().items():
            positive_months[split][month] = positive_months[split].get(month, 0) + int(n)

    counts: dict[tuple[str, str], dict[str, int]] = {}
    models = sorted(set(splits["model"].astype(object)))
    for split in SPLITS:
        for model in models:
            part = splits[(splits["split"] == split) & (splits["model"].astype(object) == model)]
            rows = label_rows.get((split, model), dict.fromkeys(names.values(), 0))
            counts[(split, model)] = {"drives": len(part), "failed_drives": int(part["ever_failed"].sum()), **rows}
        counts[(split, "all")] = {
            key: sum(counts[(split, m)][key] for m in models)
            for key in ("drives", "failed_drives", "positive", "negative", "censored")}
    return {"counts": counts, "models": models,
            "dates": {s: {"min": date_range[s][0].date().isoformat() if s in date_range else "none",
                          "max": date_range[s][1].date().isoformat() if s in date_range else "none",
                          "positive_by_month": dict(sorted(positive_months[s].items()))}
                      for s in SPLITS}}


def report_lines(result: dict, stats: dict) -> list[str]:
    totals = result["totals"]
    lines = [
        f"a2_labels rows={result['rows']} positive={totals['positive']} "
        f"negative={totals['negative']} censored={totals['censored']}",
        f"a2_params seed={SPLIT_SEED} fractions="
        + ",".join(f"{s}:{f:g}" for s, f in zip(SPLITS, FRACTIONS)) + " strata=model,ever_failed",
    ]
    for model in stats["models"] + ["all"]:
        for split in SPLITS:
            c = stats["counts"][(split, model)]
            lines.append(f"a2_split split={split} model={model} drives={c['drives']} "
                         f"failed_drives={c['failed_drives']} positive={c['positive']} "
                         f"negative={c['negative']} censored={c['censored']}")
    for split in SPLITS:
        d = stats["dates"][split]
        months = ",".join(f"{m}:{n}" for m, n in d["positive_by_month"].items())
        lines.append(f"a2_split_dates split={split} date_min={d['min']} date_max={d['max']} "
                     f"positive_by_month={months}")
    return lines


def update_preprocessing(lines: list[str]) -> None:
    body = "\n".join(f"- `{ln}`" for ln in lines)
    section = (
        "## A2 — Horizon labels and drive split\n\n"
        "- Inputs: `data/processed/backblaze_timelines.parquet`, "
        "`data/processed/backblaze_drive_index.parquet`; decision record: `data_pipeline/A2_DECISION.md`\n"
        "- Label rule: the 30-day rule of section B2 (`timelines.row_labels`), imported unchanged. "
        "`backblaze_labels.parquet` holds `serial_number`, `date`, `model` and `label` "
        "(1 positive, 0 negative, -1 censored) for every timelines row, in the timelines row order\n"
        "- Split rule: unit = drive; 70/15/15 train/val/test; drives sorted by `serial_number` and shuffled "
        f"with a seeded generator (seed {SPLIT_SEED}) inside each stratum (`model`, `ever_failed`); "
        "largest-remainder rounding per stratum. `backblaze_drive_split.parquet` has one row per drive "
        "of the drive index, including the fully censored drives\n"
        "- The split is not time-forward: drives span the whole window, so the three splits overlap in "
        "dates (`a2_split_dates` lines give the date range and the positive rows by month of each split)\n"
        "- Censored rows (label -1) are unknown outcomes and are excluded from training and scoring\n"
        "- Counts from the last run of `backblaze/labels.py`:\n\n"
        f"{body}\n"
    )
    text = PREPROCESSING.read_text(encoding="utf-8")
    new, n = re.subn(r"## A2 — Horizon labels and drive split\n.*?(?=\n## |\n?\Z)", section.rstrip("\n"),
                     text, count=1, flags=re.S)
    if n == 0:
        new = text.rstrip("\n") + "\n\n" + section.rstrip("\n") + "\n"
    PREPROCESSING.write_text(new, encoding="utf-8")


def update_manifest(entries: list[dict]) -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    names = {e["file"] for e in entries}
    data["files"] = [e for e in data.get("files", []) if e.get("file") not in names] + entries
    MANIFEST.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    tmp_labels, tmp_split = tmp_path(OUT_LABELS), tmp_path(OUT_SPLIT)
    try:
        PROCESSED.mkdir(parents=True, exist_ok=True)
        result = build_labels(IN_TIMELINES, IN_INDEX, tmp_labels,
                              check=verify_expected)
        splits = assign_splits(pd.read_parquet(IN_INDEX))
        pq.write_table(pa.Table.from_pandas(splits, preserve_index=False), tmp_split)
        stats = split_statistics(tmp_labels, splits)
        tmp_labels.replace(OUT_LABELS)
        tmp_split.replace(OUT_SPLIT)
    except BaseException:
        tmp_labels.unlink(missing_ok=True)
        tmp_split.unlink(missing_ok=True)
        raise
    lines = report_lines(result, stats)
    entries = []
    for path, rows in ((OUT_LABELS, result["rows"]), (OUT_SPLIT, len(splits))):
        rel = path.relative_to(ROOT).as_posix()
        digest = sha256_of(path)
        lines.append(f"output={rel} rows={rows} sha256={digest}")
        entries.append({"file": rel, "rows": rows, "sha256": digest})
    print("\n".join(lines))
    update_preprocessing(lines)
    update_manifest(entries)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
