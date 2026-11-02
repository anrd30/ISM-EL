"""
B2: per-drive timelines from the B1 extract.

Produces two tables.

timelines (one row per drive-day): only backward-looking columns. Every
value on a row is a function of that drive's rows with date <= the row date,
so truncating the input at any date leaves all earlier rows unchanged.

drive index (one row per drive): whole-drive facts (model, first_seen,
last_seen, days_observed, failed_in_window, failure_date). These depend on
the full observation window, are never features, and are joined to the
timelines only at label construction.

Windows are calendar-based. For a window of N days, the trailing difference
of a column is value(today) - value(earliest observed row with
date >= today - N). It is NaN when fewer than MIN_OBS_DIFF rows fall in
[today - N, today] or either endpoint is NaN. No value is imputed. A column
with no value for a model (smart_187_raw and smart_188_raw for the Toshiba
model) gives NaN level and difference features for that model.

Horizon label counts (reported, not stored): for a row on date d of a drive
with last_seen L and failure_date F (if any),
  positive  : F exists and d <= F <= d + HORIZON_DAYS;
  negative  : F is absent or later than d + HORIZON_DAYS, and L >= d + HORIZON_DAYS;
  censored  : every other row. A drive without a recorded failure that has no row
              on or after d + HORIZON_DAYS, whether it left the data or the horizon
              extends past the last date of the input, is censored, not negative.
The first and last date come from the input.

Memory: the input is partitioned by a hash of serial_number into BUCKETS
temporary Parquet files and each bucket is processed on its own, because the
features of one drive depend only on that drive's rows.

Usage:
    python -I timelines.py [filtered_parquet]

Outputs:
    data/processed/backblaze_timelines.parquet
    data/processed/backblaze_drive_index.parquet
    manifest.json (entries upserted)
    PREPROCESSING.md (B2 section rewritten)
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
INPUT = PROCESSED / "backblaze_filtered.parquet"
OUT_TIMELINES = PROCESSED / "backblaze_timelines.parquet"
OUT_INDEX = PROCESSED / "backblaze_drive_index.parquet"
MANIFEST = ROOT / "manifest.json"
PREPROCESSING = ROOT / "PREPROCESSING.md"

SMART_COLUMNS = (
    "smart_5_raw", "smart_9_raw", "smart_12_raw", "smart_187_raw", "smart_188_raw",
    "smart_193_raw", "smart_194_raw", "smart_197_raw", "smart_198_raw", "smart_199_raw",
)
WINDOWS_DAYS = (7, 30)
# smart_188_raw packs several counters, so its level is not comparable across drives.
NO_LEVEL = frozenset({"smart_188_raw"})
WINDOWS_OVERRIDE = {"smart_188_raw": (7,)}
MIN_OBS_DIFF = 2

HORIZON_DAYS = 30
LABEL_POSITIVE, LABEL_NEGATIVE, LABEL_CENSORED = 1, 0, -1
NO_DAY = np.iinfo("int64").max // 4  # failure day of a drive with no recorded failure
BUCKETS = 16

KEY = ["serial_number", "date"]
CORE_IN = ["serial_number", "model", "date", "capacity_bytes", "failure"]
DAY_STRIDE = 1_000_000  # larger than any day ordinal; packs (drive, day) into one sortable int64


def feature_spec() -> list[tuple[str, str, int | None]]:
    """(output column, source column, window days or None for the level)."""
    spec: list[tuple[str, str, int | None]] = []
    for col in SMART_COLUMNS:
        if col not in NO_LEVEL:
            spec.append((col, col, None))
        for n in WINDOWS_OVERRIDE.get(col, WINDOWS_DAYS):
            spec.append((f"{col}_diff{n}d", col, n))
    return spec


def day_ordinals(dates: pd.Series) -> np.ndarray:
    return pd.to_datetime(dates).to_numpy().astype("datetime64[D]").astype("int64")


def build_timelines(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(KEY, kind="stable").reset_index(drop=True)
    n = len(df)
    codes = pd.factorize(df["serial_number"], sort=True)[0].astype("int64")
    day = day_ordinals(df["date"])
    key = codes * DAY_STRIDE + day
    if n and not (np.diff(key) > 0).all():
        raise ValueError("duplicate (serial_number, date) rows")

    pos = np.arange(n)
    start = np.searchsorted(codes, codes, side="left")
    has_prev = pos > start
    prev_day = day[np.maximum(pos - 1, 0)]

    out = pd.DataFrame({
        "serial_number": df["serial_number"].astype("string"),
        "date": pd.to_datetime(df["date"]),
        "model": df["model"].astype("string"),
        "capacity_bytes": df["capacity_bytes"].astype("Int64"),
        "failure": df["failure"].astype("int8"),
        "gap_days_before": pd.array(np.where(has_prev, day - prev_day, 0), dtype="Int32"),
        "days_since_first_seen": (day - day[start]).astype("int32"),
        "first_seen": pd.to_datetime(day[start].astype("datetime64[D]")),
    })
    out.loc[~has_prev, "gap_days_before"] = pd.NA

    window_start: dict[int, np.ndarray] = {}
    features: dict[str, np.ndarray] = {}
    for name, src, win in feature_spec():
        values = df[src].to_numpy(dtype="float64")
        if win is None:
            features[name] = values
            continue
        if win not in window_start:
            window_start[win] = np.searchsorted(key, key - win, side="left")
        idx = window_start[win]
        diff = values - values[idx]
        diff[(pos - idx + 1) < MIN_OBS_DIFF] = np.nan
        features[name] = diff

    for name, arr in features.items():
        out[name] = arr
    for name, arr in features.items():
        out[f"{name}_missing"] = np.isnan(arr)
    return out


def build_drive_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(date=pd.to_datetime(df["date"]))
    g = df.groupby("serial_number", sort=True)
    index = pd.DataFrame({
        "model": g["model"].first(),
        "first_seen": g["date"].min(),
        "last_seen": g["date"].max(),
        "days_observed": g["date"].size().astype("int32"),
    })
    failed = df[df["failure"] == 1].groupby("serial_number")["date"].min()
    index["failure_date"] = failed.reindex(index.index)
    index["failed_in_window"] = index["failure_date"].notna()
    return index.reset_index()[
        ["serial_number", "model", "first_seen", "last_seen",
         "days_observed", "failed_in_window", "failure_date"]
    ]


def classify_rows(day: np.ndarray, failure_day: np.ndarray, last_day: np.ndarray,
                  horizon: int = HORIZON_DAYS) -> np.ndarray:
    """Per-row horizon label from day ordinals; failure_day is NO_DAY for a drive with no failure."""
    lead = failure_day - day
    positive = (lead >= 0) & (lead <= horizon)
    negative = (lead > horizon) & (last_day >= day + horizon)
    labels = np.full(len(day), LABEL_CENSORED, dtype="int8")
    labels[negative] = LABEL_NEGATIVE
    labels[positive] = LABEL_POSITIVE
    return labels


def _day_numbers(dates: pd.Series, missing: int | None = None) -> np.ndarray:
    parsed = pd.to_datetime(dates)
    null = parsed.isna().to_numpy()
    days = parsed.fillna(pd.Timestamp("1970-01-01")).to_numpy().astype("datetime64[D]").astype("int64")
    if missing is not None:
        days[null] = missing
    return days


def row_labels(timeline: pd.DataFrame, index: pd.DataFrame, horizon: int = HORIZON_DAYS) -> np.ndarray:
    """Horizon label of each timeline row, from the drive index's failure_date and last_seen."""
    per_drive = index.set_index("serial_number")
    where = pd.Index(per_drive.index.astype(object)).get_indexer(timeline["serial_number"].astype(object))
    if (where < 0).any():
        raise ValueError("timeline rows for drives missing from the drive index")
    failure_day = _day_numbers(per_drive["failure_date"], missing=NO_DAY)[where]
    last_day = _day_numbers(per_drive["last_seen"])[where]
    return classify_rows(day_ordinals(timeline["date"]), failure_day, last_day, horizon)


def label_counts(labels: np.ndarray) -> dict[str, int]:
    return {"positive": int((labels == LABEL_POSITIVE).sum()),
            "negative": int((labels == LABEL_NEGATIVE).sum()),
            "censored": int((labels == LABEL_CENSORED).sum())}


def drive_label_flags(timeline: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Per drive: whether it has a positive row and whether it has a negative row."""
    flags = pd.DataFrame({"serial_number": timeline["serial_number"].astype(object).to_numpy(),
                          "has_positive": labels == LABEL_POSITIVE,
                          "has_negative": labels == LABEL_NEGATIVE})
    return flags.groupby("serial_number", sort=True).any()


def drive_split_counts(index: pd.DataFrame, flags: pd.DataFrame) -> dict[str, int]:
    """Drives with at least one labelled row (eligible), by what they provide."""
    joined = flags.reindex(index["serial_number"].astype(object).to_numpy()).fillna(False).astype(bool)
    positive = joined["has_positive"]
    negative = joined["has_negative"]
    eligible = positive | negative
    return {"drives": len(joined), "eligible": int(eligible.sum()),
            "with_positive": int(positive.sum()),
            "negative_only": int((negative & ~positive).sum()),
            "all_censored": int((~eligible).sum())}


def label_lines(label_totals: dict[str, dict[str, int]], index: pd.DataFrame,
                flags: pd.DataFrame) -> list[str]:
    """Report lines for the horizon label counts and the drive split, overall and per model."""
    models = sorted(label_totals)
    overall = {k: sum(t[k] for t in label_totals.values()) for k in ("positive", "negative", "censored")}
    lines = [f"labels_h{HORIZON_DAYS} model=all positive={overall['positive']} "
             f"negative={overall['negative']} censored={overall['censored']}"]
    for m in models:
        t = label_totals[m]
        lines.append(f"labels_h{HORIZON_DAYS} model={m} positive={t['positive']} "
                     f"negative={t['negative']} censored={t['censored']}")
    for name, sub in [("all", index)] + [(m, index[index["model"] == m]) for m in models]:
        s = drive_split_counts(sub, flags)
        lines.append(f"label_drives model={name} drives={s['drives']} eligible={s['eligible']} "
                     f"with_positive={s['with_positive']} negative_only={s['negative_only']} "
                     f"all_censored={s['all_censored']}")
    return lines


def to_arrow(df: pd.DataFrame, schema: pa.Schema | None = None) -> pa.Table:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.date
    return pa.Table.from_pandas(out, schema=schema, preserve_index=False)


def sha256_of(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def partition_by_drive(src: Path, columns: list[str], tmpdir: Path, buckets: int) -> list[Path]:
    """Split the input row group by row group into one Parquet file per serial_number hash bucket."""
    pf = pq.ParquetFile(src)
    writers: dict[int, pq.ParquetWriter] = {}
    schema = pa.schema([pf.schema_arrow.field(c) for c in columns])
    try:
        for i in range(pf.num_row_groups):
            table = pf.read_row_group(i, columns=columns)
            ids = pd.util.hash_array(table["serial_number"].to_pandas().to_numpy(dtype=object)) % buckets
            for b in np.unique(ids):
                if b not in writers:
                    writers[b] = pq.ParquetWriter(tmpdir / f"bucket_{b}.parquet", schema)
                writers[b].write_table(table.filter(pa.array(ids == b)))
    finally:
        for w in writers.values():
            w.close()
    return [tmpdir / f"bucket_{b}.parquet" for b in sorted(writers)]


def date_span(src: Path) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    column = pq.read_table(src, columns=["date"]).column("date")
    bounds = pc.min_max(column)
    if bounds["min"].as_py() is None:
        return None, None
    return pd.Timestamp(bounds["min"].as_py()), pd.Timestamp(bounds["max"].as_py())


def update_manifest(entries: list[dict]) -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    names = {e["file"] for e in entries}
    data["files"] = [e for e in data.get("files", []) if e.get("file") not in names] + entries
    MANIFEST.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def update_preprocessing(lines: list[str]) -> None:
    body = "\n".join(f"- `{ln}`" for ln in lines)
    spec = ", ".join(f"`{n}`" for n, _, _ in feature_spec())
    section = (
        "## B2 — Per-drive timelines\n\n"
        "- Input: `data/processed/backblaze_filtered.parquet` (B1; window, models and columns in "
        "`data_pipeline/A1_DECISION.md`)\n"
        f"- SMART columns: {', '.join(f'`{c}`' for c in SMART_COLUMNS)}\n"
        "- `smart_187_raw` and `smart_188_raw` have no value for the Toshiba model; their level and "
        "difference features and the matching `_missing` flags are NaN and True for that model. "
        "`smart_12_raw` is kept because it is non-null in more than 99% of rows for every kept "
        "model\n"
        f"- Windows (calendar days, trailing): {', '.join(str(w) for w in WINDOWS_DAYS)}; "
        "`smart_188_raw` uses 7 only and has no level column\n"
        f"- Minimum observations for a difference: {MIN_OBS_DIFF}\n"
        f"- Feature columns: {spec}\n"
        "- Missingness: one `<feature>_missing` boolean per feature column; "
        "no value is imputed\n"
        "- Timelines table holds backward-looking columns only; `last_seen`, "
        "`days_observed`, `failed_in_window` and `failure_date` are in the drive index, which "
        "also carries `model`\n"
        f"- Horizon labels, counted per drive-day, not stored: horizon {HORIZON_DAYS} days. "
        "Positive: the drive's recorded failure date lies in [d, d + 30]. Negative: no failure within "
        "30 days and the drive has a row on or after d + 30. Censored: every other row, including "
        "rows whose 30-day horizon extends past the last date of the input and rows of drives that "
        "left the data without a recorded failure. The first and last date are taken from the input\n"
        "- Eligible drives: drives with at least one positive or negative row; `all_censored` drives "
        "have none\n"
        "- Counts from the last run of `backblaze/timelines.py`:\n\n"
        f"{body}\n\n"
    )
    text = PREPROCESSING.read_text(encoding="utf-8")
    new, n = re.subn(r"## B2 — Per-drive timelines\n.*?(?=\n## B3 )",
                     section.rstrip("\n") + "\n", text, count=1, flags=re.S)
    if n != 1:
        raise SystemExit("B2 section not found in PREPROCESSING.md")
    PREPROCESSING.write_text(new, encoding="utf-8")


def main(argv: list[str]) -> int:
    src = Path(argv[1]) if len(argv) > 1 else INPUT
    columns = CORE_IN + list(SMART_COLUMNS)
    rows_in = pq.ParquetFile(src).metadata.num_rows
    first_date, last_date = date_span(src)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    feature_names = [n for n, _, _ in feature_spec()]
    missing_totals = dict.fromkeys(feature_names, 0)
    timeline_rows = timeline_columns = 0
    gap_rows = first_rows = multi_failure = 0
    index_parts: list[pd.DataFrame] = []
    label_totals: dict[str, dict[str, int]] = {}
    flag_parts: list[pd.DataFrame] = []
    model_rows: dict[str, int] = {}

    with tempfile.TemporaryDirectory(dir=PROCESSED) as tmp:
        parts = partition_by_drive(src, columns, Path(tmp), BUCKETS)
        if not parts:
            empty = pq.read_table(src, columns=columns).to_pandas()
            empty_path = Path(tmp) / "bucket_empty.parquet"
            pq.write_table(pa.Table.from_pandas(empty, preserve_index=False), empty_path)
            parts = [empty_path]
        writer = None
        try:
            for part in parts:
                df = pq.read_table(part).to_pandas()
                timelines = build_timelines(df)
                index = build_drive_index(df[CORE_IN])
                labels = row_labels(timelines, index)

                if writer is None:
                    table = to_arrow(timelines)
                    schema = table.schema
                    writer = pq.ParquetWriter(OUT_TIMELINES, schema)
                else:
                    table = to_arrow(timelines, schema)
                writer.write_table(table)

                timeline_rows += len(timelines)
                timeline_columns = timelines.shape[1]
                gap_rows += int((timelines["gap_days_before"] > 1).sum())
                first_rows += int(timelines["gap_days_before"].isna().sum())
                multi_failure += int((df[df["failure"] == 1].groupby("serial_number").size() > 1).sum())
                for name in feature_names:
                    missing_totals[name] += int(timelines[f"{name}_missing"].sum())
                for model, g in timelines.groupby(timelines["model"].astype(object)):
                    model_rows[str(model)] = model_rows.get(str(model), 0) + len(g)
                    counts = label_counts(labels[g.index.to_numpy()])
                    totals = label_totals.setdefault(str(model), dict.fromkeys(counts, 0))
                    for k, v in counts.items():
                        totals[k] += v
                flag_parts.append(drive_label_flags(timelines, labels))
                index_parts.append(index)
        finally:
            if writer is not None:
                writer.close()

    index = (pd.concat(index_parts, ignore_index=True)
             .sort_values("serial_number", kind="stable").reset_index(drop=True))
    flags = pd.concat(flag_parts)
    pq.write_table(to_arrow(index), OUT_INDEX)

    models = sorted(model_rows)
    span_first = first_date.date().isoformat() if first_date is not None else "none"
    span_last = last_date.date().isoformat() if last_date is not None else "none"
    lines = [
        f"rows_in={rows_in} drives={len(index)}",
        f"window_first={span_first} window_last={span_last} horizon_days={HORIZON_DAYS}",
        f"timelines_rows={timeline_rows} timelines_columns={timeline_columns}",
        f"drive_index_rows={len(index)} failed_in_window={int(index['failed_in_window'].sum())}",
    ]
    for m in models:
        sub = index[index["model"] == m]
        lines.append(f"model={m} timelines_rows={model_rows[m]} drives={len(sub)} "
                     f"failed_in_window={int(sub['failed_in_window'].sum())}")
    lines += label_lines(label_totals, index, flags)
    lines += [
        f"drives_with_multiple_failure_rows={multi_failure}",
        f"rows_with_gap_days_before_gt_1={gap_rows}",
        f"rows_first_observation={first_rows}",
    ]
    for name in feature_names:
        lines.append(f"{name}_missing={missing_totals[name]}")
    entries = []
    for path, rows in ((OUT_TIMELINES, timeline_rows), (OUT_INDEX, len(index))):
        rel = path.relative_to(ROOT).as_posix()
        digest = sha256_of(path)
        lines.append(f"output={rel} rows={rows} sha256={digest}")
        entries.append({"file": rel, "rows": rows, "sha256": digest})
    print("\n".join(lines))
    update_manifest(entries)
    update_preprocessing(lines)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
