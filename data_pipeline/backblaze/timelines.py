"""
B2: per-drive timelines from the B1 extract.

Produces two tables.

timelines (one row per drive-day): only backward-looking columns. Every
value on a row is a function of that drive's rows with date <= the row date,
so truncating the input at any date leaves all earlier rows unchanged.

drive index (one row per drive): whole-drive facts (last_seen, days_observed,
failed_in_window, failure_date). These depend on the full observation window,
are never features, and are joined to the timelines only at label construction.

Windows are calendar-based. For a window of N days, the trailing difference
of a column is value(today) - value(earliest observed row with
date >= today - N). It is NaN when fewer than MIN_OBS_DIFF rows fall in
[today - N, today] or either endpoint is NaN. No value is imputed.

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
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
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
    "smart_193_raw", "smart_194_raw", "smart_197_raw", "smart_198_raw",
)
WINDOWS_DAYS = (7, 30)
# smart_188_raw packs several counters, so its level is not comparable across drives.
NO_LEVEL = frozenset({"smart_188_raw"})
WINDOWS_OVERRIDE = {"smart_188_raw": (7,)}
MIN_OBS_DIFF = 2

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


def to_arrow(df: pd.DataFrame) -> pa.Table:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.date
    return pa.Table.from_pandas(out, preserve_index=False)


def sha256_of(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


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
        f"- Input: `data/processed/backblaze_filtered.parquet` (provisional B1 subset)\n"
        f"- SMART columns: {', '.join(f'`{c}`' for c in SMART_COLUMNS)}\n"
        f"- Windows (calendar days, trailing): {', '.join(str(w) for w in WINDOWS_DAYS)}; "
        "`smart_188_raw` uses 7 only and has no level column\n"
        f"- Minimum observations for a difference: {MIN_OBS_DIFF}\n"
        f"- Feature columns: {spec}\n"
        "- Missingness: one `<feature>_missing` boolean per feature column; "
        "no value is imputed\n"
        "- Timelines table holds backward-looking columns only; `last_seen`, "
        "`days_observed`, `failed_in_window` and `failure_date` are in the drive index\n"
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
    df = pq.read_table(src, columns=CORE_IN + list(SMART_COLUMNS)).to_pandas()
    rows_in = len(df)

    timelines = build_timelines(df)
    index = build_drive_index(df[CORE_IN])

    PROCESSED.mkdir(parents=True, exist_ok=True)
    pq.write_table(to_arrow(timelines), OUT_TIMELINES)
    pq.write_table(to_arrow(index), OUT_INDEX)

    multi_failure = int((df[df["failure"] == 1].groupby("serial_number").size() > 1).sum())
    lines = [
        f"rows_in={rows_in} drives={len(index)}",
        f"timelines_rows={len(timelines)} timelines_columns={timelines.shape[1]}",
        f"drive_index_rows={len(index)} failed_in_window={int(index['failed_in_window'].sum())}",
        f"drives_with_multiple_failure_rows={multi_failure}",
        f"rows_with_gap_days_before_gt_1={int((timelines['gap_days_before'] > 1).sum())}",
        f"rows_first_observation={int(timelines['gap_days_before'].isna().sum())}",
    ]
    for name, _, _ in feature_spec():
        lines.append(f"{name}_missing={int(timelines[f'{name}_missing'].sum())}")
    entries = []
    for path, frame in ((OUT_TIMELINES, timelines), (OUT_INDEX, index)):
        rel = path.relative_to(ROOT).as_posix()
        digest = sha256_of(path)
        lines.append(f"output={rel} rows={len(frame)} sha256={digest}")
        entries.append({"file": rel, "rows": len(frame), "sha256": digest})
    print("\n".join(lines))
    update_manifest(entries)
    update_preprocessing(lines)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
