"""
B1: Backblaze Drive Stats extract.

Reads the daily CSVs of one quarter, aligns every file to one column list
(the union of all per-file headers), keeps only the selected drive models,
and writes one Parquet file with one row per drive per day. SMART values are
carried through unmodified; feature engineering belongs to the timeline stage.

Alignment: a column absent from a given day's file is filled with NaN for
that day's rows. Columns are never dropped because they are missing in some
files, and quarters are never concatenated here.

Usage:
    python -I extract.py [quarter_dir]

Outputs:
    data/processed/backblaze_filtered.parquet
    manifest.json (entry upserted)
    PREPROCESSING.md (B1 section rewritten)
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUARTER_DIR = ROOT / "data" / "raw" / "backblaze_2022Q1"
OUTPUT = ROOT / "data" / "processed" / "backblaze_filtered.parquet"
MANIFEST = ROOT / "manifest.json"
PREPROCESSING = ROOT / "PREPROCESSING.md"

# TODO: quarter/models/SMART columns (provisional, not ratified)
MODELS = ("ST4000DM000", "ST8000NM0055", "ST12000NM0008")
SMART_SUFFIXES = ("_raw",)

CORE_COLUMNS = ["serial_number", "model", "date", "capacity_bytes", "failure"]
KEY = ["serial_number", "date"]
SMART_RE = re.compile(r"^smart_(\d+)_(raw|normalized)$")


def smart_sort_key(col: str) -> tuple[int, str]:
    m = SMART_RE.match(col)
    return int(m.group(1)), m.group(2)


def read_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


def build_column_list(headers: dict[str, list[str]]) -> list[str]:
    """Core columns, then every selected SMART column seen in any file."""
    seen: set[str] = set()
    for cols in headers.values():
        seen.update(cols)
    smart = [c for c in seen if SMART_RE.match(c) and c.endswith(SMART_SUFFIXES)]
    return CORE_COLUMNS + sorted(smart, key=smart_sort_key)


def align_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Reindex to the aligned column list; absent columns become NaN."""
    return df.reindex(columns=columns)


def read_day(path: Path, header: list[str], columns: list[str]) -> pd.DataFrame:
    present = [c for c in columns if c in header]
    smart = [c for c in present if c not in CORE_COLUMNS]
    dtypes: dict[str, object] = {"serial_number": "string", "model": "string",
                                 "date": "string", "capacity_bytes": "Int64",
                                 "failure": "Int64"}
    dtypes.update({c: "float64" for c in smart})
    df = pd.read_csv(path, usecols=present, dtype=dtypes)
    return df


def parse_dates(raw: pd.Series) -> tuple[pd.Series, int]:
    """Parse ISO dates; fall back to MM-DD-YYYY. Returns (dates, non-ISO row count)."""
    iso = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")
    alt = pd.to_datetime(raw[iso.isna()], format="%m-%d-%Y", errors="coerce")
    out = iso.copy()
    out[alt.index] = alt
    if out.isna().any():
        raise SystemExit(f"unparseable date values: {raw[out.isna()].unique()[:3].tolist()}")
    return out, int(iso.isna().sum())


def arrow_schema(columns: list[str]) -> pa.Schema:
    fields = [
        pa.field("serial_number", pa.string()),
        pa.field("model", pa.string()),
        pa.field("date", pa.date32()),
        pa.field("capacity_bytes", pa.int64()),
        pa.field("failure", pa.int8()),
    ]
    fields += [pa.field(c, pa.float64()) for c in columns if c not in CORE_COLUMNS]
    return pa.schema(fields)


def sha256_of(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def extract(quarter_dir: Path) -> dict:
    files = sorted(quarter_dir.glob("*.csv"))
    if not files:
        raise SystemExit(f"no CSV files in {quarter_dir}")

    headers = {f.name: read_header(f) for f in files}
    columns = build_column_list(headers)
    smart_cols = [c for c in columns if c not in CORE_COLUMNS]
    missing_per_file = {
        name: sum(1 for c in smart_cols if c not in hdr) for name, hdr in headers.items()
    }
    schema = arrow_schema(columns)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_suffix(".parquet.tmp")

    rows_in = rows_models = dup_dropped = date_mismatch = null_capacity = non_iso = 0
    per_model = {m: {"rows": 0, "failures": 0, "drives": set()} for m in MODELS}
    nonnull = dict.fromkeys(smart_cols, 0)
    rows_out = 0

    with pq.ParquetWriter(tmp, schema) as writer:
        for f in files:
            day = read_day(f, headers[f.name], columns)
            rows_in += len(day)
            day = day[day["model"].isin(MODELS)]
            rows_models += len(day)
            day = align_frame(day, columns)

            parsed, n_alt = parse_dates(day["date"])
            non_iso += n_alt
            day = day.assign(date=parsed)

            before = len(day)
            day = day.drop_duplicates(subset=KEY, keep="first")
            dup_dropped += before - len(day)

            if day["failure"].isna().any():
                raise SystemExit(f"{f.name}: null in failure column")
            date_mismatch += int((day["date"].dt.strftime("%Y-%m-%d") != f.stem).sum())
            null_capacity += int(day["capacity_bytes"].isna().sum())

            for model, g in day.groupby("model"):
                s = per_model[str(model)]
                s["rows"] += len(g)
                s["failures"] += int(g["failure"].sum())
                s["drives"].update(g["serial_number"])
            for c in smart_cols:
                nonnull[c] += int(day[c].notna().sum())

            out = day.copy()
            out["date"] = out["date"].dt.date
            out["failure"] = out["failure"].astype("int8")
            out["capacity_bytes"] = out["capacity_bytes"].astype("Int64")
            writer.write_table(pa.Table.from_pandas(out, schema=schema, preserve_index=False))
            rows_out += len(out)

    tmp.replace(OUTPUT)

    return {
        "quarter_dir": quarter_dir,
        "files": files,
        "columns": columns,
        "smart_cols": smart_cols,
        "missing_per_file": missing_per_file,
        "rows_in": rows_in,
        "rows_models": rows_models,
        "dup_dropped": dup_dropped,
        "rows_out": rows_out,
        "date_mismatch": date_mismatch,
        "non_iso": non_iso,
        "null_capacity": null_capacity,
        "per_model": per_model,
        "all_null_cols": [c for c in smart_cols if nonnull[c] == 0],
        "output_sha256": sha256_of(OUTPUT),
    }


def report_lines(r: dict) -> list[str]:
    lines = [
        f"quarter_dir={r['quarter_dir'].name} daily_files={len(r['files'])}",
        f"models={','.join(MODELS)}",
        f"columns_out={len(r['columns'])} smart_columns={len(r['smart_cols'])}",
        f"smart_columns_all_null={len(r['all_null_cols'])}",
        f"files_missing_some_smart_column={sum(1 for n in r['missing_per_file'].values() if n)}",
        f"rows_in={r['rows_in']}",
        f"rows_after_model_subset={r['rows_models']} "
        f"dropped_other_models={r['rows_in'] - r['rows_models']}",
        f"duplicate_serial_date_dropped={r['dup_dropped']}",
        f"rows_out={r['rows_out']}",
        f"rows_date_not_iso={r['non_iso']}",
        f"rows_date_differs_from_filename={r['date_mismatch']}",
        f"rows_null_capacity_bytes={r['null_capacity']}",
    ]
    total_rows = total_fail = 0
    for m, s in r["per_model"].items():
        lines.append(f"model={m} drive_days={s['rows']} drives={len(s['drives'])} "
                     f"failures={s['failures']}")
        total_rows += s["rows"]
        total_fail += s["failures"]
    afr = total_fail / total_rows * 365 if total_rows else float("nan")
    lines.append(f"pooled drive_days={total_rows} failures={total_fail} afr={afr:.4%}")
    lines.append(f"output={OUTPUT.relative_to(ROOT).as_posix()} sha256={r['output_sha256']}")
    return lines


def update_preprocessing(r: dict, lines: list[str], source_digests: dict[str, str]) -> None:
    body = "\n".join(f"- `{ln}`" for ln in lines)
    section = (
        "## B1 — Backblaze extract\n\n"
        f"- Quarter: `{r['quarter_dir'].name}` (provisional)\n"
        f"- Models kept: `{', '.join(MODELS)}` (provisional)\n"
        f"- SMART columns kept: every `smart_*{'|'.join(SMART_SUFFIXES)}` column present in any "
        "daily file (provisional)\n"
        "- Source file digests: `manifest.json` key `sources`\n"
        "- Counts from the last run of `backblaze/extract.py`:\n\n"
        f"{body}\n\n"
        "- `smart_columns_all_null` counts aligned SMART columns that hold no value for the "
        "kept models; they are retained so the column list stays fixed.\n\n"
    )
    text = PREPROCESSING.read_text(encoding="utf-8")
    new, n = re.subn(r"## B1 — Backblaze extract\n.*?(?=\n## B2 )", section.rstrip("\n") + "\n",
                     text, count=1, flags=re.S)
    if n != 1:
        raise SystemExit("B1 section not found in PREPROCESSING.md")
    PREPROCESSING.write_text(new, encoding="utf-8")


def update_manifest(r: dict, source_digests: dict[str, str]) -> None:
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rel = OUTPUT.relative_to(ROOT).as_posix()
    entry = {"file": rel, "rows": r["rows_out"], "sha256": r["output_sha256"]}
    data["files"] = [e for e in data.get("files", []) if e.get("file") != rel] + [entry]
    data.setdefault("sources", {})["backblaze"] = source_digests
    MANIFEST.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str]) -> int:
    quarter_dir = Path(argv[1]) if len(argv) > 1 else DEFAULT_QUARTER_DIR
    r = extract(quarter_dir)
    lines = report_lines(r)
    print("\n".join(lines))
    source_digests = {f.name: sha256_of(f) for f in r["files"]}
    update_preprocessing(r, lines, source_digests)
    update_manifest(r, source_digests)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
