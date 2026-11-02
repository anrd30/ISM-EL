"""
B1: Backblaze Drive Stats extract.

Reads the daily CSVs of one or more quarter directories in date order, keeps
only the selected drive models and the fixed column list (core columns plus
SMART_COLUMNS), and writes one Parquet file with one row per drive per day.
SMART values are carried through unmodified; feature engineering belongs to
the timeline stage.

Alignment: a column absent from a given day's file, or not reported for a
model, is NaN in the output. Columns are never dropped and never filled with
a value.

File checks (SystemExit on failure): every file name is an ISO date; no date
appears in more than one file; the only calendar days from the first to the
last date without a file are the dates of the files listed in EXCLUDED_FILES;
every kept row's date equals its file name. A file listed in EXCLUDED_FILES
(key: "<quarter directory>/<file name>") is not read into the output; its rows
of the kept models are counted and reported. Each daily file is filtered to the kept models before the next is read,
and the output is written one file at a time.

With no argument the four 2022 quarter directories are read and the measured
failures and drive-days are compared with the recorded values; a difference
aborts the run before the output is replaced.

Usage:
    python -I extract.py [quarter_dir ...]

Outputs:
    data/processed/backblaze_filtered.parquet
    manifest.json (entry upserted)
    PREPROCESSING.md (B1 section rewritten)
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
QUARTER_DIRS = tuple(ROOT / "data" / "raw" / f"backblaze_2022Q{q}" for q in (1, 2, 3, 4))
OUTPUT = ROOT / "data" / "processed" / "backblaze_filtered.parquet"
MANIFEST = ROOT / "manifest.json"
PREPROCESSING = ROOT / "PREPROCESSING.md"

MODELS = ("ST4000DM000", "ST8000NM0055", "ST12000NM0008", "TOSHIBA MG07ACA14TA")
SEAGATE_MODELS = MODELS[:3]
SMART_COLUMNS = (
    "smart_5_raw", "smart_9_raw", "smart_12_raw", "smart_187_raw", "smart_188_raw",
    "smart_193_raw", "smart_194_raw", "smart_197_raw", "smart_198_raw", "smart_199_raw",
)

EXCLUDED_FILES = {
    "backblaze_2022Q4/2022-12-10.csv": "all rows dated 2022-12-11; copy of the 2022-12-11 file",
}

# Measured values recorded in A1_DECISION.md for the four 2022 quarters.
EXPECTED_FILES = 364
EXPECTED_DATE_RANGE = ("2022-01-01", "2022-12-31")
EXPECTED_PER_MODEL = {  # model -> (failures, drive-days)
    "ST4000DM000": (633, 6_681_069),
    "ST8000NM0055": (348, 5_215_692),
    "ST12000NM0008": (404, 7_272_074),
    "TOSHIBA MG07ACA14TA": (385, 13_883_449),
}
EXPECTED_POOLED = (1_770, 33_052_284)  # (failures, drive-days)
EXPECTED_SEAGATE_FAILURES_BY_QUARTER = {
    "backblaze_2022Q1": 276, "backblaze_2022Q2": 343,
    "backblaze_2022Q3": 433, "backblaze_2022Q4": 333,
}

CORE_COLUMNS = ["serial_number", "model", "date", "capacity_bytes", "failure"]
KEY = ["serial_number", "date"]
SMART_RE = re.compile(r"^smart_(\d+)_(raw|normalized)$")


def build_column_list() -> list[str]:
    """Core columns, then SMART_COLUMNS in order."""
    return CORE_COLUMNS + list(SMART_COLUMNS)


def read_header(path: Path) -> list[str]:
    return list(pd.read_csv(path, nrows=0).columns)


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
    return pd.read_csv(path, usecols=present, dtype=dtypes)


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


def collect_files(quarter_dirs: Sequence[Path]) -> tuple[
        list[tuple[dt.date, Path, Path]], list[tuple[dt.date, Path, Path]]]:
    """(included, excluded) as (date, file, quarter directory) in date order, after the file checks."""
    entries: list[tuple[dt.date, Path, Path]] = []
    excluded: list[tuple[dt.date, Path, Path]] = []
    for qdir in quarter_dirs:
        found = sorted(qdir.glob("*.csv"))
        if not found:
            raise SystemExit(f"no CSV files in {qdir}")
        for f in found:
            try:
                day = dt.date.fromisoformat(f.stem)
            except ValueError:
                raise SystemExit(f"{f}: file name is not an ISO date") from None
            target = excluded if f"{qdir.name}/{f.name}" in EXCLUDED_FILES else entries
            target.append((day, f, qdir))
    entries.sort(key=lambda e: e[0])
    excluded.sort(key=lambda e: e[0])
    if not entries:
        raise SystemExit("no CSV files left after exclusions")
    for (d1, f1, _), (d2, f2, _) in zip(entries, entries[1:]):
        if d1 == d2:
            raise SystemExit(f"date {d1.isoformat()} appears in more than one file: {f1}, {f2}")
    first, last = entries[0][0], entries[-1][0]
    span = (last - first).days + 1
    present = {e[0] for e in entries}
    absent = {first + dt.timedelta(days=i) for i in range(span)} - present
    permitted = {e[0] for e in excluded}
    if not absent <= permitted:
        unlisted = sorted(d.isoformat() for d in absent - permitted)
        raise SystemExit(
            f"{len(entries)} files for {span} calendar days "
            f"({first.isoformat()} to {last.isoformat()}); dates without a file: {unlisted}")
    return entries, excluded


def extract(quarter_dirs: Path | Sequence[Path],
            check: Callable[[dict], None] | None = None) -> dict:
    """Write the filtered extract. `check(result)` runs before the output replaces any
    previous file and aborts the run by raising."""
    if isinstance(quarter_dirs, Path):
        quarter_dirs = [quarter_dirs]
    quarter_dirs = list(quarter_dirs)
    entries, excluded = collect_files(quarter_dirs)
    files = [e[1] for e in entries]
    excluded_rows = 0
    for _, f, _ in excluded:
        models = pd.read_csv(f, usecols=["model"], dtype="string")["model"]
        excluded_rows += int(models.isin(MODELS).sum())

    headers = {f.name: read_header(f) for f in files}
    columns = build_column_list()
    smart_cols = [c for c in columns if c not in CORE_COLUMNS]
    missing_per_file = {
        name: sum(1 for c in smart_cols if c not in hdr) for name, hdr in headers.items()
    }
    schema = arrow_schema(columns)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT.with_suffix(".parquet.tmp")

    rows_in = rows_models = dup_dropped = date_mismatch = null_capacity = non_iso = 0
    per_model = {m: {"rows": 0, "failures": 0, "drives": set()} for m in MODELS}
    per_quarter = {q.name: {"files": 0, "failures": 0, "seagate_failures": 0} for q in quarter_dirs}
    nonnull = {m: dict.fromkeys(smart_cols, 0) for m in MODELS}
    rows_out = 0

    try:
        with pq.ParquetWriter(tmp, schema) as writer:
            for _, f, qdir in entries:
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
                mismatched = int((day["date"].dt.strftime("%Y-%m-%d") != f.stem).sum())
                if mismatched:
                    raise SystemExit(f"{f.name}: {mismatched} rows with a date different from the file name")
                null_capacity += int(day["capacity_bytes"].isna().sum())

                quarter = per_quarter[qdir.name]
                quarter["files"] += 1
                for model, g in day.groupby("model"):
                    s = per_model[str(model)]
                    failures = int(g["failure"].sum())
                    s["rows"] += len(g)
                    s["failures"] += failures
                    s["drives"].update(g["serial_number"])
                    quarter["failures"] += failures
                    if str(model) in SEAGATE_MODELS:
                        quarter["seagate_failures"] += failures
                    for c in smart_cols:
                        nonnull[str(model)][c] += int(g[c].notna().sum())

                out = day.copy()
                out["date"] = out["date"].dt.date
                out["failure"] = out["failure"].astype("int8")
                out["capacity_bytes"] = out["capacity_bytes"].astype("Int64")
                writer.write_table(pa.Table.from_pandas(out, schema=schema, preserve_index=False))
                rows_out += len(out)

        result = {
            "quarter_dirs": quarter_dirs,
            "files": files,
            "first_date": entries[0][0],
            "last_date": entries[-1][0],
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
            "per_quarter": per_quarter,
            "all_null_cols": [c for c in smart_cols if all(nonnull[m][c] == 0 for m in MODELS)],
            "all_null_by_model": {m: [c for c in smart_cols if nonnull[m][c] == 0] for m in MODELS},
            "excluded_files": [f"{e[2].name}/{e[1].name}" for e in excluded],
            "rows_excluded": excluded_rows,
            "verified": False,
        }
        if check is not None:
            check(result)
            result["verified"] = True
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    tmp.replace(OUTPUT)
    result["output_sha256"] = sha256_of(OUTPUT)
    return result


def verify_expected(r: dict) -> None:
    """Compare the measured counts with the recorded values; SystemExit on any difference."""
    problems = []

    def compare(name: str, expected, measured) -> None:
        if expected != measured:
            problems.append(f"{name}: expected {expected}, measured {measured}")

    compare("files", EXPECTED_FILES, len(r["files"]))
    compare("date_range", EXPECTED_DATE_RANGE,
            (r["first_date"].isoformat(), r["last_date"].isoformat()))
    for model, (failures, drive_days) in EXPECTED_PER_MODEL.items():
        s = r["per_model"][model]
        compare(f"{model} failures", failures, s["failures"])
        compare(f"{model} drive_days", drive_days, s["rows"])
    compare("pooled failures", EXPECTED_POOLED[0], sum(s["failures"] for s in r["per_model"].values()))
    compare("pooled drive_days", EXPECTED_POOLED[1], sum(s["rows"] for s in r["per_model"].values()))
    for quarter, failures in EXPECTED_SEAGATE_FAILURES_BY_QUARTER.items():
        compare(f"{quarter} seagate failures", failures,
                r["per_quarter"].get(quarter, {}).get("seagate_failures"))
    if problems:
        raise SystemExit("verification against A1_DECISION.md failed:\n" + "\n".join(problems))


def report_lines(r: dict) -> list[str]:
    lines = [
        f"quarter_dirs={','.join(q.name for q in r['quarter_dirs'])}",
        f"daily_files={len(r['files'])} date_first={r['first_date'].isoformat()} "
        f"date_last={r['last_date'].isoformat()}",
    ]
    for q in r["quarter_dirs"]:
        s = r["per_quarter"][q.name]
        lines.append(f"quarter={q.name} files={s['files']} failures={s['failures']} "
                     f"seagate_failures={s['seagate_failures']}")
    lines += [
        f"excluded_files={','.join(r['excluded_files']) or 'none'} rows_excluded={r['rows_excluded']}",
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
                     f"failures={s['failures']} "
                     f"smart_columns_all_null={','.join(r['all_null_by_model'][m]) or 'none'}")
        total_rows += s["rows"]
        total_fail += s["failures"]
    afr = total_fail / total_rows * 365 if total_rows else float("nan")
    lines.append(f"pooled drive_days={total_rows} failures={total_fail} afr={afr:.4%}")
    lines.append(f"verification_against_a1_decision={'PASS' if r['verified'] else 'not run'}")
    lines.append(f"output={OUTPUT.relative_to(ROOT).as_posix()} sha256={r['output_sha256']}")
    return lines


def excluded_text(r: dict) -> str:
    if not r["excluded_files"]:
        return "none"
    return "; ".join(f"`{name}` ({EXCLUDED_FILES[name]})" for name in r["excluded_files"]) + \
        f"; {r['rows_excluded']} rows of the kept models are not in the output and the date is a " \
        "permitted gap in the calendar-span check"


def update_preprocessing(r: dict, lines: list[str], source_digests: dict[str, str]) -> None:
    body = "\n".join(f"- `{ln}`" for ln in lines)
    section = (
        "## B1 — Backblaze extract\n\n"
        "- Window, models and SMART columns: `data_pipeline/A1_DECISION.md`\n"
        f"- Quarter directories: {', '.join(f'`{q.name}`' for q in r['quarter_dirs'])}, read in "
        "date order; the file count must equal the number of calendar days from the first to the "
        "last date, no date may appear in two files, and every kept row's date must equal its "
        "file name\n"
        f"- Excluded files: {excluded_text(r)}\n"
        f"- Models kept: `{', '.join(MODELS)}`\n"
        f"- SMART columns kept: {', '.join(f'`{c}`' for c in SMART_COLUMNS)}\n"
        "- A column that a model does not report (`smart_187_raw`, `smart_188_raw` for the Toshiba "
        "model) is NaN for that model; it is not dropped and not filled\n"
        "- Source file digests: `manifest.json` key `sources`\n"
        "- A run over the four default directories compares the per-model failures and drive-days, "
        "the pooled totals, the file count, the date range and the per-quarter Seagate failures "
        "with the values in `A1_DECISION.md` and aborts on any difference\n"
        "- Counts from the last run of `backblaze/extract.py`:\n\n"
        f"{body}\n\n"
        "- `smart_columns_all_null` counts aligned SMART columns that hold no value for any kept "
        "model; the per-model entry lists the columns with no value for that model.\n\n"
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
    dirs = [Path(a) for a in argv[1:]] or list(QUARTER_DIRS)
    default_run = tuple(dirs) == QUARTER_DIRS
    r = extract(dirs, check=verify_expected if default_run else None)
    lines = report_lines(r)
    print("\n".join(lines))
    source_digests = {f.name: sha256_of(f) for f in r["files"]}
    update_preprocessing(r, lines, source_digests)
    update_manifest(r, source_digests)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
