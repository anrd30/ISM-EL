"""
Unit tests for backblaze/extract.py (B1) on small synthetic daily CSVs.

All path constants of the module are redirected into tmp_path, so no test
touches the real manifest, PREPROCESSING.md or processed data.
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

pytest.importorskip("pyarrow")

from data_pipeline.backblaze import extract  # noqa: E402

HEADER_A = ("date,serial_number,model,capacity_bytes,failure,"
            "smart_5_raw,smart_5_normalized,smart_9_raw,smart_12_raw")
HEADER_B = ("date,serial_number,model,capacity_bytes,failure,"
            "smart_5_raw,smart_187_raw,smart_12_raw")

# 2022-01-01 uses MM-DD-YYYY dates; one other-model row; one duplicate (serial, date).
DAY1 = "\n".join([
    HEADER_A,
    "01-01-2022,A1,ST4000DM000,4000787030016,0,5,100,1000,",
    "01-01-2022,A2,ST8000NM0055,8001563222016,0,6,100,2000,",
    "01-01-2022,X1,OTHER,1000204886016,0,7,100,3000,",
    "01-01-2022,A1,ST4000DM000,4000787030016,0,5,100,1000,",
]) + "\n"

# 2022-01-02 is ISO, lacks smart_9_raw, has smart_187_raw, and holds one row
# whose date differs from the filename.
DAY2 = "\n".join([
    HEADER_B,
    "2022-01-02,A1,ST4000DM000,4000787030016,1,5,9,",
    "2022-01-02,A3,ST12000NM0008,12000138625024,0,8,0,",
    "2022-01-03,A2,ST8000NM0055,8001563222016,0,6,1,",
]) + "\n"


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / "pipe"
    (root / "data" / "processed").mkdir(parents=True)
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"files": [{"file": "other.parquet", "rows": 1, "sha256": "x"}]}))
    prep = root / "PREPROCESSING.md"
    prep.write_text("# P\n\n## B1 — Backblaze extract\n\n- stub\n\n## B2 — Per-drive timelines\n\n- keep\n", encoding="utf-8")
    quarter = tmp_path / "q"
    quarter.mkdir()
    monkeypatch.setattr(extract, "ROOT", root)
    monkeypatch.setattr(extract, "OUTPUT", root / "data" / "processed" / "backblaze_filtered.parquet")
    monkeypatch.setattr(extract, "MANIFEST", manifest)
    monkeypatch.setattr(extract, "PREPROCESSING", prep)
    monkeypatch.setattr(extract, "DEFAULT_QUARTER_DIR", quarter)
    return quarter


def write_days(quarter: Path) -> None:
    (quarter / "2022-01-01.csv").write_text(DAY1)
    (quarter / "2022-01-02.csv").write_text(DAY2)


def test_parse_dates_iso_and_fallback():
    raw = pd.Series(["2022-01-05", "01-31-2022", "2022-02-01"])
    parsed, n_alt = extract.parse_dates(raw)
    assert n_alt == 1
    assert parsed.dt.strftime("%Y-%m-%d").tolist() == ["2022-01-05", "2022-01-31", "2022-02-01"]


def test_parse_dates_all_iso_and_empty():
    parsed, n_alt = extract.parse_dates(pd.Series(["2022-03-01"]))
    assert n_alt == 0 and parsed.iloc[0] == pd.Timestamp("2022-03-01")
    parsed, n_alt = extract.parse_dates(pd.Series([], dtype="string"))
    assert n_alt == 0 and len(parsed) == 0


@pytest.mark.parametrize("bad", ["not-a-date", "2022-13-45", "31-31-2022"])
def test_parse_dates_unparseable_raises(bad):
    with pytest.raises(SystemExit, match="unparseable date values"):
        extract.parse_dates(pd.Series(["2022-01-01", bad]))


def test_sha256_of_matches_hashlib(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc" * 1000)
    assert extract.sha256_of(p) == hashlib.sha256(b"abc" * 1000).hexdigest()


def test_arrow_schema_types():
    schema = extract.arrow_schema(extract.CORE_COLUMNS + ["smart_5_raw"])
    assert [f.name for f in schema] == extract.CORE_COLUMNS + ["smart_5_raw"]
    assert str(schema.field("date").type) == "date32[day]"
    assert str(schema.field("failure").type) == "int8"
    assert str(schema.field("smart_5_raw").type) == "double"


def test_extract_counts_and_output(env):
    write_days(env)
    r = extract.extract(env)

    assert r["rows_in"] == 7
    assert r["rows_models"] == 6
    assert r["rows_in"] - r["rows_models"] == 1
    assert r["dup_dropped"] == 1
    assert r["rows_out"] == 5
    assert r["non_iso"] == 3
    assert r["date_mismatch"] == 1
    assert r["null_capacity"] == 0
    assert r["smart_cols"] == ["smart_5_raw", "smart_9_raw", "smart_12_raw", "smart_187_raw"]
    assert r["all_null_cols"] == ["smart_12_raw"]
    assert r["missing_per_file"] == {"2022-01-01.csv": 1, "2022-01-02.csv": 1}
    pm = r["per_model"]
    assert (pm["ST4000DM000"]["rows"], pm["ST4000DM000"]["failures"]) == (2, 1)
    assert pm["ST8000NM0055"]["rows"] == 2 and pm["ST12000NM0008"]["rows"] == 1
    assert pm["ST4000DM000"]["drives"] == {"A1"}

    out = pq.read_table(extract.OUTPUT).to_pandas()
    assert len(out) == 5
    assert "X1" not in set(out["serial_number"])
    assert list(out.columns) == r["columns"]
    from_day1 = out["smart_9_raw"].notna()
    assert from_day1.sum() == 2
    assert out.loc[from_day1, "smart_187_raw"].isna().all()
    assert out.loc[~from_day1, "smart_187_raw"].notna().all()
    assert r["output_sha256"] == extract.sha256_of(extract.OUTPUT)
    assert not extract.OUTPUT.with_suffix(".parquet.tmp").exists()


def test_extract_no_csv_files_raises(env):
    with pytest.raises(SystemExit, match="no CSV files"):
        extract.extract(env)


def test_extract_unparseable_date_raises(env):
    (env / "2022-01-01.csv").write_text(
        HEADER_A + "\nnot-a-date,A1,ST4000DM000,4000787030016,0,5,100,1000,\n")
    with pytest.raises(SystemExit, match="unparseable date values"):
        extract.extract(env)


def test_extract_null_failure_raises(env):
    (env / "2022-01-01.csv").write_text(
        HEADER_A + "\n2022-01-01,A1,ST4000DM000,4000787030016,,5,100,1000,\n")
    with pytest.raises(SystemExit, match="null in failure column"):
        extract.extract(env)


def test_extract_counts_null_capacity(env):
    (env / "2022-01-01.csv").write_text(
        HEADER_A + "\n2022-01-01,A1,ST4000DM000,,0,5,100,1000,\n")
    assert extract.extract(env)["null_capacity"] == 1


def test_extract_with_no_kept_rows(env):
    (env / "2022-01-01.csv").write_text(
        HEADER_A + "\n2022-01-01,X1,OTHER,1000204886016,0,7,100,3000,\n")
    r = extract.extract(env)
    assert r["rows_out"] == 0 and r["rows_models"] == 0
    assert any(line.endswith("afr=nan%") for line in extract.report_lines(r))
    assert pq.read_table(extract.OUTPUT).num_rows == 0


def test_report_lines_contents(env):
    write_days(env)
    lines = extract.report_lines(extract.extract(env))
    assert "rows_in=7" in lines
    assert "duplicate_serial_date_dropped=1" in lines
    assert "rows_date_not_iso=3" in lines
    assert "rows_date_differs_from_filename=1" in lines
    assert "model=ST4000DM000 drive_days=2 drives=1 failures=1" in lines
    assert any(l.startswith("pooled drive_days=5 failures=1 afr=") for l in lines)
    assert lines[-1].startswith("output=data/processed/backblaze_filtered.parquet sha256=")


def test_update_preprocessing_replaces_only_b1_section(env):
    write_days(env)
    r = extract.extract(env)
    extract.update_preprocessing(r, ["rows_in=7"], {})
    text = extract.PREPROCESSING.read_text(encoding="utf-8")
    assert "- `rows_in=7`" in text
    assert "- stub" not in text
    assert "## B2 — Per-drive timelines\n\n- keep" in text
    extract.update_preprocessing(r, ["rows_in=7"], {})
    assert extract.PREPROCESSING.read_text(encoding="utf-8") == text


def test_update_preprocessing_missing_section_raises(env):
    extract.PREPROCESSING.write_text("# P\n\n## B2 — Per-drive timelines\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="B1 section not found"):
        extract.update_preprocessing({"quarter_dir": env}, [], {})


def test_update_manifest_upserts(env):
    r = {"rows_out": 5, "output_sha256": "abc"}
    extract.update_manifest(r, {"2022-01-01.csv": "d1"})
    extract.update_manifest({"rows_out": 6, "output_sha256": "def"}, {"2022-01-01.csv": "d2"})
    data = json.loads(extract.MANIFEST.read_text(encoding="utf-8"))
    files = {e["file"]: e for e in data["files"]}
    assert set(files) == {"other.parquet", "data/processed/backblaze_filtered.parquet"}
    assert files["data/processed/backblaze_filtered.parquet"]["rows"] == 6
    assert files["data/processed/backblaze_filtered.parquet"]["sha256"] == "def"
    assert data["sources"]["backblaze"] == {"2022-01-01.csv": "d2"}


def test_main_with_explicit_directory(env, capsys):
    write_days(env)
    assert extract.main(["extract.py", str(env)]) == 0
    printed = capsys.readouterr().out
    assert "rows_out=5" in printed
    data = json.loads(extract.MANIFEST.read_text(encoding="utf-8"))
    assert {e["file"] for e in data["files"]} == {
        "other.parquet", "data/processed/backblaze_filtered.parquet"}
    assert data["sources"]["backblaze"]["2022-01-02.csv"] == hashlib.sha256(
        (env / "2022-01-02.csv").read_bytes()).hexdigest()
    entry = next(e for e in data["files"] if e["file"].endswith("filtered.parquet"))
    assert entry["rows"] == 5 and entry["sha256"] == extract.sha256_of(extract.OUTPUT)
    assert "rows_out=5" in extract.PREPROCESSING.read_text(encoding="utf-8")


def test_main_default_directory_and_rerun_is_idempotent(env):
    write_days(env)
    assert extract.main(["extract.py"]) == 0
    first = (extract.PREPROCESSING.read_text(encoding="utf-8"),
             extract.MANIFEST.read_text(encoding="utf-8"))
    assert extract.main(["extract.py"]) == 0
    assert (extract.PREPROCESSING.read_text(encoding="utf-8"),
            extract.MANIFEST.read_text(encoding="utf-8")) == first


def test_cli_entry_point_in_subprocess(tmp_path):
    pipe = tmp_path / "pipe"
    (pipe / "backblaze").mkdir(parents=True)
    shutil.copy(Path(extract.__file__), pipe / "backblaze" / "extract.py")
    (pipe / "manifest.json").write_text('{"files": []}')
    (pipe / "PREPROCESSING.md").write_text("## B1 — Backblaze extract\n\n- stub\n\n## B2 — x\n", encoding="utf-8")
    quarter = tmp_path / "q"
    quarter.mkdir()
    write_days(quarter)
    res = subprocess.run(
        [sys.executable, "-I", str(pipe / "backblaze" / "extract.py"), str(quarter)],
        capture_output=True, text=True, cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "rows_out=5" in res.stdout
    assert (pipe / "data" / "processed" / "backblaze_filtered.parquet").exists()
    assert json.loads((pipe / "manifest.json").read_text())["files"][0]["rows"] == 5


def test_model_filter_constants():
    assert extract.MODELS == ("ST4000DM000", "ST8000NM0055", "ST12000NM0008")
