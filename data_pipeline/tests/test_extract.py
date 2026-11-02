"""
Unit tests for backblaze/extract.py (B1) on small synthetic daily CSVs.

All path constants of the module are redirected into tmp_path, so no test
touches the real manifest, PREPROCESSING.md, processed data or raw data.
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

# 2022-01-02 is ISO, lacks smart_9_raw and has smart_187_raw.
DAY2 = "\n".join([
    HEADER_B,
    "2022-01-02,A1,ST4000DM000,4000787030016,1,5,9,",
    "2022-01-02,A3,ST12000NM0008,12000138625024,0,8,0,",
    "2022-01-02,A2,ST8000NM0055,8001563222016,0,6,1,",
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
    monkeypatch.setattr(extract, "QUARTER_DIRS", (quarter,))
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
    assert r["date_mismatch"] == 0
    assert r["null_capacity"] == 0
    assert r["smart_cols"] == list(extract.SMART_COLUMNS)
    assert r["all_null_cols"] == ["smart_12_raw", "smart_188_raw", "smart_193_raw", "smart_194_raw",
                                  "smart_197_raw", "smart_198_raw", "smart_199_raw"]
    assert r["missing_per_file"] == {"2022-01-01.csv": 7, "2022-01-02.csv": 7}
    assert r["first_date"].isoformat() == "2022-01-01" and r["last_date"].isoformat() == "2022-01-02"
    assert r["per_quarter"] == {"q": {"files": 2, "failures": 1, "seagate_failures": 1}}
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


def test_a_date_that_differs_from_the_file_name_aborts_the_run(env):
    (env / "2022-01-02.csv").write_text(
        HEADER_A + "\n2022-01-02,A1,ST4000DM000,4000787030016,0,5,100,1000,\n"
        "2022-01-03,A2,ST8000NM0055,8001563222016,0,6,100,2000,\n")
    with pytest.raises(SystemExit, match=r"2022-01-02\.csv: 1 rows with a date different from the file name"):
        extract.extract(env)
    assert not extract.OUTPUT.exists()


def test_a_non_iso_date_that_differs_from_the_file_name_also_aborts(env):
    (env / "2022-01-01.csv").write_text(
        HEADER_A + "\n01-05-2022,A1,ST4000DM000,4000787030016,0,5,100,1000,\n"
        "01-01-2022,A2,ST8000NM0055,8001563222016,0,6,100,2000,\n")
    with pytest.raises(SystemExit, match=r"2022-01-01\.csv: 1 rows with a date different"):
        extract.extract(env)


def test_rows_in_other_models_are_not_checked_against_the_file_name(env):
    (env / "2022-01-01.csv").write_text(
        HEADER_A + "\n2022-01-01,A1,ST4000DM000,4000787030016,0,5,100,1000,\n"
        "2030-09-09,X1,OTHER,1000204886016,0,7,100,3000,\n")
    assert extract.extract(env)["rows_out"] == 1


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
    assert "rows_date_differs_from_filename=0" in lines
    assert any(l.startswith("model=ST4000DM000 drive_days=2 drives=1 failures=1 ") for l in lines)
    assert "daily_files=2 date_first=2022-01-01 date_last=2022-01-02" in lines
    assert "quarter=q files=2 failures=1 seagate_failures=1" in lines
    assert "verification_against_a1_decision=not run" in lines
    assert "excluded_files=none rows_excluded=0" in lines
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
        extract.update_preprocessing(
            {"quarter_dirs": [env], "excluded_files": [], "rows_excluded": 0}, [], {})


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


def test_main_with_explicit_directory(env, capsys, monkeypatch):
    monkeypatch.setattr(extract, "QUARTER_DIRS", (env.parent / "not_the_input",))
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


def test_main_default_directory_and_rerun_is_idempotent(recorded):
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
    assert extract.MODELS == ("ST4000DM000", "ST8000NM0055", "ST12000NM0008", "TOSHIBA MG07ACA14TA")
    assert extract.SEAGATE_MODELS == extract.MODELS[:3]
    assert len(extract.QUARTER_DIRS) == 4
    assert [d.name for d in extract.QUARTER_DIRS] == [f"backblaze_2022Q{q}" for q in (1, 2, 3, 4)]


def test_recorded_values_are_consistent():
    failures = sum(f for f, _ in extract.EXPECTED_PER_MODEL.values())
    days = sum(d for _, d in extract.EXPECTED_PER_MODEL.values())
    assert (failures, days) == extract.EXPECTED_POOLED == (1770, 33_052_284)
    assert sum(extract.EXPECTED_SEAGATE_FAILURES_BY_QUARTER.values()) == sum(
        extract.EXPECTED_PER_MODEL[m][0] for m in extract.SEAGATE_MODELS)
    assert set(extract.EXPECTED_PER_MODEL) == set(extract.MODELS)
    assert extract.EXPECTED_FILES == 364
    assert extract.EXCLUDED_FILES == {
        "backblaze_2022Q4/2022-12-10.csv": "all rows dated 2022-12-11; copy of the 2022-12-11 file"}


def test_the_smart_column_list_is_the_timeline_input_and_nothing_else():
    from data_pipeline.backblaze import timelines
    assert tuple(timelines.SMART_COLUMNS) == extract.SMART_COLUMNS
    assert extract.build_column_list() == extract.CORE_COLUMNS + list(extract.SMART_COLUMNS)


# -- several quarter directories, file checks, four models, recorded-value verification --

TOSHIBA = "TOSHIBA MG07ACA14TA"


def write_csv(directory: Path, day: str, rows: list[str], header: str = HEADER_A) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{day}.csv").write_text("\n".join([header, *rows]) + "\n")


def test_several_quarter_directories_are_read_in_date_order(env, tmp_path):
    later, earlier = tmp_path / "later", tmp_path / "earlier"
    write_csv(later, "2022-01-03", ["2022-01-03,A1,ST4000DM000,4000787030016,0,5,100,1000,"])
    write_csv(later, "2022-01-04", ["2022-01-04,A1,ST4000DM000,4000787030016,1,5,100,1000,"])
    write_csv(earlier, "2022-01-01", ["2022-01-01,A1,ST4000DM000,4000787030016,0,5,100,1000,"])
    write_csv(earlier, "2022-01-02", ["2022-01-02,A1,ST4000DM000,4000787030016,0,5,100,1000,"])
    r = extract.extract([later, earlier])
    out = pq.read_table(extract.OUTPUT).to_pandas()
    assert [str(d) for d in out["date"]] == ["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04"]
    assert [f.name for f in r["files"]] == [f"2022-01-0{i}.csv" for i in (1, 2, 3, 4)]
    assert r["per_quarter"] == {"later": {"files": 2, "failures": 1, "seagate_failures": 1},
                                "earlier": {"files": 2, "failures": 0, "seagate_failures": 0}}
    assert r["rows_out"] == 4
    assert extract.report_lines(r)[0] == "quarter_dirs=later,earlier"


def test_a_single_directory_given_as_a_path_or_a_list_gives_the_same_result(env):
    write_days(env)
    as_path = extract.extract(env)
    as_list = extract.extract([env])
    assert as_path["rows_out"] == as_list["rows_out"] == 5
    assert as_path["output_sha256"] == as_list["output_sha256"]


def test_a_date_present_in_two_files_aborts_the_run(env, tmp_path):
    other = tmp_path / "other"
    write_csv(env, "2022-01-01", ["2022-01-01,A1,ST4000DM000,4000787030016,0,5,100,1000,"])
    write_csv(other, "2022-01-01", ["2022-01-01,A2,ST8000NM0055,8001563222016,0,6,100,2000,"])
    with pytest.raises(SystemExit, match=r"date 2022-01-01 appears in more than one file"):
        extract.extract([env, other])
    assert not extract.OUTPUT.exists()


def test_the_file_count_must_equal_the_calendar_days(env):
    write_csv(env, "2022-01-01", ["2022-01-01,A1,ST4000DM000,4000787030016,0,5,100,1000,"])
    write_csv(env, "2022-01-04", ["2022-01-04,A1,ST4000DM000,4000787030016,0,5,100,1000,"])
    with pytest.raises(SystemExit, match=r"2 files for 4 calendar days \(2022-01-01 to 2022-01-04\)"):
        extract.extract(env)
    assert not extract.OUTPUT.exists()


def test_a_file_name_that_is_not_an_iso_date_aborts(env):
    (env / "day1.csv").write_text(HEADER_A + "\n")
    with pytest.raises(SystemExit, match="file name is not an ISO date"):
        extract.extract(env)


def test_an_empty_directory_among_several_aborts(env, tmp_path):
    write_days(env)
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no CSV files in"):
        extract.extract([env, tmp_path / "empty"])


def test_the_four_model_filter_keeps_toshiba_and_drops_other_models(env):
    write_csv(env, "2022-01-01", [
        "2022-01-01,A1,ST4000DM000,4000787030016,0,5,100,1000,",
        f"2022-01-01,T1,{TOSHIBA},14000519643136,1,5,100,1000,",
        "2022-01-01,T2,TOSHIBA MG08ACA16TE,16000900661248,1,5,100,1000,",
        "2022-01-01,X1,OTHER,1000204886016,0,7,100,3000,",
    ])
    r = extract.extract(env)
    out = pq.read_table(extract.OUTPUT).to_pandas()
    assert sorted(out["model"]) == sorted(["ST4000DM000", TOSHIBA])
    assert r["rows_models"] == 2 and r["rows_in"] == 4
    assert r["per_model"][TOSHIBA]["failures"] == 1 and r["per_model"][TOSHIBA]["drives"] == {"T1"}


def test_columns_a_model_does_not_report_stay_nan_for_that_model(env):
    header = ("date,serial_number,model,capacity_bytes,failure,smart_5_raw,smart_187_raw,"
              "smart_188_raw,smart_199_raw")
    write_csv(env, "2022-01-01", [
        "2022-01-01,A1,ST4000DM000,4000787030016,0,5,3,4,6",
        f"2022-01-01,T1,{TOSHIBA},14000519643136,0,5,,,6",
    ], header=header)
    r = extract.extract(env)
    out = pq.read_table(extract.OUTPUT).to_pandas().set_index("serial_number")
    assert out.loc["T1", ["smart_187_raw", "smart_188_raw"]].isna().all()
    assert out.loc["T1", "smart_199_raw"] == 6 and out.loc["A1", "smart_187_raw"] == 3
    assert r["all_null_by_model"][TOSHIBA] == [
        "smart_9_raw", "smart_12_raw", "smart_187_raw", "smart_188_raw", "smart_193_raw",
        "smart_194_raw", "smart_197_raw", "smart_198_raw"]
    assert "smart_187_raw" not in r["all_null_by_model"]["ST4000DM000"]
    assert not (out[["smart_187_raw", "smart_188_raw"]].fillna(-1) == 0).any().any()


@pytest.fixture
def recorded(env, monkeypatch):
    """Recorded values that match the two-day fixture."""
    write_days(env)
    monkeypatch.setattr(extract, "EXPECTED_FILES", 2)
    monkeypatch.setattr(extract, "EXPECTED_DATE_RANGE", ("2022-01-01", "2022-01-02"))
    monkeypatch.setattr(extract, "EXPECTED_PER_MODEL", {
        "ST4000DM000": (1, 2), "ST8000NM0055": (0, 2), "ST12000NM0008": (0, 1), TOSHIBA: (0, 0)})
    monkeypatch.setattr(extract, "EXPECTED_POOLED", (1, 5))
    monkeypatch.setattr(extract, "EXPECTED_SEAGATE_FAILURES_BY_QUARTER", {"q": 1})
    return env


def test_verification_passes_on_matching_values_and_is_reported(recorded):
    r = extract.extract(recorded, check=extract.verify_expected)
    assert r["verified"] is True
    assert "verification_against_a1_decision=PASS" in extract.report_lines(r)


def test_default_run_verifies_against_the_recorded_values(recorded, capsys):
    assert extract.main(["extract.py"]) == 0
    assert "verification_against_a1_decision=PASS" in capsys.readouterr().out


@pytest.mark.parametrize("constant, value, message", [
    ("EXPECTED_PER_MODEL", {"ST4000DM000": (2, 2), "ST8000NM0055": (0, 2), "ST12000NM0008": (0, 1),
                            TOSHIBA: (0, 0)}, "ST4000DM000 failures: expected 2, measured 1"),
    ("EXPECTED_PER_MODEL", {"ST4000DM000": (1, 3), "ST8000NM0055": (0, 2), "ST12000NM0008": (0, 1),
                            TOSHIBA: (0, 0)}, "ST4000DM000 drive_days: expected 3, measured 2"),
    ("EXPECTED_POOLED", (2, 5), "pooled failures: expected 2, measured 1"),
    ("EXPECTED_POOLED", (1, 6), "pooled drive_days: expected 6, measured 5"),
    ("EXPECTED_FILES", 3, "files: expected 3, measured 2"),
    ("EXPECTED_DATE_RANGE", ("2022-01-01", "2022-01-03"), "date_range"),
    ("EXPECTED_SEAGATE_FAILURES_BY_QUARTER", {"q": 2}, "q seagate failures: expected 2, measured 1"),
])
def test_a_count_that_differs_from_the_recorded_value_aborts_without_writing(
        recorded, monkeypatch, constant, value, message):
    monkeypatch.setattr(extract, constant, value)
    with pytest.raises(SystemExit, match="verification against A1_DECISION.md failed") as err:
        extract.main(["extract.py"])
    assert message in str(err.value)
    assert not extract.OUTPUT.exists()
    assert not extract.OUTPUT.with_suffix(".parquet.tmp").exists()
    assert "- stub" in extract.PREPROCESSING.read_text(encoding="utf-8")


def test_a_failed_verification_keeps_the_previous_output(recorded, monkeypatch):
    extract.extract(recorded)
    before = extract.sha256_of(extract.OUTPUT)
    monkeypatch.setattr(extract, "EXPECTED_POOLED", (9, 9))
    with pytest.raises(SystemExit):
        extract.extract(recorded, check=extract.verify_expected)
    assert extract.sha256_of(extract.OUTPUT) == before


def test_a_run_over_other_directories_does_not_compare_with_the_recorded_values(env, tmp_path, capsys):
    other = tmp_path / "other"
    write_csv(other, "2022-01-01", ["2022-01-01,A1,ST4000DM000,4000787030016,0,5,100,1000,"])
    assert extract.main(["extract.py", str(other)]) == 0
    assert "verification_against_a1_decision=not run" in capsys.readouterr().out


def test_preprocessing_section_lists_the_quarter_directories_and_references_the_decision(recorded):
    r = extract.extract(recorded)
    extract.update_preprocessing(r, extract.report_lines(r), {})
    text = extract.PREPROCESSING.read_text(encoding="utf-8")
    assert "A1_DECISION.md" in text and "provisional" not in text
    assert "`q`" in text and "TOSHIBA MG07ACA14TA" in text


# -- named excluded files --

def write_three_days(directory: Path, middle_date_in_rows: str = "2022-01-03") -> None:
    write_csv(directory, "2022-01-01", ["2022-01-01,A1,ST4000DM000,4000787030016,0,5,100,1000,"])
    write_csv(directory, "2022-01-02", [
        f"{middle_date_in_rows},A1,ST4000DM000,4000787030016,1,5,100,1000,",
        f"{middle_date_in_rows},A2,ST8000NM0055,8001563222016,0,6,100,2000,",
        f"{middle_date_in_rows},X1,OTHER,1000204886016,0,7,100,3000,",
    ])
    write_csv(directory, "2022-01-03", ["2022-01-03,A1,ST4000DM000,4000787030016,0,5,100,1000,"])


def test_an_excluded_file_is_skipped_and_reported(env, monkeypatch):
    monkeypatch.setattr(extract, "EXCLUDED_FILES", {"q/2022-01-02.csv": "test reason"})
    write_three_days(env)
    r = extract.extract(env)
    out = pq.read_table(extract.OUTPUT).to_pandas()
    assert [str(d) for d in out["date"]] == ["2022-01-01", "2022-01-03"]
    assert r["rows_out"] == 2 and r["excluded_files"] == ["q/2022-01-02.csv"]
    assert r["rows_excluded"] == 2          # kept-model rows only; the OTHER row is not counted
    assert [f.name for f in r["files"]] == ["2022-01-01.csv", "2022-01-03.csv"]
    assert r["per_model"]["ST4000DM000"]["failures"] == 0
    lines = extract.report_lines(r)
    assert "excluded_files=q/2022-01-02.csv rows_excluded=2" in lines
    assert "daily_files=2 date_first=2022-01-01 date_last=2022-01-03" in lines
    extract.update_preprocessing(r, lines, {})
    text = extract.PREPROCESSING.read_text(encoding="utf-8")
    assert "`q/2022-01-02.csv` (test reason)" in text and "permitted gap" in text


def test_a_mismatching_file_that_is_not_excluded_still_aborts(env, monkeypatch):
    monkeypatch.setattr(extract, "EXCLUDED_FILES", {"q/2022-01-03.csv": "test reason"})
    write_three_days(env)
    with pytest.raises(SystemExit, match=r"2022-01-02\.csv: 2 rows with a date different from the file name"):
        extract.extract(env)
    assert not extract.OUTPUT.exists()


def test_a_gap_that_is_not_listed_fails_the_span_check(env, monkeypatch):
    monkeypatch.setattr(extract, "EXCLUDED_FILES", {"q/2022-01-02.csv": "test reason"})
    write_csv(env, "2022-01-01", ["2022-01-01,A1,ST4000DM000,4000787030016,0,5,100,1000,"])
    write_csv(env, "2022-01-02", ["2022-01-03,A1,ST4000DM000,4000787030016,0,5,100,1000,"])
    write_csv(env, "2022-01-04", ["2022-01-04,A1,ST4000DM000,4000787030016,0,5,100,1000,"])
    with pytest.raises(SystemExit, match=r"2 files for 4 calendar days .*\['2022-01-03'\]"):
        extract.extract(env)
    assert not extract.OUTPUT.exists()


def test_an_excluded_name_in_another_directory_is_not_skipped(env, monkeypatch):
    monkeypatch.setattr(extract, "EXCLUDED_FILES", {"other/2022-01-02.csv": "test reason"})
    write_three_days(env)
    with pytest.raises(SystemExit, match="rows with a date different from the file name"):
        extract.extract(env)
