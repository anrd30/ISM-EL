"""
Unit tests for tencent/build_disk_index.py (B4) on synthetic subscription data.

Path constants of the module are redirected into tmp_path, so no test touches
the real manifest, PREPROCESSING.md or processed data.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("pyarrow")

from contracts import contracts  # noqa: E402
from data_pipeline.tencent import build_disk_index as bd  # noqa: E402
from data_pipeline.tencent import extract_load, inspect_schema  # noqa: E402

# Source order: disk_uid, capacity, disk_attr, disk_type, user_type, vm_cpu, vm_memory.
SUBSCRIPTION = "\n".join([
    "d1,50,1,1,0,2,4",
    "d2,50,1,1,0,1,1",
    "d2,50,1,1,0,1,1",        # repeated, identical
    "dX,50,1,1,0,2,4",
    "dX,100,0,0,1,4,8",       # repeated, attributes differ: first row is kept
    "dY,20,1,0,0,1,2",        # no load data
]) + "\n"

PREPROCESSING_TEXT = (
    "# P\n\n## B3 — Tencent extract\n\n- `conflicting_uid=dX`\n\n"
    "## B4 — Metadata join\n\n- stub\n\n## B5 — Four intensity profiles\n\n- keep\n"
)


def write_clean(path: Path, groups: list[list[str]], statistics: bool = True) -> None:
    schema = pa.schema([pa.field("disk_uid", pa.string()), pa.field("timestamp", pa.int64())])
    with pq.ParquetWriter(path, schema, write_statistics=statistics) as w:
        for uids in groups:
            w.write_table(pa.table({"disk_uid": uids, "timestamp": list(range(len(uids)))},
                                   schema=schema))


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / "pipe"
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"files": [{"file": "other.parquet", "rows": 1, "sha256": "x"}]}))
    prep = root / "PREPROCESSING.md"
    prep.write_text(PREPROCESSING_TEXT, encoding="utf-8")
    clean = processed / "tencent_load_clean.parquet"
    write_clean(clean, [["d1", "d1"], ["d2"], ["dX", "dX"], ["dZ"]])   # dZ: load only
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "disk_subscript_info").write_text(SUBSCRIPTION)
    for name, value in {"ROOT": root, "PROCESSED": processed, "CLEAN": clean,
                        "OUT_INDEX": processed / "tencent_disk_index.parquet",
                        "MANIFEST": manifest, "PREPROCESSING": prep,
                        "DEFAULT_DATASET": raw}.items():
        monkeypatch.setattr(bd, name, value)
    return raw


def sub_frame(text: str = SUBSCRIPTION) -> pd.DataFrame:
    import io
    return bd.read_subscription(io.StringIO(text))


def test_columns_match_contract_and_other_readers():
    assert bd.DISK_INDEX_COLUMNS == contracts.DISK_INDEX_COLUMNS
    assert bd.DISK_INDEX_COLUMNS.index("disk_capacity") == 1
    assert list(bd.DISK_INDEX_COLUMNS) == inspect_schema.SUB_COLUMNS == extract_load.SUB_COLUMNS


def test_read_subscription_assigns_source_order():
    sub = sub_frame()
    assert list(sub.columns) == list(contracts.DISK_INDEX_COLUMNS)
    assert sub.iloc[0].to_dict() == {"disk_uid": "d1", "disk_capacity": 50, "disk_attr": 1,
                                     "disk_type": 1, "user_type": 0, "vm_cpu": 2, "vm_memory": 4}
    assert all(str(sub[c].dtype) == "int64" for c in bd.VALUE_COLUMNS)


def test_read_subscription_rejects_non_integer_values():
    with pytest.raises(SystemExit, match="non-integer columns"):
        sub_frame("d1,50,1,1,0,2.5,4\n")


def test_read_subscription_rejects_an_empty_field_as_missing():
    with pytest.raises(SystemExit, match="missing fields"):
        sub_frame("d1,50,1,1,0,,4\n")


@pytest.mark.parametrize(
    "text, message",
    [
        ("d1,50,1,1,0,2,4,9\nd2,50,1,1,0,1,1\n", "8 columns, expected 7"),   # 8-field first row
        ("d1,50,1,1,0,2,4\nd2,50,1,1,0,1,1,9\n", "Expected 7 fields"),       # 8-field later row
        ("d1,50,1,1,0,2,4\nd2,50,1,1,0,1\n", "missing fields"),               # too few fields
        ("d1,50,1,1,0,2\n", "6 columns, expected 7"),
        ("", "No columns to parse"),
    ],
    ids=["extra_first_row", "extra_later_row", "too_few_later_row", "all_rows_short", "empty"],
)
def test_read_subscription_rejects_wrong_field_counts(text, message):
    with pytest.raises(SystemExit, match=message) as err:
        sub_frame(text)
    assert "subscription" in str(err.value)


def test_dedupe_keeps_first_row_and_reports_conflicts():
    index, info = bd.dedupe_subscription(sub_frame())
    assert index["disk_uid"].tolist() == ["d1", "d2", "dX", "dY"]
    assert index.loc[index["disk_uid"] == "dX", "disk_capacity"].iloc[0] == 50
    assert index.loc[index["disk_uid"] == "dX", "vm_memory"].iloc[0] == 4
    assert info == {"rows_in": 6, "rows_out": 4, "duplicate_rows_dropped": 2,
                    "repeated_uids": 2, "conflicting_uids": ["dX"]}


def test_dedupe_without_duplicates():
    index, info = bd.dedupe_subscription(sub_frame("d1,50,1,1,0,2,4\nd2,50,1,1,0,1,1\n"))
    assert len(index) == 2 and info["duplicate_rows_dropped"] == 0
    assert info["repeated_uids"] == 0 and info["conflicting_uids"] == []


def test_deduped_frame_satisfies_the_contract_validator():
    assert contracts.validate_disk_index_frame(bd.dedupe_subscription(sub_frame())[0]) is None


def test_build_calls_the_contract_validator_on_the_index(env, monkeypatch):
    calls = []
    original = contracts.validate_disk_index_frame

    def spy(frame):
        calls.append(frame.copy())
        return original(frame)

    monkeypatch.setattr(bd.contracts, "validate_disk_index_frame", spy)
    bd.build(env)
    assert len(calls) == 1
    assert list(calls[0].columns) == list(contracts.DISK_INDEX_COLUMNS)
    assert calls[0]["disk_uid"].tolist() == ["d1", "d2", "dX", "dY"]


def _duplicate_uid(good):
    return pd.concat([good, good.iloc[:1]], ignore_index=True)


def _float_vm_cpu(good):
    return good.astype({"vm_cpu": "float64"})


def _reversed_columns(good):
    return good[list(reversed(good.columns))]


@pytest.mark.parametrize(
    "break_frame, message",
    [(_duplicate_uid, "duplicate"), (_float_vm_cpu, "vm_cpu"), (_reversed_columns, "order")],
    ids=["duplicate_disk_uid", "wrong_dtype", "wrong_column_order"],
)
def test_build_rejects_a_broken_frame_before_writing(env, monkeypatch, break_frame, message):
    good = bd.dedupe_subscription(sub_frame())[0]
    monkeypatch.setattr(bd, "dedupe_subscription", lambda sub: (break_frame(good), {}))
    with pytest.raises(contracts.ContractViolation, match=message):
        bd.build(env)
    assert not bd.OUT_INDEX.exists()
    assert not bd.OUT_INDEX.with_suffix(".parquet.tmp").exists()


def test_documented_conflicting_uids():
    assert bd.documented_conflicting_uids(PREPROCESSING_TEXT) == ["dX"]
    two = PREPROCESSING_TEXT.replace("- `conflicting_uid=dX`", "- `conflicting_uid=b`\n- `conflicting_uid=a`")
    assert bd.documented_conflicting_uids(two) == ["a", "b"]
    assert bd.documented_conflicting_uids("# nothing\n") == []


def test_clean_disk_uids_from_statistics(tmp_path):
    p = tmp_path / "clean.parquet"
    write_clean(p, [["a", "a"], ["b"]])
    assert bd.clean_disk_uids(p) == {"a", "b"}


def test_clean_disk_uids_without_usable_statistics(tmp_path):
    p = tmp_path / "nostats.parquet"
    write_clean(p, [["a", "a"], ["b", "c"]], statistics=False)
    assert bd.clean_disk_uids(p) == {"a", "b", "c"}
    q = tmp_path / "mixed.parquet"
    write_clean(q, [["a", "b"]])                       # min != max: rows are read
    assert bd.clean_disk_uids(q) == {"a", "b"}


def test_compare_ids():
    assert bd.compare_ids({"a", "b"}, {"b", "c"}) == (["a"], ["c"])
    assert bd.compare_ids({"a"}, {"a"}) == ([], [])


def test_build_outputs_and_flags_mismatches(env):
    r = bd.build(env)
    assert r["rows_out"] == 4 and r["duplicate_rows_dropped"] == 2
    assert r["subscription_only"] == ["dY"] and r["load_only"] == ["dZ"]
    assert r["documented_conflicting_uids"] == ["dX"] == r["conflicting_uids"]

    out = pq.read_table(bd.OUT_INDEX)
    assert out.column_names == list(contracts.DISK_INDEX_COLUMNS)
    assert [str(f.type) for f in out.schema] == ["string"] + ["int64"] * 6
    index = out.to_pandas()
    assert index["disk_uid"].tolist() == ["d1", "d2", "dX", "dY"]     # dY kept: no load rows
    assert "dZ" not in set(index["disk_uid"])
    assert r["output_sha256"] == bd.sha256_of(bd.OUT_INDEX)
    assert not bd.OUT_INDEX.with_suffix(".parquet.tmp").exists()


def test_build_requires_clean_table(env):
    bd.CLEAN.unlink()
    with pytest.raises(SystemExit, match="missing tencent_load_clean.parquet"):
        bd.build(env)


def test_report_lines(env):
    lines = bd.report_lines(bd.build(env))
    assert "subscription_rows_in=6" in lines
    assert "duplicate_rows_dropped=2 repeated_uids=2" in lines
    assert "disk_index_rows=4" in lines
    assert "conflicting_uids=1 documented_in_b3=1 match=True" in lines
    assert "conflicting_uid=dX" in lines
    assert "subscription_only_uids=1 load_only_uids=1" in lines
    assert "subscription_only_uid=dY" in lines and "load_only_uid=dZ" in lines
    assert lines[-1].startswith("output=data/processed/tencent_disk_index.parquet rows=4 sha256=")


def test_report_flags_documented_mismatch(env):
    bd.PREPROCESSING.write_text(PREPROCESSING_TEXT.replace("dX", "dQ"), encoding="utf-8")
    assert "conflicting_uids=1 documented_in_b3=1 match=False" in bd.report_lines(bd.build(env))


def test_report_lists_at_most_ten_mismatches():
    r = {"rows_in": 0, "rows_out": 0, "duplicate_rows_dropped": 0, "repeated_uids": 0,
         "conflicting_uids": [], "documented_conflicting_uids": [],
         "subscription_only": [f"s{i:02d}" for i in range(25)], "load_only": [],
         "output_sha256": "h"}
    lines = bd.report_lines(r)
    assert "subscription_only_uids=25 load_only_uids=0" in lines
    assert sum(l.startswith("subscription_only_uid=") for l in lines) == bd.LISTED_MISMATCHES


def test_update_preprocessing_replaces_only_b4_section(env):
    bd.update_preprocessing(["disk_index_rows=4", "conflicting_uid=dX"])
    text = bd.PREPROCESSING.read_text(encoding="utf-8")
    assert "- `disk_index_rows=4`" in text and "- stub" not in text
    assert "## B3 — Tencent extract\n\n- `conflicting_uid=dX`" in text
    assert "## B5 — Four intensity profiles\n\n- keep" in text
    bd.update_preprocessing(["disk_index_rows=4", "conflicting_uid=dX"])
    assert bd.PREPROCESSING.read_text(encoding="utf-8") == text


def test_update_preprocessing_missing_section_raises(env):
    bd.PREPROCESSING.write_text("# P\n\n## B5 — x\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="B4 section not found"):
        bd.update_preprocessing([])


def test_update_manifest_upserts(env):
    bd.update_manifest({"rows_out": 4, "output_sha256": "a"})
    bd.update_manifest({"rows_out": 5, "output_sha256": "b"})
    files = json.loads(bd.MANIFEST.read_text(encoding="utf-8"))["files"]
    assert [e["file"] for e in files] == ["other.parquet", "data/processed/tencent_disk_index.parquet"]
    assert files[1]["rows"] == 5 and files[1]["sha256"] == "b"


def test_main_explicit_and_default_root_are_idempotent(env, capsys):
    assert bd.main(["build_disk_index.py", str(env)]) == 0
    assert "disk_index_rows=4" in capsys.readouterr().out
    first = (bd.PREPROCESSING.read_text(encoding="utf-8"), bd.MANIFEST.read_text(encoding="utf-8"))
    assert bd.main(["build_disk_index.py"]) == 0
    assert (bd.PREPROCESSING.read_text(encoding="utf-8"), bd.MANIFEST.read_text(encoding="utf-8")) == first
    assert "conflicting_uid=dX" in first[0]


def test_cli_entry_point_in_subprocess(tmp_path, env):
    repo = tmp_path / "repo"
    pipe = repo / "data_pipeline"
    (pipe / "tencent").mkdir(parents=True)
    (pipe / "data" / "processed").mkdir(parents=True)
    shutil.copytree(Path(contracts.__file__).parent, repo / "contracts",
                    ignore=shutil.ignore_patterns("__pycache__", "test_*"))
    shutil.copy(Path(bd.__file__), pipe / "tencent" / "build_disk_index.py")
    shutil.copy(bd.CLEAN, pipe / "data" / "processed" / "tencent_load_clean.parquet")
    (pipe / "manifest.json").write_text('{"files": []}')
    (pipe / "PREPROCESSING.md").write_text(PREPROCESSING_TEXT, encoding="utf-8")
    res = subprocess.run(
        [sys.executable, "-I", str(pipe / "tencent" / "build_disk_index.py"), str(env)],
        capture_output=True, text=True, cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "disk_index_rows=4" in res.stdout
    assert (pipe / "data" / "processed" / "tencent_disk_index.parquet").exists()
    assert len(json.loads((pipe / "manifest.json").read_text())["files"]) == 1
