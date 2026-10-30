"""
Unit tests for backblaze/timelines.py (B2) on small synthetic drive histories.

Expected values are worked out by hand from the window rule: the N-day
difference is value(today) - value(earliest observed row with
date >= today - N), NaN when fewer than MIN_OBS_DIFF rows lie in
[today - N, today] or either endpoint is NaN. Path constants of the module
are redirected into tmp_path so no test touches real outputs.
"""
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("pyarrow")

from data_pipeline.backblaze import timelines  # noqa: E402

NAN = float("nan")


def day(n: int) -> dt.date:
    return dt.date(2022, 1, n)


def make_frame(drives: dict[str, list[tuple]], model: str = "M") -> pd.DataFrame:
    """drives: serial -> [(day_of_january, smart_5_raw, failure), ...]."""
    rows = []
    for serial, history in drives.items():
        for d, s5, failure in history:
            row = {"serial_number": serial, "model": model, "date": day(d),
                   "capacity_bytes": 4_000_000_000_000, "failure": failure}
            row.update({c: 1.0 for c in timelines.SMART_COLUMNS})
            row["smart_5_raw"] = s5
            rows.append(row)
    return pd.DataFrame(rows)


def series(out: pd.DataFrame, serial: str, col: str) -> list:
    return out.loc[out["serial_number"] == serial, col].tolist()


WINDOW_DRIVE = [(1, 10.0, 0), (2, 12.0, 0), (10, 20.0, 0), (11, NAN, 0),
                (12, 30.0, 0), (18, 50.0, 0)]


def test_feature_spec_has_no_mean_and_drops_188_level():
    names = [n for n, _, _ in timelines.feature_spec()]
    assert len(names) == 25
    assert "smart_188_raw" not in names
    assert "smart_188_raw_diff30d" not in names and "smart_188_raw_diff7d" in names
    assert not any("mean" in n for n in names)
    assert timelines.MIN_OBS_DIFF == 2


def test_diff_nan_when_window_has_too_few_observations():
    out = timelines.build_timelines(make_frame({"S": WINDOW_DRIVE}))
    d7 = series(out, "S", "smart_5_raw_diff7d")
    # d1 and d10 have a single row in their 7-day window.
    assert np.isnan(d7[0]) and np.isnan(d7[2])
    assert d7[1] == 2.0
    flags = series(out, "S", "smart_5_raw_diff7d_missing")
    assert flags[0] and not flags[1] and flags[2]


def test_diff_nan_when_an_endpoint_is_nan():
    out = timelines.build_timelines(make_frame({"S": WINDOW_DRIVE}))
    d7 = series(out, "S", "smart_5_raw_diff7d")
    d30 = series(out, "S", "smart_5_raw_diff30d")
    # d11: own value NaN. d18: earliest row in [11, 18] is d11, whose value is NaN.
    assert np.isnan(d7[3]) and np.isnan(d7[5])
    assert d7[4] == 10.0                      # d12 vs d10 (window [5, 12])
    assert np.isnan(d30[3])                   # own value NaN
    assert d30[2] == 10.0 and d30[4] == 20.0 and d30[5] == 40.0   # all measured from d1


def test_missing_flags_equal_nan_mask_for_every_feature():
    out = timelines.build_timelines(make_frame({"S": WINDOW_DRIVE, "T": WINDOW_DRIVE[:3]}))
    for name, _, _ in timelines.feature_spec():
        assert out[f"{name}_missing"].tolist() == out[name].isna().tolist()
    assert series(out, "S", "smart_5_raw_missing") == [False, False, False, True, False, False]


def test_windows_do_not_cross_drives():
    frame = make_frame({"S": [(1, 10.0, 0), (2, 12.0, 0)], "T": [(2, 5.0, 0), (3, 9.0, 0)]})
    out = timelines.build_timelines(frame)
    t7 = series(out, "T", "smart_5_raw_diff7d")
    assert np.isnan(t7[0]) and t7[1] == 4.0


def test_structural_columns():
    out = timelines.build_timelines(make_frame({"S": WINDOW_DRIVE[:3]}))
    assert series(out, "S", "gap_days_before")[1:] == [1, 8]
    assert pd.isna(series(out, "S", "gap_days_before")[0])
    assert series(out, "S", "days_since_first_seen") == [0, 1, 9]
    assert set(out["first_seen"]) == {pd.Timestamp("2022-01-01")}
    assert out["failure"].dtype == np.dtype("int8")


def test_unsorted_input_is_sorted_and_duplicates_raise():
    shuffled = make_frame({"S": WINDOW_DRIVE[:3]}).iloc[::-1]
    out = timelines.build_timelines(shuffled)
    assert out["date"].is_monotonic_increasing
    dup = pd.concat([shuffled, shuffled.iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        timelines.build_timelines(dup)


def drive_index_frame() -> pd.DataFrame:
    return make_frame({
        "ALIVE": [(d, 1.0, 0) for d in range(1, 6)],
        "FAILS": [(1, 1.0, 0), (2, 1.0, 0), (3, 1.0, 0), (4, 1.0, 1)],
        "GAPPY": [(1, 1.0, 0), (3, 1.0, 0), (9, 1.0, 0)],
        "DOUBLE": [(1, 1.0, 0), (2, 1.0, 1), (3, 1.0, 1)],
    })


def test_drive_index_multi_drive():
    idx = timelines.build_drive_index(drive_index_frame()).set_index("serial_number")
    assert list(idx.index) == ["ALIVE", "DOUBLE", "FAILS", "GAPPY"]
    assert list(timelines.build_drive_index(drive_index_frame()).columns) == [
        "serial_number", "model", "first_seen", "last_seen",
        "days_observed", "failed_in_window", "failure_date"]

    alive = idx.loc["ALIVE"]
    assert (alive["first_seen"], alive["last_seen"]) == (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-01-05"))
    assert alive["days_observed"] == 5 and not alive["failed_in_window"]
    assert pd.isna(alive["failure_date"])

    fails = idx.loc["FAILS"]
    assert fails["failed_in_window"] and fails["failure_date"] == pd.Timestamp("2022-01-04")
    assert fails["last_seen"] == pd.Timestamp("2022-01-04") and fails["days_observed"] == 4

    gappy = idx.loc["GAPPY"]
    assert gappy["days_observed"] == 3 and gappy["last_seen"] == pd.Timestamp("2022-01-09")
    assert not gappy["failed_in_window"]

    assert idx.loc["DOUBLE", "failure_date"] == pd.Timestamp("2022-01-02")
    assert idx["model"].eq("M").all()
    assert idx["days_observed"].dtype == np.dtype("int32")
    assert idx["failed_in_window"].dtype == np.dtype("bool")


def test_to_arrow_converts_datetimes_to_date32():
    frame = timelines.build_timelines(make_frame({"S": WINDOW_DRIVE[:3]}))
    table = timelines.to_arrow(frame)
    assert table.schema.field("date").type == pa.date32()
    assert table.schema.field("first_seen").type == pa.date32()
    index_table = timelines.to_arrow(timelines.build_drive_index(drive_index_frame()))
    assert index_table.schema.field("failure_date").type == pa.date32()
    assert index_table.column("failure_date").null_count == 2


def test_sha256_of_matches_hashlib(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"xyz" * 500)
    assert timelines.sha256_of(p) == hashlib.sha256(b"xyz" * 500).hexdigest()


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / "pipe"
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"files": [{"file": "other.parquet", "rows": 1, "sha256": "x"}]}))
    prep = root / "PREPROCESSING.md"
    prep.write_text("# P\n\n## B2 — Per-drive timelines\n\n- stub\n\n## B3 — Tencent extract\n\n- keep\n",
                    encoding="utf-8")
    src = processed / "backblaze_filtered.parquet"
    frame = drive_index_frame()
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), src)
    for name, value in {
        "ROOT": root, "PROCESSED": processed, "INPUT": src, "MANIFEST": manifest,
        "PREPROCESSING": prep,
        "OUT_TIMELINES": processed / "backblaze_timelines.parquet",
        "OUT_INDEX": processed / "backblaze_drive_index.parquet",
    }.items():
        monkeypatch.setattr(timelines, name, value)
    return frame


def test_update_manifest_upserts(env):
    entry = {"file": "data/processed/backblaze_timelines.parquet", "rows": 1, "sha256": "a"}
    timelines.update_manifest([entry])
    timelines.update_manifest([{**entry, "rows": 2, "sha256": "b"}])
    files = json.loads(timelines.MANIFEST.read_text(encoding="utf-8"))["files"]
    assert [e["file"] for e in files] == ["other.parquet", entry["file"]]
    assert files[1]["rows"] == 2 and files[1]["sha256"] == "b"


def test_update_preprocessing_replaces_only_b2_section(env):
    timelines.update_preprocessing(["rows_in=14"])
    text = timelines.PREPROCESSING.read_text(encoding="utf-8")
    assert "- `rows_in=14`" in text and "- stub" not in text
    assert "## B3 — Tencent extract\n\n- keep" in text
    timelines.update_preprocessing(["rows_in=14"])
    assert timelines.PREPROCESSING.read_text(encoding="utf-8") == text


def test_update_preprocessing_missing_section_raises(env):
    timelines.PREPROCESSING.write_text("# P\n\n## B3 — x\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="B2 section not found"):
        timelines.update_preprocessing([])


def test_main_default_input(env, capsys):
    assert timelines.main(["timelines.py"]) == 0
    printed = capsys.readouterr().out
    assert "rows_in=15 drives=4" in printed
    assert "drive_index_rows=4 failed_in_window=2" in printed
    assert "drives_with_multiple_failure_rows=1" in printed
    assert "rows_first_observation=4" in printed
    assert "rows_with_gap_days_before_gt_1=" in printed

    tl = pq.read_table(timelines.OUT_TIMELINES).to_pandas()
    ix = pq.read_table(timelines.OUT_INDEX).to_pandas()
    assert len(tl) == 15 and len(ix) == 4
    assert {"last_seen", "failed_in_window"}.isdisjoint(tl.columns)

    files = {e["file"]: e for e in json.loads(timelines.MANIFEST.read_text(encoding="utf-8"))["files"]}
    assert set(files) == {"other.parquet", "data/processed/backblaze_timelines.parquet",
                          "data/processed/backblaze_drive_index.parquet"}
    assert files["data/processed/backblaze_timelines.parquet"]["rows"] == 15
    assert files["data/processed/backblaze_drive_index.parquet"]["sha256"] == \
        timelines.sha256_of(timelines.OUT_INDEX)
    assert "drive_index_rows=4" in timelines.PREPROCESSING.read_text(encoding="utf-8")


def test_main_explicit_input_and_rerun_is_idempotent(env, tmp_path):
    other = tmp_path / "other_input.parquet"
    pq.write_table(pa.Table.from_pandas(env, preserve_index=False), other)
    assert timelines.main(["timelines.py", str(other)]) == 0
    first = (timelines.PREPROCESSING.read_text(encoding="utf-8"),
             timelines.MANIFEST.read_text(encoding="utf-8"))
    assert timelines.main(["timelines.py", str(other)]) == 0
    assert (timelines.PREPROCESSING.read_text(encoding="utf-8"),
            timelines.MANIFEST.read_text(encoding="utf-8")) == first


def test_cli_entry_point_in_subprocess(tmp_path):
    pipe = tmp_path / "pipe"
    (pipe / "backblaze").mkdir(parents=True)
    (pipe / "data" / "processed").mkdir(parents=True)
    shutil.copy(Path(timelines.__file__), pipe / "backblaze" / "timelines.py")
    (pipe / "manifest.json").write_text('{"files": []}')
    (pipe / "PREPROCESSING.md").write_text("## B2 — Per-drive timelines\n\n- stub\n\n## B3 — x\n",
                                           encoding="utf-8")
    src = pipe / "data" / "processed" / "backblaze_filtered.parquet"
    pq.write_table(pa.Table.from_pandas(drive_index_frame(), preserve_index=False), src)
    res = subprocess.run([sys.executable, "-I", str(pipe / "backblaze" / "timelines.py")],
                         capture_output=True, text=True, cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "timelines_rows=15" in res.stdout
    assert (pipe / "data" / "processed" / "backblaze_timelines.parquet").exists()
    assert len(json.loads((pipe / "manifest.json").read_text())["files"]) == 2
