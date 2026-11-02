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
    """Day n counted from 2022-01-01 (day 1)."""
    return dt.date(2022, 1, 1) + dt.timedelta(days=n - 1)


def make_frame(drives: dict[str, list[tuple]], model: str = "M") -> pd.DataFrame:
    """drives: serial -> [(day number from 2022-01-01 = 1, smart_5_raw, failure), ...]."""
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
    assert len(names) == 28
    assert {"smart_199_raw", "smart_199_raw_diff7d", "smart_199_raw_diff30d"} <= set(names)
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


def test_columns_with_no_values_for_a_model_give_nan_features_not_zero():
    frame = pd.concat([make_frame({"S": WINDOW_DRIVE}, model="SEAGATE"),
                       make_frame({"T": WINDOW_DRIVE}, model="TOSHIBA")], ignore_index=True)
    frame.loc[frame["model"] == "TOSHIBA", ["smart_187_raw", "smart_188_raw"]] = np.nan
    out = timelines.build_timelines(frame)
    no_values = out[out["serial_number"] == "T"]
    for name in ("smart_187_raw", "smart_187_raw_diff7d", "smart_187_raw_diff30d", "smart_188_raw_diff7d"):
        assert no_values[name].isna().all(), name
        assert no_values[f"{name}_missing"].all(), name
    with_values = out[out["serial_number"] == "S"]
    assert with_values["smart_187_raw"].notna().all()
    assert with_values["smart_187_raw_diff7d"].notna().any()
    assert no_values["smart_5_raw"].notna().sum() == 5 and no_values["smart_5_raw_diff7d"].notna().any()


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


def label_setup(drives: dict[str, tuple[str, list[tuple]]]):
    frame = pd.concat([make_frame({serial: history}, model=model)
                       for serial, (model, history) in drives.items()], ignore_index=True)
    timeline = timelines.build_timelines(frame)
    index = timelines.build_drive_index(frame[timelines.CORE_IN])
    return timeline, index, timelines.row_labels(timeline, index)


def counts_for(timeline, labels, serial) -> dict[int, int]:
    mask = (timeline["serial_number"] == serial).to_numpy()
    values, counts = np.unique(labels[mask], return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


def daily(first: int, last: int, failure_day: int | None = None) -> list[tuple]:
    return [(d, 1.0, 1 if d == failure_day else 0) for d in range(first, last + 1)]


LABEL_DRIVES = {
    "FAILS": ("A", daily(1, 40, failure_day=40)),
    "ALIVE": ("A", daily(1, 61)),
    "GONE": ("B", daily(1, 20)),
    "EDGE": ("B", [(1, 1.0, 0), (31, 1.0, 0)]),
}


def test_classify_rows_boundaries():
    no = timelines.NO_DAY
    day_ = np.zeros(6, dtype="int64")
    failure = np.array([30, 31, no, no, -1, 0], dtype="int64")
    last = np.array([30, 31, 30, 29, 100, 0], dtype="int64")
    # failure 30 d ahead is positive; 31 d ahead is negative (drive seen at +31);
    # no failure and seen at +30 is negative; seen only to +29 is censored;
    # a failure before the row is censored; a failure on the row's own day is positive.
    assert timelines.classify_rows(day_, failure, last, 30).tolist() == [1, 0, 0, -1, -1, 1]


def test_horizon_labels_of_a_failing_drive():
    timeline, _, labels = label_setup(LABEL_DRIVES)
    # d = 10..40 are within 30 days of the failure on day 40; d = 1..9 saw the drive alive at d + 30
    assert counts_for(timeline, labels, "FAILS") == {1: 31, 0: 9}


def test_rows_whose_horizon_passes_the_last_date_are_censored_not_negative():
    timeline, _, labels = label_setup(LABEL_DRIVES)
    assert counts_for(timeline, labels, "ALIVE") == {0: 31, -1: 30}
    alive = timeline[timeline["serial_number"] == "ALIVE"]
    mask = (labels[(timeline["serial_number"] == "ALIVE").to_numpy()] == -1)
    assert alive.loc[mask, "date"].min() == pd.Timestamp(day(32))


def test_a_drive_that_leaves_without_a_failure_is_censored():
    timeline, _, labels = label_setup(LABEL_DRIVES)
    assert counts_for(timeline, labels, "GONE") == {-1: 20}


def test_a_negative_needs_a_row_on_or_after_the_horizon_end():
    timeline, _, labels = label_setup(LABEL_DRIVES)
    assert counts_for(timeline, labels, "EDGE") == {0: 1, -1: 1}


def test_label_counts_do_not_depend_on_the_calendar_position_of_the_window():
    shifted = {serial: (model, [(d + 200, s5, f) for d, s5, f in history])
               for serial, (model, history) in LABEL_DRIVES.items()}
    base = label_setup(LABEL_DRIVES)
    moved = label_setup(shifted)
    assert timelines.label_counts(base[2]) == timelines.label_counts(moved[2])
    assert moved[0]["date"].min() == pd.Timestamp(day(201))


def test_rows_after_a_recorded_failure_are_censored():
    timeline, _, labels = label_setup({"DOUBLE": ("A", [(1, 1.0, 0), (2, 1.0, 1), (3, 1.0, 1)])})
    assert counts_for(timeline, labels, "DOUBLE") == {1: 2, -1: 1}


def test_label_counts_and_drive_split():
    timeline, index, labels = label_setup(LABEL_DRIVES)
    assert timelines.label_counts(labels) == {"positive": 31, "negative": 41, "censored": 51}
    flags = timelines.drive_label_flags(timeline, labels)
    assert timelines.drive_split_counts(index, flags) == {
        "drives": 4, "eligible": 3, "with_positive": 1, "negative_only": 2, "all_censored": 1}
    by_model = timelines.drive_split_counts(index[index["model"] == "B"], flags)
    assert by_model == {"drives": 2, "eligible": 1, "with_positive": 0, "negative_only": 1, "all_censored": 1}


def test_label_lines_report_overall_and_per_model():
    timeline, index, labels = label_setup(LABEL_DRIVES)
    totals = {}
    for model in ("A", "B"):
        mask = (timeline["model"] == model).to_numpy()
        totals[model] = timelines.label_counts(labels[mask])
    lines = timelines.label_lines(totals, index, timelines.drive_label_flags(timeline, labels))
    assert lines == [
        "labels_h30 model=all positive=31 negative=41 censored=51",
        "labels_h30 model=A positive=31 negative=40 censored=30",
        "labels_h30 model=B positive=0 negative=1 censored=21",
        "label_drives model=all drives=4 eligible=3 with_positive=1 negative_only=2 all_censored=1",
        "label_drives model=A drives=2 eligible=2 with_positive=1 negative_only=1 all_censored=0",
        "label_drives model=B drives=2 eligible=1 with_positive=0 negative_only=1 all_censored=1",
    ]


def test_row_labels_reject_rows_of_drives_missing_from_the_index():
    timeline, index, _ = label_setup(LABEL_DRIVES)
    with pytest.raises(ValueError, match="missing from the drive index"):
        timelines.row_labels(timeline, index[index["serial_number"] != "ALIVE"])


def test_drive_index_carries_the_model_of_each_drive():
    _, index, _ = label_setup(LABEL_DRIVES)
    assert dict(zip(index["serial_number"], index["model"])) == {
        "ALIVE": "A", "EDGE": "B", "FAILS": "A", "GONE": "B"}


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
    assert "window_first=2022-01-01 window_last=2022-01-09 horizon_days=30" in printed
    assert "model=M timelines_rows=15 drives=4 failed_in_window=2" in printed
    # FAILS: days 1-4 positive; DOUBLE: days 1-2 positive, day 3 follows the recorded failure;
    # ALIVE and GAPPY end less than 30 days after every row.
    assert "labels_h30 model=all positive=6 negative=0 censored=9" in printed
    assert "label_drives model=all drives=4 eligible=2 with_positive=2 negative_only=0 all_censored=2" in printed

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


@pytest.mark.parametrize("buckets", [1, 3])
def test_partitioning_by_drive_does_not_change_the_output(env, monkeypatch, buckets):
    monkeypatch.setattr(timelines, "BUCKETS", buckets)
    assert timelines.main(["timelines.py"]) == 0
    stored = pq.read_table(timelines.OUT_TIMELINES).to_pandas()
    stored = stored.assign(date=pd.to_datetime(stored["date"]),
                           first_seen=pd.to_datetime(stored["first_seen"]))
    stored = stored.sort_values(timelines.KEY, kind="stable").reset_index(drop=True)
    expected = timelines.build_timelines(env)
    pd.testing.assert_frame_equal(stored, expected, check_exact=True, check_dtype=False)
    index = pq.read_table(timelines.OUT_INDEX).to_pandas()
    assert index["serial_number"].tolist() == ["ALIVE", "DOUBLE", "FAILS", "GAPPY"]


def test_main_reports_rows_and_drives_and_labels_per_model(env, tmp_path, capsys):
    two_models = pd.concat([
        make_frame({"A1": daily(1, 40, failure_day=40), "A2": daily(1, 61)}, model="SEAGATE"),
        make_frame({"T1": daily(1, 20)}, model="TOSHIBA"),
    ], ignore_index=True)
    src = tmp_path / "two_models.parquet"
    pq.write_table(pa.Table.from_pandas(two_models, preserve_index=False), src)
    assert timelines.main(["timelines.py", str(src)]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert "window_first=2022-01-01 window_last=2022-03-02 horizon_days=30" in printed
    assert "model=SEAGATE timelines_rows=101 drives=2 failed_in_window=1" in printed
    assert "model=TOSHIBA timelines_rows=20 drives=1 failed_in_window=0" in printed
    assert "labels_h30 model=SEAGATE positive=31 negative=40 censored=30" in printed
    assert "labels_h30 model=TOSHIBA positive=0 negative=0 censored=20" in printed
    assert "label_drives model=TOSHIBA drives=1 eligible=0 with_positive=0 negative_only=0 all_censored=1" in printed
    index = pq.read_table(timelines.OUT_INDEX).to_pandas().set_index("serial_number")
    assert index["model"].to_dict() == {"A1": "SEAGATE", "A2": "SEAGATE", "T1": "TOSHIBA"}


def test_preprocessing_section_states_the_label_rule_and_the_input_decision(env):
    timelines.update_preprocessing(["rows_in=14"])
    text = timelines.PREPROCESSING.read_text(encoding="utf-8")
    assert "A1_DECISION.md" in text and "provisional" not in text
    assert "smart_199_raw" in text and "Censored" in text and "30 days" in text


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
