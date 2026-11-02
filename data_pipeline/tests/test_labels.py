"""
Unit tests for backblaze/labels.py (A2) on small synthetic drive histories.

All path constants of the module are redirected into tmp_path, so no test reads
data/raw or data/processed or touches the real manifest or PREPROCESSING.md.
"""
import datetime as dt
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("pyarrow")

from data_pipeline.backblaze import labels, timelines  # noqa: E402

FIRST_DAY = dt.date(2022, 1, 1)
MODELS = ("MA", "MB")


def day(n: int) -> dt.date:
    return FIRST_DAY + dt.timedelta(days=n - 1)


def core_rows(serial: str, model: str, first: int, last: int, fail_day: int | None = None) -> list[dict]:
    rows = []
    for d in range(first, last + 1):
        row = {"serial_number": serial, "model": model, "date": day(d),
               "capacity_bytes": 4_000_000_000_000, "failure": int(d == fail_day)}
        row.update({c: float(d) for c in timelines.SMART_COLUMNS})
        rows.append(row)
    return rows


def synthetic_frame(failing: int = 10, healthy: int = 10) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for k in range(failing):
            rows += core_rows(f"{model}-F{k:02d}", model, 1, 35, fail_day=35)
        for k in range(healthy):
            rows += core_rows(f"{model}-H{k:02d}", model, 1, 45)
    rows += core_rows("MA-CENSORED", "MA", 45, 45)
    return pd.DataFrame(rows)


def build_inputs(directory, frame: pd.DataFrame, row_group_size: int = 1000):
    timeline = timelines.build_timelines(frame)
    index = timelines.build_drive_index(frame[timelines.CORE_IN])
    directory.mkdir(parents=True, exist_ok=True)
    timelines_path, index_path = directory / "timelines.parquet", directory / "index.parquet"
    pq.write_table(timelines.to_arrow(timeline), timelines_path, row_group_size=row_group_size)
    pq.write_table(timelines.to_arrow(index), index_path)
    return timelines_path, index_path, timeline, index


def expected_counts(timeline: pd.DataFrame, index: pd.DataFrame) -> dict:
    row_labels = timelines.row_labels(timeline, index)
    by_model = {m: timelines.label_counts(row_labels[(timeline["model"] == m).to_numpy()]) for m in MODELS}
    return {"rows": len(timeline), "by_model": by_model,
            "totals": {k: sum(v[k] for v in by_model.values()) for k in ("positive", "negative", "censored")}}


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / "pipe"
    processed = root / "data" / "processed"
    timelines_path, index_path, timeline, index = build_inputs(processed, synthetic_frame(), row_group_size=97)
    (root / "manifest.json").write_text(json.dumps({"files": [{"file": "other.parquet", "rows": 1, "sha256": "x"}]}))
    (root / "PREPROCESSING.md").write_text("# P\n\n## B6 — This file\n\n- keep\n", encoding="utf-8")
    expected = expected_counts(timeline, index)
    for name, value in {
        "ROOT": root, "PROCESSED": processed, "IN_TIMELINES": timelines_path, "IN_INDEX": index_path,
        "OUT_LABELS": processed / "backblaze_labels.parquet",
        "OUT_SPLIT": processed / "backblaze_drive_split.parquet",
        "MANIFEST": root / "manifest.json", "PREPROCESSING": root / "PREPROCESSING.md",
        "EXPECTED_ROWS": expected["rows"], "EXPECTED_TOTALS": expected["totals"],
        "EXPECTED_BY_MODEL": expected["by_model"],
    }.items():
        monkeypatch.setattr(labels, name, value)
    return {"timeline": timeline, "index": index, "expected": expected}


# -- labels --

def test_labels_equal_the_b2_rule_in_the_timelines_order(tmp_path):
    frame = synthetic_frame()
    timelines_path, index_path, timeline, index = build_inputs(tmp_path / "in", frame, row_group_size=50)
    out = tmp_path / "labels.parquet"
    result = labels.build_labels(timelines_path, index_path, out)
    table = pq.read_table(out)
    assert table.schema.equals(labels.LABEL_SCHEMA)
    assert table.column_names == ["serial_number", "date", "model", "label"]
    got = table.to_pandas()
    assert len(got) == len(timeline) == result["rows"]
    assert got["serial_number"].tolist() == timeline["serial_number"].tolist()
    assert [pd.Timestamp(d) for d in got["date"]] == timeline["date"].tolist()
    assert got["model"].tolist() == timeline["model"].astype(object).tolist()
    assert got["label"].to_numpy().tolist() == timelines.row_labels(timeline, index).tolist()
    assert set(got["label"].unique()) == {-1, 0, 1}
    assert result["totals"] == expected_counts(timeline, index)["totals"]
    assert result["by_model"] == expected_counts(timeline, index)["by_model"]


def test_the_row_group_size_does_not_change_the_labels(tmp_path):
    frame = synthetic_frame(failing=3, healthy=3)
    outputs = []
    for size in (7, 100_000):
        timelines_path, index_path, _, _ = build_inputs(tmp_path / f"in{size}", frame, row_group_size=size)
        out = tmp_path / f"labels{size}.parquet"
        labels.build_labels(timelines_path, index_path, out)
        outputs.append(pq.read_table(out).to_pandas())
    pd.testing.assert_frame_equal(outputs[0], outputs[1])


def test_only_keys_and_the_model_are_read_from_the_timelines(tmp_path, monkeypatch):
    frame = synthetic_frame(failing=2, healthy=2)
    timelines_path, index_path, _, _ = build_inputs(tmp_path / "in", frame)
    requested = []
    original = pq.ParquetFile.read_row_group

    def spy(self, i, columns=None, **kwargs):
        requested.append(tuple(columns or ()))
        return original(self, i, columns=columns, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", spy)
    labels.build_labels(timelines_path, index_path, tmp_path / "out.parquet")
    assert requested and set(requested) == {("serial_number", "date")}


def test_verify_expected_accepts_the_recorded_counts_and_names_each_difference(env):
    expected = env["expected"]
    labels.verify_expected({"rows": expected["rows"], "totals": expected["totals"],
                            "by_model": expected["by_model"]})
    wrong_model = {m: dict(c) for m, c in expected["by_model"].items()}
    wrong_model["MA"]["positive"] += 1
    wrong_totals = dict(expected["totals"], censored=expected["totals"]["censored"] + 1)
    with pytest.raises(SystemExit) as err:
        labels.verify_expected({"rows": expected["rows"] + 1, "totals": wrong_totals, "by_model": wrong_model})
    message = str(err.value)
    assert "label verification failed" in message
    assert f"rows: expected {expected['rows']}, measured {expected['rows'] + 1}" in message
    assert "totals:" in message and "MA:" in message


def test_a_model_without_a_recorded_value_is_reported(env):
    expected = env["expected"]
    by_model = dict(expected["by_model"], MC={"positive": 0, "negative": 0, "censored": 0})
    with pytest.raises(SystemExit, match=r"models not in the recorded values: \['MC'\]"):
        labels.verify_expected({"rows": expected["rows"], "totals": expected["totals"], "by_model": by_model})


def test_main_publishes_both_outputs_and_registers_only_those_entries(env, capsys):
    assert labels.main(["labels.py"]) == 0
    printed = capsys.readouterr().out
    assert f"a2_labels rows={env['expected']['rows']} " in printed
    assert labels.OUT_LABELS.exists() and labels.OUT_SPLIT.exists()
    assert not list(labels.PROCESSED.glob("*.tmp"))
    files = {e["file"]: e for e in json.loads(labels.MANIFEST.read_text(encoding="utf-8"))["files"]}
    assert set(files) == {"other.parquet", "data/processed/backblaze_labels.parquet",
                          "data/processed/backblaze_drive_split.parquet"}
    assert files["other.parquet"] == {"file": "other.parquet", "rows": 1, "sha256": "x"}
    assert files["data/processed/backblaze_labels.parquet"]["rows"] == env["expected"]["rows"]
    assert files["data/processed/backblaze_drive_split.parquet"]["rows"] == len(env["index"])
    assert files["data/processed/backblaze_labels.parquet"]["sha256"] == labels.sha256_of(labels.OUT_LABELS)
    text = labels.PREPROCESSING.read_text(encoding="utf-8")
    assert "## A2 — Horizon labels and drive split" in text and "- keep" in text
    assert "not time-forward" in text and "excluded from training and scoring" in text


def test_rerunning_main_leaves_the_documents_and_outputs_unchanged(env):
    labels.main(["labels.py"])
    first = (labels.PREPROCESSING.read_text(encoding="utf-8"), labels.MANIFEST.read_text(encoding="utf-8"),
             labels.OUT_LABELS.read_bytes(), labels.OUT_SPLIT.read_bytes())
    labels.main(["labels.py"])
    assert (labels.PREPROCESSING.read_text(encoding="utf-8"), labels.MANIFEST.read_text(encoding="utf-8"),
            labels.OUT_LABELS.read_bytes(), labels.OUT_SPLIT.read_bytes()) == first


def test_a_failed_verification_publishes_nothing_and_leaves_no_temporary_file(env, monkeypatch):
    monkeypatch.setattr(labels, "EXPECTED_ROWS", env["expected"]["rows"] + 1)
    for path in (labels.OUT_LABELS, labels.OUT_SPLIT):
        path.write_bytes(b"previous")
    manifest = labels.MANIFEST.read_bytes()
    with pytest.raises(SystemExit, match="label verification failed"):
        labels.main(["labels.py"])
    assert labels.OUT_LABELS.read_bytes() == b"previous" and labels.OUT_SPLIT.read_bytes() == b"previous"
    assert not list(labels.PROCESSED.glob("*.tmp"))
    assert labels.MANIFEST.read_bytes() == manifest
    assert "A2" not in labels.PREPROCESSING.read_text(encoding="utf-8")


def test_an_exception_in_a_late_row_group_leaves_no_temporary_file(env, monkeypatch):
    calls = []
    original = timelines.row_labels

    def failing(frame, index, horizon=timelines.HORIZON_DAYS):
        calls.append(1)
        if len(calls) == 3:
            raise RuntimeError("late row group failure")
        return original(frame, index, horizon)

    monkeypatch.setattr(timelines, "row_labels", failing)
    labels.OUT_LABELS.write_bytes(b"previous")
    with pytest.raises(RuntimeError, match="late row group failure"):
        labels.main(["labels.py"])
    assert len(calls) == 3
    assert labels.OUT_LABELS.read_bytes() == b"previous" and not labels.OUT_SPLIT.exists()
    assert not list(labels.PROCESSED.glob("*.tmp"))


def test_an_input_without_row_groups_gives_an_empty_label_table(tmp_path):
    timelines_path, index_path, _, _ = build_inputs(tmp_path / "in", synthetic_frame(1, 1))
    empty = pq.read_table(timelines_path).slice(0, 0)
    pq.write_table(empty, timelines_path)
    out = tmp_path / "labels.parquet"
    result = labels.build_labels(timelines_path, index_path, out)
    assert result["rows"] == 0 and result["totals"] == {"positive": 0, "negative": 0, "censored": 0}
    assert pq.read_table(out).num_rows == 0


# -- split --

@pytest.mark.parametrize("n", list(range(0, 60)) + [385, 633, 18940, 38579])
def test_largest_remainder_sums_to_n_and_stays_within_one_of_the_quota(n):
    shares = labels.largest_remainder(n)
    assert sum(shares) == n and all(s >= 0 for s in shares)
    assert all(abs(s - n * f) < 1 for s, f in zip(shares, labels.FRACTIONS))


def test_largest_remainder_rounding_edges():
    assert labels.largest_remainder(0) == [0, 0, 0]
    assert labels.largest_remainder(1) == [1, 0, 0]
    assert labels.largest_remainder(2) == [2, 0, 0]
    assert labels.largest_remainder(3) == [2, 1, 0]
    assert labels.largest_remainder(10) == [7, 2, 1]
    assert labels.largest_remainder(20) == [14, 3, 3]


def split_index(failed_per_model: dict[str, int], healthy_per_model: dict[str, int]) -> pd.DataFrame:
    rows = []
    for model in sorted(set(failed_per_model) | set(healthy_per_model)):
        for k in range(failed_per_model.get(model, 0)):
            rows.append((f"{model}-F{k:03d}", model, True))
        for k in range(healthy_per_model.get(model, 0)):
            rows.append((f"{model}-H{k:03d}", model, False))
    frame = pd.DataFrame(rows, columns=["serial_number", "model", "failed_in_window"])
    return frame.sample(frac=1, random_state=5).reset_index(drop=True)


def test_the_split_is_deterministic_and_independent_of_the_input_row_order():
    index = split_index({"MA": 40, "MB": 25}, {"MA": 300, "MB": 120})
    first = labels.assign_splits(index)
    again = labels.assign_splits(index)
    shuffled = labels.assign_splits(index.sample(frac=1, random_state=11).reset_index(drop=True))
    pd.testing.assert_frame_equal(first, again)
    pd.testing.assert_frame_equal(first, shuffled)
    assert labels.assign_splits(index, seed=labels.SPLIT_SEED + 1)["split"].tolist() != first["split"].tolist()


def test_the_split_has_one_row_per_drive_and_the_documented_columns():
    index = split_index({"MA": 40, "MB": 25}, {"MA": 300, "MB": 120})
    split = labels.assign_splits(index)
    assert list(split.columns) == labels.SPLIT_COLUMNS
    assert len(split) == len(index) and split["serial_number"].is_unique
    assert set(split["serial_number"]) == set(index["serial_number"])
    assert set(split["split"]) == {"train", "val", "test"}
    assert split["serial_number"].is_monotonic_increasing
    flags = index.set_index("serial_number")["failed_in_window"]
    assert split.set_index("serial_number")["ever_failed"].equals(flags.loc[split["serial_number"]].rename("ever_failed"))


def test_every_stratum_is_within_one_drive_of_the_target_shares():
    index = split_index({"MA": 40, "MB": 25, "MC": 7}, {"MA": 301, "MB": 121, "MC": 53})
    split = labels.assign_splits(index)
    for (model, ever), group in split.groupby(["model", "ever_failed"]):
        for name, fraction in zip(labels.SPLITS, labels.FRACTIONS):
            assert abs((group["split"] == name).sum() - len(group) * fraction) < 1, (model, ever, name)


def test_a_drive_is_in_exactly_one_split_and_the_assignment_is_not_by_name_order():
    index = split_index({"MA": 40}, {"MA": 200})
    split = labels.assign_splits(index)
    assert split.groupby("serial_number")["split"].nunique().eq(1).all()
    train_names = split.loc[split["split"] == "train", "serial_number"]
    assert set(train_names) != set(split["serial_number"].iloc[:len(train_names)])
    assert split.loc[split["split"] == "test", "serial_number"].iloc[0] > split["serial_number"].iloc[0]


def test_strata_with_one_or_two_failed_drives():
    index = split_index({"MA": 1, "MB": 2}, {"MA": 30, "MB": 30})
    split = labels.assign_splits(index)
    failed = split[split["ever_failed"]]
    assert failed[failed["model"] == "MA"]["split"].tolist() == ["train"]
    assert failed[failed["model"] == "MB"]["split"].tolist() == ["train", "train"]
    assert len(split) == 63


def test_an_empty_stratum_is_not_created_and_an_empty_index_gives_an_empty_split():
    only_healthy = split_index({}, {"MA": 20})
    split = labels.assign_splits(only_healthy)
    assert len(split) == 20 and not split["ever_failed"].any()
    empty = labels.assign_splits(only_healthy.iloc[0:0])
    assert list(empty.columns) == labels.SPLIT_COLUMNS and len(empty) == 0


def test_a_fully_censored_drive_gets_a_split(env):
    split = labels.assign_splits(env["index"])
    row = split[split["serial_number"] == "MA-CENSORED"]
    assert len(row) == 1 and row.iloc[0]["split"] in labels.SPLITS and not row.iloc[0]["ever_failed"]
    assert len(split) == len(env["index"])


def test_split_statistics_counts_drives_rows_dates_and_positive_months(env, tmp_path):
    labels.build_labels(labels.IN_TIMELINES, labels.IN_INDEX, tmp_path / "labels.parquet")
    split = labels.assign_splits(env["index"])
    stats = labels.split_statistics(tmp_path / "labels.parquet", split)
    timeline = env["timeline"]
    row_labels = timelines.row_labels(timeline, env["index"])
    by_serial = split.set_index("serial_number")["split"]
    row_split = by_serial.loc[timeline["serial_number"].astype(object)].to_numpy()
    for name in labels.SPLITS:
        for model in MODELS:
            mask = (row_split == name) & (timeline["model"] == model).to_numpy()
            counts = stats["counts"][(name, model)]
            assert (counts["positive"], counts["negative"], counts["censored"]) == (
                int((row_labels[mask] == 1).sum()), int((row_labels[mask] == 0).sum()),
                int((row_labels[mask] == -1).sum()))
            drives = split[(split["split"] == name) & (split["model"] == model)]
            assert counts["drives"] == len(drives) and counts["failed_drives"] == int(drives["ever_failed"].sum())
        total = stats["counts"][(name, "all")]
        assert total["drives"] == int((split["split"] == name).sum())
        positive_rows = int(((row_split == name) & (row_labels == 1)).sum())
        assert total["positive"] == positive_rows
        assert sum(stats["dates"][name]["positive_by_month"].values()) == positive_rows
    assert sum(stats["counts"][(n, "all")]["drives"] for n in labels.SPLITS) == len(split)
    assert stats["dates"]["train"]["min"] == "2022-01-01" and stats["dates"]["train"]["max"] == "2022-02-14"


def test_split_statistics_rejects_label_rows_of_unknown_drives(env, tmp_path):
    labels.build_labels(labels.IN_TIMELINES, labels.IN_INDEX, tmp_path / "labels.parquet")
    split = labels.assign_splits(env["index"])
    with pytest.raises(ValueError, match="missing from the split table"):
        labels.split_statistics(tmp_path / "labels.parquet", split.iloc[1:])


def test_report_lines_have_one_split_line_per_model_and_split_plus_all(env, tmp_path):
    result = labels.build_labels(labels.IN_TIMELINES, labels.IN_INDEX, tmp_path / "labels.parquet")
    stats = labels.split_statistics(tmp_path / "labels.parquet", labels.assign_splits(env["index"]))
    lines = labels.report_lines(result, stats)
    assert lines[0].startswith(f"a2_labels rows={result['rows']} ")
    assert lines[1] == f"a2_params seed={labels.SPLIT_SEED} fractions=train:0.7,val:0.15,test:0.15 strata=model,ever_failed"
    assert sum(l.startswith("a2_split split=") for l in lines) == 3 * (len(MODELS) + 1)
    assert sum(l.startswith("a2_split_dates split=") for l in lines) == 3


def test_update_preprocessing_appends_then_replaces_only_the_a2_section(env):
    labels.update_preprocessing(["a2_labels rows=1"])
    first = labels.PREPROCESSING.read_text(encoding="utf-8")
    assert first.count("## A2 — Horizon labels and drive split") == 1 and "- `a2_labels rows=1`" in first
    labels.PREPROCESSING.write_text(first + "\n## B9 — later\n\n- later\n", encoding="utf-8")
    labels.update_preprocessing(["a2_labels rows=2"])
    second = labels.PREPROCESSING.read_text(encoding="utf-8")
    assert "- `a2_labels rows=2`" in second and "rows=1`" not in second
    assert "## B9 — later\n\n- later" in second and "- keep" in second


def test_the_cli_runs_in_isolated_mode_from_a_copy_of_the_package(tmp_path):
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(labels.__file__).resolve().parents[2]
    copy = tmp_path / "repo"
    (copy / "data_pipeline" / "backblaze").mkdir(parents=True)
    for name in ("__init__.py",):
        (copy / "data_pipeline" / name).write_text("")
    for name in ("__init__.py", "labels.py", "timelines.py"):
        shutil.copy(repo / "data_pipeline" / "backblaze" / name, copy / "data_pipeline" / "backblaze" / name)
    pipe = copy / "data_pipeline"
    build_inputs(pipe / "data" / "processed", synthetic_frame(1, 1))
    (pipe / "data" / "processed" / "timelines.parquet").rename(pipe / "data" / "processed" / "backblaze_timelines.parquet")
    (pipe / "data" / "processed" / "index.parquet").rename(pipe / "data" / "processed" / "backblaze_drive_index.parquet")
    (pipe / "manifest.json").write_text('{"files": []}')
    (pipe / "PREPROCESSING.md").write_text("# P\n", encoding="utf-8")
    res = subprocess.run([sys.executable, "-I", str(pipe / "backblaze" / "labels.py")],
                         capture_output=True, text=True, cwd=tmp_path)
    assert res.returncode != 0
    assert "label verification failed" in res.stderr
    assert not list((pipe / "data" / "processed").glob("*.tmp"))
