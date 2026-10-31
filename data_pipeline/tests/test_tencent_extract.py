"""
Unit tests for tencent/extract_load.py (B3) on a small synthetic dataset.

Path constants of the module are redirected into tmp_path, so no test
touches the real manifest, PREPROCESSING.md or processed data.
"""
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

pytest.importorskip("pyarrow")

from contracts import contracts  # noqa: E402
from data_pipeline.tencent import extract_load as ex  # noqa: E402

# Source order: timestamp, read_iops, read_bw, write_iops, write_bw, usage.
# d1 (grid origin 1000, step 300): one decrease (1600 -> 1300), an exact duplicate
# (1900), a conflicting timestamp (2200 twice), a negative value (2500), a
# three-slot gap (1900 -> 3100) and an off-grid row (3250).
D1 = "\n".join([
    "1000,1,10,2,20,100",
    "1600,1,10,2,20,100",
    "1300,1,10,2,20,100",
    "1900,1,10,2,20,100",
    "1900,1,10,2,20,100",
    "2200,1,10,2,20,100",
    "2200,9,10,2,20,100",
    "2500,1,10,-1,20,100",
    "3100,1,10,2,20,100",
    "3250,1,10,2,20,100",
]) + "\n"
D2 = "\n".join([
    "5000,0,0,0,0,50",
    "5300,3,100,4,50,50",
    "5600,3,100,4,50,50",
    "5900,0,0,0,0,50",
    "6200,3,100,4,50,50",
]) + "\n"
D3 = "7000,0,0,-5,0,0\n"   # every row invalid
SUBSCRIPTION = "\n".join([
    "d1,50,1,1,0,2,4",
    "d2,50,1,1,0,1,1",
    "d2,50,1,1,0,1,1",
    "dX,50,1,1,0,2,4",
    "dX,100,0,1,0,2,4",
]) + "\n"


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    root = tmp_path / "pipe"
    processed = root / "data" / "processed"
    processed.mkdir(parents=True)
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"files": [{"file": "other.parquet", "rows": 1, "sha256": "x"}]}))
    prep = root / "PREPROCESSING.md"
    prep.write_text("# P\n\n## B3 — Tencent extract\n\n- stub\n\n## B4 — Metadata join\n\n- keep\n",
                    encoding="utf-8")
    raw = tmp_path / "raw"
    (raw / "disk_load_data").mkdir(parents=True)
    (raw / "disk_subscript_info").write_text(SUBSCRIPTION)
    for name, text in (("d1", D1), ("d2", D2), ("d3", D3)):
        (raw / "disk_load_data" / name).write_text(text)
    (raw / "disk_load_data" / ".DS_Store").write_bytes(b"\x00junk")
    for name, value in {
        "ROOT": root, "PROCESSED": processed, "MANIFEST": manifest, "PREPROCESSING": prep,
        "OUT_CLEAN": processed / "tencent_load_clean.parquet",
        "OUT_SUMMARY": processed / "tencent_disk_summary.parquet",
        "OUT_RUNS": processed / "tencent_disk_runs.parquet",
        "DEFAULT_DATASET": raw,
    }.items():
        monkeypatch.setattr(ex, name, value)
    return raw


def parsed(text: str) -> pd.DataFrame:
    return ex.parse_trace(text.encode(), "t")


def test_parse_trace_reorders_to_clean_columns():
    df = parsed("1000,1,10,2,20,100\n")
    assert list(df.columns) == ex.CLEAN_COLUMNS
    row = df.iloc[0].to_dict()
    assert row == {"timestamp": 1000, "read_iops": 1, "write_iops": 2,
                   "read_bw_kbps": 10, "write_bw_kbps": 20, "disk_usage_mb": 100}
    assert all(dtype == np.dtype("int64") for dtype in df.dtypes)


def test_parse_trace_rejects_non_integer_values():
    with pytest.raises(SystemExit, match="non-integer columns"):
        ex.parse_trace(b"1000,1,10.5,2,20,100\n", "diskX")


GOOD_ROW = "1000,1,10,2,20,100"
WRONG_FIELD_COUNTS = [
    (GOOD_ROW + ",9\n1300,1,10,2,20,100\n", "7 columns, expected 6"),    # extra field on the first row
    (GOOD_ROW + "\n1300,1,10,2,20,100,9\n", "Expected 6 fields"),        # extra field on a later row
    (GOOD_ROW + "\n1300,1,10,2,20\n", "missing fields"),                 # too few fields on a later row
    ("1000,1,10,2,20\n1300,1,10,2,20,100\n", "Expected 5 fields"),       # short first row, longer later row
    ("1000,1,10,2,20\n", "5 columns, expected 6"),                       # every row too short
    ("", "No columns to parse"),                                          # empty file
]


@pytest.mark.parametrize("text, message", WRONG_FIELD_COUNTS,
                         ids=["extra_first_row", "extra_later_row", "too_few_later_row",
                              "short_first_row", "all_rows_short", "empty"])
def test_parse_trace_rejects_wrong_field_counts_naming_the_file(text, message):
    with pytest.raises(SystemExit, match=message) as err:
        ex.parse_trace(text.encode(), "diskX")
    assert "diskX" in str(err.value)


SUBSCRIPTION_ROW = "d1,50,1,1,0,2,4"
WRONG_SUBSCRIPTION_FIELD_COUNTS = [
    (SUBSCRIPTION_ROW + ",9\nd2,50,1,1,0,1,1\n", "8 columns, expected 7"),
    (SUBSCRIPTION_ROW + "\nd2,50,1,1,0,1,1,9\n", "Expected 7 fields"),
    (SUBSCRIPTION_ROW + "\nd2,50,1,1,0,1\n", "missing fields"),
    ("d1,50,1,1,0,2\n", "6 columns, expected 7"),
    ("", "No columns to parse"),
]


@pytest.mark.parametrize("text, message", WRONG_SUBSCRIPTION_FIELD_COUNTS,
                         ids=["extra_first_row", "extra_later_row", "too_few_later_row",
                              "all_rows_short", "empty"])
def test_read_subscription_rejects_wrong_field_counts(tmp_path, text, message):
    path = tmp_path / "sub"
    path.write_text(text)
    with pytest.raises(SystemExit, match=message) as err:
        ex.read_subscription(path)
    assert "subscription" in str(err.value)


def test_read_subscription_accepts_exact_field_counts(tmp_path):
    path = tmp_path / "sub"
    path.write_text(SUBSCRIPTION_ROW + "\nd2,50,1,1,0,1,1\n")
    sub = ex.read_subscription(path)
    assert list(sub.columns) == ex.SUB_COLUMNS and len(sub) == 2


def test_clean_trace_counts_and_order():
    clean, stats, _ = ex.clean_trace(parsed(D1))
    assert stats == {"rows_raw": 10, "out_of_order_rows": 1, "exact_duplicate_rows": 1,
                     "conflict_timestamps": 1, "conflict_rows_dropped": 2,
                     "negative_rows_dropped": 1}
    assert clean["timestamp"].tolist() == [1000, 1300, 1600, 1900, 3100, 3250]
    assert list(clean.columns) == ex.CLEAN_COLUMNS


def test_clean_trace_empty_and_all_invalid():
    empty, stats, _ = ex.clean_trace(parsed("1000,1,1,1,1,1\n").iloc[0:0])
    assert len(empty) == 0 and stats["out_of_order_rows"] == 0
    clean, stats, _ = ex.clean_trace(parsed(D3))
    assert len(clean) == 0 and stats["negative_rows_dropped"] == 1


def test_trace_summary_gap_run_and_grid():
    clean, _, _ = ex.clean_trace(parsed(D1))
    s = ex.trace_summary(clean)
    assert (s["rows_clean"], s["first_ts"], s["last_ts"]) == (6, 1000, 3250)
    assert s["breaks"] == 2 and s["missing_slots"] == 3
    assert s["longest_run_rows"] == 4 and s["off_grid_rows"] == 1
    assert (s["mean_total_bw_kbps"], s["p95_total_bw_kbps"], s["max_total_bw_kbps"]) == (30.0, 30.0, 30)
    assert s["zero_row_fraction"] == 0.0


def test_trace_summary_contiguous_with_zero_rows():
    clean, _, _ = ex.clean_trace(parsed(D2))
    s = ex.trace_summary(clean)
    assert s["breaks"] == 0 and s["missing_slots"] == 0 and s["longest_run_rows"] == 5
    assert s["zero_row_fraction"] == pytest.approx(0.4)
    assert s["mean_total_bw_kbps"] == pytest.approx(90.0) and s["max_total_bw_kbps"] == 150


def test_trace_summary_empty():
    s = ex.trace_summary(ex.clean_trace(parsed(D3))[0])
    assert s["rows_clean"] == 0 and s["first_ts"] is None and s["longest_run_rows"] == 0


def test_clean_window_converts_with_the_contract_to_replay_frame():
    clean, _, _ = ex.clean_trace(parsed(D2))
    frame = contracts.to_replay_frame(clean)
    assert list(frame.columns) == list(contracts.REPLAY_COLUMNS)
    assert frame["t_seconds"].tolist() == [0, 300, 600, 900, 1200]
    contracts.validate_replay_frame(frame)
    assert ex.STEP_S == contracts.CONTROL_INTERVAL_S


def test_clean_window_with_a_gap_or_no_rows_is_rejected_by_the_contract():
    clean, _, _ = ex.clean_trace(parsed(D1))
    with pytest.raises(contracts.ContractViolation, match="contiguous"):
        contracts.to_replay_frame(clean)
    with pytest.raises(contracts.ContractViolation, match="contiguous"):
        contracts.to_replay_frame(clean.iloc[0:0])
    contracts.validate_replay_frame(contracts.to_replay_frame(clean.iloc[:4]))


def test_dedupe_subscription_keeps_first_and_lists_conflicts(tmp_path):
    p = tmp_path / "sub"
    p.write_text(SUBSCRIPTION)
    deduped, info = ex.dedupe_subscription(ex.read_subscription(p))
    assert list(deduped["disk_uid"]) == ["d1", "d2", "dX"]
    assert deduped.loc[deduped["disk_uid"] == "dX", "disk_capacity"].iloc[0] == 50
    assert info == {"rows": 5, "unique_uids": 3, "repeated_uids": 2, "conflicting_uids": ["dX"]}


def test_sha256_helpers(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"abc")
    assert ex.sha256_of(p) == ex.sha256_bytes(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_extract_outputs_and_totals(dataset):
    r = ex.extract(dataset)
    assert r["files"] == 3
    assert r["totals"] == {
        "rows_raw": 16, "rows_clean": 11, "out_of_order_rows": 1, "exact_duplicate_rows": 1,
        "conflict_timestamps": 1, "conflict_rows_dropped": 2, "negative_rows_dropped": 2,
        "off_grid_rows": 1, "missing_slots": 3}
    assert r["disks_with"] == {"out_of_order": 1, "exact_duplicates": 1, "conflicts": 1,
                               "negatives": 2, "off_grid": 1, "breaks": 1}
    assert r["subscription"]["in_subscription_only"] == 1      # dX
    assert r["subscription"]["in_load_only"] == 1              # d3
    assert r["full_length_disks"] == 0

    pf = pq.ParquetFile(ex.OUT_CLEAN)
    assert pf.metadata.num_row_groups == 2           # d3 has no clean rows
    assert [pf.metadata.row_group(i).num_rows for i in range(2)] == [6, 5]
    clean = pq.read_table(ex.OUT_CLEAN).to_pandas()
    assert list(clean.columns) == ["disk_uid"] + ex.CLEAN_COLUMNS
    assert clean["disk_uid"].tolist() == ["d1"] * 6 + ["d2"] * 5
    assert not (clean[ex.CLEAN_COLUMNS] < 0).any().any()

    summary = pq.read_table(ex.OUT_SUMMARY).to_pandas().set_index("disk_uid")
    assert list(summary.index) == ["d1", "d2", "d3"]
    assert summary.loc["d1", "missing_slots"] == 3 and summary.loc["d2", "longest_run_rows"] == 5
    assert summary.loc["d3", "rows_clean"] == 0 and pd.isna(summary.loc["d3", "first_ts"])
    assert r["source_sha256"]["load_files"] == 3
    assert not ex.OUT_CLEAN.with_suffix(".parquet.tmp").exists()


def test_extract_without_load_files_raises(dataset):
    for p in (dataset / "disk_load_data").iterdir():
        p.unlink()
    with pytest.raises(SystemExit, match="no load files"):
        ex.extract(dataset)


def test_report_lines(dataset):
    lines = ex.report_lines(ex.extract(dataset))
    assert "load_files=3" in lines and "rows_raw=16" in lines and "rows_clean=11" in lines
    assert "out_of_order_rows=1 disks=1" in lines
    assert "conflict_timestamps=1 conflict_rows_dropped=2 disks=1" in lines
    assert "negative_rows_dropped=2 disks=2" in lines
    assert "disks_with_breaks=1 missing_slots=3" in lines
    assert ("subscription_rows=5 unique_uids=3 repeated_uids=2 conflicting_uids=1") in lines
    assert "conflicting_uid=dX" in lines
    assert any(l.startswith("output=data/processed/tencent_load_clean.parquet rows=11 sha256=")
               for l in lines)
    assert any(l.startswith("output=data/processed/tencent_disk_summary.parquet rows=3 sha256=")
               for l in lines)
    assert any(l.startswith("output=data/processed/tencent_disk_runs.parquet rows=") for l in lines)


def test_update_preprocessing_replaces_only_b3_section(dataset):
    ex.update_preprocessing(["rows_raw=16", "conflicting_uid=dX"])
    text = ex.PREPROCESSING.read_text(encoding="utf-8")
    assert "- `conflicting_uid=dX`" in text and "- stub" not in text
    assert "## B4 — Metadata join\n\n- keep" in text
    ex.update_preprocessing(["rows_raw=16", "conflicting_uid=dX"])
    assert ex.PREPROCESSING.read_text(encoding="utf-8") == text


def test_update_preprocessing_missing_section_raises(dataset):
    ex.PREPROCESSING.write_text("# P\n\n## B4 — x\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="B3 section not found"):
        ex.update_preprocessing([])


def test_update_manifest_upserts(dataset):
    r = {"clean_rows": 11, "files": 3, "clean_sha256": "a", "summary_sha256": "b",
         "runs_rows": 5, "runs_sha256": "r", "source_sha256": {"combined": "c"}}
    ex.update_manifest(r)
    ex.update_manifest({**r, "clean_rows": 12})
    data = json.loads(ex.MANIFEST.read_text(encoding="utf-8"))
    files = {e["file"]: e for e in data["files"]}
    assert set(files) == {"other.parquet", "data/processed/tencent_load_clean.parquet",
                          "data/processed/tencent_disk_summary.parquet",
                          "data/processed/tencent_disk_runs.parquet"}
    assert files["data/processed/tencent_load_clean.parquet"]["rows"] == 12
    assert data["sources"]["tencent"] == {"combined": "c"}


def test_main_explicit_and_default_root_are_idempotent(dataset, capsys):
    assert ex.main(["extract_load.py", str(dataset)]) == 0
    assert "rows_clean=11" in capsys.readouterr().out
    first = (ex.PREPROCESSING.read_text(encoding="utf-8"), ex.MANIFEST.read_text(encoding="utf-8"))
    assert ex.main(["extract_load.py"]) == 0
    assert (ex.PREPROCESSING.read_text(encoding="utf-8"), ex.MANIFEST.read_text(encoding="utf-8")) == first
    entry = next(e for e in json.loads(first[1])["files"] if e["file"].endswith("clean.parquet"))
    assert entry["sha256"] == ex.sha256_of(ex.OUT_CLEAN)


def test_cli_entry_point_in_subprocess(tmp_path, dataset):
    pipe = tmp_path / "cli"
    (pipe / "tencent").mkdir(parents=True)
    shutil.copy(Path(ex.__file__), pipe / "tencent" / "extract_load.py")
    (pipe / "manifest.json").write_text('{"files": []}')
    (pipe / "PREPROCESSING.md").write_text("## B3 — Tencent extract\n\n- stub\n\n## B4 — x\n",
                                           encoding="utf-8")
    res = subprocess.run(
        [sys.executable, "-I", str(pipe / "tencent" / "extract_load.py"), str(dataset)],
        capture_output=True, text=True, cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "rows_clean=11" in res.stdout
    assert (pipe / "data" / "processed" / "tencent_load_clean.parquet").exists()
    assert len(json.loads((pipe / "manifest.json").read_text())["files"]) == 3


def row(ts, **kw):
    v = {"ri": 1, "rb": 10, "wi": 2, "wb": 20, "u": 100, **kw}
    return f"{ts},{v['ri']},{v['rb']},{v['wi']},{v['wb']},{v['u']}"


# Conflicting timestamps of known size. Consecutive on the grid: 1000..2200; isolated: 4000.
CONFLICTS = "\n".join([
    row(1000), row(1000, u=200),                          # usage only
    row(1300, rb=100), row(1300, rb=101),                 # rounding level (1 KB/s)
    row(1600, rb=100), row(1600, rb=105),                 # small (5 / 105)
    row(1900, ri=10), row(1900, ri=20),                   # large (10 / 20)
    row(2200, ri=5), row(2200, ri=6), row(2200, ri=7),    # three copies, 2 / 7
    row(4000), row(4000, u=101),                          # usage only, separate block
]) + "\n"


def test_conflict_detail_buckets_and_blocks():
    clean, stats, c = ex.clean_trace(parsed(CONFLICTS))
    assert len(clean) == 0 and stats["conflict_timestamps"] == 6 and stats["conflict_rows_dropped"] == 13
    assert c["block_lengths"].tolist() == [5, 1]
    assert c["rows_per_ts"].tolist() == [2, 2, 2, 2, 3, 2]
    assert c["usage_spread"].tolist() == [100, 0, 0, 0, 0, 1]
    assert c["load_abs"].tolist() == [0, 1, 5, 10, 2, 0]
    assert c["load_rel"][3] == pytest.approx(0.5) and c["load_rel"][2] == pytest.approx(5 / 105)


def test_conflict_lines_report_buckets():
    _, _, c = ex.clean_trace(parsed(CONFLICTS))
    lines = ex.conflict_lines(c, disks_with_out_of_order=4)
    assert "conflict_diagnostic timestamps=6" in lines
    assert "conflict_load_identical_usage_differs=2" in lines
    assert "conflict_load_rounding_level(abs<=1)=1" in lines
    assert "conflict_load_small(rel<=0.1)=1" in lines
    assert "conflict_load_large(rel>0.1)=2" in lines
    assert "conflict_rows_per_timestamp pairs=5 more_than_two=1 max=3" in lines
    assert any(l.startswith("conflict_blocks=2 block_rows median=3 ") and l.endswith("multiple_of_288=0")
               for l in lines)
    assert "disks_with_conflicts_and_out_of_order=4" in lines
    assert ex.conflict_lines(None) == ["conflict_diagnostic timestamps=0"]


def test_extract_returns_conflict_detail_and_summary_columns(dataset):
    r = ex.extract(dataset)
    assert r["conflicts"]["load_abs"].tolist() == [8]
    assert r["disks_conflict_and_out_of_order"] == 1
    summary = pq.read_table(ex.OUT_SUMMARY).to_pandas().set_index("disk_uid")
    assert summary.loc["d2", "median_total_bw_kbps"] == 150.0
    assert summary.loc["d2", "p99_total_bw_kbps"] == 150.0
    assert summary.loc["d1", "median_total_bw_kbps"] == 30.0


def test_trace_runs():
    clean, _, _ = ex.clean_trace(parsed(D1))
    starts, lengths = ex.trace_runs(clean)
    assert starts.tolist() == [1000, 3100, 3250] and lengths.tolist() == [4, 1, 1]
    s0, l0 = ex.trace_runs(clean.iloc[0:0])
    assert len(s0) == 0 and len(l0) == 0


def test_runs_table_written_by_extract(dataset):
    ex.extract(dataset)
    runs = pq.read_table(ex.OUT_RUNS).to_pandas()
    assert runs["disk_uid"].tolist() == ["d1", "d1", "d1", "d2"]
    assert runs["run_rows"].tolist() == [4, 1, 1, 5]


def runs_frame(rows):
    return pd.DataFrame(rows, columns=["disk_uid", "run_start_ts", "run_rows"])


def test_shared_window_stats():
    runs = runs_frame([
        ("A", 0, 600),         # window starts 0..312 (units of 300 s)
        ("B", 30000, 400),     # 100..212, overlaps A
        ("C", 120000, 288),    # 400..400, overlaps nobody
        ("D", 0, 100),         # shorter than a day
        ("E", 150, 300),       # other sampling offset
        ("A", 900000, 300),    # second run of A: counted once
    ])
    w = ex.shared_window_stats(runs)
    assert w == {"disks_with_window": 4, "offset_classes": 2, "shared_disks": 2,
                 "largest_group": 2, "largest_group_start_ts": 30000}


def test_shared_window_stats_none_qualify():
    w = ex.shared_window_stats(runs_frame([("A", 0, 100)]))
    assert w["disks_with_window"] == 0 and w["largest_group"] == 0
    assert w["largest_group_start_ts"] is None
    assert ex.shared_window_stats(runs_frame([]))["disks_with_window"] == 0


def test_fmt_quantiles():
    assert ex.fmt_quantiles([]) == "median=nan p90=nan p99=nan max=nan"
    assert ex.fmt_quantiles([1, 2, 3, 4, 5]).startswith("median=3 ")
    assert ex.fmt_quantiles([1, 2, 3, 4, 5]).endswith("max=5")


def test_aggregation_lines_on_small_dataset(dataset):
    ex.extract(dataset)
    lines = ex.aggregation_lines(ex.OUT_SUMMARY, ex.OUT_RUNS)
    assert "agg_disks=3 disks_with_clean_rows=2" in lines
    assert "agg_burstiness_p99_over_median disks_defined=2 disks_median_zero=0 " \
           "median=1 p90=1 p99=1 max=1" in lines
    assert "agg_disks_with_24h_window=0 sampling_offset_classes=0" in lines
    assert "agg_largest_24h_group=0" in lines


def test_aggregation_lines_with_shared_windows(dataset, tmp_path):
    ex.extract(dataset)
    crafted = tmp_path / "runs.parquet"
    pq.write_table(ex.runs_table(
        ["A", "B"], [np.array([0, 900000]), np.array([30000])],
        [np.array([600, 300]), np.array([400])]), crafted)
    lines = ex.aggregation_lines(ex.OUT_SUMMARY, crafted)
    assert "agg_disks_with_24h_window=2 sampling_offset_classes=1" in lines
    assert "agg_disks_sharing_a_24h_window=2" in lines
    assert "agg_largest_24h_group=2 window_start_ts=30000 window_start_utc=1970-01-01 08:20" in lines


def test_aggregation_lines_zero_median_disks_are_excluded(dataset):
    ex.extract(dataset)
    summary = pq.read_table(ex.OUT_SUMMARY).to_pandas()
    summary.loc[summary["disk_uid"] == "d2", "median_total_bw_kbps"] = 0.0
    pq.write_table(ex.pa.Table.from_pandas(summary, preserve_index=False), ex.OUT_SUMMARY)
    lines = ex.aggregation_lines(ex.OUT_SUMMARY, ex.OUT_RUNS)
    assert any(l.startswith("agg_burstiness_p99_over_median disks_defined=1 disks_median_zero=1 ")
               for l in lines)


def test_main_prints_conflict_and_aggregation_sections(dataset, capsys):
    assert ex.main(["extract_load.py", str(dataset)]) == 0
    out = capsys.readouterr().out
    assert "conflict_diagnostic timestamps=1" in out
    assert "agg_mean_total_bw_kbps " in out and "agg_peak_total_bw_kbps " in out
    assert "tencent_disk_runs.parquet" in ex.PREPROCESSING.read_text(encoding="utf-8")


def test_main_aggregation_only_does_not_rewrite_outputs(dataset, capsys):
    ex.extract(dataset)
    before = (ex.OUT_SUMMARY.read_bytes(), ex.MANIFEST.read_text(encoding="utf-8"),
              ex.PREPROCESSING.read_text(encoding="utf-8"))
    assert ex.main(["extract_load.py", "--aggregation-only"]) == 0
    assert "agg_disks=3" in capsys.readouterr().out
    assert before == (ex.OUT_SUMMARY.read_bytes(), ex.MANIFEST.read_text(encoding="utf-8"),
                      ex.PREPROCESSING.read_text(encoding="utf-8"))
