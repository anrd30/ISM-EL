"""Tests for the out-of-distribution evaluation and the workload-selection check.

Neither test needs ``hazard/artifacts/v1/`` or the Tencent pipeline outputs: the
evaluation helpers are tested on generator output and the committed holdout
format, and the selection check on a small synthetic pipeline dataset.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent.evaluate_ood import PHASE_SHIFT_STEPS, write_phase_shift, write_synthetic
from agent.residual_features import DAY_STEPS
from sim.scenario import load_replay_csv


def test_synthetic_days_differ_and_load(tmp_path: Path) -> None:
    path = write_synthetic("diurnal", 3, tmp_path)
    trace = load_replay_csv(path)
    fg = np.asarray(trace.fg_bw_mbps, dtype=np.float64)
    assert len(fg) == 3 * DAY_STEPS
    assert not np.array_equal(fg[:DAY_STEPS], fg[DAY_STEPS : 2 * DAY_STEPS])
    frame = pd.read_csv(path)
    assert (np.diff(frame["t_seconds"]) == 300).all()


def test_phase_shift_rotates_each_day(tmp_path: Path) -> None:
    rows = 2 * DAY_STEPS + 10
    source = pd.DataFrame({
        "t_seconds": np.arange(rows, dtype=np.int64) * 300,
        "read_iops": np.arange(rows, dtype=np.int64),
        "write_iops": np.arange(rows, dtype=np.int64),
        "read_bw_kbps": np.arange(rows, dtype=np.int64) * 10,
        "write_bw_kbps": np.arange(rows, dtype=np.int64) * 10,
        "disk_usage_mb": np.full(rows, 1000, dtype=np.int64),
    })
    src_path = tmp_path / "src.csv"
    source.to_csv(src_path, index=False)
    out = pd.read_csv(write_phase_shift(src_path, tmp_path / "out"))
    assert len(out) == rows
    np.testing.assert_array_equal(out["read_iops"].to_numpy()[:DAY_STEPS], np.arange(DAY_STEPS))
    day1 = out["read_iops"].to_numpy()[DAY_STEPS : 2 * DAY_STEPS]
    np.testing.assert_array_equal(day1, np.roll(np.arange(DAY_STEPS, 2 * DAY_STEPS), -PHASE_SHIFT_STEPS))
    np.testing.assert_array_equal(out["read_iops"].to_numpy()[2 * DAY_STEPS :], np.arange(2 * DAY_STEPS, rows))
    np.testing.assert_array_equal(out["t_seconds"], source["t_seconds"])


def test_workload_selection_check_on_a_small_fake_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    from agent import workload_selection_check as wsc
    from data_pipeline.tencent import build_profiles as bp
    from data_pipeline.tencent import extract_load as el

    rng = np.random.default_rng(0)
    rows = bp.LONG_WINDOW_ROWS + bp.DAY_ROWS
    start = 1_600_000_200 - (1_600_000_200 % el.STEP_S)
    ts = start + np.arange(rows, dtype="int64") * el.STEP_S
    phase = 2 * np.pi * (ts % 86400) / 86400
    tables, runs, summary = [], [], []
    for i in range(40):
        uid = f"disk{i:02d}"
        daily = (i % 4 == 0) * 15000 * (1 + np.sin(phase))
        burst = (i % 4 == 1) * 60000 * (rng.random(rows) < 0.02)
        bw = (5000 + 200 * i + daily + burst + rng.normal(0, 500, rows)).clip(10).astype("int64")
        frame = pd.DataFrame({"disk_uid": uid, "timestamp": ts, "read_iops": bw // 8, "write_iops": bw // 8,
                              "read_bw_kbps": bw // 2, "write_bw_kbps": bw - bw // 2, "disk_usage_mb": 1000})
        tables.append(pa.Table.from_pandas(frame[["disk_uid", "timestamp", *el.VALUE_COLUMNS]], preserve_index=False))
        runs.append({"disk_uid": uid, "run_start_ts": int(start), "run_rows": rows})
        summary.append({"disk_uid": uid, "mean_total_bw_kbps": float(bw.mean()), "zero_row_fraction": 0.0})
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    with pq.ParquetWriter(processed / "clean.parquet", tables[0].schema) as writer:
        for table in tables:
            writer.write_table(table)
    pd.DataFrame(runs).to_parquet(processed / "runs.parquet")
    pd.DataFrame(summary).to_parquet(processed / "summary.parquet")
    monkeypatch.setattr(bp, "IN_CLEAN", processed / "clean.parquet")
    monkeypatch.setattr(bp, "IN_RUNS", processed / "runs.parquet")
    monkeypatch.setattr(bp, "IN_SUMMARY", processed / "summary.parquet")
    monkeypatch.setattr(bp, "ROOT", tmp_path)

    # The committed long window, as the pipeline would build it on this data.
    r = pd.read_parquet(processed / "runs.parquet")
    window = el.shared_window_stats(r, bp.LONG_WINDOW_ROWS)["largest_group_start_ts"]
    uids = bp.eligible_disks(r, window, bp.LONG_WINDOW_ROWS)
    pf, groups = bp.row_group_map(processed / "clean.parquet")
    windows, periodicity = bp.collect(pf, groups, uids, window, bp.LONG_WINDOW_ROWS)
    cand = bp.candidate_frame(pd.read_parquet(processed / "summary.parquet"), periodicity)
    built = bp.build_profile("diurnal", bp.PROFILE_SPECS["diurnal"], cand, windows,
                             {u: i for i, u in enumerate(uids)}, window, bp.LONG_WINDOW_ROWS)
    (tmp_path / "profiles").mkdir()
    built["frame"].to_csv(tmp_path / "profiles" / "profile_diurnal_long.csv", index=False)
    (tmp_path / "profiles" / "profile_diurnal_long.meta.json").write_text(json.dumps({"disk_uids": built["uids"]}))

    assert wsc.main(["--out", str(tmp_path / "out")]) == 0
    result = json.loads((tmp_path / "out" / "selection_check.json").read_text())
    assert result["step1"]["ranked_candidates"] == 40
    assert result["step1"]["selected_min_percentile_rank"] > 50
    for name in ("bursty_long", "idle_long"):
        for suffix in ("", "_train", "_holdout"):
            assert (tmp_path / "out" / f"profile_{name}{suffix}.csv").is_file()
    assert result["step2"]["diurnal_long"]["daily_strength"] > result["step2"]["bursty_long"]["daily_strength"]

    (tmp_path / "profiles" / "profile_diurnal_long.meta.json").write_text(json.dumps({"disk_uids": ["other"]}))
    with pytest.raises(SystemExit):
        wsc.main(["--out", str(tmp_path / "out2")])
