"""
Unit tests for tencent/build_profiles.py (B5) on synthetic volumes of known character.

Path constants and the profile specs of the module are redirected, so no test
touches the real manifest, PREPROCESSING.md, profiles or processed data.
"""
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

from contracts import contracts  # noqa: E402
from data_pipeline.tencent import build_profiles as bp  # noqa: E402
from data_pipeline.tencent import extract_load as el  # noqa: E402

T0 = 1596672000            # multiple of 86400 and of 300
ROWS = 16 * 288
PREPROCESSING_TEXT = ("# P\n\n## B4 — Metadata join\n\n- keep4\n\n"
                      "## B5 — Four intensity profiles\n\n- stub\n\n## B6 — Acceptance\n\n- keep6\n")


def trace(total, t0=T0):
    total = np.asarray(total, dtype="int64")
    n = len(total)
    read = total * 2 // 3
    iops = np.where(total > 0, 5, 0)
    return pd.DataFrame({
        "timestamp": t0 + np.arange(n, dtype="int64") * 300, "read_iops": iops, "write_iops": iops,
        "read_bw_kbps": read, "write_bw_kbps": total - read,
        "disk_usage_mb": np.full(n, 1000, dtype="int64"),
    })


def make_traces() -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(7)
    phase = 2 * np.pi * (np.arange(ROWS) % 288) / 288
    traces = {}
    for k in range(4):
        traces[f"hi{k}"] = trace(60000 - 5000 * k + rng.integers(0, 100, ROWS))
    traces["dn0"] = trace((5000 * (1 + 0.9 * np.sin(phase)) + rng.integers(0, 50, ROWS)).astype(int))
    traces["dn1"] = trace((5000 * (1 + 0.6 * np.sin(phase + 1)) + rng.integers(0, 1500, ROWS)).astype(int))
    for k, height in enumerate([50000, 30000, 10000, 2000]):
        spikes = (rng.random(ROWS) < 0.03) * height
        traces[f"bu{k}"] = trace(100 + rng.integers(0, 5, ROWS) + spikes)
    for k, level in enumerate([0, 0, 0, 50, 200]):
        traces[f"id{k}"] = trace(np.full(ROWS, level))
    traces["short"] = trace(np.full(3 * 288, 1000))
    traces["late"] = trace(np.full(ROWS, 3000), t0=T0 + 600 * 300)
    return traces


def write_world(root: Path, traces: dict[str, pd.DataFrame]) -> dict[str, Path]:
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    paths = {k: processed / f"tencent_{k}.parquet" for k in ("disk_summary", "disk_runs", "disk_index")}
    paths["load_clean"] = processed / "tencent_load_clean.parquet"
    uids = sorted(traces)
    with pq.ParquetWriter(paths["load_clean"], el.CLEAN_SCHEMA) as writer:
        for uid in uids:
            writer.write_table(el.clean_table(uid, traces[uid]), row_group_size=len(traces[uid]))
    rows = []
    for uid in uids:
        row = dict.fromkeys(el.SUMMARY_FIELDS, 0)
        row.update(el.trace_summary(traces[uid]))
        row["disk_uid"] = uid
        rows.append(row)
    pq.write_table(el.summary_table(rows), paths["disk_summary"])
    starts, lengths, run_uids = [], [], []
    for uid in uids:
        s, ln = el.trace_runs(traces[uid])
        starts.append(s)
        lengths.append(ln)
        run_uids.append(uid)
    pq.write_table(el.runs_table(run_uids, starts, lengths), paths["disk_runs"])
    index = pd.DataFrame({"disk_uid": uids, "disk_capacity": 50, "disk_attr": 1,
                          "disk_type": [i % 2 for i in range(len(uids))],
                          "user_type": [(i // 2) % 2 for i in range(len(uids))],
                          "vm_cpu": 2, "vm_memory": 4})
    pq.write_table(pa.Table.from_pandas(index, preserve_index=False), paths["disk_index"])
    return paths


@pytest.fixture
def world(tmp_path, monkeypatch):
    root = tmp_path / "pipe"
    traces = make_traces()
    paths = write_world(root, traces)
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"files": [{"file": "other.parquet", "rows": 1, "sha256": "x"}]}))
    prep = root / "PREPROCESSING.md"
    prep.write_text(PREPROCESSING_TEXT, encoding="utf-8")
    processed = root / "data" / "processed"
    for name, value in {
        "ROOT": root, "PROCESSED": processed, "IN_CLEAN": paths["load_clean"],
        "IN_SUMMARY": paths["disk_summary"], "IN_RUNS": paths["disk_runs"],
        "IN_INDEX": paths["disk_index"], "OUT_DISKS": processed / "tencent_profile_disks.parquet",
        "OUT_CANDIDATES": processed / "tencent_profile_candidates.parquet",
        "PROFILES_DIR": root / "profiles", "MANIFEST": manifest, "PREPROCESSING": prep,
    }.items():
        monkeypatch.setattr(bp, name, value)
    specs = {k: dict(v) for k, v in bp.PROFILE_SPECS.items()}
    specs["peak"]["start_n"] = 2
    monkeypatch.setattr(bp, "PROFILE_SPECS", specs)
    return traces


def window_totals(traces, uids):
    """Total bandwidth of the first 288 samples (the window at T0) per disk, brute force."""
    return {u: (traces[u]["read_bw_kbps"] + traces[u]["write_bw_kbps"]).to_numpy()[:288] for u in uids}


def full_mean(traces, uid):
    return float((traces[uid]["read_bw_kbps"] + traces[uid]["write_bw_kbps"]).mean())


def window_ratio(traces, uid):
    t = window_totals(traces, [uid])[uid]
    median = np.median(t)
    return np.percentile(t, 99) / median if median > 0 else np.nan


def p95_curve(order, traces):
    cum = np.cumsum([window_totals(traces, [u])[u] for u in order], axis=0)
    return np.array([np.percentile(c, 95) for c in cum]) / 1000.0


def expected_best_n(order, traces, target_mbps, mode="closest"):
    p95 = p95_curve(order, traces)
    if mode == "not_above":
        within = np.flatnonzero(p95 <= target_mbps * (1.0 + bp.CEILING_TOLERANCE))
        return int(within[-1]) + 1 if len(within) else 1
    return int(np.argmin(np.abs(p95 - target_mbps))) + 1


def test_constants_match_contract():
    assert bp.STEP_S == contracts.CONTROL_INTERVAL_S and bp.DAY_ROWS == 288
    assert bp.KBPS_PER_MBPS == contracts.kbps_to_mbps(1000.0) * 1000.0
    assert bp.PEAK_TARGET_MBPS == 0.8 * contracts.BW_TOTAL_MBPS
    assert bp.BURSTY_TARGET_MBPS == 0.6 * contracts.BW_TOTAL_MBPS
    assert bp.CEILING_TOLERANCE == 0.005
    assert {n: bp.PROFILE_SPECS[n]["mode"] for n in bp.PROFILE_NAMES} == {
        "idle": "not_above", "diurnal": "not_above", "bursty": "not_above", "peak": "not_above"}
    assert bp.VALUE_COLUMNS == list(contracts.REPLAY_COLUMNS[1:])


def test_daily_stats_sine_flat_and_constant():
    ts = T0 + np.arange(10 * 288) * 300
    phase = 2 * np.pi * (np.arange(10 * 288) % 288) / 288
    sine = bp.daily_stats(ts, 1000 + 500 * np.sin(phase))
    assert sine["daily_strength"] == pytest.approx(1.0) and sine["days"] == 10
    assert sine["daily_amplitude_kbps"] == pytest.approx(1000.0, rel=1e-3)
    noise = bp.daily_stats(ts, np.random.default_rng(0).integers(0, 100, len(ts)).astype(float))
    assert noise["daily_strength"] < 0.2
    flat = bp.daily_stats(ts, np.full(len(ts), 7.0))
    assert np.isnan(flat["daily_strength"]) and flat["daily_amplitude_kbps"] == 0.0


def test_eligible_disks_requires_full_covering_run():
    runs = pd.DataFrame([
        ("a", 1000, 400), ("b", 1000, 287), ("c", 1300, 400), ("d", 0, 200), ("d", 900, 288),
    ], columns=["disk_uid", "run_start_ts", "run_rows"])
    assert bp.eligible_disks(runs, 1000) == ["a"]
    assert bp.eligible_disks(runs, 900) == ["d"]


def test_rank_uids_orders_ties_and_drops_nan():
    cand = pd.DataFrame({"disk_uid": ["c", "a", "b", "d"], "k": [5.0, 5.0, 9.0, np.nan]})
    assert bp.rank_uids(cand, "k", ascending=False) == ["b", "a", "c"]
    assert bp.rank_uids(cand, "k", ascending=True) == ["a", "c", "b"]


def test_tune_n_picks_closest_and_smaller_on_tie():
    assert bp.tune_n(np.array([1.0, 8.0, 12.0, 30.0]), 10.0) == 2
    assert bp.tune_n(np.array([1.0, 9.0, 11.0]), 10.0) == 2
    assert bp.tune_n(np.array([5.0]), 100.0) == 1


def test_tune_n_not_above_takes_largest_n_within_target():
    curve = np.array([45.8, 78.7, 114.0, 144.4, 173.1, 195.5])
    assert bp.tune_n(curve, 160.0, "closest") == 5
    assert bp.tune_n(curve, 160.0, "not_above") == 4
    assert bp.tune_n(curve, 144.4, "not_above") == 4
    with pytest.raises(ValueError, match="no N satisfies the ceiling"):
        bp.tune_n(curve, 10.0, "not_above")                   # none qualifies
    assert bp.tune_n(np.array([1.0, 2.0]), 100.0, "not_above") == 2


def test_tune_n_raises_when_nothing_qualifies_stating_target_ceiling_and_p95_at_n1():
    with pytest.raises(ValueError) as err:
        bp.tune_n(np.array([50.0, 60.0, 70.0]), 40.0, "not_above")
    msg = str(err.value)
    assert "target 40 MB/s" in msg and "ceiling 40.2 MB/s" in msg and "p95 at N=1 50 MB/s" in msg


def test_tune_n_closest_never_raises():
    assert bp.tune_n(np.array([50.0, 60.0]), 10.0, "closest") == 1


def test_tune_n_not_above_allows_half_percent_tolerance():
    curve = np.array([19.0, 20.00025, 20.09, 20.11, 25.0])
    assert bp.tune_n(curve, 20.0, "not_above") == 3          # 20.09 <= 20.1
    assert bp.tune_n(np.array([99.0, 100.5, 100.51]), 100.0, "not_above") == 2
    assert bp.tune_n(np.array([99.0, 100.51]), 100.0, "not_above") == 1


def test_total_bw_stats_and_zero_median():
    s = bp.total_bw_stats(np.array([0, 0, 0, 1000.0]))
    assert np.isnan(s["p99_over_median"]) and s["max_mbps"] == 1.0
    t = bp.total_bw_stats(np.array([100.0] * 99 + [10000.0]))
    assert t["p99_over_median"] == pytest.approx((np.percentile([100.0] * 99 + [10000.0], 99)) / 100)
    assert t["p95_mbps"] == pytest.approx(0.1)


def test_row_group_map_with_statistics_and_mixed_group(tmp_path):
    p = tmp_path / "c.parquet"
    with pq.ParquetWriter(p, el.CLEAN_SCHEMA) as w:
        w.write_table(el.clean_table("a", trace([1, 2])), row_group_size=2)
        mixed = pa.concat_tables([el.clean_table("b", trace([3])), el.clean_table("c", trace([4]))])
        w.write_table(mixed, row_group_size=2)
    pf, groups = bp.row_group_map(p)
    assert groups == {"a": 0, "b": 1, "c": 1}
    assert bp.read_disk(pf, 1, "c")["read_bw_kbps"].tolist() == [2]


def test_collect_windows_and_error_on_incomplete_window(world, tmp_path):
    pf, groups = bp.row_group_map(bp.IN_CLEAN)
    windows, per = bp.collect(pf, groups, ["hi0", "short"], T0)
    expected = world["hi0"][bp.VALUE_COLUMNS].to_numpy()[:288]
    assert np.array_equal(windows[0], expected) and windows.shape == (2, 288, 5)
    assert per["disk_uid"].tolist() == ["hi0", "short"] and per["days"].tolist() == [16.0, 3.0]
    assert list(per.columns) == ["disk_uid", "days", "daily_strength", "daily_amplitude_kbps",
                                 "window_p99_over_median", "median_read_bytes_per_io",
                                 "median_write_bytes_per_io"]
    assert per["window_p99_over_median"].tolist() == pytest.approx(
        [window_ratio(world, "hi0"), window_ratio(world, "short")])
    with pytest.raises(ValueError, match="no complete window"):
        bp.collect(pf, groups, ["late"], T0)
    with pytest.raises(ValueError, match="no complete window"):
        bp.collect(pf, groups, ["short"], T0 + 2 * 86400 + 300)


def test_candidate_frame_flags(world):
    pf, groups = bp.row_group_map(bp.IN_CLEAN)
    _, per = bp.collect(pf, groups, ["bu0", "id0", "short"], T0)
    cand = bp.candidate_frame(pd.read_parquet(bp.IN_SUMMARY), per).set_index("disk_uid")
    assert cand.loc["bu0", "burstiness"] == pytest.approx(window_ratio(world, "bu0"))
    assert cand.loc["bu0", "burstiness"] > 100
    assert np.isnan(cand.loc["id0", "burstiness"])                    # window median 0
    assert np.isnan(cand.loc["short", "daily_strength"])              # fewer than MIN_DAYS
    assert np.isnan(cand.loc["id0", "daily_strength"])                # constant trace


def test_build_selects_by_rule_and_aggregates(world):
    r = bp.build()
    assert r["window_start"] == T0 and r["candidates"] == 16 and r["diurnal_candidates"] == 10
    by = {p["name"]: p for p in r["profiles"]}
    eligible = [u for u in world if u != "late"]

    by_mean = sorted(eligible, key=lambda u: (-full_mean(world, u), u))
    assert by["peak"]["uids"] == by_mean[: by["peak"]["n"]] == ["hi0", "hi1"]
    assert by["peak"]["n"] == expected_best_n(by_mean, world, 160.0, "not_above") == 2
    curve = p95_curve(by_mean, world)
    assert curve[1] <= 160.0 < curve[2] and by["peak"]["p95_mbps"] == pytest.approx(curve[1])
    ratios = {u: window_ratio(world, u) for u in eligible}
    by_ratio = sorted((u for u in eligible if np.isfinite(ratios[u])), key=lambda u: (-ratios[u], u))
    assert by_ratio[0] in {"bu0", "bu1", "bu2", "bu3"}
    assert by["bursty"]["uids"] == by_ratio[: by["bursty"]["n"]]
    assert by["bursty"]["n"] == expected_best_n(by_ratio, world, 120.0, "not_above")
    assert by["bursty"]["tuned"]
    assert by["diurnal"]["uids"][:2] == ["dn0", "dn1"]
    by_mean_asc = sorted(eligible, key=lambda u: (full_mean(world, u), u))
    assert by["idle"]["uids"] == by_mean_asc[: by["idle"]["n"]]
    assert by["idle"]["uids"][:3] == ["id0", "id1", "id2"]
    for name in ("diurnal", "idle"):
        assert by[name]["tuned"] and by[name]["n"] >= 2

    for p in r["profiles"]:
        w = window_totals(world, p["uids"])
        total = sum(w.values())
        frame = p["frame"]
        assert (frame["read_bw_kbps"] + frame["write_bw_kbps"]).to_numpy().tolist() == total.tolist()
        assert frame["disk_usage_mb"].tolist() == [1000 * p["n"]] * 288
        assert frame["t_seconds"].tolist() == list(range(0, 288 * 300, 300))
        contracts.validate_replay_frame(frame)
    assert by["peak"]["p95_at_start_n"] is not None and by["peak"]["start_n"] == 2


def test_tuned_n_matches_brute_force_for_idle_and_diurnal(world):
    r = bp.build()
    by = {p["name"]: p for p in r["profiles"]}
    pf, groups = bp.row_group_map(bp.IN_CLEAN)
    uids = bp.eligible_disks(pd.read_parquet(bp.IN_RUNS), T0)
    _, per = bp.collect(pf, groups, uids, T0)
    cand = bp.candidate_frame(pd.read_parquet(bp.IN_SUMMARY), per)
    cand = cand[~cand["implausible_io"]]
    for name, key, asc, target, mode in (
            ("idle", "mean_total_bw_kbps", True, 20.0, "not_above"),
            ("diurnal", "daily_strength", False, 100.0, "not_above"),
            ("bursty", "burstiness", False, 120.0, "not_above"),
            ("peak", "mean_total_bw_kbps", False, 160.0, "not_above")):
        order = bp.rank_uids(cand, key, asc)
        assert by[name]["n"] == expected_best_n(order, world, target, mode)
        assert by[name]["uids"] == order[: by[name]["n"]]


def test_start_n_beyond_candidates_is_reported_as_nan(world, monkeypatch):
    specs = {k: dict(v) for k, v in bp.PROFILE_SPECS.items()}
    specs["peak"]["start_n"] = 999
    monkeypatch.setattr(bp, "PROFILE_SPECS", specs)
    r = bp.build()
    peak = next(p for p in r["profiles"] if p["name"] == "peak")
    assert peak["p95_at_start_n"] is None
    assert "profile=peak start_n=999 p95_mbps_at_start_n=nan" in bp.report_lines(r)


def test_bursty_needs_enough_candidates(world, monkeypatch):
    specs = {k: dict(v) for k, v in bp.PROFILE_SPECS.items()}
    specs["bursty"]["fixed_n"] = 99
    monkeypatch.setattr(bp, "PROFILE_SPECS", specs)
    with pytest.raises(ValueError, match="fixed N is 99"):
        bp.build()


def test_fixed_n_overrides_tuning(world, monkeypatch):
    specs = {k: dict(v) for k, v in bp.PROFILE_SPECS.items()}
    specs["bursty"]["fixed_n"] = 3
    monkeypatch.setattr(bp, "PROFILE_SPECS", specs)
    bursty = next(p for p in bp.build()["profiles"] if p["name"] == "bursty")
    assert bursty["n"] == 3 and not bursty["tuned"] and len(bursty["uids"]) == 3


def test_build_profile_without_candidates_raises(world):
    cand = pd.DataFrame({"disk_uid": ["a"], "burstiness": [np.nan], "implausible_io": [False]})
    with pytest.raises(ValueError, match="no ranked candidates"):
        bp.build_profile("bursty", bp.PROFILE_SPECS["bursty"], cand,
                         np.zeros((1, 288, 5), dtype="int64"), {"a": 0}, T0)


def test_build_without_shared_window_raises(world):
    pq.write_table(el.runs_table(["x"], [np.array([T0])], [np.array([10])]), bp.IN_RUNS)
    with pytest.raises(SystemExit, match="no complete 24 h window"):
        bp.build()


def test_composition_counts(world):
    r = bp.build()
    index = pd.read_parquet(bp.IN_INDEX)
    comp = bp.composition(index, ["hi0", "hi1", "hi2"])
    sel = index[index["disk_uid"].isin(["hi0", "hi1", "hi2"])]
    assert comp["user_type"] == {int(k): int(v) for k, v in sel["user_type"].value_counts().sort_index().items()}
    assert sum(comp["disk_type"].values()) == 3
    assert all(set(p["composition"]) == {"user_type", "disk_type"} for p in r["profiles"])


def test_main_writes_validated_files(world, capsys):
    assert bp.main(["build_profiles.py"]) == 0
    out = capsys.readouterr().out
    assert "window_start_ts=1596672000 window_start_utc=2020-08-06 00:00" in out
    frames = {}
    for name in bp.PROFILE_NAMES:
        parquet = pq.read_table(bp.PROCESSED / f"tencent_profile_{name}.parquet")
        assert parquet.column_names == list(contracts.REPLAY_COLUMNS)
        assert all(str(f.type) == "int64" for f in parquet.schema)
        frame = parquet.to_pandas()
        csv = pd.read_csv(bp.PROFILES_DIR / f"profile_{name}.csv")
        pd.testing.assert_frame_equal(frame, csv)
        contracts.validate_replay_frame(frame)
        contracts.validate_replay_frame(csv)
        assert len(frame) == 288
        frames[name] = frame
    disks = pq.read_table(bp.OUT_DISKS).to_pandas()
    assert set(disks["profile"]) == set(bp.PROFILE_NAMES)
    top = disks[(disks["profile"] == "bursty") & (disks["rank"] == 1)]["disk_uid"].iloc[0]
    assert top in {"bu0", "bu1", "bu2", "bu3"}
    candidates = pq.read_table(bp.OUT_CANDIDATES)
    assert candidates.num_rows == 16
    assert candidates.column_names == ["disk_uid", "days", "daily_strength",
                                       "daily_amplitude_kbps", "window_p99_over_median",
                                       "median_read_bytes_per_io", "median_write_bytes_per_io",
                                       "implausible_io"]


def test_profiles_are_distinguishable(world):
    r = bp.build()
    stats = {}
    for p in r["profiles"]:
        total = (p["frame"]["read_bw_kbps"] + p["frame"]["write_bw_kbps"]).to_numpy(dtype=float)
        stats[p["name"]] = (round(p["p95_mbps"], 3), round(p["p99_over_median"], 3), tuple(total[:24]))
    assert len({s[2] for s in stats.values()}) == 4                     # no two identical series
    assert len({(s[0], s[1]) for s in stats.values()}) == 4             # distinct (p95, p99/median)
    by = {p["name"]: p for p in r["profiles"]}
    assert by["bursty"]["p99_over_median"] > by["diurnal"]["p99_over_median"] > by["peak"]["p99_over_median"]
    assert by["peak"]["p95_mbps"] > by["idle"]["p95_mbps"]
    series = {p["name"]: (p["frame"]["read_bw_kbps"] + p["frame"]["write_bw_kbps"]).to_numpy(float)
              for p in r["profiles"]}
    assert abs(np.corrcoef(series["bursty"], series["peak"])[0, 1]) < 0.95


def test_report_lines(world):
    r = bp.build()
    lines = bp.report_lines(r)
    assert lines[0].startswith("window_start_ts=1596672000 ")
    assert "candidate_disks=16 diurnal_candidates=10" in lines
    peak = next(l for l in lines if l.startswith("profile=peak n_disks="))
    assert "label=aggregate_of_2_highest-mean-bandwidth_volumes" in peak
    assert "tuned=True target_p95_mbps=160" in peak
    bursty = next(l for l in lines if l.startswith("profile=bursty n_disks="))
    assert "tuned=True target_p95_mbps=120" in bursty
    assert "rule=\"highest window-local p99/median of total bandwidth\"" in bursty
    assert any(l.startswith("profile=peak start_n=2 p95_mbps_at_start_n=") for l in lines)
    assert any(l.startswith("profile=idle mean_total_bw_kbps_range=") for l in lines)
    assert any(l.startswith("bursty_p99_over_median=") and "synthetic_reference_range=8.5-9.8" in l
               and l.endswith("ranking_statistic=window_p99_over_median selected_on_reported_statistic=True")
               for l in lines)
    assert any(l.startswith("overlap_idle_peak=") for l in lines)
    assert sum(l.startswith("output=") for l in lines) == 4 + 4 + 4 + 2 + 1
    png_line = next(l for l in lines if l.startswith("output=profiles/profile_comparison.png"))
    assert " rows=" not in png_line and " sha256=" in png_line


def test_report_flags_flat_bursty_profile(world):
    r = bp.build()
    next(p for p in r["profiles"] if p["name"] == "bursty")["p99_over_median"] = 3.0
    assert "below_reference=True" in " ".join(bp.report_lines(r))
    next(p for p in r["profiles"] if p["name"] == "bursty")["p99_over_median"] = float("nan")
    text = " ".join(bp.report_lines(r))
    assert "bursty_p99_over_median=nan" in text and "below_reference=undefined" in text


def test_plot_is_written_with_one_panel_per_profile(world):
    r = bp.build()
    png = bp.PROFILES_DIR / "profile_comparison.png"
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(r["profiles"]) == 4
    bp.plot_profiles(r["profiles"], r["window_start"], png.with_name("again.png"))
    assert png.with_name("again.png").read_bytes() == png.read_bytes()


def test_fmt():
    assert bp.fmt(None) == "nan" and bp.fmt(float("nan")) == "nan"
    assert bp.fmt(160.0) == "160" and bp.fmt(0.123456) == "0.1235"


def test_manifest_and_preprocessing_record_each_profile(world):
    assert bp.main(["build_profiles.py"]) == 0
    manifest = json.loads(bp.MANIFEST.read_text(encoding="utf-8"))
    assert set(manifest["profiles"]) == set(bp.PROFILE_NAMES)
    peak = manifest["profiles"]["peak"]
    assert peak["n_disks"] == 2 and peak["tuned"] is True and peak["target_p95_mbps"] == 160.0
    assert peak["label"] == ("aggregate of 2 highest-mean-bandwidth volumes "
                             "(selected extremes, not a representative sample)")
    assert peak["selection_rule"] == "highest mean total bandwidth"
    assert peak["n_selection_mode"] == "not_above" and peak["p95_mbps"] <= 160.0
    bursty = manifest["profiles"]["bursty"]
    assert bursty["tuned"] is True and bursty["target_p95_mbps"] == 120.0
    assert bursty["n_selection_mode"] == "not_above"
    assert "selection on the outcome" in bursty["disclosure"]
    assert "high partly by construction" in bursty["disclosure"]
    assert all("not a representative sample" in m["disclosure"] for m in manifest["profiles"].values())
    files = {e["file"] for e in manifest["files"]}
    assert "profiles/profile_idle.csv" in files and "data/processed/tencent_profile_peak.parquet" in files
    assert "profiles/profile_comparison.png" in files
    png_entry = next(e for e in manifest["files"] if e["file"] == "profiles/profile_comparison.png")
    assert png_entry["rows"] is None and len(png_entry["sha256"]) == 64
    assert {n: manifest["profiles"][n]["n_selection_mode"] for n in bp.PROFILE_NAMES} == {
        "idle": "not_above", "diurnal": "not_above", "bursty": "not_above", "peak": "not_above"}
    assert "other.parquet" in files
    text = bp.PREPROCESSING.read_text(encoding="utf-8")
    assert "- `profile=peak n_disks=2" in text and "## B4 — Metadata join\n\n- keep4" in text
    assert "selection on the outcome" in text and "high partly by construction" in text
    assert "window-local p99/median" in text and "TODO" not in text
    assert "symmetric distance is the wrong fit for a one-sided feasibility ceiling" in text
    assert "target x 1.005" in text and "profiles/profile_comparison.png" in text
    assert "## B6 — Acceptance\n\n- keep6" in text and "- stub" not in text
    assert "not a representative sample" in text


def test_manifest_records_null_when_burstiness_undefined(world):
    r = bp.build()
    next(p for p in r["profiles"] if p["name"] == "bursty")["p99_over_median"] = float("nan")
    bp.update_manifest(r)
    assert json.loads(bp.MANIFEST.read_text(encoding="utf-8"))["profiles"]["bursty"]["p99_over_median"] is None


def test_update_preprocessing_missing_section_raises(world):
    bp.PREPROCESSING.write_text("# P\n\n## B6 — x\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="B5 section not found"):
        bp.update_preprocessing([])


def test_main_is_idempotent(world):
    assert bp.main(["build_profiles.py"]) == 0
    first = (bp.PREPROCESSING.read_text(encoding="utf-8"), bp.MANIFEST.read_text(encoding="utf-8"),
             (bp.PROFILES_DIR / "profile_peak.csv").read_bytes())
    assert bp.main(["build_profiles.py"]) == 0
    assert (bp.PREPROCESSING.read_text(encoding="utf-8"), bp.MANIFEST.read_text(encoding="utf-8"),
            (bp.PROFILES_DIR / "profile_peak.csv").read_bytes()) == first


def test_cli_entry_point_in_subprocess(tmp_path, world):
    repo = tmp_path / "repo"
    pipe = repo / "data_pipeline"
    (pipe / "tencent").mkdir(parents=True)
    for package in ("contracts", "sim", "hazard", "workload"):
        shutil.copytree(Path(contracts.__file__).parents[1] / package, repo / package,
                        ignore=shutil.ignore_patterns("__pycache__", "test_*"))
    for name in ("__init__.py", "extract_load.py"):
        shutil.copy(Path(bp.__file__).parent / name, pipe / "tencent" / name)
    shutil.copy(Path(bp.__file__).parent / "build_profiles.py", pipe / "tencent" / "build_profiles.py")
    shutil.copytree(bp.PROCESSED, pipe / "data" / "processed")
    (pipe / "manifest.json").write_text('{"files": []}')
    (pipe / "PREPROCESSING.md").write_text(PREPROCESSING_TEXT, encoding="utf-8")
    res = subprocess.run([sys.executable, "-I", str(pipe / "tencent" / "build_profiles.py")],
                         capture_output=True, text=True, cwd=tmp_path)
    assert res.returncode == 0, res.stderr
    assert "profile=bursty n_disks=" in res.stdout
    assert len(pd.read_csv(pipe / "profiles" / "profile_peak.csv")) == 288
    assert (pipe / "profiles" / "profile_peak.meta.json").exists()


# -- provenance sidecars, latency ladder, IOPS facts --------------------------------------


def profile_frame(read_kbps, write_kbps, read_iops=None, write_iops=None):
    """A 288-row replay frame from per-row bandwidth arrays (KB/s)."""
    n = len(read_kbps)
    return pd.DataFrame({
        "t_seconds": np.arange(n, dtype="int64") * 300,
        "read_iops": np.asarray(read_iops if read_iops is not None else np.full(n, 10), dtype="int64"),
        "write_iops": np.asarray(write_iops if write_iops is not None else np.full(n, 20), dtype="int64"),
        "read_bw_kbps": np.asarray(read_kbps, dtype="int64"),
        "write_bw_kbps": np.asarray(write_kbps, dtype="int64"),
        "disk_usage_mb": np.full(n, 1000, dtype="int64"),
    })


def brute_force_ladder(frame):
    """Per-action violating share, with plain float arithmetic and the configured constants."""
    from sim.config import SimConfig

    cfg = SimConfig()
    limit_ms = contracts.LATENCY_LIMIT_FACTOR * cfg.latency_target_ms
    fg = ((frame["read_bw_kbps"] + frame["write_bw_kbps"]) / 1000.0).tolist()
    fractions = {}
    for action in contracts.ACTIONS_MBPS:
        violating = 0
        for load in fg:
            headroom = max(0.0, cfg.rho_max * contracts.BW_TOTAL_MBPS - load)
            applied = min(float(action), headroom)
            rho = min(cfg.rho_max, (load + applied) / contracts.BW_TOTAL_MBPS)
            violating += (cfg.latency_unloaded_ms / (1.0 - rho)) > limit_ms
        fractions[int(action)] = violating / len(fg)
    return fractions


def test_latency_ladder_matches_a_brute_force_count_on_a_hand_built_frame():
    """100 MB/s for half the day, 10 MB/s for the rest: only the 120 MB/s action violates
    (at 100 MB/s its applied share is capped at 96 MB/s and rho reaches 0.98)."""
    read = np.r_[np.full(144, 60000), np.full(144, 6000)]
    write = np.r_[np.full(144, 40000), np.full(144, 4000)]
    frame = profile_frame(read, write)
    ladder = bp.latency_ladder(frame, "hand")
    assert ladder["violation_fraction_by_action_mbps"] == pytest.approx(brute_force_ladder(frame))
    assert ladder["violation_fraction_by_action_mbps"] == pytest.approx({20: 0.0, 50: 0.0, 80: 0.0, 120: 0.5})
    assert ladder["safe_actions"] == 3
    assert list(ladder["violation_fraction_by_action_mbps"]) == list(contracts.ACTIONS_MBPS)


def test_latency_ladder_when_the_load_alone_exceeds_the_utilisation_ceiling():
    """At 198 MB/s the headroom is 0, every action is capped to 0 and rho is clamped to 0.98."""
    frame = profile_frame(np.full(288, 120000), np.full(288, 78000))
    ladder = bp.latency_ladder(frame)
    assert ladder["violation_fraction_by_action_mbps"] == pytest.approx(brute_force_ladder(frame))
    assert set(ladder["violation_fraction_by_action_mbps"].values()) == {1.0}
    assert ladder["safe_actions"] == 0


def test_latency_ladder_counts_a_partial_share_against_the_safe_limit():
    """At 190 MB/s every action violates (applied allocation capped at 6 MB/s, rho 0.98), so the
    violating share is the same for all four: 3 rows of 288 (1.04%) is within
    LATENCY_VIOLATION_FRACTION_MAX, 20 rows (6.9%) is not."""
    for violating_rows, safe in ((3, 4), (20, 0)):
        total = np.full(288, 10000)
        total[:violating_rows] = 190000
        frame = profile_frame(total // 2, total - total // 2)
        ladder = bp.latency_ladder(frame)
        assert ladder["violation_fraction_by_action_mbps"] == pytest.approx(brute_force_ladder(frame))
        assert ladder["safe_actions"] == safe


def test_latency_ladder_rejects_a_frame_that_breaks_the_replay_schema():
    frame = profile_frame(np.full(288, 1000), np.full(288, 1000))
    frame.loc[5, "read_bw_kbps"] = -1
    with pytest.raises(contracts.ContractViolation):
        bp.latency_ladder(frame)


def test_iops_facts_on_a_hand_built_frame():
    read_iops = np.r_[np.full(10, 25000), np.zeros(10, dtype=int), np.full(268, 100)]
    write_iops = np.full(288, 50)
    read_kbps = np.r_[np.full(10, 1000), np.zeros(10, dtype=int), np.full(268, 400)]
    frame = profile_frame(read_kbps, np.full(288, 100), read_iops, write_iops)
    facts = bp.iops_facts(frame)
    assert facts["max_read_iops"] == 25000 and facts["max_write_iops"] == 50
    assert facts["fraction_rows_read_iops_above_obs_scale"] == pytest.approx(10 / 288)
    assert facts["fraction_rows_write_iops_above_obs_scale"] == 0.0
    # bytes per I/O = KB/s x 1000 / IOPS, over the rows with IOPS > 0 (the 10 zero-IOPS rows are excluded)
    assert facts["median_bytes_per_read_io"] == pytest.approx(400 * 1000 / 100)
    assert facts["median_bytes_per_write_io"] == pytest.approx(100 * 1000 / 50)
    assert contracts.OBS_SCALES["fg_read_iops"] == 20000.0


def test_iops_facts_without_any_active_row_has_no_median():
    frame = profile_frame(np.zeros(288, dtype=int), np.zeros(288, dtype=int),
                          np.zeros(288, dtype=int), np.zeros(288, dtype=int))
    facts = bp.iops_facts(frame)
    assert facts["median_bytes_per_read_io"] is None and facts["median_bytes_per_write_io"] is None
    assert facts["max_read_iops"] == 0


def test_sidecars_are_written_deterministically_and_read_by_the_simulator(world):
    from sim import scenario

    r = bp.build()
    for p in r["profiles"]:
        sidecar = bp.PROFILES_DIR / f"profile_{p['name']}.meta.json"
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        assert set(payload) == {"profile", "synthetic", "source", "n_disks", "window_start_ts",
                                "window_start_utc", "label", "disclosure", "selection_rule"}
        assert payload["profile"] == p["name"] and payload["synthetic"] is False
        assert payload["source"] == "Tencent Cloud Volume Workload Traces"
        assert payload["n_disks"] == p["n"] and payload["window_start_ts"] == T0
        assert payload["window_start_utc"] == "2020-08-06T00:00:00Z"
        assert payload["label"] == bp.profile_label(p)
        assert payload["selection_rule"] == bp.PROFILE_SPECS[p["name"]]["rule"]
        assert payload["disclosure"] == bp.PROFILE_SPECS[p["name"]]["disclosure"]
        raw = sidecar.read_bytes()
        assert raw == (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
        assert b"\r" not in raw and raw.endswith(b"}\n")

        trace = scenario.load_replay_csv(bp.PROFILES_DIR / f"profile_{p['name']}.csv")
        assert trace.synthetic is False and trace.profile == p["name"]


def test_sidecars_are_listed_in_the_manifest_without_a_row_count(world):
    assert bp.main(["build_profiles.py"]) == 0
    manifest = json.loads(bp.MANIFEST.read_text(encoding="utf-8"))
    entries = {e["file"]: e for e in manifest["files"]}
    for name in bp.PROFILE_NAMES:
        entry = entries[f"profiles/profile_{name}.meta.json"]
        assert entry["rows"] is None
        assert entry["sha256"] == bp.sha256_of(bp.PROFILES_DIR / f"profile_{name}.meta.json")


def test_sidecars_are_identical_on_a_rerun(world):
    bp.build()
    first = {n: (bp.PROFILES_DIR / f"profile_{n}.meta.json").read_bytes() for n in bp.PROFILE_NAMES}
    bp.build()
    assert first == {n: (bp.PROFILES_DIR / f"profile_{n}.meta.json").read_bytes() for n in bp.PROFILE_NAMES}


def test_manifest_and_report_record_the_ladder_and_iops_facts(world):
    assert bp.main(["build_profiles.py"]) == 0
    block = json.loads(bp.MANIFEST.read_text(encoding="utf-8"))["profiles"]
    text = bp.PREPROCESSING.read_text(encoding="utf-8")
    for name in bp.PROFILE_NAMES:
        entry = block[name]
        assert list(entry["violation_fraction_by_action_mbps"]) == [str(a) for a in contracts.ACTIONS_MBPS]
        assert 0 <= entry["safe_actions"] <= len(contracts.ACTIONS_MBPS)
        assert set(entry["iops_facts"]) == {
            "max_read_iops", "max_write_iops", "fraction_rows_read_iops_above_obs_scale",
            "fraction_rows_write_iops_above_obs_scale", "median_bytes_per_read_io",
            "median_bytes_per_write_io"}
        assert f"profile={name} safe_actions={entry['safe_actions']} violation_fraction_by_action_mbps=" in text
        assert f"profile={name} max_read_iops={entry['iops_facts']['max_read_iops']} " in text
    assert "Latency ladder" in text and "IOPS facts" in text and "Sidecars" in text


def test_undershoot_wording_names_the_ladder_check(world):
    assert bp.main(["build_profiles.py"]) == 0
    text = bp.PREPROCESSING.read_text(encoding="utf-8")
    assert "undershoot can add a safe action" in text and "latency_ladder" in text
    assert "only makes the profile lighter" not in text
    doc = " ".join(bp.__doc__.split())
    assert "only makes the profile" not in doc and "can add a safe action" in doc


# -- physical-plausibility eligibility rule ------------------------------------------------


def tiny_io_volume():
    """70 MB/s (above every other synthetic volume) at about 47 bytes per read I/O."""
    frame = trace(np.full(ROWS, 70000))
    frame["read_iops"] = 1_000_000
    frame["write_iops"] = 1_000_000
    return frame


def world_with_tiny_io_volume():
    traces = make_traces()
    traces["tiny"] = tiny_io_volume()
    write_world(bp.ROOT, traces)
    return traces


def test_median_bytes_per_io_uses_only_rows_with_iops_above_zero():
    bw = np.array([1000.0, 5000.0, 3000.0, 0.0, 9999.0])
    iops = np.array([10.0, 50.0, 0.0, 0.0, 100.0])
    # rows with iops > 0: 1000*1000/10, 5000*1000/50, 9999*1000/100 -> 100000, 100000, 99990
    assert bp.median_bytes_per_io(bw, iops) == pytest.approx(100000.0)
    assert np.isnan(bp.median_bytes_per_io(np.array([5.0, 6.0]), np.zeros(2)))
    assert bp.MIN_BYTES_PER_IO == 512


def test_candidate_frame_flags_implausible_io_below_512_in_either_direction():
    """Strictly below 512 excludes; exactly 512 and an undefined (NaN) median do not."""
    uids = ["read_low", "at_threshold", "write_low", "both_undefined", "read_undefined_write_ok"]
    periodicity = pd.DataFrame({
        "disk_uid": uids, "days": 16.0, "daily_strength": 0.5, "daily_amplitude_kbps": 1.0,
        "window_p99_over_median": 2.0,
        "median_read_bytes_per_io": [511.99, 512.0, 4000.0, np.nan, np.nan],
        "median_write_bytes_per_io": [4000.0, 512.0, 100.0, np.nan, 600.0],
    })
    summary = pd.DataFrame({"disk_uid": uids, "mean_total_bw_kbps": 1.0, "zero_row_fraction": 0.0})
    cand = bp.candidate_frame(summary, periodicity).set_index("disk_uid")
    assert cand["implausible_io"].to_dict() == {
        "read_low": True, "at_threshold": False, "write_low": True,
        "both_undefined": False, "read_undefined_write_ok": False}
    assert cand["implausible_io"].dtype == bool


def test_collect_computes_the_median_bytes_per_io_of_the_whole_trace(world):
    pf, groups = bp.row_group_map(bp.IN_CLEAN)
    _, per = bp.collect(pf, groups, ["hi0", "id0"], T0)
    per = per.set_index("disk_uid")
    trace_hi0 = world["hi0"]
    expected = np.median(trace_hi0["read_bw_kbps"].to_numpy(float) * 1000.0 / trace_hi0["read_iops"].to_numpy(float))
    assert per.loc["hi0", "median_read_bytes_per_io"] == pytest.approx(expected)
    assert np.isnan(per.loc["id0", "median_read_bytes_per_io"])      # zero IOPS everywhere
    assert np.isnan(per.loc["id0", "median_write_bytes_per_io"])


def test_a_volume_with_tiny_bytes_per_io_is_excluded_from_every_profile(world, monkeypatch):
    world_with_tiny_io_volume()
    r = bp.build()
    assert r["candidates"] == 17 and r["excluded"] == 1
    for p in r["profiles"]:
        assert "tiny" not in p["uids"], p["name"]
        assert p["candidates"] <= 16
    candidates = pd.read_parquet(bp.OUT_CANDIDATES).set_index("disk_uid")
    assert bool(candidates.loc["tiny", "implausible_io"]) is True
    assert candidates.loc["tiny", "median_read_bytes_per_io"] < bp.MIN_BYTES_PER_IO
    assert int(candidates["implausible_io"].sum()) == 1

    # control: without the rule the same volume is the highest-mean candidate and enters the peak profile
    monkeypatch.setattr(bp, "MIN_BYTES_PER_IO", 0)
    control = bp.build()
    assert control["excluded"] == 0
    assert "tiny" in next(p for p in control["profiles"] if p["name"] == "peak")["uids"]


def test_the_rule_does_not_depend_on_any_profile_target(world, monkeypatch):
    world_with_tiny_io_volume()
    excluded = bp.build()["excluded"]
    specs = {k: dict(v) for k, v in bp.PROFILE_SPECS.items()}
    for spec in specs.values():
        spec["target_mbps"] = spec["target_mbps"] * 0.5
    monkeypatch.setattr(bp, "PROFILE_SPECS", specs)
    assert bp.build()["excluded"] == excluded == 1


def test_a_volume_with_zero_iops_everywhere_is_not_excluded(world):
    r = bp.build()
    assert r["excluded"] == 0
    candidates = pd.read_parquet(bp.OUT_CANDIDATES).set_index("disk_uid")
    for uid in ("id0", "id1", "id2"):                       # zero bandwidth and zero IOPS in every row
        assert np.isnan(candidates.loc[uid, "median_read_bytes_per_io"])
        assert np.isnan(candidates.loc[uid, "median_write_bytes_per_io"])
        assert bool(candidates.loc[uid, "implausible_io"]) is False
    idle = next(p for p in r["profiles"] if p["name"] == "idle")
    assert {"id0", "id1", "id2"} <= set(idle["uids"])


def test_the_candidates_table_has_the_three_new_columns(world):
    bp.build()
    table = pq.read_table(bp.OUT_CANDIDATES)
    assert table.column_names == ["disk_uid", "days", "daily_strength", "daily_amplitude_kbps",
                                  "window_p99_over_median", "median_read_bytes_per_io",
                                  "median_write_bytes_per_io", "implausible_io"]
    assert str(table.schema.field("implausible_io").type) == "bool"
    assert table.num_rows == 16


def test_report_manifest_and_text_record_the_eligibility_rule(world):
    world_with_tiny_io_volume()
    assert bp.main(["build_profiles.py"]) == 0
    text = bp.PREPROCESSING.read_text(encoding="utf-8")
    assert "- `min_bytes_per_io=512 candidates_excluded_by_rule=1 candidates_after_rule=16`" in text
    for name in bp.PROFILE_NAMES:
        assert f"- `profile={name} ranked_candidates=" in text
    assert "Eligibility rule" in text and "one disk sector" in text
    assert "unreliable IOPS counter" in text and "does not depend on any profile target" in text
    assert "before any ranking" in text and "IOPS facts" in text
    manifest = json.loads(bp.MANIFEST.read_text(encoding="utf-8"))
    assert manifest["profile_eligibility"] == {
        "min_bytes_per_io": 512, "candidates": 17, "excluded_implausible_io": 1}
    assert all(0 < entry["ranked_candidates"] <= 16 for entry in manifest["profiles"].values())
    # the sidecars carry the new n_disks and label
    for name in bp.PROFILE_NAMES:
        sidecar = json.loads((bp.PROFILES_DIR / f"profile_{name}.meta.json").read_text(encoding="utf-8"))
        assert sidecar["n_disks"] == manifest["profiles"][name]["n_disks"]

