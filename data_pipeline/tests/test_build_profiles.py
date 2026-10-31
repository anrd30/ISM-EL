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
    assert bp.tune_n(curve, 10.0, "not_above") == 1           # none qualifies: smallest N
    assert bp.tune_n(np.array([1.0, 2.0]), 100.0, "not_above") == 2


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
                                 "window_p99_over_median"]
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
    cand = pd.DataFrame({"disk_uid": ["a"], "burstiness": [np.nan]})
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
                                       "daily_amplitude_kbps", "window_p99_over_median"]


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
    assert sum(l.startswith("output=") for l in lines) == 4 + 4 + 2 + 1
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
    shutil.copytree(Path(contracts.__file__).parent, repo / "contracts",
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
