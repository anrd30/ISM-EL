"""Tests for the synthetic workload generator.

These check schema identity with ``contracts.contracts``, determinism, profile
separation and bandwidth/IOPS consistency of the generated files. They do not
evaluate realism: the generated data is synthetic by construction.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contracts.contracts import (
    BW_TOTAL_MBPS,
    CONTROL_INTERVAL_S,
    REPLAY_COLUMNS,
    REPLAY_DTYPES,
    ContractViolation,
    kbps_to_mbps,
    validate_replay_frame,
    validate_replay_grid,
)
from workload.generate import (
    GENERATOR_MODULE,
    GENERATOR_VERSION,
    MEAN_REQUEST_KB,
    N_ROWS,
    READ_FRACTION,
    SYNTHETIC_MARKER_FILE,
    P95_FRACTION_OF_BUDGET,
    PROFILE_NAMES,
    generate_profile,
    main,
    p95_total_bw_mbps,
)

# Stated tolerance on p95 total bandwidth relative to its target fraction of
# BW_TOTAL_MBPS (relative error). The generator scales each series to its
# target, so this bounds calibration and integer-rounding error (the p95 is
# measured on the rounded integer columns), not statistical variation between
# seeds.
P95_REL_TOL = 0.02

# Stated range (KB) for the implied mean request size of every row,
# total_bw_kbps / (read_iops + write_iops).
REQUEST_KB_RANGE = (16.0, 64.0)

SEEDS = range(10)


def _total_kbps(frame: pd.DataFrame) -> np.ndarray:
    return (frame["read_bw_kbps"] + frame["write_bw_kbps"]).to_numpy()


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    return {name: generate_profile(name, seed=0) for name in PROFILE_NAMES}


def test_profile_names() -> None:
    assert PROFILE_NAMES == ("idle", "diurnal", "bursty", "peak")


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_shape_and_columns(frames: dict[str, pd.DataFrame], name: str) -> None:
    frame = frames[name]
    assert len(frame) == 288 == N_ROWS
    assert tuple(frame.columns) == REPLAY_COLUMNS


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_time_grid_valid(frames: dict[str, pd.DataFrame], name: str) -> None:
    t = frames[name]["t_seconds"]
    validate_replay_grid(t.to_numpy())
    assert t.iloc[0] == 0
    assert t.iloc[-1] == (288 - 1) * CONTROL_INTERVAL_S
    assert pd.api.types.is_integer_dtype(t)


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_dtypes_match_contract(frames: dict[str, pd.DataFrame], name: str) -> None:
    frame = frames[name]
    for column in REPLAY_COLUMNS:
        assert frame[column].dtype == REPLAY_DTYPES[column], column
    validate_replay_frame(frame)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_validate_replay_frame_passes_for_all_seeds(name: str, seed: int) -> None:
    validate_replay_frame(generate_profile(name, seed))


def test_float_frame_rejected_by_contract(frames: dict[str, pd.DataFrame]) -> None:
    # Guards against the generator regressing to float columns: the contract
    # check that generate_profile relies on must reject them.
    floaty = frames["peak"].astype(np.float64)
    with pytest.raises(ContractViolation):
        validate_replay_frame(floaty)
    one_float = frames["peak"].copy()
    one_float["read_iops"] = one_float["read_iops"].astype(np.float64)
    with pytest.raises(ContractViolation, match="read_iops"):
        validate_replay_frame(one_float)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_no_column_all_zero(name: str, seed: int) -> None:
    # Integer quantisation must not collapse any column to zero. t_seconds has
    # a zero first row but is non-zero overall; the others must be positive
    # in every row, since quantisation of the smallest profile stays well
    # above 1 unit.
    frame = generate_profile(name, seed)
    for column in REPLAY_COLUMNS:
        assert (frame[column] != 0).any(), column
    for column in REPLAY_COLUMNS[1:]:
        assert (frame[column] > 0).all(), column


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_no_nan_no_negative(frames: dict[str, pd.DataFrame], name: str) -> None:
    values = frames[name].to_numpy(dtype=np.float64)
    assert np.isfinite(values).all()
    assert (values >= 0).all()


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_total_bandwidth_within_group_budget(frames: dict[str, pd.DataFrame], name: str) -> None:
    peak_mbps = _total_kbps(frames[name]).max() / 1000.0
    assert peak_mbps <= BW_TOTAL_MBPS + 1e-6


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_same_seed_identical(name: str) -> None:
    pd.testing.assert_frame_equal(generate_profile(name, 7), generate_profile(name, 7), check_exact=True)


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_different_seeds_differ(name: str) -> None:
    a, b = generate_profile(name, 1), generate_profile(name, 2)
    assert not a.equals(b)


def test_profiles_use_independent_streams() -> None:
    # Under one seed, profiles must not share a noise stream.
    idle = generate_profile("idle", 0)
    peak = generate_profile("peak", 0)
    ci = np.corrcoef(_total_kbps(idle), _total_kbps(peak))[0, 1]
    assert abs(ci) < 0.5


def test_global_numpy_state_untouched() -> None:
    np.random.seed(123)
    expected = np.random.random(3)
    np.random.seed(123)
    for name in PROFILE_NAMES:
        generate_profile(name, 5)
    np.testing.assert_array_equal(np.random.random(3), expected)


def test_p95_strictly_ordered(frames: dict[str, pd.DataFrame]) -> None:
    p95 = [p95_total_bw_mbps(frames[name]) for name in PROFILE_NAMES]
    assert p95 == sorted(p95)
    assert len(set(p95)) == len(p95)
    assert p95[0] < p95[1] < p95[2] < p95[3]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_p95_near_target(name: str, seed: int) -> None:
    """Check the scaling step, not the process.

    The generator imposes each profile's p95 by construction (it rescales the
    shape so the empirical p95 equals the target). This test therefore verifies
    that the scaling and rounding are implemented correctly. It is not a
    validation result and does not show that any process reaches a given
    fraction of the group budget.
    """
    target_mbps = P95_FRACTION_OF_BUDGET[name] * BW_TOTAL_MBPS
    measured = p95_total_bw_mbps(generate_profile(name, seed))
    assert measured == pytest.approx(target_mbps, rel=P95_REL_TOL)


def test_target_fractions() -> None:
    assert P95_FRACTION_OF_BUDGET == {"idle": 0.10, "diurnal": 0.50, "bursty": 0.60, "peak": 0.80}


def test_bandwidth_is_kbps_not_mbps(frames: dict[str, pd.DataFrame]) -> None:
    # In KB/s the peak profile's p95 is ~160,000; a MB/s value would be ~160.
    p95_kbps = float(np.percentile(_total_kbps(frames["peak"]), 95))
    assert p95_kbps == pytest.approx(0.8 * BW_TOTAL_MBPS * 1000.0, rel=P95_REL_TOL)
    assert kbps_to_mbps(p95_kbps) == pytest.approx(0.8 * BW_TOTAL_MBPS, rel=P95_REL_TOL)


@pytest.mark.parametrize("seed", SEEDS)
def test_bursty_more_dispersed_than_diurnal(seed: int) -> None:
    # p99/median compares the extreme upper tail with the typical level. It is
    # invariant to overall scale, so it measures burstiness independently of the
    # (different) means and p95 targets of the two profiles. A smooth cycle has
    # a tail close to its peak and a median not far below it, giving a small
    # ratio; short bursts over a low baseline put p99 far above the median.
    def ratio(name: str) -> float:
        total = _total_kbps(generate_profile(name, seed))
        return float(np.percentile(total, 99) / np.median(total))

    bursty, diurnal = ratio("bursty"), ratio("diurnal")
    assert bursty > diurnal
    assert bursty > 2.0 * diurnal


def test_diurnal_has_single_peak_and_night_trough() -> None:
    total = _total_kbps(generate_profile("diurnal", 0))
    per_hour = total.reshape(24, -1).mean(axis=1)  # 12 intervals of 300 s per hour
    assert 10 <= int(np.argmax(per_hour)) <= 18
    assert int(np.argmin(per_hour)) in (22, 23, 0, 1, 2, 3, 4, 5, 6)
    assert per_hour.min() < 0.5 * per_hour.max()


def test_idle_is_flat() -> None:
    total = _total_kbps(generate_profile("idle", 0))
    assert total.std() / total.mean() < 0.15


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_iops_consistent_with_bandwidth(name: str, seed: int) -> None:
    frame = generate_profile(name, seed)
    iops = (frame["read_iops"] + frame["write_iops"]).to_numpy()
    assert (iops > 0).all()
    implied_kb = _total_kbps(frame) / iops
    low, high = REQUEST_KB_RANGE
    assert implied_kb.min() >= low
    assert implied_kb.max() <= high
    # Reads and writes individually, too.
    for iops_col, bw_col in (("read_iops", "read_bw_kbps"), ("write_iops", "write_bw_kbps")):
        per_op = (frame[bw_col] / frame[iops_col]).to_numpy()
        assert per_op.min() >= low and per_op.max() <= high


@pytest.mark.parametrize("name", ("diurnal", "bursty"))
def test_iops_tracks_bandwidth(frames: dict[str, pd.DataFrame], name: str) -> None:
    # IOPS must follow bandwidth over time rather than being drawn independently.
    # Only the profiles with large variation make this correlation meaningful.
    frame = frames[name]
    corr = np.corrcoef(_total_kbps(frame), (frame["read_iops"] + frame["write_iops"]).to_numpy())[0, 1]
    assert corr > 0.9


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_read_fraction(frames: dict[str, pd.DataFrame], name: str) -> None:
    frame = frames[name]
    share = frame["read_bw_kbps"].sum() / _total_kbps(frame).sum()
    assert share == pytest.approx(0.7, abs=0.03)


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_disk_usage_slow_drift(frames: dict[str, pd.DataFrame], name: str) -> None:
    usage = frames[name]["disk_usage_mb"].to_numpy()
    assert (np.diff(usage) >= 0).all()
    assert (usage.max() - usage.min()) / usage.mean() < 1e-3


def test_unknown_profile_raises() -> None:
    with pytest.raises(ValueError):
        generate_profile("nonexistent")
    with pytest.raises(ValueError):
        generate_profile("")


def test_cli_writes_files_and_reports(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--out-dir", str(tmp_path / "out"), "--seed", "4"]) == 0
    printed = capsys.readouterr().out.splitlines()
    assert "synthetic" in printed[0].lower()
    assert len(printed) == 1 + len(PROFILE_NAMES)
    for line, name in zip(printed[1:], PROFILE_NAMES):
        assert line.startswith(f"profile_{name}.csv: rows=288 p95_total_bw_mbps=")
        csv_path = tmp_path / "out" / f"profile_{name}.csv"
        on_disk = pd.read_csv(csv_path)
        assert tuple(on_disk.columns) == REPLAY_COLUMNS
        pd.testing.assert_frame_equal(on_disk, generate_profile(name, 4), check_exact=True)


def test_cli_writes_sidecars(tmp_path: Path) -> None:
    out = tmp_path / "out"
    main(["--out-dir", str(out), "--seed", "6"])
    marker = json.loads((out / SYNTHETIC_MARKER_FILE).read_text(encoding="utf-8"))
    assert marker["synthetic"] is True
    assert marker["generator_module"] == GENERATOR_MODULE
    assert "not trace data" in marker["statement"]
    for name in PROFILE_NAMES:
        meta = json.loads((out / f"profile_{name}.meta.json").read_text(encoding="utf-8"))
        assert meta["synthetic"] is True
        assert meta["generator_module"] == GENERATOR_MODULE
        assert meta["generator_version"] == GENERATOR_VERSION
        assert meta["profile"] == name
        assert meta["seed"] == 6
        assert meta["target_p95_fraction_of_bw_total"] == P95_FRACTION_OF_BUDGET[name]
        assert meta["measured_p95_total_bw_mbps"] == pytest.approx(
            p95_total_bw_mbps(generate_profile(name, 6)), abs=1e-6
        )
        assert meta["p95_imposed_by_construction"] is True
        assert meta["mean_request_kb"] == MEAN_REQUEST_KB[name]
        assert meta["read_fraction"] == READ_FRACTION
        datetime.fromisoformat(meta["generated_at_utc"])
    # The CSV itself carries no marker: header is exactly REPLAY_COLUMNS.
    header = (out / "profile_idle.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(REPLAY_COLUMNS)


def test_cli_output_byte_identical_per_seed(tmp_path: Path) -> None:
    main(["--out-dir", str(tmp_path / "a"), "--seed", "9"])
    main(["--out-dir", str(tmp_path / "b"), "--seed", "9"])
    main(["--out-dir", str(tmp_path / "c"), "--seed", "10"])
    differs = False
    for name in PROFILE_NAMES:
        fname = f"profile_{name}.csv"
        a = (tmp_path / "a" / fname).read_bytes()
        assert a == (tmp_path / "b" / fname).read_bytes()
        differs |= a != (tmp_path / "c" / fname).read_bytes()
    assert differs
