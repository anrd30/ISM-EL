"""Optional pre-failure load history in the reset ``info`` (contract 1.5.0)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from baselines.evaluate import _pinned_config, evaluate
from baselines.policies import FixedRate, NeedsHistory, Policy
from contracts.contracts import (
    BW_TOTAL_MBPS,
    CONTRACT_VERSION,
    PRE_FAILURE_HISTORY_ROWS,
    RESET_HISTORY_INFO_KEYS,
)
from sim.config import SimConfig, v0_config, v1_eval_config
from sim.recovery_env import RecoveryEnv
from sim.scenario import trace_from_frame

REPO = Path(__file__).resolve().parents[1]
HOLDOUT = REPO / "data_pipeline" / "profiles" / "profile_diurnal_long_holdout.csv"
LONG = REPO / "data_pipeline" / "profiles" / "profile_diurnal_long.csv"
EVIDENCE = REPO / "baselines" / "evidence" / "diurnal-long-sigma0.25" / "episodes.csv"
needs_holdout = pytest.mark.skipif(not HOLDOUT.exists(), reason=f"{HOLDOUT.name} absent")
needs_v1 = pytest.mark.skipif(not (REPO / "hazard" / "artifacts" / "v1").exists(), reason="hazard/artifacts/v1 absent")
H = PRE_FAILURE_HISTORY_ROWS


def _config(history_rows: int = 0) -> SimConfig:
    return v0_config(history_rows=history_rows)


def _pinned(config: SimConfig) -> SimConfig:
    return _pinned_config(config, HOLDOUT)[0]


def _trace():
    return trace_from_frame(pd.read_csv(HOLDOUT), "holdout", False)


def test_contract_version_and_constants() -> None:
    assert CONTRACT_VERSION == "1.5.0"
    assert PRE_FAILURE_HISTORY_ROWS == 576
    assert RESET_HISTORY_INFO_KEYS == ("history_fg_bw_mbps", "history_read_iops", "history_write_iops")
    assert SimConfig(workload_profile="idle").history_rows == 0


@pytest.mark.parametrize("bad", [-1, 1.5, True, "576"])
def test_history_rows_must_be_a_non_negative_integer(bad: object) -> None:
    with pytest.raises(ValueError):
        v0_config(history_rows=bad)


@needs_holdout
def test_off_by_default_adds_nothing_to_the_reset_info_and_changes_no_start_row() -> None:
    plain = RecoveryEnv(_pinned(v0_config()))
    explicit = RecoveryEnv(_pinned(_config(0)))
    for seed in range(5):
        obs_a, info_a = plain.reset(seed=seed)
        obs_b, info_b = explicit.reset(seed=seed)
        assert not any(key in info_a for key in RESET_HISTORY_INFO_KEYS + ("history_rows",))
        assert info_a.keys() == info_b.keys()
        assert np.array_equal(obs_a, obs_b) and info_a["trace_start_row"] == info_b["trace_start_row"]


@needs_holdout
def test_history_ends_at_the_row_before_the_start_row_and_the_reset_observation_is_the_start_row() -> None:
    trace = _trace()
    env = RecoveryEnv(_pinned(_config(H)))
    for seed in range(20):
        obs, info = env.reset(seed=seed)
        start = info["trace_start_row"]
        assert start >= H
        fg, read, write = (info[key] for key in RESET_HISTORY_INFO_KEYS)
        assert fg.shape == read.shape == write.shape == (H,) and fg.dtype == np.float64
        assert fg[-1] == trace.fg_bw_mbps[start - 1] and fg[0] == trace.fg_bw_mbps[start - H]
        assert read[-1] == trace.fg_read_iops[start - 1] and write[-1] == trace.fg_write_iops[start - 1]
        assert np.array_equal(fg, trace.fg_bw_mbps[start - H : start])
        # the reset observation carries the start row itself, not the last history row
        assert float(obs[2]) * BW_TOTAL_MBPS == pytest.approx(trace.fg_bw_mbps[start], rel=1e-6)
        assert info["history_rows"] == H


@needs_holdout
def test_start_rows_below_the_history_length_are_mapped_and_all_others_are_unchanged() -> None:
    n_rows = len(_trace().fg_bw_mbps)
    plain, hist = RecoveryEnv(_pinned(_config(0))), RecoveryEnv(_pinned(_config(H)))
    remapped = 0
    for seed in range(20):
        obs_p, info_p = plain.reset(seed=seed)
        obs_h, info_h = hist.reset(seed=seed)
        drawn = info_p["trace_start_row"]
        assert info_h["trace_start_row_drawn"] == drawn
        if drawn >= H:
            assert info_h["trace_start_row"] == drawn and not info_h["trace_start_row_remapped"]
            assert np.array_equal(obs_p, obs_h)
        else:
            remapped += 1
            assert info_h["trace_start_row"] == H + (drawn * (n_rows - H)) // H
            assert info_h["trace_start_row_remapped"]
        assert H <= info_h["trace_start_row"] < n_rows
        # fleet, hazard and failure streams are the same whether or not the start row moved
        assert obs_p[5] == obs_h[5] and obs_p[6] == obs_h[6]
    assert remapped >= 1


def test_history_longer_than_the_trace_is_rejected_at_reset() -> None:
    env = RecoveryEnv(v0_config(workload_profile="idle", history_rows=10**6))
    with pytest.raises(ValueError):
        env.reset(seed=0)


class _Recorder:
    """A NeedsHistory policy that records what the harness hands it and requests 120 MB/s."""

    deployable = False
    name = "history_recorder"

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.order: list[str] = []

    def reset(self) -> None:
        self.order.append("reset")

    def set_history(self, fg_bw: np.ndarray, read_iops: np.ndarray, write_iops: np.ndarray) -> None:
        self.order.append("set_history")
        self.calls.append((fg_bw, read_iops, write_iops))

    def __call__(self, obs: np.ndarray) -> int:
        self.order.append("call")
        return 3


@needs_holdout
def test_the_harness_passes_the_history_after_reset_and_before_the_first_call() -> None:
    recorder = _Recorder()
    assert isinstance(recorder, NeedsHistory)
    frame = evaluate(recorder, profile=HOLDOUT, seeds=[0, 1], config=_config(H))
    assert recorder.order[:3] == ["reset", "set_history", "call"]
    assert recorder.order.count("set_history") == 2 and len(recorder.calls) == 2
    trace = _trace()
    for (fg, _, _), start in zip(recorder.calls, frame["trace_start_row"]):
        assert np.array_equal(fg, trace.fg_bw_mbps[start - H : start])


@needs_holdout
def test_a_history_policy_without_history_rows_is_rejected() -> None:
    with pytest.raises(ValueError, match="history_rows"):
        evaluate(_Recorder(), profile=HOLDOUT, seeds=[0], config=_config(0))


@needs_holdout
def test_a_policy_without_set_history_is_unchanged_on_seeds_whose_start_row_is_unchanged() -> None:
    policy: Policy = FixedRate(50)
    off = evaluate(policy, profile=HOLDOUT, seeds=list(range(8)), config=_config(0)).set_index("seed")
    on = evaluate(policy, profile=HOLDOUT, seeds=list(range(8)), config=_config(H)).set_index("seed")
    same = off.index[off["trace_start_row"] >= H]
    assert len(same) >= 1
    for column in ("total_reward", "steps", "completion_hours", "latency_violation_fraction", "trace_start_row"):
        assert off.loc[same, column].tolist() == on.loc[same, column].tolist()


@needs_v1
@pytest.mark.skipif(not EVIDENCE.exists(), reason="committed baselines evidence absent")
def test_flag_off_reproduces_committed_evidence_rows_exactly() -> None:
    committed = pd.read_csv(EVIDENCE)
    config = v1_eval_config(hazard_error_sigma=0.25)
    for name in ("fixed_50", "fixed_120"):
        rows = committed[committed["policy"] == name].set_index("seed")
        frame = evaluate(FixedRate(int(name.split("_")[1])), profile=LONG, seeds=[0, 1, 2], config=config).set_index("seed")
        for column in ("total_reward", "steps", "latency_violation_fraction", "p_failure_mean", "trace_start_row",
                       "mean_delivered_mbps"):
            assert frame[column].tolist() == pytest.approx(rows.loc[frame.index, column].tolist(), rel=1e-12), (name, column)
