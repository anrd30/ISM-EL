"""Predictive controller: forecasters, selection, residual belief, history warm-up, harness hand-over."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from agent.predictive_controller import (
    DAY_STEPS,
    FALLBACK_OFFSETS_MBPS,
    MIN_RESIDUALS,
    RESIDUAL_WINDOW,
    PredictiveController,
    PredictiveControllerNoHistory,
    RuleWithHistory,
)
from baselines.evaluate import evaluate
from baselines.policies import NeedsHistory, Policy, RiskAwareRuleBased
from sim.config import v0_config


@dataclass
class _Obs:
    fg_bw: float


def _series(n: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return 60.0 + 25.0 * np.sin(2.0 * np.pi * t / DAY_STEPS) + rng.normal(0.0, 4.0, n)


def _controller(mode: str = "adaptive", **kwargs) -> PredictiveController:
    policy = PredictiveController(v0_config(), mode=mode, **kwargs)  # type: ignore[arg-type]
    policy.reset()
    return policy


def _feed(policy: PredictiveController, values) -> tuple[np.ndarray, np.ndarray]:
    out = (np.array([]), np.array([]))
    for v in values:
        out = policy._belief(_Obs(float(v)))  # type: ignore[arg-type]
    return out


def test_protocols_and_identity() -> None:
    policy = _controller()
    assert isinstance(policy, Policy) and isinstance(policy, NeedsHistory) and isinstance(policy, RiskAwareRuleBased)
    assert policy.deployable is True
    assert isinstance(RuleWithHistory(v0_config()), NeedsHistory)
    with pytest.raises(ValueError):
        PredictiveController(v0_config(), mode="unknown")  # type: ignore[arg-type]


def test_early_belief_is_the_forecast_plus_the_fixed_offsets() -> None:
    samples, weights = _feed(_controller(), [50.0])
    assert samples == pytest.approx(50.0 + np.array(FALLBACK_OFFSETS_MBPS))
    assert weights == pytest.approx(np.full(len(FALLBACK_OFFSETS_MBPS), 1.0 / len(FALLBACK_OFFSETS_MBPS)))


def test_persistence_forecast_is_the_last_value_and_residuals_are_its_errors() -> None:
    policy = _controller("persist")
    _feed(policy, [40.0, 44.0, 41.0, 47.0, 45.0])
    assert policy._pending["persist"] == 45.0
    assert list(policy._residuals["persist"]) == [4.0, -3.0, 6.0, -2.0]


def test_seasonal_forecast_is_yesterday_at_the_target_time_plus_the_level_change() -> None:
    series = _series(3 * DAY_STEPS)
    policy = _controller("seasonal")
    _feed(policy, series[: 2 * DAY_STEPS])
    h = [float(x) for x in series[: 2 * DAY_STEPS]]
    expected = h[len(h) - DAY_STEPS] + (h[-1] - h[-1 - DAY_STEPS])
    assert policy._pending["seasonal"] == pytest.approx(expected)


def test_ar_forecast_recovers_an_exact_ar_process() -> None:
    n = 600
    x = np.zeros(n)
    x[:3] = [50.0, 51.0, 49.0]
    for i in range(3, n):
        x[i] = 0.5 * x[i - 1] + 0.3 * x[i - 2] + 0.1 * x[i - 3] + 6.0
    policy = _controller("ar")
    _feed(policy, x[:-1])
    assert policy._pending["ar"] == pytest.approx(x[-1], rel=1e-6)


def test_adaptive_selection_follows_the_lowest_recent_squared_error() -> None:
    policy = _controller()
    _feed(policy, _series(2 * DAY_STEPS + 60))
    means = {k: float(np.mean(policy._squared[k])) for k in policy._pending if len(policy._squared[k]) >= 12}
    assert policy._chosen() == min(means, key=means.get)  # type: ignore[arg-type]


def test_belief_is_the_chosen_forecast_plus_its_last_residuals_clipped_at_zero() -> None:
    policy = _controller()
    samples, weights = _feed(policy, _series(DAY_STEPS + 100))
    chosen = policy._chosen()
    pool = np.fromiter(policy._residuals[chosen], dtype=float)
    assert MIN_RESIDUALS <= len(pool) <= RESIDUAL_WINDOW
    assert samples == pytest.approx(np.maximum(0.0, policy._pending[chosen] + pool))
    assert weights.sum() == pytest.approx(1.0) and np.allclose(weights, weights[0])


def test_history_fills_the_residual_pools() -> None:
    history = _series(576)
    with_history = _controller()
    with_history.set_history(history, history, history)
    assert len(with_history._residuals["persist"]) >= MIN_RESIDUALS
    assert len(with_history._h) == 576
    assert len(_controller()._residuals["persist"]) == 0


def test_history_replay_is_deterministic() -> None:
    history = _series(576)
    follow = _series(700)[576:]
    runs = []
    for _ in range(2):
        policy = _controller()
        policy.set_history(history, history, history)
        runs.append([policy._belief(_Obs(float(v)))[0] for v in follow])  # type: ignore[arg-type]
    assert all(np.array_equal(a, b) for a, b in zip(*runs))


def test_the_no_history_arm_opts_out_of_the_history_protocol() -> None:
    policy = PredictiveControllerNoHistory(v0_config())
    assert not isinstance(policy, NeedsHistory) and isinstance(policy, PredictiveController)
    assert isinstance(_controller(), NeedsHistory)


def test_a_short_history_is_stored_without_replay() -> None:
    policy = _controller()
    policy.set_history(np.arange(100.0), np.zeros(100), np.zeros(100))
    assert len(policy._h) == 100 and len(policy._residuals["persist"]) == 0


def test_rule_with_history_prefills_the_rules_buffers() -> None:
    policy = RuleWithHistory(v0_config())
    policy.reset()
    assert len(policy._recent) == 0
    policy.set_history(np.full(30, 55.0), np.zeros(30), np.zeros(30))
    assert len(policy._recent) > 0


def test_the_controller_runs_through_the_harness_with_and_without_history() -> None:
    holdout = Path(__file__).resolve().parents[1] / "data_pipeline" / "profiles" / "profile_diurnal_long_holdout.csv"
    if not holdout.exists():
        pytest.skip("holdout profile absent")
    off = evaluate(PredictiveControllerNoHistory(v0_config()), profile=holdout, seeds=[0, 1], config=v0_config())
    assert len(off) == 2 and off["completed"].all()
    with pytest.raises(ValueError, match="history_rows"):
        evaluate(PredictiveController(v0_config()), profile=holdout, seeds=[0], config=v0_config())
    on_config = v0_config(history_rows=576)
    on = evaluate(PredictiveController(on_config), profile=holdout, seeds=[0, 1], config=on_config)
    assert len(on) == 2 and on["completed"].all()
    again = evaluate(PredictiveController(on_config), profile=holdout, seeds=[0, 1], config=on_config)
    assert on["total_reward"].tolist() == again["total_reward"].tolist()
    ignored = evaluate(PredictiveControllerNoHistory(on_config), profile=holdout, seeds=[0, 1], config=on_config)
    assert ignored["trace_start_row"].tolist() == on["trace_start_row"].tolist()
