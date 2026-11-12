"""Log-barrier latency cost: bounded, sharp near the boundary, and the default path unchanged."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from agent.ppo import BARRIER_CAP, BARRIER_DELTA, PPOAgent, PPOConfig, latency_barrier
from agent.rollout import RolloutBatch, RolloutCollector
from sim.config import v0_config

TARGET = 15.0
LIMIT = 1.5 * TARGET

# Captured from RolloutCollector(v0_config(workload_profile="bursty"), seed=0).collect(PPOAgent(seed=0), 600)
# before RolloutBatch gained latency_now.
ROLLOUT_REFERENCE = {
    "obs_sum": "0x1.662f1510868c0p+11",
    "next_obs_sum": "0x1.667434675c4c0p+11",
    "log_probs_sum": "-0x1.9fd785a300000p+9",
    "values_sum": "0x1.6a1ea92200000p+7",
    "rewards_sum": "-0x1.9b4b07246fefdp+9",
    "latency_violated_sum": "0x1.9000000000000p+5",
    "exposure_sum": "0x1.de1191c126600p+0",
    "actions_sum": "920",
    "terminated_sum": "1",
    "truncated_sum": "0",
    "infeasible_sum": "0",
    "n_episodes": "1",
    "episode_returns": "-0x1.4176691d13b70p+9",
}


def _rollout_summary() -> dict[str, str]:
    torch.set_num_threads(1)
    agent = PPOAgent(seed=0)
    b = RolloutCollector(v0_config(workload_profile="bursty"), seed=0).collect(agent, 600)
    out: dict[str, str] = {}
    for name in ("obs", "next_obs", "log_probs", "values", "rewards", "latency_violated", "exposure"):
        out[f"{name}_sum"] = float(getattr(b, name).astype(np.float64).sum()).hex()
    for name in ("actions", "terminated", "truncated", "infeasible"):
        out[f"{name}_sum"] = str(int(getattr(b, name).sum()))
    out["n_episodes"] = str(len(b.episodes))
    out["episode_returns"] = ",".join(float(e.episode_return).hex() for e in b.episodes)
    return out


def test_default_rollout_matches_pre_edit_reference() -> None:
    """Existing batch fields are unchanged by the added latency_now; floats compared at rel 1e-6."""
    actual = _rollout_summary()
    assert actual.keys() == ROLLOUT_REFERENCE.keys()
    for key, expected in ROLLOUT_REFERENCE.items():
        if key.endswith("_sum") and "0x" in expected:
            assert math.isclose(float.fromhex(actual[key]), float.fromhex(expected), rel_tol=1e-6, abs_tol=1e-9), key
        elif key == "episode_returns":
            a, e = actual[key].split(","), expected.split(",")
            assert len(a) == len(e)
            assert all(math.isclose(float.fromhex(x), float.fromhex(y), rel_tol=1e-6) for x, y in zip(a, e))
        else:
            assert actual[key] == expected, key


def test_rollout_latency_now_is_consistent_with_violation_flag() -> None:
    batch = RolloutCollector(v0_config(workload_profile="bursty"), seed=0).collect(PPOAgent(seed=0), 300)
    assert batch.latency_now is not None and batch.latency_now.shape == (300,)
    assert np.array_equal(batch.latency_now > LIMIT, batch.latency_violated.astype(bool))


def test_barrier_is_zero_at_zero_latency_and_small_far_from_the_limit() -> None:
    cost = latency_barrier(np.array([0.0, 2.0, 4.0]), TARGET)
    assert cost[0] == 0.0
    assert np.all(cost < 0.25)


def test_barrier_grows_sharply_toward_the_boundary() -> None:
    latencies = np.array([2.0, 4.0, 10.0, 15.0, 20.0, 22.0, 22.4])
    cost = latency_barrier(latencies, TARGET)
    assert np.all(np.diff(cost) > 0)
    assert cost[-1] > 15 * cost[1]


def test_barrier_at_and_past_the_boundary_is_finite_bounded_and_monotone() -> None:
    at = latency_barrier(np.array([LIMIT]), TARGET)[0]
    assert at == pytest.approx(-math.log(BARRIER_DELTA) + 1.0)
    past = latency_barrier(np.array([23.0, 25.0, 30.0, 100.0, 1e9]), TARGET)
    assert np.all(np.isfinite(past))
    assert np.all(past <= BARRIER_CAP)
    assert np.all(np.diff(np.concatenate([[at], past])) >= 0)
    assert past[-1] == BARRIER_CAP


def test_barrier_is_continuous_and_differentiable_at_the_switch() -> None:
    switch = LIMIT * (1.0 - BARRIER_DELTA)
    eps = 1e-6
    lo, mid, hi = latency_barrier(np.array([switch - eps, switch, switch + eps]), TARGET)
    assert mid == pytest.approx(-math.log(BARRIER_DELTA))
    assert lo == pytest.approx(mid, abs=1e-5)
    slope_left = (mid - latency_barrier(np.array([switch - eps]), TARGET)[0]) / eps
    slope_right = (hi - mid) / eps
    assert slope_left == pytest.approx(slope_right, rel=1e-3)


def _batch(latency_now: np.ndarray | None, infeasible: np.ndarray | None = None) -> RolloutBatch:
    n = 6
    return RolloutBatch(
        obs=np.zeros((n, 12), dtype=np.float32),
        next_obs=np.zeros((n, 12), dtype=np.float32),
        actions=np.zeros(n, dtype=np.int64),
        log_probs=np.zeros(n, dtype=np.float32),
        values=np.zeros(n, dtype=np.float32),
        rewards=np.arange(n, dtype=np.float64),
        terminated=np.zeros(n, dtype=bool),
        truncated=np.zeros(n, dtype=bool),
        latency_violated=np.array([0, 0, 0, 1, 1, 1], dtype=np.float32),
        exposure=np.zeros(n, dtype=np.float32),
        infeasible=infeasible,
        latency_now=latency_now,
    )


def test_barrier_replaces_the_latency_lagrangian_charge() -> None:
    lat = np.array([1.0, 5.0, 12.0, 21.0, 23.0, 40.0])
    batch = _batch(lat)
    agent = PPOAgent(PPOConfig(), seed=0, barrier_cost=True, barrier_coef=0.1, barrier_latency_target_ms=TARGET)
    agent.lambda_latency.value = 1.5
    expected = 0.1 * batch.rewards - 0.1 * latency_barrier(lat, TARGET)
    assert np.allclose(agent.penalised_rewards(batch), expected, rtol=0, atol=1e-12)
    agent.lambda_latency.value = 0.0
    assert np.allclose(agent.penalised_rewards(batch), expected, rtol=0, atol=1e-12)


def test_barrier_masks_infeasible_episode_steps() -> None:
    lat = np.full(6, 30.0)
    mask = np.array([True, True, False, False, False, False])
    agent = PPOAgent(PPOConfig(), seed=0, barrier_cost=True, barrier_coef=0.1, barrier_latency_target_ms=TARGET)
    pen = agent.penalised_rewards(_batch(lat, mask))
    assert pen[0] == 0.0 and pen[1] == 0.1 * 1.0
    assert pen[2] < 0.1 * 2.0


def test_barrier_flag_guards() -> None:
    with pytest.raises(ValueError):
        PPOAgent(barrier_cost=True)
    with pytest.raises(ValueError):
        PPOAgent(barrier_cost=True, marginal_violation_cost=True, barrier_latency_target_ms=TARGET)
    agent = PPOAgent(barrier_cost=True, barrier_latency_target_ms=TARGET)
    with pytest.raises(ValueError):
        agent.penalised_rewards(_batch(None))


def test_default_agent_ignores_latency_now() -> None:
    plain = PPOAgent(PPOConfig(), seed=0)
    lat = np.linspace(0.0, 40.0, 6)
    assert np.array_equal(plain.penalised_rewards(_batch(lat)), plain.penalised_rewards(_batch(None)))


def test_barrier_rejects_non_finite_latency_now() -> None:
    agent = PPOAgent(barrier_cost=True, barrier_latency_target_ms=TARGET)
    with pytest.raises(ValueError):
        agent.penalised_rewards(_batch(np.full(6, np.nan)))
