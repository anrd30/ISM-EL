"""Marginal-violation cost: default path unchanged, counterfactual leaves the real step untouched."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from agent.min_action_counterfactual import MinActionCounterfactual, min_action_violation
from agent.ppo import PPOAgent, PPOConfig, ppo_loss
from agent.rollout import RolloutBatch
from sim.config import v0_config
from sim.recovery_env import RecoveryEnv

# float.hex() values of the computation in _reference_run, captured before the opt-in was added.
REFERENCE = {
    "latency_cost_sum": "0x1.8c00000000000p+6",
    "risk_cost_sum": "0x1.d8553e60c0000p+6",
    "penalised_sum": "-0x1.d1e5b7c18c6c6p-2",
    "update_approx_kl": "0x1.fe07ad4000000p-3",
    "update_clip_fraction": "0x1.b500000000000p-1",
    "update_entropy": "0x1.62de304000000p+0",
    "update_policy_loss": "0x1.0c1aba4000000p-3",
    "update_value_loss": "0x1.64b8004000000p-3",
    "multipliers0_latency_violation_fraction": "0x1.a585969e9a586p-3",
    "multipliers0_lambda_latency": "0x1.3f1f303833f20p-4",
    "multipliers0_lambda_risk": "0x0.0p+0",
    "multipliers1_latency_violation_fraction": "0x1.a585969e9a586p-3",
    "multipliers1_lambda_latency": "0x1.3f1f303833f20p-3",
    "multipliers1_lambda_risk": "0x0.0p+0",
    "ppo_loss_total": "-0x1.baa1b20000000p-3",
}


def _batch(n: int = 512) -> RolloutBatch:
    rng = np.random.default_rng(1234)
    return RolloutBatch(
        obs=rng.random((n, 12), dtype=np.float32),
        next_obs=rng.random((n, 12), dtype=np.float32),
        actions=rng.integers(0, 4, n).astype(np.int64),
        log_probs=np.log(rng.uniform(0.1, 0.9, n)).astype(np.float32),
        values=rng.normal(size=n).astype(np.float32),
        rewards=rng.normal(size=n),
        terminated=(rng.random(n) < 0.01),
        truncated=np.zeros(n, dtype=bool),
        latency_violated=(rng.random(n) < 0.2).astype(np.float32),
        exposure=rng.uniform(0, 0.01, n).astype(np.float32),
        infeasible=(rng.random(n) < 0.05),
    )


def _reference_run() -> dict[str, str]:
    torch.set_num_threads(1)
    out: dict[str, str] = {}
    agent = PPOAgent(PPOConfig(), seed=0)
    batch = _batch()
    lat, risk = agent.step_costs(batch)
    out["latency_cost_sum"] = float(lat.sum()).hex()
    out["risk_cost_sum"] = float(risk.sum()).hex()
    out["penalised_sum"] = float(agent.penalised_rewards(batch).sum()).hex()
    stats = agent.update(batch, progress=0.0)
    for k in sorted(stats):
        out[f"update_{k}"] = float(stats[k]).hex()
    for i in range(2):
        m = agent.update_multipliers(batch.episodes, batch)
        for k in ("latency_violation_fraction", "lambda_latency", "lambda_risk"):
            out[f"multipliers{i}_{k}"] = float(m[k]).hex()
    lp = torch.log(torch.tensor([0.3, 0.5, 0.2]))
    t = ppo_loss(lp, lp + 0.1, torch.tensor([1.0, -0.5, 0.2]), torch.tensor([0.1, 0.2, 0.3]),
                 torch.tensor([0.2, 0.1, 0.4]), torch.tensor([1.0, 1.1, 0.9]), 0.2, 0.5, 0.01)
    out["ppo_loss_total"] = float(t.total).hex()
    return out


def test_default_path_reproduces_pre_edit_reference() -> None:
    """Relative tolerance 1e-6: float32 reductions differ by single ULPs across platforms."""
    actual = _reference_run()
    assert actual.keys() == REFERENCE.keys()
    for key, expected in REFERENCE.items():
        assert math.isclose(float.fromhex(actual[key]), float.fromhex(expected), rel_tol=1e-6, abs_tol=0.0), key


def test_flag_requires_per_step_counterfactual_of_batch_length() -> None:
    agent = PPOAgent(PPOConfig(), seed=0, marginal_violation_cost=True)
    batch = _batch()
    with pytest.raises(ValueError):
        agent.step_costs(batch)
    agent.set_min_action_violated(np.zeros(len(batch) - 1))
    with pytest.raises(ValueError):
        agent.update_multipliers(batch.episodes, batch)


def test_marginal_cost_equals_raw_violation_when_min_action_does_not_violate() -> None:
    batch = _batch()
    raw = PPOAgent(PPOConfig(), seed=0)
    marginal = PPOAgent(PPOConfig(), seed=0, marginal_violation_cost=True)
    marginal.set_min_action_violated(np.zeros(len(batch)))
    assert np.array_equal(raw.step_costs(batch)[0], marginal.step_costs(batch)[0])
    a, b = raw.update_multipliers(batch.episodes, batch), marginal.update_multipliers(batch.episodes, batch)
    for key in ("latency_violation_fraction", "lambda_latency", "lambda_risk"):
        assert a[key] == b[key]


def test_marginal_cost_is_zero_on_exogenous_intervals_and_never_negative() -> None:
    batch = _batch()
    agent = PPOAgent(PPOConfig(mask_infeasible_cost=False), seed=0, marginal_violation_cost=True)
    agent.set_min_action_violated(np.ones(len(batch)))
    assert not agent.step_costs(batch)[0].any()
    batch.latency_violated[:] = 0.0
    assert not agent.step_costs(batch)[0].any()


def _config():
    return v0_config(workload_profile="peak", fleet_degradation=1.0)


def _run(env, actions: list[int], seed: int):
    env.reset(seed=seed)
    trace = []
    for a in actions:
        obs, reward, term, trunc, info = env.step(a)
        base = env._base if isinstance(env, MinActionCounterfactual) else env
        trace.append((obs.tobytes(), reward, term, trunc, tuple(sorted((k, repr(v)) for k, v in info.items())),
                      float(base._hazard).hex(), float(base._work).hex()))
        if term or trunc:
            break
    return trace


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_counterfactual_call_leaves_real_step_state_bit_identical(seed: int) -> None:
    rng = np.random.default_rng(seed)
    actions = rng.integers(0, 4, 300).tolist()
    plain = _run(RecoveryEnv(_config()), actions, seed)
    wrapped = _run(MinActionCounterfactual(RecoveryEnv(_config()), _config()), actions, seed)
    assert plain == wrapped


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_counterfactual_equals_real_violation_when_the_request_is_the_minimum_action(seed: int) -> None:
    env = MinActionCounterfactual(RecoveryEnv(_config()), _config())
    env.reset(seed=seed)
    for _ in range(300):
        _, _, term, trunc, info = env.step(0)
        if term or trunc:
            break
    records = env.drain()
    assert len(records["real_violated"]) > 0
    assert np.array_equal(records["min_action_violated"], records["real_violated"])


def test_min_action_violation_matches_filter_floor() -> None:
    cfg = _config()
    violated, floor = min_action_violation(0.0, 1000.0, 0.0, cfg)
    assert (violated, floor) == (False, 0)
    _, floor_high = min_action_violation(0.05, 3000.0, 0.0, cfg)
    assert floor_high > 0
    assert min_action_violation(0.0, 1000.0, 190.0, cfg)[0] is True


def test_wrapper_rejects_latency_filter_configs() -> None:
    cfg = v0_config(latency_filter=True)
    with pytest.raises(ValueError):
        MinActionCounterfactual(RecoveryEnv(cfg), cfg)
