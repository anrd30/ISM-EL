"""Exact hindsight optimum: planned value against executed reward, and its relation to the other policies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from baselines.dp_optimum import DPHindsightOptimum, DPSolution, solve_hindsight
from baselines.evaluate import _pinned_config, evaluate
from baselines.policies import NeedsEnv, OracleLookahead, RiskAwareRuleBased
from contracts.contracts import REBUILD_WORK_GB
from sim.config import v0_config, v1_eval_config
from sim.recovery_env import RecoveryEnv

REPO = Path(__file__).resolve().parents[1]
HOLDOUT = REPO / "data_pipeline" / "profiles" / "profile_diurnal_long_holdout.csv"
needs_holdout = pytest.mark.skipif(not HOLDOUT.exists(), reason=f"{HOLDOUT.name} absent")
needs_v1 = pytest.mark.skipif(not (REPO / "hazard" / "artifacts" / "v1").exists(), reason="hazard/artifacts/v1 absent")


def _execute(config, seed: int) -> tuple[float, float, int, int, str]:
    """Run the optimum on one scenario: (planned cost, executed cost, survivors lost, steps, termination)."""
    env = RecoveryEnv(config)
    policy = DPHindsightOptimum(config)
    policy.bind(env)
    obs, _ = env.reset(seed=seed)
    policy.reset()
    start = None
    total = 0.0
    steps = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        total += reward
        steps += 1
        if start is None:
            start = info["n_survivors"]
        if terminated or truncated:
            break
    assert policy.planned_cost is not None and start is not None
    return policy.planned_cost, -total, start - info["n_survivors"], steps, info["termination_reason"]


def test_policy_identity_and_protocol() -> None:
    policy = DPHindsightOptimum(v0_config())
    assert policy.name == "DP_hindsight_optimum (not deployable)"
    assert policy.deployable is False
    assert isinstance(policy, NeedsEnv)
    with pytest.raises(RuntimeError):
        policy(np.zeros(12, dtype=np.float32))


def test_solution_value_is_zero_without_work_and_increases_with_work() -> None:
    config = v0_config()
    solution = solve_hindsight([40.0] * 300, 1.0e-4, config, grid_gb=10.0, work_max_gb=2000.0)
    assert isinstance(solution, DPSolution)
    assert solution.value[0] == 0.0
    assert (np.diff(solution.value) >= -1e-9).all()
    assert solution.planned_cost(0.0) == 0.0
    assert solution.planned_cost(1000.0) > 0.0
    assert solution.action.shape == (300, 201)


def test_planned_cost_equals_executed_cost_on_episodes_without_failures_under_the_v0_config() -> None:
    config, _ = _pinned_config(v0_config(), "diurnal")
    checked = 0
    for seed in range(7):
        planned, executed, lost, steps, reason = _execute(config, seed)
        if lost == 0 and reason == "recovery_complete":
            assert executed == pytest.approx(planned, abs=1e-6), seed
            checked += 1
    assert checked == 7


def test_planned_cost_differs_by_the_grid_effect_where_a_risk_floor_boundary_falls_between_grid_points() -> None:
    """v0 seed 7 has a high hazard; the executed cost is below the interpolated plan by less than 1."""
    config, _ = _pinned_config(v0_config(), "diurnal")
    planned, executed, lost, steps, reason = _execute(config, 7)
    assert lost == 0 and reason == "recovery_complete"
    assert 0.0 < planned - executed < 1.0


def test_optimum_is_not_worse_than_the_rule_or_the_look_ahead_on_failure_free_episodes_under_the_v0_config() -> None:
    config, _ = _pinned_config(v0_config(), "bursty")
    checked = 0
    for seed in range(6):
        planned, executed, lost, steps, reason = _execute(config, seed)
        if lost:
            continue
        for policy in (RiskAwareRuleBased(config), OracleLookahead(config)):
            row = evaluate(policy, profile="bursty", seeds=[seed], config=v0_config()).iloc[0]
            assert -planned >= row["total_reward"] - 1e-6, (seed, policy.name)
        checked += 1
    assert checked >= 1


@needs_holdout
@needs_v1
@pytest.mark.parametrize("seed, cost", [(0, 44.23), (2, 47.04)])
def test_planned_cost_equals_executed_cost_on_the_holdout(seed: int, cost: float) -> None:
    config, _ = _pinned_config(v1_eval_config(hazard_error_sigma=0.25), HOLDOUT)
    planned, executed, lost, steps, reason = _execute(config, seed)
    assert lost == 0 and reason == "recovery_complete"
    assert executed == pytest.approx(planned, abs=1e-6)
    assert executed == pytest.approx(cost, abs=5e-3)


def test_work_grid_covers_the_rebuild_work() -> None:
    solution = solve_hindsight([20.0] * 5, 1.0e-4, v0_config(), grid_gb=500.0)
    assert (len(solution.value) - 1) * 500.0 == REBUILD_WORK_GB
