"""Optional exposure-weighted objective, the unforced infeasible-budget option and the hazard-weighted policies."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from baselines.dp_optimum import DPHindsightOptimum, solve_hindsight
from baselines.evaluate import _pinned_config
from baselines.policies import DEFAULT_PROGRESS_VALUE, FixedRate, RiskAwareRuleBased
from baselines.risk_objective import HazardWeightedRule, calibrate_hazard_ref
from contracts.contracts import ACTIONS_MBPS, Observation
from sim.config import RiskObjective, SimConfig, v0_config
from sim.recovery_env import RecoveryEnv


def _pinned(**fields):
    return _pinned_config(v0_config(**fields), "diurnal")[0]


def _run(config: SimConfig, seed: int, action: int = 1) -> tuple[float, int, int, float]:
    """Total reward, steps, survivors lost and the reset-time true hazard for a constant request."""
    env = RecoveryEnv(config)
    _, info = env.reset(seed=seed)
    hazard0 = float(info["hazard_true_per_hour"])
    survivors0 = None
    total, steps = 0.0, 0
    while True:
        _, reward, term, trunc, info = env.step(action)
        total += reward
        steps += 1
        survivors0 = survivors0 if survivors0 is not None else info["n_survivors"]
        if term or trunc:
            break
    return total, steps, survivors0 - info["n_survivors"], hazard0


def test_defaults_are_the_contract_reward_and_the_forced_filter() -> None:
    config = v0_config()
    assert config.risk_objective is None and config.infeasible_budget_max_rate is True


@pytest.mark.parametrize("bad", [0.0, -1e-5, float("nan"), float("inf"), True, "1e-4"])
def test_risk_objective_needs_a_positive_finite_reference(bad: object) -> None:
    with pytest.raises(ValueError):
        RiskObjective(bad)  # type: ignore[arg-type]


def test_weight_one_leaves_the_reward_unchanged_and_other_weights_scale_the_degraded_term() -> None:
    base = _pinned()
    total, steps, lost, hazard0 = _run(base, seed=0)
    assert lost == 0
    same = dataclasses.replace(base, risk_objective=RiskObjective(hazard0))
    assert _run(same, 0)[0] == pytest.approx(total, abs=1e-9)
    halved = dataclasses.replace(base, risk_objective=RiskObjective(2.0 * hazard0))
    new_total = _run(halved, 0)[0]
    assert new_total - total == pytest.approx(0.5 * base.reward.beta * steps, rel=1e-9)


def test_calibrated_reference_is_the_mean_initial_true_hazard() -> None:
    config = _pinned()
    seeds = list(range(5))
    env = RecoveryEnv(config)
    expected = float(np.mean([env.reset(seed=s)[1]["hazard_true_per_hour"] for s in seeds]))
    assert calibrate_hazard_ref(config, seeds) == pytest.approx(expected)
    assert calibrate_hazard_ref(dataclasses.replace(config, risk_objective=RiskObjective(1.0)), seeds) == pytest.approx(expected)


def _infeasible_step(config: SimConfig, requested: int) -> dict:
    env = RecoveryEnv(config)
    env.reset(seed=0)
    env._hazard = 1.0  # a group hazard at which no action meets the risk budget
    return env.step(requested)[4]


def test_infeasible_budget_forces_the_largest_action_by_default_and_applies_the_request_when_unforced() -> None:
    forced = _infeasible_step(_pinned(), 0)
    assert forced["alloc_applied_mbps"] == ACTIONS_MBPS[-1] and forced["violated_risk_budget"]
    free = _infeasible_step(_pinned(infeasible_budget_max_rate=False), 0)
    assert free["alloc_applied_mbps"] == ACTIONS_MBPS[0] and free["violated_risk_budget"] and not free["alloc_clipped_by_filter"]
    mid = _infeasible_step(_pinned(infeasible_budget_max_rate=False), 2)
    assert mid["alloc_applied_mbps"] == ACTIONS_MBPS[2]


def test_a_feasible_budget_is_filtered_the_same_in_both_modes() -> None:
    for mode in (True, False):
        env = RecoveryEnv(_pinned(infeasible_budget_max_rate=mode))
        env.reset(seed=0)
        info = env.step(0)[4]
        assert info["alloc_applied_mbps"] == ACTIONS_MBPS[0] and not info["violated_risk_budget"]


def _obs(hazard: float, work: float = 12000.0, fg: float = 60.0, uncertainty: float = 0.0) -> np.ndarray:
    o = Observation(fg_read_iops=1000.0, fg_write_iops=1000.0, fg_bw=fg, latency_now=3.0, latency_target=15.0,
                    hazard=hazard, hazard_uncertainty=uncertainty, redundancy_left=1.0, progress=0.0, work_remaining=work,
                    alloc_current=0.0, bw_available=130.0)
    return o.to_vector(normalized=True)


def test_hazard_weighting_scales_the_progress_value_and_equals_the_rule_at_weight_one() -> None:
    config = v0_config()
    ref = 1.0e-4
    weighted = HazardWeightedRule(config, ref)
    rule = RiskAwareRuleBased(config)
    for fg in (20.0, 60.0, 100.0):
        obs = _obs(ref, fg=fg)
        assert weighted(obs) == rule(obs)
        assert weighted.progress_value == pytest.approx(DEFAULT_PROGRESS_VALUE)
    weighted(_obs(1.5e-4))
    assert weighted.progress_value == pytest.approx(DEFAULT_PROGRESS_VALUE * 1.5)
    weighted(_obs(0.5e-4))
    assert weighted.progress_value == pytest.approx(DEFAULT_PROGRESS_VALUE * 0.5)


def test_unforced_policy_has_floor_zero_only_when_no_action_meets_the_budget() -> None:
    """The hazard field saturates at 1.8e-4 per hour, at which the largest action is still feasible; an infeasible
    inflated hazard needs the uncertainty term (``k`` = 1 with an uncertainty of 1.8e-4)."""
    config = v0_config()
    forced = HazardWeightedRule(config, 1e-4, k=1.0)
    free = HazardWeightedRule(config, 1e-4, k=1.0, unforced=True)
    infeasible = Observation.from_vector(_obs(1.8e-4, uncertainty=1.8e-4))
    feasible = Observation.from_vector(_obs(1.0e-5, uncertainty=1.0e-5))
    assert forced._floor(infeasible) == len(ACTIONS_MBPS) - 1 and free._floor(infeasible) == 0
    assert free._floor(feasible) == forced._floor(feasible)
    # with k = 0 the saturated hazard never shows an infeasible budget, so the unforced policy equals the forced one
    plain = HazardWeightedRule(config, 1e-4, unforced=True)
    assert plain._floor(Observation.from_vector(_obs(1.8e-4))) == HazardWeightedRule(config, 1e-4)._floor(
        Observation.from_vector(_obs(1.8e-4)))


def test_hazard_weighted_rule_rejects_a_bad_reference() -> None:
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError):
            HazardWeightedRule(v0_config(), bad)


def test_the_optimum_plans_the_weighted_objective_and_equals_the_executed_cost() -> None:
    config = _pinned()
    ref = 2.0 * calibrate_hazard_ref(config, [0])
    weighted = dataclasses.replace(config, risk_objective=RiskObjective(ref))
    for cfg in (config, weighted):
        env = RecoveryEnv(cfg)
        policy = DPHindsightOptimum(cfg)
        policy.bind(env)
        obs, _ = env.reset(seed=0)
        policy.reset()
        total = 0.0
        while True:
            obs, reward, term, trunc, _ = env.step(policy(obs))
            total += reward
            if term or trunc:
                break
        assert -total == pytest.approx(policy.planned_cost, abs=1e-6)
    plain = solve_hindsight([40.0] * 200, 1e-4, config, grid_gb=20.0, work_max_gb=2000.0)
    heavy = solve_hindsight([40.0] * 200, 1e-4, config, grid_gb=20.0, work_max_gb=2000.0, degraded_weight=3.0)
    assert heavy.planned_cost(1000.0) > plain.planned_cost(1000.0)


def test_the_fixed_rate_policy_is_unaffected_by_the_options() -> None:
    config = dataclasses.replace(_pinned(), risk_objective=RiskObjective(1e-4), infeasible_budget_max_rate=False)
    assert FixedRate(50)(np.zeros(12, dtype=np.float32)) == 1
    assert config.risk_objective is not None
