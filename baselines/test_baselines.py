"""Tests for the baseline policies and the paired evaluation harness.

Tests marked ``slow`` run many episodes and are skipped unless ``--runslow`` is
passed or ``BASELINES_RUN_SLOW`` is set (see ``conftest.py``). Everything here
exercises the simulator in ``sim/``; no result is a statement about a real
system.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from baselines.evaluate import (
    EPISODE_COLUMNS,
    check_pairing,
    compare,
    evaluate,
    main,
    paired_delta,
    summarise,
    to_markdown,
)
from baselines.policies import (
    BeliefConfig,
    FixedRate,
    LatencyRule,
    NeedsEnv,
    OracleLookahead,
    Policy,
    RiskAwareRuleBased,
    SlowestRiskFeasible,
    WorkloadOnlyThrottle,
    baseline_policies,
    diagnostic_policies,
    fixed_rate_policies,
    peek_next_fg_mbps,
)
from contracts.contracts import (
    ACTIONS_MBPS,
    BW_TOTAL_MBPS,
    LATENCY_VIOLATION_FRACTION_MAX,
    MAX_EPISODE_STEPS,
    N_ACTIONS,
    OBS_DIM,
    RISK_BUDGET_EPS,
    ContractViolation,
    Observation,
    TerminationReason,
    latency_degradation,
    latency_violation,
)
from sim.config import SimConfig
from sim.recovery_env import RecoveryEnv
from sim.safety_filter import exposure_probability
from workload.generate import PROFILE_NAMES, generate_profile

HEALTHY = SimConfig(fleet_degradation=0.0)


def _bound(policy: Policy, config: SimConfig | None = None) -> Policy:
    """Bind an oracle to a freshly reset environment so it can be called."""
    if isinstance(policy, NeedsEnv):
        env = RecoveryEnv(config)
        env.reset(seed=0)
        policy.bind(env)
    return policy


def _all_policies(config: SimConfig | None = None) -> list[Policy]:
    return [*baseline_policies(config), *diagnostic_policies(config)]


def _obs_sample(n: int = 300) -> np.ndarray:
    """Random, saturated and zero normalised observations, float32 like the environment's."""
    rng = np.random.default_rng(0)
    rows = [rng.random(OBS_DIM) for _ in range(n)]
    rows.append(np.zeros(OBS_DIM))
    rows.append(np.ones(OBS_DIM))
    for i in range(OBS_DIM):
        e = np.zeros(OBS_DIM)
        e[i] = 1.0
        rows.append(e)
        f = np.ones(OBS_DIM)
        f[i] = 0.0
        rows.append(f)
    # Hazard saturated with work remaining, and no work remaining with hazard.
    saturated = np.full(OBS_DIM, 0.3)
    saturated[5] = 1.0
    saturated[6] = 1.0
    saturated[9] = 1.0
    rows.append(saturated)
    done = saturated.copy()
    done[9] = 0.0
    rows.append(done)
    return np.asarray(rows, dtype=np.float32)


# --------------------------------------------------------------------------
# Policies: interface and validity
# --------------------------------------------------------------------------


def test_every_policy_satisfies_the_protocol() -> None:
    for policy in _all_policies():
        assert isinstance(policy, Policy), policy.name
        assert isinstance(policy.name, str) and policy.name


def test_policy_names_are_unique_and_oracle_is_labelled() -> None:
    names = [p.name for p in _all_policies()]
    assert len(set(names)) == len(names)
    for policy in _all_policies():
        if isinstance(policy, OracleLookahead):
            assert "ORACLE" in policy.name and "not deployable" in policy.name
            assert policy.deployable is False
        else:
            assert "ORACLE" not in policy.name
            assert policy.deployable is True


def test_four_fixed_rates_cover_the_action_set() -> None:
    fixed = fixed_rate_policies()
    assert [p.mbps for p in fixed] == list(ACTIONS_MBPS)
    assert [p(np.zeros(OBS_DIM, dtype=np.float32)) for p in fixed] == list(range(N_ACTIONS))


def test_fixed_rate_rejects_values_outside_the_action_set() -> None:
    with pytest.raises(ValueError):
        FixedRate(30)


def test_every_policy_returns_a_valid_action_for_a_wide_sample() -> None:
    sample = _obs_sample()
    for policy in _all_policies():
        _bound(policy)
        policy.reset()
        for obs in sample:
            action = policy(obs)
            assert type(action) is int, (policy.name, type(action))
            assert 0 <= action < N_ACTIONS, (policy.name, action)


def test_oracle_requires_an_environment() -> None:
    with pytest.raises(RuntimeError):
        OracleLookahead()(np.zeros(OBS_DIM, dtype=np.float32))


def test_policy_reset_restores_the_first_call_behaviour() -> None:
    sample = _obs_sample(60)
    for policy in (WorkloadOnlyThrottle(), RiskAwareRuleBased()):
        first = [policy(o) for o in sample]
        policy.reset()
        second = [policy(o) for o in sample]
        assert first == second, policy.name


# --------------------------------------------------------------------------
# Risk signal: invariance of the blind policy, sensitivity of the aware one
# --------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["random", "zero", "one", "mixed"])
def test_workload_only_is_invariant_to_hazard_fields(variant: str) -> None:
    rng = np.random.default_rng(1)
    base = rng.random((120, OBS_DIM)).astype(np.float32)
    changed = base.copy()
    if variant == "random":
        changed[:, 5:7] = rng.random((120, 2))
    elif variant == "zero":
        changed[:, 5:7] = 0.0
    elif variant == "one":
        changed[:, 5:7] = 1.0
    else:
        changed[:, 5] = 1.0
        changed[:, 6] = 0.0
    a = WorkloadOnlyThrottle()
    b = WorkloadOnlyThrottle()
    assert [a(o) for o in base] == [b(o) for o in changed]


def test_workload_only_is_not_invariant_to_the_fields_it_reads() -> None:
    """Guard against a vacuous invariance test: other fields do change the action."""
    low = np.zeros(OBS_DIM, dtype=np.float32)
    high = low.copy()
    low[2], high[2] = 0.05, 0.95
    assert WorkloadOnlyThrottle()(low) != WorkloadOnlyThrottle()(high)


def _obs(**fields: float) -> np.ndarray:
    """Normalised observation with every field zero except those given (already normalised)."""
    names = list(Observation.__dataclass_fields__)
    v = np.zeros(OBS_DIM, dtype=np.float32)
    for name, value in fields.items():
        v[names.index(name)] = value
    return v


def test_risk_aware_depends_on_the_hazard() -> None:
    """Heavy foreground load wants the slowest rebuild; a high hazard must overrule it."""
    base = {"fg_bw": 0.7, "work_remaining": 1.0, "redundancy_left": 0.5}
    calm = RiskAwareRuleBased()(_obs(hazard=0.1, **base))
    risky = RiskAwareRuleBased()(_obs(hazard=0.5, **base))
    assert risky > calm


def test_risk_aware_uses_the_uncertainty_only_through_k() -> None:
    base = {"fg_bw": 0.7, "work_remaining": 1.0, "redundancy_left": 0.5, "hazard": 0.3}
    uncertainties = np.linspace(0.0, 0.5, 51)
    with_margin = {RiskAwareRuleBased(k=1.0)(_obs(hazard_uncertainty=float(u), **base)) for u in uncertainties}
    without = {RiskAwareRuleBased(k=0.0)(_obs(hazard_uncertainty=float(u), **base)) for u in uncertainties}
    assert len(with_margin) > 1  # the margin moves the action somewhere on this sweep
    assert len(without) == 1  # k = 0 ignores the field


def test_risk_floor_matches_the_contract_exposure_formula() -> None:
    """The floor is the smallest action with P < eps, found independently of the filter."""
    rng = np.random.default_rng(2)
    policy = RiskAwareRuleBased(k=0.0)
    for _ in range(400):
        hazard = float(rng.uniform(0.0, 3.0e-4))
        work = float(rng.uniform(0.0, 12000.0))
        o = Observation(0.0, 0.0, 0.0, 0.0, 15.0, hazard, 0.0, 1.0, 0.0, work, 0.0, 0.0)
        feasible = [exposure_probability(hazard, work, float(a)) < RISK_BUDGET_EPS for a in ACTIONS_MBPS]
        expected = feasible.index(True) if any(feasible) else N_ACTIONS - 1
        assert policy.risk_floor_index(o) == expected


def test_risk_floor_is_monotone_in_k() -> None:
    rng = np.random.default_rng(3)
    for _ in range(200):
        hazard = float(rng.uniform(0.0, 1.8e-4))
        unc = float(rng.uniform(0.0, 0.3e-4))
        work = float(rng.uniform(0.0, 12000.0))
        o = Observation(0.0, 0.0, 0.0, 0.0, 15.0, hazard, unc, 1.0, 0.0, work, 0.0, 0.0)
        floors = [RiskAwareRuleBased(k=k).risk_floor_index(o) for k in (0.0, 1.0, 2.0, 4.0)]
        assert floors == sorted(floors)


def test_slowest_risk_feasible_is_the_risk_floor() -> None:
    rng = np.random.default_rng(4)
    slow = SlowestRiskFeasible()
    aware = RiskAwareRuleBased()
    for obs in rng.random((100, OBS_DIM)).astype(np.float32):
        assert slow(obs) == aware.risk_floor_index(Observation.from_vector(obs))


def test_risk_aware_never_requests_less_than_the_floor() -> None:
    rng = np.random.default_rng(5)
    policy = RiskAwareRuleBased()
    for obs in rng.random((300, OBS_DIM)).astype(np.float32):
        assert policy(obs) >= policy.risk_floor_index(Observation.from_vector(obs))


# --------------------------------------------------------------------------
# Latency prediction and the selection rule
# --------------------------------------------------------------------------


def test_prediction_equals_the_simulators_realised_latency() -> None:
    """With the next foreground load known exactly, the predicted latency is exact."""
    for profile in PROFILE_NAMES:
        cfg = dataclasses.replace(SimConfig(), workload_profile=profile)
        rule = LatencyRule(cfg)
        env = RecoveryEnv(cfg)
        env.reset(seed=11)
        rng = np.random.default_rng(12)
        for _ in range(300):
            fg_next = peek_next_fg_mbps(env)
            _, _, terminated, truncated, info = env.step(int(rng.integers(0, N_ACTIONS)))
            predicted = float(rule.predicted_latency_ms(fg_next, info["alloc_applied_mbps"]))
            assert predicted == pytest.approx(info["latency_now"], rel=1e-12), profile
            assert float(rule.delivered_mbps(fg_next, info["alloc_applied_mbps"])) == pytest.approx(
                info["alloc_effective_mbps"], rel=1e-12
            )
            if terminated or truncated:
                break


def test_peek_matches_the_load_the_next_step_executes_across_a_trace_wrap() -> None:
    cfg = dataclasses.replace(SimConfig(), workload_profile="bursty")
    env = RecoveryEnv(cfg)
    env.reset(seed=5)
    wraps = 0
    for _ in range(700):
        peeked = peek_next_fg_mbps(env)
        _, _, terminated, truncated, info = env.step(0)
        assert peeked == info["fg_bw_mbps"]
        wraps = max(wraps, info["trace_wraps"])
        if terminated or truncated:
            break
    assert wraps >= 1  # the sample did cross the end of the 288-row trace


def test_peek_requires_a_running_episode() -> None:
    with pytest.raises(RuntimeError):
        peek_next_fg_mbps(RecoveryEnv())


def test_array_cost_terms_match_the_contract_helpers() -> None:
    """The vectorised D and V inside ``LatencyRule.score`` equal the contract's helpers."""
    cfg = SimConfig()
    rule = LatencyRule(cfg)
    reward = cfg.reward
    for fg in np.linspace(0.0, 199.0, 60):
        for alloc in ACTIONS_MBPS:
            latency = float(rule.predicted_latency_ms(fg, float(alloc)))
            expected = reward.alpha * latency_degradation(latency, cfg.latency_target_ms) + reward.gamma * latency_violation(
                latency, cfg.latency_target_ms
            )
            got = rule.score(np.array([fg]), np.array([1.0]), float(alloc), progress_value=0.0)
            assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_load_rule_is_monotone_in_foreground_load() -> None:
    """On a first call the chosen allocation never rises as the foreground load rises."""
    for selection in ("progress_value", "ceiling"):
        actions = [
            WorkloadOnlyThrottle(selection=selection)(_obs(fg_bw=float(f))) for f in np.linspace(0.0, 1.0, 101)
        ]
        assert actions == sorted(actions, reverse=True), selection
        assert actions[0] == N_ACTIONS - 1  # no load: fastest
        assert actions[-1] == 0  # saturated: slowest


def test_load_rule_never_requests_an_allocation_it_predicts_to_violate_when_another_fits() -> None:
    cfg = SimConfig()
    rule = LatencyRule(cfg)
    policy = WorkloadOnlyThrottle(cfg)
    for fg in np.linspace(0.0, 195.0, 80):
        action = policy(_obs(fg_bw=float(fg) / BW_TOTAL_MBPS))
        fits = [float(rule.predicted_latency_ms(fg, float(a))) <= rule.limit_ms for a in ACTIONS_MBPS]
        if any(fits):
            assert fits[action], fg


def test_belief_config_validation() -> None:
    with pytest.raises(ValueError):
        BeliefConfig(level_window=0)
    with pytest.raises(ValueError):
        BeliefConfig(window=0)
    with pytest.raises(ValueError):
        BeliefConfig(last_weight=1.5)
    with pytest.raises(ValueError):
        RiskAwareRuleBased(selection="bogus")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RiskAwareRuleBased(k=-1.0)


def test_belief_weights_form_a_distribution() -> None:
    policy = WorkloadOnlyThrottle()
    rng = np.random.default_rng(6)
    for _ in range(40):
        o = Observation.from_vector(rng.random(OBS_DIM).astype(np.float32))
        samples, weights = policy._belief(o)  # noqa: SLF001 - checking the internal contract
        assert samples.shape == weights.shape
        assert weights.sum() == pytest.approx(1.0)
        assert (samples >= 0.0).all() and (weights >= 0.0).all()
        assert samples[-1] == o.fg_bw


def test_oracle_action_has_the_lowest_true_score_at_every_step() -> None:
    """Per step, the oracle's action minimises the objective both rules optimise.

    Walks a trajectory driven by the deployable rule, and at every step compares
    the true one-interval score (exact next load, as the simulator will realise
    it) of the action each policy would take from that same state. The oracle
    optimises it exactly within the risk-feasible set, so it can never be worse,
    and on a bursty trace it must be strictly better somewhere or the test is
    vacuous.
    """
    cfg = dataclasses.replace(SimConfig(), workload_profile="bursty")
    rule = LatencyRule(cfg)
    policy = RiskAwareRuleBased(cfg)
    oracle = OracleLookahead(cfg)
    env = RecoveryEnv(cfg)
    oracle.bind(env)
    obs, _ = env.reset(seed=21)
    policy.reset()
    strictly_better = 0

    def true_score(fg_next: float, action_index: int) -> float:
        return rule.score(np.array([fg_next]), np.array([1.0]), float(ACTIONS_MBPS[action_index]), oracle.progress_value)

    for _ in range(600):
        fg_next = peek_next_fg_mbps(env)
        action = policy(obs)
        oracle_action = oracle(obs)
        assert true_score(fg_next, oracle_action) <= true_score(fg_next, action) + 1e-9
        strictly_better += true_score(fg_next, oracle_action) < true_score(fg_next, action) - 1e-9
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    assert strictly_better > 0


# --------------------------------------------------------------------------
# Episodes through the harness
# --------------------------------------------------------------------------


def test_fixed_20_completes_on_idle_within_the_horizon() -> None:
    # Healthy fleet: no clipping, so the delivered rate is exactly 20 MB/s.
    df = evaluate(FixedRate(20), profile="idle", seeds=[0, 1], config=HEALTHY)
    assert (df["termination_reason"] == TerminationReason.RECOVERY_COMPLETE.value).all()
    assert (df["steps"] == 2000).all() and (df["steps"] < MAX_EPISODE_STEPS).all()
    assert df["completion_hours"].tolist() == pytest.approx([2000 * 300 / 3600.0] * 2)
    assert df["mean_delivered_mbps"].tolist() == pytest.approx([20.0, 20.0])
    # Default (random) fleets: the filter may speed it up but it still completes.
    df = evaluate(FixedRate(20), profile="idle", seeds=[0, 1, 2, 3])
    assert df["completed"].all() and (df["steps"] <= 2000).all()


def test_risk_aware_is_never_clipped_by_the_filter_on_a_healthy_fleet() -> None:
    for profile in PROFILE_NAMES:
        df = evaluate(RiskAwareRuleBased(HEALTHY), profile=profile, seeds=[0, 1], config=HEALTHY)
        assert (df["clip_fraction"] == 0.0).all(), profile
        assert not df["risk_budget_violated"].any(), profile


def test_workload_only_is_clipped_by_the_filter_when_the_hazard_is_high() -> None:
    """The blind policy relies on the filter; the aware one does not (same scenarios)."""
    cfg = SimConfig(fleet_degradation=0.6)
    blind = evaluate(WorkloadOnlyThrottle(cfg), profile="peak", seeds=[0, 1], config=cfg)
    aware = evaluate(RiskAwareRuleBased(cfg), profile="peak", seeds=[0, 1], config=cfg)
    assert blind["clip_fraction"].mean() > 0.0
    assert (aware["clip_fraction"] == 0.0).all()


def test_evaluate_rows_come_from_info() -> None:
    """Cross-check the aggregated columns against an independent loop over ``info``."""
    cfg = dataclasses.replace(SimConfig(), workload_profile="diurnal")
    policy = FixedRate(80)
    row = evaluate(policy, profile="diurnal", seeds=[3]).iloc[0]

    env = RecoveryEnv(cfg)
    env.reset(seed=3)
    infos, rewards = [], []
    while True:
        _, reward, terminated, truncated, info = env.step(policy(np.zeros(OBS_DIM, dtype=np.float32)))
        infos.append(info)
        rewards.append(reward)
        if terminated or truncated:
            break
    degradation = [latency_degradation(i["latency_now"], cfg.latency_target_ms) for i in infos]
    assert row["steps"] == len(infos)
    assert row["termination_reason"] == infos[-1]["termination_reason"]
    assert row["total_reward"] == pytest.approx(sum(rewards))
    assert row["latency_degradation_mean"] == pytest.approx(np.mean(degradation))
    assert row["latency_degradation_p95"] == pytest.approx(np.percentile(degradation, 95))
    assert row["latency_violation_fraction"] == pytest.approx(np.mean([i["violated_latency"] for i in infos]))
    assert row["latency_budget_met"] == (row["latency_violation_fraction"] <= LATENCY_VIOLATION_FRACTION_MAX)
    assert row["p_failure_mean"] == pytest.approx(np.mean([i["p_failure_before_complete"] for i in infos]))
    assert row["p_failure_max"] == pytest.approx(max(i["p_failure_before_complete"] for i in infos))
    assert row["risk_budget_violated"] == any(i["violated_risk_budget"] for i in infos)
    assert row["clip_fraction"] == pytest.approx(np.mean([i["alloc_clipped_by_filter"] for i in infos]))
    assert row["mean_delivered_mbps"] == pytest.approx(np.mean([i["alloc_effective_mbps"] for i in infos]))
    assert row["fleet_level"] == infos[-1]["fleet_degradation_level"]


def test_evaluate_returns_the_documented_columns() -> None:
    df = evaluate(FixedRate(120), profile="idle", seeds=[0, 1])
    assert list(df.columns) == list(EPISODE_COLUMNS)
    assert df["policy"].eq("fixed_120").all() and df["profile"].eq("idle").all()
    assert df["seed"].tolist() == [0, 1]


def test_evaluate_is_deterministic_for_every_policy() -> None:
    for make in (
        lambda: FixedRate(80),
        lambda: WorkloadOnlyThrottle(),
        lambda: RiskAwareRuleBased(),
        lambda: OracleLookahead(),
    ):
        first = evaluate(make(), profile="diurnal", seeds=[0, 1])
        second = evaluate(make(), profile="diurnal", seeds=[0, 1])
        pd.testing.assert_frame_equal(first, second)


def test_reusing_a_policy_instance_gives_the_same_result() -> None:
    policy = RiskAwareRuleBased()
    first = evaluate(policy, profile="diurnal", seeds=[0, 1])
    second = evaluate(policy, profile="diurnal", seeds=[0, 1])
    pd.testing.assert_frame_equal(first, second)


def test_seed_order_does_not_change_an_episode() -> None:
    forward = evaluate(RiskAwareRuleBased(), profile="diurnal", seeds=[0, 1]).set_index("seed")
    backward = evaluate(RiskAwareRuleBased(), profile="diurnal", seeds=[1, 0]).set_index("seed")
    pd.testing.assert_frame_equal(forward.sort_index(), backward.sort_index())


def test_compare_evaluates_every_policy_on_the_identical_scenarios() -> None:
    policies = [FixedRate(80), FixedRate(120), WorkloadOnlyThrottle(), RiskAwareRuleBased(), OracleLookahead()]
    df = compare(policies, profiles=["idle", "diurnal"], seeds=[0, 1, 2])
    assert len(df) == len(policies) * 2 * 3
    for (_, _), block in df.groupby(["profile", "seed"]):
        assert len(block) == len(policies)
        assert block["fleet_level"].nunique() == 1
        assert block["initial_fg_bw_mbps"].nunique() == 1
        assert block["initial_budget_infeasible"].nunique() == 1
    # The scenarios themselves differ across seeds, so the check above is not vacuous.
    assert df[df["profile"] == "diurnal"].groupby("seed")["fleet_level"].first().nunique() == 3
    # Row order: profile, then policy in the order given, then seed.
    assert df["profile"].tolist() == ["idle"] * 15 + ["diurnal"] * 15
    assert df["policy"].iloc[:3].tolist() == ["fixed_80"] * 3


def test_check_pairing_detects_a_scenario_mismatch() -> None:
    df = compare([FixedRate(50), FixedRate(80)], profiles=["idle"], seeds=[0, 1])
    check_pairing(df)
    broken = df.copy()
    broken.loc[(broken["policy"] == "fixed_80") & (broken["seed"] == 1), "fleet_level"] += 0.1
    with pytest.raises(RuntimeError):
        check_pairing(broken)
    missing = df[~((df["policy"] == "fixed_80") & (df["seed"] == 1))]
    with pytest.raises(RuntimeError):
        check_pairing(missing)


def test_scenarios_do_not_depend_on_the_policy() -> None:
    """Same seed, different actions: the first observation and the fleet agree."""
    cfg = dataclasses.replace(SimConfig(), workload_profile="bursty")
    first = RecoveryEnv(cfg)
    second = RecoveryEnv(cfg)
    obs_a, info_a = first.reset(seed=9)
    obs_b, info_b = second.reset(seed=9)
    for _ in range(25):
        first.step(0)
        second.step(3)
    assert np.array_equal(obs_a, obs_b)
    assert info_a["fleet_degradation_level"] == info_b["fleet_degradation_level"]


def test_evaluate_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError):
        evaluate(FixedRate(20), profile="nope", seeds=[0])
    with pytest.raises(ValueError):
        evaluate(FixedRate(20), profile="idle", seeds=[])
    with pytest.raises(ValueError):
        evaluate(FixedRate(20), profile="idle", seeds=[0, 0])
    with pytest.raises(ValueError):
        evaluate(FixedRate(20), profile="idle", seeds=[0], config=SimConfig(workload_csv="x.csv"))
    with pytest.raises(ValueError):
        compare([FixedRate(20), FixedRate(20)], profiles=["idle"], seeds=[0])
    with pytest.raises(ValueError):
        compare([], profiles=["idle"], seeds=[0])


def test_evaluate_rejects_a_policy_built_for_a_different_latency_model() -> None:
    other = SimConfig(latency_unloaded_ms=3.0)
    with pytest.raises(ValueError):
        evaluate(RiskAwareRuleBased(other), profile="idle", seeds=[0])


def test_evaluate_pins_the_profile_without_mutating_the_config() -> None:
    cfg = SimConfig()
    df = evaluate(FixedRate(120), profile="peak", seeds=[0], config=cfg)
    assert cfg.workload_profile is None
    assert df["profile"].iloc[0] == "peak"


# --------------------------------------------------------------------------
# Summaries and rendering
# --------------------------------------------------------------------------


def _small_comparison() -> pd.DataFrame:
    return compare([FixedRate(80), FixedRate(120), OracleLookahead()], profiles=["idle", "diurnal"], seeds=[0, 1])


def test_summarise_has_one_row_per_policy_and_profile() -> None:
    df = _small_comparison()
    summary = summarise(df)
    assert len(summary) == 3 * 2
    assert summary["profile"].tolist() == ["idle"] * 3 + ["diurnal"] * 3
    assert (summary["n"] == 2).all()
    assert (summary["completed"] + summary["horizon_expired"] + summary["terminal_failure"] == summary["n"]).all()
    row = summary[(summary["profile"] == "idle") & (summary["policy"] == "fixed_120")].iloc[0]
    expected = df[(df["profile"] == "idle") & (df["policy"] == "fixed_120")]
    assert row["reward"] == pytest.approx(expected["total_reward"].mean())
    assert row["viol_frac"] == pytest.approx(expected["latency_violation_fraction"].mean())


def test_paired_delta_of_a_policy_with_itself_is_zero_and_signs_are_sensible() -> None:
    df = _small_comparison()
    same = paired_delta(df, "fixed_120", "fixed_120")
    assert (same["mean_diff"].dropna() == 0.0).all()
    diff = paired_delta(df, "fixed_120", "fixed_80", metrics=["completion_hours"])
    assert (diff["mean_diff"] < 0).all()  # the faster rebuild finishes sooner
    with pytest.raises(ValueError):
        paired_delta(df, "fixed_120", "absent")


def test_to_markdown_renders_a_table_and_labels_the_oracle() -> None:
    summary = summarise(_small_comparison())
    text = to_markdown(summary)
    lines = text.splitlines()
    assert lines[0].startswith("| profile | policy |")
    assert set(lines[1]) <= {"|", "-"}
    assert "ORACLE" in text.splitlines()[-1] or "ORACLE rows read" in text
    plain = to_markdown(pd.DataFrame({"policy": ["a"], "x": [float("nan")], "flag": [True], "n": [3]}))
    assert "n/a" in plain and "true" in plain and "ORACLE" not in plain


def test_cli_writes_the_episode_csv_and_the_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "results"
    assert main(["--profiles", "idle", "--seeds", "2", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    episodes = pd.read_csv(out / "episodes.csv")
    assert list(episodes.columns) == list(EPISODE_COLUMNS)
    assert len(episodes) == 7 * 2  # seven policies, two seeds, one profile
    summary_text = (out / "summary.md").read_text(encoding="utf-8")
    assert summary_text.strip() in printed
    assert "ORACLE_lookahead (not deployable)" in summary_text
    assert "simulated" in summary_text.lower() and "uncalibrated" in summary_text.lower()


def test_cli_rejects_unknown_profiles(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--profiles", "nonsense", "--seeds", "1", "--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        main(["--profiles", "idle", "--seeds", "0", "--out", str(tmp_path)])


# --------------------------------------------------------------------------
# Replay-CSV workloads and provenance
# --------------------------------------------------------------------------

REPO_PROFILES = Path(__file__).resolve().parents[1] / "data_pipeline" / "profiles"


def _write_csv(
    directory: Path,
    name: str = "idle",
    *,
    seed: int = 0,
    sidecar: dict[str, Any] | None = None,
) -> Path:
    """Write ``profile_<name>.csv`` from the generator, with an optional sidecar."""
    path = directory / f"profile_{name}.csv"
    generate_profile(name, seed).to_csv(path, index=False)
    if sidecar is not None:
        path.with_suffix(".meta.json").write_text(json.dumps(sidecar), encoding="utf-8")
    return path


def test_evaluate_and_compare_accept_a_csv_path_with_the_generator_schema(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path, sidecar={"profile": "idle", "synthetic": False})
    from_csv = evaluate(FixedRate(80), profile=csv, seeds=[0, 1])
    from_generator = evaluate(FixedRate(80), profile="idle", seeds=[0, 1])
    assert list(from_csv.columns) == list(from_generator.columns) == list(EPISODE_COLUMNS)
    assert from_csv["workload_source"].eq(csv.as_posix()).all()
    assert from_generator["workload_source"].eq("idle").all()
    assert from_csv["profile"].eq("idle").all()

    both = compare([FixedRate(50), FixedRate(80)], profiles=[csv], seeds=[0, 1])
    assert list(both.columns) == list(EPISODE_COLUMNS)
    assert len(both) == 4
    mixed = compare([FixedRate(50)], profiles=[csv, "peak"], seeds=[0])
    assert mixed["profile"].tolist() == ["idle", "peak"]


def test_a_str_is_always_a_profile_name_and_a_path_is_always_a_file(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path, sidecar={"profile": "idle", "synthetic": False})
    with pytest.raises(ValueError, match="pathlib.Path"):
        evaluate(FixedRate(80), profile=str(csv), seeds=[0])
    with pytest.raises(FileNotFoundError):
        evaluate(FixedRate(80), profile=tmp_path / "idle", seeds=[0])  # not the profile "idle"


def test_workload_synthetic_is_false_true_or_none_by_provenance(tmp_path: Path) -> None:
    with_sidecar = _write_csv(tmp_path, "idle", sidecar={"profile": "idle", "synthetic": False})
    bare_dir = tmp_path / "bare"
    bare_dir.mkdir()
    without_sidecar = _write_csv(bare_dir, "idle")
    assert not without_sidecar.with_suffix(".meta.json").exists()
    odd_dir = tmp_path / "odd"
    odd_dir.mkdir()
    non_boolean = _write_csv(odd_dir, "idle", sidecar={"profile": "idle", "synthetic": "no"})

    assert evaluate(FixedRate(80), profile=with_sidecar, seeds=[0])["workload_synthetic"].iloc[0] is False
    assert evaluate(FixedRate(80), profile="idle", seeds=[0])["workload_synthetic"].iloc[0] is True
    unknown = evaluate(FixedRate(80), profile=without_sidecar, seeds=[0, 1])
    assert unknown["workload_synthetic"].map(lambda v: v is None).all()
    assert evaluate(FixedRate(80), profile=non_boolean, seeds=[0])["workload_synthetic"].iloc[0] is None
    # Without a sidecar the label falls back to the file stem; nothing is inferred.
    assert unknown["profile"].eq("profile_idle").all()


def test_real_profile_csvs_carry_the_sidecar_flag() -> None:
    for name in PROFILE_NAMES:
        path = REPO_PROFILES / f"profile_{name}.csv"
        row = evaluate(FixedRate(80), profile=path, seeds=[0]).iloc[0]
        assert row["profile"] == name
        assert row["workload_synthetic"] is False


def test_pairing_holds_for_csv_runs_and_a_mismatched_scenario_still_raises(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path, sidecar={"profile": "idle", "synthetic": False})
    df = compare([FixedRate(50), FixedRate(80), RiskAwareRuleBased()], profiles=[csv], seeds=[0, 1, 2])
    check_pairing(df)
    for column, change in (
        ("fleet_level", lambda s: s + 0.1),
        ("trace_start_row", lambda s: s + 1),
        ("initial_fg_bw_mbps", lambda s: s + 1.0),
        ("workload_source", lambda s: s + "-other"),
        ("workload_synthetic", lambda s: s.map(lambda v: None)),
    ):
        broken = df.copy()
        rows = (broken["policy"] == "fixed_80") & (broken["seed"] == 1)
        broken[column] = broken[column].astype(object)
        broken.loc[rows, column] = change(broken.loc[rows, column])
        with pytest.raises(RuntimeError):
            check_pairing(broken)


def test_one_label_from_two_sources_is_rejected_not_merged(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path, sidecar={"profile": "idle", "synthetic": False})
    with pytest.raises(ValueError, match="separate calls"):
        compare([FixedRate(50)], profiles=[csv, "idle"], seeds=[0])
    with pytest.raises(ValueError):
        compare([FixedRate(50)], profiles=[csv, csv], seeds=[0])
    # Concatenating the two sources by hand is caught by the pairing check and the summaries.
    merged = pd.concat(
        [
            evaluate(FixedRate(50), profile=csv, seeds=[0]),
            evaluate(FixedRate(50), profile="idle", seeds=[0]),
        ],
        ignore_index=True,
    )
    with pytest.raises(RuntimeError):
        check_pairing(merged)
    with pytest.raises(ValueError):
        summarise(merged)
    with pytest.raises(ValueError):
        paired_delta(merged, "fixed_50", "fixed_50")


def test_same_seed_gives_the_same_fleet_and_start_row_for_csv_and_generator(tmp_path: Path) -> None:
    # Different trace content (generator seed 5), same number of rows.
    csv = _write_csv(tmp_path, seed=5, sidecar={"profile": "idle", "synthetic": False})
    seeds = [0, 1, 2, 3]
    from_csv = evaluate(FixedRate(80), profile=csv, seeds=seeds)
    from_generator = evaluate(FixedRate(80), profile="idle", seeds=seeds)
    assert from_csv["fleet_level"].tolist() == from_generator["fleet_level"].tolist()
    assert from_csv["trace_start_row"].tolist() == from_generator["trace_start_row"].tolist()
    assert from_csv["initial_budget_infeasible"].tolist() == from_generator["initial_budget_infeasible"].tolist()
    # The check is not vacuous: scenarios differ across seeds and the sources are different data.
    assert from_csv["fleet_level"].nunique() == len(seeds)
    assert from_csv["trace_start_row"].nunique() > 1
    assert from_csv["workload_source"].iloc[0] != from_generator["workload_source"].iloc[0]


def test_a_missing_file_and_an_invalid_csv_raise_clearly(tmp_path: Path) -> None:
    missing = tmp_path / "absent.csv"
    with pytest.raises(FileNotFoundError, match="absent.csv"):
        evaluate(FixedRate(80), profile=missing, seeds=[0])
    with pytest.raises(FileNotFoundError):
        compare([FixedRate(80)], profiles=[missing], seeds=[0])
    with pytest.raises(FileNotFoundError):
        evaluate(FixedRate(80), profile=tmp_path, seeds=[0])  # a directory is not a replay file

    good = _write_csv(tmp_path, sidecar={"profile": "idle", "synthetic": False})
    frame = pd.read_csv(good)
    bad_columns = tmp_path / "bad_columns.csv"
    frame.drop(columns=["disk_usage_mb"]).to_csv(bad_columns, index=False)
    with pytest.raises(ContractViolation):
        evaluate(FixedRate(80), profile=bad_columns, seeds=[0])
    negative = tmp_path / "negative.csv"
    frame.assign(read_iops=-1).to_csv(negative, index=False)
    with pytest.raises(ContractViolation):
        compare([FixedRate(80)], profiles=[negative], seeds=[0])
    with pytest.raises(ValueError):
        evaluate(FixedRate(80), profile=good, seeds=[0], config=SimConfig(workload_csv=good))


def test_summarise_reports_the_provenance_of_each_profile(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path, sidecar={"profile": "idle", "synthetic": False})
    bare = tmp_path / "bare"
    bare.mkdir()
    unlabeled = _write_csv(bare, "peak")
    df = compare([FixedRate(80)], profiles=[csv, unlabeled, "diurnal"], seeds=[0])
    summary = summarise(df).set_index("profile")["workload_synthetic"].to_dict()
    assert summary == {"idle": "false", "profile_peak": "unknown", "diurnal": "true"}


def test_cli_runs_a_directory_of_csvs_and_records_provenance(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    _write_csv(profiles, "idle", sidecar={"profile": "idle", "synthetic": False})
    _write_csv(profiles, "peak")  # no sidecar
    out = tmp_path / "results"
    assert main(["--workload-dir", str(profiles), "--seeds", "1", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    episodes = pd.read_csv(out / "episodes.csv")
    assert list(episodes.columns) == list(EPISODE_COLUMNS)
    assert len(episodes) == 7 * 2
    flags = episodes.groupby("profile")["workload_synthetic"].first().to_dict()
    assert flags == {"idle": "False", "profile_peak": "unknown"}
    assert set(episodes["workload_source"]) == {(profiles / "profile_idle.csv").as_posix(), (profiles / "profile_peak.csv").as_posix()}
    summary_text = (out / "summary.md").read_text(encoding="utf-8")
    assert summary_text.strip() in printed
    assert "synthetic: false" in summary_text and "synthetic: unknown" in summary_text
    assert "uncalibrated" in summary_text.lower()
    assert "synthetic workloads" not in summary_text


def test_rows_carry_the_sha256_of_the_csv_bytes(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path, sidecar={"profile": "idle", "synthetic": False})
    row = evaluate(FixedRate(80), profile=csv, seeds=[0]).iloc[0]
    assert row["workload_sha256"] == hashlib.sha256(csv.read_bytes()).hexdigest()
    assert evaluate(FixedRate(80), profile="idle", seeds=[0])["workload_sha256"].iloc[0] is None
    # Same name, different bytes: the digest changes while the path does not.
    csv.write_bytes(generate_profile("idle", 9).to_csv(index=False).encode())
    again = evaluate(FixedRate(80), profile=csv, seeds=[0]).iloc[0]
    assert again["workload_source"] == row["workload_source"] and again["workload_sha256"] != row["workload_sha256"]


def test_summary_prints_the_sidecar_disclosure_or_states_it_is_absent(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    profiles = tmp_path / "profiles"
    profiles.mkdir()
    _write_csv(profiles, "idle", sidecar={"profile": "idle", "synthetic": False, "disclosure": "selected extremes, not a representative sample"})
    _write_csv(profiles, "bursty", sidecar={"profile": "bursty", "synthetic": False, "label": "aggregate label text"})
    _write_csv(profiles, "diurnal", sidecar={"profile": "diurnal", "synthetic": False})
    _write_csv(profiles, "peak")
    (profiles / "profile_peak.meta.json").write_text("{not json", encoding="utf-8")
    out = tmp_path / "out"
    assert main(["--workload-dir", str(profiles), "--seeds", "1", "--out", str(out)]) == 0
    text = (out / "summary.md").read_text(encoding="utf-8")
    capsys.readouterr()
    assert "disclosure: selected extremes, not a representative sample" in text
    assert "disclosure: aggregate label text" in text
    assert "disclosure: absent (sidecar has no disclosure or label field)" in text
    assert "disclosure: absent (no readable sidecar)" in text
    assert hashlib.sha256((profiles / "profile_idle.csv").read_bytes()).hexdigest() in text
    generated = tmp_path / "gen"
    assert main(["--profiles", "idle", "--seeds", "1", "--out", str(generated)]) == 0
    assert "disclosure: not applicable (generator profile)" in (generated / "summary.md").read_text(encoding="utf-8")


def test_cli_accepts_single_csv_files_and_rejects_bad_workload_arguments(tmp_path: Path) -> None:
    csv = _write_csv(tmp_path, sidecar={"profile": "idle", "synthetic": False})
    out = tmp_path / "out"
    assert main(["--workload-csv", str(csv), "--seeds", "1", "--out", str(out)]) == 0
    assert len(pd.read_csv(out / "episodes.csv")) == 7
    empty = tmp_path / "empty"
    empty.mkdir()
    broken = tmp_path / "broken.csv"
    pd.read_csv(csv).drop(columns=["write_iops"]).to_csv(broken, index=False)
    for argv in (
        ["--workload-dir", str(empty)],
        ["--workload-dir", str(tmp_path / "absent")],
        ["--workload-csv", str(tmp_path / "absent.csv")],
        ["--workload-csv", str(broken)],
        ["--workload-csv", str(csv), "--profiles", "idle"],
    ):
        with pytest.raises(SystemExit):
            main([*argv, "--seeds", "1", "--out", str(tmp_path / "x")])


# --------------------------------------------------------------------------
# Slow: many episodes
# --------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("level", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_risk_aware_is_clipped_only_when_the_hazard_field_is_saturated(level: float) -> None:
    """The filter acts on the policy's requests only where the policy could not see the hazard.

    With ``k > 0`` the policy's floor sits above the filter's, so the only way
    the filter can still raise a request is a hazard above the observation's
    scale, which the clipped field cannot show.
    """
    cfg = SimConfig(fleet_degradation=level)
    for profile in PROFILE_NAMES:
        env = RecoveryEnv(dataclasses.replace(cfg, workload_profile=profile))
        policy = RiskAwareRuleBased(cfg, k=1.0)
        for seed in range(4):
            obs, _ = env.reset(seed=seed)
            policy.reset()
            while True:
                action = policy(obs)
                saturated = float(obs[5]) >= 1.0 - 1e-6
                obs, _, terminated, truncated, info = env.step(action)
                if info["alloc_clipped_by_filter"]:
                    assert saturated, (profile, level, seed)
                if terminated or truncated:
                    break


@pytest.mark.slow
def test_literal_oracle_is_no_worse_than_the_literal_rule_on_latency_degradation() -> None:
    """The spec's sanity check, applied to the rules it describes.

    Both use the largest allocation whose predicted latency fits the limit; the
    oracle predicts from the true next load and the rule from the last one. A
    wrong forecast overshoots the limit and costs latency degradation, so the
    oracle cannot be worse across a decent sample. (With the default selector
    this inequality is not implied: the oracle spends its better information on
    speed, so its mean degradation can be higher while its reward is better.)
    """
    seeds = list(range(12))
    for profile in PROFILE_NAMES:
        rule = evaluate(RiskAwareRuleBased(selection="ceiling"), profile=profile, seeds=seeds)
        oracle = evaluate(OracleLookahead(selection="ceiling"), profile=profile, seeds=seeds)
        assert oracle["latency_degradation_mean"].mean() <= rule["latency_degradation_mean"].mean() + 1e-12, profile
        assert oracle["latency_violation_fraction"].mean() <= rule["latency_violation_fraction"].mean() + 1e-12, profile


@pytest.mark.slow
def test_default_oracle_has_no_worse_reward_than_the_rule_on_average() -> None:
    seeds = list(range(12))
    for profile in ("diurnal", "bursty"):
        rule = evaluate(RiskAwareRuleBased(), profile=profile, seeds=seeds)
        oracle = evaluate(OracleLookahead(), profile=profile, seeds=seeds)
        assert oracle["total_reward"].mean() >= rule["total_reward"].mean(), profile


@pytest.mark.slow
def test_risk_aware_is_not_beaten_by_the_best_fixed_rate_on_average() -> None:
    """A strong baseline must match the best constant on each profile (paired seeds).

    On ``peak`` the best constant is also the best policy, so the rule is
    allowed 0.5% below it; everywhere else it must be at least as good.
    """
    seeds = list(range(16))
    policies: list[Any] = [*fixed_rate_policies(), RiskAwareRuleBased(HEALTHY)]
    df = compare(policies, profiles=["diurnal", "bursty", "peak"], seeds=seeds, config=HEALTHY)
    means = df.groupby(["profile", "policy"])["total_reward"].mean().unstack()
    for profile in means.index:
        best_fixed = means.loc[profile, [p.name for p in fixed_rate_policies()]].max()
        assert means.loc[profile, "risk_aware_rule_based"] >= best_fixed - 0.005 * abs(best_fixed), profile


@pytest.mark.slow
def test_compare_is_deterministic_over_a_larger_run() -> None:
    kwargs: dict[str, Any] = dict(profiles=["diurnal", "peak"], seeds=list(range(6)))
    first = compare(baseline_policies(), **kwargs)
    second = compare(baseline_policies(), **kwargs)
    pd.testing.assert_frame_equal(first, second)
