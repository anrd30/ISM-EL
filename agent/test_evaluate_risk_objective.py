"""Environments, hazard-weighted controller and tables of the exposure-weighted objective evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent.evaluate_risk_objective import (
    RULE,
    WEIGHTED_RULE,
    HazardWeightedController,
    _forced_trade,
    _gains,
    _summary,
    build_environments,
)
from baselines.policies import DEFAULT_PROGRESS_VALUE, NeedsHistory
from contracts.contracts import Observation
from sim.config import v0_config


def test_three_environments_differ_only_in_the_objective_and_the_infeasible_rule() -> None:
    envs = build_environments(v0_config(), 2.0e-4)
    assert list(envs) == ["contract", "objective", "objective_unforced"]
    assert envs["contract"].risk_objective is None and envs["contract"].infeasible_budget_max_rate
    assert envs["objective"].risk_objective.hazard_ref_per_hour == 2.0e-4 and envs["objective"].infeasible_budget_max_rate
    assert envs["objective_unforced"].risk_objective is not None and not envs["objective_unforced"].infeasible_budget_max_rate


def test_the_weighted_controller_keeps_the_history_protocol_and_scales_its_progress_value() -> None:
    policy = HazardWeightedController(v0_config(), 1.0e-4)
    assert isinstance(policy, NeedsHistory)
    obs = Observation(fg_read_iops=1000.0, fg_write_iops=1000.0, fg_bw=60.0, latency_now=3.0, latency_target=15.0,
                      hazard=1.5e-4, hazard_uncertainty=0.0, redundancy_left=1.0, progress=0.0, work_remaining=12000.0,
                      alloc_current=0.0, bw_available=130.0).to_vector(normalized=True)
    policy.reset()
    assert isinstance(policy(obs), int)
    assert policy.progress_value == pytest.approx(DEFAULT_PROGRESS_VALUE * 1.5)


def _frame() -> pd.DataFrame:
    rows = []
    for env in ("objective",):
        for policy, delta in ((RULE, 0.0), (WEIGHTED_RULE, 2.0)):
            for seed in range(4):
                rows.append({"environment": env, "policy": policy, "seed": seed,
                             "total_reward": -10.0 * (seed + 1) + (delta if seed < 3 else 0.0),
                             "latency_violation_fraction": 0.05, "data_loss_probability": 1e-4,
                             "completion_hours": 40.0, "mean_delivered_mbps": 80.0,
                             "fixed20_clip_fraction": 0.9 if seed == 3 else 0.0, "fixed20_latency_cost": 0.0,
                             "risk_budget_violated": seed == 3})
    return pd.DataFrame(rows)


def test_gains_count_the_scenarios_where_an_arm_differs_from_the_rule() -> None:
    gains = _gains(_frame()).set_index(["policy", "class"])
    row = gains.loc[(WEIGHTED_RULE, "primary_controllable")]
    assert row["n"] == 3 and row["gain_over_rule"] == pytest.approx(2.0) and row["scenarios_differing_from_rule"] == 3
    assert gains.loc[(WEIGHTED_RULE, "all"), "scenarios_differing_from_rule"] == 3


def test_summary_has_the_three_scenario_classes() -> None:
    summary = _summary(_frame())
    assert set(summary["class"]) == {"all", "primary_controllable", "filter_forced"}
    forced = summary[(summary["class"] == "filter_forced") & (summary["policy"] == RULE)].iloc[0]
    assert forced["n"] == 1 and forced["total_reward"] == pytest.approx(-40.0)


def test_forced_trade_lists_the_requested_seeds() -> None:
    trade = _forced_trade(_frame(), [3])
    assert set(trade["seed"]) == {3} and len(trade) == 2
    assert np.allclose(trade["data_loss_probability"], 1e-4)
