"""Gap tables and arms of the predictive-controller evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent.evaluate_predictive import CONTROLLER, OPTIMUM, RULE, SEASONAL, build_arms, gap_table
from baselines.policies import NeedsHistory
from sim.config import v0_config


def _frame() -> pd.DataFrame:
    seeds = list(range(6))
    rows = []
    rule = np.array([-10.0, -20.0, -30.0, -40.0, -50.0, -1000.0])
    clip = [0.0, 0.0, 0.0, 0.0, 0.0, 0.9]
    cost = [0.0, 0.0, 0.0, 0.0, 0.0, 900.0]
    gains = {
        RULE: np.zeros(6),
        SEASONAL: np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
        CONTROLLER: np.array([2.0, 2.0, 2.0, 2.0, 2.0, -5.0]),
        OPTIMUM: np.array([4.0, 4.0, 4.0, 4.0, 4.0, 1.0]),
    }
    viol = {RULE: 0.04, SEASONAL: 0.04, CONTROLLER: 0.03, OPTIMUM: 0.02}
    for name, gain in gains.items():
        for i, seed in enumerate(seeds):
            rows.append({"profile": "p", "policy": name, "seed": seed, "total_reward": rule[i] + gain[i],
                         "latency_violation_fraction": viol[name], "fixed20_clip_fraction": clip[i],
                         "fixed20_latency_cost": cost[i], "filter_forced": clip[i] >= 0.1})
    return pd.DataFrame(rows)


def test_gap_table_blocks_counts_and_share_of_optimum() -> None:
    table = gap_table(_frame(), {CONTROLLER: [CONTROLLER]}, {RULE: [RULE]})
    by = table.set_index("block")
    assert set(by.index) == {"all", "primary", "clip_0.01", "clip_0.25", "fixed20_cost"}
    assert by.loc["all", "n"] == 6 and by.loc["primary", "n"] == 5
    assert by.loc["primary", "mean_gain"] == pytest.approx(2.0)
    assert by.loc["primary", "better"] == 5 and by.loc["primary", "worse"] == 0
    assert by.loc["all", "mean_gain"] == pytest.approx((10.0 - 5.0) / 6)
    assert by.loc["primary", "percent_of_optimum"] == pytest.approx(50.0)
    assert by.loc["primary", "viol_frac_diff"] == pytest.approx(-0.01)


def test_gap_table_averages_several_policies_within_a_scenario_first() -> None:
    frame = _frame()
    a = frame[frame["policy"] == CONTROLLER].assign(policy="residual_q a")
    b = a.assign(policy="residual_q b", total_reward=a["total_reward"] + 2.0)
    both = pd.concat([frame, a, b], ignore_index=True)
    table = gap_table(both, {"q": ["residual_q a", "residual_q b"]}, {RULE: [RULE]})
    primary = table[table["block"] == "primary"].iloc[0]
    assert primary["mean_gain"] == pytest.approx(3.0)  # model a gains 2 and model b gains 4 in every scenario
    assert primary["n"] == 5


def test_gap_table_without_an_optimum_leaves_the_share_empty() -> None:
    frame = _frame()
    table = gap_table(frame[frame["policy"] != OPTIMUM], {CONTROLLER: [CONTROLLER]}, {RULE: [RULE]})
    assert table["percent_of_optimum"].isna().all()


def test_arms_have_unique_names_and_the_history_arms_are_the_only_protocol_users() -> None:
    arms = build_arms(v0_config())
    names = [a.name for a in arms]
    assert len(set(names)) == len(names)
    assert OPTIMUM in names and CONTROLLER in names and RULE in names and SEASONAL in names
    history_users = {a.name for a in arms if isinstance(a, NeedsHistory)}
    assert history_users == {"rule+history", "predictive_controller[seasonal]+history", CONTROLLER}
