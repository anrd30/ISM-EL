"""Exogenous share of violations: X, C, f, equation (3) checks and the training counters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent.exogenous_trap import D, carriers, eq3_check, main, training_x, x_summary, x_table
from baselines.evaluate import evaluate
from baselines.policies import FixedRate
from sim.config import v0_config


def _episodes() -> pd.DataFrame:
    rows = []
    # two scenarios; fixed_20 violations are all exogenous, more aggressive rates add violations
    for policy, steps, exo, viol in (
        ("fixed_20", [100, 100], [10, 0], [0.10, 0.0]),
        ("fixed_50", [100, 100], [10, 0], [0.12, 0.02]),
        ("fixed_80", [100, 100], [10, 0], [0.20, 0.05]),
        ("fixed_120", [100, 100], [10, 0], [0.50, 0.30]),
    ):
        for seed in range(2):
            rows.append({"profile": "p", "policy": policy, "seed": seed, "steps": steps[seed],
                         "exogenous_violations": exo[seed], "latency_violation_fraction": viol[seed]})
    return pd.DataFrame(rows)


def test_x_c_and_f_per_scenario() -> None:
    table = x_table(_episodes()).set_index(["policy", "seed"])
    assert table.loc[("fixed_20", 0), "X"] == pytest.approx(0.10) and table.loc[("fixed_20", 0), "f"] == pytest.approx(1.0)
    assert np.isnan(table.loc[("fixed_20", 1), "f"])  # C = 0
    assert table.loc[("fixed_50", 0), "f"] == pytest.approx(0.10 / 0.12)
    assert bool(table.loc[("fixed_20", 0), "X_above_d"]) and not bool(table.loc[("fixed_20", 1), "X_above_d"])


def test_summary_reports_x_against_d_and_the_scenarios_above() -> None:
    summary = x_summary(_episodes()).set_index("policy")
    assert summary.loc["fixed_20", "X_mean"] == pytest.approx(0.05)
    assert summary.loc["fixed_20", "scenarios_X_above_d"] == "0"
    assert summary.loc["fixed_120", "f_ratio_of_means"] == pytest.approx(0.05 / 0.40)
    assert D == 0.05


def test_carriers_orders_scenarios_by_x() -> None:
    c = carriers(_episodes(), "fixed_50")
    assert c["seed"].tolist() == [0, 1] and c["share_of_total_X"].iloc[0] == pytest.approx(1.0)


def test_eq3_f_is_one_at_the_smallest_action_and_falls_with_aggressiveness() -> None:
    check = eq3_check(_episodes()).iloc[0]
    assert bool(check["f20_is_one"]) and bool(check["f_decreasing"]) and bool(check["f_non_increasing"])
    assert check["f_fixed_20"] == pytest.approx(1.0) and check["f_fixed_120"] < check["f_fixed_80"] < check["f_fixed_50"]
    broken = _episodes()
    broken.loc[broken["policy"] == "fixed_120", "latency_violation_fraction"] = [0.10, 0.0]
    assert not bool(eq3_check(broken).iloc[0]["f_decreasing"]) and not bool(eq3_check(broken).iloc[0]["f_non_increasing"])


def test_a_rate_that_adds_no_violation_keeps_f_at_one_so_f_is_non_increasing_but_not_decreasing() -> None:
    episodes = _episodes()
    episodes.loc[episodes["policy"] == "fixed_50", "latency_violation_fraction"] = [0.10, 0.0]
    check = eq3_check(episodes).iloc[0]
    assert check["f_fixed_50"] == pytest.approx(1.0)
    assert bool(check["f_non_increasing"]) and not bool(check["f_decreasing"])


def test_eq3_on_real_episodes_under_the_v0_config() -> None:
    frames = [evaluate(FixedRate(r), profile="peak", seeds=[0, 1, 2], config=v0_config()) for r in (20, 50, 80, 120)]
    check = eq3_check(pd.concat(frames, ignore_index=True)).iloc[0]
    assert bool(check["f20_is_one"])
    assert check["f_fixed_20"] == pytest.approx(1.0) and check["f_fixed_120"] < check["f_fixed_20"]


def test_training_counters_give_x_c_f_and_the_marginal_identity(tmp_path: Path) -> None:
    def write(name: str, real: list[int], marginal: list[int], steps: int = 1000) -> Path:
        path = tmp_path / name / "l3_stats.csv"
        path.parent.mkdir()
        exo = [r - m for r, m in zip(real, marginal)]
        pd.DataFrame({"iteration": range(len(real)), "n_steps": steps, "real_violations": real,
                      "min_action_violations": exo, "marginal_cost_steps": marginal,
                      "floor_above_min_steps": 0}).to_csv(path, index=False)
        return path

    files = [write("seed0", [80, 40], [10, 5]), write("seed1", [60, 100], [20, 30])]
    table = training_x(files).set_index("source")
    pooled = table.loc["pooled"]
    assert pooled["real_violations"] == 280 and pooled["marginal_cost_steps"] == 65
    assert pooled["min_action_violations"] == pooled["exogenous_real_minus_marginal"] == 215
    assert pooled["X"] == pytest.approx(215 / 4000) and pooled["C"] == pytest.approx(280 / 4000)
    assert pooled["f"] == pytest.approx(215 / 280)
    assert table.loc["seed0", "share_of_iterations_X_above_d"] == pytest.approx(0.5)


def test_cli_writes_the_tables(tmp_path: Path) -> None:
    src = tmp_path / "episodes.csv"
    _episodes().to_csv(src, index=False)
    assert main(["tables", "--episodes", str(src), "--out", str(tmp_path / "out")]) == 0
    for name in ("x_summary.csv", "x_by_scenario.csv", "eq3_check.csv"):
        assert (tmp_path / "out" / name).exists()
