"""Completion-time against violation frontiers: grids, achievable curve, constrained optimum and dominance."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent.frontier import MUS, PROGRESS_VALUES, V_GRID, achievable, analyse_points
from baselines.dp_optimum import DPHindsightOptimum
from baselines.evaluate import evaluate
from sim.config import v0_config


def test_grids_cover_the_registered_ranges() -> None:
    assert len(MUS) >= 25 and MUS[0] == pytest.approx(1e-2) and MUS[-1] == pytest.approx(1e4)
    assert len(PROGRESS_VALUES) >= 12 and PROGRESS_VALUES[0] == pytest.approx(1e-4) and PROGRESS_VALUES[-1] == pytest.approx(1e-1)
    assert np.allclose(np.diff(np.log10(MUS)), np.diff(np.log10(MUS))[0])


def test_achievable_curve_is_the_least_hours_with_violation_at_most_v() -> None:
    curve = achievable(np.array([0.30, 0.10, 0.02]), np.array([30.0, 40.0, 60.0]))
    at = lambda v: curve[int(round(v / (V_GRID[1] - V_GRID[0])))]
    assert np.isinf(at(0.01)) and at(0.02) == 60.0 and at(0.05) == 60.0 and at(0.10) == 40.0 and at(0.30) == 30.0
    assert (np.diff(curve[np.isfinite(curve)]) <= 1e-12).all()
    assert achievable(np.array([0.1]), np.array([np.nan])).max() == np.inf


def _points() -> pd.DataFrame:
    rows = []
    for seed in range(4):
        forced = seed == 3
        for kind, pts in (("dp", [(0.60, 29.0), (0.04, 33.0), (0.0, 36.0)]),
                          ("controller", [(0.60, 29.5), (0.04, 34.0), (0.0, 38.0)]),
                          ("rule", [(0.60, 30.0), (0.04, 36.0), (0.0, 42.0)])):
            for i, (v, h) in enumerate(pts):
                rows.append({"population": "P", "kind": kind, "param": i, "seed": seed, "hours": h, "violation": v,
                             "filter_forced": forced})
    return pd.DataFrame(rows)


def test_constrained_optimum_areas_and_dominance() -> None:
    scen, summary = analyse_points(_points())
    assert (scen["H_dp"] == 33.0).all() and (scen["H_controller"] == 34.0).all() and (scen["H_rule"] == 36.0).all()
    assert scen["ctrl_dominates"].all()
    assert (scen["area_controller_minus_dp"] > 0).all() and (scen["area_rule_minus_dp"] > scen["area_controller_minus_dp"]).all()
    primary = summary[summary["class"] == "primary"].iloc[0]
    assert primary["n"] == 3 and primary["hours_at_limit_dp"] == 33.0 and primary["rule_minus_controller_hours"] == 2.0
    assert primary["share_ctrl_dominates"] == 1.0 and summary[summary["class"] == "all"].iloc[0]["n"] == 4


def test_a_curve_that_is_worse_somewhere_does_not_dominate() -> None:
    points = _points()
    points.loc[(points["kind"] == "controller") & (points["param"] == 0), "hours"] = 31.0  # worse than the rule at 0.60
    scen, _ = analyse_points(points)
    assert not scen["ctrl_dominates"].any() and not scen["ctrl_weakly_dominates"].any()


def test_the_violation_price_trades_time_for_violations_under_the_v0_config() -> None:
    config = v0_config()
    cheap = evaluate(DPHindsightOptimum(config, violation_mu=1e-2), profile="bursty", seeds=[0, 1, 2], config=config)
    dear = evaluate(DPHindsightOptimum(config, violation_mu=1e4), profile="bursty", seeds=[0, 1, 2], config=config)
    assert (dear["latency_violation_fraction"].to_numpy() <= cheap["latency_violation_fraction"].to_numpy() + 1e-12).all()
    assert (dear["completion_hours"].to_numpy() >= cheap["completion_hours"].to_numpy() - 1e-9).all()
    assert cheap["policy"].iloc[0] == "DP_frontier mu=0.01 (not deployable)"
