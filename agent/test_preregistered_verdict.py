"""The pre-registered verdict function on synthetic frames and on the committed evidence of the reference populations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from agent.evaluate_predictive import CONTROLLER, OPTIMUM, RULE
from agent.preregistered_verdict import MIN_CONTROLLABLE, population_verdict, verdict

EVIDENCE = Path(__file__).resolve().parent / "evidence" / "predictive-controller-2026-11-19"


def _frame(gains: list[float], viol_diff: float, clip: list[float] | None = None, profile: str = "p") -> pd.DataFrame:
    n = len(gains)
    clip = clip or [0.0] * n
    rule = np.linspace(-50.0, -10.0, n)
    rows = []
    for name, reward, viol in (
        (RULE, rule, 0.04),
        (CONTROLLER, rule + np.array(gains), 0.04 + viol_diff),
        (OPTIMUM, rule + 20.0, 0.03),
    ):
        for i in range(n):
            rows.append({"profile": profile, "policy": name, "seed": i, "total_reward": reward[i],
                         "latency_violation_fraction": viol, "fixed20_clip_fraction": clip[i],
                         "fixed20_latency_cost": 0.0, "filter_forced": clip[i] >= 0.1})
    return pd.DataFrame(rows)


def test_both_conditions_met() -> None:
    gains = [3.0, 4.0, 5.0, 6.0, 3.5, 4.5, 5.5, 6.5, 4.0, 5.0]
    r = population_verdict(_frame(gains, -0.001))
    assert r["n"] == 10 and r["better"] == 10 and r["condition_1"] and r["condition_2"] and r["met"]
    assert r["t"] > 2 and r["percent_of_optimum"] == pytest.approx(100.0 * np.mean(gains) / 20.0)


def test_a_small_t_fails_condition_1() -> None:
    gains = [3.0, -4.0, 5.0, -6.0, 3.5, -4.5, 5.5, -6.5, 4.0, -5.0]
    r = population_verdict(_frame(gains, -0.001))
    assert not r["condition_1"] and r["condition_2"] and not r["met"]


def test_a_positive_violation_difference_fails_condition_2() -> None:
    gains = [3.0, 4.0, 5.0, 6.0, 3.5, 4.5, 5.5, 6.5, 4.0, 5.0]
    r = population_verdict(_frame(gains, +0.0005))
    assert r["condition_1"] and not r["condition_2"] and not r["met"]
    assert population_verdict(_frame(gains, 0.0))["condition_2"]


def test_forced_scenarios_are_excluded_from_the_primary_definition() -> None:
    gains = [3.0, 4.0, 5.0, 6.0, 3.5, 4.5, 5.5, 6.5, 4.0, -500.0]
    clip = [0.0] * 9 + [0.9]
    primary = population_verdict(_frame(gains, -0.001, clip), "primary")
    everything = population_verdict(_frame(gains, -0.001, clip), "all")
    assert primary["n"] == 9 and everything["n"] == 10
    assert primary["met"] and not everything["met"]


def test_fewer_than_the_minimum_controllable_scenarios_cannot_meet_the_criterion() -> None:
    gains = [3.0, 4.0, 5.0, 6.0, 3.5, 4.5, 5.5]
    assert len(gains) < MIN_CONTROLLABLE
    assert not population_verdict(_frame(gains, -0.001))["met"]


def test_the_verdict_needs_every_population() -> None:
    good = _frame([3.0, 4.0, 5.0, 6.0, 3.5, 4.5, 5.5, 6.5, 4.0, 5.0], -0.001, profile="C")
    bad = _frame([3.0, -4.0, 5.0, -6.0, 3.5, -4.5, 5.5, -6.5, 4.0, -5.0], -0.001, profile="D")
    assert verdict([good, good.assign(profile="D")])[0] is True
    met, text = verdict([good, bad])
    assert met is False and "**FAIL**" in text and "Criterion as registered" in text


@pytest.mark.skipif(not (EVIDENCE / "holdout" / "episodes.csv").exists(), reason="evidence absent")
def test_reference_populations_reproduce_the_decision_record_figures() -> None:
    holdout = pd.read_csv(EVIDENCE / "holdout" / "episodes.csv")
    altb = pd.read_csv(EVIDENCE / "altB" / "episodes.csv")
    h, a = population_verdict(holdout), population_verdict(altb)
    assert (h["n"], a["n"]) == (16, 16)
    assert h["mean_gain"] == pytest.approx(15.147, abs=5e-4) and h["t"] == pytest.approx(10.11, abs=5e-3)
    assert a["mean_gain"] == pytest.approx(13.836, abs=5e-4) and a["t"] == pytest.approx(3.68, abs=5e-3)
    assert h["percent_of_optimum"] == pytest.approx(94.4, abs=0.05) and a["percent_of_optimum"] == pytest.approx(40.2, abs=0.05)
