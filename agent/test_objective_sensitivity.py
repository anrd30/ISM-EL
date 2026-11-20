"""Settings grid, configurations, arms and ordering analysis of the objective sensitivity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent.objective_sensitivity import (
    CONTROLLER,
    FIXED_SUFFIX,
    ORACLE,
    OPTIMUM,
    PROGRESS_RATE_MBPS,
    RULE,
    SEASONAL,
    Setting,
    analyse_frames,
    headline,
    build_arms,
    make_config,
    settings,
)
from baselines.policies import DEFAULT_PROGRESS_VALUE
from sim.config import v0_config


def test_grid_has_the_default_and_one_factor_settings() -> None:
    names = [s.name for s in settings()]
    assert names[0] == "default" and len(names) == len(set(names)) == 15
    by = {s.name: s for s in settings()}
    assert [by[f"beta={b}"].beta for b in ("0.01", "0.03", "0.3", "1")] == [0.01, 0.03, 0.3, 1.0]
    assert [by[f"gamma={g}"].gamma for g in ("1", "3", "30", "100")] == [1.0, 3.0, 30.0, 100.0]
    assert by["alpha=0.3"].alpha == 0.3 and by["alpha=3"].alpha == 3.0
    assert by["latency_target_ms=12"].latency_target_ms == 12.0 and by["latency_unloaded_ms=2.5"].latency_unloaded_ms == 2.5
    assert all(s.beta == 0.1 and s.gamma == 10.0 and s.alpha == 1.0 for s in settings() if s.name.startswith("latency"))


def test_progress_value_is_beta_over_25_and_is_0_004_at_the_default() -> None:
    assert PROGRESS_RATE_MBPS == 25.0
    assert Setting("default").progress_value == pytest.approx(DEFAULT_PROGRESS_VALUE)
    assert Setting("b", beta=0.3).progress_value == pytest.approx(0.012)


def test_config_carries_the_setting_and_the_selector_reads_it() -> None:
    s = Setting("x", alpha=3.0, beta=0.3, gamma=30.0, latency_target_ms=18.0, latency_unloaded_ms=2.5)
    config = make_config(v0_config(), s)
    assert (config.reward.alpha, config.reward.beta, config.reward.gamma) == (3.0, 0.3, 30.0)
    assert config.latency_target_ms == 18.0 and config.latency_unloaded_ms == 2.5
    arms = {a.name: a for a in build_arms(config, s, with_fixed_variant=True)}
    rule = arms[RULE]
    assert rule.progress_value == pytest.approx(0.012) and rule.latency_rule.config.reward.gamma == 30.0
    assert rule.latency_rule.limit_ms == pytest.approx(1.5 * 18.0)
    assert arms[SEASONAL].progress_value == pytest.approx(0.012)
    assert arms[CONTROLLER].progress_value == pytest.approx(0.012) and arms[ORACLE].progress_value == pytest.approx(0.012)
    assert arms[RULE + FIXED_SUFFIX].progress_value == pytest.approx(DEFAULT_PROGRESS_VALUE)
    assert arms[ORACLE + FIXED_SUFFIX].progress_value == pytest.approx(DEFAULT_PROGRESS_VALUE)
    assert OPTIMUM in arms


def _frame(shift: float, swap: bool = False) -> pd.DataFrame:
    rows = []
    base = {OPTIMUM: 10.0, ORACLE: 8.0, CONTROLLER: 7.0, SEASONAL: 3.0, RULE: 0.0}
    if swap:
        base[ORACLE], base[CONTROLLER] = base[CONTROLLER], base[ORACLE]
    for seed in range(6):
        for policy, gain in base.items():
            rows.append({"profile": "p", "policy": policy, "seed": seed, "total_reward": -50.0 - seed + gain + shift * (seed % 2),
                         "latency_violation_fraction": 0.03, "completion_hours": 40.0, "steps": 480,
                         "fixed20_clip_fraction": 0.9 if seed == 5 else 0.0, "fixed20_latency_cost": 0.0,
                         "filter_forced": seed == 5, "trace_start_row": 0})
    return pd.DataFrame(rows)


def test_analysis_orders_policies_and_compares_with_the_default() -> None:
    table = analyse_frames({("A", "default"): _frame(0.0), ("A", "beta=0.3"): _frame(0.0, swap=True)})
    t = table[(table["class"] == "primary")].set_index("setting")
    assert t.loc["default", "pooled_order"] == "DP>oracle>ctrl>seas>rule"
    assert t.loc["beta=0.3", "pooled_order"] == "DP>ctrl>oracle>seas>rule"
    assert bool(t.loc["default", "matches_default_pooled"]) and not bool(t.loc["beta=0.3", "matches_default_pooled"])
    assert t.loc["beta=0.3", "share_scenarios_same_order_as_default_scenario"] == 0.0
    assert t.loc["default", "n"] == 5
    assert t.loc["default", "dp_minus_rule"] == pytest.approx(10.0)
    assert t.loc["default", "ctrl_share"] == pytest.approx(0.7) and t.loc["default", "oracle_share"] == pytest.approx(0.8)
    assert t.loc["default", "dp_minus_rule_pct_of_rule_cost"] == pytest.approx(100.0 * 10.0 / np.mean([50, 51, 52, 53, 54]))
    all_rows = table[table["class"] == "all"].set_index("setting")
    assert all_rows.loc["default", "n"] == 6


def test_headline_lists_the_ordering_exceptions_and_the_small_t() -> None:
    table = analyse_frames({("A", "default"): _frame(0.0), ("A", "beta=0.3"): _frame(0.0, swap=True)})
    text, exceptions = headline(table)
    assert "| default |" in text and "same order" in text
    assert "beta=0.3, population A, primary" in exceptions and "DP>ctrl>oracle>seas>rule" in exceptions
    assert "default, population A" not in exceptions
