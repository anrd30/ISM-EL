"""Scenario flags, exogenous violations, data-loss probability and the summary blocks of the harness."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from baselines import evaluate as ev
from baselines.evaluate import (
    EPISODE_COLUMNS,
    FILTER_FORCED_CLIP_FRACTION,
    _minimum_action_violates,
    _pinned_config,
    compare,
    evaluate,
    forced_seed_sets,
    main,
    pareto_table,
    recovery_time_distribution,
    scenario_blocks,
)
from baselines.policies import FixedRate
from contracts.contracts import ACTIONS_MBPS, BW_TOTAL_MBPS, CONTROL_INTERVAL_S, Observation
from sim.config import v0_config, v1_eval_config
from sim.recovery_env import RecoveryEnv
from sim.safety_filter import apply_safety_filter

REPO = Path(__file__).resolve().parents[1]
HOLDOUT = REPO / "data_pipeline" / "profiles" / "profile_diurnal_long_holdout.csv"
needs_holdout = pytest.mark.skipif(not HOLDOUT.exists(), reason=f"{HOLDOUT.name} absent")
needs_v1 = pytest.mark.skipif(
    not (REPO / "hazard" / "artifacts" / "v1").exists(), reason="hazard/artifacts/v1 absent"
)

EVAL = v1_eval_config(hazard_error_sigma=0.25)
PINNED_SEEDS = [0, 3, 9, 17, 19]
PINNED_FORCED = {3, 9, 17, 19}


def _forced(frame: pd.DataFrame) -> set[int]:
    return {int(s) for s in frame.loc[frame["filter_forced"], "seed"]}


def test_new_columns_are_documented_and_typed() -> None:
    frame = evaluate(FixedRate(120), profile="idle", seeds=[0, 1], config=v0_config())
    assert list(frame.columns) == list(EPISODE_COLUMNS)
    for name in (
        "latency_cost",
        "exogenous_violations",
        "data_loss_probability",
        "fixed20_clip_fraction",
        "fixed20_latency_cost",
        "filter_forced",
    ):
        assert name in frame.columns
        assert frame[name].notna().all()
    assert frame["filter_forced"].dtype == bool
    assert frame["data_loss_probability"].between(0.0, 1.0).all()
    assert (frame["exogenous_violations"] >= 0).all()


def test_latency_cost_is_the_latency_part_of_the_reward() -> None:
    config = v0_config()
    frame = evaluate(FixedRate(80), profile="bursty", seeds=[0, 1, 2], config=config)
    beta = config.reward.beta
    for _, row in frame.iterrows():
        degraded_steps = (-row["total_reward"] - row["latency_cost"]) / beta
        assert degraded_steps == pytest.approx(round(degraded_steps), abs=1e-6)
        assert 0 <= round(degraded_steps) <= row["steps"]


def test_minimum_action_formula_matches_the_environment_latency() -> None:
    config = v0_config()
    env = RecoveryEnv(config)
    env.reset(seed=0)
    for _ in range(40):
        _, _, terminated, truncated, info = env.step(0)
        fg = info["fg_bw_mbps"]
        # fixed_20 with no safety-filter raise: the minimum action is the applied action.
        if info["alloc_applied_mbps"] == ACTIONS_MBPS[0]:
            assert _minimum_action_violates(fg, 0, config) == info["violated_latency"]
        if terminated or truncated:
            break
    # load above the utilisation ceiling: nothing is delivered and latency sits at the ceiling value
    assert _minimum_action_violates(config.rho_max * BW_TOTAL_MBPS + 1.0, 0, config)
    # load far below the threshold
    assert not _minimum_action_violates(10.0, 0, config)
    # the same load violates with a larger floor
    fg = 0.8 * BW_TOTAL_MBPS
    assert not _minimum_action_violates(fg, 0, config) or _minimum_action_violates(fg, 3, config)
    assert _minimum_action_violates(fg, 3, config)


def test_fixed_20_exogenous_violations_equal_its_violations_when_the_floor_is_unclipped() -> None:
    """With the floor equal to the applied action (fixed_20), the counterfactual is the realised violation."""
    frame = evaluate(FixedRate(20), profile="peak", seeds=[0, 1], config=v0_config())
    for _, row in frame.iterrows():
        assert row["exogenous_violations"] == round(row["latency_violation_fraction"] * row["steps"])
    assert (frame["exogenous_violations"] > 0).any()


@needs_holdout
@needs_v1
def test_floor_from_the_observation_equals_the_applied_floor_when_the_hazard_is_below_the_scale() -> None:
    config, _ = _pinned_config(EVAL, HOLDOUT)
    env = RecoveryEnv(config)
    checked = 0
    for seed in (0, 3):
        obs, _ = env.reset(seed=seed)
        while True:
            seen = Observation.from_vector(obs)
            floor = apply_safety_filter(0, seen.hazard, seen.work_remaining).applied_index
            obs, _, terminated, truncated, info = env.step(0)
            assert float(ACTIONS_MBPS[floor]) == info["alloc_applied_mbps"]
            checked += 1
            if terminated or truncated:
                break
    assert checked > 300


@needs_holdout
@needs_v1
def test_pinned_filter_forced_scenarios_on_the_holdout() -> None:
    frame = evaluate(FixedRate(120), profile=HOLDOUT, seeds=PINNED_SEEDS, config=EVAL)
    assert _forced(frame) == PINNED_FORCED
    by_seed = frame.set_index("seed")
    assert not by_seed.loc[0, "filter_forced"] and by_seed.loc[9, "filter_forced"]
    assert by_seed.loc[9, "fixed20_latency_cost"] == pytest.approx(2743.2253, abs=1e-3)
    assert by_seed.loc[17, "fixed20_latency_cost"] == pytest.approx(1349.9192, abs=1e-3)
    assert by_seed.loc[19, "fixed20_latency_cost"] == pytest.approx(1336.4996, abs=1e-3)
    assert (by_seed.loc[[0, 3], "fixed20_latency_cost"] < 50).all()
    sets = forced_seed_sets(frame).set_index("definition")
    assert sets.loc["fixed20_cost", "forced_seeds"] == "9 17 19"
    assert sets.loc["primary", "forced_seeds"] == "3 9 17 19"


@needs_holdout
@needs_v1
def test_forcing_does_not_depend_on_the_evaluated_policy() -> None:
    a = evaluate(FixedRate(20), profile=HOLDOUT, seeds=[0, 9], config=EVAL)
    b = evaluate(FixedRate(120), profile=HOLDOUT, seeds=[0, 9], config=EVAL)
    for column in ("filter_forced", "fixed20_clip_fraction", "fixed20_latency_cost"):
        assert a[column].tolist() == b[column].tolist()


def test_data_loss_probability_follows_the_true_group_hazard() -> None:
    config = v0_config()
    frame = evaluate(FixedRate(20), profile="idle", seeds=[0], config=config)
    env = RecoveryEnv(ev._pinned_config(config, "idle")[0])
    env.reset(seed=0)
    total = 0.0
    while True:
        _, _, terminated, truncated, info = env.step(0)
        total += info["hazard_true_per_hour"]
        if terminated or truncated:
            break
    mean = total * CONTROL_INTERVAL_S / 3600.0
    expected = 1.0 - math.exp(-mean) * (1.0 + mean)
    assert frame["data_loss_probability"].iloc[0] == pytest.approx(expected, rel=1e-9, abs=1e-15)
    assert 0.0 < expected < 1.0


def _two_policy_frame() -> pd.DataFrame:
    return compare([FixedRate(80), FixedRate(120)], profiles=["idle", "peak"], seeds=[0, 1, 2], config=v0_config())


def test_scenario_blocks_partition_the_rows() -> None:
    frame = _two_policy_frame()
    frame.loc[frame["seed"] == 1, "filter_forced"] = True  # a flag is a scenario property: set it for every policy
    blocks = scenario_blocks(frame)
    assert len(blocks["controllable"]) + len(blocks["filter_forced"]) == len(blocks["all"]) == len(frame)
    forced = set(blocks["filter_forced"]["seed"])
    assert 1 in forced and forced == set(frame.loc[frame["filter_forced"], "seed"])
    assert not forced & set(blocks["controllable"]["seed"])


def test_forced_seed_sets_use_the_documented_thresholds() -> None:
    frame = pd.DataFrame(
        {
            "profile": ["p"] * 4,
            "seed": [0, 1, 2, 3],
            "fixed20_clip_fraction": [0.0, 0.02, 0.10, 0.30],
            "fixed20_latency_cost": [0.0, 60.0, 0.0, 10.0],
        }
    )
    sets = forced_seed_sets(frame).set_index("definition")["forced_seeds"].to_dict()
    assert sets == {"primary": "2 3", "clip_0.01": "1 2 3", "clip_0.25": "3", "fixed20_cost": "1"}
    assert FILTER_FORCED_CLIP_FRACTION == 0.10


def test_pareto_and_recovery_time_tables() -> None:
    frame = _two_policy_frame()
    pareto = pareto_table(frame)
    assert len(pareto) == 4
    assert {"completion_p50_h", "completion_p95_h", "completion_p99_h", "viol_frac", "lat_budget_met",
            "p_fail_mean", "p_fail_delivered", "exog_viol", "data_loss_prob"} <= set(pareto.columns)
    assert (pareto["completion_p50_h"] <= pareto["completion_p95_h"]).all()
    assert (pareto["completion_p95_h"] <= pareto["completion_p99_h"]).all()
    dist = recovery_time_distribution(frame)
    assert (dist["p99"] <= dist["max"] + 1e-12).all() and (dist["n_completed"] <= dist["n"]).all()
    row = dist[(dist["profile"] == "idle") & (dist["policy"] == "fixed_120")].iloc[0]
    hours = frame[(frame["profile"] == "idle") & (frame["policy"] == "fixed_120")]["completion_hours"]
    assert row["mean"] == pytest.approx(hours.mean())
    assert row["p50"] == pytest.approx(np.percentile(hours, 50))


def test_cli_writes_the_blocks_and_the_distribution(tmp_path: Path) -> None:
    out = tmp_path / "results"
    assert main(["--profiles", "idle", "--seeds", "2", "--out", str(out)]) == 0
    text = (out / "summary.md").read_text(encoding="utf-8")
    for heading in ("## All scenarios", "## Controllable scenarios", "## Filter-forced scenarios",
                    "## Pareto table", "## Recovery-time distribution"):
        assert heading in text
    assert "simulated data-loss probability" in text
    assert (out / "pareto.csv").exists() and (out / "recovery_time_distribution.csv").exists()
    episodes = pd.read_csv(out / "episodes.csv")
    assert list(episodes.columns) == list(EPISODE_COLUMNS)
