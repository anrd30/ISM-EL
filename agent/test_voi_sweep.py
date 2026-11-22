"""Noisy look-ahead for the value-of-information sweep."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agent.voi_sweep import N_QUANTILES, NoisyOracle, _share_table
from baselines.evaluate import evaluate
from baselines.policies import OracleLookahead
from sim.config import v0_config


def _rewards(policy, seeds) -> list[float]:
    return evaluate(policy, profile="bursty", seeds=seeds, config=v0_config())["total_reward"].tolist()


def test_zero_sigma_is_the_look_ahead_oracle() -> None:
    config = v0_config()
    assert _rewards(NoisyOracle(config, 0.0), [0, 1]) == _rewards(OracleLookahead(config), [0, 1])


def test_noisy_oracle_is_reproducible_and_reseeded_per_episode() -> None:
    config = v0_config()
    first = _rewards(NoisyOracle(config, 7.0, seed=5), [0, 1, 2])
    second = _rewards(NoisyOracle(config, 7.0, seed=5), [0, 1, 2])
    other = _rewards(NoisyOracle(config, 7.0, seed=6), [0, 1, 2])
    assert first == second
    assert first != other


def test_sigma_must_be_finite_and_non_negative() -> None:
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            NoisyOracle(v0_config(), bad)


def test_belief_has_equal_weights_and_the_requested_spread() -> None:
    from sim.recovery_env import RecoveryEnv

    config = v0_config()
    policy = NoisyOracle(config, 10.0, seed=1)
    env = RecoveryEnv(config)
    policy.bind(env)
    env.reset(seed=0)
    policy.reset()
    samples, weights = policy._belief(None)  # type: ignore[arg-type]
    assert len(samples) == len(weights) == N_QUANTILES
    assert weights.sum() == pytest.approx(1.0) and np.allclose(weights, 1.0 / N_QUANTILES)
    assert np.diff(samples).min() >= 0.0
    unclipped = samples[samples > 0.0]
    if len(unclipped) == len(samples):
        assert samples.std() == pytest.approx(10.0, rel=0.1)


def test_share_table_divides_by_the_optimum_gain_on_the_controllable_scenarios() -> None:
    seeds = list(range(4))
    base = pd.DataFrame({"total_reward": [-10.0, -20.0, -30.0, -1000.0], "fixed20_clip_fraction": [0.0, 0.0, 0.0, 0.9],
                         "fixed20_latency_cost": [0.0, 0.0, 0.0, 900.0]}, index=pd.Index(seeds, name="seed"))
    dp = base.copy()
    dp["total_reward"] = base["total_reward"] + [4.0, 4.0, 4.0, 1.0]
    half = base.copy()
    half["total_reward"] = base["total_reward"] + [2.0, 2.0, 2.0, 0.0]
    table = _share_table({3.0: half}, base, dp, exclude=[3])
    primary = table[table["definition"] == "primary"].iloc[0]
    assert primary["n_controllable"] == 3
    assert primary["gain_over_rule"] == pytest.approx(2.0)
    assert primary["percent_of_optimum"] == pytest.approx(50.0)
    assert set(table["definition"]) >= {"primary", "clip_0.01", "clip_0.25", "fixed20_cost", "excluded_seeds"}
