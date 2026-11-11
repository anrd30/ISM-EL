"""Tests for the learned controllers: the forecast PPO agent, the seasonal-rule policy search, the residual network, and evaluation guards.

Every test uses ``v0_config`` so none needs ``hazard/artifacts/v1/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from agent.evaluate_learned import main as evaluate_main
from agent.evaluate_learned import strata_table
from agent.seasonal_rule import SeasonalForecastRule
from agent.residual_features import (
    DAY_STEPS,
    FEATURE_DIM,
    LEVEL_WINDOW,
    N_RESIDUAL,
    NO_CHANGE_INDEX,
    RESIDUAL_DELTAS,
    ResidualFeatures,
    compose_action,
)
from agent.residual_policy import ResidualActor, ResidualPolicy, load_residual_checkpoint
from agent.train_residual import ResidualTrainConfig, build_sim_config, split_time_forward, train
from agent.train_residual import main as train_main
from baselines.policies import RiskAwareRuleBased
from contracts.contracts import N_ACTIONS, OBS_DIM
from sim.config import v0_config
from sim.recovery_env import RecoveryEnv

REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / "data_pipeline" / "profiles" / "profile_diurnal.csv"
SEASONAL_START = DAY_STEPS + LEVEL_WINDOW - 1
SEASONAL_FLAG = OBS_DIM + N_ACTIONS + 1 + N_ACTIONS + 12


def _config():
    return v0_config(hazard_error_sigma=0.25, workload_profile="diurnal")


def _episode_obs(n: int, seed: int = 3) -> list[np.ndarray]:
    env = RecoveryEnv(_config())
    obs, _ = env.reset(seed=seed)
    out = [obs]
    while len(out) < n:
        obs, _, term, trunc, _ = env.step(1)
        out.append(obs)
        if term or trunc:
            obs, _ = env.reset()
            out.append(obs)
    return out[:n]


def test_compose_action_clips_to_floor_and_top() -> None:
    assert RESIDUAL_DELTAS[NO_CHANGE_INDEX] == 0
    assert compose_action(2, 0, NO_CHANGE_INDEX) == 2
    assert compose_action(0, 0, 0) == 0
    assert compose_action(N_ACTIONS - 1, 0, N_RESIDUAL - 1) == N_ACTIONS - 1
    assert compose_action(2, 2, 0) == 2
    with pytest.raises(ValueError):
        compose_action(1, 0, N_RESIDUAL)


def test_untrained_policy_is_the_rule() -> None:
    cfg = _config()
    env_a, env_b = RecoveryEnv(cfg), RecoveryEnv(cfg)
    rule = RiskAwareRuleBased(cfg)
    policy = ResidualPolicy(ResidualActor(), cfg)
    for seed in (0, 1):
        obs_a, _ = env_a.reset(seed=seed)
        obs_b, _ = env_b.reset(seed=seed)
        rule.reset()
        policy.reset()
        done = False
        while not done:
            a, b = rule(obs_a), policy(obs_b)
            assert a == b
            obs_a, _, term, trunc, _ = env_a.step(a)
            obs_b, _, _, _, _ = env_b.step(b)
            done = term or trunc
    assert policy.n_changed == 0 and policy.n_steps > 0


def test_features_are_causal() -> None:
    seq = _episode_obs(SEASONAL_START + 40)
    cut = SEASONAL_START + 10
    altered = [o.copy() for o in seq]
    for o in altered[cut + 1 :]:
        o[2] = 1.0 - o[2]
    fa, fb = ResidualFeatures(RiskAwareRuleBased(_config())), ResidualFeatures(RiskAwareRuleBased(_config()))
    fa.reset()
    fb.reset()
    for t in range(len(seq)):
        xa, _ = fa.update(seq[t])
        xb, _ = fb.update(altered[t])
        if t <= cut:
            np.testing.assert_array_equal(xa, xb)


def test_seasonal_block_switches_on_after_a_day() -> None:
    seq = _episode_obs(SEASONAL_START + 2)
    feats = ResidualFeatures(RiskAwareRuleBased(_config()))
    feats.reset()
    flags = []
    for o in seq:
        x, step = feats.update(o)
        assert x.shape == (FEATURE_DIM,) and x.dtype == np.float32
        assert np.all(np.isfinite(x))
        assert 0 <= step.floor <= step.action < N_ACTIONS
        flags.append(x[SEASONAL_FLAG])
    assert all(f == 0.0 for f in flags[:SEASONAL_START])
    assert flags[SEASONAL_START] == 1.0


def test_seasonal_rule_equals_rule_before_a_day() -> None:
    seq = _episode_obs(SEASONAL_START)
    rule, seasonal = RiskAwareRuleBased(_config()), SeasonalForecastRule(_config())
    rule.reset()
    seasonal.reset()
    for o in seq:
        assert rule(o) == seasonal(o)


def test_train_smoke_writes_loadable_checkpoints(tmp_path: Path) -> None:
    cfg = ResidualTrainConfig(
        iterations=2, n_envs=2, steps_per_env=16, workload_csv=str(PROFILE), hazard="v0",
        out_dir=str(tmp_path / "seed0"), val_every=1, val_seeds=2, minibatch_size=16, val_days=0,
    )
    out = train(cfg, echo=False)
    for name in ("ckpt_best.pt", "ckpt_final.pt", "log.csv"):
        assert (out / name).is_file()
    meta = load_residual_checkpoint(out / "ckpt_final.pt")["meta"]
    assert meta["hazard"] == "v0" and meta["eval_fleet"] is False
    policy = ResidualPolicy.from_checkpoint(out / "ckpt_final.pt", build_sim_config("v0", False, 0.25))
    assert 0 <= policy(_episode_obs(1)[0]) < N_ACTIONS


def test_checkpoint_feature_version_is_checked(tmp_path: Path) -> None:
    cfg = ResidualTrainConfig(
        iterations=1, n_envs=1, steps_per_env=8, workload_csv=str(PROFILE), hazard="v0",
        out_dir=str(tmp_path / "s"), val_every=1, val_seeds=1, minibatch_size=8, val_days=0,
    )
    out = train(cfg, echo=False)
    ckpt = load_residual_checkpoint(out / "ckpt_final.pt")
    ckpt["meta"]["feature_version"] = 999
    torch.save(ckpt, tmp_path / "bad.pt")
    with pytest.raises(ValueError):
        ResidualPolicy.from_checkpoint(tmp_path / "bad.pt", build_sim_config("v0", False, 0.25))


def test_evaluation_refuses_in_sample_workload(tmp_path: Path) -> None:
    cfg = ResidualTrainConfig(
        iterations=1, n_envs=1, steps_per_env=8, workload_csv=str(PROFILE), hazard="v0",
        out_dir=str(tmp_path / "s"), val_every=1, val_seeds=1, minibatch_size=8, val_days=0,
    )
    out = train(cfg, echo=False)
    with pytest.raises(SystemExit):
        evaluate_main([
            "--checkpoint", str(out / "ckpt_final.pt"), "--workload-csv", str(PROFILE),
            "--hazard", "v0", "--seeds", "1", "--out", str(tmp_path / "eval"),
        ])
    with pytest.raises(SystemExit):
        evaluate_main([
            "--checkpoint", str(out / "ckpt_final.pt"), "--workload-csv", str(PROFILE),
            "--hazard", "v0", "--hazard-error-sigma", "0.0", "--no-strict-holdout", "--seeds", "1",
            "--out", str(tmp_path / "eval"),
        ])


def test_evaluation_runs_and_writes_strata(tmp_path: Path) -> None:
    cfg = ResidualTrainConfig(
        iterations=1, n_envs=1, steps_per_env=8, workload_csv=str(PROFILE), hazard="v0",
        out_dir=str(tmp_path / "s"), val_every=1, val_seeds=1, minibatch_size=8, val_days=0,
    )
    out = train(cfg, echo=False)
    code = evaluate_main([
        "--checkpoint", str(out / "ckpt_final.pt"), "--workload-csv", str(PROFILE), "--no-strict-holdout",
        "--hazard", "v0", "--seeds", "2", "--diagnostics", "--out", str(tmp_path / "eval"),
    ])
    assert code == 0
    assert (tmp_path / "eval" / "summary.md").is_file()
    assert (tmp_path / "eval" / "strata.csv").is_file()


def test_strata_table_counts() -> None:
    import pandas as pd

    rows = []
    for seed, (rule_r, treat_r, infeasible) in enumerate([(-60, -55, False), (-70, -72, False), (-2000, -1990, True)]):
        rows.append({"policy": "risk_aware_rule_based", "seed": seed, "total_reward": rule_r,
                     "initial_budget_infeasible": infeasible})
        rows.append({"policy": "t", "seed": seed, "total_reward": treat_r, "initial_budget_infeasible": infeasible})
    table = strata_table(pd.DataFrame(rows), ["t"]).set_index("stratum")
    assert table.loc["all", "n"] == 3 and table.loc["all", "n_better"] == 2
    assert table.loc["initial_budget_infeasible=True (pre-registered)", "n"] == 1
    assert table.filter(like="typical", axis=0)["gap"].iloc[0] == pytest.approx(1.5)


def test_cli_guards() -> None:
    with pytest.raises(SystemExit):
        train_main(["--eval-fleet", "--hazard", "v0", "--iterations", "1"])
    with pytest.raises(SystemExit):
        train_main(["--val-seed-start", "0", "--hazard", "v0", "--iterations", "1"])


def test_split_time_forward_is_contiguous_and_rebased(tmp_path: Path) -> None:
    fit, val = split_time_forward(str(PROFILE), 96, tmp_path)
    src = PROFILE.read_bytes().splitlines()
    fit_lines = Path(fit).read_bytes().splitlines()
    val_lines = Path(val).read_bytes().splitlines()
    assert fit_lines[0] == val_lines[0] == src[0]
    assert fit_lines[1:] == src[1:-96]
    assert len(val_lines) == 97
    assert val_lines[1].split(b",")[0] == b"0"
    assert [line.split(b",")[1:] for line in val_lines[1:]] == [line.split(b",")[1:] for line in src[-96:]]
    with pytest.raises(ValueError):
        split_time_forward(str(PROFILE), 200, tmp_path)


def test_train_with_time_forward_validation(tmp_path: Path) -> None:
    cfg = ResidualTrainConfig(
        iterations=1, n_envs=1, steps_per_env=8, workload_csv=str(PROFILE), hazard="v0",
        out_dir=str(tmp_path / "s"), val_every=1, val_seeds=1, minibatch_size=8, val_days=0,
    )
    import dataclasses

    from agent import residual_features

    # profile_diurnal has one day; hold back a third of it by patching the day length used for the split.
    cfg = dataclasses.replace(cfg, val_days=1)
    original = residual_features.DAY_STEPS
    import agent.train_residual as tr

    tr.DAY_STEPS = 96
    try:
        out = train(cfg, echo=False)
    finally:
        tr.DAY_STEPS = original
    meta = load_residual_checkpoint(out / "ckpt_final.pt")["meta"]
    assert meta["fit_csv"].endswith("_fit.csv") and meta["val_csv"].endswith("_val.csv")
    assert meta["fit_csv_sha256"] != meta["val_csv_sha256"]


def test_seasonal_rule_rejects_bad_parameters() -> None:
    with pytest.raises(ValueError):
        SeasonalForecastRule(_config(), mix=1.5)
    with pytest.raises(ValueError):
        SeasonalForecastRule(_config(), spread=-1.0)


def test_seasonal_rule_mix_zero_is_the_rule() -> None:
    seq = _episode_obs(SEASONAL_START + 30)
    rule, seasonal = RiskAwareRuleBased(_config()), SeasonalForecastRule(_config(), mix=0.0)
    rule.reset()
    seasonal.reset()
    for o in seq:
        assert rule(o) == seasonal(o)


def test_search_decode_respects_bounds() -> None:
    from agent.search_seasonal import decode

    params = decode("seasonal", np.array([10.0, -10.0, 10.0]))
    assert params["progress_scale"] == pytest.approx(3.0)
    assert params["spread"] == pytest.approx(0.2)
    assert 0.98 < params["mix"] < 1.0
    assert decode("rule", np.array([0.0])) == {"progress_scale": 1.0}


def test_search_smoke_and_evaluation(tmp_path: Path) -> None:
    import json

    from agent.search_seasonal import SearchConfig, search

    results = []
    for base in ("seasonal", "rule"):
        cfg = SearchConfig(base=base, workload_csv=str(PROFILE), hazard="v0", out_dir=str(tmp_path / base),
                           generations=1, population=3, elite=1, episodes=1, val_days=0, val_seeds=2)
        out = search(cfg, echo=False)
        result = json.loads((out / "search.json").read_text(encoding="utf-8"))
        assert result["base"] == base and result["generation"] == 1
        results.append(str(out / "search.json"))
    code = evaluate_main(["--search-result", *results, "--workload-csv", str(PROFILE), "--no-strict-holdout",
                          "--hazard", "v0", "--seeds", "2", "--out", str(tmp_path / "eval")])
    assert code == 0
    text = (tmp_path / "eval" / "summary.md").read_text(encoding="utf-8")
    assert "seasonal_search seed0" in text and "Seasonal search against the progress-price search" in text
    with pytest.raises(SystemExit):
        evaluate_main(["--search-result", results[0], "--workload-csv", str(PROFILE), "--hazard", "v0",
                       "--seeds", "1", "--out", str(tmp_path / "eval2")])


def test_cached_hazard_returns_identical_outputs() -> None:
    from agent.search_seasonal import CachedHazard
    from hazard.v0 import HeuristicHazardV0

    model = HeuristicHazardV0()
    cached = CachedHazard(model)
    features = _config().feature_fn(np.random.default_rng(0), 0.5, 20000.0, 0.1)
    first = cached(features, 720.0)
    second = cached(features, 720.0)
    assert first == second == model(features, 720.0)
    assert cached.hits == 1 and cached.misses == 1


def test_forecast_features_are_causal_and_masked() -> None:
    from agent.forecast_ppo import FEATURE_DIM, ForecastActor, ForecastFeatures

    seq = _episode_obs(SEASONAL_START + 30)
    cut = SEASONAL_START + 5
    altered = [o.copy() for o in seq]
    for o in altered[cut + 1 :]:
        o[2] = 1.0 - o[2]
    fa, fb = ForecastFeatures(), ForecastFeatures()
    fa.reset()
    fb.reset()
    actor = ForecastActor()
    for t in range(len(seq)):
        xa, floor = fa.update(seq[t])
        xb, _ = fb.update(altered[t])
        assert xa.shape == (FEATURE_DIM,) and np.all(np.isfinite(xa))
        if t <= cut:
            np.testing.assert_array_equal(xa, xb)
        logits = actor(torch.as_tensor(xa).unsqueeze(0))
        assert int(logits.argmax(dim=-1).item()) >= floor
        assert bool(torch.all(logits[0, :floor] < -1e8))


def test_forecast_ppo_trains_and_evaluates(tmp_path: Path) -> None:
    from agent.forecast_ppo import ForecastPPOConfig, train as train_ppo

    outs = {}
    for init in ("bc", "scratch"):
        cfg = ForecastPPOConfig(init=init, workload_csv=str(PROFILE), hazard="v0", out_dir=str(tmp_path / init),
                                iterations=1, n_envs=2, steps_per_env=16, pool_size=3, bc_episodes=1, bc_epochs=1,
                                val_every=1, val_days=0, val_seeds=1, minibatch_size=16)
        outs[init] = train_ppo(cfg, echo=False)
        for name in ("ckpt_init.pt", "ckpt_best.pt", "ckpt_final.pt", "log.csv"):
            assert (outs[init] / name).is_file()
    code = evaluate_main(["--ppo-checkpoint", str(outs["bc"] / "ckpt_best.pt"), str(outs["bc"] / "ckpt_init.pt"),
                          "--workload-csv", str(PROFILE), "--no-strict-holdout", "--hazard", "v0", "--seeds", "1",
                          "--out", str(tmp_path / "eval")])
    assert code == 0
    text = (tmp_path / "eval" / "summary.md").read_text(encoding="utf-8")
    assert "forecast_ppo_bc seed0" in text and "forecast_bc_only seed0" in text
    with pytest.raises(SystemExit):
        evaluate_main(["--ppo-checkpoint", str(outs["scratch"] / "ckpt_init.pt"), "--workload-csv", str(PROFILE),
                       "--no-strict-holdout", "--hazard", "v0", "--seeds", "1", "--out", str(tmp_path / "e2")])
    with pytest.raises(SystemExit):
        evaluate_main(["--ppo-checkpoint", str(outs["bc"] / "ckpt_best.pt"), "--workload-csv", str(PROFILE),
                       "--hazard", "v0", "--seeds", "1", "--out", str(tmp_path / "e3")])
