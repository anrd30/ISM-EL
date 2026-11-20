"""Tests for the residual Q-policy. All use ``v0_config`` and need no hazard artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from agent.evaluate_learned import main as evaluate_main
from agent.evaluate_ood import _policies
from agent.residual_features import NO_CHANGE_INDEX, N_RESIDUAL, ResidualFeatures, compose_action
from agent.residual_q import QConfig, ResidualQPolicy, allowed_moves, counterfactual_values, train
from agent.seasonal_rule import SeasonalForecastRule
from contracts.contracts import compute_reward, degraded_cost, latency_degradation, latency_violation
from sim.config import v0_config
from sim.recovery_env import RecoveryEnv

REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / "data_pipeline" / "profiles" / "profile_diurnal.csv"


def _config():
    return v0_config(hazard_error_sigma=0.25, workload_profile="diurnal")


class _Constant:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.value)


def test_keeping_the_rule_action_is_never_masked() -> None:
    cfg = _config()
    env = RecoveryEnv(cfg)
    feats = ResidualFeatures(SeasonalForecastRule(cfg))
    obs, _ = env.reset(seed=1)
    feats.reset()
    for _ in range(50):
        _, step = feats.update(obs)
        mask = allowed_moves(obs, step, cfg)
        assert mask.shape == (N_RESIDUAL,) and mask[NO_CHANGE_INDEX]
        obs, _, terminated, truncated, _ = env.step(step.action)
        if terminated or truncated:
            break


def test_counterfactual_value_matches_environment_reward_for_the_taken_move() -> None:
    cfg = _config()
    env = RecoveryEnv(cfg)
    feats = ResidualFeatures(SeasonalForecastRule(cfg))
    obs, _ = env.reset(seed=2)
    feats.reset()
    for _ in range(40):
        _, step = feats.update(obs)
        for k in range(N_RESIDUAL):
            action = compose_action(step.action, step.floor, k)
            if action == step.action:
                break
        obs, reward, terminated, truncated, info = env.step(step.action)
        values = counterfactual_values(feats.rule, step, float(info["fg_bw_mbps"]))
        latency = float(info["latency_now"])
        expected = compute_reward(latency_degradation(latency, cfg.latency_target_ms), degraded_cost(False),
                                  latency_violation(latency, cfg.latency_target_ms), cfg.reward)
        progress = values[NO_CHANGE_INDEX] - expected
        if not info["alloc_clipped_by_filter"]:
            assert progress == pytest.approx(0.004 * float(info["alloc_effective_mbps"]), abs=1e-9)
        if terminated or truncated:
            break


def test_policy_with_unreachable_margin_is_the_seasonal_rule() -> None:
    cfg = _config()
    models = {0: _Constant(5.0), 2: _Constant(5.0)}
    policy = ResidualQPolicy(models, cfg, margin=1e9)
    rule = SeasonalForecastRule(cfg)
    env_a, env_b = RecoveryEnv(cfg), RecoveryEnv(cfg)
    obs_a, _ = env_a.reset(seed=3)
    obs_b, _ = env_b.reset(seed=3)
    policy.reset()
    rule.reset()
    while True:
        a, b = policy(obs_a), rule(obs_b)
        assert a == b
        obs_a, _, terminated, truncated, _ = env_a.step(a)
        obs_b, _, _, _, _ = env_b.step(b)
        if terminated or truncated:
            break
    assert policy.n_moved == 0


def test_policy_respects_the_latency_ceiling() -> None:
    from contracts.contracts import BW_TOTAL_MBPS
    from sim.latency_filter import latency_ceiling_index

    cfg = _config()
    policy = ResidualQPolicy({0: _Constant(-1.0), 2: _Constant(100.0)}, cfg, margin=0.0)
    shadow = ResidualFeatures(SeasonalForecastRule(cfg))
    env = RecoveryEnv(cfg)
    obs, _ = env.reset(seed=4)
    policy.reset()
    shadow.reset()
    moved = 0
    while True:
        _, step = shadow.update(obs)
        action = policy(obs)
        if action != step.action:
            moved += 1
            ceiling, _ = latency_ceiling_index(float(obs[2]) * BW_TOTAL_MBPS, cfg)
            assert action <= max(ceiling, step.floor)
            assert action == min(step.action + 1, 3)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    assert moved > 0
    with pytest.raises(ValueError):
        ResidualQPolicy({}, cfg, margin=-1.0)


def test_train_load_and_evaluate(tmp_path: Path) -> None:
    cfg = QConfig(workload_csv=str(PROFILE), hazard="v0", out_dir=str(tmp_path / "seed0"), episodes=1,
                  val_days=0, val_seeds=2, max_iter=5)
    out = train(cfg, echo=False)
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["margin"] in (0.1, 0.25, 0.5) and meta["eval_fleet"] is False
    assert (out / "model.pkl").is_file() and (out / "validation.csv").is_file()
    code = evaluate_main(["--q-model", str(out), "--workload-csv", str(PROFILE), "--no-strict-holdout",
                          "--hazard", "v0", "--seeds", "2", "--diagnostics", "--out", str(tmp_path / "eval")])
    assert code == 0
    assert "residual_q seed0" in (tmp_path / "eval" / "summary.md").read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        evaluate_main(["--q-model", str(out), "--workload-csv", str(PROFILE), "--hazard", "v0", "--seeds", "1",
                       "--out", str(tmp_path / "eval2")])
    names = [p.name for p in _policies(_config(), [], [out])]
    assert names[-1] == "residual_q seed0"
