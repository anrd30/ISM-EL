"""Tests for the constrained PPO agent.

These tests check the implementation's mechanics: shapes, the
truncation/termination bootstrap rule, GAE arithmetic, clipping, multiplier
dynamics, the ablation mask, determinism, the policy adapter and a short
training loop. The smoke training test only shows that the loop runs and does
not degrade catastrophically over a few iterations. It is not evidence that the
agent learns, and no test here asserts anything about the quality of a trained
policy.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest
import torch

import sim.recovery_env as recovery_env_module
from agent.ablation import HAZARD_INDICES, AblatedEnv, mask_observation
from agent.checkpoint import load_checkpoint, save_checkpoint
from agent.networks import Actor, Critic
from agent.policy_adapter import LearnedPolicy
from agent.ppo import (
    LagrangeMultiplier,
    PPOAgent,
    PPOConfig,
    compute_gae,
    ppo_loss,
    resolve_next_values,
)
from agent.rollout import EpisodeRecord, RolloutBatch, RolloutCollector
import agent.train as train_module
from agent.train import DEFAULT_ROLLOUT_STEPS, LOG_COLUMNS, TrainConfig, train, _checkpoint_payload
from baselines.evaluate import evaluate
from baselines.policies import Policy
from sim.scenario import v1_eval_drive_features
from hazard.v1 import ARTIFACT_DIR as V1_ARTIFACT_DIR
from contracts.contracts import (
    ACTIONS_MBPS,
    N_ACTIONS,
    OBS_DIM,
    RISK_BUDGET_EPS,
    TerminationReason,
    action_to_mbps,
    termination_flags,
)
from sim.config import v0_config
from sim.recovery_env import RecoveryEnv

IDLE = v0_config(workload_profile="idle")


#: The default ``SimConfig`` uses the fitted v1 hazard, whose artifacts are
#: gitignored (~218 MB, regenerate with ``python -m hazard.fit_v1``). Tests that
#: go through the default config skip without them, as the hazard and scenario
#: suites already do, rather than failing on a missing file.
_V1_PRESENT = (V1_ARTIFACT_DIR / "meta.json").exists()
_needs_v1 = pytest.mark.skipif(not _V1_PRESENT, reason="hazard/artifacts/v1 absent")

def _flags(reason: TerminationReason) -> tuple[bool, bool]:
    return termination_flags(reason)


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


def test_network_shapes_dtypes_and_architecture() -> None:
    actor, critic = Actor(), Critic()
    x = torch.rand(7, OBS_DIM)
    logits, value = actor(x), critic(x)
    assert logits.shape == (7, N_ACTIONS) and logits.dtype == torch.float32
    assert value.shape == (7,) and value.dtype == torch.float32
    assert [m.out_features for m in actor.net if isinstance(m, torch.nn.Linear)] == [64, 64, N_ACTIONS]
    assert [m.out_features for m in critic.net if isinstance(m, torch.nn.Linear)] == [64, 64, 1]
    assert not set(map(id, actor.parameters())) & set(map(id, critic.parameters()))


def test_policy_head_is_small_so_initial_policy_is_near_uniform() -> None:
    actor = Actor()
    probs = torch.softmax(actor(torch.rand(32, OBS_DIM)), dim=-1)
    assert float((probs - 1.0 / N_ACTIONS).abs().max()) < 0.05


def test_hidden_weights_are_orthogonal() -> None:
    layer = Actor().net[2]
    w = layer.weight.detach() / math.sqrt(2.0)
    assert torch.allclose(w @ w.T, torch.eye(w.shape[0]), atol=1e-4)


def test_actions_are_valid_indices_and_map_through_action_to_mbps() -> None:
    agent = PPOAgent(seed=3)
    gen = torch.Generator().manual_seed(0)
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(200):
        action, log_prob, value = agent.act(rng.random(OBS_DIM, dtype=np.float32), gen)
        assert isinstance(action, int) and 0 <= action < N_ACTIONS
        assert action_to_mbps(action) == float(ACTIONS_MBPS[action])
        assert math.isfinite(log_prob) and log_prob <= 0.0 and math.isfinite(value)
        seen.add(action)
    assert len(seen) > 1  # near-uniform initial policy samples more than one action


# ---------------------------------------------------------------------------
# Truncation versus termination
# ---------------------------------------------------------------------------

_BIAS_MESSAGE = (
    "horizon expiry was treated as termination (or recovery/failure as truncation): the value-bootstrapping "
    "bias this prevents is V(s')=0 at an artificial cutoff, which under-values states leading to long "
    "rebuilds and biases the agent against slow allocations"
)


@pytest.mark.parametrize(
    ("reason", "expect_bootstrap"),
    [
        (TerminationReason.HORIZON_EXPIRED, True),
        (TerminationReason.RECOVERY_COMPLETE, False),
        (TerminationReason.TERMINAL_FAILURE, False),
    ],
)
def test_truncation_bootstraps_and_termination_does_not(reason: TerminationReason, expect_bootstrap: bool) -> None:
    terminated_flag, truncated_flag = _flags(reason)
    terminated = np.array([False, False, terminated_flag])
    truncated = np.array([False, False, truncated_flag])
    values = np.array([1.0, 1.0, 1.0])
    successor_value = 50.0  # V(s') of the true successor of the last step
    truncation_values = np.array([0.0, 0.0, successor_value])
    next_values = resolve_next_values(values, terminated, truncated, truncation_values, last_value=-999.0)

    expected_last = successor_value if expect_bootstrap else 0.0
    assert next_values[-1] == expected_last, _BIAS_MESSAGE
    assert next_values[0] == values[1] and next_values[1] == values[2]

    rewards = np.array([0.0, 0.0, 1.0])
    adv, _ = compute_gae(rewards, values, next_values, terminated, truncated, gamma=0.9, gae_lambda=0.0)
    assert math.isclose(adv[-1], 1.0 + 0.9 * expected_last - 1.0), _BIAS_MESSAGE


def test_final_step_without_episode_end_uses_last_value() -> None:
    values = np.array([1.0, 2.0])
    flags = np.array([False, False])
    nxt = resolve_next_values(values, flags, flags, np.zeros(2), last_value=7.0)
    assert list(nxt) == [2.0, 7.0]


def test_real_rollout_truncation_is_bootstrapped_from_critic(monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end: a horizon-truncated episode from the environment gets V(s') in its advantage."""
    monkeypatch.setattr(recovery_env_module, "MAX_EPISODE_STEPS", 6)
    agent = PPOAgent(seed=0)
    collector = RolloutCollector(v0_config(workload_profile="idle", fleet_degradation=0.0), seed=0)
    batch = collector.collect(agent, 6)
    assert batch.truncated[-1] and not batch.terminated[-1], "horizon expiry must be truncation"
    assert batch.episodes[-1].reason == TerminationReason.HORIZON_EXPIRED.value

    adv, _ = agent.advantages(batch)
    v_next = float(agent.value(batch.next_obs[-1:])[0])
    penalised = agent.penalised_rewards(batch)
    expected = penalised[-1] + agent.config.gamma * v_next - float(batch.values[-1])
    assert math.isclose(adv[-1], expected, rel_tol=1e-6, abs_tol=1e-9), _BIAS_MESSAGE
    assert v_next != 0.0


# ---------------------------------------------------------------------------
# GAE
# ---------------------------------------------------------------------------


def test_gae_matches_hand_computation() -> None:
    rewards = np.array([1.0, 2.0, 3.0])
    values = np.array([0.5, 1.0, 1.5])
    none = np.zeros(3, dtype=bool)
    nxt = resolve_next_values(values, none, none, np.zeros(3), last_value=2.0)
    adv, ret = compute_gae(rewards, values, nxt, none, none, gamma=0.9, gae_lambda=0.5)
    # deltas: 1 + 0.9*1.0 - 0.5, 2 + 0.9*1.5 - 1.0, 3 + 0.9*2.0 - 1.5; decay 0.9*0.5 = 0.45
    d = [1.4, 2.35, 3.3]
    a2 = d[2]
    a1 = d[1] + 0.45 * a2
    a0 = d[0] + 0.45 * a1
    assert np.allclose(adv, [a0, a1, a2])
    assert np.allclose(ret, np.array([a0, a1, a2]) + values)
    assert math.isclose(a0, 3.12575)


def test_gae_recursion_is_cut_at_episode_boundary() -> None:
    rewards = np.array([1.0, 2.0, 3.0])
    values = np.array([0.5, 1.0, 1.5])
    terminated = np.array([False, True, False])
    truncated = np.zeros(3, dtype=bool)
    nxt = resolve_next_values(values, terminated, truncated, np.zeros(3), last_value=2.0)
    adv, _ = compute_gae(rewards, values, nxt, terminated, truncated, gamma=0.9, gae_lambda=0.5)
    assert math.isclose(adv[1], 2.0 - 1.0)  # terminal: no successor value
    assert math.isclose(adv[0], 1.4 + 0.45 * adv[1])  # step 2 is another episode and must not leak into 0 or 1
    assert math.isclose(adv[2], 3.0 + 0.9 * 2.0 - 1.5)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def test_ratio_clipping_binds_when_policy_moves_far() -> None:
    old = torch.zeros(4)
    adv = torch.tensor([1.0, 2.0, -1.0, -2.0])
    new = torch.tensor([1.0, 1.0, -1.5, -1.5], requires_grad=True)  # ratios e^1, e^1, e^-1.5, e^-1.5
    terms = ppo_loss(new, old, adv, torch.zeros(4), torch.zeros(4), torch.ones(4), 0.2, 0.5, 0.0)
    # Each sample is on the clipped branch: A>0 with r>1.2 -> 1.2*A; A<0 with r<0.8 -> 0.8*A.
    expected = -np.mean([1.2 * 1.0, 1.2 * 2.0, 0.8 * -1.0, 0.8 * -2.0])
    assert math.isclose(float(terms.policy_loss), expected, rel_tol=1e-6)
    assert float(terms.clip_fraction) == 1.0
    terms.policy_loss.backward()
    assert torch.count_nonzero(new.grad) == 0  # clipped samples carry no policy gradient


def test_unclipped_samples_carry_gradient_and_loss_is_finite() -> None:
    old = torch.zeros(3)
    new = torch.tensor([0.05, -0.05, 0.0], requires_grad=True)
    adv = torch.tensor([1.0, -1.0, 0.5])
    terms = ppo_loss(new, old, adv, torch.randn(3), torch.randn(3), torch.full((3,), 1.2), 0.2, 0.5, 0.01)
    assert float(terms.clip_fraction) == 0.0
    assert torch.isfinite(terms.total) and torch.isfinite(terms.approx_kl)
    terms.total.backward()
    assert torch.count_nonzero(new.grad) == 3


def test_update_runs_changes_weights_and_stays_finite() -> None:
    agent = PPOAgent(PPOConfig(epochs=2, minibatch_size=64), seed=1)
    batch = RolloutCollector(IDLE, seed=1).collect(agent, 256)
    before = [p.detach().clone() for p in agent.actor.parameters()]
    stats = agent.update(batch)
    assert all(math.isfinite(v) for v in stats.values())
    assert any(not torch.equal(b, p) for b, p in zip(before, agent.actor.parameters()))


def test_entropy_coefficient_anneals_linearly() -> None:
    agent = PPOAgent()
    assert agent.entropy_coef(0.0) == agent.config.ent_coef
    assert math.isclose(agent.entropy_coef(1.0), agent.config.ent_coef_final)
    assert agent.config.gamma == 0.999


# ---------------------------------------------------------------------------
# Multipliers and constraint signals
# ---------------------------------------------------------------------------


def test_multiplier_rises_when_violated_and_decays_when_satisfied() -> None:
    lam = LagrangeMultiplier("latency", limit=0.05, lr=0.5)
    history = [lam.update(0.30) for _ in range(5)]
    assert all(b > a for a, b in zip(history, history[1:])) and history[0] > 0.0
    peak = lam.value
    decayed = [lam.update(0.0) for _ in range(3)]
    assert all(b < a for a, b in zip([peak, *decayed], decayed))
    for _ in range(100):
        lam.update(0.0)
    assert lam.value == 0.0  # projected onto the non-negative half line
    assert math.isclose(LagrangeMultiplier("x", 0.05, 0.5).update(0.30), 0.5 * 0.25)


def test_risk_multiplier_is_normalised_by_the_budget() -> None:
    agent = PPOAgent()
    over = [EpisodeRecord(0.0, 10, 0.0, 2 * RISK_BUDGET_EPS, "recovery_complete", "idle")]
    out = agent.update_multipliers(over)
    assert math.isclose(out["lambda_risk"], agent.config.lr_lambda_risk * 1.0)  # (2 eps - eps) / eps
    under = [EpisodeRecord(0.0, 10, 0.0, 0.0, "recovery_complete", "idle")]
    for _ in range(10):
        agent.update_multipliers(under)
    assert agent.lambda_risk.value == 0.0


def test_both_multipliers_move_independently_from_episode_records() -> None:
    agent = PPOAgent()
    episodes = [EpisodeRecord(-5.0, 100, 0.4, 0.5 * RISK_BUDGET_EPS, "recovery_complete", "peak")]
    for _ in range(3):
        out = agent.update_multipliers(episodes)
    assert out["lambda_latency"] > 0.0  # latency violated
    assert out["lambda_risk"] == 0.0  # reliability satisfied


def test_no_finished_episode_leaves_multipliers_unchanged() -> None:
    agent = PPOAgent()
    agent.lambda_latency.value = 0.7
    out = agent.update_multipliers([])
    assert agent.lambda_latency.value == 0.7 and math.isnan(out["latency_violation_fraction"])


def test_multiplier_bound_clips_and_records_each_binding_update() -> None:
    lam = LagrangeMultiplier("latency", limit=0.05, lr=0.5, max_value=1.0)
    assert lam.update(1.0) == pytest.approx(0.475) and not lam.last_capped
    assert lam.update(1.0) == pytest.approx(0.95) and not lam.last_capped
    assert lam.update(1.0) == 1.0 and lam.last_capped  # 0.95 + 0.475 > 1.0
    assert lam.update(1.0) == 1.0 and lam.last_capped  # stays on the bound
    lam.update(0.0)  # constraint satisfied: decays off the bound
    assert lam.value < 1.0 and not lam.last_capped
    assert lam.n_updates == 5 and lam.n_capped == 2


def test_unbounded_multiplier_never_reports_binding() -> None:
    lam = LagrangeMultiplier("latency", limit=0.05, lr=0.5)
    for _ in range(50):
        lam.update(1.0)
    assert lam.n_capped == 0 and not lam.last_capped and lam.value > 20.0


def test_default_config_bounds_the_multipliers_and_reports_binding() -> None:
    agent = PPOAgent()
    assert math.isfinite(agent.config.lambda_max)
    always_violating = [EpisodeRecord(-5.0, 100, 1.0, 0.0, "recovery_complete", "peak")]
    flags = [agent.update_multipliers(always_violating)["lambda_latency_capped"] for _ in range(20)]
    assert agent.lambda_latency.value == agent.config.lambda_max
    assert flags[0] == 0.0 and flags[-1] == 1.0
    assert agent.update_multipliers([])["lambda_latency_capped"] == 0.0
    clone = PPOAgent()
    clone.load_state(agent.state())
    assert (clone.lambda_latency.n_capped, clone.lambda_latency.n_updates) == (
        agent.lambda_latency.n_capped,
        agent.lambda_latency.n_updates,
    )


def _sim_config_built_by_main(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *flags: str):
    """Run agent.train's CLI far enough to capture the SimConfig it builds, then stop.

    The collector is replaced before it can touch the hazard model, so this works on a
    checkout without hazard/artifacts/v1/: v1_eval_config() constructs lazily and only
    loads the artifacts when the model is first called.
    """
    seen: dict[str, object] = {}

    class _Stop(Exception):
        pass

    def _capture(sim_cfg, **kwargs):
        seen["config"] = sim_cfg
        raise _Stop

    monkeypatch.setattr(train_module, "RolloutCollector", _capture)
    with pytest.raises(_Stop):
        train_module.main(["--iterations", "1", "--out", str(tmp_path), *flags])
    return seen["config"]


def test_cli_eval_fleet_selects_the_held_out_feature_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _sim_config_built_by_main(monkeypatch, tmp_path, "--eval-fleet", "--hazard-error-sigma", "0.25")
    assert cfg.feature_fn is v1_eval_drive_features
    assert cfg.hazard_error_sigma == 0.25
    assert cfg.label == "eval-fleet,sigma=0.25"


def test_cli_without_eval_fleet_keeps_the_train_split_table(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = _sim_config_built_by_main(monkeypatch, tmp_path, "--hazard-error-sigma", "0.5")
    assert cfg.feature_fn is not v1_eval_drive_features
    assert cfg.label == "train-fleet,sigma=0.5"


def test_checkpoint_records_the_fleet_split_and_sigma() -> None:
    meta = _checkpoint_payload(PPOAgent(seed=0), TrainConfig(eval_fleet=True, hazard_error_sigma=0.25), 1)["meta"]
    assert meta["eval_fleet"] is True and meta["hazard_error_sigma"] == 0.25
    assert _checkpoint_payload(PPOAgent(seed=0), TrainConfig(), 1)["meta"]["eval_fleet"] is False


def test_default_rollout_spans_several_default_episodes() -> None:
    # The shortest episode under any allocation is 334 steps (always the largest allocation).
    assert DEFAULT_ROLLOUT_STEPS >= 8 * 334 and TrainConfig().rollout_steps == DEFAULT_ROLLOUT_STEPS


def test_penalties_are_read_from_info_signals_not_the_reward() -> None:
    agent = PPOAgent(PPOConfig(reward_scale=1.0))
    agent.lambda_latency.value = 2.0
    agent.lambda_risk.value = 3.0
    n = 3
    batch = RolloutBatch(
        obs=np.zeros((n, OBS_DIM), np.float32),
        next_obs=np.zeros((n, OBS_DIM), np.float32),
        actions=np.zeros(n, np.int64),
        log_probs=np.zeros(n, np.float32),
        values=np.zeros(n, np.float32),
        rewards=np.array([-1.0, -1.0, -1.0]),  # identical rewards in every step
        terminated=np.zeros(n, bool),
        truncated=np.zeros(n, bool),
        latency_violated=np.array([0.0, 1.0, 0.0], np.float32),
        exposure=np.array([0.0, 0.0, 2.0 * RISK_BUDGET_EPS], np.float32),
    )
    assert np.allclose(agent.penalised_rewards(batch), [-1.0, -1.0 - 2.0, -1.0 - 3.0 * 1.0])


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------


def test_mask_zeroes_exactly_fields_5_and_6() -> None:
    assert HAZARD_INDICES == (5, 6)
    v = np.linspace(0.1, 1.0, OBS_DIM, dtype=np.float32)
    masked = mask_observation(v)
    keep = [i for i in range(OBS_DIM) if i not in (5, 6)]
    assert masked[5] == 0.0 and masked[6] == 0.0
    assert np.array_equal(masked[keep], v[keep])
    assert v[5] != 0.0 and v[6] != 0.0  # input not modified
    assert np.array_equal(mask_observation(v, enabled=False), v)
    batch = np.tile(v, (4, 1))
    assert np.all(mask_observation(batch)[:, [5, 6]] == 0.0)


def test_ablated_env_changes_only_the_observation() -> None:
    raw = RecoveryEnv(IDLE)
    wrapped = AblatedEnv(RecoveryEnv(IDLE), enabled=True)
    obs_r, info_r = raw.reset(seed=5)
    obs_w, info_w = wrapped.reset(seed=5)
    assert obs_r[5] > 0.0  # the underlying environment does carry a hazard
    assert obs_w[5] == 0.0 and obs_w[6] == 0.0
    keep = [i for i in range(OBS_DIM) if i not in (5, 6)]
    assert np.array_equal(obs_r[keep], obs_w[keep]) and info_r == info_w
    for action in (0, 3, 1, 2, 0):
        o_r, r_r, te_r, tr_r, i_r = raw.step(action)
        o_w, r_w, te_w, tr_w, i_w = wrapped.step(action)
        assert (r_r, te_r, tr_r, i_r) == (r_w, te_w, tr_w, i_w)
        assert o_w[5] == 0.0 and o_w[6] == 0.0 and np.array_equal(o_r[keep], o_w[keep])
        assert o_r[5] > 0.0
    assert wrapped.config is wrapped.env.config  # attribute pass-through


def test_disabled_wrapper_is_a_pass_through() -> None:
    a, b = RecoveryEnv(IDLE), AblatedEnv(RecoveryEnv(IDLE), enabled=False)
    assert np.array_equal(a.reset(seed=2)[0], b.reset(seed=2)[0])


def test_ablation_is_a_flag_on_the_same_collector_code() -> None:
    agent = PPOAgent(seed=0)
    on = RolloutCollector(IDLE, seed=0, ablate_hazard=True).collect(agent, 20)
    off = RolloutCollector(IDLE, seed=0, ablate_hazard=False).collect(agent, 20)
    assert np.all(on.obs[:, [5, 6]] == 0.0) and np.any(off.obs[:, [5, 6]] != 0.0)


# ---------------------------------------------------------------------------
# Rollout determinism
# ---------------------------------------------------------------------------


def _same(a: RolloutBatch, b: RolloutBatch) -> bool:
    arrays = ("obs", "next_obs", "actions", "log_probs", "values", "rewards", "terminated", "truncated",
              "latency_violated", "exposure")
    return all(np.array_equal(getattr(a, k), getattr(b, k)) for k in arrays) and a.episodes == b.episodes


def test_same_seed_gives_identical_rollout() -> None:
    first = RolloutCollector(IDLE, seed=11).collect(PPOAgent(seed=2), 300)
    second = RolloutCollector(IDLE, seed=11).collect(PPOAgent(seed=2), 300)
    assert _same(first, second)
    other = RolloutCollector(IDLE, seed=12).collect(PPOAgent(seed=2), 300)
    assert not _same(first, other)


def test_collector_carries_the_running_episode_across_batches() -> None:
    agent = PPOAgent(seed=0)
    one = RolloutCollector(IDLE, seed=4)
    a, b = one.collect(agent, 100), one.collect(agent, 100)
    whole = RolloutCollector(IDLE, seed=4).collect(agent, 200)
    assert np.array_equal(np.concatenate([a.obs, b.obs]), whole.obs)
    assert np.array_equal(np.concatenate([a.actions, b.actions]), whole.actions)


def test_rollout_signals_come_from_info() -> None:
    batch = RolloutCollector(IDLE, seed=0).collect(PPOAgent(seed=0), 50)
    assert set(np.unique(batch.latency_violated)) <= {0.0, 1.0}
    assert np.all((batch.exposure >= 0.0) & (batch.exposure <= 1.0))
    assert batch.exposure.max() > 0.0  # read from info["p_failure_before_complete"]


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def test_checkpoint_write_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "ckpt.pt"
    save_checkpoint(target, {"x": torch.ones(2)})

    def boom(obj: object, handle: object) -> None:
        handle.write(b"partial")  # type: ignore[attr-defined]
        raise RuntimeError("disk full")

    monkeypatch.setattr(torch, "save", boom)
    with pytest.raises(RuntimeError):
        save_checkpoint(target, {"x": torch.zeros(2)})
    assert torch.equal(load_checkpoint(target)["x"], torch.ones(2))  # previous file intact
    assert [p.name for p in tmp_path.iterdir()] == ["ckpt.pt"]  # no stray temp file


# ---------------------------------------------------------------------------
# Policy adapter
# ---------------------------------------------------------------------------


def test_adapter_satisfies_policy_protocol_and_defaults_to_deterministic() -> None:
    policy = LearnedPolicy(Actor())
    assert isinstance(policy, Policy)
    assert policy.deterministic is True and isinstance(policy.name, str)
    obs = np.random.default_rng(0).random(OBS_DIM, dtype=np.float32)
    first = policy(obs)
    assert isinstance(first, int) and 0 <= first < N_ACTIONS
    assert all(policy(obs) == first for _ in range(5))
    policy.reset()


def test_stochastic_mode_samples_and_reset_reseeds() -> None:
    policy = LearnedPolicy(Actor(), deterministic=False, seed=9)
    obs = np.full(OBS_DIM, 0.5, dtype=np.float32)
    run = [policy(obs) for _ in range(40)]
    assert len(set(run)) > 1 and all(0 <= a < N_ACTIONS for a in run)
    policy.reset()
    assert [policy(obs) for _ in range(40)] == run


def test_adapter_applies_the_ablation_mask() -> None:
    actor = Actor()
    base = np.random.default_rng(1).random(OBS_DIM, dtype=np.float32)
    changed = base.copy()
    changed[[5, 6]] = 1.0 - base[[5, 6]]
    with torch.no_grad():
        for p in actor.parameters():
            p.add_(0.3 * torch.randn_like(p))  # make the actor sensitive to its inputs
    ablated = LearnedPolicy(actor, ablate_hazard=True)
    seen_logits = []
    for o in (base, changed):
        x = torch.as_tensor(mask_observation(o, True)).unsqueeze(0)
        seen_logits.append(actor(x))
    assert torch.equal(seen_logits[0], seen_logits[1])
    assert ablated(base) == ablated(changed)


@_needs_v1
def test_adapter_runs_inside_the_paired_harness() -> None:
    policy = LearnedPolicy(Actor(), name="ppo_untrained")
    frame = evaluate(policy, profile="idle", seeds=[0, 1])
    assert list(frame["policy"]) == ["ppo_untrained", "ppo_untrained"]
    assert list(frame["seed"]) == [0, 1]
    assert frame["steps"].min() > 0
    again = evaluate(policy, profile="idle", seeds=[0, 1])
    assert list(again["total_reward"]) == list(frame["total_reward"])  # repeatable, same scenarios


# ---------------------------------------------------------------------------
# Smoke training
# ---------------------------------------------------------------------------


@_needs_v1
def test_smoke_training_runs_logs_and_does_not_degrade_catastrophically(tmp_path: Path) -> None:
    """A few iterations on ``idle``: the loop runs and the log is written.

    This is a smoke test, not evidence of learning. The bound only rules out a
    collapse (mean return more than doubling in magnitude between the first and
    last iterations) in a handful of noisy iterations.
    """
    cfg = TrainConfig(iterations=5, rollout_steps=2048, seed=0, profile="idle", out_dir=str(tmp_path),
                      checkpoint_every=2, ppo=PPOConfig(epochs=2, minibatch_size=256))
    train(cfg, echo=False)

    with open(tmp_path / "log.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5 and tuple(rows[0].keys()) == LOG_COLUMNS
    for row in rows:
        assert math.isfinite(float(row["entropy"])) and math.isfinite(float(row["approx_kl"]))
        assert float(row["lambda_latency"]) >= 0.0 and float(row["lambda_risk"]) >= 0.0
        assert (int(row["n_recovery_complete"]) + int(row["n_terminal_failure"])
                + int(row["n_horizon_expired"])) == int(row["n_episodes"])
    returns = np.array([float(r["mean_return"]) for r in rows])
    assert np.isfinite(returns).any()
    first, last = np.nanmean(returns[:2]), np.nanmean(returns[-2:])
    assert last >= first - abs(first), f"mean return fell from {first:.2f} to {last:.2f}"

    assert (tmp_path / "ckpt_latest.pt").exists() and (tmp_path / "ckpt_final.pt").exists()
    assert not list(tmp_path.glob("*.tmp"))
    restored = LearnedPolicy.from_checkpoint(tmp_path / "ckpt_final.pt")
    assert restored.ablate_hazard is False and restored.name == "ppo_lagrangian"
    assert 0 <= restored(np.zeros(OBS_DIM, dtype=np.float32)) < N_ACTIONS
