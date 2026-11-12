"""Training loop and command line for the residual controller.

Usage::

    python -m agent.train_residual --seeds 0 1 2 --iterations 200 \\
        --workload-csv data_pipeline/profiles/profile_diurnal_long_train.csv \\
        --hazard-error-sigma 0.25 --out runs/residual

For each seed, ``<out>/seed<k>/`` receives ``log.csv`` (one row per
iteration), ``ckpt_final.pt`` (the last iteration) and ``ckpt_best.pt`` (the
iteration with the best paired validation gap against ``RiskAwareRuleBased``).

What is learned. A correction to ``RiskAwareRuleBased``: each interval, one
of ``RESIDUAL_DELTAS`` (one action index down, unchanged, one up) is added to
the rule's action (:mod:`agent.residual_features`). The untrained policy is
the rule (:mod:`agent.residual_policy`). The policy reads only observations
from the current episode, the same information the rule has.

Method ``counterfactual`` (default). In this simulator the foreground load is
exogenous (no action changes it) and an interval's latency depends only on
that interval's allocation. So once an interval has executed, the
one-interval outcome of every correction is known exactly: with the
interval's realised load and the simulator's latency model
(``LatencyRule``), each correction is scored as
``-(alpha * D + gamma * V) + progress_value * delivered``, the contract's
latency terms with contract coefficients plus the rule's own price for
rebuild progress (``DEFAULT_PROGRESS_VALUE``, which stands in for the
degraded-interval cost that progress saves later). The network's three
outputs are trained by squared error to the advantage of each correction over
no correction, on a replay buffer of the last ``buffer_iterations`` rollouts,
and the policy takes the correction with the highest predicted advantage.
This is action-value learning on a one-step decision with full counterfactual
feedback from the simulator. Rollouts take a uniformly random correction with
probability ``epsilon``; the states visited depend on the policy only through
the remaining rebuild work. Squared error, not a robust loss: the decision
turns on the mean advantage, where frequent small gains from a faster
allocation are weighed against rare violations costing ``gamma``, and a robust
loss tracks something closer to the median, which ignores those violations.

Method ``ppo``. Clipped PPO on the sampled correction, with GAE over the
environment reward plus potential-based shaping
``Phi(s) = -beta * work_remaining / (ref_rate_mbps * interval)``. Kept for
comparison. On the training half of ``profile_diurnal_long`` it did not move
the deterministic policy away from the rule by a useful margin in 300
iterations: random exploration of the faster correction under high load
draws violation penalties, its probability falls in every state, and the
gradient that would raise it again in the states where it is right is
proportional to that probability.

Latency constraint. Both methods carry a multiplier ``lambda`` on
``violated_latency``, updated once per iteration by projected gradient ascent
against ``LATENCY_VIOLATION_FRACTION_MAX``. The measurement is pooled over
steps (violating steps over all steps, not a mean of per-episode fractions,
which over-weights short episodes). With ``counterfactual`` it is the
violation fraction of the greedy policy's own choices on the iteration's
states, read from the counterfactual outcomes, so exploration does not
inflate it; with ``ppo`` it is the sampled policy's.

Fleet. The default draws training fleets from the train-split feature table
(``SimConfig``). ``--eval-fleet`` would draw them from the held-out table that
evaluation uses, and is refused unless ``--allow-eval-fleet-training`` is also
given.

Validation. The training workload is cut time-forward: its last
``val_days`` days are held back for validation and training sees only the
earlier part (both parts are written under ``<out>/split/``). Every
``--val-every`` iterations the deterministic policy and the rule are run on
``--val-seeds`` scenario seeds starting at ``--val-seed-start`` of the
validation part, and ``ckpt_best.pt`` keeps the iteration with the best paired
gap. Validating on later days, not on the same days from other start rows, is
what makes the gap measure forecasting rather than recall of the training
days: with ``val_days=0`` (validation on the training days themselves) three
seeds reached validation gaps of +9.2 to +14.5 and then scored -5.9 to +3.5 on
the held-out days. The validation seeds are disjoint from the report's
evaluation seeds, which start at 0, and from the rule's tuning seeds 1000 to
1099. Nothing from the evaluation workload is used. ``val_gap`` of
``ckpt_best.pt`` is the maximum over validations and is biased upward by that
selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agent.residual_features import (
    DAY_STEPS,
    FEATURE_DIM,
    FEATURE_VERSION,
    N_RESIDUAL,
    NO_CHANGE_INDEX,
    RESIDUAL_DELTAS,
    ResidualFeatures,
    compose_action,
)
from agent.residual_policy import ResidualActor, ResidualCritic, ResidualPolicy
from baselines.evaluate import evaluate
from baselines.policies import DEFAULT_PROGRESS_VALUE, RiskAwareRuleBased
from contracts.contracts import (
    ACTIONS_MBPS,
    CONTROL_INTERVAL_S,
    LATENCY_VIOLATION_FRACTION_MAX,
    Observation,
    TerminationReason,
)
from sim.config import SimConfig, v0_config, v1_eval_config
from sim.recovery_env import RecoveryEnv

__all__ = ["METHODS", "ResidualTrainConfig", "LOG_COLUMNS", "REF_RATE_MBPS", "build_sim_config", "train", "main"]

#: Default rebuild rate assumed by the shaping potential, MB/s. It sets the value
#: the shaped reward assigns to one MB/s of progress for one interval,
#: ``beta / REF_RATE_MBPS``. 50 MB/s equals ``0.5 * DEFAULT_PROGRESS_VALUE`` of
#: ``baselines.policies``, the best constant progress value for the rule on
#: the training half of ``profile_diurnal_long`` (40 tuning seeds from 1000,
#: grid 0.33x to 3x; the optimum is flat between 0.33x and 1x). With a discount
#: below 1 the shaping is not exactly policy-invariant, so this value biases
#: the learned corrections; 80 MB/s, the rule's mean delivered rate, produced a
#: policy that slowed the rebuild in every state.
REF_RATE_MBPS: float = 50.0

#: Training methods: ``counterfactual`` (default) and ``ppo``. See the module docstring.
METHODS: tuple[str, ...] = ("counterfactual", "ppo")

LOG_COLUMNS: tuple[str, ...] = (
    "iteration",
    "env_steps",
    "n_episodes",
    "mean_return",
    "mean_episode_length",
    "latency_violation_fraction",
    "constraint_violation_fraction",
    "lambda_latency",
    "frac_changed",
    "frac_down",
    "frac_up",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "policy_loss",
    "value_loss",
    "n_terminal_failure",
    "val_gap",
    "val_gap_sem",
    "val_frac_changed",
    "elapsed_s",
)


@dataclass(frozen=True)
class ResidualTrainConfig:
    """Run configuration. Every learning hyperparameter is an untuned default."""

    method: str = "counterfactual"
    iterations: int = 300
    n_envs: int = 8
    steps_per_env: int = 512
    seed: int = 0
    workload_csv: str = "data_pipeline/profiles/profile_diurnal_long_train.csv"
    hazard: str = "v1"
    eval_fleet: bool = False
    hazard_error_sigma: float = 0.25
    out_dir: str = "runs/residual/seed0"
    discount: float = 0.9
    gae_lambda: float = 0.9
    lr: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 1024
    clip_ratio: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    reward_scale: float = 0.1
    ref_rate_mbps: float = REF_RATE_MBPS
    progress_value: float = DEFAULT_PROGRESS_VALUE
    cf_lr: float = 1e-3
    epsilon: float = 0.1
    buffer_iterations: int = 25
    cf_epochs: int = 2
    cf_minibatch_size: int = 256
    latency_limit: float = LATENCY_VIOLATION_FRACTION_MAX
    lr_lambda: float = 0.05
    lambda_max: float = 10.0
    val_every: int = 10
    val_seeds: int = 40
    val_seed_start: int = 5000
    val_days: int = 3
    weight_decay: float = 0.0


def build_sim_config(hazard: str, eval_fleet: bool, sigma: float, workload_csv: str | None = None) -> SimConfig:
    """``SimConfig`` for the given risk model and fleet table.

    ``hazard="v1"`` is the simulator default (fitted model; needs
    ``hazard/artifacts/v1/``). ``hazard="v0"`` is the unfitted placeholder,
    for machines without the artifacts; its results are not comparable with v1
    results.

    Raises:
        ValueError: on an unknown ``hazard``, or ``eval_fleet`` with v0 (v0 has one fleet generator).
    """
    fleet = "eval-fleet" if eval_fleet else "train-fleet"
    fields: dict[str, Any] = {"hazard_error_sigma": float(sigma), "label": f"{hazard},{fleet},sigma={sigma:g}"}
    if workload_csv is not None:
        fields["workload_csv"] = workload_csv
    if hazard == "v1":
        return v1_eval_config(**fields) if eval_fleet else SimConfig(**fields)
    if hazard == "v0":
        if eval_fleet:
            raise ValueError("eval_fleet has no meaning with hazard='v0'")
        return v0_config(**fields)
    raise ValueError(f"hazard must be 'v1' or 'v0', got {hazard!r}")


def split_time_forward(csv_path: str, val_rows: int, out_dir: Path) -> tuple[str, str]:
    """Write the first rows and the last ``val_rows`` rows of a replay CSV as two files.

    Rows are copied byte for byte, with the same line terminator as the source;
    the validation part's ``t_seconds`` is rebased to start at 0. A sidecar is
    written beside each part from the source's sidecar (``profile`` suffixed
    ``_fit`` / ``_val``, ``rows`` updated). Returns ``(fit_path, val_path)``.

    Raises:
        ValueError: if ``val_rows`` leaves fewer than ``val_rows`` rows to fit on.
    """
    raw = Path(csv_path).read_bytes()
    newline = b"\r\n" if b"\r\n" in raw else b"\n"
    lines = [line for line in raw.split(newline) if line]
    header, body = lines[0], lines[1:]
    if not 0 < val_rows <= len(body) - val_rows:
        raise ValueError(f"val_rows={val_rows} does not fit a trace of {len(body)} rows")
    fit_rows, val_part = body[:-val_rows], body[-val_rows:]
    t0 = int(val_part[0].split(b",")[0])
    val_part = [b",".join([str(int(r.split(b",")[0]) - t0).encode()] + r.split(b",")[1:]) for r in val_part]
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(csv_path).stem
    sidecar_path = Path(csv_path).with_suffix(".meta.json")
    try:
        meta = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            meta = {}
    except (OSError, ValueError):
        meta = {}
    paths = []
    for suffix, rows in (("fit", fit_rows), ("val", val_part)):
        path = out_dir / f"{stem}_{suffix}.csv"
        path.write_bytes(newline.join([header, *rows]) + newline)
        part_meta = dict(meta)
        part_meta["profile"] = f"{meta.get('profile', stem)}_{suffix}"
        part_meta["rows"] = len(rows)
        part_meta["split_from"] = Path(csv_path).as_posix()
        path.with_suffix(".meta.json").write_text(json.dumps(part_meta, indent=2), encoding="utf-8")
        paths.append(str(path))
    return paths[0], paths[1]


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _potential(obs: np.ndarray, beta: float, ref_rate_mbps: float) -> float:
    work_gb = Observation.from_vector(obs, normalized=True).work_remaining
    return -beta * work_gb * 1000.0 / (ref_rate_mbps * CONTROL_INTERVAL_S)


@dataclass
class _Lane:
    """One environment and its per-episode state."""

    env: RecoveryEnv
    feats: ResidualFeatures
    obs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    x: np.ndarray = field(default_factory=lambda: np.zeros(0))
    rule_action: int = 0
    floor: int = 0
    ep_return: float = 0.0
    ep_len: int = 0

    def start(self, seed: int | None) -> None:
        self.obs, _ = self.env.reset(seed=seed)
        self.feats.reset()
        self._observe()
        self.ep_return, self.ep_len = 0.0, 0

    def _observe(self) -> None:
        self.x, step = self.feats.update(self.obs)
        self.rule_action, self.floor = step.action, step.floor


class _Multiplier:
    def __init__(self, lr: float, limit: float, lam_max: float) -> None:
        self.value, self.lr, self.limit, self.lam_max = 0.0, lr, limit, lam_max
        self.n_capped = 0

    def update(self, measured: float) -> None:
        raw = self.value + self.lr * (measured - self.limit) / self.limit
        self.n_capped += int(raw > self.lam_max)
        self.value = float(min(self.lam_max, max(0.0, raw)))


def _validate(actor: ResidualActor, cfg: ResidualTrainConfig, rule_rewards: np.ndarray,
              val_csv: str) -> tuple[float, float, float]:
    sim = build_sim_config(cfg.hazard, cfg.eval_fleet, cfg.hazard_error_sigma)
    policy = ResidualPolicy(actor, sim, name="residual_val")
    seeds = list(range(cfg.val_seed_start, cfg.val_seed_start + cfg.val_seeds))
    frame = evaluate(policy, profile=Path(val_csv), seeds=seeds, config=sim)
    actor.train()
    diff = frame["total_reward"].to_numpy(dtype=np.float64) - rule_rewards
    sem = float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else float("nan")
    changed = policy.n_changed / max(1, policy.n_steps)
    return float(diff.mean()), sem, float(changed)


def _payload(actor: ResidualActor, critic: ResidualCritic, cfg: ResidualTrainConfig, iteration: int,
             trained_sha: str, val: tuple[float, float], fit_csv: str, val_csv: str) -> dict[str, Any]:
    return {
        "actor": actor.state_dict(),
        "critic": critic.state_dict(),
        "meta": {
            "iteration": int(iteration),
            "seed": int(cfg.seed),
            "method": cfg.method,
            "feature_version": int(FEATURE_VERSION),
            "residual_deltas": ",".join(str(d) for d in RESIDUAL_DELTAS),
            "trained_on": str(cfg.workload_csv),
            "trained_on_sha256": trained_sha,
            "hazard": cfg.hazard,
            "eval_fleet": bool(cfg.eval_fleet),
            "hazard_error_sigma": float(cfg.hazard_error_sigma),
            "val_gap": float(val[0]),
            "val_gap_sem": float(val[1]),
            "val_seeds": f"{cfg.val_seed_start}..{cfg.val_seed_start + cfg.val_seeds - 1}",
            "fit_csv": Path(fit_csv).as_posix(),
            "fit_csv_sha256": _sha256(fit_csv),
            "val_csv": Path(val_csv).as_posix(),
            "val_csv_sha256": _sha256(val_csv),
            "config": {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in asdict(cfg).items()},
        },
    }


def _save(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def _counterfactual_rewards(lane: _Lane, fg_mbps: float, rule: RiskAwareRuleBased,
                            progress_value: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-correction one-interval objective and violation flag, at the interval's realised load.

    For each correction ``k`` the action ``compose_action(rule_action, floor, k)`` is
    scored as ``-(alpha * D + gamma * V) + progress_value * delivered`` with the
    simulator's latency model (``LatencyRule``, tested against the environment)
    and the foreground load the interval actually had. That load is exogenous:
    no action changes it. The rebuild-progress term stands in for the
    degraded-interval cost the progress saves later, priced as the rule prices it.
    """
    reward_cfg = rule.config.reward
    target = rule.config.latency_target_ms
    limit = rule.latency_rule.limit_ms
    values = np.zeros(N_RESIDUAL, dtype=np.float64)
    violations = np.zeros(N_RESIDUAL, dtype=np.float64)
    for k in range(N_RESIDUAL):
        mbps = float(ACTIONS_MBPS[compose_action(lane.rule_action, lane.floor, k)])
        latency = float(rule.latency_rule.predicted_latency_ms(fg_mbps, mbps))
        degradation = max(0.0, (latency - target) / target)
        violated = float(latency > limit)
        delivered = float(rule.latency_rule.delivered_mbps(fg_mbps, mbps))
        values[k] = -(reward_cfg.alpha * degradation + reward_cfg.gamma * violated) + progress_value * delivered
        violations[k] = violated
    return values, violations


def train(cfg: ResidualTrainConfig, echo: bool = True) -> Path:
    """Run training and return the run directory."""
    if cfg.iterations < 1 or cfg.n_envs < 1 or cfg.steps_per_env < 1 or cfg.val_every < 1:
        raise ValueError("iterations, n_envs, steps_per_env and val_every must be >= 1")
    if cfg.method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {cfg.method!r}")
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    trained_sha = _sha256(cfg.workload_csv)
    if cfg.val_days > 0:
        fit_csv, val_csv = split_time_forward(cfg.workload_csv, cfg.val_days * DAY_STEPS, out / "split")
    else:
        fit_csv = val_csv = cfg.workload_csv
    sim = build_sim_config(cfg.hazard, cfg.eval_fleet, cfg.hazard_error_sigma, fit_csv)
    beta = float(sim.reward.beta)

    lanes = [_Lane(RecoveryEnv(sim), ResidualFeatures(RiskAwareRuleBased(sim))) for _ in range(cfg.n_envs)]
    for i, lane in enumerate(lanes):
        lane.start(seed=cfg.seed * 1_000_003 + 7919 * (i + 1))

    actor, critic = ResidualActor(), ResidualCritic()
    params = list(actor.parameters()) + (list(critic.parameters()) if cfg.method == "ppo" else [])
    optimiser = torch.optim.AdamW(params, lr=cfg.lr if cfg.method == "ppo" else cfg.cf_lr, eps=1e-5,
                                  weight_decay=cfg.weight_decay)
    buffer: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    multiplier = _Multiplier(cfg.lr_lambda, cfg.latency_limit, cfg.lambda_max)

    val_eval_sim = build_sim_config(cfg.hazard, cfg.eval_fleet, cfg.hazard_error_sigma)
    val_seed_list = list(range(cfg.val_seed_start, cfg.val_seed_start + cfg.val_seeds))
    rule_frame = evaluate(RiskAwareRuleBased(val_eval_sim), profile=Path(val_csv), seeds=val_seed_list,
                          config=val_eval_sim)
    rule_rewards = rule_frame["total_reward"].to_numpy(dtype=np.float64)
    best_gap = -math.inf
    val = (float("nan"), float("nan"))

    T, N = cfg.steps_per_env, cfg.n_envs
    started = time.monotonic()
    env_steps = 0
    with open(out / "log.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LOG_COLUMNS))
        writer.writeheader()
        for iteration in range(1, cfg.iterations + 1):
            xs = np.zeros((T, N, FEATURE_DIM), dtype=np.float32)
            acts = np.zeros((T, N), dtype=np.int64)
            logps = np.zeros((T, N), dtype=np.float32)
            vals = np.zeros((T, N), dtype=np.float32)
            rews = np.zeros((T, N), dtype=np.float64)
            viols = np.zeros((T, N), dtype=np.float64)
            dones = np.zeros((T, N), dtype=np.float64)
            boot = np.zeros((T, N), dtype=np.float64)
            cf_values = np.zeros((T, N, N_RESIDUAL), dtype=np.float64)
            cf_viols = np.zeros((T, N, N_RESIDUAL), dtype=np.float64)
            changed = down = up = 0
            finished: list[tuple[float, int]] = []
            n_failure = 0
            for t in range(T):
                x = np.stack([lane.x for lane in lanes])
                with torch.no_grad():
                    xt = torch.as_tensor(x)
                    out_t = actor(xt)
                    if cfg.method == "ppo":
                        dist = torch.distributions.Categorical(logits=out_t)
                        a = dist.sample()
                        lp = dist.log_prob(a)
                        v = critic(xt)
                    else:
                        explore = torch.as_tensor(rng.random(N) < cfg.epsilon)
                        a = torch.where(explore, torch.as_tensor(rng.integers(0, N_RESIDUAL, N)), out_t.argmax(dim=-1))
                        lp = torch.zeros(N)
                        v = torch.zeros(N)
                xs[t], acts[t], logps[t], vals[t] = x, a.numpy(), lp.numpy(), v.numpy()
                for n, lane in enumerate(lanes):
                    action = compose_action(lane.rule_action, lane.floor, int(acts[t, n]))
                    changed += int(action != lane.rule_action)
                    down += int(action < lane.rule_action)
                    up += int(action > lane.rule_action)
                    phi = _potential(lane.obs, beta, cfg.ref_rate_mbps)
                    next_obs, reward, terminated, truncated, info = lane.env.step(action)
                    cf_values[t, n], cf_viols[t, n] = _counterfactual_rewards(
                        lane, float(info["fg_bw_mbps"]), lane.feats.rule, cfg.progress_value
                    )
                    lane.ep_return += float(reward)
                    lane.ep_len += 1
                    viols[t, n] = float(bool(info["violated_latency"]))
                    reason = info.get("termination_reason")
                    if reason == TerminationReason.TERMINAL_FAILURE.value:
                        shaped = 0.0
                        n_failure += 1
                    elif reason == TerminationReason.RECOVERY_COMPLETE.value:
                        shaped = -phi
                    else:
                        shaped = cfg.discount * _potential(next_obs, beta, cfg.ref_rate_mbps) - phi
                    rews[t, n] = cfg.reward_scale * (float(reward) + shaped)
                    if terminated or truncated:
                        dones[t, n] = 1.0
                        if truncated and not terminated and cfg.method == "ppo":
                            lane.obs = next_obs
                            lane._observe()
                            with torch.no_grad():
                                boot[t, n] = float(critic(torch.as_tensor(lane.x).unsqueeze(0)).item())
                        finished.append((lane.ep_return, lane.ep_len))
                        lane.start(seed=None)
                    else:
                        lane.obs = next_obs
                        lane._observe()
            env_steps += T * N
            pooled_violation = float(viols.mean())
            constraint_measure = pooled_violation
            flat = T * N
            bx = torch.as_tensor(xs.reshape(flat, FEATURE_DIM))
            stats: dict[str, list[float]] = {
                "entropy": [], "approx_kl": [], "clip_fraction": [], "policy_loss": [], "value_loss": []
            }

            if cfg.method == "counterfactual":
                buffer.append((xs.reshape(flat, FEATURE_DIM), cf_values.reshape(flat, N_RESIDUAL),
                               cf_viols.reshape(flat, N_RESIDUAL)))
                del buffer[: max(0, len(buffer) - cfg.buffer_iterations)]
                bx_all = torch.as_tensor(np.concatenate([b[0] for b in buffer]))
                values = np.concatenate([b[1] for b in buffer])
                violations = np.concatenate([b[2] for b in buffer])
                penalised = values - multiplier.value * violations
                target = torch.as_tensor(penalised - penalised[:, [NO_CHANGE_INDEX]], dtype=torch.float32)
                n_all = len(target)
                for _ in range(cfg.cf_epochs):
                    perm = torch.as_tensor(rng.permutation(n_all))
                    for start in range(0, n_all, cfg.cf_minibatch_size):
                        idx = perm[start : start + cfg.cf_minibatch_size]
                        pred = actor(bx_all[idx])
                        loss = torch.nn.functional.mse_loss(pred, target[idx])
                        optimiser.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                        optimiser.step()
                        stats["value_loss"].append(float(loss.detach()))
                with torch.no_grad():
                    greedy = actor(bx).argmax(dim=-1).numpy()
                    chosen = target[-flat:][torch.arange(flat), torch.as_tensor(greedy)]
                constraint_measure = float(cf_viols.reshape(flat, N_RESIDUAL)[np.arange(flat), greedy].mean())
                stats["policy_loss"] = [float(-chosen.mean())]
                stats["entropy"] = [float("nan")]
                stats["approx_kl"] = [float("nan")]
                stats["clip_fraction"] = [float("nan")]
            else:
                rews_pen = rews - cfg.reward_scale * multiplier.value * viols
                with torch.no_grad():
                    last_v = critic(torch.as_tensor(np.stack([lane.x for lane in lanes]))).numpy().astype(np.float64)
                adv = np.zeros((T, N), dtype=np.float64)
                gae = np.zeros(N, dtype=np.float64)
                for t in reversed(range(T)):
                    next_v = last_v if t == T - 1 else vals[t + 1].astype(np.float64)
                    next_v = np.where(dones[t] > 0, boot[t], next_v)
                    delta = rews_pen[t] + cfg.discount * next_v - vals[t]
                    gae = delta + cfg.discount * cfg.gae_lambda * (1.0 - dones[t]) * gae
                    adv[t] = gae
                returns = adv + vals
                ba = torch.as_tensor(acts.reshape(flat))
                blp = torch.as_tensor(logps.reshape(flat))
                badv = torch.as_tensor(adv.reshape(flat), dtype=torch.float32)
                badv = (badv - badv.mean()) / (badv.std() + 1e-8)
                bret = torch.as_tensor(returns.reshape(flat), dtype=torch.float32)
                for _ in range(cfg.epochs):
                    perm = torch.as_tensor(rng.permutation(flat))
                    epoch_kl = []
                    for start in range(0, flat, cfg.minibatch_size):
                        idx = perm[start : start + cfg.minibatch_size]
                        dist = torch.distributions.Categorical(logits=actor(bx[idx]))
                        lp = dist.log_prob(ba[idx])
                        ratio = torch.exp(lp - blp[idx])
                        s1 = ratio * badv[idx]
                        s2 = torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * badv[idx]
                        pl = -torch.min(s1, s2).mean()
                        vl = 0.5 * ((critic(bx[idx]) - bret[idx]) ** 2).mean()
                        ent = dist.entropy().mean()
                        loss = pl + cfg.vf_coef * vl - cfg.ent_coef * ent
                        optimiser.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                        optimiser.step()
                        with torch.no_grad():
                            kl = float(((ratio - 1) - torch.log(ratio)).mean())
                            epoch_kl.append(kl)
                            stats["approx_kl"].append(kl)
                            stats["clip_fraction"].append(float(((ratio - 1).abs() > cfg.clip_ratio).float().mean()))
                            stats["entropy"].append(float(ent))
                            stats["policy_loss"].append(float(pl))
                            stats["value_loss"].append(float(vl))
                    if cfg.target_kl > 0 and float(np.mean(epoch_kl)) > 1.5 * cfg.target_kl:
                        break
            multiplier.update(constraint_measure)

            if iteration % cfg.val_every == 0 or iteration == cfg.iterations:
                gap, sem, val_changed = _validate(actor, cfg, rule_rewards, val_csv)
                val = (gap, sem)
                if gap > best_gap:
                    best_gap = gap
                    _save(out / "ckpt_best.pt", _payload(actor, critic, cfg, iteration, trained_sha, val, fit_csv, val_csv))
            else:
                gap = sem = val_changed = float("nan")

            row = {
                "iteration": iteration,
                "env_steps": env_steps,
                "n_episodes": len(finished),
                "mean_return": float(np.mean([r for r, _ in finished])) if finished else float("nan"),
                "mean_episode_length": float(np.mean([n for _, n in finished])) if finished else float("nan"),
                "latency_violation_fraction": pooled_violation,
                "constraint_violation_fraction": constraint_measure,
                "lambda_latency": multiplier.value,
                "frac_changed": changed / (T * N),
                "frac_down": down / (T * N),
                "frac_up": up / (T * N),
                **{k: float(np.mean(v)) if v else float("nan") for k, v in stats.items()},
                "n_terminal_failure": n_failure,
                "val_gap": gap,
                "val_gap_sem": sem,
                "val_frac_changed": val_changed,
                "elapsed_s": time.monotonic() - started,
            }
            writer.writerow(row)
            handle.flush()
            if echo:
                print(
                    f"seed={cfg.seed} method={cfg.method} iter={iteration}/{cfg.iterations} steps={env_steps} "
                    f"episodes={row['n_episodes']} mean_return={row['mean_return']:.2f} "
                    f"viol_pooled={pooled_violation:.4f} lambda={multiplier.value:.3f} "
                    f"changed={row['frac_changed']:.3f} down={row['frac_down']:.3f} up={row['frac_up']:.3f} "
                    f"entropy={row['entropy']:.3f} kl={row['approx_kl']:.4f} "
                    f"val_gap={gap:+.2f}+-{sem:.2f} elapsed_s={row['elapsed_s']:.0f}",
                    flush=True,
                )
    _save(out / "ckpt_final.pt", _payload(actor, critic, cfg, cfg.iterations, trained_sha, val, fit_csv, val_csv))
    if echo:
        print(f"seed={cfg.seed} best_val_gap={best_gap:+.3f} lambda_capped={multiplier.n_capped}/{cfg.iterations} "
              f"wrote {out}")
    return out


def _parser() -> argparse.ArgumentParser:
    d = ResidualTrainConfig()
    p = argparse.ArgumentParser(prog="python -m agent.train_residual", description=__doc__.split("\n\n")[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--method", choices=list(METHODS), default=d.method)
    p.add_argument("--iterations", type=int, default=d.iterations)
    p.add_argument("--n-envs", type=int, default=d.n_envs)
    p.add_argument("--steps-per-env", type=int, default=d.steps_per_env)
    p.add_argument("--workload-csv", default=d.workload_csv)
    p.add_argument("--hazard", choices=["v1", "v0"], default=d.hazard,
                   help="v1: fitted model (needs hazard/artifacts/v1); v0: placeholder, not comparable with v1")
    p.add_argument("--eval-fleet", action="store_true", help="train on the held-out fleet table (refused by default)")
    p.add_argument("--allow-eval-fleet-training", action="store_true")
    p.add_argument("--hazard-error-sigma", type=float, default=d.hazard_error_sigma)
    p.add_argument("--out", default="runs/residual")
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--ent-coef", type=float, default=d.ent_coef)
    p.add_argument("--discount", type=float, default=d.discount)
    p.add_argument("--ref-rate-mbps", type=float, default=d.ref_rate_mbps, help="shaping potential rate")
    p.add_argument("--val-every", type=int, default=d.val_every)
    p.add_argument("--val-seeds", type=int, default=d.val_seeds)
    p.add_argument("--val-seed-start", type=int, default=d.val_seed_start)
    p.add_argument("--val-days", type=int, default=d.val_days,
                   help="last days of the training CSV held back for validation; 0 validates on the training days")
    p.add_argument("--weight-decay", type=float, default=d.weight_decay)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.eval_fleet and not args.allow_eval_fleet_training:
        parser.error("--eval-fleet trains on the held-out fleet table; add --allow-eval-fleet-training to do so")
    if args.val_seed_start < 100 and args.val_seed_start + args.val_seeds > 0:
        parser.error("--val-seed-start overlaps the evaluation seeds 0..99")
    for seed in args.seeds:
        cfg = ResidualTrainConfig(
            method=args.method,
            iterations=args.iterations,
            n_envs=args.n_envs,
            steps_per_env=args.steps_per_env,
            seed=seed,
            workload_csv=args.workload_csv,
            hazard=args.hazard,
            eval_fleet=args.eval_fleet,
            hazard_error_sigma=args.hazard_error_sigma,
            out_dir=os.path.join(args.out, f"seed{seed}"),
            lr=args.lr,
            ent_coef=args.ent_coef,
            discount=args.discount,
            ref_rate_mbps=args.ref_rate_mbps,
            val_every=args.val_every,
            val_seeds=args.val_seeds,
            val_seed_start=args.val_seed_start,
            val_days=args.val_days,
            weight_decay=args.weight_decay,
        )
        train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
