"""PPO agent with load-forecast inputs: features, network, policy adapter, training loop and CLI.

Usage::

    python -m agent.forecast_ppo --seeds 0 1 2 --init bc \\
        --workload-csv data_pipeline/profiles/profile_diurnal_long_train.csv \\
        --hazard-error-sigma 0.25 --out runs/forecast_ppo
    python -m agent.forecast_ppo --seeds 0 1 2 --init scratch ...

For each seed, ``<out>/<init>/seed<k>/`` receives ``log.csv``, ``ckpt_init.pt``
(the policy before PPO: the imitation result with ``--init bc``, the random
network with ``--init scratch``), ``ckpt_best.pt`` (best validation gap over
all iterations, including iteration 0) and ``ckpt_final.pt``.

Policy. A neural network chooses the rebuild allocation directly: one of the
four ``ACTIONS_MBPS``, with the actions the safety filter would raise masked
out. No rule runs inside it. It is trained with clipped PPO on the contract
reward (contract coefficients; nothing overridden).

Inputs (``FEATURE_DIM``), all built from observations of the current episode,
the same information ``risk_aware_rule_based`` has:

- the 12 normalised observation fields;
- a load-forecast block (7): availability of the same-time-yesterday
  forecast (a day plus an hour of this episode observed); the next-interval
  forecast, ``fg[t + 1 - DAY] + (level_now - level_one_day_earlier)`` when
  available and the last observed load otherwise; the one-hour median level;
  the last observed load; the spread of the last hour's one-interval changes;
  the spread of this episode's forecast errors; and the change the previous
  day showed over the next interval;
- the risk floor, one-hot (4): the smallest action the safety filter accepts,
  computed from the observation with the filter's own function.

The forecast inputs exist because the 12 contract fields carry no
time-of-day information, while ``profile_diurnal_long`` repeats day to day
(next-interval RMSE 2.0 MB/s from the forecast, 10.1 MB/s from the last
value; see ``agent/SEASONAL_SEARCH_DECISION.md``).

Initialisation. ``--init bc`` first fits the network by behaviour cloning to
the actions of the hand-built ``SeasonalForecastRule`` (neutral parameters)
on the training days, then runs PPO from there with a KL penalty towards the
cloned policy (``kl_coef``). ``--init scratch`` runs PPO from a random
network. The earlier PPO in ``agent/PPO_COLLAPSE_DECISION.md`` started from
scratch and collapsed through action starvation; the warm start is the remedy
tested here, and the scratch arm measures what it contributes.

Reward for learning. The contract reward, plus potential-based shaping
``Phi(s) = -beta * work_remaining / (REF_RATE_MBPS * interval)`` (zero after
completion, no shaping credit on terminal failure), minus
``lambda * violated_latency`` with ``lambda`` updated per iteration from the
violation fraction pooled over the iteration's steps. Discount 0.99.

Validation and fleets. As in :mod:`agent.search_seasonal`: the last
``val_days`` days of the training workload are held back and the policy is
scored against the rule on ``val_seeds`` scenarios of them; training fleets
come from the train-split table; training scenarios cycle through a pool of
``pool_size`` scenario seeds, so each fleet is scored by the risk model once
(:class:`agent.search_seasonal.CachedHazard`).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import math
import os
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from agent.residual_features import DAY_STEPS, LEVEL_WINDOW
from agent.search_seasonal import CachedHazard
from agent.seasonal_rule import SeasonalForecastRule
from agent.train_residual import build_sim_config, split_time_forward
from baselines.evaluate import evaluate
from baselines.policies import RiskAwareRuleBased
from contracts.contracts import BW_TOTAL_MBPS, CONTROL_INTERVAL_S, LATENCY_VIOLATION_FRACTION_MAX, N_ACTIONS, OBS_DIM, \
    Observation, TerminationReason
from sim.config import SimConfig
from sim.recovery_env import RecoveryEnv
from sim.safety_filter import apply_safety_filter

__all__ = [
    "FEATURE_DIM",
    "FEATURE_VERSION",
    "INITS",
    "ForecastFeatures",
    "ForecastActor",
    "ForecastCritic",
    "ForecastPPOPolicy",
    "ForecastPPOConfig",
    "train",
    "main",
]

FEATURE_DIM: int = OBS_DIM + 7 + N_ACTIONS
FEATURE_VERSION: int = 1
INITS: tuple[str, ...] = ("bc", "scratch")
HIDDEN: int = 64
LOAD_SCALE: float = 50.0
#: Rebuild rate assumed by the shaping potential, MB/s: the rule's mean delivered rate on the training days.
REF_RATE_MBPS: float = 80.0
_MASKED: float = -1e9
#: First scenario seed of the training pool; disjoint from evaluation (0..99), tuning (1000..1099),
#: validation (5000..) and the policy-search seeds (100000..).
_POOL_BASE: int = 300_000


class ForecastFeatures:
    """Per-episode history and the feature vector. ``reset`` per episode, ``update`` per observation."""

    def __init__(self) -> None:
        self._fg: list[float] = []
        self._errors: deque[float] = deque(maxlen=DAY_STEPS)
        self._last_forecast: float | None = None

    def reset(self) -> None:
        self._fg = []
        self._errors.clear()
        self._last_forecast = None

    def update(self, obs: np.ndarray) -> tuple[np.ndarray, int]:
        """Consume one observation; return ``(features, risk_floor_index)``."""
        arr = np.asarray(obs, dtype=np.float64)
        if arr.shape != (OBS_DIM,):
            raise ValueError(f"observation must have shape ({OBS_DIM},), got {arr.shape}")
        o = Observation.from_vector(arr, normalized=True)
        fg = self._fg
        fg.append(o.fg_bw)
        if self._last_forecast is not None:
            self._errors.append(o.fg_bw - self._last_forecast)
        t = len(fg) - 1
        now = fg[t]
        recent = fg[max(0, t - LEVEL_WINDOW + 1) : t + 1]
        level = float(np.median(recent))
        steps = np.diff(recent) if len(recent) > 1 else np.zeros(1)
        available = t >= DAY_STEPS + LEVEL_WINDOW - 1
        if available:
            level_then = float(np.median(fg[t - DAY_STEPS - LEVEL_WINDOW + 1 : t - DAY_STEPS + 1]))
            forecast = max(0.0, fg[t + 1 - DAY_STEPS] + level - level_then)
            ahead = (fg[t + 1 - DAY_STEPS] - fg[t - DAY_STEPS]) / LOAD_SCALE
            self._last_forecast = forecast
        else:
            forecast, ahead = now, 0.0
            self._last_forecast = None
        spread = float(np.std(self._errors)) / LOAD_SCALE if len(self._errors) >= 12 else 0.0
        floor = apply_safety_filter(0, o.hazard, o.work_remaining).applied_index
        x = np.zeros(FEATURE_DIM, dtype=np.float32)
        x[:OBS_DIM] = arr
        x[OBS_DIM : OBS_DIM + 7] = (
            float(available),
            forecast / BW_TOTAL_MBPS,
            level / BW_TOTAL_MBPS,
            now / BW_TOTAL_MBPS,
            float(np.std(steps)) / LOAD_SCALE,
            spread,
            ahead,
        )
        x[OBS_DIM + 7 + floor] = 1.0
        return x, int(floor)


def _mlp(out_dim: int, head_gain: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = FEATURE_DIM
    for _ in range(2):
        linear = nn.Linear(width, HIDDEN)
        nn.init.orthogonal_(linear.weight, gain=math.sqrt(2.0))
        nn.init.zeros_(linear.bias)
        layers += [linear, nn.Tanh()]
        width = HIDDEN
    head = nn.Linear(width, out_dim)
    nn.init.orthogonal_(head.weight, gain=head_gain)
    nn.init.zeros_(head.bias)
    layers.append(head)
    return nn.Sequential(*layers)


def _mask(logits: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Mask actions below the risk floor (encoded one-hot in the last ``N_ACTIONS`` features)."""
    floor = x[:, -N_ACTIONS:].argmax(dim=-1, keepdim=True)
    idx = torch.arange(N_ACTIONS, device=x.device).unsqueeze(0)
    return logits.masked_fill(idx < floor, _MASKED)


class ForecastActor(nn.Module):
    """Features ``(B, FEATURE_DIM)`` to masked action logits ``(B, N_ACTIONS)``."""

    def __init__(self) -> None:
        super().__init__()
        self.net: nn.Sequential = _mlp(N_ACTIONS, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _mask(self.net(x), x)


class ForecastCritic(nn.Module):
    """Features ``(B, FEATURE_DIM)`` to values ``(B,)``."""

    def __init__(self) -> None:
        super().__init__()
        self.net: nn.Sequential = _mlp(1, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ForecastPPOPolicy:
    """Trained actor as a ``baselines.policies.Policy``.

    Acts by the argmax of the logits when ``deterministic``; otherwise samples from the softmax of
    the logits with a ``torch.Generator`` seeded by ``seed`` and re-seeded by ``reset``.
    """

    deployable: bool = True

    def __init__(
        self,
        actor: ForecastActor,
        config: SimConfig,
        name: str = "forecast_ppo",
        deterministic: bool = True,
        seed: int = 0,
    ) -> None:
        self.deterministic: bool = deterministic
        self._seed: int = int(seed)
        self._generator: torch.Generator = torch.Generator().manual_seed(self._seed)
        self.actor: ForecastActor = actor.eval()
        self.config: SimConfig = config
        self.name: str = name
        self.features: ForecastFeatures = ForecastFeatures()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | os.PathLike[str],
        config: SimConfig,
        name: str,
        deterministic: bool = True,
        seed: int = 0,
    ) -> "ForecastPPOPolicy":
        """Build from a checkpoint written by :func:`train`.

        Raises:
            ValueError: if the checkpoint's feature version differs from this code's.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if int(ckpt["meta"]["feature_version"]) != FEATURE_VERSION:
            raise ValueError(f"checkpoint feature_version={ckpt['meta']['feature_version']}, code {FEATURE_VERSION}")
        actor = ForecastActor()
        actor.load_state_dict(ckpt["actor"])
        return cls(actor, config, name=name, deterministic=deterministic, seed=seed)

    def reset(self) -> None:
        self.features.reset()
        self._generator.manual_seed(self._seed)

    @torch.no_grad()
    def __call__(self, obs: np.ndarray) -> int:
        x, _ = self.features.update(obs)
        logits = self.actor(torch.as_tensor(x).unsqueeze(0))
        if self.deterministic:
            return int(logits.argmax(dim=-1).item())
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, 1, generator=self._generator).item())


@dataclass(frozen=True)
class ForecastPPOConfig:
    """Run configuration. Every learning hyperparameter is an untuned default."""

    init: str = "bc"
    seed: int = 0
    workload_csv: str = "data_pipeline/profiles/profile_diurnal_long_train.csv"
    hazard: str = "v1"
    hazard_error_sigma: float = 0.25
    out_dir: str = "runs/forecast_ppo/bc/seed0"
    iterations: int = 120
    n_envs: int = 8
    steps_per_env: int = 512
    pool_size: int = 300
    bc_episodes: int = 60
    bc_epochs: int = 40
    discount: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 1e-4
    epochs: int = 4
    minibatch_size: int = 512
    clip_ratio: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.001
    kl_coef: float = 0.1
    max_grad_norm: float = 0.5
    reward_scale: float = 0.1
    lr_lambda: float = 0.05
    lambda_max: float = 10.0
    val_every: int = 10
    val_days: int = 3
    val_seeds: int = 40
    val_seed_start: int = 5000


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _potential(obs: np.ndarray, beta: float) -> float:
    return -beta * Observation.from_vector(obs, normalized=True).work_remaining * 1000.0 / (
        REF_RATE_MBPS * CONTROL_INTERVAL_S
    )


def _behaviour_clone(actor: ForecastActor, sim: SimConfig, fit_csv: str, seeds: list[int], epochs: int,
                     rng: np.random.Generator) -> float:
    """Fit ``actor`` to ``SeasonalForecastRule`` actions on ``seeds``; return training accuracy."""
    teacher = SeasonalForecastRule(sim)
    feats = ForecastFeatures()
    env = RecoveryEnv(dataclasses.replace(sim, workload_csv=fit_csv))
    xs, ys = [], []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        teacher.reset()
        feats.reset()
        while True:
            x, _ = feats.update(obs)
            action = teacher(obs)
            xs.append(x)
            ys.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
    bx, by = torch.as_tensor(np.stack(xs)), torch.as_tensor(np.array(ys))
    optimiser = torch.optim.Adam(actor.parameters(), lr=1e-3)
    for _ in range(epochs):
        perm = torch.as_tensor(rng.permutation(len(bx)))
        for start in range(0, len(bx), 256):
            idx = perm[start : start + 256]
            loss = nn.functional.cross_entropy(actor(bx[idx]), by[idx])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    with torch.no_grad():
        return float((actor(bx).argmax(dim=-1) == by).float().mean())


def _validate(actor: ForecastActor, sim: SimConfig, val_csv: str, seeds: list[int], rule_rewards: np.ndarray
              ) -> tuple[float, float]:
    frame = evaluate(ForecastPPOPolicy(actor, sim, "validation"), profile=Path(val_csv), seeds=seeds, config=sim)
    actor.train()
    diff = frame["total_reward"].to_numpy(dtype=np.float64) - rule_rewards
    sem = float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else float("nan")
    return float(diff.mean()), sem


def _payload(actor: ForecastActor, cfg: ForecastPPOConfig, iteration: int, val: tuple[float, float],
             extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "actor": actor.state_dict(),
        "meta": {
            "iteration": int(iteration),
            "seed": int(cfg.seed),
            "init": cfg.init,
            "feature_version": FEATURE_VERSION,
            "hazard": cfg.hazard,
            "eval_fleet": False,
            "hazard_error_sigma": float(cfg.hazard_error_sigma),
            "val_gap": float(val[0]),
            "val_gap_sem": float(val[1]),
            "val_seeds": f"{cfg.val_seed_start}..{cfg.val_seed_start + cfg.val_seeds - 1}",
            **extra,
            "config": {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in asdict(cfg).items()},
        },
    }


def _save(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


LOG_COLUMNS: tuple[str, ...] = (
    "iteration", "env_steps", "n_episodes", "mean_return", "latency_violation_fraction", "lambda_latency",
    "entropy", "approx_kl", "kl_to_init", "clip_fraction", "value_loss", "action_mean_mbps", "val_gap",
    "val_gap_sem", "elapsed_s",
)


def train(cfg: ForecastPPOConfig, echo: bool = True) -> Path:
    """Run training and return the run directory.

    Raises:
        ValueError: on an unknown ``init`` or non-positive sizes.
    """
    if cfg.init not in INITS:
        raise ValueError(f"init must be one of {INITS}, got {cfg.init!r}")
    if min(cfg.iterations, cfg.n_envs, cfg.steps_per_env, cfg.pool_size, cfg.val_every) < 1:
        raise ValueError("iterations, n_envs, steps_per_env, pool_size and val_every must be >= 1")
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if cfg.val_days > 0:
        fit_csv, val_csv = split_time_forward(cfg.workload_csv, cfg.val_days * DAY_STEPS, out / "split")
    else:
        fit_csv = val_csv = cfg.workload_csv
    base = build_sim_config(cfg.hazard, False, cfg.hazard_error_sigma)
    sim = dataclasses.replace(base, hazard_model=CachedHazard(base.hazard_model))
    beta = float(sim.reward.beta)
    pool = list(range(_POOL_BASE + cfg.seed * 10_000, _POOL_BASE + cfg.seed * 10_000 + cfg.pool_size))
    val_seeds = list(range(cfg.val_seed_start, cfg.val_seed_start + cfg.val_seeds))
    rule_rewards = evaluate(RiskAwareRuleBased(sim), profile=Path(val_csv), seeds=val_seeds,
                            config=sim)["total_reward"].to_numpy(dtype=np.float64)

    actor, critic = ForecastActor(), ForecastCritic()
    bc_accuracy = float("nan")
    if cfg.init == "bc":
        bc_accuracy = _behaviour_clone(actor, sim, fit_csv, pool[: cfg.bc_episodes], cfg.bc_epochs, rng)
    anchor = ForecastActor()
    anchor.load_state_dict(actor.state_dict())
    anchor.eval()
    extra = {
        "trained_on": Path(cfg.workload_csv).as_posix(),
        "trained_on_sha256": _sha256(cfg.workload_csv),
        "fit_csv_sha256": _sha256(fit_csv),
        "val_csv_sha256": _sha256(val_csv),
        "bc_accuracy": bc_accuracy,
    }
    val = _validate(actor, sim, val_csv, val_seeds, rule_rewards)
    _save(out / "ckpt_init.pt", _payload(actor, cfg, 0, val, extra))
    _save(out / "ckpt_best.pt", _payload(actor, cfg, 0, val, extra))
    best_gap = val[0]
    if echo:
        print(f"init={cfg.init} seed={cfg.seed} iteration=0 bc_accuracy={bc_accuracy:.3f} "
              f"val_gap={val[0]:+.2f}+-{val[1]:.2f} elapsed_s={time.monotonic() - started:.0f}", flush=True)

    train_sim = dataclasses.replace(sim, workload_csv=fit_csv)
    envs = [RecoveryEnv(train_sim) for _ in range(cfg.n_envs)]
    feats = [ForecastFeatures() for _ in range(cfg.n_envs)]
    state: list[dict[str, Any]] = []
    for env, ft in zip(envs, feats):
        obs, _ = env.reset(seed=int(rng.choice(pool)))
        ft.reset()
        x, _ = ft.update(obs)
        state.append({"obs": obs, "x": x, "ret": 0.0})
    params = list(actor.parameters()) + list(critic.parameters())
    optimiser = torch.optim.Adam(params, lr=cfg.lr, eps=1e-5)
    lam = 0.0
    ent_coef = cfg.ent_coef if cfg.init == "bc" else max(cfg.ent_coef, 0.01)
    kl_coef = cfg.kl_coef if cfg.init == "bc" else 0.0
    T, N = cfg.steps_per_env, cfg.n_envs
    env_steps = 0
    with open(out / "log.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LOG_COLUMNS), lineterminator="\n")
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
            finished: list[float] = []
            for t in range(T):
                x = np.stack([s["x"] for s in state])
                with torch.no_grad():
                    xt = torch.as_tensor(x)
                    dist = torch.distributions.Categorical(logits=actor(xt))
                    a = dist.sample()
                    xs[t], acts[t], logps[t], vals[t] = x, a.numpy(), dist.log_prob(a).numpy(), critic(xt).numpy()
                for n, (env, ft, s) in enumerate(zip(envs, feats, state)):
                    phi = _potential(s["obs"], beta)
                    obs, reward, terminated, truncated, info = env.step(int(acts[t, n]))
                    s["ret"] += float(reward)
                    reason = info.get("termination_reason")
                    if reason == TerminationReason.TERMINAL_FAILURE.value:
                        shaped = 0.0
                    elif reason == TerminationReason.RECOVERY_COMPLETE.value:
                        shaped = -phi
                    else:
                        shaped = cfg.discount * _potential(obs, beta) - phi
                    rews[t, n] = cfg.reward_scale * (float(reward) + shaped)
                    viols[t, n] = float(bool(info["violated_latency"]))
                    if terminated or truncated:
                        dones[t, n] = 1.0
                        if truncated and not terminated:
                            xb, _ = ft.update(obs)
                            with torch.no_grad():
                                boot[t, n] = float(critic(torch.as_tensor(xb).unsqueeze(0)).item())
                        finished.append(s["ret"])
                        obs, _ = env.reset(seed=int(rng.choice(pool)))
                        ft.reset()
                        s["ret"] = 0.0
                    s["obs"] = obs
                    s["x"], _ = ft.update(obs)
            env_steps += T * N
            pooled = float(viols.mean())
            penalised = rews - cfg.reward_scale * lam * viols
            with torch.no_grad():
                last_v = critic(torch.as_tensor(np.stack([s["x"] for s in state]))).numpy().astype(np.float64)
            adv = np.zeros((T, N), dtype=np.float64)
            gae = np.zeros(N, dtype=np.float64)
            for t in reversed(range(T)):
                next_v = last_v if t == T - 1 else vals[t + 1].astype(np.float64)
                next_v = np.where(dones[t] > 0, boot[t], next_v)
                delta = penalised[t] + cfg.discount * next_v - vals[t]
                gae = delta + cfg.discount * cfg.gae_lambda * (1.0 - dones[t]) * gae
                adv[t] = gae
            returns = adv + vals
            flat = T * N
            bx = torch.as_tensor(xs.reshape(flat, FEATURE_DIM))
            ba = torch.as_tensor(acts.reshape(flat))
            blp = torch.as_tensor(logps.reshape(flat))
            badv = torch.as_tensor(adv.reshape(flat), dtype=torch.float32)
            badv = (badv - badv.mean()) / (badv.std() + 1e-8)
            bret = torch.as_tensor(returns.reshape(flat), dtype=torch.float32)
            stats: dict[str, list[float]] = {k: [] for k in ("entropy", "approx_kl", "kl_to_init", "clip_fraction",
                                                             "value_loss")}
            for _ in range(cfg.epochs):
                perm = torch.as_tensor(rng.permutation(flat))
                for start in range(0, flat, cfg.minibatch_size):
                    idx = perm[start : start + cfg.minibatch_size]
                    logits = actor(bx[idx])
                    dist = torch.distributions.Categorical(logits=logits)
                    lp = dist.log_prob(ba[idx])
                    ratio = torch.exp(lp - blp[idx])
                    pl = -torch.min(ratio * badv[idx],
                                    torch.clamp(ratio, 1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * badv[idx]).mean()
                    vl = 0.5 * ((critic(bx[idx]) - bret[idx]) ** 2).mean()
                    ent = dist.entropy().mean()
                    loss = pl + cfg.vf_coef * vl - ent_coef * ent
                    if kl_coef > 0:
                        with torch.no_grad():
                            anchor_logp = torch.log_softmax(anchor(bx[idx]), dim=-1)
                        logp_all = torch.log_softmax(logits, dim=-1)
                        kl = (logp_all.exp() * (logp_all - anchor_logp)).sum(dim=-1).mean()
                        loss = loss + kl_coef * kl
                        stats["kl_to_init"].append(float(kl.detach()))
                    optimiser.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
                    optimiser.step()
                    with torch.no_grad():
                        stats["approx_kl"].append(float(((ratio - 1) - torch.log(ratio)).mean()))
                        stats["clip_fraction"].append(float(((ratio - 1).abs() > cfg.clip_ratio).float().mean()))
                        stats["entropy"].append(float(ent.detach()))
                        stats["value_loss"].append(float(vl.detach()))
            raw = lam + cfg.lr_lambda * (pooled - LATENCY_VIOLATION_FRACTION_MAX) / LATENCY_VIOLATION_FRACTION_MAX
            lam = float(min(cfg.lambda_max, max(0.0, raw)))
            gap = sem = float("nan")
            if iteration % cfg.val_every == 0 or iteration == cfg.iterations:
                gap, sem = _validate(actor, sim, val_csv, val_seeds, rule_rewards)
                if gap > best_gap:
                    best_gap = gap
                    _save(out / "ckpt_best.pt", _payload(actor, cfg, iteration, (gap, sem), extra))
            mbps = float(np.mean(np.array([20, 50, 80, 120])[acts]))
            row = {
                "iteration": iteration, "env_steps": env_steps, "n_episodes": len(finished),
                "mean_return": float(np.mean(finished)) if finished else float("nan"),
                "latency_violation_fraction": pooled, "lambda_latency": lam,
                **{k: (float(np.mean(v)) if v else float("nan")) for k, v in stats.items()},
                "action_mean_mbps": mbps, "val_gap": gap, "val_gap_sem": sem,
                "elapsed_s": time.monotonic() - started,
            }
            writer.writerow(row)
            handle.flush()
            if echo:
                print(f"init={cfg.init} seed={cfg.seed} iteration={iteration}/{cfg.iterations} steps={env_steps} "
                      f"viol={pooled:.4f} lambda={lam:.3f} entropy={row['entropy']:.3f} "
                      f"kl_to_init={row['kl_to_init']:.4f} mbps={mbps:.1f} val_gap={gap:+.2f}+-{sem:.2f} "
                      f"elapsed_s={row['elapsed_s']:.0f}", flush=True)
    _save(out / "ckpt_final.pt", _payload(actor, cfg, cfg.iterations, (gap, sem), extra))
    if echo:
        print(f"init={cfg.init} seed={cfg.seed} best_val_gap={best_gap:+.3f} wrote {out}")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    d = ForecastPPOConfig()
    p = argparse.ArgumentParser(prog="python -m agent.forecast_ppo", description=__doc__.split("\n\n")[0])
    p.add_argument("--init", choices=list(INITS), default=d.init)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--workload-csv", default=d.workload_csv)
    p.add_argument("--hazard", choices=["v1", "v0"], default=d.hazard,
                   help="v1: fitted model (needs hazard/artifacts/v1); v0: placeholder, not comparable with v1")
    p.add_argument("--hazard-error-sigma", type=float, default=d.hazard_error_sigma)
    p.add_argument("--iterations", type=int, default=d.iterations)
    p.add_argument("--pool-size", type=int, default=d.pool_size)
    p.add_argument("--lr", type=float, default=d.lr)
    p.add_argument("--kl-coef", type=float, default=d.kl_coef)
    p.add_argument("--val-days", type=int, default=d.val_days)
    p.add_argument("--val-seeds", type=int, default=d.val_seeds)
    p.add_argument("--val-every", type=int, default=d.val_every)
    p.add_argument("--out", default="runs/forecast_ppo")
    args = p.parse_args(argv)
    for seed in args.seeds:
        train(ForecastPPOConfig(
            init=args.init, seed=seed, workload_csv=args.workload_csv, hazard=args.hazard,
            hazard_error_sigma=args.hazard_error_sigma, out_dir=os.path.join(args.out, args.init, f"seed{seed}"),
            iterations=args.iterations, pool_size=args.pool_size,
            lr=args.lr if args.init == "bc" else max(args.lr, 3e-4), kl_coef=args.kl_coef,
            val_days=args.val_days, val_seeds=args.val_seeds, val_every=args.val_every,
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
