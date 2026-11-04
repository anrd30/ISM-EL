"""Serial rollout collection against ``RecoveryEnv``.

One environment is stepped for a fixed number of control intervals per batch.
Episodes that end inside the batch are reset and collection continues; an
episode still running when the batch fills carries over into the next call, so
no step is discarded and no bootstrap is faked at an arbitrary cut.

Stored per step: observation, next observation, action, log-probability,
value estimate, reward, the contract's ``(terminated, truncated)`` flags, and
the two constraint signals read from ``info``: ``violated_latency`` and the
exposure ``p_failure_before_complete``. The constraint signals come from
``info`` and are never inferred from the reward, because the contract makes
``info`` the authoritative constraint record.

Reproducibility. Action sampling uses a ``torch.Generator`` owned by the
collector and each episode is reset with a seed derived from ``(seed,
episode_index)``. Two collectors built with the same seed, configuration and
agent weights produce identical batches.

The next observation is stored for every step, not only for the last one. At a
truncated step the learner needs ``V(s')`` for the true successor state, which
is the observation the environment returned before the collector reset it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import torch

from agent.ablation import AblatedEnv
from contracts.contracts import OBS_DIM, TerminationReason
from sim.config import SimConfig
from sim.recovery_env import RecoveryEnv

__all__ = [
    "RELIABILITY_KEY",
    "EpisodeRecord",
    "RolloutBatch",
    "RolloutCollector",
    "Acting",
    "episode_seed",
]

#: ``info`` key read as the per-step exposure signal.
RELIABILITY_KEY: str = "p_failure_before_complete"

_SEED_BOUND: int = 2**31 - 1


class Acting(Protocol):
    """What the collector needs from an agent."""

    def act(
        self, obs: np.ndarray, generator: torch.Generator | None = None, deterministic: bool = False
    ) -> tuple[int, float, float]:
        """Return ``(action, log_prob, value)`` for one observation."""
        ...


def episode_seed(seed: int, index: int) -> int:
    """Environment seed for the ``index``-th episode of a collector seeded ``seed``."""
    state = np.random.SeedSequence([int(seed), int(index)]).generate_state(1, dtype=np.uint32)
    return int(state[0]) % _SEED_BOUND


@dataclass(frozen=True)
class EpisodeRecord:
    """Summary of one finished episode, built only from ``info`` and rewards.

    Attributes:
        episode_return: Sum of environment rewards (unscaled, no penalties).
        length: Number of control intervals.
        latency_violation_fraction: Fraction of intervals with
            ``info["violated_latency"]`` true.
        max_exposure: Maximum over the episode of the exposure signal
            (``info["p_failure_before_complete"]`` unless overridden).
        reason: ``TerminationReason`` value string.
        profile: Workload profile name reported by the environment.
    """

    episode_return: float
    length: int
    latency_violation_fraction: float
    max_exposure: float
    reason: str
    profile: str


@dataclass
class RolloutBatch:
    """Arrays of length ``T`` (steps) plus the episodes finished inside them."""

    obs: np.ndarray  # (T, 12) float32, observation the policy acted on
    next_obs: np.ndarray  # (T, 12) float32, observation returned by step()
    actions: np.ndarray  # (T,) int64
    log_probs: np.ndarray  # (T,) float32
    values: np.ndarray  # (T,) float32, V(obs) at collection time
    rewards: np.ndarray  # (T,) float64, environment reward
    terminated: np.ndarray  # (T,) bool
    truncated: np.ndarray  # (T,) bool
    latency_violated: np.ndarray  # (T,) float32 in {0, 1}, from info
    exposure: np.ndarray  # (T,) float32, from info
    episodes: list[EpisodeRecord] = field(default_factory=list)

    def __len__(self) -> int:
        return int(self.actions.shape[0])

    def termination_counts(self) -> dict[str, int]:
        """Number of finished episodes per ``TerminationReason`` value (all three keys)."""
        counts = Counter(ep.reason for ep in self.episodes)
        return {r.value: int(counts.get(r.value, 0)) for r in TerminationReason}


class RolloutCollector:
    """Collect fixed-length batches from one ``RecoveryEnv``.

    Args:
        config: Simulator configuration (workload profile, fleet level, ...).
        seed: Seeds both action sampling and the per-episode environment seeds.
        ablate_hazard: Zero observation fields 5 and 6 before the policy sees
            them (see :mod:`agent.ablation`).
        reliability_key: ``info`` key used as the per-step exposure signal.
    """

    def __init__(
        self,
        config: SimConfig,
        seed: int,
        ablate_hazard: bool = False,
        reliability_key: str = RELIABILITY_KEY,
    ) -> None:
        self.env: AblatedEnv = AblatedEnv(RecoveryEnv(config), enabled=ablate_hazard)
        self.seed: int = int(seed)
        self.reliability_key: str = reliability_key
        self._generator: torch.Generator = torch.Generator().manual_seed(self.seed)
        self._episode_index: int = 0
        self._obs: np.ndarray | None = None
        self._profile: str = ""
        self._reset_accumulators()

    def _reset_accumulators(self) -> None:
        self._ep_return: float = 0.0
        self._ep_length: int = 0
        self._ep_violations: int = 0
        self._ep_max_exposure: float = 0.0

    def _start_episode(self) -> None:
        obs, info = self.env.reset(seed=episode_seed(self.seed, self._episode_index))
        self._episode_index += 1
        self._obs = obs
        self._profile = str(info["workload_profile"])
        self._reset_accumulators()

    def collect(self, agent: Acting, n_steps: int) -> RolloutBatch:
        """Run ``n_steps`` control intervals and return them as a batch.

        Raises:
            ValueError: if ``n_steps`` is not a positive integer.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        obs_buf = np.zeros((n_steps, OBS_DIM), dtype=np.float32)
        next_buf = np.zeros((n_steps, OBS_DIM), dtype=np.float32)
        actions = np.zeros(n_steps, dtype=np.int64)
        log_probs = np.zeros(n_steps, dtype=np.float32)
        values = np.zeros(n_steps, dtype=np.float32)
        rewards = np.zeros(n_steps, dtype=np.float64)
        terminated = np.zeros(n_steps, dtype=bool)
        truncated = np.zeros(n_steps, dtype=bool)
        violated = np.zeros(n_steps, dtype=np.float32)
        exposure = np.zeros(n_steps, dtype=np.float32)
        episodes: list[EpisodeRecord] = []

        for t in range(n_steps):
            if self._obs is None:
                self._start_episode()
            assert self._obs is not None
            action, log_prob, value = agent.act(self._obs, self._generator)
            next_obs, reward, term, trunc, info = self.env.step(action)

            obs_buf[t] = self._obs
            next_buf[t] = next_obs
            actions[t] = action
            log_probs[t] = log_prob
            values[t] = value
            rewards[t] = reward
            terminated[t] = term
            truncated[t] = trunc
            violated[t] = 1.0 if info["violated_latency"] else 0.0
            exposure[t] = float(info[self.reliability_key])

            self._ep_return += float(reward)
            self._ep_length += 1
            self._ep_violations += int(bool(info["violated_latency"]))
            self._ep_max_exposure = max(self._ep_max_exposure, float(info[self.reliability_key]))

            if term or trunc:
                episodes.append(
                    EpisodeRecord(
                        episode_return=self._ep_return,
                        length=self._ep_length,
                        latency_violation_fraction=self._ep_violations / self._ep_length,
                        max_exposure=self._ep_max_exposure,
                        reason=str(info["termination_reason"]),
                        profile=self._profile,
                    )
                )
                self._obs = None
            else:
                self._obs = next_obs

        return RolloutBatch(
            obs=obs_buf,
            next_obs=next_buf,
            actions=actions,
            log_probs=log_probs,
            values=values,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            latency_violated=violated,
            exposure=exposure,
            episodes=episodes,
        )
