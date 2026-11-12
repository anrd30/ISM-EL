"""Clipped PPO with two Lagrangian constraints.

Objective. The agent maximises the (scaled) environment reward subject to two
constraints, each enforced through its own Lagrange multiplier::

    maximise   E[ sum_t reward_t ]
    subject to latency:     E[ fraction of steps with violated_latency ] <= LATENCY_VIOLATION_FRACTION_MAX
               reliability: E[ max_t p_failure_before_complete ]         <= RISK_BUDGET_EPS

Both constraint quantities are read from ``info`` (carried through
``RolloutBatch``), never from the reward. The contract makes ``info`` the
authoritative constraint record so that a reward penalty is never mistaken for
evidence that a constraint held.

The two constraints have different shapes and are handled separately:

- Latency is a per-step rate. Its per-step cost is the indicator
  ``violated_latency``; the multiplier measurement is the mean of that
  indicator over every step of the batch (a step-pooled rate), not a mean of
  per-episode fractions. Episodes differ in length and the fraction is
  strongly negatively correlated with length (fast, high-allocation episodes
  are short and violate more), so an unweighted mean over finished episodes
  reads about 0.025 above the step-pooled rate at the same policy.
- Reliability is an episode-level quantity: the largest exposure reached in
  the episode (the same statistic the evaluation harness reports as
  ``p_failure_max``). A maximum is not additive over steps, so the per-step
  cost used for credit assignment is the exceedance
  ``min(max(0, p_t - eps) / eps, risk_cost_cap)``, which is zero while the
  budget holds and grows with the size of the breach. The multiplier update
  uses the episode-level measurement, with violation divided by ``eps`` so the
  multiplier is dimensionless and its learning rate is comparable to the
  latency one.

Multipliers. ``lambda <- max(0, lambda + lr * (measured - limit) / scale)``,
projected gradient ascent on the constraint violation, with a learning rate
separate from the policy's. They are updated once per iteration from the
episodes that finished in that iteration's batch and are logged every
iteration. Expected behaviour: a multiplier rises while its constraint is
violated and decays towards zero while it is satisfied, so a multiplier that
stays flat is either an inactive constraint or a bug. Because the safety filter
raises any unsafe action to a feasible one when one exists, the measured
exposure is expected to stay under the budget except when no action is
feasible, which the policy can influence only through how much work remains;
the reliability multiplier is therefore expected to be inactive most of the
time. That is a property of the simulator's filter, not evidence about the
policy.

Bounded multipliers. Both multipliers are projected onto ``[0, lambda_max]``.
The latency constraint is not satisfiable everywhere: on the ``peak`` profile
under a degraded fleet no allocation keeps the violation fraction under the
limit (``sim/LIMITATIONS.md`` sections 13 and 14); with the default profile
mix, a quarter of the episodes are ``peak``, and the lowest fixed allocation
already violates on about a third of ``peak`` steps. An unbounded multiplier on an unsatisfiable constraint grows for as long
as training runs, and once large it dominates the reward on the profiles where
the constraint is satisfiable, pushing the policy to the slowest allocation
there too. The bound makes the penalty finite. Where it binds, the constraint
is no longer enforced by the multiplier: the penalty is a constant, and the
measured violation fraction must be read from the logs (``lambda_*_capped``
marks every iteration on which an update was clipped).

Penalised reward. Each step contributes
``reward_scale * reward - lambda_latency * cost_latency - lambda_risk * cost_risk``
to a single GAE and a single critic. The critic's target therefore drifts as the
multipliers move; the multipliers change slowly relative to the policy
(``lr_lambda_*`` are small), but this non-stationarity is not removed.

Discount. ``gamma = 0.999``, not the reflex 0.99. An episode runs up to
``MAX_EPISODE_STEPS = 2400`` intervals, and the main thing the agent has to
learn is that finishing the rebuild sooner is better. At 0.99 the effective
horizon is about 100 steps, so most of an episode lies beyond it and the agent
would not see the end of its own rebuild. At 0.999 the effective horizon is
about 1000 steps. The cost is higher-variance value targets, which is why GAE
``lambda`` stays at 0.95 and advantages are normalised per batch.

Truncation versus termination. Horizon expiry is truncation and recovery
completion or data loss is termination (``contracts.termination_flags``). On
truncation the target bootstraps ``V(s')`` of the true successor state; on
termination the value after the step is zero. Treating a horizon cutoff as
terminal bootstraps zero at an artificial boundary and under-values states that
lead to long rebuilds, biasing the agent against slow allocations, which is
the trade-off this project measures. ``resolve_next_values`` is the single place
this is decided.

Nothing here has been trained to a result; the defaults are untuned.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

from agent.networks import Actor, Critic
from agent.rollout import EpisodeRecord, RolloutBatch
from contracts.contracts import LATENCY_LIMIT_FACTOR, LATENCY_VIOLATION_FRACTION_MAX, RISK_BUDGET_EPS

__all__ = [
    "PPOConfig",
    "LagrangeMultiplier",
    "LossTerms",
    "PPOAgent",
    "resolve_next_values",
    "compute_gae",
    "ppo_loss",
    "latency_barrier",
]

BARRIER_DELTA: float = 0.05
BARRIER_CAP: float = 10.0


def latency_barrier(latency_now_ms: np.ndarray, latency_target_ms: float) -> np.ndarray:
    """Log-barrier cost of the latency margin, bounded past the boundary.

    With ``limit = LATENCY_LIMIT_FACTOR * latency_target_ms`` and the normalised margin
    ``d = (limit - latency_now) / limit`` (positive inside the limit, 0 at it, negative past
    it), the cost is ``-log(d)`` for ``d >= BARRIER_DELTA``. Below ``BARRIER_DELTA`` it continues
    along the tangent at ``BARRIER_DELTA`` (value ``-log(delta)``, slope ``-1/delta``), which keeps
    the function continuous and differentiable and finite for every ``d``. The result is capped at
    ``BARRIER_CAP``. The cost is 0 at zero latency (``d = 1``) and increases monotonically as
    ``d`` decreases.
    """
    limit = LATENCY_LIMIT_FACTOR * float(latency_target_ms)
    d = (limit - np.asarray(latency_now_ms, dtype=np.float64)) / limit
    inside = -np.log(np.maximum(d, BARRIER_DELTA))
    tangent = np.where(d < BARRIER_DELTA, -(d - BARRIER_DELTA) / BARRIER_DELTA, 0.0)
    return np.minimum(inside + tangent, BARRIER_CAP)


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters. Every default is an untuned choice.

    Attributes:
        gamma: Discount; 0.999 deliberately (see the module docstring).
        gae_lambda: GAE lambda.
        clip_ratio: PPO ratio clipping range.
        vf_coef: Value-loss coefficient.
        ent_coef: Entropy bonus at the start of training.
        ent_coef_final: Entropy bonus at the end; linear in training progress.
            Equal to ``ent_coef`` by default, which disables the anneal. With
            advantages normalised to unit scale, a bonus that decays to 0.001
            is small next to the policy gradient, and on the default profile
            mix the policy reached about 0.15 nats (uniform over four actions
            is 1.386) within 100 iterations of 2048 steps, concentrating on
            the smallest allocation. The same collapse occurred with the
            latency penalty switched off, so it is not caused by the
            multiplier: on three of the four profiles one of the two smallest
            allocations has the best fixed-allocation return, so the gradient
            points the same way in most states. A constant 0.01 held the entropy
            between about 0.75 and 1.06 nats over 60 iterations of 8192 steps
            in a single-seed comparison. Untuned.
        lr: Adam step size for actor and critic.
        epochs: Passes over each batch.
        minibatch_size: Minibatch size within an epoch.
        max_grad_norm: Global gradient-norm clip.
        target_kl: If set, stop the epochs early once the mean approximate KL
            of an epoch exceeds ``1.5 * target_kl``.
        reward_scale: Multiplier on the environment reward before the penalties
            are subtracted; keeps the value targets near unit scale. Multiplier
            values are in units of the scaled reward.
        lr_lambda_latency: Step size of the latency multiplier.
        lr_lambda_risk: Step size of the reliability multiplier (acting on the
            violation divided by ``risk_limit``).
        lambda_max: Upper bound on either multiplier (infinite means none), in
            units of the scaled reward. With ``reward_scale = 0.1`` a fully
            violating step on the ``diurnal`` and ``peak`` profiles carries an
            environment reward term of the order of one scaled unit, so 2.0
            lets the latency penalty reach a small multiple of what the reward
            already charges for a violation but not exceed it by an order of
            magnitude. The value is an untuned design choice; the rationale
            for bounding at all is in :class:`LagrangeMultiplier`.
        risk_cost_cap: Cap on the per-step normalised exposure exceedance.
        latency_limit: Allowed mean fraction of violating steps per episode.
        risk_limit: Allowed episode exposure (the budget ``eps``).
        latency_window: Number of most recent per-iteration latency
            measurements averaged before the multiplier update; 1 uses the
            current iteration's measurement alone.
        action_prob_floor: Minimum probability of every action. The policy
            distribution is ``floor + (1 - n_actions * floor) * softmax(logits)``;
            sampling, the stored log-probabilities and the PPO loss all use this
            distribution. 0 leaves the softmax unchanged. Must satisfy
            ``n_actions * floor < 1``. This is the initial value when a decay
            is configured.
        action_prob_floor_hold: Number of leading iterations at the initial
            floor when a decay is configured.
        action_prob_floor_end: Iteration at which the floor reaches 0 after a
            linear decay from the end of the hold; 0 disables the decay and
            keeps the floor constant. The floor of an iteration is fixed for
            that iteration's collection and update.
        mask_infeasible_cost: Episodes flagged ``infeasible`` (latency target
            violated at the reset observation, before any recovery bandwidth is
            applied) are excluded from the latency multiplier measurement, and
            their per-step latency cost is set to zero. The reliability
            measurement and cost are unaffected.
    """

    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    ent_coef_final: float = 0.01
    lr: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 256
    max_grad_norm: float = 0.5
    target_kl: float | None = None
    reward_scale: float = 0.1
    lr_lambda_latency: float = 0.5
    lr_lambda_risk: float = 0.05
    lambda_max: float = 2.0
    risk_cost_cap: float = 10.0
    latency_limit: float = LATENCY_VIOLATION_FRACTION_MAX
    risk_limit: float = RISK_BUDGET_EPS
    mask_infeasible_cost: bool = True
    latency_window: int = 1
    action_prob_floor: float = 0.0
    action_prob_floor_hold: int = 0
    action_prob_floor_end: int = 0


class LagrangeMultiplier:
    """A non-negative multiplier updated by projected gradient ascent.

    ``update(measured)`` applies ``value <- clip(value + lr * (measured - limit)
    / scale, 0, max_value)``. The multiplier rises while ``measured > limit``
    and decays towards zero while ``measured < limit``; it stays exactly flat
    only when ``measured == limit`` or when it is pinned at zero by a
    satisfied constraint.

    The upper bound is a projection, not a feasibility argument: when a
    constraint cannot be met, the unbounded multiplier grows without limit and
    the penalty term eventually outweighs the reward term for every state. The
    bound stops that growth. An update *binds* when the unclipped step exceeds
    ``max_value``; ``last_capped``, ``n_capped`` and ``n_updates`` record this
    so that a multiplier sitting on its bound is distinguishable from one that
    settled there because the constraint held.

    Args:
        name: Label used in logs.
        limit: The constraint threshold ``d``.
        lr: Step size.
        scale: Divisor on the violation, making the multiplier dimensionless.
        initial: Starting value (must be >= 0).
        max_value: Upper bound.
    """

    def __init__(
        self,
        name: str,
        limit: float,
        lr: float,
        scale: float = 1.0,
        initial: float = 0.0,
        max_value: float = math.inf,
    ) -> None:
        if initial < 0.0 or scale <= 0.0 or lr < 0.0:
            raise ValueError("initial and lr must be >= 0 and scale > 0")
        self.name: str = name
        self.limit: float = float(limit)
        self.lr: float = float(lr)
        self.scale: float = float(scale)
        self.max_value: float = float(max_value)
        self.value: float = float(initial)
        self.last_capped: bool = False
        self.n_capped: int = 0
        self.n_updates: int = 0

    def update(self, measured: float) -> float:
        """Apply one ascent step and return the new value.

        Sets ``last_capped`` to whether the unclipped step exceeded
        ``max_value`` and advances the ``n_capped`` / ``n_updates`` counters.
        """
        if not math.isfinite(measured):
            raise ValueError(f"measured constraint value for {self.name!r} is not finite: {measured!r}")
        stepped = self.value + self.lr * (measured - self.limit) / self.scale
        self.last_capped = stepped > self.max_value
        self.n_updates += 1
        self.n_capped += int(self.last_capped)
        self.value = min(self.max_value, max(0.0, stepped))
        return self.value


def resolve_next_values(
    values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    truncation_values: np.ndarray,
    last_value: float,
) -> np.ndarray:
    """Return ``V(s_{t+1})`` for every step, honouring termination and truncation.

    Args:
        values: ``V(s_t)`` for the ``T`` collected steps.
        terminated: Step ended the episode by absorbing state (value after it is 0).
        truncated: Step ended the episode by cutoff (bootstrap from the true successor).
        truncation_values: ``V(s'_t)`` for every step; only entries at truncated
            steps are read.
        last_value: ``V`` of the observation after the final step; used only if
            that step did not end an episode.

    Per step: terminated gives 0; truncated gives ``truncation_values[t]``; any
    other step gives ``values[t + 1]``, or ``last_value`` for the final step.
    """
    n = len(values)
    nxt = np.empty(n, dtype=np.float64)
    nxt[:-1] = values[1:]
    nxt[-1] = last_value
    nxt = np.where(truncated, truncation_values, nxt)
    nxt = np.where(terminated, 0.0, nxt)
    return nxt


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalised advantage estimates and value targets.

    ``delta_t = r_t + gamma * next_values_t - values_t`` (``next_values`` already
    resolved by :func:`resolve_next_values`, so termination is a zero there and
    truncation a bootstrapped value). The advantage recursion
    ``A_t = delta_t + gamma * lambda * A_{t+1}`` is cut at every episode
    boundary, terminated or truncated, because the following step belongs to a
    different episode. Returns ``(advantages, returns)`` with
    ``returns = advantages + values``.
    """
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float64)
    last = 0.0
    for t in range(n - 1, -1, -1):
        boundary = bool(terminated[t] or truncated[t])
        delta = rewards[t] + gamma * next_values[t] - values[t]
        last = delta + gamma * gae_lambda * (0.0 if boundary else 1.0) * last
        adv[t] = last
    return adv, adv + values


@dataclass
class LossTerms:
    """PPO loss and diagnostics for one minibatch (tensors; ``total`` carries the graph)."""

    total: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor


def ppo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    values: torch.Tensor,
    returns: torch.Tensor,
    entropy: torch.Tensor,
    clip_ratio: float,
    vf_coef: float,
    ent_coef: float,
) -> LossTerms:
    """Clipped surrogate, value MSE and entropy bonus.

    ``policy_loss = -mean(min(r * A, clip(r, 1 - c, 1 + c) * A))`` with
    ``r = exp(new - old)``. ``approx_kl`` is the estimator
    ``mean((r - 1) - log r)``; ``clip_fraction`` is the share of samples with
    ``|r - 1| > c``.
    """
    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    policy_loss = -torch.minimum(unclipped, clipped).mean()
    value_loss = ((values - returns) ** 2).mean()
    entropy_mean = entropy.mean()
    total = policy_loss + vf_coef * value_loss - ent_coef * entropy_mean
    with torch.no_grad():
        approx_kl = ((ratio - 1.0) - log_ratio).mean()
        clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean()
    return LossTerms(total, policy_loss, value_loss, entropy_mean, approx_kl, clip_fraction)


class PPOAgent:
    """Actor, critic, optimiser and the two multipliers.

    Args:
        config: Hyperparameters.
        seed: Seeds network initialisation and minibatch shuffling.
        marginal_violation_cost: When True, the per-step latency cost and the latency
            multiplier measurement use ``max(0, violated_latency - min_action_violated)``
            instead of ``violated_latency``, where ``min_action_violated`` is the violation
            the smallest action would have produced at the same step
            (``agent.min_action_counterfactual``), supplied per step through
            :meth:`set_min_action_violated` before each :meth:`step_costs` or
            :meth:`update_multipliers` call. False (default) leaves every computation
            unchanged.
        barrier_cost: When True, the latency term of the penalised reward is
            ``barrier_coef * latency_barrier(latency_now)`` instead of
            ``lambda_latency * violated_latency``; it replaces the latency Lagrangian charge.
            ``lambda_latency`` is still updated and logged from the measured violation
            fraction but multiplies nothing. The reliability term is unchanged. Needs
            ``barrier_latency_target_ms`` and a batch carrying ``latency_now``. Cannot be
            combined with ``marginal_violation_cost``.
        barrier_coef: Weight of the barrier cost; the caller may change the attribute
            ``barrier_coef`` between iterations to anneal it.
        barrier_latency_target_ms: Latency target the barrier margin is measured against.
    """

    def __init__(
        self,
        config: PPOConfig | None = None,
        seed: int = 0,
        marginal_violation_cost: bool = False,
        barrier_cost: bool = False,
        barrier_coef: float = 0.05,
        barrier_latency_target_ms: float | None = None,
    ) -> None:
        if barrier_cost and marginal_violation_cost:
            raise ValueError("barrier_cost and marginal_violation_cost are mutually exclusive")
        if barrier_cost and barrier_latency_target_ms is None:
            raise ValueError("barrier_cost needs barrier_latency_target_ms")
        self.barrier_cost: bool = bool(barrier_cost)
        self.barrier_coef: float = float(barrier_coef)
        self.barrier_latency_target_ms: float | None = barrier_latency_target_ms
        self.marginal_violation_cost: bool = bool(marginal_violation_cost)
        self._min_action_violated: np.ndarray | None = None
        self.config: PPOConfig = config if config is not None else PPOConfig()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.actor: Actor = Actor()
            self.critic: Critic = Critic()
        self.optimizer: torch.optim.Adam = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.config.lr, eps=1e-5
        )
        self.lambda_latency = LagrangeMultiplier(
            "latency",
            limit=self.config.latency_limit,
            lr=self.config.lr_lambda_latency,
            max_value=self.config.lambda_max,
        )
        self.lambda_risk = LagrangeMultiplier(
            "reliability",
            limit=self.config.risk_limit,
            lr=self.config.lr_lambda_risk,
            scale=self.config.risk_limit,
            max_value=self.config.lambda_max,
        )
        self._latency_history: deque[float] = deque(maxlen=max(1, int(self.config.latency_window)))
        self.action_prob_floor: float = self.config.action_prob_floor
        self._shuffle: torch.Generator = torch.Generator().manual_seed(int(seed) + 1)

    # ------------------------------------------------------------------
    # Acting
    # ------------------------------------------------------------------

    def floor_at(self, iteration: int) -> float:
        """Floor in force during 1-based ``iteration`` under the configured hold and linear decay."""
        cfg = self.config
        if cfg.action_prob_floor_end <= 0:
            return cfg.action_prob_floor
        if iteration <= cfg.action_prob_floor_hold:
            return cfg.action_prob_floor
        if iteration >= cfg.action_prob_floor_end:
            return 0.0
        span = cfg.action_prob_floor_end - cfg.action_prob_floor_hold
        return cfg.action_prob_floor * (cfg.action_prob_floor_end - iteration) / span

    def _floored_log_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Log-probabilities of the floored softmax (plain log-softmax when the floor is 0)."""
        floor = self.action_prob_floor
        if floor <= 0.0:
            return torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        return torch.log(floor + (1.0 - logits.shape[-1] * floor) * probs)

    @torch.no_grad()
    def act(
        self, obs: np.ndarray, generator: torch.Generator | None = None, deterministic: bool = False
    ) -> tuple[int, float, float]:
        """Return ``(action index, log-probability, value)`` for one observation."""
        x = torch.as_tensor(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
        log_probs = self._floored_log_probs(self.actor(x))
        if deterministic:
            action = int(torch.argmax(log_probs, dim=-1).item())
        else:
            action = int(torch.multinomial(log_probs.exp(), 1, generator=generator).item())
        return action, float(log_probs[0, action].item()), float(self.critic(x).item())

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> np.ndarray:
        """Critic values for a batch of observations ``(B, 12)``."""
        return self.critic(torch.as_tensor(np.asarray(obs, dtype=np.float32))).numpy().astype(np.float64)

    # ------------------------------------------------------------------
    # Constraint signals and advantages
    # ------------------------------------------------------------------

    def set_min_action_violated(self, min_action_violated: np.ndarray) -> None:
        """Set the per-step smallest-action violation (``(T,)``, aligned with the next batch)."""
        self._min_action_violated = np.asarray(min_action_violated, dtype=np.float64)

    def _marginal_cost(self, violated: np.ndarray, batch: RolloutBatch) -> np.ndarray:
        """Per-step ``max(0, violated - min_action_violated)``.

        Raises:
            ValueError: if the smallest-action violations were not set for a batch of this length.
        """
        cf = self._min_action_violated
        if cf is None or cf.shape != violated.shape:
            raise ValueError(
                "marginal_violation_cost needs set_min_action_violated() with one value per step of the batch"
            )
        return np.maximum(0.0, violated - cf)

    def step_costs(self, batch: RolloutBatch) -> tuple[np.ndarray, np.ndarray]:
        """Per-step ``(latency cost, reliability cost)`` from the batch's ``info`` signals."""
        cfg = self.config
        latency_cost = batch.latency_violated.astype(np.float64)
        if self.marginal_violation_cost:
            latency_cost = self._marginal_cost(latency_cost, batch)
        if cfg.mask_infeasible_cost and batch.infeasible is not None:
            latency_cost = np.where(batch.infeasible, 0.0, latency_cost)
        excess = np.maximum(0.0, batch.exposure.astype(np.float64) - cfg.risk_limit) / cfg.risk_limit
        return latency_cost, np.minimum(excess, cfg.risk_cost_cap)

    def penalised_rewards(self, batch: RolloutBatch) -> np.ndarray:
        """``reward_scale * reward - lambda_latency * c_latency - lambda_risk * c_risk``."""
        latency_cost, risk_cost = self.step_costs(batch)
        if self.barrier_cost:
            return (
                self.config.reward_scale * batch.rewards
                - self.barrier_coef * self._barrier_step_cost(batch)
                - self.lambda_risk.value * risk_cost
            )
        return (
            self.config.reward_scale * batch.rewards
            - self.lambda_latency.value * latency_cost
            - self.lambda_risk.value * risk_cost
        )

    def _barrier_step_cost(self, batch: RolloutBatch) -> np.ndarray:
        """Per-step ``latency_barrier``; steps of infeasible episodes cost 0 when ``mask_infeasible_cost``.

        Raises:
            ValueError: if the batch carries no finite ``latency_now``.
        """
        if batch.latency_now is None or not np.all(np.isfinite(batch.latency_now)):
            raise ValueError("barrier_cost needs a batch with finite latency_now")
        cost = latency_barrier(batch.latency_now, float(self.barrier_latency_target_ms))
        if self.config.mask_infeasible_cost and batch.infeasible is not None:
            cost = np.where(batch.infeasible, 0.0, cost)
        return cost

    def advantages(self, batch: RolloutBatch) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(advantages, returns)`` for the batch, with correct bootstrapping."""
        truncation_values = np.zeros(len(batch), dtype=np.float64)
        if batch.truncated.any():
            truncation_values[batch.truncated] = self.value(batch.next_obs[batch.truncated])
        last_value = 0.0
        if not (batch.terminated[-1] or batch.truncated[-1]):
            last_value = float(self.value(batch.next_obs[-1:])[0])
        next_values = resolve_next_values(
            batch.values.astype(np.float64), batch.terminated, batch.truncated, truncation_values, last_value
        )
        return compute_gae(
            self.penalised_rewards(batch),
            batch.values.astype(np.float64),
            next_values,
            batch.terminated,
            batch.truncated,
            self.config.gamma,
            self.config.gae_lambda,
        )

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def entropy_coef(self, progress: float) -> float:
        """Entropy coefficient at training ``progress`` in ``[0, 1]`` (linear anneal)."""
        p = min(1.0, max(0.0, progress))
        return self.config.ent_coef + p * (self.config.ent_coef_final - self.config.ent_coef)

    def update(self, batch: RolloutBatch, progress: float = 0.0) -> dict[str, float]:
        """Run the PPO epochs on ``batch`` and return mean diagnostics.

        Does not touch the multipliers; call :meth:`update_multipliers` after.
        """
        cfg = self.config
        adv_np, ret_np = self.advantages(batch)
        adv_np = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)

        obs = torch.as_tensor(batch.obs)
        actions = torch.as_tensor(batch.actions)
        old_log_probs = torch.as_tensor(batch.log_probs)
        adv = torch.as_tensor(adv_np, dtype=torch.float32)
        ret = torch.as_tensor(ret_np, dtype=torch.float32)
        ent_coef = self.entropy_coef(progress)

        n = len(batch)
        sums = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0}
        count = 0
        for _ in range(cfg.epochs):
            order = torch.randperm(n, generator=self._shuffle)
            epoch_kl = 0.0
            epoch_batches = 0
            for start in range(0, n, cfg.minibatch_size):
                idx = order[start : start + cfg.minibatch_size]
                dist = torch.distributions.Categorical(logits=self._floored_log_probs(self.actor(obs[idx])))
                terms = ppo_loss(
                    dist.log_prob(actions[idx]),
                    old_log_probs[idx],
                    adv[idx],
                    self.critic(obs[idx]),
                    ret[idx],
                    dist.entropy(),
                    cfg.clip_ratio,
                    cfg.vf_coef,
                    ent_coef,
                )
                self.optimizer.zero_grad(set_to_none=True)
                terms.total.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), cfg.max_grad_norm
                )
                self.optimizer.step()
                sums["policy_loss"] += float(terms.policy_loss)
                sums["value_loss"] += float(terms.value_loss)
                sums["entropy"] += float(terms.entropy)
                sums["approx_kl"] += float(terms.approx_kl)
                sums["clip_fraction"] += float(terms.clip_fraction)
                epoch_kl += float(terms.approx_kl)
                epoch_batches += 1
                count += 1
            if cfg.target_kl is not None and epoch_kl / epoch_batches > 1.5 * cfg.target_kl:
                break
        return {k: v / count for k, v in sums.items()}

    def update_multipliers(
        self, episodes: Sequence[EpisodeRecord], batch: RolloutBatch | None = None
    ) -> dict[str, float]:
        """Update both multipliers from the episodes finished in the last batch.

        Measurements: mean over episodes of the latency violation fraction, and
        mean over episodes of the maximum exposure. With no finished episode
        there is no measurement and neither multiplier moves (the returned
        measurements are NaN). Episodes flagged ``infeasible`` are left out of
        the latency measurement when ``mask_infeasible_cost`` is set; if every
        finished episode is flagged, the latency multiplier does not move and
        its measurement is NaN, while the reliability multiplier still updates.
        ``n_infeasible_episodes`` counts the flagged episodes.

        With ``batch`` given, the latency measurement is the mean of
        ``latency_violated`` over the batch's steps (excluding steps of
        infeasible episodes when ``mask_infeasible_cost`` is set), averaged
        over the last ``latency_window`` iterations; it exists whenever the
        batch has counted steps, including when no episode finished. Without
        ``batch`` it is the mean over episodes' violation fractions.
        ``episode_violation_fraction`` is that episode-weighted mean in both
        cases.
        ``lambda_latency_capped`` and
        ``lambda_risk_capped`` are 1.0 when this iteration's update was
        clipped by ``lambda_max`` and 0.0 otherwise (including when no update
        happened).
        """
        if not episodes and batch is None:
            nan = float("nan")
            return {
                "latency_violation_fraction": nan,
                "episode_violation_fraction": nan,
                "max_exposure": nan,
                "lambda_latency": self.lambda_latency.value,
                "lambda_risk": self.lambda_risk.value,
                "lambda_latency_capped": 0.0,
                "lambda_risk_capped": 0.0,
                "n_infeasible_episodes": 0,
            }
        n_infeasible = sum(bool(e.infeasible) for e in episodes)
        counted = [e for e in episodes if not (self.config.mask_infeasible_cost and e.infeasible)]
        exposure = float(np.mean([e.max_exposure for e in episodes])) if episodes else float("nan")
        episode_latency = float(np.mean([e.latency_violation_fraction for e in counted])) if counted else float("nan")
        latency = episode_latency
        if batch is not None:
            steps = batch.latency_violated.astype(np.float64)
            if self.marginal_violation_cost:
                steps = self._marginal_cost(steps, batch)
            if self.config.mask_infeasible_cost and batch.infeasible is not None:
                steps = steps[~batch.infeasible]
            latency = float(steps.mean()) if len(steps) else float("nan")
            if not math.isnan(latency):
                self._latency_history.append(latency)
                latency = float(np.mean(self._latency_history))
        if not math.isnan(latency):
            lambda_latency = self.lambda_latency.update(latency)
            latency_capped = float(self.lambda_latency.last_capped)
        else:
            latency = float("nan")
            lambda_latency = self.lambda_latency.value
            latency_capped = 0.0
        return {
            "latency_violation_fraction": latency,
            "episode_violation_fraction": episode_latency,
            "max_exposure": exposure,
            "lambda_latency": lambda_latency,
            "lambda_risk": self.lambda_risk.update(exposure) if episodes else self.lambda_risk.value,
            "lambda_latency_capped": latency_capped,
            "lambda_risk_capped": float(self.lambda_risk.last_capped) if episodes else 0.0,
            "n_infeasible_episodes": n_infeasible,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def state(self) -> dict[str, Any]:
        """Serialisable state: weights, optimiser, multipliers, hyperparameters."""
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lambda_latency": self.lambda_latency.value,
            "lambda_risk": self.lambda_risk.value,
            "cap_counts": {
                "latency": [self.lambda_latency.n_capped, self.lambda_latency.n_updates],
                "risk": [self.lambda_risk.n_capped, self.lambda_risk.n_updates],
            },
            "ppo_config": asdict(self.config),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        """Restore from :meth:`state` (the hyperparameters are not overwritten)."""
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        self.lambda_latency.value = float(state["lambda_latency"])
        self.lambda_risk.value = float(state["lambda_risk"])
        counts = state.get("cap_counts")
        if counts is not None:
            self.lambda_latency.n_capped, self.lambda_latency.n_updates = (int(v) for v in counts["latency"])
            self.lambda_risk.n_capped, self.lambda_risk.n_updates = (int(v) for v in counts["risk"])
