"""Non-learned baseline policies for the rebuild-bandwidth controller.

Every policy maps the environment's normalised 12-vector to an action index
and nothing else, so a baseline sees exactly what a learned agent sees. The
policies are comparators for a learned controller; their value depends on how
hard they are to beat, so each is built to be as good as a competent
non-learning design can be. All results obtained with them come from the
simulator in ``sim/``, whose constants are design decisions and whose latency
model is uncalibrated (``sim/LIMITATIONS.md``).

Policies
--------
``FixedRate``
    Constant allocation; four instances bracket the achievable range.
``WorkloadOnlyThrottle``
    Chooses the allocation from foreground load alone. Never reads
    observation fields 5 and 6 (``hazard``, ``hazard_uncertainty``).
``RiskAwareRuleBased``
    The strong baseline: the same load rule, restricted to the allocations
    that satisfy the reliability budget (optionally under a hazard inflated by
    ``k`` uncertainty standard deviations; ``k = 0`` by default).
``OracleLookahead``
    ``RiskAwareRuleBased`` given the next interval's foreground load. An upper
    reference for one-interval forecasting, not a deployable controller.
``SlowestRiskFeasible``, and the ``selection="ceiling"`` variants
    Diagnostics that show what the spec's literal rules do; see below.

What the load rule does
-----------------------
The environment gives the controller no look-ahead: the observation describes
the interval just executed and the allocation chosen now applies to the next
one. Every non-oracle policy therefore holds a belief about the next
interval's foreground bandwidth, a set of weighted scenarios built from
observations only:

- ``level``: the median of the last ``level_window`` observed values, a
  burst-robust estimate of the current load.
- ``residuals``: for each of the last ``window`` intervals, how far the
  observed value was from the level just before it arrived.
- Scenarios: ``level + residual`` for every stored residual (equal weights
  summing to ``1 - last_weight``), plus the newest observed value itself with
  weight ``last_weight`` (burst persistence).

Slow drift of the load (a daily cycle) moves the level and leaves the residuals
small; bursts that arrive at random leave the level alone and appear in the
residuals at their observed frequency and height. The belief is an online
estimate rebuilt each episode (``reset`` clears it); nothing in it is trained.

For each candidate allocation the policy predicts, with the simulator's own
latency model (:class:`LatencyRule`), the latency and delivered rebuild rate
under every foreground scenario in the belief. Two selectors then choose:

``selection="progress_value"`` (default)
    Minimise ``E[alpha * D + gamma * V] - progress_value * E[delivered]``.
    ``D`` and ``V`` are the contract's latency degradation and violation
    terms, so the rule trades latency cost against finishing sooner using the
    reward's own coefficients. The reward charges ``beta`` per interval until
    the rebuild ends, so extra progress saves time; ``progress_value`` is that
    saving per MB/s, ``beta / average_rate``. This selector spends spare
    latency headroom on speed, and it backs off when the belief says a burst
    is likely, which a single-point forecast cannot do.
``selection="ceiling"``
    The literal reading of the spec: the largest allocation whose predicted
    latency, at the last observed foreground bandwidth, stays inside the
    limit (``LATENCY_LIMIT_FACTOR * latency_target``). Kept so its cost
    against the default is measurable.

Why the default is not the literal rule. The literal rule runs each interval
as close to the latency limit as the last interval's load allows. Foreground
load changes between intervals, and one violating interval costs ``gamma = 10``
against ``beta = 0.1`` for one more interval of rebuild, so a rule with no
margin gives away reward. Measured on 40 tuning seeds with a healthy fleet
(simulated), the ceiling rule's mean total reward was -371.6 on ``bursty`` and
-1371.7 on ``peak``, against -332.7 and -887.2 for a constant 20 MB/s: worse
than not controlling at all, which is a baseline too weak to compare against.
The default rule scored -276.9 and -887.3 on the same seeds.

Tuning disclosure. ``level_window``, ``window``, ``last_weight`` and
``progress_value`` were chosen by sweeping grids on scenario seeds 1000 to 1099
(never the report's evaluation seeds, which start at 0), with the default
random fleet, one global setting for all profiles and ``k = 0``. The grids were
coarse and the selection judged by mean total reward across profiles, with the
latency-budget attainment rate checked. The values reflect the synthetic
profiles of ``workload.generate`` and this simulator's constants and carry no
claim beyond them.

Denormalisation. Policies recover interpretable units with
``Observation.from_vector(obs, normalized=True)``. The vector was clipped to
``[0, 1]`` by ``to_vector``, so the recovery is lossy for any raw value above
its field's scale. This matters most for ``hazard``: a group hazard at or above
``1.8e-4`` per hour reads as exactly ``1.8e-4``, so a policy cannot tell how far
above it the true value is (``contracts/INTERFACES.md`` section 3). The values
are float32 in transit, about 1e-7 relative error.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from contracts.contracts import (
    ACTIONS_MBPS,
    BW_TOTAL_MBPS,
    LATENCY_LIMIT_FACTOR,
    N_ACTIONS,
    RISK_BUDGET_EPS,
    Observation,
)
from sim.config import SimConfig
from sim.safety_filter import apply_safety_filter

__all__ = [
    "Policy",
    "NeedsEnv",
    "LatencyRule",
    "BeliefConfig",
    "FixedRate",
    "WorkloadOnlyThrottle",
    "RiskAwareRuleBased",
    "OracleLookahead",
    "SlowestRiskFeasible",
    "peek_next_fg_mbps",
    "fixed_rate_policies",
    "baseline_policies",
    "diagnostic_policies",
    "DEFAULT_HAZARD_UNCERTAINTY_K",
    "DEFAULT_LEVEL_WINDOW",
    "DEFAULT_WINDOW",
    "DEFAULT_LAST_WEIGHT",
    "DEFAULT_PROGRESS_VALUE",
]

#: Hazard standard deviations added before the exposure test. Zero by default
#: because a positive value costs reward in this simulator and buys nothing
#: measurable; see ``RiskAwareRuleBased``. Pass ``k=1.0`` for the margin variant.
DEFAULT_HAZARD_UNCERTAINTY_K: float = 0.0
#: Intervals over which the load level is taken as a median: one hour.
DEFAULT_LEVEL_WINDOW: int = 12
#: Residual history length in intervals: one day at ``CONTROL_INTERVAL_S``.
DEFAULT_WINDOW: int = 288
#: Belief weight on the newest foreground value (the rest is spread over the residuals).
DEFAULT_LAST_WEIGHT: float = 0.5
#: Reward units per MB/s of expected delivered rebuild rate; ``beta / 25 MB/s``.
DEFAULT_PROGRESS_VALUE: float = 0.004

Selection = Literal["progress_value", "ceiling"]

_ACTIONS: tuple[float, ...] = tuple(float(a) for a in ACTIONS_MBPS)
_TIE_TOLERANCE: float = 1e-12


@runtime_checkable
class Policy(Protocol):
    """A controller: normalised observation in, action index out.

    ``name`` labels the policy in every output. ``reset`` clears per-episode
    state and is called by the harness before each episode.
    """

    name: str

    def __call__(self, obs: np.ndarray) -> int: ...

    def reset(self) -> None: ...


@runtime_checkable
class NeedsEnv(Protocol):
    """A policy that reads the environment directly (an oracle).

    The evaluation harness calls ``bind(env)`` once before the first episode.
    """

    def bind(self, env: Any) -> None: ...


@dataclass(frozen=True)
class LatencyRule:
    """The simulator's latency model, restated for prediction, and two selectors.

    Attributes:
        config: Simulator configuration supplying ``latency_unloaded_ms``,
            ``rho_max``, ``latency_target_ms`` and the reward coefficients.
        ceiling_factor: Used by :meth:`choose_ceiling`; the latency ceiling is
            ``ceiling_factor * latency_target_ms``.
    """

    config: SimConfig
    ceiling_factor: float = LATENCY_LIMIT_FACTOR

    @property
    def ceiling_ms(self) -> float:
        """Latency above which :meth:`choose_ceiling` rejects a candidate."""
        return self.ceiling_factor * self.config.latency_target_ms

    @property
    def limit_ms(self) -> float:
        """The contract's violation threshold, ``LATENCY_LIMIT_FACTOR * target``."""
        return LATENCY_LIMIT_FACTOR * self.config.latency_target_ms

    def delivered_mbps(self, fg_mbps: np.ndarray | float, alloc_mbps: float) -> np.ndarray:
        """Rebuild rate the simulator delivers: ``min(alloc, max(0, rho_max * BW - fg))``."""
        headroom = np.maximum(0.0, self.config.rho_max * BW_TOTAL_MBPS - np.asarray(fg_mbps, dtype=np.float64))
        return np.minimum(alloc_mbps, headroom)

    def predicted_latency_ms(self, fg_mbps: np.ndarray | float, alloc_mbps: float) -> np.ndarray:
        """Mean latency the simulator reports for ``alloc_mbps`` at ``fg_mbps``.

        Vectorised over ``fg_mbps``. It is ``latency_unloaded_ms / (1 - rho)``
        with ``rho = min(rho_max, (fg + delivered) / BW_TOTAL_MBPS)``, the
        expression of ``sim.config.latency_mean_ms``; ``rho`` is already
        inside that function's clamp. The environment builds ``rho`` in a
        private method, so the two one-line formulas are restated here and
        pinned by a test against the simulator's realised latency.
        """
        fg = np.asarray(fg_mbps, dtype=np.float64)
        delivered = self.delivered_mbps(fg, alloc_mbps)
        rho = np.minimum(self.config.rho_max, (fg + delivered) / BW_TOTAL_MBPS)
        return np.asarray(self.config.latency_unloaded_ms / (1.0 - rho))

    def choose_ceiling(self, fg_mbps: float, lowest_index: int = 0) -> int:
        """Largest index in ``[lowest_index, N_ACTIONS)`` whose predicted latency fits the ceiling.

        If none fits, the index with the lowest predicted latency, ties
        resolved to the smaller allocation. A tie arises when the forecast
        headroom, not the request, limits the delivered rate; that headroom is
        uncertain, and a large request would do harm if the foreground load
        then falls, so the cautious tie-break is used.
        """
        latencies = [float(self.predicted_latency_ms(fg_mbps, _ACTIONS[i])) for i in range(lowest_index, N_ACTIONS)]
        for offset in range(len(latencies) - 1, -1, -1):
            if latencies[offset] <= self.ceiling_ms:
                return lowest_index + offset
        best = min(latencies)
        for offset, value in enumerate(latencies):
            if value <= best + _TIE_TOLERANCE:
                return lowest_index + offset
        raise AssertionError("unreachable: a minimum always exists")  # pragma: no cover

    def score(
        self,
        fg_samples: np.ndarray,
        weights: np.ndarray,
        alloc_mbps: float,
        progress_value: float,
    ) -> float:
        """Expected latency cost minus the value of expected progress, for one allocation.

        Over foreground scenarios ``fg_samples`` with probabilities
        ``weights``::

            E[alpha * D + gamma * V] - progress_value * E[delivered]

        ``D`` is the contract's latency degradation, ``V`` its violation
        indicator, ``alpha`` and ``gamma`` the configured ``RewardConfig``
        coefficients, and ``delivered`` the rebuild rate the simulator would
        deliver. The reward charges ``beta`` for every interval until the
        rebuild ends, so a faster interval saves ``beta`` times the fraction
        of an average interval's progress it adds; ``progress_value`` prices
        that saving per MB/s and equals ``beta / average_rate``. The array
        forms of ``D`` and ``V`` are tested against the contract's helpers.
        """
        reward = self.config.reward
        latency = self.predicted_latency_ms(fg_samples, alloc_mbps)
        target = self.config.latency_target_ms
        degradation = np.maximum(0.0, (latency - target) / target)
        violation = (latency > self.limit_ms).astype(np.float64)
        cost = float(weights @ (reward.alpha * degradation + reward.gamma * violation))
        progress = float(weights @ self.delivered_mbps(fg_samples, alloc_mbps))
        return cost - progress_value * progress

    def choose_progress_value(
        self,
        fg_samples: np.ndarray,
        weights: np.ndarray,
        progress_value: float,
        lowest_index: int = 0,
    ) -> int:
        """Index in ``[lowest_index, N_ACTIONS)`` with the lowest :meth:`score`.

        Ties resolve to the smaller allocation.
        """
        best_index = lowest_index
        best_score = float("inf")
        for i in range(lowest_index, N_ACTIONS):
            value = self.score(fg_samples, weights, _ACTIONS[i], progress_value)
            if value < best_score - _TIE_TOLERANCE:
                best_score, best_index = value, i
        return best_index


@dataclass(frozen=True)
class BeliefConfig:
    """Parameters of the foreground belief (see the module docstring).

    Attributes:
        level_window: Intervals over which the load level is a median.
        window: Number of residuals kept.
        last_weight: Probability mass on the newest observed value; the rest is
            spread evenly over the residual scenarios. ``1.0`` is persistence.
    """

    level_window: int = DEFAULT_LEVEL_WINDOW
    window: int = DEFAULT_WINDOW
    last_weight: float = DEFAULT_LAST_WEIGHT

    def __post_init__(self) -> None:
        if self.level_window < 1:
            raise ValueError(f"level_window must be >= 1, got {self.level_window!r}")
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window!r}")
        if not 0.0 <= self.last_weight <= 1.0:
            raise ValueError(f"last_weight must be in [0, 1], got {self.last_weight!r}")


class _Throttle:
    """Shared machinery: a foreground belief and a selector over allocations."""

    def __init__(
        self,
        config: SimConfig | None,
        selection: Selection,
        belief: BeliefConfig,
        progress_value: float,
        ceiling_factor: float,
    ) -> None:
        if selection not in ("progress_value", "ceiling"):
            raise ValueError(f"selection must be 'progress_value' or 'ceiling', got {selection!r}")
        if not progress_value >= 0.0:
            raise ValueError(f"progress_value must be >= 0, got {progress_value!r}")
        self.config: SimConfig = config if config is not None else SimConfig()
        self.selection: Selection = selection
        self.belief_config: BeliefConfig = belief
        self.progress_value: float = float(progress_value)
        self.latency_rule: LatencyRule = LatencyRule(self.config, ceiling_factor)
        self._recent: deque[float] = deque(maxlen=belief.level_window)
        self._residuals: deque[float] = deque(maxlen=belief.window)

    def reset(self) -> None:
        """Clear the foreground history."""
        self._recent.clear()
        self._residuals.clear()

    def _belief(self, o: Observation) -> tuple[np.ndarray, np.ndarray]:
        """Record the newest foreground value; return ``(samples, weights)``.

        The newest observed value is last in ``samples``. Before a residual
        exists (the first call) the belief is that value alone.
        """
        x = o.fg_bw
        if self._recent:
            self._residuals.append(x - float(np.median(np.fromiter(self._recent, dtype=np.float64))))
        self._recent.append(x)
        last_weight = self.belief_config.last_weight
        if not self._residuals or last_weight >= 1.0:
            return np.array([x]), np.array([1.0])
        level = float(np.median(np.fromiter(self._recent, dtype=np.float64)))
        residuals = np.fromiter(self._residuals, dtype=np.float64)
        samples = np.maximum(0.0, level + residuals)
        weights = np.full(samples.size, (1.0 - last_weight) / samples.size)
        return np.append(samples, x), np.append(weights, last_weight)

    def _select(self, samples: np.ndarray, weights: np.ndarray, lowest_index: int) -> int:
        if self.selection == "progress_value":
            return self.latency_rule.choose_progress_value(samples, weights, self.progress_value, lowest_index)
        return self.latency_rule.choose_ceiling(float(samples[-1]), lowest_index)


class FixedRate:
    """Always request one allocation from ``ACTIONS_MBPS``.

    Four instances (one per action) bracket the achievable range: the slowest
    has the lowest foreground cost and the longest exposure window, the fastest
    the opposite. The safety filter may still raise the allocation when the
    risk budget requires it; the policy itself never reads the observation.
    """

    deployable: bool = True

    def __init__(self, mbps: int) -> None:
        if mbps not in ACTIONS_MBPS:
            raise ValueError(f"mbps must be one of {ACTIONS_MBPS}, got {mbps!r}")
        self.mbps: int = int(mbps)
        self._index: int = ACTIONS_MBPS.index(mbps)
        self.name: str = f"fixed_{self.mbps}"

    def __call__(self, obs: np.ndarray) -> int:
        return self._index

    def reset(self) -> None:
        """No per-episode state."""


class WorkloadOnlyThrottle(_Throttle):
    """Risk-blind throttle: chooses the allocation from foreground load alone.

    It uses the load rule described in the module docstring and nothing else.
    Observation fields 5 (``hazard``) and 6 (``hazard_uncertainty``) are never
    read, so changing them cannot change its action; that invariance is what
    makes it the ablation of the risk signal at the policy level.

    It shares every part of the load rule with :class:`RiskAwareRuleBased`
    (belief, selector, parameters), so the two differ only in the risk floor.

    The ablation is not clean at the environment level. The environment's
    safety filter still raises any request that violates the budget to the
    smallest feasible allocation, so the allocation actually applied is
    ``max(request, risk floor)`` whatever this policy knows. With the
    uncertainty margin ``k = 0``, :class:`RiskAwareRuleBased` and this policy
    differ in outcome only where the policy's own request would otherwise have
    been clipped (and in ``alloc_clipped_by_filter``). The filter, not this
    policy, keeps its exposure inside the budget.
    """

    deployable: bool = True

    def __init__(
        self,
        config: SimConfig | None = None,
        *,
        selection: Selection = "progress_value",
        belief: BeliefConfig | None = None,
        progress_value: float = DEFAULT_PROGRESS_VALUE,
        ceiling_factor: float = LATENCY_LIMIT_FACTOR,
        name: str = "workload_only_throttle",
    ) -> None:
        super().__init__(config, selection, belief or BeliefConfig(), progress_value, ceiling_factor)
        self.name: str = name

    def __call__(self, obs: np.ndarray) -> int:
        o = Observation.from_vector(obs, normalized=True)
        samples, weights = self._belief(o)
        return self._select(samples, weights, 0)


class RiskAwareRuleBased(_Throttle):
    """Strong rule-based baseline: the load rule inside the reliability budget.

    Decision rule, evaluated every interval from the observation alone:

    1. Inflate the hazard: ``h = hazard + k * hazard_uncertainty`` (``k``
       defaults to :data:`DEFAULT_HAZARD_UNCERTAINTY_K`).
    2. Find the risk floor with the safety filter's own function:
       ``apply_safety_filter(0, h, work_remaining)`` returns the smallest
       action whose exposure ``1 - exp(-h * T_remaining)`` is below
       ``RISK_BUDGET_EPS``. The feasible set is that action and every faster
       one, because exposure falls as the allocation rises. The exposure
       formula is the contract's and is not restated here.
    3. Among the feasible allocations choose by the load rule of the module
       docstring (``selection``).
    4. If no allocation is feasible even at the inflated hazard, the floor is
       the largest allocation, the filter's own degradation rule.

    Precedence is deliberate. Reliability is the hard constraint and latency is
    what is optimised inside it: when the feasible allocations all cost
    latency, the rule accepts the latency violation rather than relax the
    budget. This is the precedence the safety filter applies. The comparison
    against a learned policy would be unfair if the baseline could trade budget
    for latency while the learned policy could not.

    Why the load rule picks the best feasible allocation, not the slowest. The
    spec's wording, "the slowest risk-feasible action that keeps predicted
    latency within the limit", collapses to "the slowest risk-feasible
    action", because predicted latency rises with the allocation: the slowest
    member of the feasible set has the lowest latency, so the latency condition
    never changes the answer. That rule is :class:`SlowestRiskFeasible`. At
    baseline hazard all four actions are feasible, so it runs 20 MB/s for about
    167 hours on every workload, and any learned policy would beat it by
    rebuilding faster when load permits. A win of that kind needs no learning,
    so it would say nothing about the project's claim. Spare headroom is worth
    spending because the reward charges every degraded interval.

    Hazard uncertainty. ``k`` scales the margin ``hazard_uncertainty`` adds to
    the hazard before the exposure test. The default is ``k = 0``, so the
    uncertainty is supplied and available but not used. The reason is
    measured, not assumed. In this simulator failures are drawn from the same
    point estimate the observation reports, so the uncertainty carries no
    information about realised failures; a positive ``k`` only raises the risk
    floor and with it the latency cost, and the filter already keeps the
    budget on the point estimate. Measured on 100 tuning seeds (seeds 1000 to
    1099, simulated, random fleets, mean total reward): ``k = 1`` scored
    -6710.8 on ``peak`` against -6326.9 for a constant 20 MB/s, and -285.2 on
    ``diurnal`` against -269.4 for a constant 50 MB/s, while ``k = 0`` scored
    -6326.9 and -239.2. A baseline that a constant policy beats is the
    straw-man outcome this class exists to avoid, so ``k > 0`` is a
    sensitivity variant (``diagnostic_policies``), not the comparator. In a
    deployment whose failures do depend on the estimate's error the trade-off
    could differ. The uncertainty also rests on the contract's independence
    assumption and is optimistic under correlated failure.

    Where a learned policy could still beat it. The rule is close to the greedy
    optimum for what one observation and an in-episode load history can say: it
    uses every supplied signal, keeps the reliability precedence, and spends
    spare latency headroom. What it cannot do is plan across intervals. The
    remaining advantages are temporal: (a)
    anticipation, for example rebuilding hard during an idle stretch because a
    peak is coming, which requires learning the workload's periodicity from the
    repeating trace; (b) a better short-term forecast than the empirical belief
    above, for example that a burst of a given height tends to last several
    intervals; (c) multi-interval trade-offs that a one-interval rule does not
    weigh. If a learned policy does not beat this baseline, the honest
    conclusion is that anticipation is not worth much in this simulator, which
    is a result rather than a failure. The comparison carries the simulator's
    limits: uncalibrated latency, synthetic workloads, and an independence
    assumption in the hazard aggregate that makes the group look safer than it
    is.

    Hazard saturation. Field 5 saturates at ``1.8e-4`` per hour. When the true
    group hazard is above that, the policy sees ``1.8e-4``, underestimates the
    exposure of the slower allocations once the remaining work has shrunk, and
    may request an allocation the filter then raises. Clipping by the filter
    is therefore expected only on observations whose hazard field is saturated
    (``test_baselines.py`` checks this), and the filter, not the policy,
    enforces the budget there.
    """

    deployable: bool = True

    def __init__(
        self,
        config: SimConfig | None = None,
        *,
        k: float = DEFAULT_HAZARD_UNCERTAINTY_K,
        selection: Selection = "progress_value",
        belief: BeliefConfig | None = None,
        progress_value: float = DEFAULT_PROGRESS_VALUE,
        ceiling_factor: float = LATENCY_LIMIT_FACTOR,
        eps: float = RISK_BUDGET_EPS,
        name: str = "risk_aware_rule_based",
    ) -> None:
        if not k >= 0.0:
            raise ValueError(f"k must be >= 0, got {k!r}")
        super().__init__(config, selection, belief or BeliefConfig(), progress_value, ceiling_factor)
        self.k: float = float(k)
        self.eps: float = float(eps)
        self.name: str = name

    def risk_floor_index(self, o: Observation) -> int:
        """Smallest action satisfying the budget at the inflated hazard.

        Returns the largest index when no action does, which is the safety
        filter's degradation rule.
        """
        h = o.hazard + self.k * o.hazard_uncertainty
        return apply_safety_filter(0, h, o.work_remaining, self.eps).applied_index

    def __call__(self, obs: np.ndarray) -> int:
        o = Observation.from_vector(obs, normalized=True)
        samples, weights = self._belief(o)
        return self._select(samples, weights, self.risk_floor_index(o))


def peek_next_fg_mbps(env: Any) -> float:
    """Foreground bandwidth the next ``env.step`` will execute, in MB/s.

    ``RecoveryEnv`` exposes no look-ahead, so this reads its private fields:
    step ``t`` executes trace row ``(offset + t) % len(trace)`` and ``_t`` is
    the number of steps already taken. Those three attributes
    (``_trace``, ``_offset``, ``_t``) are an implementation detail of the
    simulator; ``test_baselines.py`` checks the result against the
    ``fg_bw_mbps`` the following step reports.

    Raises:
        RuntimeError: if the environment has not been reset.
    """
    trace = getattr(env, "_trace", None)
    if trace is None:
        raise RuntimeError("environment has no running episode; call reset() first")
    row = (int(env._offset) + int(env._t) + 1) % len(trace)
    return float(trace.fg_bw_mbps[row])


class OracleLookahead(RiskAwareRuleBased):
    """ORACLE: the risk-aware rule given the next interval's foreground load.

    NOT A DEPLOYABLE CONTROLLER. No real controller knows the load of the
    interval it is choosing for; this one reads it from the trace through
    :func:`peek_next_fg_mbps`, so its belief is the exact next value. It
    replaces only the belief; the risk floor, selector, parameters and
    precedence are those of :class:`RiskAwareRuleBased`. The harness binds it
    to the environment with :meth:`bind`.

    What it bounds. It bounds the value of knowing the next interval, that is
    of removing the forecast error of a one-interval rule. It does not bound
    the value of anticipating several intervals ahead: it is still a
    one-interval greedy rule and cannot rebuild hard now because a peak
    arrives later. A small gap to ``RiskAwareRuleBased`` therefore shows that
    next-interval knowledge is worth little; it does not show that a learned
    policy cannot gain from multi-interval anticipation. A large gap does not
    show a learnable gain either: what the oracle knows includes things no
    policy can predict, such as the interval in which a random burst starts.
    """

    deployable: bool = False

    def __init__(
        self,
        config: SimConfig | None = None,
        *,
        k: float = DEFAULT_HAZARD_UNCERTAINTY_K,
        selection: Selection = "progress_value",
        progress_value: float = DEFAULT_PROGRESS_VALUE,
        ceiling_factor: float = LATENCY_LIMIT_FACTOR,
        eps: float = RISK_BUDGET_EPS,
    ) -> None:
        label = "" if k == DEFAULT_HAZARD_UNCERTAINTY_K else f" k={k:g}"
        super().__init__(
            config,
            k=k,
            selection=selection,
            belief=BeliefConfig(level_window=1, window=1, last_weight=1.0),
            progress_value=progress_value,
            ceiling_factor=ceiling_factor,
            eps=eps,
            name=f"ORACLE_lookahead{label} (not deployable)",
        )
        self._env: Any = None

    def bind(self, env: Any) -> None:
        """Attach the environment whose trace is read."""
        self._env = env

    def _belief(self, o: Observation) -> tuple[np.ndarray, np.ndarray]:
        if self._env is None:
            raise RuntimeError("OracleLookahead is not bound to an environment; call bind(env)")
        return np.array([peek_next_fg_mbps(self._env)]), np.array([1.0])


class SlowestRiskFeasible:
    """Diagnostic: request the slowest allocation that satisfies the budget.

    The literal reading of "among the risk-feasible actions, prefer the slowest
    one that also keeps predicted latency within the limit". Because predicted
    latency rises with the allocation, the latency condition never changes the
    answer, and the result is the risk floor of :class:`RiskAwareRuleBased`.
    It is not a strong baseline and is reported only to show how much of the
    strong baseline's advantage comes from using spare headroom.
    """

    deployable: bool = True

    def __init__(
        self,
        config: SimConfig | None = None,
        *,
        k: float = DEFAULT_HAZARD_UNCERTAINTY_K,
        eps: float = RISK_BUDGET_EPS,
    ) -> None:
        self.config: SimConfig = config if config is not None else SimConfig()
        self._inner: RiskAwareRuleBased = RiskAwareRuleBased(self.config, k=k, eps=eps)
        self.name: str = "slowest_risk_feasible (literal spec)"

    def __call__(self, obs: np.ndarray) -> int:
        return self._inner.risk_floor_index(Observation.from_vector(obs, normalized=True))

    def reset(self) -> None:
        """No per-episode state."""


def fixed_rate_policies() -> list[FixedRate]:
    """One :class:`FixedRate` per entry of ``ACTIONS_MBPS``, slowest first."""
    return [FixedRate(a) for a in ACTIONS_MBPS]


def baseline_policies(config: SimConfig | None = None) -> list[Policy]:
    """Seven policies: the four fixed rates, the risk-blind throttle, the
    risk-aware rule, and last the oracle (an upper reference, not a comparator).
    """
    cfg = config if config is not None else SimConfig()
    return [
        *fixed_rate_policies(),
        WorkloadOnlyThrottle(cfg),
        RiskAwareRuleBased(cfg),
        OracleLookahead(cfg),
    ]


def diagnostic_policies(config: SimConfig | None = None) -> list[Policy]:
    """Extra policies reported alongside the baselines, never as comparators.

    The literal-spec selection rules, which differ from the default only in how
    they pick among the feasible allocations, and the risk-aware rule with an
    uncertainty margin ``k = 1``, whose difference from the default measures
    the margin's cost.
    """
    cfg = config if config is not None else SimConfig()
    return [
        SlowestRiskFeasible(cfg),
        RiskAwareRuleBased(cfg, selection="ceiling", name="risk_aware_ceiling_persistence (literal spec)"),
        RiskAwareRuleBased(cfg, k=1.0, name="risk_aware_rule_based k=1"),
    ]
