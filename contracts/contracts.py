"""Frozen interface contract for the recovery-bandwidth controller project.

This module is the single shared definition of the observation vector, the
action set, the reward, the required ``info`` keys, the termination reasons,
the hazard-model signature, the replay-file schema and the numeric constants
used by the four workstreams (RL agent and simulator, statistical risk model,
data pipeline, evaluation and report).

It is frozen at ``CONTRACT_VERSION``. Any change requires sign-off from all
four teammates and a row in the changelog of ``contracts/INTERFACES.md``.

Every numeric value below is a design decision, not a measurement. Nothing in
this module implies that a model has been trained, a simulator calibrated or a
result obtained.

Dependencies: Python 3.11+, numpy, standard library only.
"""

from __future__ import annotations

import enum
import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "CONTRACT_VERSION",
    "CONTROL_INTERVAL_S",
    "HORIZON_DAYS",
    "HORIZON_HOURS",
    "RISK_BUDGET_EPS",
    "LATENCY_LIMIT_FACTOR",
    "LATENCY_VIOLATION_FRACTION_MAX",
    "MAX_EPISODE_STEPS",
    "ACTIONS_MBPS",
    "N_ACTIONS",
    "GROUP_DATA_DRIVES",
    "GROUP_PARITY_DRIVES",
    "GROUP_SIZE",
    "GROUP_TOLERATED_FAILURES",
    "DRIVE_CAPACITY_GB",
    "REBUILD_WORK_GB",
    "BW_TOTAL_MBPS",
    "OBS_FIELDS",
    "OBS_SCALES",
    "OBS_DIM",
    "INFO_KEYS",
    "REPLAY_COLUMNS",
    "REPLAY_DTYPES",
    "DISK_INDEX_COLUMNS",
    "ContractViolation",
    "Observation",
    "TerminationReason",
    "termination_flags",
    "RewardConfig",
    "HazardModel",
    "action_to_mbps",
    "validate_info",
    "latency_degradation",
    "degraded_cost",
    "latency_violation",
    "compute_reward",
    "hazard_p_to_per_hour",
    "hazard_uncertainty_to_per_hour",
    "aggregate_group_hazard",
    "kbps_to_mbps",
    "validate_replay_grid",
    "validate_replay_frame",
    "validate_disk_index_frame",
    "to_replay_frame",
]


class ContractViolation(Exception):
    """Raised when a value crosses a workstream boundary and breaks the contract."""


# --------------------------------------------------------------------------
# Frozen numeric constants (design decisions; rationale in INTERFACES.md)
# --------------------------------------------------------------------------

CONTRACT_VERSION: str = "1.4.0"

CONTROL_INTERVAL_S: int = 300  # equals the Tencent trace sampling interval
HORIZON_DAYS: int = 30
HORIZON_HOURS: float = 720.0  # hazard model horizon H
# Bounds P(at least one additional failure before recovery completes), per
# episode. This is an exposure probability, not a data-loss probability.
RISK_BUDGET_EPS: float = 5e-3
LATENCY_LIMIT_FACTOR: float = 1.5  # L_max = LATENCY_LIMIT_FACTOR * latency_target
LATENCY_VIOLATION_FRACTION_MAX: float = 0.05
# 200 h at 300 s. Must exceed the slowest action's completion time
# (REBUILD_WORK_GB at min(ACTIONS_MBPS)), or a fixed-rate baseline can never finish.
MAX_EPISODE_STEPS: int = 2400
ACTIONS_MBPS: tuple[int, ...] = (20, 50, 80, 120)  # prototype set; illustrative, not measured device limits
N_ACTIONS: int = 4

GROUP_DATA_DRIVES: int = 8
GROUP_PARITY_DRIVES: int = 2
GROUP_TOLERATED_FAILURES: int = 2
GROUP_SIZE: int = GROUP_DATA_DRIVES + GROUP_PARITY_DRIVES  # 10; survivors at episode start: GROUP_SIZE - 1 = 9
DRIVE_CAPACITY_GB: float = 12000.0
REBUILD_WORK_GB: float = 12000.0  # one drive's worth of reconstruction
BW_TOTAL_MBPS: float = 200.0  # shared budget per group


# --------------------------------------------------------------------------
# Observation vector
# --------------------------------------------------------------------------

OBS_FIELDS: tuple[str, ...] = (
    "fg_read_iops",
    "fg_write_iops",
    "fg_bw",
    "latency_now",
    "latency_target",
    "hazard",
    "hazard_uncertainty",
    "redundancy_left",
    "progress",
    "work_remaining",
    "alloc_current",
    "bw_available",
)

OBS_DIM: int = 12

# Normalisation scales are frozen constants, never computed from data: a
# data-dependent scale would leak dataset statistics into the policy input.
OBS_SCALES: dict[str, float] = {
    "fg_read_iops": 20000.0,  # IOPS
    "fg_write_iops": 20000.0,  # IOPS
    "fg_bw": BW_TOTAL_MBPS,  # MB/s
    "latency_now": 50.0,  # ms
    "latency_target": 50.0,  # ms
    # Group-level aggregate: (GROUP_SIZE - 1) * 2e-5. 2e-5 per drive saturates at
    # ~8.3x the baseline per-drive hazard (2.41e-6 /h), the multiple at which
    # even the fastest action exceeds RISK_BUDGET_EPS.
    "hazard": 1.8e-4,  # per hour, sum over surviving drives
    "hazard_uncertainty": 1.8e-4,  # per hour, sum in quadrature over surviving drives
    "redundancy_left": float(GROUP_TOLERATED_FAILURES),  # count
    "progress": 1.0,  # fraction
    "work_remaining": REBUILD_WORK_GB,  # GB
    "alloc_current": float(max(ACTIONS_MBPS)),  # MB/s
    "bw_available": BW_TOTAL_MBPS,  # MB/s
}


@dataclass(frozen=True, slots=True)
class Observation:
    """One control-interval observation. Field order equals ``OBS_FIELDS``.

    Units and suppliers:

    - ``fg_read_iops``, ``fg_write_iops`` (IOPS), ``fg_bw`` (MB/s): data pipeline.
    - ``latency_now``, ``latency_target`` (ms): simulator.
    - ``hazard``, ``hazard_uncertainty`` (per hour): risk model. Group-level
      aggregates over the surviving drives, built with
      :func:`aggregate_group_hazard` from per-drive horizon probabilities.
      Exposure remaining is ``1 - exp(-hazard * T_remaining)`` with
      ``T_remaining`` in hours.
    - ``redundancy_left`` (count), ``progress`` (fraction), ``work_remaining``
      (GB), ``alloc_current`` (MB/s), ``bw_available`` (MB/s): simulator.

    ``redundancy_left``: a rebuild episode begins after one drive has failed,
    so it starts at 1; a second failure takes it to 0 and a third is data
    loss. The value 2 is the healthy state and never occurs during an episode;
    it exists in the valid range and as the normalisation scale so that the
    field means the same thing outside a rebuild.
    """

    fg_read_iops: float
    fg_write_iops: float
    fg_bw: float
    latency_now: float
    latency_target: float
    hazard: float
    hazard_uncertainty: float
    redundancy_left: float
    progress: float
    work_remaining: float
    alloc_current: float
    bw_available: float

    def to_vector(self, normalized: bool = True) -> np.ndarray:
        """Return the observation as a float32 array of shape ``(12,)``.

        With ``normalized=True`` each entry is ``clip(value / scale, 0.0, 1.0)``
        using the frozen ``OBS_SCALES``: a negative value becomes 0.0 and a
        value above its scale saturates at 1.0. With ``normalized=False`` the
        raw values are returned in index order, unclipped.

        This method does not validate. NaN, inf and negative values pass
        through (NaN stays NaN); :meth:`validate` is the only gate for them and
        must be called explicitly. Values are computed in float64 and cast to
        float32, so an exact round trip holds only for float32-representable
        values; realistic decimals match to about 1e-6 relative.
        """
        raw = np.array([getattr(self, name) for name in OBS_FIELDS], dtype=np.float64)
        if normalized:
            scales = np.array([OBS_SCALES[name] for name in OBS_FIELDS], dtype=np.float64)
            raw = np.clip(raw / scales, 0.0, 1.0)
        return raw.astype(np.float32)

    @classmethod
    def from_vector(cls, v: Sequence[float] | np.ndarray, normalized: bool = True) -> "Observation":
        """Inverse of :meth:`to_vector`.

        With ``normalized=True`` each entry is multiplied back by its scale.
        Clipping in :meth:`to_vector` makes the round trip lossy for raw values
        above a scale; this is intended, because the policy never sees them.

        Raises:
            ContractViolation: if the input does not have exactly 12 entries.
        """
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 1 or arr.shape[0] != OBS_DIM:
            raise ContractViolation(
                f"observation vector must have shape ({OBS_DIM},), got {arr.shape}"
            )
        if normalized:
            arr = arr * np.array([OBS_SCALES[name] for name in OBS_FIELDS], dtype=np.float64)
        return cls(*(float(x) for x in arr))

    def validate(self) -> None:
        """Check raw (unnormalised) values.

        Raises:
            ContractViolation: on NaN or inf in any field, any negative value,
                ``progress`` outside ``[0, 1]``, ``redundancy_left`` outside
                ``[0, GROUP_TOLERATED_FAILURES]``, or ``latency_target <= 0``.
        """
        for name in OBS_FIELDS:
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ContractViolation(f"observation field {name!r} is not finite: {value!r}")
            if value < 0.0:
                raise ContractViolation(f"observation field {name!r} is negative: {value!r}")
        if not 0.0 <= self.progress <= 1.0:
            raise ContractViolation(f"progress must be in [0, 1], got {self.progress!r}")
        if not 0.0 <= self.redundancy_left <= float(GROUP_TOLERATED_FAILURES):
            raise ContractViolation(
                f"redundancy_left must be in [0, {GROUP_TOLERATED_FAILURES}], "
                f"got {self.redundancy_left!r}"
            )
        if self.latency_target <= 0.0:
            raise ContractViolation(f"latency_target must be > 0, got {self.latency_target!r}")


# --------------------------------------------------------------------------
# Action set
# --------------------------------------------------------------------------


def action_to_mbps(index: int) -> float:
    """Map a discrete action index to a rebuild bandwidth in MB/s.

    The set ``ACTIONS_MBPS`` is a prototype set, illustrative rather than
    measured device limits. Environments must reject bad actions through this
    function so the failure is a ``ContractViolation``, never a bare
    ``ValueError`` or ``IndexError``.

    Raises:
        ContractViolation: if ``index`` is not an integer (``bool`` is rejected)
            or lies outside ``[0, N_ACTIONS)``.
    """
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, numbers.Integral):
        raise ContractViolation(f"action index must be an int, got {type(index).__name__}")
    i = int(index)
    if not 0 <= i < N_ACTIONS:
        raise ContractViolation(f"action index {i} out of range [0, {N_ACTIONS})")
    return float(ACTIONS_MBPS[i])


# --------------------------------------------------------------------------
# Step info
# --------------------------------------------------------------------------

INFO_KEYS: tuple[str, ...] = (
    "latency_p95",
    "latency_p99",
    "violated_latency",
    "p_failure_before_complete",
    "violated_risk_budget",
    "alloc_clipped_by_filter",
    "recovery_complete",
)

_INFO_BOOL_KEYS: tuple[str, ...] = (
    "violated_latency",
    "violated_risk_budget",
    "alloc_clipped_by_filter",
    "recovery_complete",
)
_INFO_NUMERIC_KEYS: tuple[str, ...] = (
    "latency_p95",
    "latency_p99",
    "p_failure_before_complete",
)


class TerminationReason(enum.Enum):
    """Why an episode ended.

    All three reasons are distinguishable in ``info``: when an episode ends,
    the ``termination_reason`` info key carries this enum's ``.value`` (a
    ``str``, not the enum member, so ``info`` stays JSON-serialisable).

    - ``RECOVERY_COMPLETE``: rebuild finished without data loss.
    - ``TERMINAL_FAILURE``: data loss. Redundancy was already exhausted
      (``redundancy_left == 0``) and a further drive failure occurred before
      the rebuild finished. An 8+2 group with one failed drive survives a
      second failure and loses data on a third.
    - ``HORIZON_EXPIRED``: ``MAX_EPISODE_STEPS`` elapsed with neither of the above.

    ``TERMINAL_FAILURE`` (loss) is deliberately a different event from the one
    ``RISK_BUDGET_EPS`` bounds. ``p_failure_before_complete`` is the probability
    of at least one additional failure before rebuild completes (exposure); the
    group may survive that failure. Exposure is not loss.
    """

    RECOVERY_COMPLETE = "recovery_complete"
    TERMINAL_FAILURE = "terminal_failure"
    HORIZON_EXPIRED = "horizon_expired"


def termination_flags(reason: TerminationReason) -> tuple[bool, bool]:
    """Return ``(terminated, truncated)`` for a termination reason.

    ``RECOVERY_COMPLETE`` and ``TERMINAL_FAILURE`` -> ``(True, False)``;
    ``HORIZON_EXPIRED`` -> ``(False, True)``.

    The episode horizon is an artificial cutoff, not an absorbing state.
    Reporting horizon expiry as ``terminated=True`` makes a value-based learner
    bootstrap ``V(s') = 0`` at the cutoff and under-value states that lead to
    long rebuilds, which biases the agent against slow allocations. Horizon
    expiry must be truncation, so the learner still bootstraps from ``s'``.

    Raises:
        ContractViolation: if ``reason`` is not a :class:`TerminationReason`.
    """
    if not isinstance(reason, TerminationReason):
        raise ContractViolation(f"reason must be a TerminationReason, got {type(reason).__name__}")
    if reason is TerminationReason.HORIZON_EXPIRED:
        return (False, True)
    return (True, False)


def _is_real_bool(x: Any) -> bool:
    return type(x) is bool


def _is_real_number(x: Any) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, (bool, np.bool_))


def validate_info(info: Mapping[str, Any]) -> None:
    """Check the per-step ``info`` mapping against ``INFO_KEYS``.

    These keys are required because they are computed independently of the
    reward. A reward penalty on latency or risk therefore can never be mistaken
    for evidence that the corresponding constraint held: the constraint record
    is read from these keys, not inferred from reward values.

    Rules: every key in ``INFO_KEYS`` is present; the four flag keys are
    built-in ``bool`` (numpy ``bool_`` and ``int`` are rejected); the three
    numeric keys are finite and non-negative; ``p_failure_before_complete`` is
    in ``[0, 1]``. If ``termination_reason`` is present and not ``None`` it
    must be the ``.value`` string of a :class:`TerminationReason` (the enum
    member itself is rejected, to keep ``info`` JSON-serialisable).

    Additional keys are permitted and are never checked or rejected, so a lane
    may log its own diagnostics. Numeric keys accept numpy floating scalars;
    the four flag keys reject numpy bools.

    Raises:
        ContractViolation: naming the offending key.
    """
    if not isinstance(info, Mapping):
        raise ContractViolation(f"info must be a mapping, got {type(info).__name__}")
    for key in INFO_KEYS:
        if key not in info:
            raise ContractViolation(f"info key {key!r} is missing")
    for key in _INFO_BOOL_KEYS:
        if not _is_real_bool(info[key]):
            raise ContractViolation(
                f"info key {key!r} must be a built-in bool, got "
                f"{type(info[key]).__module__}.{type(info[key]).__qualname__}"
            )
    for key in _INFO_NUMERIC_KEYS:
        value = info[key]
        if not _is_real_number(value):
            raise ContractViolation(
                f"info key {key!r} must be a real number, got {type(value).__name__}"
            )
        if not math.isfinite(value) or value < 0.0:
            raise ContractViolation(f"info key {key!r} must be finite and non-negative, got {value!r}")
    if not 0.0 <= info["p_failure_before_complete"] <= 1.0:
        raise ContractViolation(
            "info key 'p_failure_before_complete' must be in [0, 1], "
            f"got {info['p_failure_before_complete']!r}"
        )
    reason = info.get("termination_reason")
    if reason is not None:
        if not isinstance(reason, str) or reason not in {r.value for r in TerminationReason}:
            raise ContractViolation(
                "info key 'termination_reason' must be the .value string of a "
                f"TerminationReason, got {reason!r}"
            )


# --------------------------------------------------------------------------
# Reward
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardConfig:
    """Reward coefficients.

    Coefficients live in configuration and are passed in; they are never
    hardcoded at a call site.
    """

    alpha: float = 1.0  # weight on latency degradation D
    beta: float = 0.1  # weight on degraded-state cost C
    gamma: float = 10.0  # weight on violation term V

    @classmethod
    def from_dict(cls, d: Mapping[str, float]) -> "RewardConfig":
        """Build from a mapping; missing keys take defaults.

        Raises:
            ContractViolation: on any key that is not a field of this class.
        """
        allowed = {f.name for f in fields(cls)}
        unknown = set(d) - allowed
        if unknown:
            raise ContractViolation(f"unknown RewardConfig keys: {sorted(unknown)}")
        return cls(**{k: float(v) for k, v in d.items()})


def latency_degradation(latency_now: float, latency_target: float) -> float:
    """Return ``max(0, (L - L0) / L0)`` with ``L = latency_now``, ``L0 = latency_target``.

    Inputs may be any real number, including numpy scalars. They are converted
    to ``float`` before computing, so the arithmetic is float64 and the result
    is a builtin ``float`` whatever the input dtype.

    Both latencies must be finite. ``max(0.0, nan)`` is ``0.0`` because every
    comparison with NaN is false, so a NaN ``latency_now`` (for example from an
    average over a control interval with no completed requests) would read as
    zero degradation: a failure that looks like a healthy system. It is
    rejected instead.

    Raises:
        ContractViolation: if an argument is not a real number, is non-finite,
            or ``latency_target <= 0``.
    """
    _require_real("latency_now", latency_now)
    _require_real("latency_target", latency_target)
    latency_now = float(latency_now)
    latency_target = float(latency_target)
    _require_finite("latency_now", latency_now)
    _require_finite("latency_target", latency_target)
    if not latency_target > 0.0:
        raise ContractViolation(f"latency_target must be > 0, got {latency_target!r}")
    return max(0.0, (latency_now - latency_target) / latency_target)


def degraded_cost(is_degraded: bool) -> float:
    """Return ``C``: 1.0 for a control interval in which the group is degraded, else 0.0.

    Degraded means at least one drive has failed and the rebuild is not yet
    complete (``redundancy_left < GROUP_TOLERATED_FAILURES``). Summed over an
    episode, ``C`` is the count of degraded intervals: a dimensionless exposure
    proxy with no hidden scaling.

    Accepts built-in ``bool`` and ``numpy.bool_``; ``int``, ``float``, ``str``
    and ``None`` are rejected. This is permissive on purpose: the natural call,
    ``degraded_cost(redundancy_left < GROUP_TOLERATED_FAILURES)``, yields
    ``numpy.bool_`` under numpy, and this is a compute helper inside the step
    loop. The strict built-in-``bool`` rule is kept at the recording boundary
    (:func:`validate_info`), where a numpy bool would hide a type error and
    serialise inconsistently.

    Raises:
        ContractViolation: if ``is_degraded`` is not a ``bool`` or ``numpy.bool_``.
    """
    if not isinstance(is_degraded, (bool, np.bool_)):
        raise ContractViolation(
            "is_degraded must be a bool or numpy.bool_, got "
            f"{type(is_degraded).__module__}.{type(is_degraded).__qualname__}"
        )
    return 1.0 if is_degraded else 0.0


def latency_violation(latency_now: float, latency_target: float) -> float:
    """Return ``V``: 1.0 if ``latency_now > LATENCY_LIMIT_FACTOR * latency_target``, else 0.0.

    ``V`` covers latency only. The reliability budget is enforced by the safety
    filter and checked at episode level through ``info``, not through ``V``.

    Inputs are converted to ``float`` before the comparison, so it is evaluated
    in float64 whatever the input dtype.

    Both latencies must be finite. A NaN ``latency_now`` makes the comparison
    false and would return ``0.0`` (no violation), reading as a healthy system
    exactly when the measurement is broken, so it is rejected.

    Raises:
        ContractViolation: if an argument is not a real number, is non-finite,
            or ``latency_target <= 0``.
    """
    _require_real("latency_now", latency_now)
    _require_real("latency_target", latency_target)
    latency_now = float(latency_now)
    latency_target = float(latency_target)
    _require_finite("latency_now", latency_now)
    _require_finite("latency_target", latency_target)
    if not latency_target > 0.0:
        raise ContractViolation(f"latency_target must be > 0, got {latency_target!r}")
    return 1.0 if latency_now > LATENCY_LIMIT_FACTOR * latency_target else 0.0


def compute_reward(
    d_latency: float,
    c_degraded: float,
    v_violation: float,
    cfg: RewardConfig = RewardConfig(),
) -> float:
    """Return ``-(alpha * D + beta * C + gamma * V)``.

    ``D`` is :func:`latency_degradation`, ``C`` is :func:`degraded_cost` and
    ``V`` is :func:`latency_violation`. All three are dimensionless and must be
    finite and non-negative; callers must obtain ``C`` and ``V`` from those
    helpers rather than defining their own.

    The reward only shapes behaviour. Reliability and latency limits are
    constraints recorded in ``info`` (see :func:`validate_info`), not terms
    whose weights could trade one against another. ``info`` is authoritative.

    The terms and coefficients are converted to ``float`` before computing, so
    the arithmetic is float64 and the result is a builtin ``float`` whatever
    the input dtype.

    Raises:
        ContractViolation: if any term is not a real number, or is non-finite
            or negative.
    """
    terms: list[float] = []
    for name, value in (
        ("d_latency", d_latency),
        ("c_degraded", c_degraded),
        ("v_violation", v_violation),
    ):
        _require_real(name, value)
        value = float(value)
        if not math.isfinite(value) or value < 0.0:
            raise ContractViolation(f"{name} must be finite and non-negative, got {value!r}")
        terms.append(value)
    d, c, v = terms
    return -(float(cfg.alpha) * d + float(cfg.beta) * c + float(cfg.gamma) * v)


# --------------------------------------------------------------------------
# Hazard interface
# --------------------------------------------------------------------------


def _require_real(name: str, value: Any) -> None:
    """Raise ``ContractViolation`` unless ``value`` is a real number (not a bool)."""
    if not _is_real_number(value):
        raise ContractViolation(f"{name} must be a real number, got {type(value).__name__}")


def _require_finite(name: str, value: float) -> None:
    """Raise ``ContractViolation`` unless ``value`` is finite."""
    if not math.isfinite(value):
        raise ContractViolation(f"{name} must be finite, got {value!r}")


def hazard_p_to_per_hour(p_horizon: float, horizon_hours: float = HORIZON_HOURS) -> float:
    """Convert a horizon failure probability to a per-hour hazard.

    Returns ``1 - (1 - p) ** (1 / horizon_hours)``. The result is a discrete
    per-hour probability, not the continuous rate ``-ln(1 - p) / H``; the two
    are nearly equal for small ``p``.

    Assumption: the hazard rate is constant across the horizon. This is what
    makes a 30-day probability usable as a per-hour rate at rebuild timescales
    (hours). It is a modelling assumption, not a measurement.

    Arguments are converted to ``float`` before computing, so the arithmetic is
    float64 and the result is a builtin ``float`` whatever the input dtype.

    ``horizon_hours`` must be finite as well as positive. An infinite horizon
    makes the exponent ``1 / H`` zero and silently returns a zero hazard, so a
    broken input would read as a perfectly safe group instead of raising.

    Raises:
        ContractViolation: unless both arguments are real numbers with
            ``0 <= p_horizon < 1`` and ``horizon_hours`` finite and ``> 0``.
    """
    _require_real("p_horizon", p_horizon)
    _require_real("horizon_hours", horizon_hours)
    p_horizon = float(p_horizon)
    horizon_hours = float(horizon_hours)
    if not 0.0 <= p_horizon < 1.0:
        raise ContractViolation(f"p_horizon must satisfy 0 <= p < 1, got {p_horizon!r}")
    if not (math.isfinite(horizon_hours) and horizon_hours > 0.0):
        raise ContractViolation(f"horizon_hours must be finite and > 0, got {horizon_hours!r}")
    # Algebraically equal to 1 - (1 - p) ** (1 / H); expm1/log1p keep precision for small p.
    return -math.expm1(math.log1p(-p_horizon) / horizon_hours)


def hazard_uncertainty_to_per_hour(
    sigma_horizon: float,
    p_horizon: float,
    horizon_hours: float = HORIZON_HOURS,
) -> float:
    """Propagate a horizon standard deviation to a per-hour standard deviation.

    ``sigma_horizon`` is one standard deviation of the estimate of
    ``p_horizon``, over the same horizon. Arguments are converted to ``float``
    before computing, so the arithmetic is float64 and the result is a builtin
    ``float`` whatever the input dtype (a numpy float32 input does not yield a
    float32 result).

    This replaces applying :func:`hazard_p_to_per_hour` to the uncertainty.
    ``1 - (1 - p) ** (1 / H)`` is a probability map: it converts a probability
    over H into a probability over one hour. Applied to a standard deviation it
    has no statistical meaning, because a dispersion does not convert the way
    the quantity it disperses does. First-order error propagation (the delta
    method) is the correct transformation. With
    ``h = f(p) = 1 - (1 - p) ** (1 / H)``::

        dh/dp   = (1 / H) * (1 - p) ** (1 / H - 1)
        sigma_h = sigma_p * dh/dp

    The derivative is evaluated at the same drive's own ``p_horizon``. For
    small ``p`` it tends to ``1 / H``, so both approaches give roughly
    ``sigma_horizon / 720`` at the default horizon and the numerical difference
    is negligible in the expected regime. The delta method is nonetheless the
    correct propagation, remains correct when ``p`` is not small, and is
    defensible as a method, whereas pushing a standard deviation through a
    probability map is not. It is a first-order approximation that ignores the
    curvature of ``f``.

    ``sigma_horizon`` must lie in ``[0, 0.5]``: the standard deviation of a
    quantity bounded in ``[0, 1]`` cannot exceed 0.5, so a larger value
    indicates a broken or mis-scaled estimate rather than genuine uncertainty,
    and accepting it silently would let a miscalibrated risk model inflate
    observation field 6 with nothing noticing. Non-finite values of
    ``sigma_horizon``, ``p_horizon`` and ``horizon_hours`` are rejected. An
    infinite horizon makes the derivative's ``1 / H`` factor zero and silently
    returns a zero uncertainty, which would read as certainty.

    Raises:
        ContractViolation: unless all three arguments are real numbers,
            ``sigma_horizon`` is finite with ``0 <= sigma_horizon <= 0.5``,
            ``p_horizon`` is finite with ``0 <= p_horizon < 1``, and
            ``horizon_hours`` is finite and ``> 0``.
    """
    _require_real("sigma_horizon", sigma_horizon)
    _require_real("p_horizon", p_horizon)
    _require_real("horizon_hours", horizon_hours)
    sigma_horizon = float(sigma_horizon)
    p_horizon = float(p_horizon)
    horizon_hours = float(horizon_hours)
    if not (math.isfinite(sigma_horizon) and 0.0 <= sigma_horizon <= 0.5):
        raise ContractViolation(
            f"sigma_horizon must be finite and satisfy 0 <= sigma <= 0.5, got {sigma_horizon!r}"
        )
    if not (math.isfinite(p_horizon) and 0.0 <= p_horizon < 1.0):
        raise ContractViolation(
            f"p_horizon must be finite and satisfy 0 <= p < 1, got {p_horizon!r}"
        )
    if not (math.isfinite(horizon_hours) and horizon_hours > 0.0):
        raise ContractViolation(f"horizon_hours must be finite and > 0, got {horizon_hours!r}")
    derivative = (1.0 / horizon_hours) * (1.0 - p_horizon) ** (1.0 / horizon_hours - 1.0)
    return sigma_horizon * derivative


def _unpack_pair(index: int, pair: Any) -> tuple[Any, Any]:
    """Return ``(p_horizon, uncertainty)`` from one ``per_drive`` element.

    The test is structural, not nominal: the element must be sized and
    indexable, of length exactly 2, and not a ``str``, ``bytes`` or mapping.
    A tuple, a list and a 1-D ``numpy.ndarray`` of length 2 all qualify, so a
    numpy-native risk model can pass its output without an unpacking shim. Both
    members must be real numbers; ``bool`` and ``numpy.bool_`` are rejected so
    a truth value never reaches a probability field.

    Raises:
        ContractViolation: naming ``per_drive[index]`` and the reason.
    """
    if (
        isinstance(pair, (str, bytes, Mapping))
        or not hasattr(pair, "__len__")
        or not hasattr(pair, "__getitem__")
        or (isinstance(pair, np.ndarray) and pair.ndim != 1)
    ):
        shape = f" with shape {pair.shape}" if isinstance(pair, np.ndarray) else ""
        raise ContractViolation(
            f"per_drive[{index}] must be a sized, indexable (p_horizon, uncertainty) pair, "
            f"got {type(pair).__name__}{shape}"
        )
    if len(pair) != 2:
        raise ContractViolation(
            f"per_drive[{index}] must have exactly 2 entries (p_horizon, uncertainty), "
            f"got {len(pair)}"
        )
    p, u = pair[0], pair[1]
    for label, member in (("p_horizon", p), ("uncertainty", u)):
        if not _is_real_number(member):
            raise ContractViolation(
                f"per_drive[{index}] {label} must be a real number, got {type(member).__name__}"
            )
    return p, u


def aggregate_group_hazard(
    per_drive: Sequence[Sequence[float]] | np.ndarray,
    horizon_hours: float = HORIZON_HOURS,
) -> tuple[float, float]:
    """Aggregate per-drive risk-model output into observation fields 5 and 6.

    ``per_drive`` holds one ``(p_horizon, uncertainty)`` pair per surviving
    drive, as a :class:`HazardModel` emits them (both over ``horizon_hours``;
    ``uncertainty`` is one standard deviation of the ``p_horizon`` estimate).
    It is a sequence of such pairs, or an ``(n, 2)`` ``numpy.ndarray`` whose
    rows are the pairs. Each pair may be a tuple, a list or a 1-D array of
    length 2.
    Each ``p_horizon`` is converted with :func:`hazard_p_to_per_hour` and each
    ``uncertainty`` with :func:`hazard_uncertainty_to_per_hour` evaluated at
    that drive's own ``p_horizon``. Returns ``(hazard, hazard_uncertainty)`` in
    per-hour units:

    - ``hazard`` is the sum of the per-drive rates, so that the exposure the
      safety filter bounds is ``1 - exp(-hazard * T_remaining)``, computable
      from the observation alone.
    - ``hazard_uncertainty`` is ``sqrt(sum of squares)`` of the converted
      per-drive uncertainties.

    Both rules assume the per-drive estimates are independent. Correlated
    failure (shared batch, temperature, vibration, age) is well documented in
    fleet data; under it the aggregate understates tail risk, i.e. makes the
    group look safer than it is, and the uncertainty is optimistic as well.

    Raises:
        ContractViolation: on an empty sequence, on more than ``GROUP_SIZE - 1``
            entries (an ``(n, 2)`` array included), on a ``per_drive`` array
            that is not 2-D with two columns, on a ``per_drive`` that is not
            iterable, on an element that is not a sized, indexable pair of
            exactly two real numbers (``None``, a scalar, a string, a 2-D array
            row, a wrong length, a non-numeric or ``bool`` member), on a
            non-finite or non-positive ``horizon_hours``,
            or on a ``p_horizon`` or ``uncertainty`` that breaks its bounds
            (``0 <= p_horizon < 1``, ``0 <= uncertainty <= 0.5``). The message
            names the pair index and what was wrong, or which of the two
            failed and the bound it broke.
    """
    _require_real("horizon_hours", horizon_hours)
    if not (math.isfinite(horizon_hours) and horizon_hours > 0.0):
        raise ContractViolation(f"horizon_hours must be finite and > 0, got {horizon_hours!r}")
    if isinstance(per_drive, np.ndarray):
        if per_drive.ndim != 2 or per_drive.shape[1] != 2:
            raise ContractViolation(
                f"per_drive as an array must have shape (n, 2), got {per_drive.shape}"
            )
    elif isinstance(per_drive, (str, bytes, Mapping)):
        raise ContractViolation(
            f"per_drive must be a sequence of (p_horizon, uncertainty) pairs, "
            f"got {type(per_drive).__name__}"
        )
    try:
        entries = list(per_drive)
    except TypeError as exc:
        raise ContractViolation(
            f"per_drive must be a sequence of (p_horizon, uncertainty) pairs, "
            f"got {type(per_drive).__name__}"
        ) from exc
    if not entries:
        raise ContractViolation("per_drive is empty")
    if len(entries) > GROUP_SIZE - 1:
        raise ContractViolation(
            f"per_drive has {len(entries)} entries, at most GROUP_SIZE - 1 = {GROUP_SIZE - 1} allowed"
        )
    rates: list[float] = []
    uncertainties: list[float] = []
    for i, pair in enumerate(entries):
        p, u = _unpack_pair(i, pair)
        try:
            rates.append(hazard_p_to_per_hour(p, horizon_hours))
        except ContractViolation as exc:
            raise ContractViolation(f"per_drive[{i}] p_horizon: {exc}") from exc
        try:
            uncertainties.append(hazard_uncertainty_to_per_hour(u, p, horizon_hours))
        except ContractViolation as exc:
            raise ContractViolation(f"per_drive[{i}] uncertainty: {exc}") from exc
    return (math.fsum(rates), math.sqrt(math.fsum(u * u for u in uncertainties)))


def kbps_to_mbps(kbps: float) -> float:
    """Convert KB/s to MB/s using the decimal convention (divide by 1000).

    The data pipeline stores source KB/s unconverted so its files remain a
    faithful record of the source. Conversion happens here, at the adapter
    boundary, exactly once.

    The input is converted to ``float`` before dividing, so the result is a
    builtin ``float`` computed in float64 whatever the input dtype.

    Raises:
        ContractViolation: if ``kbps`` is not a real number.
    """
    _require_real("kbps", kbps)
    return float(kbps) / 1000.0


@runtime_checkable
class HazardModel(Protocol):
    """Risk-model callable consumed by the simulator.

    Returns ``(p_horizon, uncertainty)``, both over the horizon
    ``horizon_hours`` (default ``HORIZON_HOURS``). ``uncertainty`` is one
    standard deviation of the estimate of ``p_horizon``, or ``0.0`` when the
    risk model provides none; it must lie in ``[0, 0.5]``. It is a dispersion
    measure, not a probability. ``p_horizon`` is a probability over H, whereas
    observation field 5 (``hazard``) is a per-hour rate.
    :func:`hazard_p_to_per_hour` is the one sanctioned bridge between a horizon
    probability and a per-hour rate, and :func:`hazard_uncertainty_to_per_hour`
    is the one for its uncertainty.
    """

    def __call__(
        self, features: Mapping[str, float], horizon_hours: float = HORIZON_HOURS
    ) -> tuple[float, float]: ...


# --------------------------------------------------------------------------
# Replay schema
# --------------------------------------------------------------------------

REPLAY_COLUMNS: tuple[str, ...] = (
    "t_seconds",
    "read_iops",
    "write_iops",
    "read_bw_kbps",
    "write_bw_kbps",
    "disk_usage_mb",
)

# Required numpy dtype of every replay column; the keys equal REPLAY_COLUMNS.
# All six are integers: the source fields are integer counts as distributed,
# and the aggregate of N disk traces into one group load is a sum of integers,
# so it stays integral. Pinning the dtype keeps a synthetic fallback generator
# and the real pipeline byte-comparable, so either can feed the simulator
# interchangeably and a consumer cannot behave differently depending on which
# source produced its input.
REPLAY_DTYPES: dict[str, np.dtype] = {
    "t_seconds": np.dtype("int64"),
    "read_iops": np.dtype("int64"),
    "write_iops": np.dtype("int64"),
    "read_bw_kbps": np.dtype("int64"),
    "write_bw_kbps": np.dtype("int64"),
    "disk_usage_mb": np.dtype("int64"),
}

# Column order is the field order of the Tencent Cloud Volume Workload Traces
# README for the subscription table, confirmed by the README's own example row
# (a2049546-cd8b-49ff-b863-c503baa79429,50,1,1,0,2.0,4.0, where 50 is the
# capacity in GB at index 1). The subscription file is headerless, so a
# positional reader depends on this order exactly. README names -> contract
# names: "product type" = disk_attr, "cloud disk type" = disk_type.
#
# RULE: if this order is ever changed again, verify it against the dataset's own
# README, never against a project document. The earlier order was copied from a
# secondhand schema description (data_pipeline/CLAUDE.md section 5), was wrong,
# and had to be corrected (see the latest INTERFACES.md changelog row). A wrong
# position does not raise; it silently mis-maps every later field.
DISK_INDEX_COLUMNS: tuple[str, ...] = (
    "disk_uid",
    "disk_capacity",
    "disk_attr",
    "disk_type",
    "user_type",
    "vm_cpu",
    "vm_memory",
)


def validate_replay_grid(t_seconds: Sequence[int] | np.ndarray) -> None:
    """Check the time grid of a replay file.

    Replay files carry episode-relative ``t_seconds`` starting at 0, not Unix
    epoch. Rebasing happens in the data pipeline, never in a consumer.

    Raises:
        ContractViolation: unless the sequence is non-empty, all entries are
            integers, it starts at 0, has no duplicates and strictly increases
            by exactly ``CONTROL_INTERVAL_S``.
    """
    values = list(t_seconds)
    if not values:
        raise ContractViolation("replay grid is empty")
    for i, t in enumerate(values):
        if isinstance(t, (bool, np.bool_)) or not isinstance(t, numbers.Integral):
            raise ContractViolation(f"t_seconds[{i}] must be an int, got {type(t).__name__}")
    ints = [int(t) for t in values]
    if ints[0] != 0:
        raise ContractViolation(f"replay grid must start at 0, got {ints[0]}")
    for i in range(1, len(ints)):
        step = ints[i] - ints[i - 1]
        if step == 0:
            raise ContractViolation(f"duplicate timestamp {ints[i]} at index {i}")
        if step != CONTROL_INTERVAL_S:
            raise ContractViolation(
                f"step at index {i} is {step} s, expected exactly {CONTROL_INTERVAL_S} s"
            )


def validate_replay_frame(df: Any) -> None:
    """Check a whole replay file held as a pandas ``DataFrame``.

    One call checks the schema and the time grid, so it is the single check a
    producer runs before writing a replay file and a consumer runs before
    reading one. Checks, in order:

    1. ``df`` is a ``pandas.DataFrame``.
    2. The columns equal ``REPLAY_COLUMNS`` exactly, in that order (not merely
       as a set).
    3. No column contains a null (NaN, ``None`` or ``pandas.NA``). This is
       checked before the dtype because a null forces an integer column to
       float or object, and the null is the more useful thing to report.
    4. Every column's dtype equals ``REPLAY_DTYPES`` (all ``int64``).
    5. No value is negative.
    6. ``validate_replay_grid`` passes on the ``t_seconds`` column.

    Raises:
        ContractViolation: naming the offending column; a dtype mismatch names
            both the expected and the actual dtype.
    """
    # pandas is imported here, not at module top level, so that importing this
    # module keeps a numpy-and-stdlib-only surface for lanes that do not use it.
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise ContractViolation(f"replay frame must be a pandas DataFrame, got {type(df).__name__}")
    columns = list(df.columns)
    expected_columns = list(REPLAY_COLUMNS)
    if columns != expected_columns:
        missing = [c for c in expected_columns if c not in columns]
        unexpected = [c for c in columns if c not in expected_columns]
        if missing or unexpected:
            detail = f"missing {missing}, unexpected {unexpected}"
        elif len(columns) != len(set(columns)):
            detail = "duplicate column names"
        else:
            detail = "same columns in the wrong order"
        raise ContractViolation(
            f"replay columns must equal {expected_columns} in order, got {columns}: {detail}"
        )
    for name in REPLAY_COLUMNS:
        if df[name].isna().any():
            raise ContractViolation(f"replay column {name!r} contains null values")
    for name in REPLAY_COLUMNS:
        expected = REPLAY_DTYPES[name]
        actual = df[name].dtype
        if actual != expected:
            raise ContractViolation(
                f"replay column {name!r} has dtype {actual}, expected {expected}"
            )
    for name in REPLAY_COLUMNS:
        negative = df[name] < 0
        if negative.any():
            position = int(np.argmax(negative.to_numpy()))
            raise ContractViolation(
                f"replay column {name!r} has a negative value at row {position}: "
                f"{df[name].iloc[position]!r}"
            )
    validate_replay_grid(df["t_seconds"].to_numpy())


def validate_disk_index_frame(df: Any) -> None:
    """Check a whole disk-index table held as a pandas ``DataFrame``.

    The disk index is the per-disk metadata file, joined to replay data on
    ``disk_uid`` only where needed (``INTERFACES.md`` section 10, decision 2). One
    call checks the schema, so it is the single check a producer runs before
    writing the file and a consumer runs before reading it. Checks, in order:

    1. ``df`` is a ``pandas.DataFrame``.
    2. The columns equal ``DISK_INDEX_COLUMNS`` exactly, in that order (not
       merely as a set).
    3. No column contains a null (NaN, ``None`` or ``pandas.NA``). This is
       checked before the dtype because a null forces an integer column to
       float or object, and the null is the more useful thing to report.
    4. ``disk_uid`` is a string column (NumPy ``object`` holding only ``str``,
       or a pandas string dtype) and every other column has dtype ``int64``.
    5. ``disk_uid`` has no duplicate value.
    6. No value in the six numeric columns is negative.

    A frame with the right columns and dtypes but no rows passes: the contract
    does not set a minimum disk count.

    Raises:
        ContractViolation: naming the offending column; a dtype mismatch names
            both the expected and the actual dtype, and a duplicate names the
            first repeated ``disk_uid`` with its row.
    """
    # pandas is imported here, not at module top level, so that importing this
    # module keeps a numpy-and-stdlib-only surface for lanes that do not use it.
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        raise ContractViolation(f"disk index frame must be a pandas DataFrame, got {type(df).__name__}")
    columns = list(df.columns)
    expected_columns = list(DISK_INDEX_COLUMNS)
    if columns != expected_columns:
        missing = [c for c in expected_columns if c not in columns]
        unexpected = [c for c in columns if c not in expected_columns]
        if missing or unexpected:
            detail = f"missing {missing}, unexpected {unexpected}"
        elif len(columns) != len(set(columns)):
            detail = "duplicate column names"
        else:
            detail = "same columns in the wrong order"
        raise ContractViolation(
            f"disk index columns must equal {expected_columns} in order, got {columns}: {detail}"
        )
    for name in DISK_INDEX_COLUMNS:
        if df[name].isna().any():
            raise ContractViolation(f"disk index column {name!r} contains null values")
    uid = df["disk_uid"]
    if uid.dtype == object:
        uid_is_string = all(isinstance(value, str) for value in uid)
    else:
        uid_is_string = isinstance(uid.dtype, pd.StringDtype)
    if not uid_is_string:
        raise ContractViolation(
            f"disk index column 'disk_uid' has dtype {uid.dtype}, expected string "
            "(object holding str, or a pandas string dtype)"
        )
    numeric_columns = DISK_INDEX_COLUMNS[1:]
    expected_int = np.dtype("int64")
    for name in numeric_columns:
        actual = df[name].dtype
        if actual != expected_int:
            raise ContractViolation(
                f"disk index column {name!r} has dtype {actual}, expected {expected_int}"
            )
    repeated = uid.duplicated()
    if repeated.any():
        position = int(np.argmax(repeated.to_numpy()))
        raise ContractViolation(
            f"disk index column 'disk_uid' has a duplicate value at row {position}: "
            f"{uid.iloc[position]!r}"
        )
    for name in numeric_columns:
        negative = df[name] < 0
        if negative.any():
            position = int(np.argmax(negative.to_numpy()))
            raise ContractViolation(
                f"disk index column {name!r} has a negative value at row {position}: "
                f"{df[name].iloc[position]!r}"
            )


def to_replay_frame(window: Any) -> Any:
    """Rebase a contiguous window of the cleaned Tencent load table to the replay schema.

    ``window`` is a pandas ``DataFrame`` with a ``timestamp`` column (source Unix
    epoch seconds, ``CONTROL_INTERVAL_S`` apart with no gap) and the five value
    columns of ``REPLAY_COLUMNS`` after ``t_seconds``; other columns are ignored.
    The result has the ``REPLAY_COLUMNS`` columns in order, all ``int64``, with
    ``t_seconds`` starting at 0 (``INTERFACES.md`` section 10, decision 4). Values
    are not converted: bandwidth stays in KB/s (decision 3).

    Raises:
        ContractViolation: if ``window`` is not a DataFrame, lacks a required
            column, or its timestamps are not a gap-free ``CONTROL_INTERVAL_S``
            grid of at least one row.
    """
    import pandas as pd

    if not isinstance(window, pd.DataFrame):
        raise ContractViolation(f"window must be a pandas DataFrame, got {type(window).__name__}")
    value_columns = list(REPLAY_COLUMNS[1:])
    missing = [c for c in ["timestamp", *value_columns] if c not in window.columns]
    if missing:
        raise ContractViolation(f"window is missing columns {missing}")
    timestamps = window["timestamp"].to_numpy(dtype="int64")
    on_grid = np.arange(len(timestamps), dtype="int64") * CONTROL_INTERVAL_S
    if len(timestamps) == 0 or not np.array_equal(timestamps - timestamps[0], on_grid):
        raise ContractViolation(f"window is not a contiguous {CONTROL_INTERVAL_S} s grid")
    out = window[value_columns].astype("int64").reset_index(drop=True)
    out.insert(0, "t_seconds", timestamps - timestamps[0])
    return out[list(REPLAY_COLUMNS)]


# --------------------------------------------------------------------------
# Import-time self-consistency
# --------------------------------------------------------------------------


def _check_internal_consistency() -> None:
    checks = {
        "Observation field order must equal OBS_FIELDS": tuple(f.name for f in fields(Observation)) == OBS_FIELDS,
        "OBS_DIM must equal len(OBS_FIELDS)": OBS_DIM == len(OBS_FIELDS),
        "OBS_SCALES keys must equal OBS_FIELDS": set(OBS_SCALES) == set(OBS_FIELDS),
        "N_ACTIONS must equal len(ACTIONS_MBPS)": N_ACTIONS == len(ACTIONS_MBPS),
        "HORIZON_HOURS must equal HORIZON_DAYS * 24": HORIZON_HOURS == HORIZON_DAYS * 24.0,
        "GROUP_TOLERATED_FAILURES must equal GROUP_PARITY_DRIVES": GROUP_TOLERATED_FAILURES == GROUP_PARITY_DRIVES,
        "all actions must fit in BW_TOTAL_MBPS": max(ACTIONS_MBPS) <= BW_TOTAL_MBPS,
        "MAX_EPISODE_STEPS must cover the slowest action's rebuild": (
            MAX_EPISODE_STEPS * CONTROL_INTERVAL_S
            >= REBUILD_WORK_GB * 1000.0 / min(ACTIONS_MBPS)
        ),
        "REPLAY_DTYPES keys must equal REPLAY_COLUMNS in order": tuple(REPLAY_DTYPES) == REPLAY_COLUMNS,
        "REBUILD_WORK_GB must equal DRIVE_CAPACITY_GB": REBUILD_WORK_GB == DRIVE_CAPACITY_GB,
        "GROUP_SIZE must equal GROUP_DATA_DRIVES + GROUP_PARITY_DRIVES": GROUP_SIZE == GROUP_DATA_DRIVES + GROUP_PARITY_DRIVES,
        "hazard scales must equal (GROUP_SIZE - 1) * 2e-5": all(
            math.isclose(OBS_SCALES[k], (GROUP_SIZE - 1) * 2e-5, rel_tol=1e-9)
            for k in ("hazard", "hazard_uncertainty")
        ),
    }
    for message, ok in checks.items():
        if not ok:
            raise ContractViolation(message)


_check_internal_consistency()
