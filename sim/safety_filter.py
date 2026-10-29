"""Safety filter: enforces the risk budget independently of the reward.

The contract defines the exposure that the filter bounds::

    T_remaining (h) = work_remaining (GB) * 1000 / alloc_mbps / 3600
    P = 1 - exp(-hazard * T_remaining)

with ``hazard`` the group-level aggregate (per hour) and ``alloc_mbps`` the
candidate allocation. ``P`` is the probability of at least one further drive
failure before the rebuild completes: exposure, not data loss.

``P`` falls monotonically as the allocation rises, so the feasible set
``{a : P(a) < RISK_BUDGET_EPS}`` is always a suffix of the sorted action set.
The filter therefore only ever moves an action up:

- if the requested action is feasible it is applied unchanged;
- otherwise the smallest feasible action above it is applied;
- if no action is feasible the largest allocation is applied. This is a
  deliberate degradation rule: the largest allocation has the shortest
  exposure window, so it is the minimum-risk choice, and the breach stays on
  record through ``violated_risk_budget``.

``P`` is computed for the *nominal* allocation. When foreground load leaves
less headroom than the allocation, the rebuild receives less than the nominal
rate and the exposure at the delivered rate is higher than ``P``. That is an
information constraint rather than an approximation: the environment has no
look-ahead, so the foreground load of the interval about to run is unknown when
the filter decides, and the delivered rate cannot be known. The simulator
reports the exposure at the delivered rate afterwards, in
``info["p_failure_at_delivered_rate"]``, so the gap is visible. See
``sim/LIMITATIONS.md``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from contracts.contracts import ACTIONS_MBPS, RISK_BUDGET_EPS

__all__ = ["FilterDecision", "exposure_probability", "apply_safety_filter"]

_SECONDS_PER_HOUR: float = 3600.0
_MB_PER_GB: float = 1000.0  # decimal convention of the contract


@dataclass(frozen=True)
class FilterDecision:
    """Outcome of one filter application.

    Attributes:
        requested_index: Action index the agent asked for.
        applied_index: Action index that will be applied.
        applied_mbps: Bandwidth of the applied action.
        p_failure: Exposure ``P`` of the applied action.
        clipped: Whether the applied action differs from the requested one.
        violated_risk_budget: Whether the applied action's own exposure is at
            or above ``RISK_BUDGET_EPS`` (feasibility is ``P < RISK_BUDGET_EPS``).
        any_feasible: Whether at least one action satisfied the budget.
    """

    requested_index: int
    applied_index: int
    applied_mbps: float
    p_failure: float
    clipped: bool
    violated_risk_budget: bool
    any_feasible: bool


def exposure_probability(hazard_per_hour: float, work_remaining_gb: float, alloc_mbps: float) -> float:
    """Return ``P = 1 - exp(-hazard * T_remaining)`` for a candidate allocation.

    ``T_remaining = work_remaining * 1000 / alloc / 3600`` hours. No work
    remaining gives 0. A zero allocation never finishes, so the exposure is 1
    for any positive hazard and 0 for zero hazard.

    Raises:
        ValueError: if an argument is negative or not finite.
    """
    for name, value in (
        ("hazard_per_hour", hazard_per_hour),
        ("work_remaining_gb", work_remaining_gb),
        ("alloc_mbps", alloc_mbps),
    ):
        if not (math.isfinite(value) and value >= 0.0):
            raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
    if work_remaining_gb == 0.0 or hazard_per_hour == 0.0:
        return 0.0
    if alloc_mbps == 0.0:
        return 1.0
    hours = work_remaining_gb * _MB_PER_GB / alloc_mbps / _SECONDS_PER_HOUR
    return -math.expm1(-hazard_per_hour * hours)


def apply_safety_filter(
    requested_index: int,
    hazard_per_hour: float,
    work_remaining_gb: float,
    eps: float = RISK_BUDGET_EPS,
) -> FilterDecision:
    """Apply the filter to a requested action index.

    ``requested_index`` must already be a valid index (the caller routes it
    through the contract's ``action_to_mbps``). The search runs upward from the
    request, so the applied action is never lower than the requested one.
    """
    probabilities = [exposure_probability(hazard_per_hour, work_remaining_gb, a) for a in ACTIONS_MBPS]
    feasible = [p < eps for p in probabilities]
    any_feasible = any(feasible)
    applied = len(ACTIONS_MBPS) - 1  # degradation rule when nothing is feasible
    for i in range(requested_index, len(ACTIONS_MBPS)):
        if feasible[i]:
            applied = i
            break
    p_applied = probabilities[applied]
    return FilterDecision(
        requested_index=requested_index,
        applied_index=applied,
        applied_mbps=float(ACTIONS_MBPS[applied]),
        p_failure=p_applied,
        clipped=applied != requested_index,
        violated_risk_budget=not p_applied < eps,
        any_feasible=any_feasible,
    )
