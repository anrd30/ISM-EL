"""Latency ceiling: clips an allocation so the interval's utilisation stays under the violation threshold.

``sim.safety_filter`` enforces the reliability budget and only ever moves an
action up. This module is the counterpart for the latency budget and only ever
moves an action down. It is opt-in (``SimConfig.latency_filter``).

The ceiling. ``sim.config.violation_rho_threshold`` is the utilisation above
which the mean latency exceeds ``LATENCY_LIMIT_FACTOR * latency_target``. The
ceiling action for a foreground load ``fg`` is the largest entry of
``ACTIONS_MBPS`` with ``(fg + alloc) / BW_TOTAL_MBPS <= threshold``. When no
entry satisfies it (the foreground load alone is at or above the threshold),
the ceiling is the smallest action. The smallest action is the least the
interval can be given; the violation then comes from the foreground load.

Precedence. The reliability floor wins. ``floor_index`` is the smallest action
the safety filter would accept at the current hazard and remaining work. The
applied action is ``max(floor_index, min(requested_index, ceiling_index))``:
the latency ceiling never lowers an action below the reliability floor, so
the risk budget the safety filter enforces is not traded for latency. When the
floor sits above the ceiling, the decision records it in
``reliability_overrode``.

Load used. ``fg_mbps`` is chosen by the caller. ``RecoveryEnv`` passes the
foreground load of the interval about to run under ``SimConfig.latency_filter``
(the same value it uses for the headroom cap on the delivered allocation; a
controller deciding at the start of the interval does not have it), and the
previous interval's load under ``SimConfig.latency_filter_causal`` (the value
observation field ``fg_bw`` carries). See ``sim/LATENCY_FILTER_DECISION.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

from contracts.contracts import ACTIONS_MBPS, BW_TOTAL_MBPS, N_ACTIONS
from sim.config import SimConfig, violation_rho_threshold

__all__ = ["LatencyFilterDecision", "latency_ceiling_index", "apply_latency_filter"]


@dataclass(frozen=True)
class LatencyFilterDecision:
    """Outcome of one latency-filter application.

    Attributes:
        requested_index: Action index handed to the filter (the safety
            filter's output).
        applied_index: Action index that will be applied.
        applied_mbps: Bandwidth of the applied action.
        ceiling_index: Largest action index under the utilisation threshold
            (0 when none is).
        ceiling_mbps: Bandwidth of the ceiling action.
        rho_threshold: The utilisation threshold used.
        clipped: Whether the applied action is below the requested one.
        ceiling_feasible: Whether at least one action satisfies the threshold.
        reliability_overrode: Whether the ceiling was below the requested
            action and the reliability floor kept the applied action above it.
    """

    requested_index: int
    applied_index: int
    applied_mbps: float
    ceiling_index: int
    ceiling_mbps: float
    rho_threshold: float
    clipped: bool
    ceiling_feasible: bool
    reliability_overrode: bool


def latency_ceiling_index(fg_mbps: float, cfg: SimConfig) -> tuple[int, bool]:
    """Return ``(ceiling index, whether any action satisfies the threshold)`` for a foreground load.

    Raises:
        ValueError: if ``fg_mbps`` is negative or not finite.
    """
    fg = float(fg_mbps)
    if not (fg >= 0.0 and fg < float("inf")):
        raise ValueError(f"fg_mbps must be finite and >= 0, got {fg_mbps!r}")
    threshold = violation_rho_threshold(cfg)
    feasible = [i for i in range(N_ACTIONS) if (fg + float(ACTIONS_MBPS[i])) / BW_TOTAL_MBPS <= threshold]
    if not feasible:
        return 0, False
    return max(feasible), True


def apply_latency_filter(
    requested_index: int, fg_mbps: float, cfg: SimConfig, floor_index: int = 0
) -> LatencyFilterDecision:
    """Clip ``requested_index`` down to the latency ceiling, never below ``floor_index``.

    Raises:
        ValueError: if an index is outside ``[0, N_ACTIONS)`` or ``fg_mbps`` is invalid.
    """
    if not 0 <= int(requested_index) < N_ACTIONS or not 0 <= int(floor_index) < N_ACTIONS:
        raise ValueError(f"indices must be in [0, {N_ACTIONS}), got {requested_index!r}, {floor_index!r}")
    ceiling, feasible = latency_ceiling_index(fg_mbps, cfg)
    requested = int(requested_index)
    applied = max(int(floor_index), min(requested, ceiling))
    return LatencyFilterDecision(
        requested_index=requested,
        applied_index=applied,
        applied_mbps=float(ACTIONS_MBPS[applied]),
        ceiling_index=ceiling,
        ceiling_mbps=float(ACTIONS_MBPS[ceiling]),
        rho_threshold=violation_rho_threshold(cfg),
        clipped=applied < requested,
        ceiling_feasible=feasible,
        reliability_overrode=requested > ceiling and applied > ceiling,
    )
