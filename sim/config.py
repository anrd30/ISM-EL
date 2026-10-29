"""Simulator constants and the latency model.

Every value in :class:`SimConfig` is a design decision, not a measurement, and
the latency model is an idealisation of a storage stack that has not been
calibrated against any device or workload. The contract owns the group
geometry, the action set, the reward coefficients' type and the limits; this
module owns only what the simulator itself has to assume.

Latency model
-------------
Foreground traffic and rebuild traffic share one bandwidth budget
``BW_TOTAL_MBPS``. With ``rho = (fg_bw + alloc_effective) / BW_TOTAL_MBPS`` the
mean latency is the single-queue form::

    latency_now = latency_unloaded_ms / (1 - rho)

Why this form and not a linear one. The slope is
``d latency / d rho = latency_unloaded_ms / (1 - rho) ** 2``, which grows
without bound as ``rho`` approaches 1. The latency cost of one more MB/s of
rebuild bandwidth is therefore small when the foreground load is light and
large when it is heavy. That load-dependent marginal cost is what makes the
trade-off non-trivial: the benefit of rebuilding faster (a shorter exposure
window) does not depend on the foreground load, so against a convex cost the
best allocation changes with load, which gives a controller something to
condition on. Under a linear model ``L = L0 * (1 + k * rho)`` the marginal
cost per MB/s is the same at every load, the comparison between a load-
independent benefit and a load-independent cost has no reason to change with
the observation, and the optimal policy tends to collapse to a constant
(usually "always rebuild at maximum"), which would make the controller
pointless. The shape of this curve is therefore the single most consequential
assumption in the project.

This is an idealisation. It treats the stack as one M/M/1-like server with a
single shared capacity; it ignores queue depth limits, caching, read/write
asymmetry, RAID geometry, device-internal garbage collection and burstiness
within a control interval. It is not calibrated against any measurement, and
its numeric consequences (where latency becomes unacceptable, how steeply it
rises) are properties of the formula, not findings about a real system. The
report's limitations section must say so.

Calibration implied by the defaults
-----------------------------------
With ``latency_unloaded_ms = 2.0`` and ``latency_target_ms = 15.0``,
``LATENCY_LIMIT_FACTOR = 1.5`` gives ``L_max = 22.5 ms``. A violation
(``latency_now > L_max``) begins where ``2.0 / (1 - rho) = 22.5``, i.e. at
``rho = 1 - 2.0 / 22.5``, about 0.9111. With ``BW_TOTAL_MBPS = 200`` that is a
combined foreground-plus-rebuild load of about 182.2 MB/s. The service-level
target (15 ms) is deliberately above the unloaded latency (2 ms): were they
equal, any load at all would register as degradation, ``V`` would be 1 almost
always, and with ``gamma = 10`` the reward would saturate at a constant and
carry no information about the action taken.

Why 15 ms. The violation threshold has to sit above the foreground utilisation
of the heaviest workload profile plus the smallest rebuild allocation, or a
latency limit that every action breaks makes a policy comparison on that
profile meaningless. The ``peak`` profile has a p95 foreground load of 0.8 of
the budget (``contracts/INTERFACES.md``, section 10, decision 5); with a
20 MB/s rebuild added the p95 utilisation is about 0.90. Targets of 5 ms
(threshold 0.733) and 8 ms (0.833) both left ``peak`` with no action inside the
latency budget. At 15 ms the threshold is 0.911, just above that 0.90, so every
profile admits at least one action inside the budget, and the number of such
actions falls as the workload gets heavier. Whether an action satisfies the
latency budget on a profile is measured, not assumed, and is a property of this
idealised model only; see ``sim/LIMITATIONS.md``.

Shaping consequence. The degradation term ``D = max(0, (L - target) / target)``
is zero while latency is below the target, that is while
``2.0 / (1 - rho) <= 15`` or ``rho <= 0.867``. The latency term is therefore
silent over most of the operating range and informative only in the band
between ``rho`` 0.867 and the violation threshold 0.911 (plus the violation
cliff beyond it).

Known limitations (uncalibrated latency model, the one-interval observation
lag, unchanged rebuild speed after a survivor is lost, nominal-rate exposure in
the filter, and others) are collected in ``sim/LIMITATIONS.md``.

Percentiles. In this queue the response time is exponentially distributed
about its mean, so the ``q`` quantile is ``-mean * ln(1 - q)``:
``p95 = mean * ln(20)`` and ``p99 = mean * ln(100)``. The reward and the
service-level target are defined on the mean; the percentiles are reported in
``info`` only.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from contracts.contracts import (
    LATENCY_LIMIT_FACTOR,
    HazardModel,
    RewardConfig,
)
from hazard.v0 import HeuristicHazardV0
from workload.generate import PROFILE_NAMES

__all__ = [
    "SimConfig",
    "latency_mean_ms",
    "latency_percentiles_ms",
    "violation_rho_threshold",
]

# Quantile multipliers of an exponential distribution: -ln(1 - q).
_P95_FACTOR: float = math.log(20.0)
_P99_FACTOR: float = math.log(100.0)


@dataclass(frozen=True)
class SimConfig:
    """Simulator configuration. Every default is a design decision.

    Attributes:
        latency_unloaded_ms: Mean service latency at zero utilisation. Design
            decision: 2 ms, an order of magnitude plausible for a networked
            block device, not measured. It is the ``L0`` of the queue formula.
        latency_target_ms: The service-level target, observation field 4 and
            the ``L0`` of the contract's degradation term. Design decision:
            15 ms, strictly above ``latency_unloaded_ms`` so that light load
            is not already a violation, and high enough that the violation
            threshold (``rho`` about 0.911) lies above the p95 foreground load
            of the heaviest profile plus the smallest allocation (see the
            module docstring).
        rho_max: Utilisation ceiling. Design decision: 0.98. It keeps
            ``1 - rho`` away from zero so latency stays finite (at most
            ``latency_unloaded_ms / (1 - rho_max)``, 100 ms with the defaults)
            and it fixes the headroom ``rho_max * BW_TOTAL_MBPS - fg_bw``.
        reward: Reward coefficients, passed to the contract's
            ``compute_reward``; never hardcoded at a call site.
        workload_profile: Pin the synthetic workload to one of
            ``workload.generate.PROFILE_NAMES``; ``None`` samples uniformly
            each episode. Ignored when ``workload_csv`` is set.
        workload_csv: Path of a replay CSV to use instead of the synthetic
            generator. Its provenance label is read from a
            ``profile_<name>.meta.json`` sidecar when one exists.
        fleet_degradation: Pin the per-episode fleet degradation level in
            ``[0, 1]`` (0 is a wholly healthy group); ``None`` samples it
            uniformly each episode. Pinning exists so that a test or an
            ablation can hold the risk regime fixed.
        drive_spread_sd: Standard deviation, in degradation-level units, of a
            drive's deviation from the fleet level. Design decision: 0.15, so
            drives in one group are strongly but not perfectly coupled.
        random_trace_offset: Start each episode at a uniformly random row of
            the replay trace rather than row 0. Design decision: on, so a
            rebuild may begin at any time of day instead of always at the
            trace's midnight.
        hazard_model: The per-drive risk model, called once per surviving
            drive at reset. Default is the heuristic v0.
        label: Name of the configuration, copied into every ``info`` as
            ``config_label`` so results can be traced to the configuration
            that produced them. ``"default"`` for the unmodified simulator;
            :func:`sim.stress.stress_config` sets a stress label.
    """

    latency_unloaded_ms: float = 2.0
    latency_target_ms: float = 15.0
    rho_max: float = 0.98
    reward: RewardConfig = field(default_factory=RewardConfig)
    workload_profile: str | None = None
    workload_csv: str | os.PathLike[str] | None = None
    fleet_degradation: float | None = None
    drive_spread_sd: float = 0.15
    random_trace_offset: bool = True
    hazard_model: HazardModel = field(default_factory=HeuristicHazardV0)
    label: str = "default"

    def __post_init__(self) -> None:
        if not (math.isfinite(self.latency_unloaded_ms) and self.latency_unloaded_ms > 0.0):
            raise ValueError(f"latency_unloaded_ms must be finite and > 0, got {self.latency_unloaded_ms!r}")
        if not (math.isfinite(self.latency_target_ms) and self.latency_target_ms > self.latency_unloaded_ms):
            raise ValueError(
                "latency_target_ms must be finite and strictly greater than latency_unloaded_ms "
                f"(otherwise any load is a violation), got {self.latency_target_ms!r}"
            )
        if not 0.0 < self.rho_max < 1.0:
            raise ValueError(f"rho_max must be in (0, 1), got {self.rho_max!r}")
        if self.workload_profile is not None and self.workload_profile not in PROFILE_NAMES:
            raise ValueError(f"workload_profile must be one of {PROFILE_NAMES}, got {self.workload_profile!r}")
        if self.workload_profile is not None and self.workload_csv is not None:
            raise ValueError("workload_profile and workload_csv are mutually exclusive")
        if self.fleet_degradation is not None and not 0.0 <= self.fleet_degradation <= 1.0:
            raise ValueError(f"fleet_degradation must be in [0, 1], got {self.fleet_degradation!r}")
        if not (math.isfinite(self.drive_spread_sd) and self.drive_spread_sd >= 0.0):
            raise ValueError(f"drive_spread_sd must be finite and >= 0, got {self.drive_spread_sd!r}")


def latency_mean_ms(rho: float, cfg: SimConfig) -> float:
    """Mean latency ``latency_unloaded_ms / (1 - rho)`` in milliseconds.

    ``rho`` is clamped to ``[0, cfg.rho_max]``. The upper clamp only matters
    when the foreground load alone exceeds ``rho_max * BW_TOTAL_MBPS``: the
    allocation formula keeps ``fg + alloc`` at or below that ceiling whenever
    there is any headroom, but it cannot lower a foreground load that is
    already above it, and without the clamp ``rho`` would reach or pass 1.
    """
    clamped = min(max(float(rho), 0.0), cfg.rho_max)
    return cfg.latency_unloaded_ms / (1.0 - clamped)


def latency_percentiles_ms(mean_ms: float) -> tuple[float, float]:
    """Return ``(p95, p99)`` of an exponentially distributed response time.

    For an exponential distribution with the given mean, the ``q`` quantile is
    ``-mean * ln(1 - q)``, so ``p95 = mean * ln(20)`` and
    ``p99 = mean * ln(100)``. That the response time is exponential about its
    mean is a property of the single-queue idealisation, not a measurement.
    """
    return float(mean_ms) * _P95_FACTOR, float(mean_ms) * _P99_FACTOR


def violation_rho_threshold(cfg: SimConfig) -> float:
    """Utilisation above which ``latency_now > LATENCY_LIMIT_FACTOR * latency_target``.

    Solves ``latency_unloaded / (1 - rho) = LATENCY_LIMIT_FACTOR * target``;
    about 0.9111 with the defaults.
    """
    return 1.0 - cfg.latency_unloaded_ms / (LATENCY_LIMIT_FACTOR * cfg.latency_target_ms)
