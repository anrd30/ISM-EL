"""Simulator constants and the latency model.

Every value in :class:`SimConfig` is a design decision, not a measurement, and
the latency model is an idealisation of a storage stack that has not been
calibrated against any device or workload. The contract owns the group
geometry, the action set, the reward coefficients' type and the limits; this
module owns only what the simulator itself has to assume.

Provenance convention
---------------------
Every numeric constant here carries exactly one provenance tag in its comment
or docstring entry:

- ``[MEASURED]``: taken from data, with the source named;
- ``[DERIVED]``: computed from other constants or from a stated model, with the
  derivation shown or referenced;
- ``[ASSUMED]``: chosen by design and never checked against data.

Why this exists: several of this project's constants were set by reasoning
before any data existed, and a wrong constant does not fail a test. It sits in
a file looking plausible while the suite passes over it. The tag makes "which
numbers have never been checked against data" a grep
(grep for the literal tag text) instead of a memory exercise. A tag is a statement
about the constant's origin, not about its correctness; ``[MEASURED]`` does not
mean validated, and no tag in this module means the simulator or its latency
model has been validated. Every simulator constant below is ``[ASSUMED]`` or
``[DERIVED]``; none is ``[MEASURED]``. The same convention applies in
``contracts/contracts.py``.

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
latency budget. At 15 ms the threshold is 0.911, just above that 0.90, so (at
a healthy fleet only; under the default randomised fleet ``peak`` admits none,
see ``sim/LIMITATIONS.md`` section 13) every
profile admits at least one action inside the budget, and the number of such
actions falls as the workload gets heavier. Whether an action satisfies the
latency budget on a profile is measured, not assumed, and is a property of this
idealised model only; see ``sim/LIMITATIONS.md``.

The statement that "every profile admits at least one action inside the
latency budget" holds for a healthy fleet only. Under the default
randomised-fleet configuration the safety filter raises the allocation on
degraded fleets and ``peak`` has zero feasible actions (measured over 40 seeds,
synthetic and real-derived profiles alike). Peak results under a degraded fleet
therefore measure the safety filter more than the policy. See
``sim/LIMITATIONS.md`` sections 13 and 14.

Inert observation fields. The latency model uses bandwidth only
(``rho = (fg_bw + alloc) / BW_TOTAL_MBPS``), so the foreground IOPS fields of
the observation (indices 0 and 1) never affect the dynamics, the reward or the
filter. No claim may be made that the controller responds to IOPS; see
``sim/LIMITATIONS.md`` section 12.

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
from pathlib import Path

from contracts.contracts import (
    LATENCY_LIMIT_FACTOR,
    HazardModel,
    RewardConfig,
)
from hazard.v0 import HeuristicHazardV0
from hazard.v1 import FittedHazardV1
from sim.scenario import (
    FleetFeatureFn,
    v0_drive_features,
    v1_drive_features,
    v1_eval_drive_features,
    v1_vendor_holdout_drive_features,
)
from workload.generate import PROFILE_NAMES

__all__ = [
    "SimConfig",
    "RiskObjective",
    "v0_config",
    "v1_eval_config",
    "vendor_holdout_config",
    "VENDOR_HOLDOUT_ARTIFACT_DIR",
    "latency_mean_ms",
    "latency_percentiles_ms",
    "violation_rho_threshold",
]

# Quantile multipliers of an exponential distribution: -ln(1 - q).
# [DERIVED] from the single-queue exponential-response-time idealisation in the
# module docstring (q = 0.95 and 0.99); the idealisation itself is [ASSUMED]
# and the factors are not measured tail ratios of any device.
_P95_FACTOR: float = math.log(20.0)
_P99_FACTOR: float = math.log(100.0)


@dataclass(frozen=True)
class RiskObjective:
    """Optional exposure-weighted degraded-state cost (off by default; ``SimConfig.risk_objective``).

    The contract reward charges ``beta * C`` per degraded interval with ``C = 1``. With this objective the
    environment charges ``beta * C * h / hazard_ref_per_hour`` instead, ``h`` being the true group hazard (per hour)
    at the start of the interval, so that an interval spent degraded costs more when the group is more likely to
    lose a further drive. ``hazard_ref_per_hour`` is the mean initial true group hazard of the evaluation fleet
    (``baselines.risk_objective.calibrate_hazard_ref``), so the mean charge per degraded interval equals the
    contract's ``beta`` at that fleet. The contract (``contracts/``) is not changed: the weight multiplies the
    ``c_degraded`` argument of ``contracts.compute_reward``.
    """

    hazard_ref_per_hour: float

    def __post_init__(self) -> None:
        if isinstance(self.hazard_ref_per_hour, bool) or not (
            isinstance(self.hazard_ref_per_hour, (int, float)) and math.isfinite(self.hazard_ref_per_hour)
            and self.hazard_ref_per_hour > 0.0
        ):
            raise ValueError(f"hazard_ref_per_hour must be a finite number > 0, got {self.hazard_ref_per_hour!r}")


@dataclass(frozen=True)
class SimConfig:
    """Simulator configuration. Every default is a design decision.

    Provenance tags (see the module docstring): ``latency_unloaded_ms``,
    ``latency_target_ms``, ``rho_max``, ``drive_spread_sd`` and
    ``hazard_error_sigma`` are all ``[ASSUMED]``. The ``reward`` coefficients are defined, and tagged, in
    ``contracts/contracts.py``. The fleet-degradation distribution (uniform on
    ``[0, 1]`` when ``fleet_degradation`` is ``None``) is ``[ASSUMED]``, and its
    default consequences for ``peak`` are in ``sim/LIMITATIONS.md`` sections 13
    and 14.

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
        risk_objective: ``RiskObjective`` or ``None`` (default, the contract reward): the degraded-state term is
            weighted by the true group hazard over a reference hazard.
        infeasible_budget_max_rate: ``True`` (default): when no action meets the risk budget the safety filter
            applies the largest action. ``False``: the filter applies the requested action in that case (the
            exposure stays on record in ``violated_risk_budget``); the feasible-budget behaviour is unchanged.
        history_rows: Number of trace rows before the episode's start row that the reset ``info`` carries
            (``contracts.RESET_HISTORY_INFO_KEYS``), the load a deployed system would have monitored before
            the failure; 0 (default) is off and leaves every episode unchanged. When positive, a drawn start
            row below ``history_rows`` is mapped onto ``[history_rows, len(trace))`` so the history never
            wraps; start rows at or above it are unchanged.
        random_trace_offset: Start each episode at a uniformly random row of
            the replay trace rather than row 0. Design decision: on, so a
            rebuild may begin at any time of day instead of always at the
            trace's midnight.
        hazard_error_sigma: Standard deviation, in log space, of the
            multiplicative error between the risk model's reported per-drive
            hazard (the estimate, which the observation and the safety filter
            see) and the hazard failures are drawn from (the truth). The true
            hazard of a drive is ``estimate * exp(N(-sigma^2/2, sigma^2))``,
            drawn once per drive per episode. ``0.0`` makes truth equal the
            estimate and reproduces the simulator without estimation error
            exactly. ``[ASSUMED]`` and left at ``0.0``: no calibration curve of
            a fitted risk model exists yet, and any non-zero value would be an
            invented number. See ``sim/LIMITATIONS.md`` section 15.
        hazard_model: The per-drive risk model, called once per surviving
            drive at reset. Default is the fitted v1
            (``hazard.v1.FittedHazardV1``, which reads its artifacts at
            construction); the heuristic v0 is selected by passing
            ``HeuristicHazardV0()`` together with ``v0_drive_features`` (see
            :func:`v0_config`).
        feature_fn: Per-drive feature builder passed to
            ``sim.scenario.sample_fleet``; it must produce the inputs of
            ``hazard_model``. Default is ``sim.scenario.v1_drive_features``,
            matching the v1 default; ``v0_drive_features`` matches v0.
        label: Name of the configuration, copied into every ``info`` as
            ``config_label`` so results can be traced to the configuration
            that produced them. ``"default"`` for the unmodified simulator;
            :func:`sim.stress.stress_config` sets a stress label.
        latency_filter: Apply the latency ceiling of :mod:`sim.latency_filter`
            after the safety filter: the applied allocation is clipped down
            to the largest action whose utilisation stays at or below
            :func:`violation_rho_threshold`, never below the reliability
            floor. ``False`` (default) leaves the simulator unchanged. The
            ceiling reads the foreground load of the interval about to run,
            which a controller deciding at the start of the interval does
            not have.
        latency_filter_causal: The same ceiling computed from the previous
            interval's foreground load, the value observation field ``fg_bw``
            carries at decision time, so it uses no load from the interval
            being decided. Mutually exclusive with ``latency_filter``.
    """

    # [ASSUMED] 2 ms unloaded latency; never checked against any device.
    latency_unloaded_ms: float = 2.0
    # [ASSUMED] 15 ms target; chosen against the synthetic profiles' p95 load
    # (see the module docstring), not against any latency measurement.
    latency_target_ms: float = 15.0
    # [ASSUMED] 0.98 utilisation ceiling; a numerical guard, not a device limit.
    rho_max: float = 0.98
    reward: RewardConfig = field(default_factory=RewardConfig)
    workload_profile: str | None = None
    workload_csv: str | os.PathLike[str] | None = None
    fleet_degradation: float | None = None
    # [ASSUMED] 0.15 per-drive spread about the fleet level; not fitted to any
    # fleet.
    drive_spread_sd: float = 0.15
    random_trace_offset: bool = True
    # [ASSUMED] 0.0: estimation error switched off. Not derived from any
    # measured calibration error; do not set non-zero without deriving it.
    hazard_error_sigma: float = 0.0
    hazard_model: HazardModel = field(default_factory=FittedHazardV1)
    feature_fn: FleetFeatureFn = field(default=v1_drive_features, kw_only=True)
    label: str = "default"
    latency_filter: bool = field(default=False, kw_only=True)
    latency_filter_causal: bool = field(default=False, kw_only=True)
    history_rows: int = field(default=0, kw_only=True)
    risk_objective: RiskObjective | None = field(default=None, kw_only=True)
    infeasible_budget_max_rate: bool = field(default=True, kw_only=True)

    def __post_init__(self) -> None:
        if isinstance(self.history_rows, bool) or not isinstance(self.history_rows, int) or self.history_rows < 0:
            raise ValueError(f"history_rows must be a non-negative integer, got {self.history_rows!r}")
        if not (math.isfinite(self.latency_unloaded_ms) and self.latency_unloaded_ms > 0.0):
            raise ValueError(f"latency_unloaded_ms must be finite and > 0, got {self.latency_unloaded_ms!r}")
        if not (math.isfinite(self.latency_target_ms) and self.latency_target_ms > self.latency_unloaded_ms):
            raise ValueError(
                "latency_target_ms must be finite and strictly greater than latency_unloaded_ms "
                f"(otherwise any load is a violation), got {self.latency_target_ms!r}"
            )
        if self.latency_filter and self.latency_filter_causal:
            raise ValueError("latency_filter and latency_filter_causal are mutually exclusive")
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
        if isinstance(self.hazard_error_sigma, bool) or not (
            isinstance(self.hazard_error_sigma, (int, float))
            and math.isfinite(self.hazard_error_sigma)
            and self.hazard_error_sigma >= 0.0
        ):
            raise ValueError(f"hazard_error_sigma must be a finite number >= 0, got {self.hazard_error_sigma!r}")


def v0_config(**overrides: object) -> SimConfig:
    """Return a ``SimConfig`` with the heuristic v0 hazard and its matching feature function.

    ``overrides`` are passed to ``SimConfig`` and may replace either of the two.
    """
    fields: dict[str, object] = {"hazard_model": HeuristicHazardV0(), "feature_fn": v0_drive_features}
    fields.update(overrides)
    return SimConfig(**fields)  # type: ignore[arg-type]


def v1_eval_config(**overrides: object) -> SimConfig:
    """Return a ``SimConfig`` with the default fitted v1 hazard and the held-out-split fleet generator.

    Only the fleet-feature table differs from ``SimConfig()``; ``overrides`` are
    passed to ``SimConfig`` and may replace either field.
    """
    fields: dict[str, object] = {"feature_fn": v1_eval_drive_features}
    fields.update(overrides)
    return SimConfig(**fields)  # type: ignore[arg-type]


#: Artifact directory of the hazard model fitted on the Seagate drive models only.
VENDOR_HOLDOUT_ARTIFACT_DIR: Path = Path(__file__).resolve().parents[1] / "hazard" / "artifacts" / "v1_seagate_only"


def vendor_holdout_config(**overrides: object) -> SimConfig:
    """Return a ``SimConfig`` with the Seagate-only hazard model and fleets drawn from the held-out drive model's feature table.

    The hazard model is ``FittedHazardV1`` loaded from ``VENDOR_HOLDOUT_ARTIFACT_DIR``; the fleet generator is
    ``v1_vendor_holdout_drive_features`` (drives of the drive model that model was not fitted on). ``overrides`` are
    passed to ``SimConfig`` and may replace either field.
    """
    fields: dict[str, object] = {
        "hazard_model": FittedHazardV1(VENDOR_HOLDOUT_ARTIFACT_DIR),
        "feature_fn": v1_vendor_holdout_drive_features,
    }
    fields.update(overrides)
    return SimConfig(**fields)  # type: ignore[arg-type]


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
