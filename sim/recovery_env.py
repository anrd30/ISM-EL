"""Rebuild-recovery environment with a gymnasium-style API (duck typed).

A group of ``GROUP_SIZE`` drives has lost one and is rebuilding it from the
nine survivors. Rebuild traffic and foreground traffic share one bandwidth
budget, ``BW_TOTAL_MBPS``. Each control interval (``CONTROL_INTERVAL_S``) the
controller picks a rebuild bandwidth from ``ACTIONS_MBPS``. Rebuilding fast
shortens the window in which a further failure could destroy data; rebuilding
slowly protects foreground latency. An episode ends when the rebuild
completes, when data is lost, or after ``MAX_EPISODE_STEPS`` intervals.

Everything here is simulated. The latency model is an idealisation that has
not been calibrated against any device (see :mod:`sim.config`), and every
constant is a design decision. Known limitations are collected in
``sim/LIMITATIONS.md``; the ones that shape this module are the one-interval
observation lag (no look-ahead, an information constraint that bounds what any
policy can learn), the stall when foreground load saturates the budget, the
constant ``C`` term, and the unchanged rebuild speed after a survivor is lost.

API
---
``reset(seed=None) -> (obs, info)`` and
``step(action) -> (obs, reward, terminated, truncated, info)``. Observations are
the contract's normalised ``(12,)`` ``float32`` vector, always produced by
building an ``Observation``, calling ``validate()`` and ``to_vector()``.
Actions are routed through ``action_to_mbps``, so a bad action raises
``ContractViolation``. ``terminated`` and ``truncated`` always come from
``termination_flags`` of the reason; they are never set independently.

Timing
------
Trace row ``k0`` (the episode's start offset) is shown in the reset
observation, with no rebuild allocation yet. Step ``t`` executes row ``k0 + t``
(modulo the trace length). Everything in the observation returned by a step
(foreground load, latency, headroom, allocation) was measured over the
interval that step just executed, so the controller never sees the load of an
interval before choosing its allocation: there is no look-ahead. The reward and
the ``info`` latency terms use the same interval.

Per-step order of events: the safety filter acts on the state at the start of
the interval; the delivered rebuild rate is ``min(applied, headroom)``; latency
follows from the resulting utilisation; progress is credited; survivor
failures are drawn for the interval. If failures and completion fall in the
same interval, the failures are resolved first, so an interval in which
redundancy is exceeded counts as data loss even if the work also reached zero.
This is the conservative reading at interval granularity.

Failure process and its simplifications
---------------------------------------
Each surviving drive fails independently within an interval with probability
``1 - (1 - h) ** (CONTROL_INTERVAL_S / 3600)``, ``h`` being that drive's
*true* per-hour hazard: the risk model's estimate times a per-drive lognormal
error multiplier drawn once per episode (``SimConfig.hazard_error_sigma``;
see :mod:`sim.scenario`). The observation, the safety filter and the exposure
it bounds use the *estimate* only, so with a positive sigma the filter is no
longer exactly protective. With the default sigma of 0 the true hazard is the
estimate and the draw consumes no randomness. A failure decrements ``redundancy_left``;
more failures than the remaining redundancy is ``TERMINAL_FAILURE``. A failed
drive stops contributing to the hazard aggregate. Rebuild speed is kept
unchanged when a survivor is lost, and the second failed drive is not itself
rebuilt. In a real array a rebuild reads from fewer sources and slows down; the
simulator does not model that, so rebuild time after a second failure is
optimistic.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from contracts.contracts import (
    ACTIONS_MBPS,
    BW_TOTAL_MBPS,
    CONTROL_INTERVAL_S,
    GROUP_TOLERATED_FAILURES,
    MAX_EPISODE_STEPS,
    REBUILD_WORK_GB,
    RESET_HISTORY_INFO_KEYS,
    RISK_BUDGET_EPS,
    Observation,
    TerminationReason,
    action_to_mbps,
    aggregate_group_hazard,
    compute_reward,
    degraded_cost,
    latency_degradation,
    latency_violation,
    termination_flags,
    validate_info,
)
from sim.config import SimConfig, latency_mean_ms, latency_percentiles_ms
from sim.latency_filter import apply_latency_filter
from sim.safety_filter import apply_safety_filter, exposure_probability
from sim.scenario import (
    HAZARD_ERROR_STREAM,
    N_SURVIVORS,
    Fleet,
    ReplayTrace,
    load_replay_csv,
    sample_fleet,
    trace_from_frame,
)
from workload.generate import PROFILE_NAMES, generate_profile

__all__ = ["RecoveryEnv", "make_env"]

_MB_PER_GB: float = 1000.0  # decimal convention of the contract
_SEED_BOUND: int = 2**31 - 1


class RecoveryEnv:
    """Single-group rebuild environment. See the module docstring.

    Args:
        config: Simulator configuration; defaults to ``SimConfig()``.

    ``reset`` must be called before ``step``, and again after an episode ends.
    """

    def __init__(self, config: SimConfig | None = None) -> None:
        self.config: SimConfig = config if config is not None else SimConfig()
        self._master: np.random.Generator = np.random.default_rng()
        self._fixed_trace: ReplayTrace | None = None
        self._failure_rng: np.random.Generator = np.random.default_rng(0)
        self._trace: ReplayTrace | None = None
        self._fleet: Fleet | None = None
        self._offset: int = 0
        self._t: int = 0
        self._active: bool = False
        self._alive: np.ndarray = np.ones(N_SURVIVORS, dtype=bool)
        self._hazard: float = 0.0
        self._hazard_unc: float = 0.0
        self._redundancy: int = 1
        self._work: float = REBUILD_WORK_GB

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        """Start a new episode and return ``(observation, info)``.

        ``seed`` re-seeds the environment: the same seed always yields the
        same scenario and, for the same actions, the same trajectory. With
        ``seed=None`` the next episode continues the environment's own seeded
        stream. The scenario (workload profile and trace, trace start offset,
        fleet degradation level, drive features) is sampled from that stream.

        The reset ``info`` has no applied action. Its
        ``p_failure_before_complete`` is the exposure of the largest action
        (the lowest the filter can reach), and ``violated_risk_budget`` is
        true when even that exceeds the budget.

        Both this ``info`` and every later ``step`` ``info`` carry
        ``trace_start_row``: the built-in ``int`` index into the replay trace
        at which this episode began (``0`` when ``random_trace_offset`` is
        off). It is fixed for the whole episode; it is the start row, not the
        current position. It exists because an episode's results are only
        reproducible if the trace offset is recorded alongside the seed: the
        offset is drawn as ``integers(0, len(trace))``, so the same seed with
        a different trace length would start at a different row. Consumers
        should read this key rather than the private ``_offset`` attribute.

        Raises:
            ValueError: if ``seed`` is not a non-negative integer.
        """
        if seed is not None:
            if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
                raise ValueError(f"seed must be a non-negative integer or None, got {seed!r}")
            self._master = np.random.default_rng(int(seed))
        episode_seed = int(self._master.integers(0, _SEED_BOUND))
        scenario_rng = np.random.default_rng([episode_seed, 0])
        self._failure_rng = np.random.default_rng([episode_seed, 1])
        error_rng = np.random.default_rng([episode_seed, HAZARD_ERROR_STREAM])

        self._trace = self._load_trace(scenario_rng, episode_seed)
        n_rows = len(self._trace)
        drawn_offset = int(scenario_rng.integers(0, n_rows)) if self.config.random_trace_offset else 0
        history_rows = self.config.history_rows
        if history_rows > 0 and history_rows >= n_rows:
            raise ValueError(f"history_rows={history_rows} needs a trace longer than it, got {n_rows} rows")
        if 0 < history_rows and drawn_offset < history_rows:
            # Deterministic map of the rows that would leave no full history onto [history_rows, n_rows).
            self._offset = history_rows + (drawn_offset * (n_rows - history_rows)) // history_rows
        else:
            self._offset = drawn_offset
        self._fleet = sample_fleet(
            scenario_rng,
            self.config.hazard_model,
            level=self.config.fleet_degradation,
            drive_spread_sd=self.config.drive_spread_sd,
            feature_fn=self.config.feature_fn,
            hazard_error_sigma=float(self.config.hazard_error_sigma),
            error_rng=error_rng,
        )

        self._t = 0
        self._active = True
        self._alive = np.ones(N_SURVIVORS, dtype=bool)
        self._redundancy = 1
        self._work = REBUILD_WORK_GB
        self._hazard, self._hazard_unc = self._group_hazard()

        row = self._offset
        fg = float(self._trace.fg_bw_mbps[row])
        avail = self._headroom(fg)
        rho = self._rho(fg, 0.0)
        latency = latency_mean_ms(rho, self.config)
        p_best = exposure_probability(self._hazard, self._work, float(max(ACTIONS_MBPS)))
        p_best_true = exposure_probability(self._true_group_hazard(), self._work, float(max(ACTIONS_MBPS)))
        obs = self._observation(row, fg, latency, 0.0, avail)
        info = self._info(
            latency=latency,
            rho=rho,
            p_failure=p_best,
            violated_risk_budget=not p_best < RISK_BUDGET_EPS,
            clipped=False,
            complete=False,
            reason=None,
            requested=0.0,
            applied=0.0,
            effective=0.0,
            avail=avail,
            fg=fg,
            p_effective=p_best,
            p_true_applied=p_best_true,
            p_true_effective=p_best_true,
        )
        if history_rows > 0:
            lo, hi = self._offset - history_rows, self._offset
            info.update(
                {
                    "history_rows": int(history_rows),
                    "trace_start_row_drawn": int(drawn_offset),
                    "trace_start_row_remapped": bool(self._offset != drawn_offset),
                    RESET_HISTORY_INFO_KEYS[0]: np.array(self._trace.fg_bw_mbps[lo:hi], dtype=np.float64),
                    RESET_HISTORY_INFO_KEYS[1]: np.array(self._trace.fg_read_iops[lo:hi], dtype=np.float64),
                    RESET_HISTORY_INFO_KEYS[2]: np.array(self._trace.fg_write_iops[lo:hi], dtype=np.float64),
                }
            )
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one control interval.

        Args:
            action: Index into ``ACTIONS_MBPS``.

        Returns:
            ``(observation, reward, terminated, truncated, info)``.

        Raises:
            ContractViolation: if ``action`` is not a valid action index.
            RuntimeError: if no episode is running (``reset`` not called, or
                the previous episode has ended).
        """
        requested_mbps = action_to_mbps(action)  # raises ContractViolation on a bad action
        if not self._active or self._trace is None or self._fleet is None:
            raise RuntimeError("step() called with no running episode; call reset() first")
        cfg = self.config

        self._t += 1
        n_rows = len(self._trace)
        position = self._offset + self._t
        row = position % n_rows
        fg = float(self._trace.fg_bw_mbps[row])

        # Safety filter on the state at the start of the interval.
        decision = apply_safety_filter(int(action), self._hazard, self._work)
        if not decision.any_feasible and not cfg.infeasible_budget_max_rate:
            requested_mbps_now = float(ACTIONS_MBPS[int(action)])
            p_requested = exposure_probability(self._hazard, self._work, requested_mbps_now)
            decision = dataclasses.replace(
                decision, applied_index=int(action), applied_mbps=requested_mbps_now, p_failure=p_requested,
                clipped=False, violated_risk_budget=not p_requested < RISK_BUDGET_EPS,
            )
        applied_mbps = decision.applied_mbps
        p_failure = decision.p_failure
        violated_risk_budget = decision.violated_risk_budget
        clipped = decision.clipped
        latency_info: dict[str, Any] = {}
        if cfg.latency_filter or cfg.latency_filter_causal:
            # Latency ceiling after the reliability filter; the reliability floor wins (sim/latency_filter.py).
            # The causal variant uses the previous interval's load, the value the policy's observation carries.
            filter_fg = fg if cfg.latency_filter else float(self._trace.fg_bw_mbps[(position - 1) % n_rows])
            floor = apply_safety_filter(0, self._hazard, self._work).applied_index
            lat = apply_latency_filter(decision.applied_index, filter_fg, cfg, floor)
            applied_mbps = lat.applied_mbps
            p_failure = exposure_probability(self._hazard, self._work, applied_mbps)
            violated_risk_budget = not p_failure < RISK_BUDGET_EPS
            clipped = lat.applied_index != int(action)
            latency_info = {
                "latency_filter_active": True,
                "latency_filter_fg_source": "oracle" if cfg.latency_filter else "causal",
                "latency_filter_fg_mbps": float(filter_fg),
                "alloc_after_reliability_filter_mbps": float(decision.applied_mbps),
                "latency_ceiling_mbps": float(lat.ceiling_mbps),
                "latency_ceiling_feasible": bool(lat.ceiling_feasible),
                "alloc_clipped_by_latency_filter": bool(lat.clipped),
                "latency_filter_overridden_by_reliability": bool(lat.reliability_overrode),
                "reliability_filter_clipped": bool(decision.clipped),
                "both_filters_fired": bool(lat.clipped and decision.clipped),
            }
        avail = self._headroom(fg)
        effective = min(applied_mbps, avail)
        p_effective = exposure_probability(self._hazard, self._work, effective)
        # Exposure at the true hazard (diagnostic only; nothing acts on it).
        true_hazard_now = self._true_group_hazard()
        p_true_applied = exposure_probability(true_hazard_now, self._work, applied_mbps)
        p_true_effective = exposure_probability(true_hazard_now, self._work, effective)

        rho = self._rho(fg, effective)
        latency = latency_mean_ms(rho, cfg)

        # Rebuild progress.
        redundancy_before = self._redundancy
        self._work = max(0.0, self._work - effective * CONTROL_INTERVAL_S / _MB_PER_GB)

        # Failures among the survivors during the interval. A fixed number of
        # draws per step keeps the random stream independent of survivor count.
        draws = self._failure_rng.random(N_SURVIVORS)
        failed = self._alive & (draws < self._fleet.interval_fail_prob)
        n_failed = int(failed.sum())
        self._alive = self._alive & ~failed
        if n_failed:
            self._hazard, self._hazard_unc = self._group_hazard()

        if n_failed > redundancy_before:
            reason: TerminationReason | None = TerminationReason.TERMINAL_FAILURE
            self._redundancy = 0
        else:
            self._redundancy = redundancy_before - n_failed
            if self._work <= 0.0:
                reason = TerminationReason.RECOVERY_COMPLETE
            elif self._t >= MAX_EPISODE_STEPS:
                reason = TerminationReason.HORIZON_EXPIRED
            else:
                reason = None

        # Reward terms come from the contract's helpers; the group was degraded
        # throughout the interval (a drive was down and the rebuild unfinished).
        d_term = latency_degradation(latency, cfg.latency_target_ms)
        c_term = degraded_cost(redundancy_before < GROUP_TOLERATED_FAILURES)
        if cfg.risk_objective is not None:
            c_term = c_term * true_hazard_now / cfg.risk_objective.hazard_ref_per_hour
        v_term = latency_violation(latency, cfg.latency_target_ms)
        reward = compute_reward(d_term, c_term, v_term, cfg.reward)

        terminated, truncated = (False, False) if reason is None else termination_flags(reason)
        if reason is not None:
            self._active = False

        obs = self._observation(row, fg, latency, effective, avail)
        info = self._info(
            latency=latency,
            rho=rho,
            p_failure=p_failure,
            violated_risk_budget=violated_risk_budget,
            clipped=clipped,
            complete=reason is TerminationReason.RECOVERY_COMPLETE,
            reason=reason,
            requested=requested_mbps,
            applied=applied_mbps,
            effective=effective,
            avail=avail,
            fg=fg,
            p_effective=p_effective,
            p_true_applied=p_true_applied,
            p_true_effective=p_true_effective,
            extra=latency_info,
        )
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_trace(self, rng: np.random.Generator, episode_seed: int) -> ReplayTrace:
        """Return this episode's foreground trace (CSV, pinned or sampled profile)."""
        cfg = self.config
        if cfg.workload_csv is not None:
            if self._fixed_trace is None:
                self._fixed_trace = load_replay_csv(cfg.workload_csv)
            return self._fixed_trace
        if cfg.workload_profile is not None:
            name = cfg.workload_profile
        else:
            name = PROFILE_NAMES[int(rng.integers(0, len(PROFILE_NAMES)))]
        # The generator is parametric, so the data is synthetic by construction.
        return trace_from_frame(generate_profile(name, episode_seed), name, True)

    def _group_hazard(self) -> tuple[float, float]:
        """Group hazard and its uncertainty over the surviving drives (per hour)."""
        assert self._fleet is not None
        if not self._alive.any():
            return 0.0, 0.0
        return aggregate_group_hazard(self._fleet.pairs[self._alive])

    def _hazard_realised_ratio(self) -> float:
        """Sum of true over sum of estimated per-hour hazard of the live drives.

        1.0 when no drive is alive. Both sums run over the same arrays in the
        same order, so the ratio is exactly 1.0 when the error is off.
        """
        assert self._fleet is not None
        if not self._alive.any():
            return 1.0
        estimated = float(self._fleet.estimated_hazard_per_hour[self._alive].sum())
        if estimated == 0.0:
            return 1.0
        return float(self._fleet.true_hazard_per_hour[self._alive].sum()) / estimated

    def _true_group_hazard(self) -> float:
        """Group hazard the failures are drawn from: reported hazard times the realised ratio."""
        return self._hazard * self._hazard_realised_ratio()

    def _headroom(self, fg_bw: float) -> float:
        """``max(0, rho_max * BW_TOTAL_MBPS - fg_bw)``, observation field 11."""
        return max(0.0, self.config.rho_max * BW_TOTAL_MBPS - fg_bw)

    def _rho(self, fg_bw: float, alloc_effective: float) -> float:
        """Utilisation ``(fg + alloc_effective) / BW_TOTAL_MBPS``, capped at ``rho_max``."""
        return min(self.config.rho_max, (fg_bw + alloc_effective) / BW_TOTAL_MBPS)

    def _observation(
        self, row: int, fg: float, latency: float, alloc_effective: float, avail: float
    ) -> np.ndarray:
        """Build, validate and normalise the observation through the contract."""
        assert self._trace is not None
        obs = Observation(
            fg_read_iops=float(self._trace.fg_read_iops[row]),
            fg_write_iops=float(self._trace.fg_write_iops[row]),
            fg_bw=fg,
            latency_now=latency,
            latency_target=self.config.latency_target_ms,
            hazard=self._hazard,
            hazard_uncertainty=self._hazard_unc,
            redundancy_left=float(self._redundancy),
            progress=1.0 - self._work / REBUILD_WORK_GB,
            work_remaining=self._work,
            alloc_current=alloc_effective,
            bw_available=avail,
        )
        obs.validate()
        return obs.to_vector(normalized=True)

    def _info(
        self,
        *,
        latency: float,
        rho: float,
        p_failure: float,
        violated_risk_budget: bool,
        clipped: bool,
        complete: bool,
        reason: TerminationReason | None,
        requested: float,
        applied: float,
        effective: float,
        avail: float,
        fg: float,
        p_effective: float,
        p_true_applied: float,
        p_true_effective: float,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Assemble and validate ``info``: contract keys plus diagnostics.

        The ``hazard_*`` and ``*_true*`` diagnostics expose the estimation
        error: the reported group hazard (what the policy and filter saw), the
        group hazard failures are drawn from, and their realised ratio, all for
        the drives alive at the end of the interval. ``p_true_applied`` and
        ``p_true_effective`` are the exposure of the applied and the delivered
        rate at the true hazard, as of the start of the interval. The contract
        keys ``p_failure_before_complete`` and ``violated_risk_budget`` stay
        defined on the estimate, because that is what the filter bounds.
        """
        assert self._trace is not None and self._fleet is not None
        p95, p99 = latency_percentiles_ms(latency)
        ratio = self._hazard_realised_ratio()
        info: dict[str, Any] = {
            "latency_p95": float(p95),
            "latency_p99": float(p99),
            "violated_latency": latency_violation(latency, self.config.latency_target_ms) == 1.0,
            "p_failure_before_complete": float(p_failure),
            "violated_risk_budget": bool(violated_risk_budget),
            "alloc_clipped_by_filter": bool(clipped),
            "recovery_complete": bool(complete),
            # Diagnostics (extra keys are permitted by the contract).
            "latency_now": float(latency),
            "rho": float(rho),
            "alloc_requested_mbps": float(requested),
            "alloc_applied_mbps": float(applied),
            "alloc_effective_mbps": float(effective),
            "bw_available_mbps": float(avail),
            "fg_bw_mbps": float(fg),
            "n_survivors": int(self._alive.sum()),
            "p_failure_at_delivered_rate": float(p_effective),
            "trace_wraps": int((self._offset + self._t) // len(self._trace)),
            "trace_start_row": int(self._offset),
            "workload_profile": self._trace.profile,
            "workload_synthetic": self._trace.synthetic,
            "fleet_degradation_level": float(self._fleet.level),
            "config_label": self.config.label,
            "hazard_error_sigma": float(self.config.hazard_error_sigma),
            "hazard_reported_per_hour": float(self._hazard),
            "hazard_true_per_hour": float(self._hazard * ratio),
            "hazard_realised_ratio": float(ratio),
            "p_failure_true_at_applied_rate": float(p_true_applied),
            "p_failure_true_at_delivered_rate": float(p_true_effective),
            "violated_risk_budget_true": bool(not p_true_applied < RISK_BUDGET_EPS),
        }
        if extra:
            info.update(extra)
        if reason is not None:
            info["termination_reason"] = reason.value
        validate_info(info)
        return info


def make_env(config: SimConfig | None = None) -> RecoveryEnv:
    """Factory for the conformance harness (``ISM_ENV_FACTORY=sim.recovery_env:make_env``)."""
    return RecoveryEnv(config)
