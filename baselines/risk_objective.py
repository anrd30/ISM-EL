"""Hazard-weighted policies and the calibration of the optional exposure-weighted objective.

Under the contract reward the cost of a degraded interval is the constant ``beta``; the hazard enters a policy's
decision only through the safety filter. ``sim.config.RiskObjective`` makes the degraded-interval cost proportional to
the true group hazard, so the value of finishing the rebuild sooner depends on the hazard. ``HazardWeightedMixin``
scales the rule's progress value (the reward units it assigns to a MB/s of rebuild progress, which stands in for the
degraded-interval cost that progress saves) by ``hazard / hazard_ref`` of the observation; the controller and the rule
are otherwise unchanged. The observation's hazard field is clipped at its scale (``OBS_SCALES["hazard"]``), which is
all a policy can see.

``unforced=True`` is the exposure-weighted choice when no action meets the risk budget: the policy's risk floor is the
smallest action instead of the largest, and the policy chooses among all actions by its own expected cost. It is meant
for an environment with ``SimConfig.infeasible_budget_max_rate = False``; in the default environment the safety filter
raises the applied action to the largest regardless.

``calibrate_hazard_ref`` is the mean initial true group hazard of a configuration's fleet over scenario seeds.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np

from baselines.policies import DEFAULT_PROGRESS_VALUE, RiskAwareRuleBased
from contracts.contracts import RISK_BUDGET_EPS, Observation
from sim.config import SimConfig
from sim.recovery_env import RecoveryEnv
from sim.safety_filter import apply_safety_filter

__all__ = ["calibrate_hazard_ref", "HazardWeightedMixin", "HazardWeightedRule"]


def calibrate_hazard_ref(config: SimConfig, seeds: range | list[int] = range(1000, 1400)) -> float:
    """Mean ``info["hazard_true_per_hour"]`` at reset over ``seeds`` for ``config``'s fleet (the workload is irrelevant).

    The default seeds 1000 to 1399 are the tuning range of the project, disjoint from the evaluation seeds 0 to 99.
    """
    config = dataclasses.replace(config, risk_objective=None, history_rows=0)
    env = RecoveryEnv(config)
    values = [float(env.reset(seed=int(s))[1]["hazard_true_per_hour"]) for s in seeds]
    mean = float(np.mean(values))
    if not (math.isfinite(mean) and mean > 0.0):
        raise ValueError(f"mean initial true hazard must be finite and > 0, got {mean!r}")
    return mean


class HazardWeightedMixin:
    """Scales ``progress_value`` by ``observed hazard / hazard_ref`` before each decision; optional unforced floor.

    Put before the policy class in the bases. The policy must define ``_belief``, ``_select`` and ``risk_floor_index``
    as ``RiskAwareRuleBased`` does.
    """

    hazard_ref: float
    unforced: bool
    base_progress_value: float

    def _init_weighting(self, hazard_ref: float, unforced: bool, base_progress_value: float) -> None:
        if not (math.isfinite(hazard_ref) and hazard_ref > 0.0):
            raise ValueError(f"hazard_ref must be finite and > 0, got {hazard_ref!r}")
        self.hazard_ref = float(hazard_ref)
        self.unforced = bool(unforced)
        self.base_progress_value = float(base_progress_value)

    def _floor(self, o: Observation) -> int:
        floor = self.risk_floor_index(o)  # type: ignore[attr-defined]
        if self.unforced:
            eps = getattr(self, "eps", RISK_BUDGET_EPS)
            k = getattr(self, "k", 0.0)
            h = o.hazard + k * o.hazard_uncertainty
            if not apply_safety_filter(0, h, o.work_remaining, eps).any_feasible:
                return 0
        return floor

    def __call__(self, obs: np.ndarray) -> int:
        o = Observation.from_vector(obs, normalized=True)
        self.progress_value = self.base_progress_value * o.hazard / self.hazard_ref  # type: ignore[attr-defined]
        samples, weights = self._belief(o)  # type: ignore[attr-defined]
        return int(self._select(samples, weights, self._floor(o)))  # type: ignore[attr-defined]


class HazardWeightedRule(HazardWeightedMixin, RiskAwareRuleBased):
    """``RiskAwareRuleBased`` with a hazard-weighted progress value (and optionally the unforced floor)."""

    def __init__(
        self,
        config: SimConfig | None,
        hazard_ref: float,
        *,
        unforced: bool = False,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        label = name or ("hazard_weighted_rule" + (" (unforced budget)" if unforced else ""))
        super().__init__(config, name=label, **kwargs)
        self._init_weighting(hazard_ref, unforced, DEFAULT_PROGRESS_VALUE)
