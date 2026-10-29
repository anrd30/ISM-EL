"""Stress configuration: an elevated hazard, for measuring data loss only.

STRESS RUNS ARE NOT DEFAULT-CONFIGURATION RESULTS. With the v0 hazard the
probability of a terminal failure (two further drive failures inside one
rebuild) is of order 1e-5 per episode, so a default-configuration run of any
feasible size observes no loss at all, and zero events in a few thousand
episodes is not an estimate of anything. A loss rate can therefore be measured
only in one of two ways: with a rare-event method, or in a run whose hazard has
been raised until losses are frequent enough to count. This module provides the
second.

The stress configuration multiplies every drive's cumulative hazard by
``STRESS_HAZARD_MULTIPLIER``. That is a deliberately unrealistic fleet. Any
loss figure produced with it must be labelled as a stress-run result, quoted
with the multiplier, and never reported as a property of the default
simulator. Every ``info`` carries ``config_label`` so the label travels with
the data; :func:`stress_config` sets it to ``"stress_hazard_x<multiplier>"``.

The multiplier is a design decision chosen only to make losses frequent enough
to count in a few hundred episodes; it is not derived from any fleet.

At this hazard level the group exposure exceeds the risk budget under every
action for essentially every scenario, so the safety filter applies the largest
allocation regardless of the requested action. A stress run therefore measures
loss under the filter's degradation rule, and says nothing about how losses
depend on the controller's choices.
See ``sim/LIMITATIONS.md``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace

from contracts.contracts import HORIZON_HOURS, HazardModel
from hazard.v0 import HeuristicHazardV0
from sim.config import SimConfig

__all__ = ["STRESS_HAZARD_MULTIPLIER", "ScaledHazard", "stress_config"]

#: Factor applied to each drive's cumulative hazard in the stress configuration.
STRESS_HAZARD_MULTIPLIER: float = 100.0

_P_MAX: float = math.nextafter(1.0, 0.0)
_UNCERTAINTY_MAX: float = 0.5


@dataclass(frozen=True)
class ScaledHazard:
    """Wrap a hazard model, multiplying its cumulative hazard by ``multiplier``.

    With the base model's horizon probability ``p`` the cumulative hazard is
    ``-ln(1 - p)``; the wrapped probability is ``1 - (1 - p) ** multiplier``,
    which stays inside ``[0, 1)`` and is monotone in both ``p`` and
    ``multiplier``. The uncertainty is scaled by the same ratio ``p' / p`` (so
    the relative width of the base model is kept) and capped at the contract's
    0.5. ``multiplier = 1`` returns the base model's output unchanged.

    Attributes:
        base: The model being scaled.
        multiplier: Cumulative-hazard factor, ``>= 1``.
    """

    base: HazardModel
    multiplier: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.multiplier) and self.multiplier >= 1.0):
            raise ValueError(f"multiplier must be finite and >= 1, got {self.multiplier!r}")

    def __call__(
        self, features: Mapping[str, float], horizon_hours: float = HORIZON_HOURS
    ) -> tuple[float, float]:
        p, uncertainty = self.base(features, horizon_hours)
        if self.multiplier == 1.0 or p <= 0.0:
            return float(p), float(uncertainty)
        scaled = min(_P_MAX, -math.expm1(self.multiplier * math.log1p(-p)))
        return float(scaled), float(min(_UNCERTAINTY_MAX, uncertainty * scaled / p))


def stress_config(
    multiplier: float = STRESS_HAZARD_MULTIPLIER, base: SimConfig | None = None
) -> SimConfig:
    """Return a configuration whose hazard is scaled to make data loss measurable.

    Args:
        multiplier: Cumulative-hazard factor.
        base: Configuration to copy (default ``SimConfig()``); only its hazard
            model and label are replaced.

    The result is labelled ``"stress_hazard_x<multiplier>"`` in every ``info``.
    Results from it are stress-run results, not default-configuration results.
    """
    config = base if base is not None else SimConfig()
    model = ScaledHazard(HeuristicHazardV0() if base is None else config.hazard_model, multiplier)
    return replace(config, hazard_model=model, label=f"stress_hazard_x{multiplier:g}")
