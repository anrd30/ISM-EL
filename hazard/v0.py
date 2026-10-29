"""v0 failure-hazard model: a constant-rate baseline with a bounded heuristic multiplier.

This module is a deliberately simple stand-in that satisfies the frozen
``contracts.contracts.HazardModel`` interface, so the simulator and the agent
can run against a hazard source before a statistical risk model is available,
and so a statistical model can replace it without any caller changing. It is
not fitted, validated or calibrated against any data. Every number below is a
stated assumption with the source given next to it. The multiplier weights and
the uncertainty rule are design choices; nothing here is a measurement, and
none of it should be described as a risk estimate in a result.

Baseline
--------
The baseline is an annualised failure rate (AFR) of 2.11 %, i.e. 0.0211
failures per drive-year. The project derives that figure from its own Q1 2022
Backblaze inspection: 276 failures over 4.78M drive-days for three pooled
Seagate models (ST4000DM000, ST8000NM0055, ST12000NM0008), and
276 / 4.78e6 * 365 = 0.02107 failures per drive-year. This module takes the
figure as a design input and does not verify it. The same figure underlies the
contract's risk-budget arithmetic (``lambda = 0.0211 / 8760`` per hour).

The AFR is read as a rate in failures per drive-year (failures divided by
drive-years), not as the probability that a drive fails within one year; the
contract's derivation does the same. With a constant rate ``lambda`` the
failure probability over a horizon of ``H`` hours is ``1 - exp(-lambda * H)``,
with ``lambda = AFR / 8760``.

Severity multiplier
-------------------
The baseline alone is constant, so a controller fed only the baseline has
nothing to learn from. The cumulative hazard over the horizon is therefore
multiplied by a factor ``m`` in ``[1, MULTIPLIER_CAP]`` computed from a few
SMART-style raw attributes:

``smart_187_raw``  reported uncorrectable errors   weight 0.35
``smart_197_raw``  current pending sector count    weight 0.35
``smart_5_raw``    reallocated sector count        weight 0.25
``smart_188_raw``  command timeouts                weight 0.10
``smart_9_raw``    power-on hours (age)            weight 0.05

Why this subset: the first three are media-error counters. Reported
uncorrectable errors and pending sectors are the attributes most often singled
out in the SMART failure-prediction literature as the strongest simple signals,
hence the largest weights; reallocated sectors are a strong but later signal
(the drive has already remapped), hence somewhat less. Command timeouts are
noisier (they also follow cabling and host faults), hence small. Age is a weak,
slowly varying covariate, hence the smallest weight and a linear rather than
logarithmic form. Other attributes are left out to keep the v0 explainable.
The weights, the saturation counts and the ordering among attributes are
assumptions; they were not estimated from data.

Each error count ``x`` maps to a severity ``s(x) = min(1, log1p(x) /
log1p(x_sat))``: zero at ``x = 0``, strictly increasing for ``0 < x < x_sat``
(any nonzero count raises the multiplier), and saturating at ``x_sat``. The
logarithm reflects that the first few errors carry most of the signal, so the
10th error adds less than the 1st. The multiplier is

    m = MULTIPLIER_CAP ** min(1, sum_i w_i * s_i)

Interpolating geometrically keeps ``m`` in ``[1, MULTIPLIER_CAP]`` by
construction and makes a severity score act on the hazard multiplicatively
rather than additively. The weights sum to 1.10, so a drive that is severe on
both strongest attributes reaches the cap without needing every attribute at
its maximum.

Range: ``m = 1`` when every error counter is zero and age is zero. For one
attribute alone, a single pending sector gives about 1.3x and ten give about
2.2x; both strongest attributes at 10 give about 5.0x; the cap (10x) is
reached once the weighted score reaches 1. Age alone contributes at most
``MULTIPLIER_CAP ** 0.05`` = 1.12x.

Cap. ``MULTIPLIER_CAP = 10`` is chosen against the contract's observation
scale. Field 5 is a group sum and saturates at 8.3x baseline group hazard,
where every action violates the risk budget. A per-drive multiplier of 10 lets
one fully degraded drive sit past the 8.3x point on a per-drive basis, so the
cap is not a ceiling below the range the controller must discriminate, and it
stays inside the 1x to 10x span the contract's risk table covers. A cap well
below 8.3 would make the all-actions-violate regime unreachable even if every
survivor were degraded; a cap far above would let one drive saturate field 5 on
its own. Note the group effect: one drive at 10x among nine survivors with the
other eight at 1x gives a group hazard of (8 + 10) / 9 = 2x baseline, a
normalised field 5 of about 0.24. Saturating field 5 requires a mean per-drive
multiple of at least 8.3, so most survivors would have to be severely degraded.
That follows from the contract's group aggregation, not from a choice made
here.

Missing features. A feature that is absent, not a number, NaN or negative
contributes severity 0, i.e. a multiplier factor of 1.0, and is counted as
missing. It is never imputed: imputing a value (a fleet mean, say) would
fabricate evidence, in either direction, that the record does not contain. The
cost of this choice is that a drive with no telemetry reads as baseline risk,
which is optimistic if missing telemetry is itself a sign of trouble; that is
the reason missing features widen the reported uncertainty. ``+inf`` is
treated as a saturated count (a reading beyond any sensible range) rather than
as missing.

Uncertainty
-----------
The returned uncertainty is a stated rule, not a measured dispersion:

    uncertainty = min(0.5, p * (REL_UNCERTAINTY_BASE
                                + REL_UNCERTAINTY_MISSING * missing_fraction))

with ``REL_UNCERTAINTY_BASE = 0.25`` (a standard deviation of 25 % of the
returned probability when every feature is present) and
``REL_UNCERTAINTY_MISSING = 0.50`` (up to 50 percentage points more when every
feature is missing). It is non-zero for every input, since ``p`` is positive,
so the agent is never told the estimate is exact, and it is capped at the
contract's 0.5 bound on the horizon standard deviation. Both constants are
assumptions.

The independence and constant-rate limitations stated in the contract
(section 8) apply unchanged: the group aggregate sums per-drive hazards as if
independent, and the per-hour rate is the horizon average under a
constant-rate assumption.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from contracts.contracts import (
    HORIZON_HOURS,
    ContractViolation,
    HazardModel,
    aggregate_group_hazard,
)

__all__ = [
    "ANNUALISED_FAILURE_RATE",
    "HOURS_PER_YEAR",
    "MULTIPLIER_CAP",
    "FEATURE_NAMES",
    "baseline_horizon_probability",
    "severity_multiplier",
    "HeuristicHazardV0",
    "group_hazard",
]

#: Design input, not verified here: 276 failures / 4.78M drive-days * 365
#: for three pooled Seagate models, the project's own Q1 2022 Backblaze inspection.
ANNUALISED_FAILURE_RATE: float = 0.0211

#: 365 d x 24 h/d; the same constant the contract's risk-budget derivation uses.
HOURS_PER_YEAR: float = 8760.0

#: Upper bound of the severity multiplier. Design choice, justified in the module docstring.
MULTIPLIER_CAP: float = 10.0

#: Weight of each feature in the severity score. Assumptions (see module
#: docstring): the ordering follows the literature's view of which attributes
#: are the strongest simple signals; the magnitudes are not estimated from data.
_WEIGHTS: dict[str, float] = {
    "smart_187_raw": 0.35,
    "smart_197_raw": 0.35,
    "smart_5_raw": 0.25,
    "smart_188_raw": 0.10,
    "smart_9_raw": 0.05,
}

#: Count at which each logarithmic severity saturates at 1. Assumptions.
_SATURATION_COUNT: dict[str, float] = {
    "smart_187_raw": 10.0,
    "smart_197_raw": 10.0,
    "smart_5_raw": 100.0,
    "smart_188_raw": 100.0,
}

#: Power-on hours at which age severity reaches 1 (5 years x 8760 h). Assumption.
_AGE_SATURATION_HOURS: float = 5.0 * HOURS_PER_YEAR

#: Names of the features the model reads, in a fixed order.
FEATURE_NAMES: tuple[str, ...] = tuple(_WEIGHTS)

#: Relative standard deviation of the returned probability with every feature present. Assumption.
REL_UNCERTAINTY_BASE: float = 0.25

#: Extra relative standard deviation when every feature is missing, scaled by the missing fraction. Assumption.
REL_UNCERTAINTY_MISSING: float = 0.50

#: The contract's upper bound on the horizon standard deviation.
_UNCERTAINTY_MAX: float = 0.5


def _baseline_cumulative_hazard(horizon_hours: float, afr: float) -> float:
    """Return ``lambda * H`` with ``lambda = afr / HOURS_PER_YEAR`` (validated)."""
    if isinstance(horizon_hours, bool) or not (
        isinstance(horizon_hours, (int, float)) and math.isfinite(horizon_hours) and horizon_hours > 0.0
    ):
        raise ContractViolation(f"horizon_hours must be finite and > 0, got {horizon_hours!r}")
    if isinstance(afr, bool) or not (isinstance(afr, (int, float)) and math.isfinite(afr) and afr >= 0.0):
        raise ContractViolation(f"afr must be finite and >= 0, got {afr!r}")
    return (float(afr) / HOURS_PER_YEAR) * float(horizon_hours)


def baseline_horizon_probability(
    horizon_hours: float = HORIZON_HOURS,
    afr: float = ANNUALISED_FAILURE_RATE,
) -> float:
    """Return the constant-rate failure probability over ``horizon_hours``.

    ``1 - exp(-lambda * H)`` with ``lambda = afr / HOURS_PER_YEAR`` failures per
    drive-hour. The conversion is computed on every call so that a change to the
    AFR or the horizon stays consistent. ``expm1`` keeps precision at the small
    values involved.

    Raises:
        ContractViolation: unless ``horizon_hours`` is finite and positive and
            ``afr`` is finite and non-negative.
    """
    return -math.expm1(-_baseline_cumulative_hazard(horizon_hours, afr))


def _read_feature(features: Mapping[str, float], name: str) -> float | None:
    """Return a usable non-negative feature value, or ``None`` if missing.

    Absent, non-numeric, NaN and negative values are all "missing". A raw SMART
    count cannot be negative, so a negative value is a malformed record, not
    evidence. ``bool`` is treated as non-numeric.
    """
    value = features.get(name)
    if value is None or isinstance(value, bool):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(x) or x < 0.0:
        return None
    return x


def _severity(name: str, x: float) -> float:
    """Map a non-negative feature value to a severity in ``[0, 1]``."""
    if name == "smart_9_raw":
        return min(1.0, x / _AGE_SATURATION_HOURS)
    if math.isinf(x):
        return 1.0
    return min(1.0, math.log1p(x) / math.log1p(_SATURATION_COUNT[name]))


def severity_multiplier(features: Mapping[str, float]) -> tuple[float, int]:
    """Return ``(multiplier, n_missing)`` for one drive's features.

    ``multiplier`` lies in ``[1, MULTIPLIER_CAP]``; the module docstring gives
    the form and what a missing feature means. ``n_missing`` counts the entries
    of ``FEATURE_NAMES`` that were absent or unusable.
    """
    score = 0.0
    n_missing = 0
    for name in FEATURE_NAMES:
        x = _read_feature(features, name)
        if x is None:
            n_missing += 1
            continue
        score += _WEIGHTS[name] * _severity(name, x)
    return MULTIPLIER_CAP ** min(1.0, score), n_missing


@dataclass(frozen=True)
class HeuristicHazardV0:
    """Heuristic v0 implementation of ``contracts.contracts.HazardModel``.

    Not fitted, validated or calibrated. Calling an instance returns
    ``(p_horizon, uncertainty)``, both over ``horizon_hours``, as builtin floats:

    - ``p_horizon = 1 - exp(-m * lambda * H)`` where ``lambda = afr / 8760`` and
      ``m`` is :func:`severity_multiplier`. The multiplier scales the cumulative
      hazard rather than the probability, so ``p_horizon`` stays strictly inside
      ``[0, 1)`` and is monotone in ``m``.
    - ``uncertainty`` follows the stated relative-width rule in the module
      docstring, capped at the contract's 0.5.

    Attributes:
        afr: Annualised failure rate in failures per drive-year.
    """

    afr: float = ANNUALISED_FAILURE_RATE

    def __call__(
        self, features: Mapping[str, float], horizon_hours: float = HORIZON_HOURS
    ) -> tuple[float, float]:
        """Return ``(p_horizon, uncertainty)`` for one drive.

        Raises:
            ContractViolation: if ``horizon_hours`` is not finite and positive,
                or ``afr`` is invalid.
        """
        cumulative_hazard = _baseline_cumulative_hazard(horizon_hours, self.afr)
        multiplier, n_missing = severity_multiplier(features)
        p = -math.expm1(-multiplier * cumulative_hazard)
        p = min(p, math.nextafter(1.0, 0.0))
        relative = REL_UNCERTAINTY_BASE + REL_UNCERTAINTY_MISSING * (n_missing / len(FEATURE_NAMES))
        uncertainty = min(_UNCERTAINTY_MAX, p * relative)
        return float(p), float(uncertainty)


def group_hazard(
    model: HazardModel,
    drive_features: Sequence[Mapping[str, float]],
    horizon_hours: float = HORIZON_HOURS,
) -> tuple[float, float]:
    """Return observation fields 5 and 6 for the surviving drives of a group.

    Calls ``model`` once per drive with ``horizon_hours`` and passes the
    resulting pairs to the contract's ``aggregate_group_hazard``; this function
    does no conversion or aggregation of its own. The result is
    ``(hazard, hazard_uncertainty)`` in per-hour, group-level units.

    Raises:
        ContractViolation: from ``aggregate_group_hazard`` for an empty
            sequence, more than ``GROUP_SIZE - 1`` drives, or invalid values.
    """
    pairs = [model(features, horizon_hours) for features in drive_features]
    return aggregate_group_hazard(pairs, horizon_hours)
