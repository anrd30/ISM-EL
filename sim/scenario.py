"""Scenario generation: foreground replay traces and the surviving-drive fleet.

Two independent parts feed an episode.

Foreground trace
----------------
A replay frame follows the contract's ``REPLAY_COLUMNS`` schema (24 hours,
``N_ROWS`` = 288 rows at ``CONTROL_INTERVAL_S``) and comes either from
``workload.generate.generate_profile`` or from a CSV. Bandwidth columns are KB/s
in the frame; they are converted to MB/s here, once, with the contract's
``kbps_to_mbps``.

An episode may run for ``MAX_EPISODE_STEPS`` = 2400 intervals, longer than the
288-row frame, so the trace is tiled cyclically. Cyclic tiling means the same
day repeats back to back, including its burst pattern and its diurnal phase;
real load would not repeat exactly. This is a simplification, and ``info``
records how many times the trace wrapped.

Provenance. The CSV format carries no synthetic marker, so a sidecar
``profile_<name>.meta.json`` next to the CSV is the out-of-band record. Its
``synthetic`` flag is carried through when it is a boolean; without a sidecar
the flag is ``None`` (unknown), never a guess.

Surviving-drive fleet
---------------------
Nine survivors each get SMART-style features (the names the v0 hazard model
reads) which are then passed to the hazard model. The features are *correlated
across drives*: an episode first draws a fleet degradation level, and each
drive's own level is that value plus a modest per-drive deviation. A group
hazard is a sum over nine drives, so degrading a single drive to the v0
multiplier cap lifts it to only about twice baseline; sampling one bad drive
per episode would never present the policy with the upper range of its most
important input. Sampling shared degradation also reflects the correlated
failure within an enclosure (shared batch, temperature, vibration, age) that
the contract's independence assumption leaves out; this generator exercises
the regime in which that assumption is weakest.

Two feature generators are provided and are selected with ``feature_fn``.

``v0_drive_features`` (the ``sample_fleet`` default) draws the five features the
v0 hazard model reads, as follows. For a drive at level ``l``, each error
counter is Poisson with mean ``expm1(l * log1p(C))`` where ``C`` is the mean
count at ``l = 1`` (``_MEAN_COUNT_AT_FULL_DEGRADATION``). That mean is zero at
``l = 0`` and grows geometrically, so the first errors are rare and later ones
accumulate quickly. Power-on hours share one batch age per episode with a small
per-drive spread. All of these shapes and constants are design decisions of
this generator, not fitted to any fleet.

``v1_drive_features`` draws the inputs of the v1 hazard model from
``sim/fleet_features_v1.json`` (built by ``hazard/build_fleet_features.py`` from
the training split). A drive's model is drawn once with the training-split
drive shares as weights. For each value column that is not structurally missing
for that model, with probability equal to the drive's level the value is drawn
uniformly from the model's failure-day values of that column, otherwise from
its healthy values. Structurally missing columns are omitted from the dict.
Values are drawn independently per column, so the joint structure of a real
drive's columns is not reproduced.

Limitation. Features are fixed for the whole episode; a real drive's counters
would change over the roughly 28 to 167 hours of a rebuild. This and the other
fleet and workload limitations are collected in ``sim/LIMITATIONS.md``.

Estimation error
----------------
The hazard model's output is an estimate. When ``hazard_error_sigma`` is
positive, each drive's true per-hour hazard is the estimate times a lognormal
multiplier::

    true_i = estimate_i * exp(Z_i),   Z_i ~ N(-sigma^2 / 2, sigma^2)

Why lognormal: hazards are positive and span orders of magnitude, so error is
naturally multiplicative, and a log-space normal keeps the multiplier
positive. Why the ``-sigma^2 / 2`` shift: ``E[exp(Z)] = exp(mu + sigma^2 / 2)``,
so ``mu = -sigma^2 / 2`` gives ``E[multiplier] = 1`` exactly. The error then
adds dispersion without shifting the fleet's mean hazard; a biased multiplier
would change the baseline failure rate and confound every comparison.

Why once per drive per episode, not per interval: the error of a risk model on
a given drive is a property of that drive's features, which this generator
holds fixed for the episode, so the same drive is mis-estimated by the same
factor throughout. Fresh noise each interval would average out over a long
rebuild (about 28 to 167 hours, hundreds of intervals) and recover the
estimate, restoring the degeneracy the error exists to remove.

Only failure draws use the true hazard. The observation, the safety filter and
the exposure it computes use the estimate. The multipliers come from their own
random stream (``HAZARD_ERROR_STREAM``), never from the stream that samples the
fleet, so switching the error on leaves every feature, estimate and workload
unchanged and a paired comparison against ``hazard_error_sigma = 0`` differs
only in the failure probabilities. With sigma 0 no random number is drawn and
the multipliers are exactly 1.0.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from contracts.contracts import (
    CONTROL_INTERVAL_S,
    GROUP_SIZE,
    HORIZON_HOURS,
    HazardModel,
    hazard_p_to_per_hour,
    kbps_to_mbps,
    validate_replay_frame,
)
from hazard.v0 import FEATURE_NAMES, HOURS_PER_YEAR

__all__ = [
    "N_SURVIVORS",
    "ReplayTrace",
    "Fleet",
    "trace_from_frame",
    "load_replay_csv",
    "sample_fleet",
    "sample_hazard_multipliers",
    "HAZARD_ERROR_STREAM",
    "FleetFeatureFn",
    "FLEET_FEATURES_PATH",
    "FLEET_FEATURES_TEST_PATH",
    "FLEET_FEATURES_VENDOR_HOLDOUT_PATH",
    "FleetTable",
    "v0_drive_features",
    "v1_drive_features",
    "v1_eval_drive_features",
    "v1_vendor_holdout_drive_features",
    "load_fleet_table",
    "draw_v1_drive_features",
]

#: Path of the empirical feature table used by the v1 generator.
FLEET_FEATURES_PATH: Path = Path(__file__).resolve().parent / "fleet_features_v1.json"

#: Path of the empirical feature table built from the held-out split, used by the evaluation generator.
FLEET_FEATURES_TEST_PATH: Path = Path(__file__).resolve().parent / "fleet_features_v1_test.json"

#: Path of the empirical feature table built from the drives of the held-out drive model (vendor-holdout split).
FLEET_FEATURES_VENDOR_HOLDOUT_PATH: Path = Path(__file__).resolve().parent / "fleet_features_vendor_holdout.json"

#: Surviving drives at episode start: one drive of the group has already failed.
N_SURVIVORS: int = GROUP_SIZE - 1

#: Index of the per-episode random stream that draws the hazard estimation
#: error, alongside stream 0 (scenario) and stream 1 (failures) of the
#: environment. [DERIVED] only in the sense of being the next unused index.
HAZARD_ERROR_STREAM: int = 2

#: Ceiling on a true per-hour hazard so that ``log1p(-h)`` stays finite when a
#: large multiplier is applied to a large estimate.
_MAX_TRUE_HAZARD_PER_HOUR: float = 1.0 - 1e-12

#: Mean error count at full degradation (level 1) per error counter. Design
#: decision of this generator: the counts at which the v0 model's severity for
#: that counter saturates, so a drive at level 1 sits at the saturation point of
#: each counter and a wholly degraded group can reach a saturated hazard
#: observation without most scenarios sitting in the saturated regime, where
#: every action violates the budget and the controller has no choice left.
_MEAN_COUNT_AT_FULL_DEGRADATION: dict[str, float] = {
    "smart_187_raw": 10.0,
    "smart_197_raw": 10.0,
    "smart_5_raw": 100.0,
    "smart_188_raw": 100.0,
}

#: Batch age range in years; a group shares a manufacturing batch, so ages are
#: close. Design decision.
_BATCH_AGE_YEARS: tuple[float, float] = (0.1, 5.0)

#: Relative per-drive spread of power-on hours around the batch age.
_AGE_SPREAD: float = 0.1


@dataclass(frozen=True)
class ReplayTrace:
    """Foreground load of one group, already converted to MB/s.

    Attributes:
        fg_read_iops: Read IOPS per row.
        fg_write_iops: Write IOPS per row.
        fg_bw_mbps: Read plus write bandwidth per row, MB/s.
        profile: Label of the workload (profile name or file label).
        synthetic: Whether the data is synthetic, ``None`` when unknown.
    """

    fg_read_iops: np.ndarray
    fg_write_iops: np.ndarray
    fg_bw_mbps: np.ndarray
    profile: str
    synthetic: bool | None

    def __len__(self) -> int:
        return int(self.fg_bw_mbps.shape[0])


def trace_from_frame(frame: pd.DataFrame, profile: str, synthetic: bool | None) -> ReplayTrace:
    """Validate a replay frame and convert its bandwidth to MB/s.

    The frame is checked with the contract's ``validate_replay_frame`` (columns,
    dtypes, nulls, negatives, time grid). Read and write KB/s are each
    converted with ``kbps_to_mbps`` and summed; this is the only conversion.

    Raises:
        ContractViolation: if the frame breaks the replay schema.
    """
    validate_replay_frame(frame)
    read_mbps = np.array([kbps_to_mbps(x) for x in frame["read_bw_kbps"].tolist()], dtype=np.float64)
    write_mbps = np.array([kbps_to_mbps(x) for x in frame["write_bw_kbps"].tolist()], dtype=np.float64)
    return ReplayTrace(
        fg_read_iops=frame["read_iops"].to_numpy(dtype=np.float64),
        fg_write_iops=frame["write_iops"].to_numpy(dtype=np.float64),
        fg_bw_mbps=read_mbps + write_mbps,
        profile=profile,
        synthetic=synthetic,
    )


def _read_sidecar(csv_path: Path) -> Mapping[str, Any] | None:
    """Return the parsed ``<stem>.meta.json`` next to a CSV, or ``None``."""
    sidecar = csv_path.with_suffix(".meta.json")
    try:
        loaded = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return loaded if isinstance(loaded, dict) else None


def load_replay_csv(path: str | os.PathLike[str]) -> ReplayTrace:
    """Load a replay CSV, carrying the sidecar's ``synthetic`` flag through.

    The workload label is the sidecar's ``profile`` when it is a string, else
    the file stem. ``synthetic`` is the sidecar's value only when it is a
    ``bool``; a missing, unreadable or malformed sidecar gives ``None``.

    Raises:
        ContractViolation: if the CSV breaks the replay schema.
        OSError: if the CSV cannot be read.
    """
    csv_path = Path(path)
    frame = pd.read_csv(csv_path)
    meta = _read_sidecar(csv_path)
    synthetic: bool | None = None
    label = csv_path.stem
    if meta is not None:
        flag = meta.get("synthetic")
        synthetic = flag if isinstance(flag, bool) else None
        name = meta.get("profile")
        if isinstance(name, str) and name:
            label = name
    return trace_from_frame(frame, label, synthetic)


@dataclass(frozen=True)
class Fleet:
    """Per-episode state of the nine surviving drives.

    Attributes:
        level: Fleet degradation level in ``[0, 1]``.
        features: SMART-style features per drive.
        pairs: ``(N_SURVIVORS, 2)`` array of ``(p_horizon, uncertainty)`` from
            the hazard model, one row per drive.
        interval_fail_prob: Probability that each drive fails within one
            control interval, computed from the *true* per-hour hazard.
        estimated_hazard_per_hour: Per-hour hazard of each drive converted
            from the model's output (what the observation and the filter see).
        true_hazard_per_hour: Per-hour hazard each drive's failures are drawn
            from: the estimate times ``hazard_multiplier``, capped just below 1.
        hazard_multiplier: Per-drive lognormal estimation-error factor, all
            ones when the error is off. Fixed for the episode.
    """

    level: float
    features: tuple[dict[str, Any], ...]
    pairs: np.ndarray
    interval_fail_prob: np.ndarray
    estimated_hazard_per_hour: np.ndarray
    true_hazard_per_hour: np.ndarray
    hazard_multiplier: np.ndarray


def sample_hazard_multipliers(rng: np.random.Generator, sigma: float, n: int = N_SURVIVORS) -> np.ndarray:
    """Draw ``n`` lognormal multipliers ``exp(N(-sigma^2/2, sigma^2))``.

    The mean of each multiplier is exactly 1 (see the module docstring). With
    ``sigma == 0`` no random number is drawn and ones are returned, so the
    stream state is untouched.

    Raises:
        ValueError: if ``sigma`` is negative or not finite.
    """
    if not (math.isfinite(sigma) and sigma >= 0.0):
        raise ValueError(f"sigma must be finite and >= 0, got {sigma!r}")
    if sigma == 0.0:
        return np.ones(n, dtype=np.float64)
    return np.exp(rng.normal(-0.5 * sigma * sigma, sigma, size=n))


#: One drive's feature builder: ``(rng, drive_level, batch_age_hours, age_noise)``
#: to the feature mapping passed to the hazard model. ``drive_level`` is in
#: ``[0, 1]``; ``batch_age_hours`` is the episode's batch age and ``age_noise``
#: the drive's standard-normal draw for the age spread.
FleetFeatureFn = Callable[[np.random.Generator, float, float, float], dict[str, Any]]


def v0_drive_features(
    rng: np.random.Generator, level: float, batch_age_hours: float, age_noise: float
) -> dict[str, Any]:
    """Return one drive's features for the v0 hazard model (see the module docstring)."""
    drive: dict[str, Any] = {}
    for name in FEATURE_NAMES:
        if name == "smart_9_raw":
            drive[name] = max(0.0, batch_age_hours * (1.0 + _AGE_SPREAD * float(age_noise)))
        else:
            mean = math.expm1(float(level) * math.log1p(_MEAN_COUNT_AT_FULL_DEGRADATION[name]))
            drive[name] = float(rng.poisson(mean))
    return drive


@dataclass(frozen=True)
class FleetTable:
    """Empirical feature pools loaded from the table written by ``hazard/build_fleet_features.py``.

    Attributes:
        value_columns: Value columns of the hazard model, in table order.
        categories: Drive model names, in table order.
        shares: Training-split drive share of each category, same order.
        pools: ``pools[model][column]`` is ``(healthy, failure)`` float arrays,
            or ``None`` when the column is structurally missing for the model.
    """

    value_columns: tuple[str, ...]
    categories: tuple[str, ...]
    shares: np.ndarray
    pools: dict[str, dict[str, tuple[np.ndarray, np.ndarray] | None]]


@lru_cache(maxsize=None)
def load_fleet_table(path: Path = FLEET_FEATURES_PATH) -> FleetTable:
    """Load and cache the empirical feature table at ``path``."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    categories = tuple(raw["model_categories"])
    shares = np.array([raw["models"][m]["share"] for m in categories], dtype=np.float64)
    pools: dict[str, dict[str, tuple[np.ndarray, np.ndarray] | None]] = {}
    for model in categories:
        pools[model] = {}
        for column, entry in raw["models"][model]["columns"].items():
            if entry["structurally_missing"]:
                pools[model][column] = None
            else:
                pools[model][column] = (
                    np.array(entry["healthy"], dtype=np.float64),
                    np.array(entry["failure"], dtype=np.float64),
                )
    return FleetTable(tuple(raw["value_columns"]), categories, shares / shares.sum(), pools)


def draw_v1_drive_features(table: FleetTable, rng: np.random.Generator, level: float) -> dict[str, Any]:
    """Return one drive's inputs for the v1 hazard model drawn from ``table``.

    The drive's model is drawn with the table's shares. Each column that is not
    structurally missing for that model takes a value from the model's failure
    pool with probability ``level`` and from its healthy pool otherwise; an
    empty pool is replaced by the other one. Structurally missing columns are
    omitted. The ``model`` key holds the drawn model name.
    """
    model = table.categories[int(rng.choice(len(table.categories), p=table.shares))]
    drive: dict[str, Any] = {}
    for column in table.value_columns:
        pool = table.pools[model].get(column)
        if pool is None:
            continue
        healthy, failure = pool
        values = failure if rng.random() < level else healthy
        if values.shape[0] == 0:
            values = healthy if values is failure else failure
        if values.shape[0] == 0:
            continue
        drive[column] = float(values[int(rng.integers(0, values.shape[0]))])
    drive["model"] = model
    return drive


def v1_drive_features(
    rng: np.random.Generator, level: float, batch_age_hours: float, age_noise: float
) -> dict[str, Any]:
    """Return one drive's v1 inputs from ``FLEET_FEATURES_PATH`` (batch age and noise are unused)."""
    return draw_v1_drive_features(load_fleet_table(FLEET_FEATURES_PATH), rng, level)


def v1_eval_drive_features(
    rng: np.random.Generator, level: float, batch_age_hours: float, age_noise: float
) -> dict[str, Any]:
    """Return one drive's v1 inputs from ``FLEET_FEATURES_TEST_PATH`` (batch age and noise are unused)."""
    return draw_v1_drive_features(load_fleet_table(FLEET_FEATURES_TEST_PATH), rng, level)


def v1_vendor_holdout_drive_features(
    rng: np.random.Generator, level: float, batch_age_hours: float, age_noise: float
) -> dict[str, Any]:
    """Return one drive's v1 inputs from ``FLEET_FEATURES_VENDOR_HOLDOUT_PATH`` (batch age and noise are unused)."""
    return draw_v1_drive_features(load_fleet_table(FLEET_FEATURES_VENDOR_HOLDOUT_PATH), rng, level)


def sample_fleet(
    rng: np.random.Generator,
    model: HazardModel,
    level: float | None = None,
    drive_spread_sd: float = 0.15,
    hazard_error_sigma: float = 0.0,
    error_rng: np.random.Generator | None = None,
    feature_fn: FleetFeatureFn = v0_drive_features,
) -> Fleet:
    """Sample a correlated fleet of surviving drives and score it.

    Args:
        rng: Source of randomness.
        model: Per-drive risk model, called once per drive.
        level: Fleet degradation level in ``[0, 1]``; ``None`` draws it
            uniformly, so that scenarios range from a wholly healthy group to
            one in which most survivors are degraded together.
        drive_spread_sd: Standard deviation of each drive's deviation from the
            fleet level.
        hazard_error_sigma: Log-space standard deviation of the multiplicative
            estimation error (see the module docstring); ``0.0`` means truth
            equals the estimate.
        error_rng: Source of the error multipliers, kept separate from ``rng``
            so the fleet is identical with and without error. Required when
            ``hazard_error_sigma > 0``.
        feature_fn: Per-drive feature builder, called once per drive with
            ``(rng, drive_level, batch_age_hours, age_noise)``. It must match
            ``model``: the default builds the v0 model's inputs.

    Each drive's failure probability per control interval is
    ``1 - (1 - h) ** (CONTROL_INTERVAL_S / 3600)`` with ``h`` the drive's
    *true* per-hour hazard: the estimate from the contract's
    ``hazard_p_to_per_hour`` times the drive's error multiplier.

    Raises:
        ValueError: if ``hazard_error_sigma`` is invalid, or positive with no
            ``error_rng``.
    """
    if not (math.isfinite(hazard_error_sigma) and hazard_error_sigma >= 0.0):
        raise ValueError(f"hazard_error_sigma must be finite and >= 0, got {hazard_error_sigma!r}")
    if hazard_error_sigma > 0.0 and error_rng is None:
        raise ValueError("error_rng is required when hazard_error_sigma > 0")
    # Own stream: drawing the multipliers cannot perturb the fleet stream.
    multipliers = (
        np.ones(N_SURVIVORS, dtype=np.float64)
        if error_rng is None
        else sample_hazard_multipliers(error_rng, hazard_error_sigma, N_SURVIVORS)
    )
    fleet_level = float(rng.uniform(0.0, 1.0)) if level is None else float(level)
    batch_age_hours = float(rng.uniform(*_BATCH_AGE_YEARS)) * HOURS_PER_YEAR
    drive_levels = np.clip(fleet_level + drive_spread_sd * rng.standard_normal(N_SURVIVORS), 0.0, 1.0)
    age_noise = rng.standard_normal(N_SURVIVORS)

    features: list[dict[str, Any]] = []
    pairs = np.empty((N_SURVIVORS, 2), dtype=np.float64)
    fail_prob = np.empty(N_SURVIVORS, dtype=np.float64)
    estimated_hazard = np.empty(N_SURVIVORS, dtype=np.float64)
    true_hazard = np.empty(N_SURVIVORS, dtype=np.float64)
    for i in range(N_SURVIVORS):
        drive = feature_fn(rng, float(drive_levels[i]), batch_age_hours, float(age_noise[i]))
        features.append(drive)
        p_horizon, uncertainty = model(drive, HORIZON_HOURS)
        pairs[i] = (p_horizon, uncertainty)
        hazard_per_hour = hazard_p_to_per_hour(p_horizon, HORIZON_HOURS)
        estimated_hazard[i] = hazard_per_hour
        true_per_hour = min(hazard_per_hour * float(multipliers[i]), _MAX_TRUE_HAZARD_PER_HOUR)
        true_hazard[i] = true_per_hour
        fail_prob[i] = -math.expm1((CONTROL_INTERVAL_S / 3600.0) * math.log1p(-true_per_hour))
    return Fleet(
        level=fleet_level,
        features=tuple(features),
        pairs=pairs,
        interval_fail_prob=fail_prob,
        estimated_hazard_per_hour=estimated_hazard,
        true_hazard_per_hour=true_hazard,
        hazard_multiplier=multipliers,
    )
