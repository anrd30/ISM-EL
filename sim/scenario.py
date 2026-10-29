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

Drive features are drawn as follows. For a drive at level ``l``, each error
counter is Poisson with mean ``expm1(l * log1p(C))`` where ``C`` is the mean
count at ``l = 1`` (``_MEAN_COUNT_AT_FULL_DEGRADATION``). That mean is zero at
``l = 0`` and grows geometrically, so the first errors are rare and later ones
accumulate quickly. Power-on hours share one batch age per episode with a small
per-drive spread. All of these shapes and constants are design decisions of
this generator, not fitted to any fleet.

Limitation. Features are fixed for the whole episode; a real drive's counters
would change over the roughly 28 to 167 hours of a rebuild. This and the other
fleet and workload limitations are collected in ``sim/LIMITATIONS.md``.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
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
]

#: Surviving drives at episode start: one drive of the group has already failed.
N_SURVIVORS: int = GROUP_SIZE - 1

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
            control interval.
    """

    level: float
    features: tuple[dict[str, float], ...]
    pairs: np.ndarray
    interval_fail_prob: np.ndarray


def sample_fleet(
    rng: np.random.Generator,
    model: HazardModel,
    level: float | None = None,
    drive_spread_sd: float = 0.15,
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

    Each drive's failure probability per control interval is
    ``1 - (1 - h) ** (CONTROL_INTERVAL_S / 3600)`` with ``h`` the drive's
    per-hour hazard from the contract's ``hazard_p_to_per_hour``.
    """
    fleet_level = float(rng.uniform(0.0, 1.0)) if level is None else float(level)
    batch_age_hours = float(rng.uniform(*_BATCH_AGE_YEARS)) * HOURS_PER_YEAR
    drive_levels = np.clip(fleet_level + drive_spread_sd * rng.standard_normal(N_SURVIVORS), 0.0, 1.0)
    age_noise = rng.standard_normal(N_SURVIVORS)

    features: list[dict[str, float]] = []
    pairs = np.empty((N_SURVIVORS, 2), dtype=np.float64)
    fail_prob = np.empty(N_SURVIVORS, dtype=np.float64)
    for i in range(N_SURVIVORS):
        drive: dict[str, float] = {}
        for name in FEATURE_NAMES:
            if name == "smart_9_raw":
                drive[name] = max(0.0, batch_age_hours * (1.0 + _AGE_SPREAD * float(age_noise[i])))
            else:
                mean = math.expm1(float(drive_levels[i]) * math.log1p(_MEAN_COUNT_AT_FULL_DEGRADATION[name]))
                drive[name] = float(rng.poisson(mean))
        features.append(drive)
        p_horizon, uncertainty = model(drive, HORIZON_HOURS)
        pairs[i] = (p_horizon, uncertainty)
        hazard_per_hour = hazard_p_to_per_hour(p_horizon, HORIZON_HOURS)
        fail_prob[i] = -math.expm1((CONTROL_INTERVAL_S / 3600.0) * math.log1p(-hazard_per_hour))
    return Fleet(
        level=fleet_level,
        features=tuple(features),
        pairs=pairs,
        interval_fail_prob=fail_prob,
    )
