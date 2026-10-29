"""Synthetic foreground-workload generator.

SYNTHETIC DATA NOTICE. Every value produced by this module is drawn from a
parametric random process defined in this file. None of it is derived from, or
fitted to, any real trace, device or application. The output files are
synthetic placeholders that share only the *schema* of real replay files; they
carry no information about any real system and must be labelled synthetic
wherever they are used or reported.

Purpose. The simulator replays a foreground workload while a rebuild competes
for the same bandwidth budget. This module produces replay files whose format
is identical to the real pipeline's output (column names and order from
``contracts.contracts.REPLAY_COLUMNS``, time step ``CONTROL_INTERVAL_S``,
episode-relative ``t_seconds`` starting at 0, bandwidth in KB/s), so a consumer
cannot distinguish the schema of a generated file from a pipeline file. Format
identity is the requirement; statistical realism is not claimed.

Each profile is the aggregate foreground load of one storage group (not of one
disk) over 24 hours. Profiles are scaled so that the 95th percentile of total
(read + write) bandwidth equals a fixed fraction of ``BW_TOTAL_MBPS``.

Calibration is imposed, not emergent. The p95 of each profile is set by
construction: the shape is multiplied by a constant chosen so that its
empirical p95 equals the target. A measured p95 of 80% of the group budget for
``peak`` (or 10%, 50%, 60% for the others) is therefore a property of this
scaling step and says nothing about any process, system or dataset; it must not
be cited as a measured or validated result.

Assumptions fixed by this generator (synthetic, not derived from data): t=0 is
midnight; the diurnal peak is near 14:00 and its trough near 02:00; used
capacity starts at 50-70% of the group's data-drive capacity. Total bandwidth
is clipped at ``BW_TOTAL_MBPS``; the clip is a guard and does not bind at the
current targets (it lies above every profile's p95 and above the tallest
bursts observed), so it plays no part in shaping the profiles.

Provenance. Alongside each ``profile_<name>.csv`` the CLI writes
``profile_<name>.meta.json`` (generator, version, profile, seed, targets,
measured p95, request size, read fraction, timestamp) and one
``_SYNTHETIC.json`` per output directory. The CSV format itself carries no
marker, so that it stays identical to the real pipeline's output; the sidecars
are the out-of-band record that a file is synthetic.

Library use::

    from workload.generate import generate_profile
    frame = generate_profile("bursty", seed=3)

Command line::

    python -m workload.generate --out-dir <dir> [--seed N]
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from contracts.contracts import (
    BW_TOTAL_MBPS,
    CONTROL_INTERVAL_S,
    DRIVE_CAPACITY_GB,
    GROUP_DATA_DRIVES,
    REPLAY_COLUMNS,
    REPLAY_DTYPES,
    kbps_to_mbps,
    validate_replay_frame,
)

__all__ = [
    "PROFILE_NAMES",
    "P95_FRACTION_OF_BUDGET",
    "N_ROWS",
    "GENERATOR_MODULE",
    "GENERATOR_VERSION",
    "SYNTHETIC_MARKER_FILE",
    "generate_profile",
    "p95_total_bw_mbps",
    "write_profiles",
    "main",
]

GENERATOR_MODULE: str = "workload.generate"
GENERATOR_VERSION: str = "0.1.0"
SYNTHETIC_MARKER_FILE: str = "_SYNTHETIC.json"

SECONDS_PER_DAY: int = 24 * 3600
N_ROWS: int = SECONDS_PER_DAY // CONTROL_INTERVAL_S

PROFILE_NAMES: tuple[str, ...] = ("idle", "diurnal", "bursty", "peak")

# Target p95 of total bandwidth, as a fraction of BW_TOTAL_MBPS. The 0.8 for
# "peak" follows the aggregation rule in contracts/INTERFACES.md section 10,
# decision 5 (peak profile p95 aggregate near 0.8 x BW_TOTAL_MBPS, 160 MB/s).
# The other fractions are design choices of this generator.
P95_FRACTION_OF_BUDGET: dict[str, float] = {
    "idle": 0.10,
    "diurnal": 0.50,
    "bursty": 0.60,
    "peak": 0.80,
}

# Mean request size (KB) per profile; all within the 16-64 KB band used for
# this workload class. IOPS is derived from bandwidth and this size.
MEAN_REQUEST_KB: dict[str, float] = {
    "idle": 24.0,
    "diurnal": 32.0,
    "bursty": 40.0,
    "peak": 32.0,
}

READ_FRACTION: float = 0.7  # mean share of total bandwidth that is reads
READ_FRACTION_SD: float = 0.03
SIZE_JITTER_SD: float = 0.08  # lognormal sigma of per-row request-size factor
SIZE_JITTER_CLIP: tuple[float, float] = (0.85, 1.15)
IOPS_JITTER_SD: float = 0.04  # independent multiplicative IOPS jitter
IOPS_JITTER_CLIP: tuple[float, float] = (0.90, 1.10)

# Initial used capacity as a fraction of the group's usable (data-drive)
# capacity, in MB (decimal, as in the contract), and the total drift per day.
USAGE_FRACTION_RANGE: tuple[float, float] = (0.50, 0.70)
USAGE_DRIFT_FRACTION_PER_DAY: float = 5e-4

_PERCENTILE: float = 95.0
_BW_CAP_KBPS: float = BW_TOTAL_MBPS * 1000.0  # foreground cannot exceed the group budget


def _ar1(rng: np.random.Generator, n: int, rho: float, sd: float) -> np.ndarray:
    """Zero-mean stationary AR(1) series with marginal standard deviation ``sd``."""
    innovations = rng.normal(0.0, 1.0, n)
    out = np.empty(n, dtype=np.float64)
    out[0] = sd * innovations[0]
    scale = sd * np.sqrt(1.0 - rho * rho)
    for i in range(1, n):
        out[i] = rho * out[i - 1] + scale * innovations[i]
    return out


def _shape_idle(rng: np.random.Generator, n: int) -> np.ndarray:
    """Flat level with mild correlated noise."""
    return np.maximum(1.0 + _ar1(rng, n, rho=0.5, sd=0.08), 0.2)


def _shape_diurnal(rng: np.random.Generator, n: int) -> np.ndarray:
    """One smooth daily cycle: trough near 02:00, peak near 14:00 (t=0 is midnight)."""
    hours = np.arange(n) * CONTROL_INTERVAL_S / 3600.0
    day = 0.5 * (1.0 + np.cos(2.0 * np.pi * (hours - 14.0) / 24.0))  # 0 at trough, 1 at peak
    cycle = 0.25 + 0.75 * day
    return cycle * np.maximum(1.0 + _ar1(rng, n, rho=0.6, sd=0.04), 0.5)


def _shape_bursty(rng: np.random.Generator, n: int) -> np.ndarray:
    """Low baseline with short rectangular bursts of random height and length."""
    baseline = np.maximum(1.0 + _ar1(rng, n, rho=0.5, sd=0.10), 0.3)
    burst = np.zeros(n, dtype=np.float64)
    starts = np.flatnonzero(rng.random(n) < 0.06)
    durations = rng.integers(1, 4, size=starts.size)  # 1-3 intervals
    heights = rng.uniform(4.0, 9.0, size=starts.size)  # multiples of the baseline
    for s, d, h in zip(starts, durations, heights):
        burst[s : s + d] = np.maximum(burst[s : s + d], h)
    return baseline + burst


def _shape_peak(rng: np.random.Generator, n: int) -> np.ndarray:
    """Sustained heavy level with slow wander and fast noise."""
    slow = _ar1(rng, n, rho=0.95, sd=0.05)
    fast = _ar1(rng, n, rho=0.3, sd=0.03)
    return np.maximum(1.0 + slow + fast, 0.5)


_SHAPES = {
    "idle": _shape_idle,
    "diurnal": _shape_diurnal,
    "bursty": _shape_bursty,
    "peak": _shape_peak,
}


def _total_bw_kbps(name: str, rng: np.random.Generator, n: int) -> np.ndarray:
    """Total bandwidth in KB/s, scaled so its empirical p95 hits the profile target.

    The p95 target is ``P95_FRACTION_OF_BUDGET[name] * BW_TOTAL_MBPS`` converted
    to KB/s (x1000, the decimal convention of the contract). The p95 is imposed
    by this scaling, not an emergent property of the shape. Scaling is applied
    to the whole series, so the shape is unchanged. Values are then clipped at
    the group budget as a guard; the clip lies above every target p95, so it
    does not move the p95, and it does not bind at the current targets.
    """
    shape = _SHAPES[name](rng, n)
    target_kbps = P95_FRACTION_OF_BUDGET[name] * BW_TOTAL_MBPS * 1000.0
    scaled = shape * (target_kbps / float(np.percentile(shape, _PERCENTILE)))
    return np.minimum(scaled, _BW_CAP_KBPS)


def generate_profile(name: str, seed: int = 0) -> pd.DataFrame:
    """Generate one synthetic 24-hour replay frame.

    The result has exactly ``REPLAY_COLUMNS`` in order and ``N_ROWS`` rows at
    ``CONTROL_INTERVAL_S`` spacing, with ``t_seconds`` starting at 0. Bandwidth
    columns are KB/s. Every column is ``int64`` as ``REPLAY_DTYPES`` requires,
    and the frame passes ``validate_replay_frame`` before it is returned.
    IOPS is derived from the (integer) bandwidth and a per-profile mean
    request size (with independent jitter), so the two stay consistent;
    ``disk_usage_mb`` is a slowly drifting near-constant.

    Quantisation order: each series is first scaled so its float p95 hits the
    target, then rounded to nearest integer (``numpy.rint``, never truncation).
    Rounding moves the total-bandwidth p95 by at most about 1 KB/s against
    targets of 20,000 to 160,000 KB/s, far inside the 2% test tolerance, so no
    pre-rounding correction is applied. The p95 reported in sidecars and by the
    CLI is measured on the rounded values actually written.

    The same ``(name, seed)`` always yields an identical frame. Randomness comes
    only from a ``numpy.random.Generator`` seeded with ``[seed, profile index]``,
    so different profiles under one seed use independent streams and global
    numpy state is never touched. All values are synthetic.

    Raises:
        ValueError: if ``name`` is not one of ``PROFILE_NAMES``.
    """
    if name not in PROFILE_NAMES:
        raise ValueError(f"unknown profile {name!r}; expected one of {PROFILE_NAMES}")
    n = N_ROWS
    rng = np.random.default_rng([seed, PROFILE_NAMES.index(name)])

    total_bw = _total_bw_kbps(name, rng, n)
    read_share = np.clip(READ_FRACTION + rng.normal(0.0, READ_FRACTION_SD, n), 0.5, 0.9)
    # Quantise bandwidth to integer KB/s. Each total is rounded to nearest and
    # the read share is rounded from it, with write = total - read, so the
    # integer read and write columns sum exactly to the rounded total.
    total_bw_i = np.rint(total_bw).astype(np.int64)
    read_bw_i = np.rint(total_bw_i * read_share).astype(np.int64)
    write_bw_i = total_bw_i - read_bw_i

    # IOPS = bandwidth / request size, with an independent jitter per column.
    size_kb = MEAN_REQUEST_KB[name] * np.clip(
        np.exp(rng.normal(0.0, SIZE_JITTER_SD, n)), *SIZE_JITTER_CLIP
    )
    read_jitter = np.clip(1.0 + rng.normal(0.0, IOPS_JITTER_SD, n), *IOPS_JITTER_CLIP)
    write_jitter = np.clip(1.0 + rng.normal(0.0, IOPS_JITTER_SD, n), *IOPS_JITTER_CLIP)
    read_iops = read_bw_i / size_kb * read_jitter
    write_iops = write_bw_i / size_kb * write_jitter

    # Used capacity: start fraction of the group's data capacity (MB), monotone
    # non-negative drift across the day.
    capacity_mb = GROUP_DATA_DRIVES * DRIVE_CAPACITY_GB * 1000.0
    start_mb = rng.uniform(*USAGE_FRACTION_RANGE) * capacity_mb
    increments = rng.random(n)
    drift = np.cumsum(increments) / increments.sum()
    usage_mb = start_mb + drift * USAGE_DRIFT_FRACTION_PER_DAY * capacity_mb

    t_seconds = np.arange(n, dtype=np.int64) * CONTROL_INTERVAL_S

    columns = {
        "t_seconds": t_seconds,
        "read_iops": np.rint(read_iops),
        "write_iops": np.rint(write_iops),
        "read_bw_kbps": read_bw_i,
        "write_bw_kbps": write_bw_i,
        "disk_usage_mb": np.rint(usage_mb),
    }
    frame = pd.DataFrame({c: np.asarray(columns[c]).astype(REPLAY_DTYPES[c]) for c in REPLAY_COLUMNS})
    validate_replay_frame(frame)
    return frame


def p95_total_bw_mbps(frame: pd.DataFrame) -> float:
    """95th percentile of read + write bandwidth of a replay frame, in MB/s."""
    total_kbps = frame["read_bw_kbps"].to_numpy() + frame["write_bw_kbps"].to_numpy()
    return kbps_to_mbps(float(np.percentile(total_kbps, _PERCENTILE)))


def write_profiles(out_dir: Path, seed: int = 0) -> list[tuple[str, int, float]]:
    """Write ``profile_<name>.csv`` and its sidecar for every profile.

    ``out_dir`` is created if missing. CSVs use ``\\n`` line endings and no
    index column, so equal frames give byte-identical files on every platform.
    Each CSV gets a ``profile_<name>.meta.json`` sidecar, and the directory gets
    one ``_SYNTHETIC.json``. Sidecars hold a generation timestamp, so they (and
    only they) differ between runs; the CSVs do not.

    Returns:
        One ``(file_name, row_count, p95_total_bw_mbps)`` tuple per CSV.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results: list[tuple[str, int, float]] = []
    for name in PROFILE_NAMES:
        frame = generate_profile(name, seed)
        file_name = f"profile_{name}.csv"
        frame.to_csv(out_dir / file_name, index=False, lineterminator="\n")
        p95 = p95_total_bw_mbps(frame)
        _write_json(out_dir / f"profile_{name}.meta.json", _profile_meta(name, seed, p95, generated_at))
        results.append((file_name, len(frame), p95))
    _write_json(out_dir / SYNTHETIC_MARKER_FILE, _directory_marker(generated_at))
    return results


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def _profile_meta(name: str, seed: int, measured_p95_mbps: float, generated_at: str) -> dict[str, Any]:
    """Provenance record for one profile file (out-of-band synthetic label)."""
    return {
        "synthetic": True,
        "generator_module": GENERATOR_MODULE,
        "generator_version": GENERATOR_VERSION,
        "profile": name,
        "seed": seed,
        "target_p95_fraction_of_bw_total": P95_FRACTION_OF_BUDGET[name],
        "bw_total_mbps": BW_TOTAL_MBPS,
        "measured_p95_total_bw_mbps": round(measured_p95_mbps, 6),
        "p95_imposed_by_construction": True,
        "mean_request_kb": MEAN_REQUEST_KB[name],
        "read_fraction": READ_FRACTION,
        "generated_at_utc": generated_at,
        "note": (
            "Parametric output of a synthetic generator; not trace data. "
            "The p95 is set by scaling, not observed."
        ),
    }


def _directory_marker(generated_at: str) -> dict[str, Any]:
    """Directory-level label: every ``profile_*.csv`` beside it is synthetic."""
    return {
        "synthetic": True,
        "generator_module": GENERATOR_MODULE,
        "generator_version": GENERATOR_VERSION,
        "generated_at_utc": generated_at,
        "statement": (
            "Every profile_*.csv in this directory is parametric output of "
            f"{GENERATOR_MODULE}, not trace data. Per-file records are in "
            "profile_<name>.meta.json."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; returns the process exit status."""
    parser = argparse.ArgumentParser(
        prog="python -m workload.generate",
        description="Write synthetic foreground-workload replay files.",
    )
    parser.add_argument("--out-dir", required=True, type=Path, help="output directory")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default 0)")
    args = parser.parse_args(argv)

    results = write_profiles(args.out_dir, args.seed)
    print("SYNTHETIC data from a parametric generator; not trace data.")
    for file_name, rows, p95 in results:
        print(f"{file_name}: rows={rows} p95_total_bw_mbps={p95:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
