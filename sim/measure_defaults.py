"""Violation fraction per constant action and profile, healthy fleet.

Reproduces the method of the table in ``sim/LIMITATIONS.md`` section 9: for each
workload profile and each action of ``ACTIONS_MBPS``, run one episode per seed
with that action chosen at every step on a healthy fleet
(``fleet_degradation = 0``, ``drive_spread_sd = 0``) and take the mean over
episodes of the share of steps with ``info["violated_latency"]``. The last
column counts the actions whose mean fraction is at most
``LATENCY_VIOLATION_FRACTION_MAX``.

Usage (from the repository root)::

    python -m sim.measure_defaults                  # SimConfig() default (v1)
    python -m sim.measure_defaults --hazard v0      # heuristic v0 with its feature function
    python -m sim.measure_defaults --seeds 3 --profiles idle peak

The output is two markdown tables (mean and worst seed) and, per profile, the
share of steps on which the safety filter reduced the requested action. The
figures are simulated and uncalibrated.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections.abc import Sequence

import numpy as np

from contracts.contracts import ACTIONS_MBPS, LATENCY_VIOLATION_FRACTION_MAX, MAX_EPISODE_STEPS
from sim.config import SimConfig, v0_config
from sim.recovery_env import RecoveryEnv
from workload.generate import PROFILE_NAMES

__all__ = ["DEFAULT_SEEDS", "measure", "format_tables", "main"]

DEFAULT_SEEDS = 20


def _episode(env: RecoveryEnv, action: int, seed: int) -> tuple[float, float]:
    """Run one constant-action episode; return (violation fraction, filter-clipped fraction)."""
    env.reset(seed=seed)
    violated: list[bool] = []
    clipped: list[bool] = []
    for _ in range(MAX_EPISODE_STEPS + 1):
        _, _, terminated, truncated, info = env.step(action)
        violated.append(bool(info["violated_latency"]))
        clipped.append(bool(info["alloc_clipped_by_filter"]))
        if terminated or truncated:
            break
    return float(np.mean(violated)), float(np.mean(clipped))


def measure(
    config: SimConfig,
    seeds: Sequence[int] = tuple(range(DEFAULT_SEEDS)),
    profiles: Sequence[str] = PROFILE_NAMES,
) -> dict[str, dict[str, list[float]]]:
    """Return per-profile lists, one entry per action in ``ACTIONS_MBPS`` order.

    Keys of each profile entry: ``mean`` and ``worst`` (violation fraction over
    seeds) and ``clipped`` (mean share of steps on which the filter reduced the
    action). ``config`` is copied with the healthy-fleet settings and the
    profile pinned; its hazard model and feature function are used as given.
    """
    out: dict[str, dict[str, list[float]]] = {}
    for profile in profiles:
        cfg = dataclasses.replace(config, workload_profile=profile, fleet_degradation=0.0, drive_spread_sd=0.0)
        env = RecoveryEnv(cfg)
        entry: dict[str, list[float]] = {"mean": [], "worst": [], "clipped": []}
        for action in range(len(ACTIONS_MBPS)):
            runs = [_episode(env, action, seed) for seed in seeds]
            entry["mean"].append(float(np.mean([v for v, _ in runs])))
            entry["worst"].append(float(np.max([v for v, _ in runs])))
            entry["clipped"].append(float(np.mean([c for _, c in runs])))
        out[profile] = entry
    return out


def format_tables(result: dict[str, dict[str, list[float]]]) -> str:
    """Return the mean table of section 9, then the worst-seed and filter-clipping tables."""
    header = "| Profile | " + " | ".join(f"{int(a)} MB/s" for a in ACTIONS_MBPS)
    lines = [header + " | Actions within the 5% budget |", "|---|" + "---|" * (len(ACTIONS_MBPS) + 1)]
    for profile, entry in result.items():
        feasible = sum(f <= LATENCY_VIOLATION_FRACTION_MAX for f in entry["mean"])
        lines.append(f"| {profile} | " + " | ".join(f"{f:.3f}" for f in entry["mean"]) + f" | {feasible} |")
    for title, key in (("worst seed", "worst"), ("share of steps clipped by the filter", "clipped")):
        lines += ["", f"{title}:", "", header + " |", "|---|" + "---|" * len(ACTIONS_MBPS)]
        for profile, entry in result.items():
            lines.append(f"| {profile} | " + " | ".join(f"{f:.3f}" for f in entry[key]) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--hazard", choices=("v1", "v0"), default="v1",
                        help="v1: SimConfig() default; v0: heuristic hazard with its feature function")
    parser.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="episodes per profile and action")
    parser.add_argument("--profiles", nargs="+", choices=PROFILE_NAMES, default=list(PROFILE_NAMES))
    args = parser.parse_args(argv)
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")
    config = SimConfig() if args.hazard == "v1" else v0_config()
    print(f"hazard={args.hazard} seeds={args.seeds} profiles={args.profiles}")
    print(format_tables(measure(config, seeds=range(args.seeds), profiles=args.profiles)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
