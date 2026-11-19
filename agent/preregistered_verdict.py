"""Verdict of ``agent/PREREGISTRATION_2026-11-19.md`` from the episode files of the fresh populations.

For each population: on the scenarios controllable under a definition, ``gain`` = total reward of
``predictive_controller (history)`` minus that of ``risk_aware_rule_based``; condition 1 is ``t = mean / sem > 2``
(``sem`` = sample standard deviation / sqrt(n) across scenarios), condition 2 is that the mean paired difference of
``latency_violation_fraction`` (controller minus rule) is not greater than 0. The criterion is met only if both
conditions hold on every population under the primary definition; a population with fewer than
``MIN_CONTROLLABLE`` controllable scenarios does not meet it.

Usage::

    python -m agent.preregistered_verdict <episodes.csv> [<episodes.csv> ...] [--out verdict.md]
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import pandas as pd

from agent.evaluate_predictive import CONTROLLER, OPTIMUM, RULE
from baselines.evaluate import forced_masks

__all__ = ["MIN_CONTROLLABLE", "T_THRESHOLD", "population_verdict", "verdict", "CRITERION_TEXT", "main"]

MIN_CONTROLLABLE = 8
T_THRESHOLD = 2.0
CRITERION_TEXT = (
    "On each of C and D, on the scenarios controllable under the primary definition: (1) t = mean / sem of the "
    "per-scenario gain (total reward of predictive_controller (history) minus that of risk_aware_rule_based) is "
    "greater than 2; and (2) the mean over the same scenarios of (latency_violation_fraction of the controller minus "
    "that of the rule) is not greater than 0. The criterion is met only if both conditions hold on both C and D."
)


def population_verdict(episodes: pd.DataFrame, definition: str = "primary") -> dict:
    """Figures and the two conditions for one population (one profile) under one definition of controllable."""
    ctrl = episodes[episodes["policy"] == CONTROLLER].set_index("seed")
    rule = episodes[episodes["policy"] == RULE].set_index("seed")
    masks = {"all": pd.Series(False, index=rule.index), **forced_masks(rule)}
    keep = ~masks[definition].reindex(ctrl.index)
    gain = (ctrl["total_reward"] - rule["total_reward"])[keep]
    viol = (ctrl["latency_violation_fraction"] - rule["latency_violation_fraction"])[keep]
    n = len(gain)
    sem = float(gain.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
    vsem = float(viol.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
    t = float(gain.mean() / sem) if n > 1 and sem > 0 else math.nan
    out = {
        "profile": str(episodes["profile"].iloc[0]), "definition": definition, "n": n,
        "better": int((gain > 0).sum()), "worse": int((gain < 0).sum()),
        "mean_gain": float(gain.mean()), "sem": sem, "t": t,
        "viol_diff": float(viol.mean()), "viol_diff_sem": vsem,
        "condition_1": bool(n >= MIN_CONTROLLABLE and t > T_THRESHOLD),
        "condition_2": bool(n >= MIN_CONTROLLABLE and viol.mean() <= 0.0),
    }
    if OPTIMUM in set(episodes["policy"]):
        dp = episodes[episodes["policy"] == OPTIMUM].set_index("seed")
        headroom = (dp["total_reward"] - rule["total_reward"])[keep]
        out["percent_of_optimum"] = float(100.0 * gain.mean() / headroom.mean())
    out["met"] = out["condition_1"] and out["condition_2"]
    return out


def verdict(frames: Sequence[pd.DataFrame]) -> tuple[bool, str]:
    """The criterion's verdict over the populations, and the report text."""
    defs = ("primary", "clip_0.01", "clip_0.25", "fixed20_cost", "all")
    lines = ["# Verdict against the pre-registration", "", "Criterion as registered:", "", f"> {CRITERION_TEXT}", ""]
    primary = []
    for frame in frames:
        rows = [population_verdict(frame, d) for d in defs]
        primary.append(rows[0])
        lines += [f"## {rows[0]['profile']}", "",
                  "| definition | n | better / worse | mean gain | sem | t | violation diff (sem) | share of optimum | "
                  "condition 1 (t > 2) | condition 2 (viol <= 0) |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(
                f"| {r['definition']} | {r['n']} | {r['better']} / {r['worse']} | {r['mean_gain']:+.3f} | {r['sem']:.3f} | "
                f"{r['t']:.2f} | {r['viol_diff']:+.5f} ({r['viol_diff_sem']:.5f}) | "
                f"{r.get('percent_of_optimum', float('nan')):.1f}% | {r['condition_1']} | {r['condition_2']} |"
            )
        lines.append("")
    met = all(r["met"] for r in primary)
    lines += ["## Verdict (primary definition, every fresh population)", "",
              f"**{'PASS' if met else 'FAIL'}**: " + "; ".join(
                  f"{r['profile']}: t = {r['t']:.2f} (condition 1 {'met' if r['condition_1'] else 'not met'}), "
                  f"violation difference {r['viol_diff']:+.5f} (condition 2 {'met' if r['condition_2'] else 'not met'})"
                  for r in primary)]
    return met, "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.preregistered_verdict", description=__doc__.split("\n\n")[0])
    parser.add_argument("episodes", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    met, text = verdict([pd.read_csv(p) for p in args.episodes])
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    print(text)
    return 0 if met else 1


if __name__ == "__main__":
    raise SystemExit(main())
