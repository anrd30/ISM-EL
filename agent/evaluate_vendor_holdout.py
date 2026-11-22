"""Paired evaluation of the rule, controller, oracle and DP optimum under a hazard model fitted without one drive vendor.

Same arms, scenarios, seeds, pairing and statistics as ``agent.evaluate_predictive`` (with its ``gap_table``); only the
hazard model and the fleet generator differ. ``--config`` selects:

- ``vendor_holdout``: ``sim.config.vendor_holdout_config`` (hazard model fitted on the Seagate drive models only, fleets drawn
  from the held-out drive model's feature table).
- ``pooled_vendor_fleet``: the default pooled hazard model with the same held-out-model fleets (separates the effect of
  the hazard model from the effect of the fleet).

Usage::

    python -m agent.evaluate_vendor_holdout --config vendor_holdout \
        --workload-csv data_pipeline/profiles/profile_diurnal_long_holdout.csv --seeds 20 --hazard-error-sigma 0.25 \
        --out agent/evidence/hazard-vendor-holdout-2026-11-20/vendor_holdout
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
from typing import Sequence

from agent.evaluate_predictive import CONTROLLER, ORACLE, OPTIMUM, RULE, SEASONAL, build_arms, gap_table
from baselines.evaluate import compare, forced_seed_sets, summarise, to_markdown
from contracts.contracts import PRE_FAILURE_HISTORY_ROWS
from sim.config import SimConfig, vendor_holdout_config
from sim.scenario import v1_vendor_holdout_drive_features

CONFIGS = ("vendor_holdout", "pooled_vendor_fleet")


def build_config(name: str, sigma: float) -> SimConfig:
    label = f"{name},sigma={sigma:g}"
    if name == "vendor_holdout":
        return vendor_holdout_config(hazard_error_sigma=sigma, label=label)
    if name == "pooled_vendor_fleet":
        return SimConfig(feature_fn=v1_vendor_holdout_drive_features, hazard_error_sigma=sigma, label=label)
    raise ValueError(f"unknown config {name!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.evaluate_vendor_holdout", description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", choices=CONFIGS, required=True)
    parser.add_argument("--workload-csv", type=Path, action="append", required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--hazard-error-sigma", type=float, default=0.25)
    parser.add_argument("--history-rows", type=int, default=PRE_FAILURE_HISTORY_ROWS)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    config = dataclasses.replace(build_config(args.config, args.hazard_error_sigma), history_rows=args.history_rows)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    arms = build_arms(config)
    episodes = compare(arms, profiles=args.workload_csv, seeds=seeds, config=config)
    args.out.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(args.out / "episodes.csv", index=False, lineterminator="\n")
    treatments = {
        SEASONAL: [SEASONAL],
        "rule+history": ["rule+history"],
        "predictive_controller[seasonal]+history": ["predictive_controller[seasonal]+history"],
        "predictive_controller[adaptive] (no history)": ["predictive_controller[adaptive] (no history)"],
        CONTROLLER: [CONTROLLER],
        ORACLE: [ORACLE],
        OPTIMUM: [OPTIMUM],
    }
    references = {RULE: [RULE], SEASONAL: [SEASONAL]}
    table = gap_table(episodes, treatments, references)
    table.to_csv(args.out / "gaps.csv", index=False, lineterminator="\n", float_format="%.6g")
    parts = [
        f"# Vendor-holdout evaluation, `{config.label}`, history_rows={args.history_rows}",
        "",
        "Scenarios are identical for every arm. Gains are paired by scenario seed; `sem` is across scenarios. `primary` is the "
        "registered definition of controllable (the safety filter raises a `fixed_20` rollout on fewer than 10% of the intervals).",
        "",
        to_markdown(forced_seed_sets(episodes)),
        "",
        "## Summary, all scenarios",
        "",
        to_markdown(summarise(episodes)),
        "",
        "## Paired gains by block",
        "",
        to_markdown(table),
    ]
    (args.out / "summary.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    print("\n".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
