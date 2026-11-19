"""The exposure-weighted objective: does the hazard change decisions, and what the infeasible-budget variants trade.

Three environments on one replay profile (evaluation fleet, ``--hazard-error-sigma``, history rows for the
controller): ``contract`` (the contract reward), ``objective`` (``SimConfig.risk_objective``: the degraded-interval
cost is ``beta * hazard_true / hazard_ref`` with ``hazard_ref`` calibrated on the fleet) and ``objective_unforced``
(the same objective with ``infeasible_budget_max_rate = False``: an infeasible risk budget no longer raises the
applied action to the largest).

Arms. ``risk_aware_rule_based`` (hazard enters only through the safety filter), ``workload_only_throttle``
(risk-blind), ``hazard_weighted_rule`` (progress value scaled by the observed hazard over ``hazard_ref``),
``predictive_controller`` and ``hazard_weighted_controller``, and ``DP_hindsight_optimum`` planned under the same
objective. In the unforced environment the weighted arms use the unforced risk floor.

Tables per profile: summary by environment, arm and scenario class (all, controllable, filter-forced); paired gains
over the rule within each environment; the number of scenarios in which an arm's outcome differs from the rule's;
and, for the filter-forced seeds, the latency violation fraction against the simulated data-loss probability.

Usage::

    python -m agent.evaluate_risk_objective --workload-csv <csv> --seeds 20 --hazard-error-sigma 0.25 --out <dir>
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from agent.predictive_controller import PredictiveController
from baselines.dp_optimum import DPHindsightOptimum
from baselines.evaluate import _build_config, compare, forced_masks, to_markdown
from baselines.policies import DEFAULT_PROGRESS_VALUE, RiskAwareRuleBased, WorkloadOnlyThrottle
from baselines.risk_objective import HazardWeightedMixin, HazardWeightedRule, calibrate_hazard_ref
from contracts.contracts import PRE_FAILURE_HISTORY_ROWS
from sim.config import RiskObjective, SimConfig

__all__ = ["HazardWeightedController", "build_environments", "main"]

RULE = "risk_aware_rule_based"
BLIND = "workload_only_throttle"
WEIGHTED_RULE = "hazard_weighted_rule"
CONTROLLER = "predictive_controller"
WEIGHTED_CONTROLLER = "hazard_weighted_controller"
OPTIMUM = "DP_hindsight_optimum (not deployable)"


class HazardWeightedController(HazardWeightedMixin, PredictiveController):
    """``PredictiveController`` with the hazard-weighted progress value (and optionally the unforced floor)."""

    def __init__(self, config: SimConfig | None, hazard_ref: float, *, unforced: bool = False, name: str | None = None) -> None:
        super().__init__(config, name=name or WEIGHTED_CONTROLLER)
        self._init_weighting(hazard_ref, unforced, DEFAULT_PROGRESS_VALUE)


def build_environments(base: SimConfig, hazard_ref: float) -> dict[str, SimConfig]:
    objective = dataclasses.replace(base, risk_objective=RiskObjective(hazard_ref))
    return {
        "contract": base,
        "objective": objective,
        "objective_unforced": dataclasses.replace(objective, infeasible_budget_max_rate=False),
    }


def _arms(name: str, config: SimConfig, hazard_ref: float) -> list[Any]:
    arms: list[Any] = [RiskAwareRuleBased(config), PredictiveController(config, name=CONTROLLER)]
    if name == "contract":
        return arms + [DPHindsightOptimum(config)]
    unforced = name == "objective_unforced"
    arms += [
        HazardWeightedRule(config, hazard_ref, unforced=unforced, name=WEIGHTED_RULE),
        HazardWeightedController(config, hazard_ref, unforced=unforced, name=WEIGHTED_CONTROLLER),
        DPHindsightOptimum(config),
    ]
    if name == "objective":
        arms.insert(1, WorkloadOnlyThrottle(config))
    return arms


def _scenario_flags(episodes: pd.DataFrame) -> pd.DataFrame:
    return episodes[episodes["policy"] == RULE].set_index("seed")


def _summary(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for env_name, block in episodes.groupby("environment", sort=False):
        flags = _scenario_flags(block)
        masks = {"all": pd.Series(False, index=flags.index), "primary_controllable": forced_masks(flags)["primary"]}
        masks["filter_forced"] = masks["primary_controllable"].copy()
        for cls, forced in masks.items():
            keep = ~forced if cls != "filter_forced" else forced
            seeds = keep[keep].index
            for policy, g in block.groupby("policy", sort=False):
                g = g[g["seed"].isin(seeds)]
                rows.append({"environment": env_name, "class": cls, "policy": policy, "n": len(g),
                             "total_reward": g["total_reward"].mean(), "viol_frac": g["latency_violation_fraction"].mean(),
                             "data_loss_probability": g["data_loss_probability"].mean(),
                             "completion_h": g["completion_hours"].mean(), "delivered_mbps": g["mean_delivered_mbps"].mean()})
    return pd.DataFrame(rows)


def _gains(episodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for env_name, block in episodes.groupby("environment", sort=False):
        flags = _scenario_flags(block)
        masks = {"all": pd.Series(False, index=flags.index), "primary_controllable": forced_masks(flags)["primary"]}
        rule = block[block["policy"] == RULE].set_index("seed")["total_reward"]
        for policy, g in block.groupby("policy", sort=False):
            if policy == RULE:
                continue
            gain = g.set_index("seed")["total_reward"] - rule
            for cls, forced in masks.items():
                d = gain[~forced.reindex(gain.index)]
                n = len(d)
                sem = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
                rows.append({"environment": env_name, "class": cls, "policy": policy, "n": n, "gain_over_rule": float(d.mean()),
                             "sem": sem, "t": float(d.mean() / sem) if sem > 0 else math.nan,
                             "scenarios_differing_from_rule": int((gain.abs() > 1e-9).sum())})
    return pd.DataFrame(rows)


def _forced_trade(episodes: pd.DataFrame, forced_seeds: Sequence[int]) -> pd.DataFrame:
    rows = []
    for env_name, block in episodes.groupby("environment", sort=False):
        for policy, g in block.groupby("policy", sort=False):
            g = g[g["seed"].isin(forced_seeds)]
            for _, r in g.iterrows():
                rows.append({"environment": env_name, "policy": policy, "seed": int(r["seed"]),
                             "viol_frac": r["latency_violation_fraction"], "data_loss_probability": r["data_loss_probability"],
                             "total_reward": r["total_reward"], "completion_h": r["completion_hours"],
                             "delivered_mbps": r["mean_delivered_mbps"], "risk_budget_violated": r["risk_budget_violated"]})
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.evaluate_risk_objective", description=__doc__.split("\n\n")[0])
    parser.add_argument("--workload-csv", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--hazard-error-sigma", type=float, default=0.25)
    parser.add_argument("--forced-seeds", default="", help="comma-separated seeds shown in the violation/data-loss table")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.eval_fleet = True
    args.latency_filter = False
    args.latency_filter_causal = False
    base = dataclasses.replace(_build_config(args), history_rows=PRE_FAILURE_HISTORY_ROWS)
    hazard_ref = calibrate_hazard_ref(base)
    frames = []
    for name, config in build_environments(base, hazard_ref).items():
        frame = compare(_arms(name, config, hazard_ref), profiles=[args.workload_csv], seeds=list(range(args.seeds)), config=config)
        frames.append(frame.assign(environment=name))
    episodes = pd.concat(frames, ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(args.out / "episodes.csv", index=False, lineterminator="\n")
    summary, gains = _summary(episodes), _gains(episodes)
    summary.to_csv(args.out / "summary.csv", index=False, lineterminator="\n", float_format="%.6g")
    gains.to_csv(args.out / "gains.csv", index=False, lineterminator="\n", float_format="%.6g")
    parts = [f"# Exposure-weighted objective, `{args.workload_csv.name}`, hazard_ref = {hazard_ref:.6g} per hour", "",
             "## Summary", "", to_markdown(summary), "", "## Gain over the rule", "", to_markdown(gains)]
    seeds = [int(s) for s in args.forced_seeds.split(",") if s.strip()]
    if seeds:
        trade = _forced_trade(episodes, seeds)
        trade.to_csv(args.out / "forced_trade.csv", index=False, lineterminator="\n", float_format="%.6g")
        parts += ["", f"## Filter-forced seeds {seeds}: violation fraction against data-loss probability", "", to_markdown(trade)]
    (args.out / "summary.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"hazard_ref={hazard_ref}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
