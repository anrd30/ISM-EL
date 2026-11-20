"""Sensitivity of the policy ordering to the objective: every setting re-solves the optimum and re-runs the policies.

For each setting of one factor around the default (alpha 1, beta 0.1, gamma 10, latency_unloaded_ms 2.0,
latency_target_ms 15.0) the ``SimConfig`` is rebuilt (``RewardConfig`` coefficients and the two latency
parameters), and under it ``DP_hindsight_optimum`` is solved again and ``risk_aware_rule_based``,
``rule_seasonal_forecast (hand-built)``, ``predictive_controller (history)`` and ``ORACLE_lookahead`` are run again.
The selector's latency terms come from ``config.reward`` (``baselines.policies.LatencyRule.score``) and its ceiling
from ``config.latency_target_ms``. Every policy that uses ``progress_value`` gets ``beta / PROGRESS_RATE_MBPS``
(``PROGRESS_RATE_MBPS = 25``, which is 0.004 at ``beta = 0.1``); the fixed-0.004 variant is run beside it for the
settings where ``beta`` differs from 0.1. The residual Q-policy is not included: it was trained at the default
objective and has no price to rescale. Stored episodes are not rescored.

Usage::

    python -m agent.objective_sensitivity run --workload-csv <csv> --label A --out <dir> [--worker i --workers n]
    python -m agent.objective_sensitivity analyse --dir <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from agent.predictive_controller import PredictiveController
from agent.seasonal_rule import SeasonalForecastRule
from baselines.dp_optimum import DPHindsightOptimum
from baselines.evaluate import _build_config, compare, forced_masks
from baselines.policies import DEFAULT_PROGRESS_VALUE, OracleLookahead, RiskAwareRuleBased
from contracts.contracts import PRE_FAILURE_HISTORY_ROWS, RewardConfig
from sim.config import SimConfig

__all__ = ["Setting", "settings", "make_config", "build_arms", "analyse_frames", "main"]

PROGRESS_RATE_MBPS: float = 25.0
RULE = "risk_aware_rule_based"
SEASONAL = "rule_seasonal_forecast (hand-built)"
CONTROLLER = "predictive_controller (history)"
ORACLE = "ORACLE_lookahead (not deployable)"
OPTIMUM = "DP_hindsight_optimum (not deployable)"
FIXED_SUFFIX = " [pv=0.004]"
SHORT = {OPTIMUM: "DP", ORACLE: "oracle", CONTROLLER: "ctrl", SEASONAL: "seas", RULE: "rule"}
ORDER_POLICIES = (OPTIMUM, ORACLE, CONTROLLER, SEASONAL, RULE)


@dataclass(frozen=True)
class Setting:
    name: str
    alpha: float = 1.0
    beta: float = 0.1
    gamma: float = 10.0
    latency_target_ms: float = 15.0
    latency_unloaded_ms: float = 2.0

    @property
    def progress_value(self) -> float:
        return self.beta / PROGRESS_RATE_MBPS


def settings() -> list[Setting]:
    """The default and every one-factor setting of the grid, default first, duplicates of the default removed."""
    out = [Setting("default")]
    for beta in (0.01, 0.03, 0.3, 1.0):
        out.append(Setting(f"beta={beta:g}", beta=beta))
    for gamma in (1.0, 3.0, 30.0, 100.0):
        out.append(Setting(f"gamma={gamma:g}", gamma=gamma))
    for alpha in (0.3, 3.0):
        out.append(Setting(f"alpha={alpha:g}", alpha=alpha))
    for target in (12.0, 18.0):
        out.append(Setting(f"latency_target_ms={target:g}", latency_target_ms=target))
    for unloaded in (1.5, 2.5):
        out.append(Setting(f"latency_unloaded_ms={unloaded:g}", latency_unloaded_ms=unloaded))
    return out


def make_config(base: SimConfig, s: Setting) -> SimConfig:
    return dataclasses.replace(
        base,
        reward=RewardConfig(alpha=s.alpha, beta=s.beta, gamma=s.gamma),
        latency_target_ms=s.latency_target_ms,
        latency_unloaded_ms=s.latency_unloaded_ms,
    )


def build_arms(config: SimConfig, s: Setting, with_fixed_variant: bool) -> list:
    pv = s.progress_value
    arms = [
        RiskAwareRuleBased(config, progress_value=pv),
        SeasonalForecastRule(config, progress_scale=pv / DEFAULT_PROGRESS_VALUE),
        PredictiveController(config, name=CONTROLLER, progress_value=pv),
        OracleLookahead(config, progress_value=pv),
        DPHindsightOptimum(config),
    ]
    if with_fixed_variant:
        arms += [
            RiskAwareRuleBased(config, progress_value=DEFAULT_PROGRESS_VALUE, name=RULE + FIXED_SUFFIX),
            SeasonalForecastRule(config, progress_scale=1.0, name=SEASONAL + FIXED_SUFFIX),
            PredictiveController(config, name=CONTROLLER + FIXED_SUFFIX, progress_value=DEFAULT_PROGRESS_VALUE),
            OracleLookahead(config, progress_value=DEFAULT_PROGRESS_VALUE),
        ]
        arms[-1].name = ORACLE + FIXED_SUFFIX
    return arms


KEEP = ["profile", "policy", "seed", "total_reward", "latency_violation_fraction", "completion_hours", "steps",
        "fixed20_clip_fraction", "fixed20_latency_cost", "filter_forced", "trace_start_row"]


def run_setting(workload: Path, s: Setting, base: SimConfig, seeds: Sequence[int]) -> pd.DataFrame:
    config = make_config(base, s)
    frame = compare(build_arms(config, s, with_fixed_variant=abs(s.beta - 0.1) > 1e-12), profiles=[workload],
                    seeds=list(seeds), config=config)
    return frame[KEEP].assign(setting=s.name)


def _stats(d: pd.Series) -> dict:
    n = len(d)
    sem = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
    mean = float(d.mean())
    t = mean / sem if n > 1 and sem > 0 else (math.copysign(math.inf, mean) if n > 1 and mean != 0 else math.nan)
    return {"n": n, "mean": mean, "sem": sem, "t": t,
            "better": int((d > 0).sum()), "worse": int((d < 0).sum())}


def _ordering(rewards: pd.Series) -> str:
    ranked = sorted(ORDER_POLICIES, key=lambda p: -rewards[p])
    return ">".join(SHORT[p] for p in ranked)


def analyse_frames(frames: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    """Per population, setting and scenario class (``all``, ``primary``): figures and ordering agreement.

    ``frames`` maps ``(population label, setting name)`` to the episode frame of that run. Columns: mean reward per
    policy, ``dp_minus_rule`` (absolute and as a percentage of the rule's cost), the oracle's and the controller's
    shares of it, the pooled ordering of the five policies and whether it equals the default setting's pooled ordering,
    the share of scenarios whose own ordering equals the default setting's ordering in that scenario, and the
    controller's gain over the rule with ``t``, better and worse. The fixed-0.004 variant is reported as
    ``ctrl_gain_fixed_pv`` and ``order_fixed_pv`` for the settings that have it.
    """
    rows = []
    for (pop, setting), frame in frames.items():
        flags = frame[frame["policy"] == RULE].set_index("seed")
        masks = {"all": pd.Series(False, index=flags.index), "primary": forced_masks(flags)["primary"]}
        reward = frame.pivot(index="seed", columns="policy", values="total_reward")
        for cls, forced in masks.items():
            keep = ~forced.reindex(reward.index)
            r = reward[keep]
            mean = r.mean()
            head = mean[OPTIMUM] - mean[RULE]
            row = {"population": pop, "setting": setting, "class": cls, "n": len(r),
                   "dp_minus_rule": head, "dp_minus_rule_pct_of_rule_cost": 100.0 * head / -mean[RULE],
                   "oracle_share": (mean[ORACLE] - mean[RULE]) / head, "ctrl_share": (mean[CONTROLLER] - mean[RULE]) / head,
                   "pooled_order": _ordering(mean)}
            for p in ORDER_POLICIES:
                row[f"reward_{SHORT[p]}"] = float(mean[p])
            g = _stats(r[CONTROLLER] - r[RULE])
            row.update({"ctrl_gain": g["mean"], "ctrl_sem": g["sem"], "ctrl_t": g["t"], "ctrl_better": g["better"],
                        "ctrl_worse": g["worse"], "ctrl_t_above_2": bool(g["t"] > 2)})
            if CONTROLLER + FIXED_SUFFIX in r.columns:
                gf = _stats(r[CONTROLLER + FIXED_SUFFIX] - r[RULE + FIXED_SUFFIX])
                fixed = pd.Series({p: float(mean[p + FIXED_SUFFIX]) for p in ORDER_POLICIES[1:]} | {OPTIMUM: float(mean[OPTIMUM])})
                row.update({"ctrl_gain_fixed_pv": gf["mean"], "ctrl_t_fixed_pv": gf["t"], "order_fixed_pv": _ordering(fixed)})
            row["_orders"] = r.apply(_ordering, axis=1)
            rows.append(row)
    table = pd.DataFrame(rows)
    out = []
    for (pop, cls), g in table.groupby(["population", "class"], sort=False):
        default = g[g["setting"] == "default"].iloc[0]
        for _, row in g.iterrows():
            same_pooled = row["pooled_order"] == default["pooled_order"]
            same_scenario = float((row["_orders"] == default["_orders"]).mean())
            same_as_pooled_default = float((row["_orders"] == default["pooled_order"]).mean())
            r = row.drop("_orders").to_dict()
            r.update({"matches_default_pooled": same_pooled, "share_scenarios_same_order_as_default_scenario": same_scenario,
                      "share_scenarios_with_default_pooled_order": same_as_pooled_default})
            out.append(r)
    return pd.DataFrame(out)


def headline(table: pd.DataFrame) -> tuple[str, str]:
    """The ordering-stability table and the list of exceptions, as markdown.

    One row per setting; per population and scenario class (primary controllable, all) the pooled ordering relative to
    the default setting ("same" or the ordering) and the controller's gain over the rule with ``t``. An exception is a
    (setting, population, class) whose pooled ordering differs from the default's or whose controller ``t`` is not above 2.
    """
    pops = list(dict.fromkeys(table["population"]))
    lines = ["| setting | " + " | ".join(f"{p} primary | {p} all" for p in pops) + " |",
             "|---|" + "---|" * (2 * len(pops))]
    exceptions = []
    for setting in dict.fromkeys(table["setting"]):
        cells = []
        for pop in pops:
            for cls in ("primary", "all"):
                r = table[(table["setting"] == setting) & (table["population"] == pop) & (table["class"] == cls)].iloc[0]
                order = "same order" if r["matches_default_pooled"] else r["pooled_order"]
                cells.append(f"{order}; ctrl {r['ctrl_gain']:+.2f}, t {r['ctrl_t']:.1f}{'' if r['ctrl_t_above_2'] else ' (t<=2)'}")
                if not r["matches_default_pooled"] or not r["ctrl_t_above_2"]:
                    exceptions.append(
                        f"- {setting}, population {pop}, {cls} (n={int(r['n'])}): pooled order {r['pooled_order']} "
                        f"(default {table[(table['setting'] == 'default') & (table['population'] == pop) & (table['class'] == cls)].iloc[0]['pooled_order']}); "
                        f"controller minus rule {r['ctrl_gain']:+.3f} +/- {r['ctrl_sem']:.3f}, t {r['ctrl_t']:.2f}, "
                        f"better/worse {int(r['ctrl_better'])}/{int(r['ctrl_worse'])}")
        lines.append(f"| {setting} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n", ("\n".join(exceptions) if exceptions else "none") + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.objective_sensitivity", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--workload-csv", type=Path, required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--seeds", type=int, default=20)
    r.add_argument("--worker", type=int, default=0)
    r.add_argument("--workers", type=int, default=1)
    a = sub.add_parser("analyse")
    a.add_argument("--dir", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "run":
        ns = argparse.Namespace(hazard_error_sigma=0.25, eval_fleet=True, latency_filter=False, latency_filter_causal=False)
        base = dataclasses.replace(_build_config(ns), history_rows=PRE_FAILURE_HISTORY_ROWS)
        args.out.mkdir(parents=True, exist_ok=True)
        for i, s in enumerate(settings()):
            if i % args.workers != args.worker:
                continue
            path = args.out / f"episodes_{args.label}_{s.name}.csv"
            if path.exists():
                continue
            run_setting(args.workload_csv, s, base, range(args.seeds)).to_csv(path, index=False, lineterminator="\n", float_format="%.10g")
            print(f"done {args.label} {s.name}", flush=True)
        return 0
    frames = {}
    for path in sorted(args.dir.glob("episodes_*.csv")):
        _, pop, setting = path.stem.split("_", 2)
        frames[(pop, setting)] = pd.read_csv(path)
    table = analyse_frames(frames)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "analysis.csv", index=False, lineterminator="\n", float_format="%.6g")
    head, exc = headline(table)
    (args.out / "headline.md").write_text(head, encoding="utf-8")
    (args.out / "exceptions.md").write_text(exc, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
