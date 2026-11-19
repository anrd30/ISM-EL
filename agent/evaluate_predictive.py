"""Paired evaluation of the predictive controller against its comparators on replay profiles.

Arms: fixed rates, ``risk_aware_rule_based``, ``rule_seasonal_forecast (hand-built)``, ``rule+history``,
``predictive_controller[seasonal]`` with history, ``predictive_controller[adaptive] (no history)``,
``predictive_controller (history)``, ``ORACLE_lookahead (not deployable)`` and ``DP_hindsight_optimum (not
deployable)``; with ``--q-model`` also the residual Q-policy run directories (zero-shot comparator). The history
arms use the contract's reset history (``--history-rows``, default ``PRE_FAILURE_HISTORY_ROWS``); every arm,
including those that ignore the history, is evaluated on the same scenarios.

Reported per profile and treatment: the paired gain over each reference on all scenarios and on the scenarios
controllable under every definition of ``baselines.evaluate.forced_masks``, with its standard error across
scenarios (learned models with several seeds averaged within each scenario first), ``t``, the counts better and
worse, the share of the optimum's gain (``DP_hindsight_optimum`` minus the reference on the same scenarios) and
the paired difference of the latency violation fraction.

Usage::

    python -m agent.evaluate_predictive --workload-csv data_pipeline/profiles/profile_diurnal_long_holdout.csv \\
        --seeds 20 --eval-fleet --hazard-error-sigma 0.25 --out runs/predictive
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from agent.predictive_controller import PredictiveController, PredictiveControllerNoHistory, RuleWithHistory
from agent.seasonal_rule import SeasonalForecastRule
from baselines.dp_optimum import DPHindsightOptimum
from baselines.evaluate import (
    _build_config,
    compare,
    forced_masks,
    forced_seed_sets,
    summarise,
    to_markdown,
)
from baselines.policies import OracleLookahead, RiskAwareRuleBased, fixed_rate_policies
from contracts.contracts import PRE_FAILURE_HISTORY_ROWS
from sim.config import SimConfig

__all__ = ["RULE", "SEASONAL", "ORACLE", "OPTIMUM", "CONTROLLER", "build_arms", "gap_table", "main"]

RULE = "risk_aware_rule_based"
SEASONAL = "rule_seasonal_forecast (hand-built)"
ORACLE = "ORACLE_lookahead (not deployable)"
OPTIMUM = "DP_hindsight_optimum (not deployable)"
CONTROLLER = "predictive_controller (history)"
Q_PREFIX = "residual_q"


def build_arms(config: SimConfig, q_dirs: Sequence[Path] = ()) -> list[Any]:
    """The comparison set, in report order. ``q_dirs`` are ``agent.residual_q`` run directories."""
    arms: list[Any] = [
        *fixed_rate_policies(),
        RiskAwareRuleBased(config),
        SeasonalForecastRule(config),
        RuleWithHistory(config),
        PredictiveController(config, mode="seasonal", name="predictive_controller[seasonal]+history"),
        PredictiveControllerNoHistory(config),
        PredictiveController(config, name=CONTROLLER),
    ]
    if q_dirs:
        from agent.residual_q import ResidualQPolicy

        arms += [ResidualQPolicy.from_dir(d, config, f"{Q_PREFIX} {d.name}") for d in q_dirs]
    arms += [OracleLookahead(config), DPHindsightOptimum(config)]
    return arms


def _scenario_series(frame: pd.DataFrame, policies: Sequence[str]) -> pd.Series:
    """Total reward per scenario seed; several policies (seeds of one learned model) are averaged within a scenario."""
    block = frame[frame["policy"].isin(policies)]
    return block.groupby("seed")["total_reward"].mean()


def _metric_series(frame: pd.DataFrame, policies: Sequence[str], column: str) -> pd.Series:
    block = frame[frame["policy"].isin(policies)]
    return block.groupby("seed")[column].mean()


def gap_table(
    episodes: pd.DataFrame,
    treatments: dict[str, Sequence[str]],
    references: dict[str, Sequence[str]],
    *,
    optimum: str = OPTIMUM,
) -> pd.DataFrame:
    """Paired gain of each treatment over each reference, by scenario block.

    ``treatments`` and ``references`` map a label to the policy names averaged within each scenario. A block is
    ``all`` or the scenarios controllable under a definition of ``forced_masks`` (``primary`` and the sensitivity
    variants). Columns: ``n``, ``mean_gain``, ``sem``, ``t``, ``better``, ``worse``, ``percent_of_optimum`` (gain
    over the optimum's gain over the same reference, NaN without an optimum row), ``viol_frac_diff`` and its
    ``viol_frac_diff_sem``.
    """
    rows: list[dict[str, Any]] = []
    for profile in dict.fromkeys(episodes["profile"]):
        block = episodes[episodes["profile"] == profile]
        flags = block[block["policy"] == next(iter(references.values()))[0]].set_index("seed")
        masks = {"all": pd.Series(False, index=flags.index), **forced_masks(flags)}
        has_optimum = optimum in set(block["policy"])
        for t_label, t_names in treatments.items():
            for r_label, r_names in references.items():
                gain = _scenario_series(block, t_names) - _scenario_series(block, r_names)
                viol = _metric_series(block, t_names, "latency_violation_fraction") - _metric_series(
                    block, r_names, "latency_violation_fraction"
                )
                head = (_scenario_series(block, [optimum]) - _scenario_series(block, r_names)) if has_optimum else None
                for name, forced in masks.items():
                    keep = ~forced.reindex(gain.index)
                    g, v = gain[keep], viol[keep]
                    n = len(g)
                    sem = float(g.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
                    vsem = float(v.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
                    rows.append(
                        {
                            "profile": profile, "treatment": t_label, "reference": r_label, "block": name, "n": n,
                            "mean_gain": float(g.mean()), "sem": sem,
                            "t": float(g.mean() / sem) if n > 1 and sem > 0 else math.nan,
                            "better": int((g > 0).sum()), "worse": int((g < 0).sum()),
                            "percent_of_optimum": (
                                float(100.0 * g.mean() / head[keep].mean()) if head is not None else math.nan
                            ),
                            "viol_frac_diff": float(v.mean()), "viol_frac_diff_sem": vsem,
                        }
                    )
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.evaluate_predictive", description=__doc__.split("\n\n")[0])
    parser.add_argument("--workload-csv", type=Path, action="append", required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--hazard-error-sigma", type=float, default=0.0)
    parser.add_argument("--eval-fleet", action="store_true")
    parser.add_argument("--history-rows", type=int, default=PRE_FAILURE_HISTORY_ROWS)
    parser.add_argument("--q-model", type=Path, nargs="*", default=[])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.latency_filter = False
    args.latency_filter_causal = False
    base = _build_config(args)
    config = dataclasses.replace(base, history_rows=args.history_rows)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    arms = build_arms(config, args.q_model)
    episodes = compare(arms, profiles=args.workload_csv, seeds=seeds, config=config)
    args.out.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(args.out / "episodes.csv", index=False, lineterminator="\n")

    q_names = [a.name for a in arms if a.name.startswith(Q_PREFIX)]
    treatments: dict[str, Sequence[str]] = {
        SEASONAL: [SEASONAL],
        "rule+history": ["rule+history"],
        "predictive_controller[seasonal]+history": ["predictive_controller[seasonal]+history"],
        "predictive_controller[adaptive] (no history)": ["predictive_controller[adaptive] (no history)"],
        CONTROLLER: [CONTROLLER],
        ORACLE: [ORACLE],
        OPTIMUM: [OPTIMUM],
    }
    if q_names:
        treatments[f"{Q_PREFIX} (seeds averaged)"] = q_names
    references: dict[str, Sequence[str]] = {RULE: [RULE], SEASONAL: [SEASONAL]}
    table = gap_table(episodes, treatments, references)
    table.to_csv(args.out / "gaps.csv", index=False, lineterminator="\n", float_format="%.6g")
    parts = [
        f"# Predictive controller and comparators, `{config.label}`, history_rows={args.history_rows}",
        "",
        "Scenarios are identical for every arm. Gains are paired by scenario seed; learned models with several seeds "
        "are averaged within each scenario first; `sem` is across scenarios. `primary` is the registered definition of "
        "controllable (the safety filter raises a `fixed_20` rollout on fewer than 10% of the intervals); the other "
        "blocks are the registered sensitivity variants.",
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
