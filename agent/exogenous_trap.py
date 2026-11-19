"""The exogenous share of latency violations: direct measurement of X and the checks of the cost-trap analysis.

Definitions, per episode. ``C`` is the latency violation fraction. ``X`` is the fraction of intervals in which the
smallest action the safety filter allows would also have violated latency (``exogenous_violations`` of
``baselines.evaluate`` divided by the number of steps). ``d`` is ``LATENCY_VIOLATION_FRACTION_MAX``. ``f = X / C`` is
the exogenous share of the violations. When the smallest allowed action is the applied one (``fixed_20``) every
violation is exogenous and ``f = 1``; a more aggressive policy adds violations the smallest action would not cause,
and ``f`` falls.

``x_summary`` and ``carriers`` report ``X`` against ``d`` overall and per scenario; ``eq3_check`` tests ``f`` across
the fixed rates; ``training_x`` reads the per-iteration counters ``real_violations`` and ``min_action_violations`` of
``agent.train --marginal-violation-cost`` (``l3_stats.csv``), where ``min_action_violations`` counts the intervals the
smallest action would also have violated, on the policy's own trajectory.

``python -m agent.exogenous_trap compare`` evaluates PPO checkpoints (with and without the marginal-cost dual update)
against the rule, the predictive controller, the look-ahead oracle and the hindsight optimum on one replay profile.

Usage::

    python -m agent.exogenous_trap tables --episodes <episodes.csv> [...] --out <dir> [--l3-stats <l3_stats.csv> ...]
    python -m agent.exogenous_trap compare --workload-csv <csv> --ppo-label <label>=<ckpt> ... --out <dir>
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from contracts.contracts import LATENCY_VIOLATION_FRACTION_MAX

__all__ = ["x_table", "x_summary", "carriers", "eq3_check", "training_x", "main"]

D = LATENCY_VIOLATION_FRACTION_MAX
FIXED_ORDER = ("fixed_20", "fixed_50", "fixed_80", "fixed_120")


def x_table(episodes: pd.DataFrame) -> pd.DataFrame:
    """Per profile, policy and scenario: ``X``, ``C``, ``f = X / C`` (NaN when ``C`` is 0) and whether ``X > d``."""
    out = episodes[["profile", "policy", "seed", "steps", "exogenous_violations", "latency_violation_fraction"]].copy()
    out["X"] = out["exogenous_violations"] / out["steps"]
    out["C"] = out["latency_violation_fraction"]
    out["f"] = np.where(out["C"] > 0, out["X"] / out["C"].where(out["C"] > 0), np.nan)
    out["X_above_d"] = out["X"] > D
    return out[["profile", "policy", "seed", "steps", "X", "C", "f", "X_above_d"]]


def x_summary(episodes: pd.DataFrame) -> pd.DataFrame:
    """Per profile and policy: mean ``X`` and ``C`` over scenarios, ``f`` as the ratio of the means, and the scenarios with ``X > d``."""
    table = x_table(episodes)
    rows = []
    for (profile, policy), g in table.groupby(["profile", "policy"], sort=False):
        mx, mc = float(g["X"].mean()), float(g["C"].mean())
        rows.append({"profile": profile, "policy": policy, "n": len(g), "X_mean": mx, "C_mean": mc,
                     "f_ratio_of_means": mx / mc if mc > 0 else math.nan, "d": D,
                     "X_mean_above_d": mx > D, "n_scenarios_X_above_d": int(g["X_above_d"].sum()),
                     "scenarios_X_above_d": " ".join(str(int(s)) for s in g.loc[g["X_above_d"], "seed"])})
    return pd.DataFrame(rows)


def carriers(episodes: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Scenarios of one policy ordered by ``X`` (largest first) with their share of the summed ``X``."""
    g = x_table(episodes)
    g = g[g["policy"] == policy].sort_values("X", ascending=False).reset_index(drop=True)
    g["share_of_total_X"] = g["X"] / g["X"].sum() if g["X"].sum() > 0 else np.nan
    return g


def eq3_check(episodes: pd.DataFrame) -> pd.DataFrame:
    """``f`` of ``fixed_20`` (must be 1 wherever ``C > 0``), then of the more aggressive fixed rates (must fall).

    One row per profile: ``f_by_rate`` for each fixed rate as the ratio of the mean ``X`` to the mean ``C``,
    ``f20_is_one`` (every scenario with ``C > 0`` has ``f = 1`` to 1e-12), ``f_non_increasing`` and
    ``f_decreasing`` (non-increasing and strictly decreasing over the rates present, in the order 20, 50, 80, 120).
    A rate that adds no violation beyond those of the smallest action keeps ``f = 1``, so the first holds where the
    second does not.
    """
    table = x_table(episodes)
    rows = []
    for profile, g in table.groupby("profile", sort=False):
        f = {}
        for name in FIXED_ORDER:
            h = g[g["policy"] == name]
            if len(h):
                f[name] = float(h["X"].mean() / h["C"].mean()) if h["C"].mean() > 0 else math.nan
        f20 = g[(g["policy"] == "fixed_20") & (g["C"] > 0)]["f"]
        values = [f[k] for k in FIXED_ORDER if k in f and not math.isnan(f[k])]
        rows.append({"profile": profile, **{f"f_{k}": v for k, v in f.items()},
                     "f20_is_one": bool(len(f20) > 0 and np.allclose(f20, 1.0, atol=1e-12)),
                     "f_non_increasing": bool(all(a >= b - 1e-12 for a, b in zip(values, values[1:]))),
                     "f_decreasing": bool(all(a > b for a, b in zip(values, values[1:])))})
    return pd.DataFrame(rows)


def training_x(stats_files: Sequence[Path]) -> pd.DataFrame:
    """``X``, ``C`` and ``f`` from ``l3_stats.csv`` files: per file and pooled, plus the share of iterations with ``X > d``.

    ``X`` is ``min_action_violations / n_steps`` and ``C`` is ``real_violations / n_steps`` (sums over iterations);
    ``exogenous`` is ``real_violations - marginal_cost_steps`` and equals ``min_action_violations``.
    """
    frames = [pd.read_csv(p).assign(source=Path(p).parent.name) for p in stats_files]
    rows = []
    for name, g in [(f.iloc[0]["source"], f) for f in frames] + [("pooled", pd.concat(frames))]:
        steps = float(g["n_steps"].sum())
        iteration_x = g["min_action_violations"] / g["n_steps"]
        rows.append({"source": name, "iterations": len(g), "steps": int(steps),
                     "real_violations": int(g["real_violations"].sum()),
                     "min_action_violations": int(g["min_action_violations"].sum()),
                     "marginal_cost_steps": int(g["marginal_cost_steps"].sum()),
                     "exogenous_real_minus_marginal": int((g["real_violations"] - g["marginal_cost_steps"]).sum()),
                     "X": float(g["min_action_violations"].sum()) / steps, "C": float(g["real_violations"].sum()) / steps,
                     "f": float(g["min_action_violations"].sum()) / float(g["real_violations"].sum()),
                     "d": D, "share_of_iterations_X_above_d": float((iteration_x > D).mean())})
    return pd.DataFrame(rows)


def _compare(args: argparse.Namespace) -> int:
    import dataclasses

    from agent.evaluate_predictive import CONTROLLER, OPTIMUM, ORACLE, RULE, gap_table
    from agent.policy_adapter import LearnedPolicy
    from agent.predictive_controller import PredictiveController
    from baselines.dp_optimum import DPHindsightOptimum
    from baselines.evaluate import _build_config, compare, to_markdown
    from baselines.policies import OracleLookahead, RiskAwareRuleBased
    from contracts.contracts import PRE_FAILURE_HISTORY_ROWS

    args.latency_filter = False
    args.latency_filter_causal = False
    config = dataclasses.replace(_build_config(args), history_rows=PRE_FAILURE_HISTORY_ROWS)
    arms = [RiskAwareRuleBased(config), PredictiveController(config, name=CONTROLLER), OracleLookahead(config),
            DPHindsightOptimum(config)]
    treatments: dict[str, list[str]] = {CONTROLLER: [CONTROLLER], ORACLE: [ORACLE], OPTIMUM: [OPTIMUM]}
    for item in args.ppo_label:
        label, _, path = item.partition("=")
        for mode, det in (("deterministic", True), ("sampled", False)):
            names = []
            for i, ckpt in enumerate(sorted(Path(path).glob("seed*/ckpt_final.pt"))):
                name = f"{label} {mode} {ckpt.parent.name}"
                arms.append(LearnedPolicy.from_checkpoint(ckpt, name=name, deterministic=det))
                names.append(name)
            treatments[f"{label} ({mode}, 3 seeds averaged)"] = names
    episodes = compare(arms, profiles=[args.workload_csv], seeds=list(range(args.seeds)), config=config)
    args.out.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(args.out / "episodes.csv", index=False, lineterminator="\n")
    table = gap_table(episodes, treatments, {RULE: [RULE]})
    table.to_csv(args.out / "gaps.csv", index=False, lineterminator="\n", float_format="%.6g")
    x_summary(episodes).to_csv(args.out / "x_summary.csv", index=False, lineterminator="\n", float_format="%.6g")
    (args.out / "summary.md").write_text(
        f"# PPO with and without the marginal-cost dual update, against the predictive controller and the optimum, `{config.label}`\n\n"
        + to_markdown(table) + "\n", encoding="utf-8")
    print(to_markdown(table))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.exogenous_trap", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("tables")
    t.add_argument("--episodes", type=Path, nargs="+", required=True)
    t.add_argument("--l3-stats", type=Path, nargs="*", default=[])
    t.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compare")
    c.add_argument("--workload-csv", type=Path, required=True)
    c.add_argument("--ppo-label", nargs="+", required=True, help="label=run_dir (containing seed*/ckpt_final.pt)")
    c.add_argument("--seeds", type=int, default=20)
    c.add_argument("--hazard-error-sigma", type=float, default=0.25)
    c.add_argument("--eval-fleet", action="store_true")
    c.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "compare":
        return _compare(args)
    episodes = pd.concat([pd.read_csv(p) for p in args.episodes], ignore_index=True)
    args.out.mkdir(parents=True, exist_ok=True)
    x_summary(episodes).to_csv(args.out / "x_summary.csv", index=False, lineterminator="\n", float_format="%.6g")
    x_table(episodes).to_csv(args.out / "x_by_scenario.csv", index=False, lineterminator="\n", float_format="%.6g")
    eq3_check(episodes).to_csv(args.out / "eq3_check.csv", index=False, lineterminator="\n", float_format="%.6g")
    if args.l3_stats:
        training_x(args.l3_stats).to_csv(args.out / "training_x.csv", index=False, lineterminator="\n", float_format="%.8g")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
