"""Completion time against latency violations, with no reward coefficient.

Per scenario three frontiers are traced:

- ``dp``: the hindsight optimum of ``baselines.dp_optimum`` with the interval cost ``1 + mu * V`` (one unit of time and
  a price ``mu`` on the violation indicator ``V``), over a log grid of ``mu``; each point is the executed
  (completion hours, latency violation fraction).
- ``rule`` and ``controller``: ``risk_aware_rule_based`` and ``predictive_controller (history)`` over a log grid of
  their own ``progress_value``.

The achievable curve of a set of points is ``H(v) = min`` completion hours over the points with violation fraction at
most ``v``. The constrained optimum is ``H(LATENCY_VIOLATION_FRACTION_MAX)``. A Lagrangian sweep finds only points on the
convex hull of the achievable set; for discrete actions a non-hull point can be missed, so the ``dp`` curve is an upper
bound on the best achievable completion time at a given violation fraction, within the resolution of the ``mu`` grid.
Episodes that do not complete have no completion time and are left out of the points.

Usage::

    python -m agent.frontier run --workload-csv <csv> --label A --kind dp|rule|controller --out <dir> [--worker i --workers n]
    python -m agent.frontier analyse --dir <dir> --out <dir>
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from agent.predictive_controller import PredictiveController
from baselines.dp_optimum import DPHindsightOptimum
from baselines.evaluate import _build_config, evaluate, forced_masks
from baselines.policies import RiskAwareRuleBased
from contracts.contracts import LATENCY_VIOLATION_FRACTION_MAX, PRE_FAILURE_HISTORY_ROWS

__all__ = ["MUS", "PROGRESS_VALUES", "achievable", "analyse_points", "main"]

MUS: np.ndarray = np.logspace(-2.0, 4.0, 31)
PROGRESS_VALUES: np.ndarray = np.logspace(-4.0, -1.0, 12)
V_GRID: np.ndarray = np.linspace(0.0, 0.7, 701)
KINDS = ("dp", "rule", "controller")


def _make(kind: str, config, param: float):
    if kind == "dp":
        return DPHindsightOptimum(config, violation_mu=float(param))
    if kind == "rule":
        return RiskAwareRuleBased(config, progress_value=float(param))
    return PredictiveController(config, progress_value=float(param))


def _params(kind: str) -> np.ndarray:
    return MUS if kind == "dp" else PROGRESS_VALUES


def achievable(violation: np.ndarray, hours: np.ndarray, grid: np.ndarray = V_GRID) -> np.ndarray:
    """``H(v)`` on ``grid``: the least completion hours among points with violation fraction at most ``v`` (inf if none)."""
    out = np.full(len(grid), np.inf)
    for v, h in zip(violation, hours):
        if math.isfinite(h):
            out = np.where(grid >= v - 1e-12, np.minimum(out, h), out)
    return out


_trapz = getattr(np, "trapezoid", None) or np.trapz


def _area(curve: np.ndarray, cap: float, grid: np.ndarray = V_GRID) -> float:
    return float(_trapz(np.where(np.isfinite(curve), curve, cap), grid))


def analyse_points(points: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-scenario results and per-population summary from the points of all kinds.

    ``points`` has columns ``population, kind, param, seed, hours, violation, filter_forced``. The scenario table has,
    per population and seed, ``H_dp``, ``H_controller``, ``H_rule`` at the violation limit, the areas of each curve,
    ``area_*_minus_dp`` and whether the controller's curve is at most the rule's everywhere on the grid (``ctrl_dominates``).
    """
    rows = []
    for (pop, seed), g in points.groupby(["population", "seed"], sort=True):
        curves = {k: achievable(h["violation"].to_numpy(), h["hours"].to_numpy()) for k, h in g.groupby("kind")}
        cap = float(np.nanmax(g["hours"].to_numpy()))
        limit = int(round(LATENCY_VIOLATION_FRACTION_MAX / (V_GRID[1] - V_GRID[0])))
        row = {"population": pop, "seed": int(seed), "filter_forced": bool(g["filter_forced"].iloc[0])}
        for k in KINDS:
            row[f"H_{k}"] = float(curves[k][limit])
            row[f"area_{k}"] = _area(curves[k], cap)
        for k in ("controller", "rule"):
            row[f"area_{k}_minus_dp"] = row[f"area_{k}"] - row["area_dp"]
        better = curves["controller"] <= curves["rule"] + 1e-9
        row["ctrl_dominates"] = bool(better.all() and (curves["controller"] < curves["rule"] - 1e-9).any())
        row["ctrl_weakly_dominates"] = bool(better.all())
        rows.append(row)
    scen = pd.DataFrame(rows)
    out = []
    for pop in scen["population"].unique():
        block = scen[scen["population"] == pop]
        for cls, keep in (("primary", block[~block["filter_forced"]]), ("all", block)):
            d = {"population": pop, "class": cls, "n": len(keep)}
            for k in KINDS:
                h = keep[f"H_{k}"].replace([np.inf], np.nan)
                d[f"hours_at_limit_{k}"] = float(h.mean())
                d[f"n_feasible_{k}"] = int(h.notna().sum())
            gap = (keep["H_rule"] - keep["H_controller"]).replace([np.inf, -np.inf], np.nan).dropna()
            d["rule_minus_controller_hours"] = float(gap.mean()) if len(gap) else math.nan
            d["rule_minus_controller_sem"] = float(gap.std(ddof=1) / math.sqrt(len(gap))) if len(gap) > 1 else math.nan
            d["area_controller_minus_dp"] = float(keep["area_controller_minus_dp"].mean())
            d["area_rule_minus_dp"] = float(keep["area_rule_minus_dp"].mean())
            d["share_ctrl_dominates"] = float(keep["ctrl_dominates"].mean())
            d["share_ctrl_weakly_dominates"] = float(keep["ctrl_weakly_dominates"].mean())
            out.append(d)
    return scen, pd.DataFrame(out)


def _plot(points: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pops = list(points["population"].unique())
    fig, axes = plt.subplots(1, len(pops), figsize=(4.6 * len(pops), 3.8), squeeze=False)
    colors = {"dp": "#1b6ca8", "controller": "#c0392b", "rule": "#555555"}
    labels = {"dp": "hindsight optimum", "controller": "predictive controller", "rule": "rule"}
    for ax, pop in zip(axes[0], pops):
        g = points[(points["population"] == pop) & (~points["filter_forced"])]
        for kind in KINDS:
            curves = [achievable(h["violation"].to_numpy(), h["hours"].to_numpy())
                      for _, h in g[g["kind"] == kind].groupby("seed")]
            cap = float(np.nanmax(g["hours"].to_numpy()))
            mean_curve = np.mean([np.where(np.isfinite(c), c, cap) for c in curves], axis=0)
            ax.plot(V_GRID, mean_curve, color=colors[kind], label=labels[kind], lw=1.8)
            m = g[g["kind"] == kind].groupby("param")[["violation", "hours"]].mean()
            ax.scatter(m["violation"], m["hours"], s=10, color=colors[kind], alpha=0.5)
        ax.axvline(LATENCY_VIOLATION_FRACTION_MAX, color="k", lw=0.8, ls="--")
        ax.set_title(f"population {pop} (controllable scenarios)")
        ax.set_xlabel("latency violation fraction")
        ax.set_xlim(0, 0.3)
    axes[0][0].set_ylabel("completion hours (mean over scenarios)")
    axes[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.frontier", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--workload-csv", type=Path, required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--kind", choices=KINDS, required=True)
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
        config = dataclasses.replace(_build_config(ns), history_rows=PRE_FAILURE_HISTORY_ROWS)
        args.out.mkdir(parents=True, exist_ok=True)
        for i, param in enumerate(_params(args.kind)):
            if i % args.workers != args.worker:
                continue
            path = args.out / f"points_{args.label}_{args.kind}_{i:02d}.csv"
            if path.exists():
                continue
            frame = evaluate(_make(args.kind, config, param), profile=args.workload_csv, seeds=list(range(args.seeds)), config=config)
            pd.DataFrame({"population": args.label, "kind": args.kind, "param": param, "seed": frame["seed"],
                          "hours": frame["completion_hours"], "violation": frame["latency_violation_fraction"],
                          "filter_forced": frame["filter_forced"]}).to_csv(path, index=False, lineterminator="\n", float_format="%.10g")
            print(f"done {args.label} {args.kind} {i}", flush=True)
        return 0
    points = pd.concat([pd.read_csv(p) for p in sorted(args.dir.glob("points_*.csv"))], ignore_index=True)
    scen, summary = analyse_points(points)
    args.out.mkdir(parents=True, exist_ok=True)
    scen.to_csv(args.out / "frontier_by_scenario.csv", index=False, lineterminator="\n", float_format="%.6g")
    summary.to_csv(args.out / "frontier_summary.csv", index=False, lineterminator="\n", float_format="%.6g")
    _plot(points, summary, args.out / "frontiers.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
