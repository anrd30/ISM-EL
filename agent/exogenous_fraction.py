"""Post-hoc share of latency violations that occur even at the smallest action.

A step's ``violated_latency`` is decided by ``latency_violation(latency_mean_ms(rho, cfg),
cfg.latency_target_ms)`` with ``rho = min(rho_max, (fg + effective) / BW_TOTAL_MBPS)`` and
``effective = min(applied, max(0, rho_max * BW_TOTAL_MBPS - fg))`` (``sim/recovery_env.py``
``step``, ``_rho``, ``_headroom``). For a fixed delivered allocation the decision is a function of the
interval's foreground load ``fg`` and ``cfg`` only. The delivered allocation of an episode
also depends on the reliability filter (``sim.safety_filter``), which raises a requested
action using the hazard and the remaining work; those depend on the episode's earlier
deliveries and are not in ``episodes.csv``. ``min_action_violates`` does not include that
raise.

An interval is *exogenous* when the smallest action in ``ACTIONS_MBPS`` violates at that
interval's foreground load. ``min_action_violates`` evaluates this with the same
``latency_mean_ms`` path as ``sim.latency_filter``.

Episode window. An episode that started at ``trace_start_row`` and ran ``steps`` intervals
read trace rows ``(trace_start_row + t) % n_rows`` for ``t = 1..steps`` (``RecoveryEnv.step``).
Both columns are present in the ``episodes.csv`` of ``baselines.evaluate`` and
``agent.evaluate_policy``; the trace is rebuilt from the workload CSV with
``sim.scenario.trace_from_frame``.

Per ``(policy, seed)``: ``exogenous_fraction`` is the share of window intervals where the
smallest action violates, ``policy_violation_fraction`` is the recorded
``latency_violation_fraction`` and ``marginal_fraction`` is their difference.

Usage::

    python -m agent.exogenous_fraction --eval-dir DIR --workload-csv CSV --out DIR
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from contracts.contracts import ACTIONS_MBPS, BW_TOTAL_MBPS, latency_violation
from sim.config import SimConfig, latency_mean_ms

__all__ = ["min_action_violates", "exogenous_fraction", "episode_window_loads", "analyse", "main"]

REFERENCE_POLICY = "fixed_20"


def min_action_violates(foreground_load: float, cfg: SimConfig) -> bool:
    """Whether the smallest action violates the latency limit at ``foreground_load`` (MB/s).

    The delivered allocation is ``min(ACTIONS_MBPS[0], headroom)`` with headroom
    ``max(0, rho_max * BW_TOTAL_MBPS - foreground_load)``, as in ``RecoveryEnv.step``.

    Raises:
        ValueError: if ``foreground_load`` is negative or not finite.
    """
    fg = float(foreground_load)
    if not (fg >= 0.0 and fg < float("inf")):
        raise ValueError(f"foreground_load must be finite and >= 0, got {foreground_load!r}")
    headroom = max(0.0, cfg.rho_max * BW_TOTAL_MBPS - fg)
    effective = min(float(ACTIONS_MBPS[0]), headroom)
    rho = min(cfg.rho_max, (fg + effective) / BW_TOTAL_MBPS)
    return latency_violation(latency_mean_ms(rho, cfg), cfg.latency_target_ms) == 1.0


def exogenous_fraction(loads: np.ndarray, cfg: SimConfig) -> float:
    """Share of the given per-interval foreground loads at which the smallest action violates."""
    if len(loads) == 0:
        return float("nan")
    return float(np.mean([min_action_violates(x, cfg) for x in loads]))


def episode_window_loads(fg_bw_mbps: np.ndarray, trace_start_row: int, steps: int) -> np.ndarray:
    """Foreground loads of intervals ``1..steps`` of an episode starting at ``trace_start_row``."""
    n_rows = len(fg_bw_mbps)
    rows = (int(trace_start_row) + np.arange(1, int(steps) + 1)) % n_rows
    return fg_bw_mbps[rows]


def analyse(eval_dir: Path, workload_csv: Path, cfg: SimConfig | None = None) -> pd.DataFrame:
    """Return one row per ``(policy, seed)`` of ``eval_dir/episodes.csv``.

    Raises:
        KeyError: if ``episodes.csv`` lacks ``trace_start_row`` or ``steps``.
    """
    from sim.scenario import trace_from_frame

    cfg = cfg or SimConfig()
    episodes = pd.read_csv(eval_dir / "episodes.csv")
    missing = {"trace_start_row", "steps", "latency_violation_fraction", "policy", "seed"} - set(episodes.columns)
    if missing:
        raise KeyError(f"{eval_dir / 'episodes.csv'} lacks columns {sorted(missing)}")
    trace = trace_from_frame(pd.read_csv(workload_csv), profile=workload_csv.stem, synthetic=None)
    rows = []
    for ep in episodes.itertuples(index=False):
        loads = episode_window_loads(trace.fg_bw_mbps, ep.trace_start_row, ep.steps)
        exo = exogenous_fraction(loads, cfg)
        own = float(ep.latency_violation_fraction)
        rows.append({"policy": ep.policy, "seed": int(ep.seed), "steps": int(ep.steps),
                     "exogenous_fraction": exo, "policy_violation_fraction": own, "marginal_fraction": own - exo})
    return pd.DataFrame(rows)


def _summary(result: pd.DataFrame, eval_dir: Path, workload_csv: Path, gate: float | None) -> str:
    means = result.groupby("policy", sort=False)[
        ["exogenous_fraction", "policy_violation_fraction", "marginal_fraction"]
    ].mean()
    lines = [
        "# Exogenous violation fraction",
        "",
        f"- evaluation directory: {eval_dir.as_posix()}",
        f"- workload: {workload_csv.as_posix()}",
        f"- {REFERENCE_POLICY} validation, max |exogenous_fraction - latency_violation_fraction|: "
        + ("n/a (policy absent)" if gate is None else f"{gate:.3e}"),
        "",
        "| policy | exogenous_fraction | policy_violation_fraction | marginal_fraction |",
        "|---|---|---|---|",
    ]
    for name, row in means.iterrows():
        lines.append(f"| {name} | {row.exogenous_fraction:.6f} | {row.policy_violation_fraction:.6f} "
                     f"| {row.marginal_fraction:.6f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m agent.exogenous_fraction", description=__doc__.split("\n\n")[0])
    p.add_argument("--eval-dir", type=Path, required=True, help="directory holding episodes.csv")
    p.add_argument("--workload-csv", type=Path, required=True, help="workload CSV the episodes were evaluated on")
    p.add_argument("--out", type=Path, required=True, help="output directory")
    args = p.parse_args(argv)

    result = analyse(args.eval_dir, args.workload_csv)
    ref = result[result["policy"] == REFERENCE_POLICY]
    gate = None if ref.empty else float((ref["exogenous_fraction"] - ref["policy_violation_fraction"]).abs().max())
    args.out.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out / "exogenous_fraction.csv", index=False, lineterminator="\n")
    (args.out / "summary.md").write_text(_summary(result, args.eval_dir, args.workload_csv, gate),
                                         encoding="utf-8", newline="\n")
    print(f"{REFERENCE_POLICY}_max_abs_diff={'n/a' if gate is None else f'{gate:.3e}'} rows={len(result)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
