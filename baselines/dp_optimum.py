"""Exact hindsight optimum of the rebuild-bandwidth problem, by backward induction. NOT DEPLOYABLE.

For one scenario the foreground trace of every future interval, the group hazard at the start and the
remaining work are known to the planner, which then chooses the action of every interval to minimise the
episode's cost ``sum(alpha * D + beta * C + gamma * V)``. The state is ``(step, remaining work)``; the work
axis is a grid of ``grid_gb`` gigabytes with linear interpolation of the value between grid points. Every
per-interval quantity comes from the environment's own definitions: the risk floor from
``sim.safety_filter.apply_safety_filter`` on the reported hazard and the grid work, the delivered rate
``min(ACTIONS_MBPS[max(action, floor)], max(0, rho_max * BW_TOTAL_MBPS - fg))``, the utilisation
``min(rho_max, (fg + delivered) / BW_TOTAL_MBPS)``, the latency ``sim.config.latency_mean_ms``, and the
reward terms from ``contracts.contracts`` (``latency_degradation``, ``degraded_cost``,
``latency_violation``, ``compute_reward``).

What it bounds. It is the optimum for this one realised trace under the hazard at the start of the episode
held constant. Drive failures during the episode change the hazard (and can end the episode); the plan does
not anticipate them, so the executed reward equals the planned value only on episodes without a failure. It
is a reference no deployable controller can reach: it reads the future trace, the initial hazard and the
trace offset from the environment's private fields (as ``OracleLookahead`` does through
``peek_next_fg_mbps``) when the harness binds it.

Usage::

    python -m baselines.dp_optimum --workload-csv data_pipeline/profiles/profile_diurnal_long_holdout.csv \\
        --seeds 20 --eval-fleet --hazard-error-sigma 0.25 --out runs/dp_optimum
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from baselines.evaluate import (
    _build_config,
    compare,
    controllable_sensitivity,
    forced_masks,
    summarise,
    to_markdown,
)
from baselines.policies import OracleLookahead, RiskAwareRuleBased
from contracts.contracts import (
    ACTIONS_MBPS,
    Observation,
    BW_TOTAL_MBPS,
    CONTROL_INTERVAL_S,
    REBUILD_WORK_GB,
    compute_reward,
    degraded_cost,
    latency_degradation,
    latency_violation,
)
from sim.config import SimConfig, latency_mean_ms
from sim.recovery_env import _MB_PER_GB
from sim.safety_filter import apply_safety_filter

__all__ = ["DPSolution", "solve_hindsight", "DPHindsightOptimum", "main"]

#: Planning horizon in steps; the plan is checked to finish the rebuild inside it.
DEFAULT_HORIZON_STEPS: int = 1200
#: Horizon of the time-against-violation plan: the episode limit, so that slow violation-free plans are available.
FRONTIER_HORIZON_STEPS: int = 2400
DEFAULT_GRID_GB: float = 1.0
#: Terminal value of unfinished work, large enough that every plan completes inside the horizon.
_UNFINISHED_BASE: float = 1.0e4
#: Terminal value of unfinished work in the time-against-violation objective (prices up to 1e4 per violation).
_FRONTIER_UNFINISHED_BASE: float = 1.0e9
_FRONTIER_UNFINISHED_PER_GB: float = 1.0e5
_UNFINISHED_PER_GB: float = 0.1
_N = len(ACTIONS_MBPS)


class DPSolution:
    """Policy table ``action[step, work_index]`` and value ``value0[work_index]`` of the first step.

    Attributes:
        action: ``int8`` array ``(horizon, n_grid)``: the action index requested at step ``j`` (0-based, the
            ``j + 1``-th environment step) with remaining work ``j``-th grid point.
        value: Cost-to-go at step 0 for every grid point.
        grid_gb: Spacing of the work grid.
    """

    def __init__(self, action: np.ndarray, value: np.ndarray, grid_gb: float) -> None:
        self.action = action
        self.value = value
        self.grid_gb = grid_gb

    def planned_cost(self, work_gb: float) -> float:
        """Interpolated cost-to-go of the whole episode from ``work_gb`` at step 0."""
        grid = np.arange(len(self.value)) * self.grid_gb
        return float(np.interp(work_gb, grid, self.value))

    def action_index(self, step: int, work_gb: float) -> int:
        """Planned action for 0-based ``step`` at ``work_gb``, from the nearest grid point."""
        j = min(step, self.action.shape[0] - 1)
        w = min(max(int(round(work_gb / self.grid_gb)), 0), self.action.shape[1] - 1)
        return int(self.action[j, w])


def _floors(hazard: float, grid: np.ndarray, max_rate_when_infeasible: bool = True) -> np.ndarray:
    """Risk floor per grid point; with ``max_rate_when_infeasible`` False an infeasible budget has floor 0."""
    out = []
    for w in grid:
        decision = apply_safety_filter(0, hazard, float(w))
        out.append(decision.applied_index if (decision.any_feasible or max_rate_when_infeasible) else 0)
    return np.array(out, dtype=np.int64)


def _interval_costs(fg: float, config: SimConfig, degraded_weight: float = 1.0) -> np.ndarray:
    """Cost ``-compute_reward(...)`` of an interval for each delivered-action index ``0 .. N-1`` at load ``fg``.

    ``degraded_weight`` multiplies the degraded-state term (``SimConfig.risk_objective``: true hazard over the
    reference hazard); 1 is the contract reward.
    """
    available = max(0.0, config.rho_max * BW_TOTAL_MBPS - fg)
    costs = np.zeros(_N)
    for i in range(_N):
        delivered = min(float(ACTIONS_MBPS[i]), available)
        rho = min(config.rho_max, (fg + delivered) / BW_TOTAL_MBPS)
        latency = latency_mean_ms(rho, config)
        costs[i] = -compute_reward(
            latency_degradation(latency, config.latency_target_ms),
            degraded_cost(True) * degraded_weight,
            latency_violation(latency, config.latency_target_ms),
            config.reward,
        )
    return costs


def _violation_indicators(fg: float, config: SimConfig) -> np.ndarray:
    """``contracts.latency_violation`` of an interval for each delivered-action index at load ``fg``."""
    available = max(0.0, config.rho_max * BW_TOTAL_MBPS - fg)
    out = np.zeros(_N)
    for i in range(_N):
        delivered = min(float(ACTIONS_MBPS[i]), available)
        rho = min(config.rho_max, (fg + delivered) / BW_TOTAL_MBPS)
        out[i] = latency_violation(latency_mean_ms(rho, config), config.latency_target_ms)
    return out


def _delivered(fg: float, config: SimConfig) -> np.ndarray:
    available = max(0.0, config.rho_max * BW_TOTAL_MBPS - fg)
    return np.array([min(float(ACTIONS_MBPS[i]), available) for i in range(_N)])


def solve_hindsight(
    fg_future_mbps: Sequence[float],
    hazard_per_hour: float,
    config: SimConfig,
    *,
    grid_gb: float = DEFAULT_GRID_GB,
    work_max_gb: float = REBUILD_WORK_GB,
    degraded_weight: float = 1.0,
    violation_mu: float | None = None,
) -> DPSolution:
    """Backward induction over ``(step, work)``.

    ``degraded_weight`` is the factor on the degraded-state cost (``config.risk_objective``); the risk floor is the
    filter's, except that with ``config.infeasible_budget_max_rate`` False an infeasible budget has floor 0.

    With ``violation_mu`` the interval cost is instead ``1 + violation_mu * V`` (one unit of time per interval and
    a price on the violation indicator) -- the objective of the completion-time-against-violation frontier, which
    uses no reward coefficient and ignores ``degraded_weight``.

    ``fg_future_mbps[j]`` is the foreground load of the ``(j + 1)``-th environment step. The interval at
    which work reaches zero is charged in full, as the environment does.
    """
    horizon = len(fg_future_mbps)
    n_grid = int(round(work_max_gb / grid_gb)) + 1
    grid = np.arange(n_grid) * grid_gb
    floors = _floors(hazard_per_hour, grid, config.infeasible_budget_max_rate)
    base, per_gb = (_UNFINISHED_BASE, _UNFINISHED_PER_GB) if violation_mu is None else (_FRONTIER_UNFINISHED_BASE, _FRONTIER_UNFINISHED_PER_GB)
    value = np.where(grid > 0, base + grid * per_gb, 0.0)
    action = np.zeros((horizon, n_grid), dtype=np.int8)
    step_gb = CONTROL_INTERVAL_S / _MB_PER_GB
    for j in range(horizon - 1, -1, -1):
        fg = float(fg_future_mbps[j])
        if violation_mu is None:
            cost_by_idx = _interval_costs(fg, config, degraded_weight)
        else:
            cost_by_idx = 1.0 + float(violation_mu) * _violation_indicators(fg, config)
        delivered_by_idx = _delivered(fg, config)
        best = None
        arg = np.zeros(n_grid, dtype=np.int8)
        for a in range(_N):
            idx = np.maximum(a, floors)
            cost = cost_by_idx[idx]
            next_work = np.maximum(0.0, grid - delivered_by_idx[idx] * step_gb)
            q = cost + np.interp(next_work, grid, value)
            q = np.where(grid > 0, q, 0.0)
            if best is None:
                best = q
            else:
                better = q < best - 1e-12
                best = np.where(better, q, best)
                arg = np.where(better, a, arg).astype(np.int8)
        value, action[j] = best, arg
    return DPSolution(action, value, grid_gb)


class DPHindsightOptimum:
    """The hindsight optimum as a policy for the harness. NOT DEPLOYABLE: it reads the future trace.

    The harness binds the environment once; at the first call of each episode the policy reads the trace,
    the start offset and the reported hazard from the environment and solves :func:`solve_hindsight`.
    """

    deployable: bool = False

    def __init__(
        self,
        config: SimConfig | None = None,
        *,
        horizon_steps: int | None = None,
        grid_gb: float = DEFAULT_GRID_GB,
        violation_mu: float | None = None,
    ) -> None:
        self.config: SimConfig = config if config is not None else SimConfig()
        self.violation_mu = violation_mu
        self.name = (
            "DP_hindsight_optimum (not deployable)"
            if violation_mu is None
            else f"DP_frontier mu={violation_mu:.6g} (not deployable)"
        )
        self.horizon_steps = (
            (DEFAULT_HORIZON_STEPS if violation_mu is None else FRONTIER_HORIZON_STEPS)
            if horizon_steps is None
            else horizon_steps
        )
        self.grid_gb = grid_gb
        self._env: Any = None
        self._solution: DPSolution | None = None
        self._step = 0
        self.planned_cost: float | None = None

    def bind(self, env: Any) -> None:
        self._env = env

    def reset(self) -> None:
        self._solution = None
        self._step = 0
        self.planned_cost = None

    def _plan(self, work_gb: float) -> None:
        env = self._env
        if env is None or getattr(env, "_trace", None) is None:
            raise RuntimeError("DPHindsightOptimum is not bound to a reset environment; call bind(env)")
        trace = np.asarray(env._trace.fg_bw_mbps, dtype=np.float64)
        offset = int(env._offset)
        future = [trace[(offset + j + 1) % len(trace)] for j in range(self.horizon_steps)]
        weight = 1.0
        if self.config.risk_objective is not None:
            weight = float(env._true_group_hazard()) / self.config.risk_objective.hazard_ref_per_hour
        self._solution = solve_hindsight(
            future, float(env._hazard), self.config, grid_gb=self.grid_gb,
            degraded_weight=weight, violation_mu=self.violation_mu,
        )
        self.planned_cost = self._solution.planned_cost(work_gb)

    def __call__(self, obs: np.ndarray) -> int:
        work_gb = _observed_work(obs)
        if self._solution is None:
            self._plan(work_gb)
        assert self._solution is not None
        action = self._solution.action_index(self._step, work_gb)
        self._step += 1
        return action


def _observed_work(obs: np.ndarray) -> float:
    return Observation.from_vector(obs).work_remaining


def _policies(config: SimConfig) -> list[Any]:
    return [RiskAwareRuleBased(config), OracleLookahead(config), DPHindsightOptimum(config)]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the rule, ``OracleLookahead`` and the optimum on one replay CSV and write the comparison."""
    parser = argparse.ArgumentParser(prog="python -m baselines.dp_optimum", description=__doc__.split("\n\n")[0])
    parser.add_argument("--workload-csv", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--hazard-error-sigma", type=float, default=0.0)
    parser.add_argument("--eval-fleet", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.latency_filter = False
    args.latency_filter_causal = False
    config = _build_config(args)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    episodes = compare(_policies(config), profiles=[args.workload_csv], seeds=seeds, config=config)
    args.out.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(args.out / "episodes.csv", index=False)
    rule, oracle, dp = (p.name for p in _policies(config))
    parts = [f"# Hindsight optimum against the rule (simulated), `{config.label}`", "", to_markdown(summarise(episodes))]
    for treatment in (oracle, dp):
        parts += ["", f"## {treatment} minus {rule}, by definition of controllable", "",
                  to_markdown(controllable_sensitivity(episodes, treatment, rule))]
    block = episodes[episodes["policy"] == rule].set_index("seed")
    gap_rows = []
    for name, forced in forced_masks(block).items():
        keep = ~forced
        head = (episodes[episodes["policy"] == dp].set_index("seed")["total_reward"] - block["total_reward"])[keep]
        orc = (episodes[episodes["policy"] == oracle].set_index("seed")["total_reward"] - block["total_reward"])[keep]
        gap_rows.append({"definition": name, "n_controllable": int(keep.sum()), "dp_minus_rule": float(head.mean()),
                         "oracle_minus_rule": float(orc.mean()),
                         "oracle_share_of_optimum": float(orc.mean() / head.mean()) if head.mean() else math.nan})
    parts += ["", "## Share of the hindsight optimum captured by `ORACLE_lookahead`", "",
              to_markdown(pd.DataFrame(gap_rows))]
    (args.out / "summary.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    print("\n".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
