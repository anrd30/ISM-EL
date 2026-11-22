"""Verification script: the predictive controller on the original scenarios, with the history built by wrap-around.

This script reads the environment's private ``_offset`` and ``_trace`` to rebuild, for each seed, the two days of
load before the original start row, wrapping to the end of the file when the start row is below 576. It exists to
compare the controller with the earlier numbers measured on those scenarios (history_rows = 0, start rows
unchanged). The controller itself and the evaluation protocol use the contract's ``history`` field (see
``agent/evaluate_predictive.py``).

Usage::

    python agent/evidence/predictive-controller-2026-11-19/reproduce_prototype_scenarios.py <workload_csv> <out_csv> [excluded seeds]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent.predictive_controller import PredictiveController, RuleWithHistory  # noqa: E402
from baselines.dp_optimum import DPHindsightOptimum  # noqa: E402
from baselines.evaluate import _pinned_config, evaluate  # noqa: E402
from baselines.policies import RiskAwareRuleBased  # noqa: E402
from contracts.contracts import MAX_EPISODE_STEPS, PRE_FAILURE_HISTORY_ROWS  # noqa: E402
from sim.config import v1_eval_config  # noqa: E402
from sim.recovery_env import RecoveryEnv  # noqa: E402


def run_with_wrapped_history(config, policy, seed: int) -> float:
    env = RecoveryEnv(config)
    obs, _ = env.reset(seed=seed)
    trace = env._trace
    n = len(trace.fg_bw_mbps)
    rows = [(env._offset - PRE_FAILURE_HISTORY_ROWS + i) % n for i in range(PRE_FAILURE_HISTORY_ROWS)]
    policy.reset()
    policy.set_history(
        np.asarray(trace.fg_bw_mbps, dtype=float)[rows],
        np.asarray(trace.fg_read_iops, dtype=float)[rows],
        np.asarray(trace.fg_write_iops, dtype=float)[rows],
    )
    total = 0.0
    for _ in range(MAX_EPISODE_STEPS + 1):
        obs, reward, terminated, truncated, _ = env.step(int(policy(obs)))
        total += float(reward)
        if terminated or truncated:
            break
    return total


def main(argv: list[str]) -> int:
    workload = Path(argv[1])
    excluded = [int(x) for x in argv[3].split(",")] if len(argv) > 3 and argv[3] else []
    config, _ = _pinned_config(v1_eval_config(hazard_error_sigma=0.25), workload)
    seeds = list(range(20))
    rule = evaluate(RiskAwareRuleBased(config), profile=workload, seeds=seeds, config=v1_eval_config(hazard_error_sigma=0.25))
    dp = evaluate(DPHindsightOptimum(config), profile=workload, seeds=seeds, config=v1_eval_config(hazard_error_sigma=0.25))
    rows = []
    for seed in seeds:
        rows.append(
            {
                "seed": seed,
                "rule": float(rule.set_index("seed").loc[seed, "total_reward"]),
                "dp": float(dp.set_index("seed").loc[seed, "total_reward"]),
                "controller_wrapped_history": run_with_wrapped_history(config, PredictiveController(config), seed),
                "rule_wrapped_history": run_with_wrapped_history(config, RuleWithHistory(config), seed),
                "controller_no_history": run_with_wrapped_history(config, _NoHistory(config), seed),
            }
        )
    table = pd.DataFrame(rows).set_index("seed")
    table.to_csv(argv[2], lineterminator="\n", float_format="%.10g")
    keep = table.drop(index=excluded)
    head = (keep["dp"] - keep["rule"]).mean()
    for name in ("controller_wrapped_history", "rule_wrapped_history", "controller_no_history"):
        gap = keep[name] - keep["rule"]
        sem = gap.std(ddof=1) / np.sqrt(len(gap))
        print(f"{name}: n={len(gap)} gap vs rule {gap.mean():+.2f} +/- {sem:.2f} (t {gap.mean() / sem:.1f}), "
              f"better {int((gap > 0).sum())}/{len(gap)}, {100 * gap.mean() / head:.1f}% of the optimum ({head:.2f})")
    return 0


class _NoHistory(PredictiveController):
    def set_history(self, *args: object) -> None:  # the arm that starts empty
        return None


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
