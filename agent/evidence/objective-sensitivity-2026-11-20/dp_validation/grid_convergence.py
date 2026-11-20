"""Grid convergence of the hindsight optimum: planned and executed cost at work-grid spacings 2, 1 and 0.5 GB.

Usage: python agent/evidence/objective-sensitivity-2026-11-20/dp_validation/grid_convergence.py <out.csv>
Runs on seeds 0 to 4 of each population (A holdout, B altB, C popC, D popD), evaluation fleet, sigma 0.25, history rows 576.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from baselines.dp_optimum import DPHindsightOptimum  # noqa: E402
from baselines.evaluate import _pinned_config  # noqa: E402
from contracts.contracts import PRE_FAILURE_HISTORY_ROWS  # noqa: E402
from sim.config import v1_eval_config  # noqa: E402
from sim.recovery_env import RecoveryEnv  # noqa: E402

POPS = {"A": "profile_diurnal_long_holdout.csv", "B": "profile_diurnal_long_altB.csv",
        "C": "profile_diurnal_long_popC.csv", "D": "profile_diurnal_long_popD.csv"}


def run(config, seed: int, grid_gb: float) -> tuple[float, float]:
    env = RecoveryEnv(config)
    policy = DPHindsightOptimum(config, grid_gb=grid_gb)
    policy.bind(env)
    obs, _ = env.reset(seed=seed)
    policy.reset()
    total = 0.0
    while True:
        obs, reward, terminated, truncated, _ = env.step(policy(obs))
        total += reward
        if terminated or truncated:
            break
    return float(policy.planned_cost), -total


def main(out: str) -> None:
    base = dataclasses.replace(v1_eval_config(hazard_error_sigma=0.25), history_rows=PRE_FAILURE_HISTORY_ROWS)
    rows = []
    for pop, name in POPS.items():
        config, _ = _pinned_config(base, Path("data_pipeline/profiles") / name)
        for seed in range(5):
            for grid in (2.0, 1.0, 0.5):
                planned, executed = run(config, seed, grid)
                rows.append({"population": pop, "seed": seed, "grid_gb": grid, "planned_cost": planned, "executed_cost": executed})
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(out, index=False, lineterminator="\n", float_format="%.8f")


if __name__ == "__main__":
    main(sys.argv[1])
