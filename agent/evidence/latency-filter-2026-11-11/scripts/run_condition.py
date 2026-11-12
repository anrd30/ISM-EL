"""Paired evaluation of the rule, the hand-built seasonal rule and the forecast-PPO checkpoints in one condition.

Usage (from a tree that contains the T9 branch plus sim/latency_filter.py):

    python run_condition.py --condition home --latency-filter 0 --out OUT_DIR

Writes ``<out>/<condition>_filter<0|1|2>.csv`` (per-episode rows from ``baselines.evaluate.compare``).
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from agent.forecast_ppo import ForecastPPOPolicy
from agent.seasonal_rule import SeasonalForecastRule
from agent.train_residual import build_sim_config
from baselines.evaluate import compare
from baselines.policies import RiskAwareRuleBased

PROFILES = Path("data_pipeline/profiles")
HOLDOUT = PROFILES / "profile_diurnal_long_holdout.csv"
CKPT = Path("agent/evidence/forecast-ppo-2026-11-11/training/bc")
# name -> (workload csv, hazard_error_sigma, pinned fleet degradation or None)
CONDITIONS = {
    "home": (HOLDOUT, 0.25, None),
    "idle": (PROFILES / "profile_idle.csv", 0.25, None),
    "diurnal_generic": (PROFILES / "profile_diurnal.csv", 0.25, None),
    "bursty": (PROFILES / "profile_bursty.csv", 0.25, None),
    "peak": (PROFILES / "profile_peak.csv", 0.25, None),
    "sigma_0": (HOLDOUT, 0.0, None),
    "sigma_0.5": (HOLDOUT, 0.5, None),
    "degraded_fleet": (HOLDOUT, 0.25, 1.0),
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--condition", choices=list(CONDITIONS), required=True)
    p.add_argument("--latency-filter", type=int, choices=[0, 1, 2], required=True,
                   help="0 none, 1 ceiling_oracle_fg (interval's own load), 2 ceiling_causal_fg (previous interval's load)")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    csv, sigma, degradation = CONDITIONS[args.condition]
    config = build_sim_config("v1", True, sigma, latency_filter=args.latency_filter == 1,
                              latency_filter_causal=args.latency_filter == 2)
    if degradation is not None:
        config = dataclasses.replace(config, fleet_degradation=degradation)
    policies = [RiskAwareRuleBased(config), SeasonalForecastRule(config)]
    policies += [ForecastPPOPolicy.from_checkpoint(CKPT / f"seed{k}" / "ckpt_best.pt", config, f"forecast_ppo_bc seed{k}")
                 for k in range(3)]
    frame = compare(policies, profiles=[csv], seeds=list(range(args.seeds)), config=config)
    frame.insert(0, "condition", args.condition)
    frame.insert(1, "latency_filter", args.latency_filter)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / f"{args.condition}_filter{args.latency_filter}.csv", index=False)
    print(args.condition, args.latency_filter, "rows", len(frame))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
