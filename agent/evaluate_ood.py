"""Out-of-distribution evaluation of the forecast PPO agent against the rule.

Usage::

    python -m agent.evaluate_ood \\
        --ppo-checkpoint agent/evidence/forecast-ppo-2026-11-11/training/bc/seed{0,1,2}/ckpt_best.pt \\
        --holdout-csv data_pipeline/profiles/profile_diurnal_long_holdout.csv \\
        --seeds 20 --out runs/eval_ood

The agent was trained on ``profile_diurnal_long``, a real trace selected for
its daily cycle, with ``hazard_error_sigma=0.25`` and randomly degraded fleets.
Each condition below moves one of those away from training. Every condition
runs ``risk_aware_rule_based``, the oracle, the hand-built
``SeasonalForecastRule`` and the agent on identical paired scenarios through
``baselines.evaluate.compare``.

Conditions:

- ``synthetic_<profile>_<n>day``, for each profile of ``workload.generate``:
  ``days`` consecutive synthetic days generated with different seeds and
  concatenated. The committed 24-hour profiles are not used. An episode lasts
  about 44 hours and the simulator loops a trace past its end, so on a
  one-day trace the load one day earlier is an exact copy of the current day
  and any same-time-yesterday forecast is perfect by construction.
  Concatenating independently generated days keeps the profile's daily shape
  and makes its noise and bursts differ from day to day.
- ``holdout_phase_shift``: the holdout trace with day ``d`` shifted by
  ``d * PHASE_SHIFT_STEPS`` intervals, so that each day runs later than the day
  before and the same time one day earlier is a different time of day.
- ``holdout_sigma_<x>``: the holdout with a different ``hazard_error_sigma``.
- ``holdout_degraded_fleet``: the holdout with ``fleet_degradation`` pinned at
  ``DEGRADED_LEVEL``.

All conditions use the held-out fleet table. The risk model is wrapped in
:class:`agent.search_seasonal.CachedHazard`, which leaves results unchanged.
Writes ``episodes.csv``, ``gaps.csv`` and ``summary.md``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from agent.forecast_ppo import ForecastPPOPolicy
from agent.residual_features import DAY_STEPS
from agent.search_seasonal import CachedHazard
from agent.seasonal_rule import SeasonalForecastRule
from agent.train_residual import build_sim_config
from baselines.evaluate import compare, to_markdown
from baselines.policies import OracleLookahead, RiskAwareRuleBased
from sim.config import SimConfig
from workload.generate import PROFILE_NAMES, generate_profile

__all__ = ["PHASE_SHIFT_STEPS", "DEGRADED_LEVEL", "write_synthetic", "write_phase_shift", "main"]

#: Shift added per day in the phase-shift condition: six hours.
PHASE_SHIFT_STEPS: int = 72
#: Pinned fleet degradation level of the degraded-fleet condition.
DEGRADED_LEVEL: float = 0.9
#: First generator seed of the synthetic days; disjoint from the generator seeds the simulator draws per episode.
_SYNTHETIC_SEED_BASE: int = 900_000
REFERENCE: str = "risk_aware_rule_based"


def _write_csv(frame: pd.DataFrame, path: Path, profile: str, synthetic: bool, note: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    meta = {"profile": profile, "synthetic": synthetic, "rows": len(frame), "label": note}
    path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def write_synthetic(name: str, days: int, directory: Path) -> Path:
    """Write ``days`` independently generated days of profile ``name`` as one replay CSV."""
    frames = [generate_profile(name, _SYNTHETIC_SEED_BASE + 1000 * PROFILE_NAMES.index(name) + d) for d in range(days)]
    frame = pd.concat(frames, ignore_index=True)
    frame["t_seconds"] = np.arange(len(frame), dtype=np.int64) * int(frame["t_seconds"].iloc[1])
    label = f"synthetic_{name}_{days}day"
    note = f"{days} synthetic days of '{name}', generator seeds differ per day"
    return _write_csv(frame, directory / f"{label}.csv", label, True, note)


def write_phase_shift(csv_path: Path, directory: Path) -> Path:
    """Write ``csv_path`` with day ``d`` rotated by ``d * PHASE_SHIFT_STEPS`` intervals."""
    frame = pd.read_csv(csv_path)
    values = frame.drop(columns=["t_seconds"]).to_numpy()
    n_days = len(frame) // DAY_STEPS
    parts = [np.roll(values[d * DAY_STEPS : (d + 1) * DAY_STEPS], -d * PHASE_SHIFT_STEPS, axis=0)
             for d in range(n_days)]
    if len(frame) % DAY_STEPS:
        parts.append(values[n_days * DAY_STEPS :])
    shifted = pd.DataFrame(np.concatenate(parts), columns=frame.columns[1:])
    shifted.insert(0, "t_seconds", frame["t_seconds"].to_numpy())
    note = f"holdout with day d shifted {PHASE_SHIFT_STEPS} intervals per day"
    return _write_csv(shifted.astype(frame.dtypes.to_dict()), directory / "holdout_phase_shift.csv",
                      "holdout_phase_shift", False, note)


def _policies(config: SimConfig, checkpoints: Sequence[Path]) -> list:
    out = [RiskAwareRuleBased(config), OracleLookahead(config), SeasonalForecastRule(config)]
    for path in checkpoints:
        seed = int(torch.load(path, map_location="cpu", weights_only=True)["meta"]["seed"])
        out.append(ForecastPPOPolicy.from_checkpoint(path, config, f"forecast_ppo_bc seed{seed}"))
    return out


def _gaps(df: pd.DataFrame, condition: str) -> list[dict[str, object]]:
    ref = df[df["policy"] == REFERENCE].set_index("seed")
    rows = []
    for name in df["policy"].unique():
        if name == REFERENCE:
            continue
        own = df[df["policy"] == name].set_index("seed")
        diff = (own["total_reward"] - ref["total_reward"]).to_numpy(dtype=np.float64)
        rows.append({
            "condition": condition,
            "policy": name,
            "n": len(diff),
            "rule_reward": float(ref["total_reward"].mean()),
            "gap": float(diff.mean()),
            "sem": float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else float("nan"),
            "n_better": int((diff > 0).sum()),
            "viol_frac": float(own["latency_violation_fraction"].mean()),
            "rule_viol_frac": float(ref["latency_violation_fraction"].mean()),
        })
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    p = argparse.ArgumentParser(prog="python -m agent.evaluate_ood", description=__doc__.split("\n\n")[0])
    p.add_argument("--ppo-checkpoint", type=Path, nargs="+", required=True)
    p.add_argument("--holdout-csv", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--days", type=int, default=4)
    p.add_argument("--hazard-error-sigma", type=float, default=0.25)
    p.add_argument("--shifted-sigmas", type=float, nargs="*", default=[0.0, 0.5])
    p.add_argument("--profiles", nargs="*", default=list(PROFILE_NAMES))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    base = build_sim_config("v1", True, args.hazard_error_sigma)
    cached = CachedHazard(base.hazard_model)
    config = dataclasses.replace(base, hazard_model=cached)
    seeds = list(range(args.seeds))
    work = args.out / "workloads"
    conditions: list[tuple[str, Path, SimConfig]] = []
    for name in args.profiles:
        path = write_synthetic(name, args.days, work)
        conditions.append((path.stem, path, config))
    conditions.append(("holdout_phase_shift", write_phase_shift(args.holdout_csv, work), config))
    for sigma in args.shifted_sigmas:
        conditions.append((f"holdout_sigma_{sigma:g}", args.holdout_csv,
                           dataclasses.replace(config, hazard_error_sigma=float(sigma),
                                               label=f"v1,eval-fleet,sigma={sigma:g}")))
    conditions.append(("holdout_degraded_fleet", args.holdout_csv,
                       dataclasses.replace(config, fleet_degradation=DEGRADED_LEVEL,
                                           label=f"v1,eval-fleet,sigma={args.hazard_error_sigma:g},level={DEGRADED_LEVEL:g}")))

    frames, gap_rows = [], []
    for condition, path, cfg in conditions:
        df = compare(_policies(cfg, args.ppo_checkpoint), profiles=[path], seeds=seeds, config=cfg)
        df.insert(0, "condition", condition)
        frames.append(df)
        gap_rows += _gaps(df, condition)
        print(f"condition={condition} done", flush=True)
    episodes = pd.concat(frames, ignore_index=True)
    gaps = pd.DataFrame(gap_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(args.out / "episodes.csv", index=False, lineterminator="\n")
    gaps.to_csv(args.out / "gaps.csv", index=False, lineterminator="\n")
    text = "\n".join([
        "# Forecast PPO agent outside its training distribution (simulated)",
        "",
        f"Fitted risk model, held-out fleet table, {len(seeds)} paired scenario seeds per condition. "
        "`gap` is total reward minus `risk_aware_rule_based`, mean over paired scenarios; conditions are "
        "defined in `agent/evaluate_ood.py`.",
        "",
        to_markdown(gaps),
    ]) + "\n"
    (args.out / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
