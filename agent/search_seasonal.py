"""Policy search over the seasonal-forecast rule, and over the rule's progress price as its comparator.

Usage::

    python -m agent.search_seasonal --base seasonal --seeds 0 1 2 \\
        --workload-csv data_pipeline/profiles/profile_diurnal_long_train.csv \\
        --hazard-error-sigma 0.25 --out runs/search
    python -m agent.search_seasonal --base rule --seeds 0 1 2 \\
        --workload-csv data_pipeline/profiles/profile_diurnal_long_train.csv \\
        --hazard-error-sigma 0.25 --out runs/search

For each seed, ``<out>/<base>/seed<k>/`` receives ``log.csv`` (one row per
generation) and ``search.json`` (the parameters with the best validation gap,
and their provenance).

Policies searched. ``--base seasonal``: ``SeasonalForecastRule`` with
parameters ``progress_scale``, ``spread`` and ``mix``
(:mod:`agent.seasonal_rule`). ``--base rule``: ``RiskAwareRuleBased``
with only ``progress_scale``. The second is the comparator for the first: the
rule's progress price was tuned on the synthetic profiles, so retuning it on
the training days is a gain available without any forecast, and only what the
seasonal search gains beyond it is attributable to the forecast.

Method. Cross-entropy method, a derivative-free policy search on episode
return. Each generation samples ``population`` parameter vectors from a
diagonal Gaussian over unconstrained coordinates (log of the scales, logit of
``mix``), runs each for ``episodes`` episodes on the training part of the
workload with the contract reward, and refits the Gaussian to the ``elite``
best by mean total reward. All candidates of a generation share the same
scenario seeds, so their comparison is paired; each generation draws new
seeds. The search has three parameters, not a network: a per-interval
correction network with 51 inputs, trained on the same days
(:mod:`agent.train_residual`), beat the rule on the days it trained on and
lost to it on later days.

Cost. With the fitted risk model, scoring a fleet at reset (nine drives
through a 25-booster ensemble) costs one to two seconds, against about 0.15 s
for the 500-odd steps of an episode. Every candidate of a generation runs the
same scenario seeds, so the same fleets are scored once per candidate. The
search therefore wraps the risk model in :class:`CachedHazard`, which returns
the stored output for a drive whose features it has already scored. The model
is a deterministic function of its inputs, so results are identical with and
without the cache.

Validation. The workload is cut time-forward as in
:func:`agent.train_residual.split_time_forward`: the last ``val_days`` days
are held back. After every generation the distribution's mean is run on
``val_seeds`` scenario seeds of the held-back days, paired with the rule on
the same seeds, and ``search.json`` keeps the best. Nothing from the
evaluation workload is used. The kept ``val_gap`` is a maximum over
generations and is biased upward by that selection.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from agent.seasonal_rule import SeasonalForecastRule
from agent.residual_features import DAY_STEPS
from agent.train_residual import build_sim_config, split_time_forward
from baselines.evaluate import evaluate
from baselines.policies import DEFAULT_PROGRESS_VALUE, RiskAwareRuleBased
from sim.config import SimConfig

__all__ = ["BASES", "CachedHazard", "SearchConfig", "decode", "make_policy", "search", "main"]

BASES: tuple[str, ...] = ("seasonal", "rule")
#: Unconstrained coordinates per base, their initial mean and standard deviation, and their bounds.
_SPACE: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "seasonal": {
        "log_progress_scale": (0.0, 0.5, math.log(0.1), math.log(3.0)),
        "log_spread": (0.0, 0.5, math.log(0.2), math.log(3.0)),
        "logit_mix": (1.1, 1.5, -4.0, 4.0),
    },
    "rule": {
        "log_progress_scale": (0.0, 0.5, math.log(0.1), math.log(3.0)),
    },
}
_SD_FLOOR: float = 0.05
_SMOOTHING: float = 0.7
#: First scenario seed used for training episodes; disjoint from evaluation (0..99), tuning (1000..1099)
#: and validation (5000..) seeds.
_TRAIN_SEED_BASE: int = 100_000


@dataclass(frozen=True)
class SearchConfig:
    """Run configuration. Every value is an untuned default."""

    base: str = "seasonal"
    seed: int = 0
    workload_csv: str = "data_pipeline/profiles/profile_diurnal_long_train.csv"
    hazard: str = "v1"
    eval_fleet: bool = False
    hazard_error_sigma: float = 0.25
    out_dir: str = "runs/search/seasonal/seed0"
    generations: int = 12
    population: int = 12
    elite: int = 4
    episodes: int = 24
    val_days: int = 3
    val_seeds: int = 40
    val_seed_start: int = 5000


class CachedHazard:
    """A ``HazardModel`` that memoises another's output by its exact inputs.

    The key is the drive's features (sorted name and value pairs) and the
    horizon. Calls with an unhashable feature value are passed through
    uncached.
    """

    def __init__(self, model: Any) -> None:
        self.model: Any = model
        self._cache: dict[tuple[Any, ...], tuple[float, float]] = {}
        self.hits: int = 0
        self.misses: int = 0

    def __call__(self, features: Any, horizon_hours: float) -> tuple[float, float]:
        try:
            key: tuple[Any, ...] | None = (float(horizon_hours), *sorted(features.items()))
            hash(key)
        except TypeError:
            key = None
        if key is not None and key in self._cache:
            self.hits += 1
            return self._cache[key]
        out = self.model(features, horizon_hours)
        self.misses += 1
        if key is not None:
            self._cache[key] = out
        return out


def decode(base: str, z: np.ndarray) -> dict[str, float]:
    """Unconstrained coordinates to policy parameters, clipped to the search bounds."""
    out: dict[str, float] = {}
    for value, (name, (_, _, lo, hi)) in zip(z, _SPACE[base].items()):
        v = float(min(hi, max(lo, value)))
        if name == "log_progress_scale":
            out["progress_scale"] = math.exp(v)
        elif name == "log_spread":
            out["spread"] = math.exp(v)
        elif name == "logit_mix":
            out["mix"] = 1.0 / (1.0 + math.exp(-v))
    return out


def make_policy(base: str, params: dict[str, float], config: SimConfig, name: str) -> RiskAwareRuleBased:
    """Build the policy a parameter dictionary describes.

    Raises:
        ValueError: on an unknown ``base``.
    """
    if base == "seasonal":
        return SeasonalForecastRule(
            config, progress_scale=params["progress_scale"], spread=params["spread"], mix=params["mix"], name=name
        )
    if base == "rule":
        return RiskAwareRuleBased(config, progress_value=DEFAULT_PROGRESS_VALUE * params["progress_scale"], name=name)
    raise ValueError(f"base must be one of {BASES}, got {base!r}")


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _mean_reward(policy: RiskAwareRuleBased, csv_path: str, seeds: list[int], config: SimConfig) -> np.ndarray:
    frame = evaluate(policy, profile=Path(csv_path), seeds=seeds, config=config)
    return frame["total_reward"].to_numpy(dtype=np.float64)


def search(cfg: SearchConfig, echo: bool = True) -> Path:
    """Run the search and return the run directory.

    Raises:
        ValueError: on an unknown base or a population/elite combination that cannot be refit.
    """
    if cfg.base not in BASES:
        raise ValueError(f"base must be one of {BASES}, got {cfg.base!r}")
    if not 1 <= cfg.elite < cfg.population or cfg.generations < 1 or cfg.episodes < 1:
        raise ValueError("need generations >= 1, episodes >= 1 and 1 <= elite < population")
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    if cfg.val_days > 0:
        fit_csv, val_csv = split_time_forward(cfg.workload_csv, cfg.val_days * DAY_STEPS, out / "split")
    else:
        fit_csv = val_csv = cfg.workload_csv
    base_sim = build_sim_config(cfg.hazard, cfg.eval_fleet, cfg.hazard_error_sigma)
    sim = dataclasses.replace(base_sim, hazard_model=CachedHazard(base_sim.hazard_model))
    names = list(_SPACE[cfg.base])
    mean = np.array([_SPACE[cfg.base][n][0] for n in names], dtype=np.float64)
    sd = np.array([_SPACE[cfg.base][n][1] for n in names], dtype=np.float64)
    val_seeds = list(range(cfg.val_seed_start, cfg.val_seed_start + cfg.val_seeds))
    rule_val = _mean_reward(RiskAwareRuleBased(sim), val_csv, val_seeds, sim)
    best: dict[str, Any] = {"val_gap": -math.inf}
    started = time.monotonic()
    columns = ["generation", "train_best", "train_elite_mean", *[f"mean_{n}" for n in names],
               *[f"sd_{n}" for n in names], *list(decode(cfg.base, mean)), "val_gap", "val_gap_sem", "elapsed_s"]
    with open(out / "log.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for generation in range(1, cfg.generations + 1):
            first = _TRAIN_SEED_BASE + cfg.seed * 10_000 + (generation - 1) * cfg.episodes
            seeds = list(range(first, first + cfg.episodes))
            candidates = mean + sd * rng.standard_normal((cfg.population, len(names)))
            scores = np.array([
                float(_mean_reward(make_policy(cfg.base, decode(cfg.base, z), sim, "candidate"), fit_csv, seeds,
                                   sim).mean())
                for z in candidates
            ])
            order = np.argsort(-scores)
            elite = candidates[order[: cfg.elite]]
            mean = _SMOOTHING * elite.mean(axis=0) + (1.0 - _SMOOTHING) * mean
            sd = np.maximum(_SD_FLOOR, _SMOOTHING * elite.std(axis=0) + (1.0 - _SMOOTHING) * sd)
            params = decode(cfg.base, mean)
            diff = _mean_reward(make_policy(cfg.base, params, sim, "validation"), val_csv, val_seeds, sim) - rule_val
            gap = float(diff.mean())
            sem = float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else float("nan")
            if gap > best["val_gap"]:
                best = {"generation": generation, "params": params, "val_gap": gap, "val_gap_sem": sem}
            row = {
                "generation": generation,
                "train_best": float(scores[order[0]]),
                "train_elite_mean": float(scores[order[: cfg.elite]].mean()),
                **{f"mean_{n}": float(v) for n, v in zip(names, mean)},
                **{f"sd_{n}": float(v) for n, v in zip(names, sd)},
                **params,
                "val_gap": gap,
                "val_gap_sem": sem,
                "elapsed_s": time.monotonic() - started,
            }
            writer.writerow(row)
            handle.flush()
            if echo:
                shown = " ".join(f"{k}={v:.3f}" for k, v in params.items())
                print(f"base={cfg.base} seed={cfg.seed} generation={generation}/{cfg.generations} {shown} "
                      f"train_elite_mean={row['train_elite_mean']:.2f} val_gap={gap:+.2f}+-{sem:.2f} "
                      f"elapsed_s={row['elapsed_s']:.0f}", flush=True)
    result = {
        "base": cfg.base,
        "seed": cfg.seed,
        **best,
        "trained_on": Path(cfg.workload_csv).as_posix(),
        "trained_on_sha256": _sha256(cfg.workload_csv),
        "fit_csv_sha256": _sha256(fit_csv),
        "val_csv_sha256": _sha256(val_csv),
        "hazard": cfg.hazard,
        "eval_fleet": cfg.eval_fleet,
        "hazard_error_sigma": cfg.hazard_error_sigma,
        "val_seeds": f"{cfg.val_seed_start}..{cfg.val_seed_start + cfg.val_seeds - 1}",
        "config": asdict(cfg),
    }
    (out / "search.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if echo:
        print(f"base={cfg.base} seed={cfg.seed} best_generation={best['generation']} params={best['params']} "
              f"val_gap={best['val_gap']:+.3f} wrote {out / 'search.json'}")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    d = SearchConfig()
    p = argparse.ArgumentParser(prog="python -m agent.search_seasonal", description=__doc__.split("\n\n")[0])
    p.add_argument("--base", choices=list(BASES), default=d.base)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--workload-csv", default=d.workload_csv)
    p.add_argument("--hazard", choices=["v1", "v0"], default=d.hazard,
                   help="v1: fitted model (needs hazard/artifacts/v1); v0: placeholder, not comparable with v1")
    p.add_argument("--hazard-error-sigma", type=float, default=d.hazard_error_sigma)
    p.add_argument("--generations", type=int, default=d.generations)
    p.add_argument("--population", type=int, default=d.population)
    p.add_argument("--elite", type=int, default=d.elite)
    p.add_argument("--episodes", type=int, default=d.episodes)
    p.add_argument("--val-days", type=int, default=d.val_days)
    p.add_argument("--val-seeds", type=int, default=d.val_seeds)
    p.add_argument("--val-seed-start", type=int, default=d.val_seed_start)
    p.add_argument("--out", default="runs/search")
    args = p.parse_args(argv)
    if args.val_seed_start < 100:
        p.error("--val-seed-start overlaps the evaluation seeds 0..99")
    for seed in args.seeds:
        search(SearchConfig(
            base=args.base, seed=seed, workload_csv=args.workload_csv, hazard=args.hazard,
            hazard_error_sigma=args.hazard_error_sigma, out_dir=os.path.join(args.out, args.base, f"seed{seed}"),
            generations=args.generations, population=args.population, elite=args.elite, episodes=args.episodes,
            val_days=args.val_days, val_seeds=args.val_seeds, val_seed_start=args.val_seed_start,
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
