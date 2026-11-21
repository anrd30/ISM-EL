"""The control arm: random search over simple policies, at a matched budget.

An iterative result means nothing on its own. This task has four actions and a handful
of useful thresholds, so blind sampling is a strong competitor, and a model that
improves over ten rounds has only shown something if it beats sampling ten policies at
random and keeping the best. This module draws that curve.

The policy family is deliberately the smallest one worth considering: a constant action,
or one threshold on one observation field choosing between two actions. The arena's
``_sine`` floor belongs to it -- over the normalised range of ``obs[2]`` the sine is
monotonic, so ``1 if sin(obs[2]) > 0.3 else 2`` is exactly a threshold at 0.305 on
foreground bandwidth, which is why it is not the nonsense baseline it was taken for.

Two steps, because the expensive part is done once::

    python -m arena.search_baseline scan           # evaluate the family, cache the scores
    python -m arena.search_baseline curve          # expected best-so-far, from the cache

``scan`` evaluates every member on the development workload and on all eight held-out
environments. ``curve`` then replays random orderings over that cache, so the budget
curve costs nothing and can be averaged over thousands of draws. Selection uses the
development score only, exactly as the iterative track does; the held-out score of
whatever got selected is what the curve reports.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import statistics
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Sequence

import numpy as np

from arena.config import DEV_PROFILE, EVAL_PROFILES, SEEDS

__all__ = ["family", "evaluate_member", "main"]

CACHE = Path(__file__).resolve().parent / "reference" / "search_baseline.json"

#: Observation fields a threshold may key on: foreground bandwidth and the headroom
#: left after it. Both are load signals, which is the structure this family can express.
FIELDS = (2, 11)
THRESHOLDS = tuple(round(0.05 * i, 2) for i in range(1, 20))
PAIRS = tuple((hi, lo) for hi, lo in itertools.product(range(4), repeat=2) if hi != lo)


def family() -> list[str]:
    """Every member's name. The name fully determines the policy; see ``_policy``."""
    names = [f"const{a}" for a in range(4)]
    names += [
        f"f{f}_t{t:.2f}_hi{hi}_lo{lo}"
        for f in FIELDS
        for t in THRESHOLDS
        for hi, lo in PAIRS
    ]
    return names


def _policy(name: str):
    """Rebuild a member from its name, so workers need nothing pickled but a string."""
    if name.startswith("const"):
        a = int(name[5:])
        return lambda obs, st: a
    field, thr, hi, lo = name.split("_")
    f, t = int(field[1:]), float(thr[1:])
    a_hi, a_lo = int(hi[2:]), int(lo[2:])
    return lambda obs, st: a_hi if obs[f] > t else a_lo


class _Plain:
    """Harness adapter without the per-step timer: these policies are ours, not a submission."""

    def __init__(self, fn, name):
        self._fn, self.name, self._state = fn, name, {}

    def reset(self) -> None:
        self._state = {}

    def __call__(self, obs) -> int:
        return int(self._fn(np.asarray(obs, dtype=np.float32), self._state))


def evaluate_member(job: tuple[str, str]) -> dict:
    """Score one family member on one environment."""
    name, env = job
    from arena.config import PROFILES, arena_config, split_env
    from arena.reference import reference_table
    from arena.score import aggregate, score_scenario
    from baselines.evaluate import evaluate

    base, variant = split_env(env)
    ref = reference_table(env).set_index("seed")
    frame = evaluate(
        _Plain(_policy(name), name),
        profile=PROFILES / base,
        seeds=list(SEEDS),
        config=arena_config(variant),
    )
    scores = [
        score_scenario(
            seed=int(r["seed"]),
            j_policy=float(r["total_reward"]),
            j_rule=float(ref.loc[int(r["seed"]), "j_rule"]),
            j_star=float(ref.loc[int(r["seed"]), "j_star"]),
            violation_fraction=float(r["latency_violation_fraction"]),
            completion_hours=float(r["completion_hours"]),
        )
        for _, r in frame.iterrows()
    ]
    return {"member": name, "env": env, "g": aggregate(scores).as_dict()["mean_g"]}


def _scan(workers: int) -> int:
    members = family()
    envs = (DEV_PROFILE, *EVAL_PROFILES)
    jobs = [(m, e) for m in members for e in envs]
    print(f"{len(members)} members x {len(envs)} environments = {len(jobs)} evaluations", flush=True)

    rows: dict[str, dict[str, float]] = {m: {} for m in members}
    with Pool(processes=workers) as pool:
        for i, row in enumerate(pool.imap_unordered(evaluate_member, jobs, chunksize=4), 1):
            rows[row["member"]][row["env"]] = row["g"]
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(
        json.dumps({"dev_profile": DEV_PROFILE, "envs": list(envs), "members": rows}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {CACHE}", flush=True)
    return 0


def _curve(draws: int, budget: int, seed: int) -> int:
    """Expected best-so-far after k samples, selecting on development score."""
    if not CACHE.is_file():
        print(f"no cache at {CACHE}; run `python -m arena.search_baseline scan` first")
        return 1
    data = json.loads(CACHE.read_text(encoding="utf-8"))
    members = [m for m, d in data["members"].items() if DEV_PROFILE in d]
    dev = {m: data["members"][m][DEV_PROFILE] for m in members}
    held = {
        m: statistics.median(
            data["members"][m][e] for e in EVAL_PROFILES if e in data["members"][m]
        )
        for m in members
    }

    rng = np.random.default_rng(seed)
    budget = min(budget, len(members))
    dev_curve = np.zeros((draws, budget))
    held_curve = np.zeros((draws, budget))
    for d in range(draws):
        order = rng.permutation(members)
        best_m, best_dev = None, -np.inf
        for k, m in enumerate(order[:budget]):
            if dev[m] > best_dev:
                best_m, best_dev = m, dev[m]
            dev_curve[d, k] = best_dev
            held_curve[d, k] = held[best_m]

    print(f"random search over {len(members)} simple policies, {draws} orderings")
    print("selection on the development score; held-out is the median over the 8 environments\n")
    print(f"{'evaluations':>12s} {'dev G':>12s} {'held-out G':>12s}")
    for k in range(budget):
        if k < 12 or (k + 1) % 10 == 0 or k == budget - 1:
            print(f"{k + 1:12d} {dev_curve[:, k].mean():12.3f} {held_curve[:, k].mean():12.3f}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arena.search_baseline")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("scan", help="evaluate the whole family and cache the scores")
    sc.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    cu = sub.add_parser("curve", help="expected best-so-far versus budget, from the cache")
    cu.add_argument("--draws", type=int, default=2000)
    cu.add_argument("--budget", type=int, default=30)
    cu.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.cmd == "scan":
        return _scan(args.workers)
    return _curve(args.draws, args.budget, args.seed)


if __name__ == "__main__":
    sys.exit(main())
