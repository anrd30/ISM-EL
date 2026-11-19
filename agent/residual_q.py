"""Deterministic residual Q-policy on top of ``SeasonalForecastRule``: fitted Q with paired counterfactual returns.

Usage::

    python -m agent.residual_q --seeds 0 1 2 \\
        --workload-csv data_pipeline/profiles/profile_diurnal_long_train.csv \\
        --hazard-error-sigma 0.25 --out runs/residual_q

For each seed, ``<out>/seed<k>/`` receives ``model.pkl`` (the fitted value
models), ``meta.json`` (selected margin, the validation table, data hashes,
configuration) and ``validation.csv``.

Policy. Each interval the policy takes the action ``SeasonalForecastRule``
would take and may move it one index down or one index up
(``RESIDUAL_DELTAS``). Moves that would exceed the causal latency ceiling of
``sim.latency_filter`` (computed from the last observed foreground load, the
value the observation carries) are masked; keeping the rule's action is never
masked. The policy takes the move with the largest predicted advantage over
keeping the rule's action if that advantage exceeds ``margin``, and keeps the
rule's action otherwise. It is deterministic by construction: the greedy
action of the value function is the policy, so there is no sampled versus
argmax distinction. With an untrained model, or a margin above every
prediction, it is ``SeasonalForecastRule`` exactly.

Learning. Fitted Q on a one-step decision with exact paired returns. In this
simulator the foreground load is exogenous and an interval's latency depends
only on that interval's allocation, so once an interval has executed, the
one-interval value of each of the three moves is known exactly at the
realised load, on the identical scenario:
``-(alpha * D + gamma * V) + progress_value * delivered``, the contract's
latency terms with contract coefficients plus the rule's own price for rebuild
progress (``DEFAULT_PROGRESS_VALUE``), which stands in for the
degraded-interval cost that progress saves later. The targets are the
advantages of moving down and moving up over keeping the rule's action, so all
variation the moves share (the scenario, the fleet, the load level) cancels in
the target. Two gradient-boosted tree regressors fit the two advantages from
the residual features (:mod:`agent.residual_features`, built on
``SeasonalForecastRule``). Data come from rollouts of the rule itself on the
training days, with a uniformly random allowed move on a fraction ``explore``
of intervals.

What the one-step reduction leaves out. A move changes the future only through
the remaining rebuild work, which the progress price values approximately. All
reported results are total episode rewards from the environment, so the
approximation can cost performance but cannot inflate a reported number.

Selection. The training workload is cut time-forward: models are fitted on
the first days and the margin is chosen from ``MARGINS`` by the paired gap
over ``SeasonalForecastRule`` on ``val_seeds`` scenarios of the last
``val_days`` days. The final models are then refitted on rollouts of the whole
training workload with the selected margin. Nothing from an evaluation
workload is used.

Serialisation. ``model.pkl`` is a pickle of scikit-learn regressors. Load only
files this module wrote.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import pickle
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from agent.residual_features import (
    DAY_STEPS,
    FEATURE_VERSION,
    N_RESIDUAL,
    NO_CHANGE_INDEX,
    ResidualFeatures,
    RuleStep,
    compose_action,
)
from agent.search_seasonal import CachedHazard
from agent.seasonal_rule import SeasonalForecastRule
from agent.train_residual import build_sim_config, split_time_forward
from baselines.evaluate import evaluate
from baselines.policies import DEFAULT_PROGRESS_VALUE
from contracts.contracts import ACTIONS_MBPS, BW_TOTAL_MBPS
from sim.config import SimConfig
from sim.latency_filter import latency_ceiling_index
from sim.recovery_env import RecoveryEnv

__all__ = [
    "MARGINS",
    "QConfig",
    "allowed_moves",
    "counterfactual_values",
    "collect",
    "fit_models",
    "ResidualQPolicy",
    "train",
    "main",
]

#: Margins tried on the validation days.
MARGINS: tuple[float, ...] = (0.1, 0.25, 0.5)
_ACTIONS: tuple[float, ...] = tuple(float(a) for a in ACTIONS_MBPS)
#: First scenario seed of training rollouts; disjoint from evaluation (0..99), tuning (1000..1099),
#: validation (5000..) and the other agents' training seeds (100000.., 300000..).
_COLLECT_SEED_BASE: int = 400_000


@dataclass(frozen=True)
class QConfig:
    """Run configuration. Every value is an untuned default."""

    seed: int = 0
    workload_csv: str = "data_pipeline/profiles/profile_diurnal_long_train.csv"
    hazard: str = "v1"
    hazard_error_sigma: float = 0.25
    out_dir: str = "runs/residual_q/seed0"
    episodes: int = 80
    explore: float = 0.1
    val_days: int = 3
    val_seeds: int = 40
    val_seed_start: int = 5000
    max_iter: int = 300
    learning_rate: float = 0.05
    min_samples_leaf: int = 200


def allowed_moves(obs: np.ndarray, step: RuleStep, config: SimConfig) -> np.ndarray:
    """Boolean mask over ``RESIDUAL_DELTAS``: a move is allowed unless it exceeds the causal latency ceiling.

    The ceiling is ``sim.latency_filter.latency_ceiling_index`` at the last observed foreground load,
    never below the reliability floor. Keeping the rule's action is always allowed.
    """
    ceiling, _ = latency_ceiling_index(float(obs[2]) * BW_TOTAL_MBPS, config)
    limit = max(ceiling, step.floor)
    return np.array(
        [k == NO_CHANGE_INDEX or compose_action(step.action, step.floor, k) <= limit for k in range(N_RESIDUAL)]
    )


def counterfactual_values(rule: SeasonalForecastRule, step: RuleStep, fg_mbps: float) -> np.ndarray:
    """One-interval value of each move at the interval's realised foreground load."""
    lr = rule.latency_rule
    reward = rule.config.reward
    target = rule.config.latency_target_ms
    out = np.zeros(N_RESIDUAL, dtype=np.float64)
    for k in range(N_RESIDUAL):
        mbps = _ACTIONS[compose_action(step.action, step.floor, k)]
        latency = float(lr.predicted_latency_ms(fg_mbps, mbps))
        degradation = max(0.0, (latency - target) / target)
        violated = float(latency > lr.limit_ms)
        delivered = float(lr.delivered_mbps(fg_mbps, mbps))
        out[k] = -(reward.alpha * degradation + reward.gamma * violated) + DEFAULT_PROGRESS_VALUE * delivered
    return out


def collect(config: SimConfig, csv_path: str, seeds: Sequence[int], explore: float,
            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Roll out ``SeasonalForecastRule`` (random allowed move on a fraction ``explore`` of intervals).

    Returns features ``(n, FEATURE_DIM)``, counterfactual values ``(n, N_RESIDUAL)`` and masks ``(n, N_RESIDUAL)``.
    """
    env = RecoveryEnv(dataclasses.replace(config, workload_csv=csv_path))
    xs, ys, ms = [], [], []
    for seed in seeds:
        feats = ResidualFeatures(SeasonalForecastRule(config))
        obs, _ = env.reset(seed=int(seed))
        feats.reset()
        while True:
            x, step = feats.update(obs)
            mask = allowed_moves(obs, step, config)
            k = NO_CHANGE_INDEX
            if rng.random() < explore:
                k = int(rng.choice(np.flatnonzero(mask)))
            obs_next, _, terminated, truncated, info = env.step(compose_action(step.action, step.floor, k))
            xs.append(x)
            ys.append(counterfactual_values(feats.rule, step, float(info["fg_bw_mbps"])))
            ms.append(mask)
            obs = obs_next
            if terminated or truncated:
                break
    return np.array(xs), np.array(ys), np.array(ms)


def fit_models(x: np.ndarray, values: np.ndarray, cfg: QConfig) -> dict[int, HistGradientBoostingRegressor]:
    """Fit the down and up advantages over keeping the rule's action."""
    advantage = values - values[:, [NO_CHANGE_INDEX]]
    models = {}
    for k in range(N_RESIDUAL):
        if k == NO_CHANGE_INDEX:
            continue
        models[k] = HistGradientBoostingRegressor(
            max_iter=cfg.max_iter, learning_rate=cfg.learning_rate, min_samples_leaf=cfg.min_samples_leaf,
            random_state=cfg.seed,
        ).fit(x, advantage[:, k])
    return models


class ResidualQPolicy:
    """Greedy residual Q-policy as a ``baselines.policies.Policy``. Deterministic."""

    deployable: bool = True

    def __init__(self, models: dict[int, Any], config: SimConfig, margin: float, name: str = "residual_q") -> None:
        if not margin >= 0.0:
            raise ValueError(f"margin must be >= 0, got {margin!r}")
        self.models: dict[int, Any] = models
        self.config: SimConfig = config
        self.margin: float = float(margin)
        self.name: str = name
        self.features: ResidualFeatures = ResidualFeatures(SeasonalForecastRule(config))
        self.n_steps: int = 0
        self.n_moved: int = 0

    @classmethod
    def from_dir(cls, run_dir: str | os.PathLike[str], config: SimConfig, name: str) -> "ResidualQPolicy":
        """Load ``model.pkl`` and ``meta.json`` written by :func:`train`.

        Raises:
            ValueError: if the feature version differs from this code's.
        """
        run = Path(run_dir)
        meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
        if int(meta["feature_version"]) != FEATURE_VERSION:
            raise ValueError(f"model feature_version={meta['feature_version']}, code {FEATURE_VERSION}")
        with open(run / "model.pkl", "rb") as handle:
            models = pickle.load(handle)
        return cls(models, config, float(meta["margin"]), name=name)

    def reset(self) -> None:
        self.features.reset()

    def __call__(self, obs: np.ndarray) -> int:
        x, step = self.features.update(obs)
        mask = allowed_moves(obs, step, self.config)
        q = np.zeros(N_RESIDUAL, dtype=np.float64)
        for k, model in self.models.items():
            q[k] = float(model.predict(x[None, :])[0])
        q[~mask] = -np.inf
        k = int(np.argmax(q)) if q.max() > self.margin else NO_CHANGE_INDEX
        action = compose_action(step.action, step.floor, k)
        self.n_steps += 1
        self.n_moved += int(action != step.action)
        return action


def _sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train(cfg: QConfig, echo: bool = True) -> Path:
    """Fit on the first days, choose the margin on the last ``val_days``, refit on the whole workload.

    Raises:
        ValueError: on non-positive sizes.
    """
    if min(cfg.episodes, cfg.val_seeds) < 1 or not 0.0 <= cfg.explore <= 1.0:
        raise ValueError("episodes and val_seeds must be >= 1 and explore in [0, 1]")
    started = time.monotonic()
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    base = build_sim_config(cfg.hazard, False, cfg.hazard_error_sigma)
    config = dataclasses.replace(base, hazard_model=CachedHazard(base.hazard_model))
    if cfg.val_days > 0:
        fit_csv, val_csv = split_time_forward(cfg.workload_csv, cfg.val_days * DAY_STEPS, out / "split")
    else:
        fit_csv = val_csv = cfg.workload_csv
    first = _COLLECT_SEED_BASE + cfg.seed * 10_000
    x, values, _ = collect(config, fit_csv, range(first, first + cfg.episodes), cfg.explore, rng)
    models = fit_models(x, values, cfg)
    val_seeds = list(range(cfg.val_seed_start, cfg.val_seed_start + cfg.val_seeds))
    reference = evaluate(SeasonalForecastRule(config), profile=Path(val_csv), seeds=val_seeds, config=config)
    ref = reference["total_reward"].to_numpy(dtype=np.float64)
    rows = []
    for margin in MARGINS:
        policy = ResidualQPolicy(models, config, margin, name=f"residual_q margin={margin:g}")
        frame = evaluate(policy, profile=Path(val_csv), seeds=val_seeds, config=config)
        diff = frame["total_reward"].to_numpy(dtype=np.float64) - ref
        rows.append({"margin": margin, "gap_vs_seasonal": float(diff.mean()),
                     "sem": float(diff.std(ddof=1) / math.sqrt(len(diff))) if len(diff) > 1 else float("nan"),
                     "n_better": int((diff > 0).sum()), "n_worse": int((diff < 0).sum()),
                     "moved_fraction": policy.n_moved / max(1, policy.n_steps),
                     "violation_fraction": float(frame["latency_violation_fraction"].mean())})
        if echo:
            r = rows[-1]
            print(f"seed={cfg.seed} margin={margin:g} val_gap_vs_seasonal={r['gap_vs_seasonal']:+.2f}+-{r['sem']:.2f} "
                  f"moved={r['moved_fraction']:.3f} elapsed_s={time.monotonic() - started:.0f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(out / "validation.csv", index=False, lineterminator="\n")
    best = table.loc[table["gap_vs_seasonal"].idxmax()]
    margin = float(best["margin"])
    if cfg.val_days > 0:
        total_days = max(1, len(pd.read_csv(cfg.workload_csv)) // DAY_STEPS)
        n_val = max(10, round(cfg.episodes * cfg.val_days / total_days))
        x_val, values_val, _ = collect(config, val_csv, range(first + cfg.episodes, first + cfg.episodes + n_val),
                                       cfg.explore, rng)
        x_all, values_all = np.concatenate([x, x_val]), np.concatenate([values, values_val])
    else:
        x_all, values_all = x, values
    final = fit_models(x_all, values_all, cfg)
    with open(out / "model.pkl", "wb") as handle:
        pickle.dump(final, handle)
    meta = {
        "seed": cfg.seed, "margin": margin, "feature_version": FEATURE_VERSION, "hazard": cfg.hazard,
        "eval_fleet": False, "hazard_error_sigma": float(cfg.hazard_error_sigma),
        "trained_on": Path(cfg.workload_csv).as_posix(), "trained_on_sha256": _sha256(cfg.workload_csv),
        "fit_csv_sha256": _sha256(fit_csv), "val_csv_sha256": _sha256(val_csv),
        "val_gap": float(best["gap_vs_seasonal"]), "val_gap_sem": float(best["sem"]),
        "val_seeds": f"{cfg.val_seed_start}..{cfg.val_seed_start + cfg.val_seeds - 1}",
        "samples_selection_fit": int(len(x)), "samples_final_fit": int(len(x_all)),
        "config": asdict(cfg),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if echo:
        print(f"seed={cfg.seed} selected_margin={margin:g} val_gap={meta['val_gap']:+.3f} wrote {out} "
              f"elapsed_s={time.monotonic() - started:.0f}", flush=True)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    d = QConfig()
    p = argparse.ArgumentParser(prog="python -m agent.residual_q", description=__doc__.split("\n\n")[0])
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--workload-csv", default=d.workload_csv)
    p.add_argument("--hazard", choices=["v1", "v0"], default=d.hazard,
                   help="v1: fitted model (needs hazard/artifacts/v1); v0: placeholder, not comparable with v1")
    p.add_argument("--hazard-error-sigma", type=float, default=d.hazard_error_sigma)
    p.add_argument("--episodes", type=int, default=d.episodes)
    p.add_argument("--val-days", type=int, default=d.val_days)
    p.add_argument("--val-seeds", type=int, default=d.val_seeds)
    p.add_argument("--out", default="runs/residual_q")
    args = p.parse_args(argv)
    if args.val_days > 0 and args.val_seeds < 2:
        p.error("--val-seeds must be >= 2")
    for seed in args.seeds:
        train(QConfig(seed=seed, workload_csv=args.workload_csv, hazard=args.hazard,
                      hazard_error_sigma=args.hazard_error_sigma, out_dir=os.path.join(args.out, f"seed{seed}"),
                      episodes=args.episodes, val_days=args.val_days, val_seeds=args.val_seeds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
