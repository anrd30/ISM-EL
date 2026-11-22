"""Value of information: ``OracleLookahead`` with a calibrated Gaussian belief of standard deviation sigma.

The look-ahead rule of ``baselines.policies.OracleLookahead`` knows the next interval's foreground load
exactly. Here its belief is replaced by a forecast ``f = true + sigma * z`` (``z`` standard normal, seeded)
with the predictive distribution ``f + sigma * q_i`` on 41 equal-probability normal quantiles, so the
belief is calibrated against the forecast error (the error of ``f`` is ``N(0, sigma)`` and the spread
reported is ``sigma``). ``sigma = 0`` is ``OracleLookahead``. The risk floor, the selector and its
parameters are those of ``RiskAwareRuleBased``.

For each sigma the report gives the paired gain over ``risk_aware_rule_based`` and its share of the gain of
``DP_hindsight_optimum`` (``baselines/dp_optimum.py``), on the scenarios that are controllable under each
definition of ``baselines.evaluate.forced_masks`` and under an explicit list of excluded seeds.

The noisy oracle is not deployable: it reads the true next load from the trace.

Usage::

    python -m agent.voi_sweep --workload-csv data_pipeline/profiles/profile_diurnal_long_holdout.csv \\
        --sigmas 0,2,5,7,10,15 --seeds 20 --eval-fleet --hazard-error-sigma 0.25 --out runs/voi_sweep
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import NormalDist
from typing import Sequence

import numpy as np
import pandas as pd

from baselines.dp_optimum import DPHindsightOptimum
from baselines.evaluate import _build_config, evaluate, forced_masks, to_markdown
from baselines.policies import OracleLookahead, RiskAwareRuleBased, peek_next_fg_mbps
from contracts.contracts import Observation
from sim.config import SimConfig

__all__ = ["N_QUANTILES", "NoisyOracle", "main"]

N_QUANTILES: int = 41
DEFAULT_BELIEF_SEED: int = 10_000
_QUANTILES = np.array([NormalDist().inv_cdf((i + 0.5) / N_QUANTILES) for i in range(N_QUANTILES)])


class NoisyOracle(OracleLookahead):
    """``OracleLookahead`` whose belief is the true next load plus Gaussian forecast error of sd ``sigma``.

    The generator of episode ``i`` (the ``i``-th ``reset`` of this object) is seeded with ``seed + i``, so
    evaluating consecutive scenario seeds starting at 0 gives episode ``i`` the stream ``seed + i``.
    """

    def __init__(self, config: SimConfig | None, sigma: float, seed: int = DEFAULT_BELIEF_SEED) -> None:
        if not (math.isfinite(sigma) and sigma >= 0.0):
            raise ValueError(f"sigma must be finite and >= 0, got {sigma!r}")
        super().__init__(config)
        self.sigma = float(sigma)
        self.belief_seed = int(seed)
        self.name = f"NOISY_ORACLE sigma={sigma:g} (not deployable)"
        self._episode = 0
        self._rng = np.random.default_rng(self.belief_seed)

    def reset(self) -> None:
        super().reset()
        self._rng = np.random.default_rng(self.belief_seed + self._episode)
        self._episode += 1

    def _belief(self, o: Observation) -> tuple[np.ndarray, np.ndarray]:
        if self._env is None:
            raise RuntimeError("NoisyOracle is not bound to an environment; call bind(env)")
        true = peek_next_fg_mbps(self._env)
        if self.sigma == 0.0:
            return np.array([true]), np.array([1.0])
        forecast = true + self.sigma * self._rng.standard_normal()
        samples = np.maximum(0.0, forecast + self.sigma * _QUANTILES)
        return samples, np.full(samples.size, 1.0 / samples.size)


def _share_table(
    frames: dict[float, pd.DataFrame], rule: pd.DataFrame, dp: pd.DataFrame, exclude: Sequence[int]
) -> pd.DataFrame:
    """Gain over the rule and share of the optimum, per sigma and definition of controllable."""
    masks = dict(forced_masks(rule))
    if exclude:
        masks["excluded_seeds"] = rule.index.isin(list(exclude))
        masks["excluded_seeds"] = pd.Series(masks["excluded_seeds"], index=rule.index)
    rows = []
    for definition, forced in masks.items():
        keep = ~forced
        headroom = (dp["total_reward"] - rule["total_reward"])[keep]
        for sigma, frame in frames.items():
            gain = (frame["total_reward"] - rule["total_reward"])[keep]
            n = int(keep.sum())
            sem = float(gain.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
            rows.append(
                {"definition": definition, "sigma_mbps": sigma, "n_controllable": n,
                 "gain_over_rule": float(gain.mean()), "sem": sem,
                 "optimum_gain": float(headroom.mean()),
                 "percent_of_optimum": float(100.0 * gain.mean() / headroom.mean())}
            )
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent.voi_sweep", description=__doc__.split("\n\n")[0])
    parser.add_argument("--workload-csv", type=Path, required=True)
    parser.add_argument("--sigmas", default="0,2,5,7,10,15", help="comma-separated forecast sd in MB/s")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--hazard-error-sigma", type=float, default=0.0)
    parser.add_argument("--eval-fleet", action="store_true")
    parser.add_argument("--exclude-seeds", default="", help="comma-separated seeds also reported as an explicit split")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.latency_filter = False
    args.latency_filter_causal = False
    config = _build_config(args)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    sigmas = [float(x) for x in args.sigmas.split(",") if x.strip()]
    exclude = [int(x) for x in args.exclude_seeds.split(",") if x.strip()]

    def run(policy) -> pd.DataFrame:
        return evaluate(policy, profile=args.workload_csv, seeds=seeds, config=config).set_index("seed")

    rule = run(RiskAwareRuleBased(config))
    dp = run(DPHindsightOptimum(config))
    frames = {s: run(NoisyOracle(config, s)) for s in sigmas}
    table = _share_table(frames, rule, dp, exclude)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "voi_sweep.csv", index=False, lineterminator="\n", float_format="%.6g")
    episodes = pd.concat(
        [rule.reset_index(), dp.reset_index()] + [f.reset_index() for f in frames.values()], ignore_index=True
    )
    episodes.to_csv(args.out / "episodes.csv", index=False, lineterminator="\n")
    text = "\n".join(
        [f"# Value of information, `{config.label}`", "",
         f"Workload `{args.workload_csv.as_posix()}`, {len(seeds)} scenarios. Gain is `NOISY_ORACLE` minus "
         "`risk_aware_rule_based`; percent of optimum is that gain over `DP_hindsight_optimum` minus the rule, "
         "on the scenarios controllable under each definition.", "", to_markdown(table)]
    )
    (args.out / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
