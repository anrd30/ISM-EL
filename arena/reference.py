"""Reference values: the rule's return and the exact optimum, per scenario.

Both are computed with the project's own evaluation harness, so a submission is
scored against exactly the trajectories the baselines produce. They are cached to
CSV because the optimum takes minutes to solve and never changes for a fixed
(profile, config, seed) triple.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from arena.config import PROFILES, REFERENCE, SEEDS, arena_config, reference_stem, split_env

__all__ = ["reference_table", "build_reference"]

_COLUMNS = ["seed", "j_rule", "j_star", "rule_violation_fraction", "rule_completion_hours"]


def _path(profile: str) -> Path:
    return REFERENCE / f"{reference_stem(profile)}.csv"


def build_reference(profile: str, *, force: bool = False) -> pd.DataFrame:
    """Compute and cache the rule and optimum returns for one profile."""
    out = _path(profile)
    if out.exists() and not force:
        return pd.read_csv(out)

    from baselines.dp_optimum import DPHindsightOptimum
    from baselines.evaluate import evaluate
    from baselines.policies import RiskAwareRuleBased

    base, variant = split_env(profile)
    cfg = arena_config(variant)
    csv = PROFILES / base
    seeds = list(SEEDS)

    rule = evaluate(RiskAwareRuleBased(cfg), profile=csv, seeds=seeds, config=cfg)
    star = evaluate(DPHindsightOptimum(cfg), profile=csv, seeds=seeds, config=cfg)

    frame = pd.DataFrame(
        {
            "seed": rule["seed"].to_numpy(),
            "j_rule": rule["total_reward"].to_numpy(),
            "j_star": star["total_reward"].to_numpy(),
            "rule_violation_fraction": rule["latency_violation_fraction"].to_numpy(),
            "rule_completion_hours": rule["completion_hours"].to_numpy(),
        }
    )[_COLUMNS]

    bad = frame[frame["j_star"] < frame["j_rule"] - 1e-9]
    if len(bad):
        raise RuntimeError(
            f"optimum below rule on seeds {list(bad['seed'])} for {profile}; "
            "the reference is invalid and must not be published"
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return frame


def reference_table(profile: str) -> pd.DataFrame:
    """Cached reference values, building them if absent.

    Raises:
        FileNotFoundError: if the profile does not exist.
    """
    base, _ = split_env(profile)
    if not (PROFILES / base).is_file():
        raise FileNotFoundError(PROFILES / base)
    return build_reference(profile)
