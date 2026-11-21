"""Frozen arena constants. Changing any of these invalidates every existing score.

Spec version is recorded in every result file. Bump it on any change here.
"""

from __future__ import annotations

from pathlib import Path

SPEC_VERSION = "1.5.0"

#: Scenario seeds. Identical for every submission, so comparisons are paired.
SEEDS: tuple[int, ...] = tuple(range(20))

#: A scenario counts toward the score only if the optimum beats the rule by more
#: than this FRACTION of the rule's own cost. Below it the safety filter forces the
#: allocation and no policy can meaningfully differ, so the normalised score would be
#: undefined or numerically unstable.
#:
#: Relative, not absolute, so that the controllability test is scale-free in the same
#: way the score is: rescaling the reward coefficients must not change which scenarios
#: count. With the rule near -330 and typical headroom near 12 (3.6%), 0.1% is a floor
#: that excludes only genuinely filter-forced scenarios.
EPSILON_REL: float = 0.001

#: Per-decision wall-clock budget. Exceeding it fails the episode.
STEP_BUDGET_S: float = 0.010

#: Whole-submission wall-clock budget, enforced by the sandbox.
SUBMISSION_BUDGET_S: float = 1800.0

#: Maximum feedback rounds in the iterative track.
MAX_ROUNDS: int = 5

#: Generation budget, identical for every backend. Part of the protocol: a model
#: truncated mid-thought is recorded as producing no policy, which is a property of
#: this number rather than of the model. Before 1.4.0 the two backends disagreed --
#: Ollama allowed 4096 and HuggingFace 2048 -- so results were not comparable across
#: them. See arena.generate.DEFAULT_MAX_NEW_TOKENS.
MAX_NEW_TOKENS: int = 8192

#: Development population. The only one a model may receive feedback from.
DEV_PROFILE = "profile_diurnal_long_train.csv"

#: Held-out environments. Never shown to a model, never used for feedback.
#:
#: An environment name is a workload file, optionally followed by ``@<variant>``. The
#: variant selects a simulator configuration from ``ENV_VARIANTS``; without one the
#: environment is the arena's original configuration. The first entry is the original
#: held-out profile, so a report that reads only the first block is unchanged.
#:
#: ``profile_diurnal_long_altB.csv``, ``_popC.csv`` and ``_popD.csv`` are the project's
#: three further 14-day periodic populations (``data_pipeline/PREPROCESSING.md`` B5b,
#: B5c), disjoint from the development and original held-out volumes in window and
#: volumes. The ``@`` entries keep the original held-out workload and change the
#: objective or the latency model.
EVAL_PROFILES: tuple[str, ...] = (
    "profile_diurnal_long_holdout.csv",
    "profile_diurnal_long_altB.csv",
    "profile_diurnal_long_popC.csv",
    "profile_diurnal_long_popD.csv",
    "profile_diurnal_long_holdout.csv@viol_cheap",
    "profile_diurnal_long_holdout.csv@degraded_costly",
    "profile_diurnal_long_holdout.csv@ssd_latency",
    "profile_diurnal_long_holdout.csv@hdd_tight",
)

#: Simulator overrides per variant. ``reward`` values are (alpha, beta, gamma) of
#: ``contracts.RewardConfig`` (latency degradation, degraded-state cost, violation
#: penalty); the arena's default is (1.0, 0.1, 10.0). The latency fields are those of
#: ``SimConfig``: unloaded latency and target in ms, whose ratio sets the utilisation at
#: which a violation starts (``1 - unloaded/target``; 0.867 by default). Rebuild rate and
#: capacity are contract constants, not configuration, so the device-class variants
#: change only the latency model.
ENV_VARIANTS: dict[str, dict[str, object]] = {
    "viol_cheap": {"reward": (1.0, 0.1, 1.0)},
    "degraded_costly": {"reward": (1.0, 1.0, 10.0)},
    "ssd_latency": {"latency_unloaded_ms": 0.1, "latency_target_ms": 2.0},
    "hdd_tight": {"latency_unloaded_ms": 5.0, "latency_target_ms": 20.0},
}

REPO = Path(__file__).resolve().parents[1]
PROFILES = REPO / "data_pipeline" / "profiles"
REFERENCE = Path(__file__).resolve().parent / "reference"
RESULTS = Path(__file__).resolve().parent / "results"
SUBMISSIONS = Path(__file__).resolve().parent / "submissions"


def split_env(name: str) -> tuple[str, str | None]:
    """Split an environment name into its workload file and its variant (None for the original)."""
    base, sep, variant = name.partition("@")
    if sep and variant not in ENV_VARIANTS:
        raise ValueError(f"unknown environment variant {variant!r} in {name!r}")
    return base, (variant if sep else None)


def reference_stem(name: str) -> str:
    """File stem of the cached reference values of an environment."""
    base, variant = split_env(name)
    return Path(base).stem + (f"@{variant}" if variant else "")


def arena_config(variant: str | None = None):
    """Simulator configuration for the arena.

    ``variant`` selects an entry of ``ENV_VARIANTS``; ``None`` is the original
    configuration, unchanged.

    Uses the heuristic v0 hazard so the arena runs from a clean clone with no
    large artifacts to download. The research results use the fitted v1 model;
    the arena's claim is about control policy quality, which v0 supports, and
    using it is what makes the arena runnable by anyone in three commands.
    """
    from sim.config import v0_config

    if variant is None:
        return v0_config(label=f"arena-{SPEC_VERSION}")
    overrides = dict(ENV_VARIANTS[variant])
    if "reward" in overrides:
        from contracts.contracts import RewardConfig

        alpha, beta, gamma = overrides["reward"]  # type: ignore[misc]
        overrides["reward"] = RewardConfig(alpha=alpha, beta=beta, gamma=gamma)
    return v0_config(label=f"arena-{SPEC_VERSION}@{variant}", **overrides)
