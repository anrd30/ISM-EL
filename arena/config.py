"""Frozen arena constants. Changing any of these invalidates every existing score.

Spec version is recorded in every result file. Bump it on any change here.
"""

from __future__ import annotations

from pathlib import Path

SPEC_VERSION = "1.4.0"

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

#: Held-out populations. Never shown to a model, never used for feedback.
EVAL_PROFILES: tuple[str, ...] = ("profile_diurnal_long_holdout.csv",)

REPO = Path(__file__).resolve().parents[1]
PROFILES = REPO / "data_pipeline" / "profiles"
REFERENCE = Path(__file__).resolve().parent / "reference"
RESULTS = Path(__file__).resolve().parent / "results"
SUBMISSIONS = Path(__file__).resolve().parent / "submissions"


def arena_config():
    """Simulator configuration for the arena.

    Uses the heuristic v0 hazard so the arena runs from a clean clone with no
    large artifacts to download. The research results use the fitted v1 model;
    the arena's claim is about control policy quality, which v0 supports, and
    using it is what makes the arena runnable by anyone in three commands.
    """
    from sim.config import v0_config

    return v0_config(label=f"arena-{SPEC_VERSION}")
