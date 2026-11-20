"""The arena score.

For scenario ``i`` with policy return ``J``, rule return ``J_rule`` and optimum
``J_star``::

    G(i) = (J - J_rule) / (J_star - J_rule)

``G = 1`` is optimal, ``G = 0`` matches the rule, ``G < 0`` is worse than the rule.
``G > 1`` is impossible and is reported as an error rather than clipped, because it
means the optimum was not optimal.

A scenario counts only if the headroom exceeds ``EPSILON_REL`` times the rule's own
cost. The test is relative rather than absolute so that it is scale-free in the same
way the score is: rescaling the reward must not change which scenarios count.

The score is dimensionless, so the reward's coefficients cannot flatter a submission.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from arena.config import EPSILON_REL

__all__ = ["ScenarioScore", "Aggregate", "score_scenario", "aggregate"]


@dataclass(frozen=True)
class ScenarioScore:
    seed: int
    j_policy: float
    j_rule: float
    j_star: float
    controllable: bool
    g: float  # nan when not controllable
    violation_fraction: float
    completion_hours: float


@dataclass
class Aggregate:
    mean_g: float
    sem_g: float
    n_controllable: int
    n_excluded: int
    n_better_than_rule: int
    mean_violation_fraction: float
    mean_completion_hours: float
    impossible: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "mean_g": self.mean_g,
            "sem_g": self.sem_g,
            "n_controllable": self.n_controllable,
            "n_excluded": self.n_excluded,
            "n_better_than_rule": self.n_better_than_rule,
            "mean_violation_fraction": self.mean_violation_fraction,
            "mean_completion_hours": self.mean_completion_hours,
            "impossible_seeds": self.impossible,
        }


def score_scenario(
    seed: int,
    j_policy: float,
    j_rule: float,
    j_star: float,
    violation_fraction: float,
    completion_hours: float,
) -> ScenarioScore:
    """Normalised score for one scenario.

    Raises:
        ValueError: if the optimum is below the rule, which means the reference
            values are wrong and nothing downstream can be trusted.
    """
    headroom = j_star - j_rule
    if headroom < -1e-9:
        raise ValueError(
            f"seed {seed}: optimum {j_star} is below the rule {j_rule}; the reference "
            "values are invalid"
        )
    controllable = headroom > EPSILON_REL * abs(j_rule)
    g = (j_policy - j_rule) / headroom if controllable else math.nan
    return ScenarioScore(
        seed=seed,
        j_policy=j_policy,
        j_rule=j_rule,
        j_star=j_star,
        controllable=controllable,
        g=g,
        violation_fraction=violation_fraction,
        completion_hours=completion_hours,
    )


def aggregate(scores: list[ScenarioScore]) -> Aggregate:
    """Mean and standard error of ``G`` across controllable scenarios.

    The standard error is taken across scenarios, not across seeds within a
    scenario: seeds sharing a scenario are near-replicates and pooling them as
    independent understates the error.
    """
    live = [s for s in scores if s.controllable]
    if not live:
        return Aggregate(math.nan, math.nan, 0, len(scores), 0, math.nan, math.nan)

    g = np.array([s.g for s in live], dtype=float)
    sem = float(g.std(ddof=1) / math.sqrt(len(g))) if len(g) > 1 else math.nan
    impossible = [s.seed for s in live if s.g > 1.0 + 1e-9]

    return Aggregate(
        mean_g=float(g.mean()),
        sem_g=sem,
        n_controllable=len(live),
        n_excluded=len(scores) - len(live),
        n_better_than_rule=int((g > 0).sum()),
        mean_violation_fraction=float(np.mean([s.violation_fraction for s in live])),
        mean_completion_hours=float(np.mean([s.completion_hours for s in live])),
        impossible=impossible,
    )
