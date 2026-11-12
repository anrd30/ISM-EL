"""Inputs and action composition for the residual controller.

The residual controller does not choose an allocation from scratch. Each
interval it chooses a correction ``delta`` in ``RESIDUAL_DELTAS`` (one action
index down, unchanged, one up) that is added to the action
``RiskAwareRuleBased`` would have taken, and the result is clipped to
``[risk_floor, N_ACTIONS - 1]``. With ``delta = 0`` everywhere the controller
is the rule exactly, which is how the policy is initialised.

Information set. Every feature is built from the sequence of observation
vectors the environment has returned in the current episode and from the
rule's own internal state, which is itself built from those observations. No
feature reads the trace, the environment or anything after the current
observation, so the controller sees exactly what ``RiskAwareRuleBased`` sees.
``ResidualFeatures.update`` is the only entry point and takes one observation
at a time; ``test_learned.py`` checks that the features at step ``t`` do not
change when later observations are altered.

Features, in order (``FEATURE_DIM`` in total):

- the 12 normalised observation fields;
- the rule's chosen action, one-hot (4);
- the rule's risk floor index divided by ``N_ACTIONS - 1`` (1);
- for each action, ``tanh`` of the rule's score minus the best score (4). The
  score is the rule's own one-step objective,
  ``E[alpha * D + gamma * V] - progress_value * E[delivered]``;
- recent load: ``(fg[t - k] - fg[t]) / LOAD_SCALE`` for ``k = 1..HISTORY`` (12);
- same time yesterday, available once ``DAY_STEPS + SEASONAL_AHEAD`` steps of
  this episode have been observed: an availability flag, the load changes
  observed one day earlier over the next ``SEASONAL_AHEAD`` intervals,
  ``(fg[t + j - DAY] - fg[t - DAY]) / LOAD_SCALE`` for ``j = 1..SEASONAL_AHEAD``,
  and the change in the one-hour median level since then (1 + 6 + 1); zeros
  while unavailable;
- elapsed intervals in this episode divided by ``2 * DAY_STEPS``, clipped to 1 (1);
- for each correction in ``RESIDUAL_DELTAS``, applied to the rule's action and
  clipped as :func:`compose_action` does: the foreground load at which that
  allocation would first violate the latency limit, minus the last observed
  load, over ``LOAD_SCALE`` (3); the same threshold minus the
  same-time-yesterday forecast ``fg[t + 1 - DAY] + level change``, zero while
  unavailable (3); and the probability of a violation under the rule's own
  foreground belief (3).

Why the last block. A violation is a step function of load plus allocation
(the limit is first exceeded at a fixed utilisation), so whether a correction
violates turns on a sharp threshold. A small tanh network fitted to
counterfactual outcomes with only the raw load features recovered
0.015 to 0.035 reward per interval in sample, against 0.058 out of sample for
gradient-boosted trees on the same data; stating the threshold distance per
correction makes that boundary linear in the inputs.

Why same time yesterday. ``profile_diurnal_long`` aggregates 44 volumes
selected for the strongest daily cycle. On its 10-day training half, the load
one day earlier predicts the next interval's load with an RMSE of 2.7 MB/s,
against 10.1 MB/s for the last observed value, and step changes correlate at
0.97 with the step changes one day earlier. ``RiskAwareRuleBased`` has no
time-of-day term in its belief. This is a property of that selected trace, not
of storage workloads in general.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baselines.policies import RiskAwareRuleBased
from contracts.contracts import ACTIONS_MBPS, BW_TOTAL_MBPS, N_ACTIONS, OBS_DIM, Observation

__all__ = [
    "RESIDUAL_DELTAS",
    "N_RESIDUAL",
    "NO_CHANGE_INDEX",
    "DAY_STEPS",
    "HISTORY",
    "SEASONAL_AHEAD",
    "LEVEL_WINDOW",
    "LOAD_SCALE",
    "FEATURE_DIM",
    "FEATURE_VERSION",
    "RuleStep",
    "ResidualFeatures",
    "compose_action",
]

#: Corrections to the rule's action index, in policy-output order.
RESIDUAL_DELTAS: tuple[int, ...] = (-1, 0, 1)
N_RESIDUAL: int = len(RESIDUAL_DELTAS)
#: Policy output index meaning "take the rule's action".
NO_CHANGE_INDEX: int = RESIDUAL_DELTAS.index(0)

#: Intervals per day at the 300 s control interval.
DAY_STEPS: int = 288
#: Number of past load values in the recent-load block.
HISTORY: int = 12
#: Intervals ahead read from the previous day.
SEASONAL_AHEAD: int = 6
#: Intervals in the median used for the level-change feature (one hour).
LEVEL_WINDOW: int = 12
#: MB/s that map to 1.0 in the load-difference features.
LOAD_SCALE: float = 50.0

FEATURE_DIM: int = OBS_DIM + N_ACTIONS + 1 + N_ACTIONS + HISTORY + (1 + SEASONAL_AHEAD + 1) + 1 + 3 * N_RESIDUAL
#: Stored in checkpoints; a checkpoint with another version is refused.
FEATURE_VERSION: int = 2

_ACTIONS: tuple[float, ...] = tuple(float(a) for a in ACTIONS_MBPS)


@dataclass(frozen=True)
class RuleStep:
    """What the rule decided for one observation, and the belief it decided on."""

    action: int
    floor: int
    scores: np.ndarray
    samples: np.ndarray
    weights: np.ndarray


def compose_action(rule_action: int, floor: int, residual_index: int) -> int:
    """Rule action plus ``RESIDUAL_DELTAS[residual_index]``, clipped to ``[floor, N_ACTIONS - 1]``.

    Raises:
        ValueError: if ``residual_index`` is out of range.
    """
    if not 0 <= residual_index < N_RESIDUAL:
        raise ValueError(f"residual_index must be in [0, {N_RESIDUAL}), got {residual_index!r}")
    return int(min(N_ACTIONS - 1, max(floor, rule_action + RESIDUAL_DELTAS[residual_index])))


class ResidualFeatures:
    """Per-episode state: the wrapped rule and the observed load history.

    ``reset`` must be called at the start of every episode; ``update`` once per
    observation, in order. ``update`` advances the rule's belief exactly as
    calling the rule on the same observation would, and returns the feature
    vector together with the rule's decision.
    """

    def __init__(self, rule: RiskAwareRuleBased) -> None:
        self.rule: RiskAwareRuleBased = rule
        self._fg: list[float] = []

    def reset(self) -> None:
        """Clear the load history and the rule's belief."""
        self.rule.reset()
        self._fg = []

    def _rule_step(self, obs: np.ndarray) -> RuleStep:
        o = Observation.from_vector(obs, normalized=True)
        samples, weights = self.rule._belief(o)
        floor = self.rule.risk_floor_index(o)
        action = self.rule._select(samples, weights, floor)
        scores = np.array(
            [
                self.rule.latency_rule.score(samples, weights, _ACTIONS[i], self.rule.progress_value)
                for i in range(N_ACTIONS)
            ],
            dtype=np.float64,
        )
        return RuleStep(action=int(action), floor=int(floor), scores=scores, samples=samples, weights=weights)

    def update(self, obs: np.ndarray) -> tuple[np.ndarray, RuleStep]:
        """Consume one observation; return ``(features, rule_step)``.

        Raises:
            ValueError: if ``obs`` does not have shape ``(OBS_DIM,)``.
        """
        arr = np.asarray(obs, dtype=np.float64)
        if arr.shape != (OBS_DIM,):
            raise ValueError(f"observation must have shape ({OBS_DIM},), got {arr.shape}")
        step = self._rule_step(arr)
        self._fg.append(float(arr[2]) * BW_TOTAL_MBPS)
        return self._features(arr, step), step

    def _features(self, obs: np.ndarray, step: RuleStep) -> np.ndarray:
        fg = self._fg
        t = len(fg) - 1
        now = fg[t]
        out = np.zeros(FEATURE_DIM, dtype=np.float32)
        i = 0
        out[i : i + OBS_DIM] = obs
        i += OBS_DIM
        out[i + step.action] = 1.0
        i += N_ACTIONS
        out[i] = step.floor / (N_ACTIONS - 1)
        i += 1
        feasible = step.scores[step.floor :]
        best = float(np.min(feasible))
        out[i : i + N_ACTIONS] = np.tanh(step.scores - best)
        i += N_ACTIONS
        for k in range(1, HISTORY + 1):
            past = fg[t - k] if t - k >= 0 else fg[0]
            out[i] = (past - now) / LOAD_SCALE
            i += 1
        # Every index read below is at most t - DAY_STEPS + SEASONAL_AHEAD < t.
        if t >= DAY_STEPS + LEVEL_WINDOW - 1:
            anchor = fg[t - DAY_STEPS]
            out[i] = 1.0
            for j in range(1, SEASONAL_AHEAD + 1):
                out[i + j] = (fg[t + j - DAY_STEPS] - anchor) / LOAD_SCALE
            level_now = float(np.median(fg[t - LEVEL_WINDOW + 1 : t + 1]))
            level_then = float(np.median(fg[t - DAY_STEPS - LEVEL_WINDOW + 1 : t - DAY_STEPS + 1]))
            out[i + SEASONAL_AHEAD + 1] = (level_now - level_then) / LOAD_SCALE
        forecast = None
        if t >= DAY_STEPS + LEVEL_WINDOW - 1:
            forecast = fg[t + 1 - DAY_STEPS] + LOAD_SCALE * float(out[i + SEASONAL_AHEAD + 1])
        i += 1 + SEASONAL_AHEAD + 1
        out[i] = min(1.0, t / (2.0 * DAY_STEPS))
        i += 1
        lr = self.rule.latency_rule
        rho_limit = 1.0 - self.rule.config.latency_unloaded_ms / lr.limit_ms
        for k in range(N_RESIDUAL):
            alloc = _ACTIONS[compose_action(step.action, step.floor, k)]
            threshold = rho_limit * BW_TOTAL_MBPS - alloc
            out[i + k] = (threshold - now) / LOAD_SCALE
            if forecast is not None:
                out[i + N_RESIDUAL + k] = (threshold - forecast) / LOAD_SCALE
            violates = (lr.predicted_latency_ms(step.samples, alloc) > lr.limit_ms).astype(np.float64)
            out[i + 2 * N_RESIDUAL + k] = float(step.weights @ violates)
        i += 3 * N_RESIDUAL
        assert i == FEATURE_DIM
        return out
