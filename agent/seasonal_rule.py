"""The rule with a same-time-yesterday load forecast, with parameters for policy search.

``SeasonalForecastRule`` is ``RiskAwareRuleBased`` whose foreground belief is
replaced, once this episode has observed a day plus an hour of load, by a
belief built from the load observed one day earlier. The point forecast is
``fg[t + 1 - DAY] + (level_now - level_one_day_earlier)``, the levels being
one-hour medians. The scenario spread is the forecast's own errors in this
episode so far (the last ``SEASONAL_ERROR_WINDOW`` of them), scaled by
``spread``. Before a day plus an hour has been observed the policy is the rule
unchanged. It reads only observations from the current episode, the same
information the rule has.

Parameters (all at their neutral value give the hand-built variant):

- ``progress_scale``: multiplies the rule's ``progress_value``, the reward
  units per MB/s of expected delivered rebuild rate.
- ``spread``: multiplies the forecast errors that form the scenario spread.
- ``mix``: probability mass on the seasonal scenarios; the rest stays on the
  rule's own belief. 1.0 replaces the rule's belief entirely.

``agent.search_seasonal`` chooses these by policy search.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from agent.residual_features import DAY_STEPS, LEVEL_WINDOW
from baselines.policies import DEFAULT_PROGRESS_VALUE, RiskAwareRuleBased
from contracts.contracts import Observation
from sim.config import SimConfig

__all__ = ["SEASONAL_ERROR_WINDOW", "SeasonalForecastRule"]

#: Forecast errors kept for the scenario spread: one day.
SEASONAL_ERROR_WINDOW: int = DAY_STEPS
_MIN_ERRORS: int = 12


class SeasonalForecastRule(RiskAwareRuleBased):
    """The rule with an in-episode same-time-yesterday load forecast.

    Raises:
        ValueError: if ``progress_scale`` or ``spread`` is negative, or ``mix`` is outside ``[0, 1]``.
    """

    def __init__(
        self,
        config: SimConfig | None = None,
        *,
        progress_scale: float = 1.0,
        spread: float = 1.0,
        mix: float = 1.0,
        name: str = "rule_seasonal_forecast (hand-built)",
    ) -> None:
        if not progress_scale >= 0.0 or not spread >= 0.0 or not 0.0 <= mix <= 1.0:
            raise ValueError(f"invalid parameters progress_scale={progress_scale!r} spread={spread!r} mix={mix!r}")
        super().__init__(config, progress_value=DEFAULT_PROGRESS_VALUE * float(progress_scale), name=name)
        self.progress_scale: float = float(progress_scale)
        self.spread: float = float(spread)
        self.mix: float = float(mix)
        self._history: list[float] = []
        self._errors: deque[float] = deque(maxlen=SEASONAL_ERROR_WINDOW)
        self._last_forecast: float | None = None

    def reset(self) -> None:
        super().reset()
        self._history = []
        self._errors.clear()
        self._last_forecast = None

    def _belief(self, o: Observation) -> tuple[np.ndarray, np.ndarray]:
        rule_samples, rule_weights = super()._belief(o)
        h = self._history
        h.append(o.fg_bw)
        if self._last_forecast is not None:
            self._errors.append(o.fg_bw - self._last_forecast)
        t = len(h) - 1
        if t < DAY_STEPS + LEVEL_WINDOW - 1:
            self._last_forecast = None
            return rule_samples, rule_weights
        level_now = float(np.median(h[t - LEVEL_WINDOW + 1 : t + 1]))
        level_then = float(np.median(h[t - DAY_STEPS - LEVEL_WINDOW + 1 : t - DAY_STEPS + 1]))
        forecast = max(0.0, h[t + 1 - DAY_STEPS] + level_now - level_then)
        self._last_forecast = forecast
        if len(self._errors) < _MIN_ERRORS:
            samples, weights = np.array([forecast]), np.array([1.0])
        else:
            errors = np.fromiter(self._errors, dtype=np.float64) * self.spread
            samples = np.maximum(0.0, forecast + errors)
            weights = np.full(samples.size, 1.0 / samples.size)
        if self.mix >= 1.0:
            return samples, weights
        return (
            np.concatenate([samples, rule_samples]),
            np.concatenate([weights * self.mix, rule_weights * (1.0 - self.mix)]),
        )
