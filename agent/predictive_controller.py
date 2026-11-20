"""Predictive-distribution controller: online forecaster selection with an empirical residual belief.

``PredictiveController`` is ``baselines.policies.RiskAwareRuleBased`` with one change: the belief about the next
interval's foreground load. The rule's selector (the progress-value expected-cost choice), its risk floor and its
parameters are untouched.

Belief. Three forecasters predict the next foreground value from the load observed so far (history included):

- ``persist``: the last value;
- ``ar``: an AR(3) model with intercept, refit by least squares every ``AR_REFIT_EVERY`` observations on the last
  ``AR_FIT_WINDOW`` values, available from ``AR_MIN_SAMPLES`` values;
- ``seasonal``: the value one day earlier at the target time plus the change between now and one day before
  now, available from ``DAY_STEPS`` values.

Each forecaster's squared error over the last ``SELECTION_WINDOW`` observations is tracked, and the forecaster with
the lowest mean (at least ``MIN_SELECTION_ERRORS`` errors needed) is chosen; ``persist`` when none qualifies. The
belief is the chosen forecast plus each of that forecaster's last ``RESIDUAL_WINDOW`` realised errors, clipped at
zero, with equal weights. Before ``MIN_RESIDUALS`` errors exist, the belief is the forecast plus
``FALLBACK_OFFSETS_MBPS`` with equal weights.

History. ``set_history`` (``baselines.policies.NeedsHistory``) hands the controller the load before the episode.
The history is replayed through the forecasters so that their residual pools are filled when the first decision is
made. Without a history the controller starts empty and is the same controller. ``PredictiveControllerNoHistory``
does not implement ``NeedsHistory``, so the harness gives it no history even when the environment provides one.

The parameters are fixed constants and were not tuned on the evaluation scenarios.

Arms for comparison, same module: ``mode`` restricts the controller to one forecaster (``"seasonal"``,
``"persist"``, ``"ar"``); ``RuleWithHistory`` is the unchanged rule with its belief buffers pre-filled from the
history.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Literal

import numpy as np

from baselines.policies import RiskAwareRuleBased
from contracts.contracts import Observation
from sim.config import SimConfig

__all__ = ["PredictiveController", "PredictiveControllerNoHistory", "RuleWithHistory", "FORECASTERS"]

DAY_STEPS: int = 288
AR_ORDER: int = 3
AR_FIT_WINDOW: int = 1500
AR_MIN_SAMPLES: int = 200
AR_REFIT_EVERY: int = 12
SELECTION_WINDOW: int = 48
MIN_SELECTION_ERRORS: int = 12
RESIDUAL_WINDOW: int = 288
MIN_RESIDUALS: int = 24
FALLBACK_OFFSETS_MBPS: tuple[float, ...] = (-20.0, -10.0, 0.0, 10.0, 20.0)
#: A history shorter than this is not replayed through the forecasters' warm-up.
_MIN_WARMUP_HISTORY: int = DAY_STEPS + 10
_WARMUP_SEED_ROWS: int = DAY_STEPS + 4

FORECASTERS: tuple[str, ...] = ("persist", "ar", "seasonal")
Mode = Literal["adaptive", "persist", "ar", "seasonal"]


@dataclass(frozen=True)
class _Sample:
    """The one field the rule's belief reads from an observation."""

    fg_bw: float


class PredictiveController(RiskAwareRuleBased):
    """The rule with a forecaster-selecting, residual-based belief. Deployable: it reads only past observations."""

    deployable: bool = True

    def __init__(
        self,
        config: SimConfig | None = None,
        *,
        mode: Mode = "adaptive",
        name: str | None = None,
    ) -> None:
        if mode not in ("adaptive",) + FORECASTERS:
            raise ValueError(f"mode must be 'adaptive' or one of {FORECASTERS}, got {mode!r}")
        super().__init__(config, name=name or f"predictive_controller[{mode}]")
        self.mode: Mode = mode
        self._clear()

    def _clear(self) -> None:
        self._h: list[float] = []
        self._pending: dict[str, float] = {}
        self._residuals = {k: deque(maxlen=RESIDUAL_WINDOW) for k in FORECASTERS}
        self._squared = {k: deque(maxlen=SELECTION_WINDOW) for k in FORECASTERS}
        self._beta: np.ndarray | None = None
        self._since_fit: int = AR_REFIT_EVERY + 1

    def reset(self) -> None:
        super().reset()
        self._clear()

    def set_history(self, fg_bw: np.ndarray, read_iops: np.ndarray, write_iops: np.ndarray) -> None:
        """Replay the pre-episode foreground bandwidth (MB/s, oldest first) through the forecasters.

        ``read_iops`` and ``write_iops`` are accepted for the protocol and not used.
        """
        history = [float(x) for x in fg_bw]
        if len(history) > _MIN_WARMUP_HISTORY:
            self._h = history[:_WARMUP_SEED_ROWS]
            for x in history[_WARMUP_SEED_ROWS:]:
                self._observe(x)
        else:
            self._h = history

    def _forecasts(self) -> dict[str, float]:
        h = self._h
        out = {"persist": h[-1]}
        if len(h) >= AR_MIN_SAMPLES:
            if self._beta is None or self._since_fit >= AR_REFIT_EVERY:
                x = np.asarray(h[-AR_FIT_WINDOW:])
                columns = [x[AR_ORDER - 1 - i : -1 - i] for i in range(AR_ORDER)]
                design = np.column_stack(columns + [np.ones(len(x) - AR_ORDER)])
                self._beta, *_ = np.linalg.lstsq(design, x[AR_ORDER:], rcond=None)
                self._since_fit = 0
            self._since_fit += 1
            out["ar"] = float(np.dot(self._beta[:AR_ORDER], [h[-1 - i] for i in range(AR_ORDER)]) + self._beta[AR_ORDER])
        if len(h) >= DAY_STEPS:
            level_change = (h[-1] - h[-1 - DAY_STEPS]) if len(h) > DAY_STEPS else 0.0
            out["seasonal"] = h[len(h) - DAY_STEPS] + level_change
        return out

    def _observe(self, x: float) -> None:
        for name, predicted in self._pending.items():
            self._residuals[name].append(x - predicted)
            self._squared[name].append((x - predicted) ** 2)
        self._h.append(x)
        self._pending = self._forecasts()

    def _chosen(self) -> str:
        forecasts = self._pending
        if self.mode == "adaptive":
            candidates = [k for k in forecasts if len(self._squared[k]) >= MIN_SELECTION_ERRORS]
            if not candidates:
                return "persist"
            return min(candidates, key=lambda k: float(np.mean(self._squared[k])))
        return self.mode if self.mode in forecasts else "persist"

    def _belief(self, o: Observation) -> tuple[np.ndarray, np.ndarray]:
        self._observe(float(o.fg_bw))
        chosen = self._chosen()
        forecast = self._pending[chosen]
        pool = self._residuals[chosen]
        if len(pool) < MIN_RESIDUALS:
            samples = forecast + np.array(FALLBACK_OFFSETS_MBPS)
            return samples, np.full(samples.size, 1.0 / samples.size)
        samples = np.maximum(0.0, forecast + np.fromiter(pool, dtype=np.float64))
        return samples, np.full(samples.size, 1.0 / samples.size)


class PredictiveControllerNoHistory(PredictiveController):
    """The controller that ignores the pre-episode history (a fairness arm); it starts every episode empty."""

    set_history = None  # type: ignore[assignment]  # not a NeedsHistory policy

    def __init__(self, config: SimConfig | None = None, *, mode: Mode = "adaptive", name: str | None = None) -> None:
        super().__init__(config, mode=mode, name=name or f"predictive_controller[{mode}] (no history)")


class RuleWithHistory(RiskAwareRuleBased):
    """``RiskAwareRuleBased`` with its belief buffers pre-filled from the pre-episode load history."""

    deployable: bool = True

    def __init__(self, config: SimConfig | None = None, *, name: str = "rule+history") -> None:
        super().__init__(config, name=name)

    def set_history(self, fg_bw: np.ndarray, read_iops: np.ndarray, write_iops: np.ndarray) -> None:
        for x in fg_bw:
            self._belief(_Sample(float(x)))  # type: ignore[arg-type]
