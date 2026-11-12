"""Per-interval latency violation the smallest action would have produced at the same state.

For the interval about to run, ``RecoveryEnv.step`` applies ``apply_safety_filter`` to the
requested action using the hazard and the remaining work at the start of the interval, caps the
result by the foreground headroom and decides the violation from the resulting utilisation.
``min_action_violation`` repeats that computation for action index 0 from the same start-of-
interval state and the same interval's foreground load. The inputs are floats and
``apply_safety_filter`` returns a new frozen value, so the call changes no state the real step
uses.

``MinActionCounterfactual`` wraps an environment, evaluates the counterfactual immediately
before each ``step`` and records it next to the real outcome. Reading the start-of-interval
state uses the private attributes ``_hazard``, ``_work``, ``_t``, ``_offset`` and ``_trace`` of
``RecoveryEnv``. The latency filters (``SimConfig.latency_filter`` and ``latency_filter_causal``)
change the applied allocation after the safety filter and are not modelled; wrapping an
environment with either set raises ``ValueError``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from contracts.contracts import BW_TOTAL_MBPS, latency_violation
from sim.config import SimConfig, latency_mean_ms
from sim.safety_filter import apply_safety_filter

__all__ = ["min_action_violation", "MinActionCounterfactual"]


def min_action_violation(hazard_per_hour: float, work_remaining_gb: float, fg_mbps: float, cfg: SimConfig) -> tuple[bool, int]:
    """Return ``(violated, safety-filter floor index)`` for action index 0 at the given state.

    ``floor index`` is the action the safety filter applies to a request of index 0.
    """
    decision = apply_safety_filter(0, float(hazard_per_hour), float(work_remaining_gb))
    avail = max(0.0, cfg.rho_max * BW_TOTAL_MBPS - float(fg_mbps))
    effective = min(decision.applied_mbps, avail)
    rho = min(cfg.rho_max, (float(fg_mbps) + effective) / BW_TOTAL_MBPS)
    violated = latency_violation(latency_mean_ms(rho, cfg), cfg.latency_target_ms) == 1.0
    return bool(violated), int(decision.applied_index)


class MinActionCounterfactual:
    """Environment wrapper recording the smallest-action counterfactual for every step.

    Args:
        env: ``AblatedEnv`` or ``RecoveryEnv``; the wrapped chain must end in a ``RecoveryEnv``.
        cfg: The ``SimConfig`` the environment was built from.

    Raises:
        ValueError: if ``cfg`` has a latency filter enabled.
    """

    def __init__(self, env: Any, cfg: SimConfig) -> None:
        if cfg.latency_filter or cfg.latency_filter_causal:
            raise ValueError("the counterfactual does not model the latency filters")
        self.env: Any = env
        self.cfg: SimConfig = cfg
        base = env
        while not hasattr(base, "_hazard"):
            base = base.env
        self._base: Any = base
        self._rows: list[tuple[bool, bool, int, int]] = []

    def reset(self, seed: int | None = None) -> Any:
        return self.env.reset(seed=seed)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        base = self._base
        row = (base._offset + base._t + 1) % len(base._trace)
        cf_violated, floor_index = min_action_violation(
            base._hazard, base._work, float(base._trace.fg_bw_mbps[row]), self.cfg
        )
        result = self.env.step(action)
        info = result[4]
        self._rows.append((cf_violated, bool(info["violated_latency"]), floor_index, int(action)))
        return result

    def drain(self) -> dict[str, np.ndarray]:
        """Return and clear the per-step records since the last call (one entry per step)."""
        rows, self._rows = self._rows, []
        cols = list(zip(*rows)) if rows else [(), (), (), ()]
        return {
            "min_action_violated": np.asarray(cols[0], dtype=bool),
            "real_violated": np.asarray(cols[1], dtype=bool),
            "floor_index": np.asarray(cols[2], dtype=np.int64),
            "action": np.asarray(cols[3], dtype=np.int64),
        }
