"""Hazard-input ablation.

The falsification test for the risk model is to remove the hazard input,
retrain, and check whether performance changes. This module provides the
removal as a single configuration flag so that the two arms differ in nothing
else: the same training code, the same network shapes (the input width stays
12, with the masked entries held at zero), the same environment.

Observation fields 5 and 6 (``hazard`` and ``hazard_uncertainty``) are set to
zero before the policy sees them. The environment, its safety filter, its
``info`` record and its reward are untouched, and the wrapper returns the
unmodified ``info``.

Expected sensitivity. The safety filter currently raises any unsafe action to
the smallest feasible one regardless of what the policy observes, so a policy
that cannot see the hazard still has its reliability budget enforced by the
filter. This ablation is therefore expected to show little difference until
that changes. The mask is built correctly regardless, and an ablated agent must
be evaluated with the same mask (``LearnedPolicy`` applies it from the
agent's own configuration).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from contracts.contracts import OBS_DIM, OBS_FIELDS

__all__ = ["HAZARD_FIELDS", "HAZARD_INDICES", "mask_observation", "AblatedEnv"]

#: Names of the observation fields removed by the ablation.
HAZARD_FIELDS: tuple[str, ...] = ("hazard", "hazard_uncertainty")
#: Indices of those fields in the observation vector (5 and 6).
HAZARD_INDICES: tuple[int, ...] = tuple(OBS_FIELDS.index(name) for name in HAZARD_FIELDS)


def mask_observation(obs: np.ndarray, enabled: bool = True) -> np.ndarray:
    """Return a copy of ``obs`` with fields 5 and 6 zeroed.

    With ``enabled=False`` the observation is returned as an unmodified copy,
    so callers can apply the function unconditionally and drive the ablation
    from one flag. Works on a single vector ``(12,)`` or a batch ``(..., 12)``.

    Raises:
        ValueError: if the last axis is not ``OBS_DIM`` long.
    """
    arr = np.array(obs, dtype=np.float32, copy=True)
    if arr.ndim < 1 or arr.shape[-1] != OBS_DIM:
        raise ValueError(f"observation must have last axis {OBS_DIM}, got shape {arr.shape}")
    if enabled:
        arr[..., list(HAZARD_INDICES)] = 0.0
    return arr


class AblatedEnv:
    """Wrap an environment so the policy sees the masked observation.

    Only the observation returned by ``reset`` and ``step`` changes. Every
    other return value, and every attribute not defined here, is passed
    through to the wrapped environment unchanged.

    Args:
        env: A ``RecoveryEnv`` (or any object with its ``reset`` / ``step``).
        enabled: When false the wrapper is a pass-through, which lets one code
            path serve both arms of the comparison.
    """

    def __init__(self, env: Any, enabled: bool = True) -> None:
        self.env: Any = env
        self.enabled: bool = bool(enabled)

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed)
        return mask_observation(obs, self.enabled), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return mask_observation(obs, self.enabled), reward, terminated, truncated, info

    def __getattr__(self, name: str) -> Any:
        # Called only for attributes not found on the wrapper itself.
        return getattr(self.env, name)
