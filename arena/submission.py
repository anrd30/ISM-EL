"""Load a submitted ``policy`` function and expose it through the harness Policy protocol.

A submission is one Python file defining::

    def policy(obs: np.ndarray, state: dict) -> int

It must be deterministic, must return an index into ``ACTIONS_MBPS``, and must keep
within the per-step time budget. ``state`` is a dict the policy may mutate to carry
memory within an episode; the arena clears it on every reset, so episodes are
independent.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from arena.config import STEP_BUDGET_S
from contracts.contracts import N_ACTIONS, OBS_DIM

__all__ = ["SubmissionError", "load_policy_fn", "SubmittedPolicy"]


class SubmissionError(RuntimeError):
    """The submission is unusable: bad signature, bad return, or over budget."""


def load_policy_fn(path: str | Path) -> Callable[[np.ndarray, dict], int]:
    """Import ``policy`` from a submission file.

    Raises:
        SubmissionError: if the file cannot be imported or defines no ``policy``.
    """
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"submission_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise SubmissionError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - any import-time failure is a bad submission
        raise SubmissionError(f"{path.name} raised on import: {exc!r}") from exc
    fn = getattr(module, "policy", None)
    if not callable(fn):
        raise SubmissionError(f"{path.name} defines no callable named 'policy'")
    return fn


class SubmittedPolicy:
    """Adapter from a submitted function to the evaluation harness.

    Deliberately has no ``config`` attribute: the harness reads ``policy.config``
    to compare latency models and would misinterpret anything we put there.
    """

    def __init__(self, fn: Callable[[np.ndarray, dict], int], name: str) -> None:
        self._fn = fn
        self.name: str = name
        self._state: dict[str, Any] = {}
        self.n_calls: int = 0
        self.max_step_s: float = 0.0
        self.total_s: float = 0.0

    def reset(self) -> None:
        """Clear per-episode memory. Episodes are independent by construction."""
        self._state = {}

    def __call__(self, obs: np.ndarray) -> int:
        arr = np.asarray(obs, dtype=np.float32)
        if arr.shape != (OBS_DIM,):
            raise SubmissionError(f"observation must be ({OBS_DIM},), got {arr.shape}")

        start = time.perf_counter()
        try:
            action = self._fn(arr.copy(), self._state)
        except Exception as exc:  # noqa: BLE001 - a raising policy is a failed submission
            raise SubmissionError(f"policy raised: {exc!r}") from exc
        elapsed = time.perf_counter() - start

        self.n_calls += 1
        self.total_s += elapsed
        self.max_step_s = max(self.max_step_s, elapsed)
        if elapsed > STEP_BUDGET_S:
            raise SubmissionError(
                f"policy took {elapsed * 1e3:.1f} ms, budget is {STEP_BUDGET_S * 1e3:.0f} ms"
            )

        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise SubmissionError(f"policy must return an int action index, got {type(action)}")
        action = int(action)
        if not 0 <= action < N_ACTIONS:
            raise SubmissionError(f"action {action} outside [0, {N_ACTIONS})")
        return action
