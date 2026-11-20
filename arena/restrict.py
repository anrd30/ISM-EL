"""Restrictions applied while a submission's own code is executing.

This is a validity control before it is a security one. The evaluation workload and
the reference optimum are files on disk; an unrestricted policy can read the future
out of the trace, which is the lookahead oracle and not a deployable controller. A
policy that did so would not be cheating deliberately -- "load the workload to plan
ahead" is a reasonable thing for a model to write -- but the resulting score would be
meaningless.

Scope matters. The guards are installed once, then gated on a flag that is raised only
while the submission's module body or its ``policy`` function is on the stack. The
harness around it legitimately reads the workload CSV and hashes it, so restricting the
whole evaluation would block the arena rather than the submission. Gating on a flag also
costs a boolean test per decision rather than a context manager, which matters at
thousands of decisions per episode.

Applied inside a subprocess (see ``arena.sandbox``), so a submission that defeats these
guards still cannot reach the parent process.
"""

from __future__ import annotations

import builtins
import importlib.abc
import importlib.machinery
import sys
from contextlib import contextmanager
from typing import Any, Iterator, Sequence

__all__ = [
    "RestrictionError",
    "ALLOWED_MODULES",
    "preload_allowed",
    "install",
    "submission_code",
]


class RestrictionError(RuntimeError):
    """A submission attempted something the arena does not permit."""


#: Everything a control policy legitimately needs. A policy is arithmetic over a
#: 12-element vector; it needs no I/O, no processes and no clock.
ALLOWED_MODULES: frozenset[str] = frozenset(
    {
        "numpy", "math", "cmath", "statistics", "itertools", "functools", "operator",
        "collections", "heapq", "bisect", "array", "dataclasses", "enum", "typing",
        "abc", "numbers", "decimal", "fractions", "copy", "warnings", "re",
    }
)

#: Blocked with a specific reason. Each is a route to the trace, the reference values,
#: the network, or the host.
_DENY_REASON = {
    "os": "filesystem and process access",
    "sys": "interpreter internals",
    "io": "file access",
    "pathlib": "filesystem access",
    "shutil": "filesystem access",
    "glob": "filesystem access",
    "tempfile": "filesystem access",
    "subprocess": "process execution",
    "socket": "network access",
    "urllib": "network access",
    "requests": "network access",
    "http": "network access",
    "pandas": "reading the workload trace or the reference values",
    "pickle": "arbitrary code execution",
    "marshal": "arbitrary code execution",
    "ctypes": "native memory access",
    "multiprocessing": "process execution",
    "threading": "concurrency is not permitted in a policy",
    "importlib": "dynamic import bypasses the allowlist",
    "builtins": "rebinding builtins bypasses the allowlist",
    "sim": "simulator internals, including the workload trace",
    "arena": "the arena's reference values, including the optimum",
    "baselines": "the reference policies",
    "contracts": "simulator internals",
    "agent": "project internals",
    "data_pipeline": "the workload data",
    "hazard": "project internals",
    "workload": "the workload generator",
}

#: Raised only while submission code is on the stack. A list so the flag survives the
#: module being re-imported under a different name.
_ACTIVE: list[bool] = [False]


def _root(name: str) -> str:
    return name.partition(".")[0]


class _Blocker(importlib.abc.MetaPathFinder):
    """Refuses imports outside the allowlist, but only from submission code."""

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: object | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if not _ACTIVE[0]:
            return None
        if fullname in ALLOWED_MODULES or _root(fullname) in ALLOWED_MODULES:
            return None  # defer to the normal finders
        reason = _DENY_REASON.get(_root(fullname), "not on the arena allowlist")
        raise RestrictionError(f"import of {fullname!r} is not permitted ({reason})")


def preload_allowed() -> None:
    """Import the allowlist before any submission runs.

    Allowed modules pull in disallowed ones internally -- numpy imports ``os`` and
    ``pickle``. Once a module is in ``sys.modules`` an ``import`` of it is a cache hit
    that never reaches the finder, so pre-loading is what makes the allowlist usable
    rather than merely strict.
    """
    for name in sorted(ALLOWED_MODULES):
        try:
            __import__(name)
        except ImportError:
            pass


def install() -> None:
    """Install the guards. Idempotent; they stay inert until :func:`submission_code`."""
    if any(isinstance(f, _Blocker) for f in sys.meta_path):
        return

    sys.meta_path.insert(0, _Blocker())
    real_open, real_input = builtins.open, builtins.input
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any):
        """Check the allowlist on every import from submission code.

        The meta-path finder alone is not enough: the arena preloads numpy's
        dependencies and the simulator, so ``import os`` or
        ``from sim.scenario import load_replay_csv`` is a ``sys.modules`` cache hit
        that never reaches a finder. The import statement always calls this, so this
        is where cache hits are caught.
        """
        if _ACTIVE[0] and name not in ALLOWED_MODULES and _root(name) not in ALLOWED_MODULES:
            reason = _DENY_REASON.get(_root(name), "not on the arena allowlist")
            raise RestrictionError(f"import of {name!r} is not permitted ({reason})")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = guarded_import  # type: ignore[assignment]

    def guarded_open(*args: Any, **kwargs: Any):
        if _ACTIVE[0]:
            raise RestrictionError(
                "file access is not permitted in a submission: the workload trace and "
                "the reference optimum are files, so reading them would be lookahead"
            )
        return real_open(*args, **kwargs)

    def guarded_input(*args: Any, **kwargs: Any):
        if _ACTIVE[0]:
            raise RestrictionError("reading stdin is not permitted in a submission")
        return real_input(*args, **kwargs)

    builtins.open = guarded_open  # type: ignore[assignment]
    builtins.input = guarded_input  # type: ignore[assignment]


@contextmanager
def submission_code() -> Iterator[None]:
    """Raise the flag while the submission's own code runs.

    Nests safely: an inner block does not lower the flag on exit if an outer one set it.
    """
    previous = _ACTIVE[0]
    _ACTIVE[0] = True
    try:
        yield
    finally:
        _ACTIVE[0] = previous
