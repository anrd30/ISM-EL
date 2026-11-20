"""Run a submission in an isolated subprocess.

Two layers. This module provides process isolation, a wall-clock cap and a stripped
environment; ``arena.restrict`` provides the import allowlist and blocks file access
inside that process. Either alone is insufficient: restrictions can in principle be
escaped, and a subprocess alone would still let a policy read the workload trace.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from arena.config import REPO, SUBMISSION_BUDGET_S

__all__ = ["run_sandboxed"]

#: Environment variables passed through. Everything else is dropped, so a submission
#: cannot read credentials or paths from the host environment.
_KEEP = ("PATH", "SYSTEMROOT", "LANG", "LC_ALL", "TMPDIR", "TEMP", "TMP")


def run_sandboxed(
    submission: str | Path,
    profile: str,
    name: str,
    *,
    timeout_s: float = SUBMISSION_BUDGET_S,
    memory_mb: int = 4096,
) -> dict:
    """Evaluate ``submission`` on ``profile`` in a subprocess.

    Returns the worker's result dict, or ``{"error": ...}``. Never raises for a
    failing submission: a crash, a timeout or a restriction violation is a result.
    """
    request = {
        "submission": str(Path(submission).resolve()),
        "profile": profile,
        "name": name,
        "memory_mb": memory_mb,
        "cpu_seconds": int(timeout_s),
        # The worker runs under -I, which implies -s and -E: no user site-packages and
        # no PYTHONPATH. On a shared machine the arena's own dependencies usually live
        # in ~/.local, so the parent's resolved path is handed over explicitly. This
        # costs nothing in isolation -- the controls that matter are the separate
        # process and the restrictions around submission code, not path discovery.
        "sys_path": [entry for entry in sys.path if entry],
    }
    env = {"PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1"}
    import os

    for key in _KEEP:
        if key in os.environ:
            env[key] = os.environ[key]

    try:
        completed = subprocess.run(
            # -I isolates the interpreter: no PYTHONPATH, no user site-packages, no
            # inherited -X options. The worker puts the repo on sys.path itself, so the
            # module is addressed by file rather than by -m.
            [sys.executable, "-I", str(Path(__file__).resolve().parent / "_worker.py")],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(REPO),
            env=env,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"timeout: exceeded {timeout_s:.0f}s"}

    if completed.returncode != 0 and not completed.stdout.strip():
        tail = (completed.stderr or "").strip().splitlines()[-3:]
        return {"error": f"worker exited {completed.returncode}: {' | '.join(tail)}"}

    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        tail = (completed.stdout or "")[-300:]
        return {"error": f"worker produced no usable result: {tail!r}"}
