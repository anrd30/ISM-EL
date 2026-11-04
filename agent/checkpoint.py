"""Atomic checkpoint writing and reading.

A checkpoint is written to a temporary file in the destination directory,
flushed and fsynced, then moved over the destination with ``os.replace``. A
crash at any point leaves either the previous checkpoint or the new one, never
a partial file. The temporary file is removed if writing fails.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch

__all__ = ["save_checkpoint", "load_checkpoint"]


def save_checkpoint(path: str | os.PathLike[str], payload: dict[str, Any]) -> Path:
    """Write ``payload`` to ``path`` atomically and return the path."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=dest.name + ".", suffix=".tmp", dir=dest.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, dest)
    except BaseException:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise
    return dest


def load_checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a checkpoint written by :func:`save_checkpoint` (tensors and plain values only)."""
    return torch.load(Path(path), map_location="cpu", weights_only=True)
