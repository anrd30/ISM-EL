"""Keys and model ids for the frontier batch, read from ``arena/.env``.

``arena/.env`` is the only place API keys and model ids are written. Importing this
module loads it, over the defaults of the committed ``arena/.env.example``; a variable
already set in the process environment wins over both. Empty values are ignored, so an
unfilled key line is "not set".

Naming: ``ARENA_MODEL_<PROVIDER>_<TIER>`` holds an OpenRouter model slug (``<vendor>/<model>``);
the one key is ``OPENROUTER_API_KEY`` (``key_for("openrouter")``).
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import dotenv_values

__all__ = ["ENV_FILE", "EXAMPLE_FILE", "load_env", "roster", "batch_roster", "model_id", "provider_of", "key_for",
           "API_PROVIDER", "DEFERRED_PROVIDERS", "pinned_provider", "display_label"]

#: Every roster model is called through this provider, by its slug.
API_PROVIDER = "openrouter"

#: Roster providers left out of the current batch.
DEFERRED_PROVIDERS = ("anthropic",)

ENV_FILE = Path(__file__).resolve().parent / ".env"
EXAMPLE_FILE = Path(__file__).resolve().parent / ".env.example"

_MODEL_PREFIX = "ARENA_MODEL_"


def load_env() -> dict[str, str]:
    """Merge ``.env.example`` and ``.env`` into ``os.environ`` (process values win) and return the merge."""
    merged: dict[str, str] = {}
    for path in (EXAMPLE_FILE, ENV_FILE):
        if path.is_file():
            merged.update({k: v for k, v in dotenv_values(path).items() if v})
    for name, value in merged.items():
        os.environ.setdefault(name, value)
    return merged


load_env()


def roster() -> dict[str, str]:
    """Roster variable -> model id, in the order of ``.env.example``."""
    names = [k for k in dotenv_values(EXAMPLE_FILE) if k.startswith(_MODEL_PREFIX)]
    missing = [n for n in names if not os.environ.get(n)]
    if missing:
        raise RuntimeError(f"model id variable not set: {', '.join(missing)}")
    return {n: os.environ[n] for n in names}


def batch_roster() -> dict[str, str]:
    """``roster()`` without the deferred providers: the models of the current batch."""
    return {v: m for v, m in roster().items() if provider_of(v) not in DEFERRED_PROVIDERS}


def model_id(var: str) -> str:
    """The model id held by roster variable ``var``."""
    return roster()[var]


def provider_of(var: str) -> str:
    """Provider name of a roster variable (``ARENA_MODEL_<PROVIDER>_<TIER>``)."""
    return var[len(_MODEL_PREFIX):].split("_", 1)[0].lower()


def key_for(provider: str) -> str:
    """The API key of ``provider``.

    Raises:
        RuntimeError: naming the variable, if it is not set.
    """
    name = f"{provider.upper()}_API_KEY"
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"environment variable {name} is not set; fill it in {ENV_FILE}")
    return value


def _var_of(slug: str) -> str | None:
    return next((v for v, m in roster().items() if m == slug), None)


def pinned_provider(slug: str) -> str | None:
    """OpenRouter provider the model ``slug`` is pinned to (``ARENA_PROVIDER_<TIER>``), or None."""
    var = _var_of(slug)
    if var is None:
        return None
    return os.environ.get("ARENA_PROVIDER_" + var[len(_MODEL_PREFIX):]) or None


def display_label(slug: str) -> str:
    """Report name of the model ``slug`` (``ARENA_LABEL_<TIER>``), else the slug."""
    var = _var_of(slug)
    label = os.environ.get("ARENA_LABEL_" + var[len(_MODEL_PREFIX):]) if var else None
    return label or slug
