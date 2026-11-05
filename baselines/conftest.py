"""Pytest configuration for ``baselines/``: the ``slow`` marker and the v0 pin.

Tests marked ``slow`` run many episodes. They are skipped unless ``--runslow``
is passed or the environment variable ``BASELINES_RUN_SLOW`` is set to a
non-empty value, so the default run stays short.

The baseline tests were written against the heuristic v0 hazard. The
``SimConfig`` default is now the fitted v1, so the autouse fixture below
replaces the ``SimConfig`` name used for default configurations in
``baselines.evaluate``, ``baselines.policies`` and ``sim.recovery_env`` with
``sim.config.v0_config``, which pairs the v0 hazard with its feature function.
Tests that pass a configuration explicitly are unaffected.
"""

from __future__ import annotations

import os

import pytest

import baselines.evaluate
import baselines.policies
import sim.recovery_env
from sim.config import v0_config


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--runslow", action="store_true", default=False, help="also run tests marked slow")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: runs many episodes; skipped unless --runslow or BASELINES_RUN_SLOW is set")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--runslow", default=False) or os.environ.get("BASELINES_RUN_SLOW"):
        return
    skip = pytest.mark.skip(reason="slow test; pass --runslow or set BASELINES_RUN_SLOW")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _default_config_is_v0(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (baselines.evaluate, baselines.policies, sim.recovery_env):
        monkeypatch.setattr(module, "SimConfig", v0_config)
