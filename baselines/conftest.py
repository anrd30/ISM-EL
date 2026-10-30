"""Pytest configuration for ``baselines/``: the ``slow`` marker.

Tests marked ``slow`` run many episodes. They are skipped unless ``--runslow``
is passed or the environment variable ``BASELINES_RUN_SLOW`` is set to a
non-empty value, so the default run stays short.
"""

from __future__ import annotations

import os

import pytest


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
