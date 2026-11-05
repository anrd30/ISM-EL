"""Tests for ``sim.measure_defaults`` with the v0 configuration, a few seeds and two profiles."""

from __future__ import annotations

from contracts.contracts import ACTIONS_MBPS
from sim.config import v0_config
from sim.measure_defaults import format_tables, main, measure


def test_measure_shape_and_idle_never_violates() -> None:
    result = measure(v0_config(), seeds=[0, 1], profiles=["idle", "peak"])
    assert set(result) == {"idle", "peak"}
    for entry in result.values():
        assert set(entry) == {"mean", "worst", "clipped"}
        assert all(len(v) == len(ACTIONS_MBPS) for v in entry.values())
        assert all(w >= m for m, w in zip(entry["mean"], entry["worst"]))
    assert result["idle"]["mean"] == [0.0] * len(ACTIONS_MBPS)
    assert result["peak"]["mean"][-1] > result["peak"]["mean"][0]


def test_format_tables_has_one_row_per_profile() -> None:
    result = measure(v0_config(), seeds=[0], profiles=["idle"])
    text = format_tables(result)
    assert "| idle | 0.000 | 0.000 | 0.000 | 0.000 | 4 |" in text
    assert "worst seed:" in text and "clipped by the filter:" in text


def test_cli_runs_with_v0(capsys) -> None:
    assert main(["--hazard", "v0", "--seeds", "1", "--profiles", "idle"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("hazard=v0 seeds=1 profiles=['idle']")
    assert "| idle |" in out
