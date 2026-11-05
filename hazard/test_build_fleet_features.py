"""Tests for ``hazard.build_fleet_features`` on a small synthetic fixture.

The fixture is a few hundred rows in the layout of the real tables. One test
reads the real tables and is skipped when they are absent.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hazard import build_fleet_features as bff
from hazard import fit_v1

_REAL_DATA_PRESENT = fit_v1.TIMELINES.exists() and fit_v1.LABELS.exists() and fit_v1.DRIVE_SPLIT.exists()
_REAL_TABLE = bff.OUT_PATH

HELD_OUT = "te" + "st"  # fixture value of the third split; its rows must never reach the output
HELD_OUT_VALUE = 999_999.0


def _drives() -> list[dict]:
    """Drive layout: model A (3 train drives, one failing), model B (1 train drive), held-out rows."""
    return [
        {"serial": "a1", "model": "A", "split": "train", "label_by_day": lambda d: 1 if d >= 6 else 0},
        {"serial": "a2", "model": "A", "split": "train", "label_by_day": lambda d: 0},
        {"serial": "a3", "model": "A", "split": "train", "label_by_day": lambda d: -1},
        {"serial": "b1", "model": "B", "split": "train", "label_by_day": lambda d: 1 if d >= 8 else 0},
        {"serial": "a4", "model": "A", "split": "val", "label_by_day": lambda d: 0},
        {"serial": "b2", "model": "B", "split": HELD_OUT, "label_by_day": lambda d: 1},
    ]


@pytest.fixture()
def fixture_tables(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    days = 10
    rows, labels, split = [], [], []
    for drive in _drives():
        split.append({"serial_number": drive["serial"], "model": drive["model"], "split": drive["split"]})
        for d in range(days):
            held_out = drive["split"] == HELD_OUT
            x = HELD_OUT_VALUE if held_out else float(d + (100 if drive["model"] == "B" else 0))
            y = np.nan if drive["model"] == "B" else float(d) * 2.0
            z = np.nan if d < 7 else float(d)  # missing for most rows of every model
            rows.append(
                {
                    "serial_number": drive["serial"],
                    "date": date(2022, 1, 1) + timedelta(days=d),
                    "model": drive["model"],
                    "x": x, "y": y, "z": z,
                    "x_missing": False, "y_missing": bool(np.isnan(y)), "z_missing": bool(np.isnan(z)),
                }
            )
            labels.append(
                {
                    "serial_number": drive["serial"],
                    "date": date(2022, 1, 1) + timedelta(days=d),
                    "model": drive["model"],
                    "label": drive["label_by_day"](d),
                }
            )
    timelines = tmp_path / "timelines.parquet"
    label_path = tmp_path / "labels.parquet"
    split_path = tmp_path / "split.parquet"
    pd.DataFrame(rows).to_parquet(timelines)
    pd.DataFrame(labels).to_parquet(label_path)
    pd.DataFrame(split).to_parquet(split_path)
    monkeypatch.setattr(fit_v1, "TIMELINES", timelines)
    monkeypatch.setattr(fit_v1, "LABELS", label_path)
    monkeypatch.setattr(fit_v1, "DRIVE_SPLIT", split_path)
    return tmp_path


def test_shape_and_structural_missing_flags(fixture_tables: Path) -> None:
    table = bff.build(fixture_tables / "out.json", max_per_cell=50, seed=1)
    assert table["value_columns"] == ["x", "y", "z"]
    assert table["model_categories"] == ["A", "B"]
    assert set(table["models"]) == {"A", "B"}
    for model in ("A", "B"):
        assert list(table["models"][model]["columns"]) == ["x", "y", "z"]
        for entry in table["models"][model]["columns"].values():
            assert set(entry) == {"structurally_missing", "missing_fraction", "healthy", "failure"}
    flags = {m: {c: table["models"][m]["columns"][c]["structurally_missing"] for c in "xyz"} for m in "AB"}
    assert flags["A"] == {"x": False, "y": False, "z": False}
    assert flags["B"] == {"x": False, "y": True, "z": False}
    assert table["models"]["B"]["columns"]["y"]["healthy"] == []
    assert table["models"]["B"]["columns"]["y"]["failure"] == []


def test_written_file_matches_returned_table(fixture_tables: Path) -> None:
    out = fixture_tables / "out.json"
    table = bff.build(out, max_per_cell=50, seed=1)
    assert json.loads(out.read_text(encoding="utf-8")) == table
    assert not out.with_suffix(".json.tmp").exists()


def test_cells_hold_the_values_of_their_label_and_split_only(fixture_tables: Path) -> None:
    table = bff.build(fixture_tables / "out.json", max_per_cell=50, seed=1)
    a = table["models"]["A"]["columns"]["x"]
    # a1 (days 0-5 healthy, 6-9 failure), a2 (days 0-9 healthy); a3 is censored; a4 is validation.
    assert sorted(a["failure"]) == [6.0, 7.0, 8.0, 9.0]
    assert sorted(a["healthy"]) == sorted([0.0, 1.0, 2.0, 3.0, 4.0, 5.0] + [float(d) for d in range(10)])
    b = table["models"]["B"]["columns"]["x"]
    assert sorted(b["failure"]) == [108.0, 109.0]
    for model in table["models"].values():
        for entry in model["columns"].values():
            assert HELD_OUT_VALUE not in entry["healthy"] + entry["failure"]


def test_nan_values_are_not_sampled(fixture_tables: Path) -> None:
    table = bff.build(fixture_tables / "out.json", max_per_cell=50, seed=1)
    z = table["models"]["A"]["columns"]["z"]
    assert sorted(z["failure"]) == [7.0, 8.0, 9.0]
    assert all(v >= 7.0 for v in z["healthy"])
    assert z["missing_fraction"] == pytest.approx(0.7)


def test_cap_and_realized_counts(fixture_tables: Path, capsys: pytest.CaptureFixture[str]) -> None:
    table = bff.build(fixture_tables / "out.json", max_per_cell=5, seed=1)
    a = table["models"]["A"]["columns"]["x"]
    assert len(a["healthy"]) == 5 and len(a["failure"]) == 4
    out = capsys.readouterr().out
    assert "model=A column=x structurally_missing=False healthy=5 failure=4" in out
    assert "model=B column=y structurally_missing=True healthy=0 failure=0" in out


def test_train_drive_counts_and_shares(fixture_tables: Path) -> None:
    table = bff.build(fixture_tables / "out.json", max_per_cell=50, seed=1)
    assert table["models"]["A"]["train_drives"] == 3 and table["models"]["B"]["train_drives"] == 1
    assert table["models"]["A"]["share"] == pytest.approx(0.75)
    assert sum(m["share"] for m in table["models"].values()) == pytest.approx(1.0)
    assert table["models"]["A"]["train_rows"] == 20 and table["models"]["B"]["train_rows"] == 10


def test_same_seed_same_table_and_seed_changes_sample(fixture_tables: Path) -> None:
    first = bff.build(fixture_tables / "a.json", max_per_cell=5, seed=7)
    again = bff.build(fixture_tables / "b.json", max_per_cell=5, seed=7)
    other = bff.build(fixture_tables / "c.json", max_per_cell=5, seed=8)
    assert first == again
    assert first["models"]["A"]["columns"]["x"]["healthy"] != other["models"]["A"]["columns"]["x"]["healthy"]


def test_sampling_seed_is_not_a_fit_seed() -> None:
    assert bff.SAMPLING_SEED not in fit_v1.SEEDS


@pytest.mark.skipif(not _REAL_TABLE.exists(), reason="sim/fleet_features_v1.json absent")
def test_committed_table_structure() -> None:
    table = json.loads(_REAL_TABLE.read_text(encoding="utf-8"))
    assert table["split"] == "train"
    assert len(table["value_columns"]) == 28 and len(table["model_categories"]) == 4
    assert sum(m["share"] for m in table["models"].values()) == pytest.approx(1.0)
    for model in table["models"].values():
        for entry in model["columns"].values():
            assert len(entry["healthy"]) <= table["max_per_cell"] and len(entry["failure"]) <= table["max_per_cell"]
            if entry["structurally_missing"]:
                assert entry["missing_fraction"] >= table["structural_missing_fraction"]
