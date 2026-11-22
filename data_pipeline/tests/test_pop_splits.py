"""Train and holdout splits of populations C and D: rows, clock, continuity, sidecars and manifest hashes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from contracts.contracts import validate_replay_frame
from data_pipeline.tencent import split_long_profile as sp

PROFILES = Path(__file__).resolve().parents[1] / "profiles"
MANIFEST = Path(__file__).resolve().parents[1] / "manifest.json"
POPS = ("popC", "popD")
needs_files = pytest.mark.skipif(
    not all((PROFILES / f"profile_diurnal_long_{p}{s}.csv").exists() for p in POPS for s in ("", "_train", "_holdout")),
    reason="a split profile is absent",
)


@needs_files
@pytest.mark.parametrize("pop", POPS)
def test_halves_are_the_first_ten_and_last_four_days_of_the_parent(pop: str) -> None:
    parent = pd.read_csv(PROFILES / f"profile_diurnal_long_{pop}.csv")
    train = pd.read_csv(PROFILES / f"profile_diurnal_long_{pop}_train.csv")
    holdout = pd.read_csv(PROFILES / f"profile_diurnal_long_{pop}_holdout.csv")
    assert (len(train), len(holdout)) == (2880, 1152) == (10 * 288, 4 * 288)
    for half in (train, holdout):
        validate_replay_frame(half)
        assert int(half["t_seconds"].iloc[0]) == 0
    cols = [c for c in parent.columns if c != "t_seconds"]
    assert train[cols].equals(parent.iloc[:2880][cols].reset_index(drop=True))
    assert holdout[cols].equals(parent.iloc[2880:][cols].reset_index(drop=True))


@needs_files
@pytest.mark.parametrize("pop", POPS)
def test_sidecars_name_the_parent_row_range_and_window(pop: str) -> None:
    parent = json.loads((PROFILES / f"profile_diurnal_long_{pop}.meta.json").read_text(encoding="utf-8"))
    train = json.loads((PROFILES / f"profile_diurnal_long_{pop}_train.meta.json").read_text(encoding="utf-8"))
    holdout = json.loads((PROFILES / f"profile_diurnal_long_{pop}_holdout.meta.json").read_text(encoding="utf-8"))
    assert train["parent_row_range"] == [0, 2880] and holdout["parent_row_range"] == [2880, 4032]
    assert train["window_start_ts"] == parent["window_start_ts"]
    assert holdout["window_end_ts"] == parent["window_end_ts"]
    assert holdout["window_start_ts"] == train["window_end_ts"] + 300
    assert train["profile"] == f"diurnal_long_{pop}_train"


@needs_files
def test_manifest_pins_the_split_files() -> None:
    files = {e["file"]: e for e in json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]}
    for pop in POPS:
        for half in ("train", "holdout"):
            for suffix in (".csv", ".meta.json"):
                rel = f"profiles/profile_diurnal_long_{pop}_{half}{suffix}"
                assert rel in files, rel
                assert files[rel]["sha256"] == hashlib.sha256((PROFILES / Path(rel).name).read_bytes()).hexdigest()


@needs_files
def test_the_splitter_regenerates_the_committed_halves_byte_for_byte(tmp_path: Path) -> None:
    for pop in POPS:
        for name in (f"profile_diurnal_long_{pop}.csv", f"profile_diurnal_long_{pop}.meta.json"):
            (tmp_path / name).write_bytes((PROFILES / name).read_bytes())
        paths = sp.split(tmp_path / f"profile_diurnal_long_{pop}.csv")
        for half, path in paths.items():
            assert path.name == f"profile_diurnal_long_{pop}_{half}.csv"
            assert path.read_bytes() == (PROFILES / path.name).read_bytes()
            side = path.with_suffix(".meta.json")
            assert side.read_bytes() == (PROFILES / side.name).read_bytes()
