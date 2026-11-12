"""The time-forward split of the long diurnal profile."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from contracts.contracts import CONTROL_INTERVAL_S, validate_replay_frame
from data_pipeline.tencent import split_long_profile as sp
from sim.scenario import load_replay_csv

SOURCE = sp.PROFILES / f"{sp.SOURCE_STEM}.csv"
_SOURCE_PRESENT = SOURCE.exists()
needs_source = pytest.mark.skipif(not _SOURCE_PRESENT, reason=f"{SOURCE.name} absent")

HALVES = ("train", "holdout")


@needs_source
def test_halves_partition_the_source_exactly() -> None:
    parent = pd.read_csv(SOURCE)
    cut = json.loads(SOURCE.with_suffix(".meta.json").read_text(encoding="utf-8"))["suggested_split"][
        "first_holdout_row"
    ]
    frames = {h: pd.read_csv(sp.PROFILES / f"{sp.SOURCE_STEM}_{h}.csv") for h in HALVES}

    assert len(frames["train"]) == cut
    assert len(frames["train"]) + len(frames["holdout"]) == len(parent)

    # Every column but the rebased clock is carried through untouched, in order.
    payload = [c for c in parent.columns if c != "t_seconds"]
    rejoined = pd.concat([frames["train"], frames["holdout"]], ignore_index=True)
    pd.testing.assert_frame_equal(rejoined[payload], parent[payload])


@needs_source
@pytest.mark.parametrize("half", HALVES)
def test_each_half_is_a_valid_replay_file_on_its_own(half: str) -> None:
    frame = pd.read_csv(sp.PROFILES / f"{sp.SOURCE_STEM}_{half}.csv")
    validate_replay_frame(frame)  # raises ContractViolation if not
    assert frame["t_seconds"].iloc[0] == 0, "each half carries its own episode-relative clock"
    assert frame["t_seconds"].iloc[-1] == (len(frame) - 1) * CONTROL_INTERVAL_S


@needs_source
@pytest.mark.parametrize("half", HALVES)
def test_sidecar_keeps_provenance_and_records_the_absolute_window(half: str) -> None:
    parent = json.loads(SOURCE.with_suffix(".meta.json").read_text(encoding="utf-8"))
    meta = json.loads((sp.PROFILES / f"{sp.SOURCE_STEM}_{half}.meta.json").read_text(encoding="utf-8"))

    assert meta["profile"] == f"{parent['profile']}_{half}"
    assert meta["synthetic"] is False, "a real trace must not be relabelled synthetic by being cut"
    assert meta["disclosure"] == parent["disclosure"]
    assert meta["disk_uids"] == parent["disk_uids"]

    start, end = meta["parent_row_range"]
    assert meta["rows"] == end - start
    # Rebasing loses the absolute clock from the CSV, so the sidecar has to carry it.
    assert meta["window_start_ts"] == parent["window_start_ts"] + start * CONTROL_INTERVAL_S


@needs_source
@pytest.mark.parametrize("half", HALVES)
def test_the_simulator_loads_each_half_and_keeps_its_label(half: str) -> None:
    trace = load_replay_csv(sp.PROFILES / f"{sp.SOURCE_STEM}_{half}.csv")
    assert trace.profile == f"diurnal_long_{half}"
    assert trace.synthetic is False
    assert len(trace.fg_bw_mbps) == len(pd.read_csv(sp.PROFILES / f"{sp.SOURCE_STEM}_{half}.csv"))


@needs_source
def test_the_two_halves_do_not_overlap_in_absolute_time() -> None:
    metas = {
        h: json.loads((sp.PROFILES / f"{sp.SOURCE_STEM}_{h}.meta.json").read_text(encoding="utf-8"))
        for h in HALVES
    }
    assert metas["train"]["window_end_ts"] < metas["holdout"]["window_start_ts"], (
        "a time-forward split is only held out if the evaluation half starts after the training half ends"
    )


@needs_source
def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    before = {h: (sp.PROFILES / f"{sp.SOURCE_STEM}_{h}.csv").stat().st_mtime_ns for h in HALVES}
    sp.split(write=False)
    after = {h: (sp.PROFILES / f"{sp.SOURCE_STEM}_{h}.csv").stat().st_mtime_ns for h in HALVES}
    assert before == after


@needs_source
def test_regenerating_is_byte_identical(tmp_path: Path) -> None:
    """The split is a pure function of the source: running it on a copy reproduces the tracked files byte for byte.

    The tracked files are only read; the split writes next to the copied source in ``tmp_path``.
    """
    sidecar = SOURCE.with_suffix(".meta.json")
    shutil.copyfile(SOURCE, tmp_path / SOURCE.name)
    shutil.copyfile(sidecar, tmp_path / sidecar.name)
    written = sp.split(tmp_path / SOURCE.name)
    for half in HALVES:
        tracked = sp.PROFILES / f"{sp.SOURCE_STEM}_{half}.csv"
        assert written[half].parent == tmp_path
        assert written[half].read_bytes() == tracked.read_bytes()
        assert written[half].with_suffix(".meta.json").read_bytes() == tracked.with_suffix(".meta.json").read_bytes()
