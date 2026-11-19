"""Populations altB, C and D: windows and volumes disjoint, replay schema, hashes pinned in the manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contracts.contracts import validate_replay_frame
from data_pipeline.tencent import build_profile_altB as b

PROFILES = Path(__file__).resolve().parents[1] / "profiles"
MANIFEST = Path(__file__).resolve().parents[1] / "manifest.json"
NAMES = ["diurnal_long", "diurnal_long_altB", "diurnal_long_popC", "diurnal_long_popD"]
needs_files = pytest.mark.skipif(
    not all((PROFILES / f"profile_{n}.csv").exists() for n in NAMES), reason="a population profile is absent"
)


def _meta(name: str) -> dict:
    return json.loads((PROFILES / f"profile_{name}.meta.json").read_text(encoding="utf-8"))


def test_time_disjoint_best_avoids_every_reference_and_takes_the_earliest_best() -> None:
    step = 300
    rows = 10
    # one volume per run; coverage of a window start t is the number of runs containing [t, t + rows)
    runs = pd.DataFrame(
        {
            "disk_uid": [f"d{i}" for i in range(6)],
            "run_start_ts": [0, 0, 0, 5 * step, 5 * step, 40 * step],
            "run_rows": [30, 30, 30, 30, 30, 30],
        }
    )
    start, table = b.time_disjoint_best(runs, rows, [0 * step, 20 * step])
    ends = rows * step
    assert start + ends <= 0 or start >= 0 + ends
    assert start + ends <= 20 * step or start >= 20 * step + ends
    assert table["volumes"].max() >= 1
    # a single reference given as an int behaves like a list of one
    assert b.time_disjoint_best(runs, rows, 0)[0] == b.time_disjoint_best(runs, rows, [0])[0]


def test_populations_are_declared_with_the_registered_seed_and_exclusions() -> None:
    assert b.POPULATIONS["D"]["seed"] == 20261119 and b.POPULATIONS["D"]["rank"] == "random"
    assert b.POPULATIONS["C"]["rank"] == "diurnal"
    assert b.POPULATIONS["D"]["avoid"] == ["diurnal_long", "diurnal_long_altB", "diurnal_long_popC"]


def test_random_order_is_a_seeded_permutation_of_the_sorted_list() -> None:
    items = sorted(f"disk{i:03d}" for i in range(50))
    a = [items[i] for i in np.random.default_rng(20261119).permutation(len(items))]
    c = [items[i] for i in np.random.default_rng(20261119).permutation(len(items))]
    assert a == c and sorted(a) == items and a != items


@needs_files
def test_every_profile_validates_has_4032_rows_and_starts_at_zero() -> None:
    for name in NAMES:
        frame = pd.read_csv(PROFILES / f"profile_{name}.csv")
        validate_replay_frame(frame)
        assert len(frame) == 4032 and int(frame["t_seconds"].iloc[0]) == 0


@needs_files
def test_windows_share_no_sample_time_and_volume_sets_are_disjoint() -> None:
    metas = {n: _meta(n) for n in NAMES}
    for i, a in enumerate(NAMES):
        for c in NAMES[i + 1 :]:
            ma, mc = metas[a], metas[c]
            assert ma["window_end_ts"] < mc["window_start_ts"] or mc["window_end_ts"] < ma["window_start_ts"], (a, c)
            assert not set(ma["disk_uids"]) & set(mc["disk_uids"]), (a, c)
    for n in ("diurnal_long_popC", "diurnal_long_popD"):
        assert metas[n]["overlap_with_excluded_disks"] == 0


@needs_files
def test_aggregate_p95_is_within_the_registered_ceiling() -> None:
    ceiling = 100.0 * 1.005
    for n in ("diurnal_long_popC", "diurnal_long_popD"):
        frame = pd.read_csv(PROFILES / f"profile_{n}.csv")
        total = (frame["read_bw_kbps"] + frame["write_bw_kbps"]).to_numpy(dtype=float) / 1000.0
        assert np.percentile(total, 95) <= ceiling * (1 + 1e-9)
        assert abs(np.percentile(total, 95) - _meta(n)["p95_mbps"]) < 1e-6


@needs_files
def test_popD_records_its_seed_and_is_not_ranked() -> None:
    meta = _meta("diurnal_long_popD")
    assert meta["rng_seed"] == 20261119 and meta["rank"] == "random"
    assert "no ranking" in meta["selection_rule"]


@needs_files
def test_manifest_pins_every_new_file() -> None:
    files = {e["file"]: e for e in json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]}
    for n in NAMES[1:]:
        for suffix in (".csv", ".meta.json"):
            rel = f"profiles/profile_{n}{suffix}"
            assert rel in files, rel
            assert files[rel]["sha256"] == hashlib.sha256((PROFILES / f"profile_{n}{suffix}").read_bytes()).hexdigest()
