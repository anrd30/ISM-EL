"""Sliding-window folds of the long diurnal profile."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from contracts.contracts import validate_replay_frame
from data_pipeline.tencent import sliding_rolling_folds as sf
from data_pipeline.tencent.extract_load import DAY_ROWS
from data_pipeline.tencent.split_long_profile import PROFILES, SOURCE_STEM

SOURCE = PROFILES / f"{SOURCE_STEM}.csv"
needs_source = pytest.mark.skipif(not SOURCE.exists(), reason=f"{SOURCE.name} absent")


@pytest.fixture(scope="module")
def folds(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return sf.make_folds(SOURCE, tmp_path_factory.mktemp("sfolds"))


@needs_source
def test_piece_sizes_and_origin(folds: dict[str, Path]) -> None:
    for fold in range(1, sf.N_FOLDS + 1):
        train = pd.read_csv(folds[f"sfold{fold}_train"])
        holdout = pd.read_csv(folds[f"sfold{fold}_holdout"])
        assert len(train) == 2304 == sf.TRAIN_DAYS * DAY_ROWS
        assert len(holdout) == 1152 == sf.HOLDOUT_DAYS * DAY_ROWS
        assert int(train["t_seconds"].iloc[0]) == 0
        assert int(holdout["t_seconds"].iloc[0]) == 0


@needs_source
def test_holdout_piece_follows_its_train_piece(folds: dict[str, Path]) -> None:
    source = pd.read_csv(SOURCE)
    for fold in range(1, sf.N_FOLDS + 1):
        bounds = sf.fold_bounds(fold)
        assert bounds["train"][1] == bounds["holdout"][0]
        for piece in ("train", "holdout"):
            start, end = bounds[piece]
            expected = source.iloc[start:end].drop(columns="t_seconds").reset_index(drop=True)
            actual = pd.read_csv(folds[f"sfold{fold}_{piece}"]).drop(columns="t_seconds")
            pd.testing.assert_frame_equal(actual, expected)


@needs_source
def test_fold3_holdout_is_byte_identical_to_committed_holdout(folds: dict[str, Path]) -> None:
    committed = PROFILES / f"{SOURCE_STEM}_holdout.csv"
    if not committed.exists():
        pytest.skip(f"{committed.name} absent")
    assert folds["sfold3_holdout"].read_bytes() == committed.read_bytes()


@needs_source
def test_every_piece_validates_and_has_a_sidecar(folds: dict[str, Path]) -> None:
    assert len(folds) == 2 * sf.N_FOLDS
    for key, path in folds.items():
        validate_replay_frame(pd.read_csv(path))
        meta = json.loads(path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        assert meta["fold"] == int(key[5])
        assert meta["fold_piece"] == key.split("_", 1)[1]
        assert b"\r" not in path.read_bytes()


@needs_source
def test_train_pieces_slide_and_are_not_nested(folds: dict[str, Path]) -> None:
    frames = {k: pd.read_csv(folds[f"sfold{k}_train"]).drop(columns="t_seconds") for k in (1, 2, 3)}
    mid = frames[2]
    for other in (1, 3):
        other_frame = frames[other]
        assert len(mid) == len(other_frame)
        assert not mid.equals(other_frame)
        assert not mid.iloc[: len(other_frame)].reset_index(drop=True).equals(other_frame)
        assert not mid.iloc[-len(other_frame):].reset_index(drop=True).equals(other_frame)
    # adjacent folds overlap in all but one day, shifted by one day
    pd.testing.assert_frame_equal(
        frames[1].iloc[DAY_ROWS:].reset_index(drop=True), mid.iloc[:-DAY_ROWS].reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        mid.iloc[DAY_ROWS:].reset_index(drop=True), frames[3].iloc[:-DAY_ROWS].reset_index(drop=True)
    )


@needs_source
def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "dry"
    assert sf.main(["--out-dir", str(out), "--dry-run"]) == 0
    assert not out.exists()
