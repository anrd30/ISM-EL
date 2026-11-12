"""Rolling-origin folds of the long diurnal profile."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from contracts.contracts import validate_replay_frame
from data_pipeline.tencent import rolling_origin_folds as ro
from data_pipeline.tencent.extract_load import DAY_ROWS
from data_pipeline.tencent.split_long_profile import PROFILES, SOURCE_STEM

SOURCE = PROFILES / f"{SOURCE_STEM}.csv"
needs_source = pytest.mark.skipif(not SOURCE.exists(), reason=f"{SOURCE.name} absent")


@pytest.fixture(scope="module")
def folds(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return ro.make_folds(SOURCE, tmp_path_factory.mktemp("folds"))


@needs_source
def test_fold1_train_is_byte_identical_to_committed_train(folds: dict[str, Path]) -> None:
    committed = PROFILES / f"{SOURCE_STEM}_train.csv"
    if not committed.exists():
        pytest.skip(f"{committed.name} absent")
    assert folds["fold1_train"].read_bytes() == committed.read_bytes()


@needs_source
def test_eval_pieces_are_one_day_starting_at_zero(folds: dict[str, Path]) -> None:
    for fold in range(1, ro.N_FOLDS + 1):
        frame = pd.read_csv(folds[f"fold{fold}_eval"])
        assert len(frame) == DAY_ROWS
        assert int(frame["t_seconds"].iloc[0]) == 0


@needs_source
def test_train_pieces_are_prefixes_of_one_another(folds: dict[str, Path]) -> None:
    frames = [pd.read_csv(folds[f"fold{fold}_train"]) for fold in range(1, ro.N_FOLDS + 1)]
    for shorter, longer in zip(frames, frames[1:]):
        assert len(longer) == len(shorter) + DAY_ROWS
        pd.testing.assert_frame_equal(longer.iloc[: len(shorter)].reset_index(drop=True), shorter)


@needs_source
def test_eval_piece_follows_its_train_piece(folds: dict[str, Path]) -> None:
    source = pd.read_csv(SOURCE)
    for fold in range(1, ro.N_FOLDS + 1):
        train_rows = len(pd.read_csv(folds[f"fold{fold}_train"]))
        eval_frame = pd.read_csv(folds[f"fold{fold}_eval"])
        expected = source.iloc[train_rows : train_rows + DAY_ROWS].drop(columns="t_seconds").reset_index(drop=True)
        pd.testing.assert_frame_equal(eval_frame.drop(columns="t_seconds"), expected)


@needs_source
def test_every_piece_validates_and_has_a_sidecar(folds: dict[str, Path]) -> None:
    assert len(folds) == 2 * ro.N_FOLDS
    for key, path in folds.items():
        validate_replay_frame(pd.read_csv(path))
        meta = path.with_suffix(".meta.json")
        assert meta.exists()
        assert f'"fold": {key[4]}' in meta.read_text(encoding="utf-8")
        assert b"\r" not in path.read_bytes()


@needs_source
def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    out = tmp_path / "dry"
    assert ro.main(["--out-dir", str(out), "--dry-run"]) == 0
    assert not out.exists()
