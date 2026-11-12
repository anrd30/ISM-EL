"""Cut ``profile_diurnal_long`` into three sliding-window train/holdout folds.

Each fold has a fixed-width train piece of ``TRAIN_DAYS`` days followed by a
holdout piece of ``HOLDOUT_DAYS`` days; the window advances one day per fold.
With the 14-day profile the folds are rows ``[0, 2304)`` / ``[2304, 3456)``,
``[288, 2592)`` / ``[2592, 3744)`` and ``[576, 2880)`` / ``[2880, 4032)``.
Fold 3's holdout piece covers the same rows as
``profile_diurnal_long_holdout.csv``. Train pieces overlap but are not nested.

Every piece, train and holdout, has ``t_seconds`` restarted at 0 and is
validated with ``contracts.validate_replay_frame`` before it is written, as in
``data_pipeline.tencent.split_long_profile``.

Usage::

    python -m data_pipeline.tencent.sliding_rolling_folds [--source CSV] [--out-dir DIR] [--dry-run]

Writes ``profile_diurnal_long_sfold{i}_train.csv`` and
``profile_diurnal_long_sfold{i}_holdout.csv`` with a ``.meta.json`` sidecar each.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from contracts.contracts import validate_replay_frame
from data_pipeline.tencent.extract_load import DAY_ROWS
from data_pipeline.tencent.split_long_profile import PROFILES, SOURCE_STEM, _half_meta, _sidecar

TRAIN_DAYS = 8
HOLDOUT_DAYS = 4
N_FOLDS = 3

__all__ = ["fold_bounds", "make_folds", "main"]


def fold_bounds(fold: int) -> dict[str, tuple[int, int]]:
    """Return the ``[start, end)`` row range of the train and holdout piece of ``fold`` (1-based)."""
    if not 1 <= fold <= N_FOLDS:
        raise ValueError(f"fold must be in [1, {N_FOLDS}], got {fold}")
    train_start = (fold - 1) * DAY_ROWS
    train_end = train_start + TRAIN_DAYS * DAY_ROWS
    return {
        "train": (train_start, train_end),
        "holdout": (train_end, train_end + HOLDOUT_DAYS * DAY_ROWS),
    }


def make_folds(
    source_csv: Path | None = None, out_dir: Path | None = None, *, write: bool = True
) -> dict[str, Path]:
    """Write the fold pieces and return their paths keyed ``sfold{i}_train`` / ``sfold{i}_holdout``.

    Raises:
        ContractViolation: if a piece is not a valid replay file.
        ValueError: if the source has fewer rows than the last fold needs.
    """
    csv_path = source_csv or (PROFILES / f"{SOURCE_STEM}.csv")
    target_dir = out_dir or PROFILES
    parent = _sidecar(csv_path.with_suffix(".meta.json"))
    frame = pd.read_csv(csv_path)

    needed = fold_bounds(N_FOLDS)["holdout"][1]
    if len(frame) < needed:
        raise ValueError(f"source has {len(frame)} rows, {needed} are needed")

    out: dict[str, Path] = {}
    if write:
        target_dir.mkdir(parents=True, exist_ok=True)
    for fold in range(1, N_FOLDS + 1):
        for piece_name, (start, end) in fold_bounds(fold).items():
            piece = frame.iloc[start:end].copy()
            piece["t_seconds"] = piece["t_seconds"].to_numpy() - int(piece["t_seconds"].iloc[0])
            piece = piece.astype(frame.dtypes.to_dict()).reset_index(drop=True)
            validate_replay_frame(piece)

            key = f"sfold{fold}_{piece_name}"
            target = target_dir / f"{SOURCE_STEM}_{key}.csv"
            if write:
                meta = _half_meta(parent, key, start, end)
                meta["fold"] = fold
                meta["fold_piece"] = piece_name
                piece.to_csv(target, index=False, lineterminator="\n")
                target.with_suffix(".meta.json").write_text(
                    json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n"
                )
            out[key] = target
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m data_pipeline.tencent.sliding_rolling_folds",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--source", type=Path, default=None, help="profile CSV to cut; defaults to the long diurnal profile")
    p.add_argument("--out-dir", type=Path, default=None, help="output directory; defaults to data_pipeline/profiles")
    p.add_argument("--dry-run", action="store_true", help="validate every piece without writing it")
    args = p.parse_args(argv)

    paths = make_folds(args.source, args.out_dir, write=not args.dry_run)
    verb = "would write" if args.dry_run else "wrote"
    for key, path in paths.items():
        print(f"{verb} {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
