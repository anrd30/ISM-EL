"""Cut ``profile_diurnal_long`` into its time-forward train and holdout halves.

The long profile carries a ``suggested_split`` in its sidecar but nothing reads
it: B-lane left it as metadata for whoever builds the real training run. This
script is that consumer. It writes two replay files, each a valid replay file in
its own right, so training and evaluation can point ``--workload-csv`` at
different halves of one continuous window without any consumer having to know
about row ranges.

Rebasing matters. ``contracts.validate_replay_grid`` requires episode-relative
``t_seconds`` starting at 0, and says rebasing happens in the data pipeline and
never in a consumer, so the holdout half's clock is restarted here.

Usage::

    python -m data_pipeline.tencent.split_long_profile

Writes, next to the source profile:

- ``profile_diurnal_long_train.csv``    rows [0, first_holdout_row)
- ``profile_diurnal_long_holdout.csv``  rows [first_holdout_row, end)

plus a ``.meta.json`` sidecar for each, carrying the parent's provenance, the
absolute window this half covers, and the row range it was cut from. Both halves
are validated with ``contracts.validate_replay_frame`` before being written.

Why a time-forward split rather than two windows: the four original profiles are
single 24-hour windows tiled to fill an episode, so the next interval's load sits
exactly 288 steps back in the agent's own history and anticipation is recoverable
by construction. One continuous 14-day window removes the tiling, and cutting it
forward in time gives an evaluation half the policy has never seen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from contracts.contracts import CONTROL_INTERVAL_S, validate_replay_frame

REPO = Path(__file__).resolve().parents[2]
PROFILES = REPO / "data_pipeline" / "profiles"
SOURCE_STEM = "profile_diurnal_long"

__all__ = ["split", "main"]


def _sidecar(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _half_meta(parent: dict, half: str, start_row: int, end_row: int) -> dict:
    """Sidecar for one half: the parent's provenance plus this half's bounds."""
    step = CONTROL_INTERVAL_S
    start_ts = int(parent["window_start_ts"]) + start_row * step
    end_ts = int(parent["window_start_ts"]) + (end_row - 1) * step
    return {
        "profile": f"{parent['profile']}_{half}",
        "synthetic": parent["synthetic"],
        "source": parent["source"],
        "label": f"{half} half of {parent['profile']} ({parent['label']})",
        "disclosure": parent["disclosure"],
        "parent_profile": parent["profile"],
        "parent_rows": int(parent["rows"]),
        "parent_row_range": [start_row, end_row],
        "rows": end_row - start_row,
        "n_disks": parent["n_disks"],
        "disk_uids": parent["disk_uids"],
        "selection_rule": parent["selection_rule"],
        "window_start_ts": start_ts,
        "window_end_ts": end_ts,
        "scaling_factor": parent["scaling_factor"],
        "split_note": (
            "Time-forward cut of a single continuous window. t_seconds is rebased to "
            "start at 0 as contracts.validate_replay_grid requires; the absolute window "
            "this half covers is window_start_ts..window_end_ts."
        ),
    }


def split(source_csv: Path | None = None, *, write: bool = True) -> dict[str, Path]:
    """Write the two halves and return their paths by name.

    Raises:
        ContractViolation: if either half is not a valid replay file.
        KeyError: if the sidecar carries no ``suggested_split``.
    """
    csv_path = source_csv or (PROFILES / f"{SOURCE_STEM}.csv")
    parent = _sidecar(csv_path.with_suffix(".meta.json"))
    frame = pd.read_csv(csv_path)

    cut = int(parent["suggested_split"]["first_holdout_row"])
    if not 0 < cut < len(frame):
        raise ValueError(f"first_holdout_row {cut} is outside the profile's {len(frame)} rows")

    out: dict[str, Path] = {}
    for half, start, end in (("train", 0, cut), ("holdout", cut, len(frame))):
        piece = frame.iloc[start:end].copy()
        # Rebase the clock: each half is a replay file in its own right.
        piece["t_seconds"] = piece["t_seconds"].to_numpy() - int(piece["t_seconds"].iloc[0])
        piece = piece.astype(frame.dtypes.to_dict()).reset_index(drop=True)
        validate_replay_frame(piece)

        target = csv_path.with_name(f"{SOURCE_STEM}_{half}.csv")
        if write:
            piece.to_csv(target, index=False, lineterminator="\n")
            target.with_suffix(".meta.json").write_text(
                json.dumps(_half_meta(parent, half, start, end), indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )
        out[half] = target
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m data_pipeline.tencent.split_long_profile",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--source", type=Path, default=None, help="profile CSV to cut; defaults to the long diurnal profile")
    p.add_argument("--dry-run", action="store_true", help="validate both halves without writing them")
    args = p.parse_args(argv)

    paths = split(args.source, write=not args.dry_run)
    verb = "would write" if args.dry_run else "wrote"
    for half, path in paths.items():
        print(f"{verb} {half}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
