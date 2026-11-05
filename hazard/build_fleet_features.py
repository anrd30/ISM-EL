"""Build ``sim/fleet_features_v1.json``: real feature-value pools of the training split.

Usage (from the repository root)::

    python -m hazard.build_fleet_features                 # train split -> OUT_PATH
    python -m hazard.build_fleet_features --split test    # test split -> OUT_PATH_TEST

Inputs are the ``hazard.fit_v1`` paths (``TIMELINES``, ``LABELS``,
``DRIVE_SPLIT``). Rows are selected for one split (rows with ``label == -1``
dropped): ``TRAIN_SPLIT`` by default, through ``hazard.fit_v1.select_rows``, or
``TEST_SPLIT`` only when ``--split test`` / ``split=TEST_SPLIT`` is given,
through the local ``_select_test_rows``; no other split is read. Timelines and labels are read in lockstep with the key check of
``hazard.fit_v1.paired_chunks``.

Per drive model and per value column the output holds:

* ``healthy``: a uniform random sample (without replacement) of up to
  ``MAX_PER_CELL`` usable values from rows with ``label == 0``;
* ``failure``: the same from rows with ``label == 1``;
* ``structurally_missing``: true when the share of rows of that model without a
  usable value is at least ``STRUCTURAL_MISSING_FRACTION``.

A value is usable when it is finite after narrowing to float32 (the dtype of
the model frame); NaN, infinite and float32-overflowing values count as
missing. Each stored value is the shortest decimal representation of its
float32. Sampling keeps, per cell and column, the usable values whose row keys
(one uniform draw per row from ``SAMPLING_SEED``) are smallest, which is a
uniform sample without replacement. A cell with fewer usable values than
``MAX_PER_CELL`` holds all of them. Realized counts are printed per cell.

Per model the file also holds ``drives``, ``share`` and ``rows``: the model's
count and fraction of the drives of the selected split in the drive-split table,
and its number of selected rows.

Format: JSON. The file is a few megabytes (28 columns x 2 cells x 4 models x
up to 2000 values), loads with the standard library and diffs readably in git.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from hazard import fit_v1

REPO = Path(__file__).resolve().parents[1]
OUT_PATH = REPO / "sim" / "fleet_features_v1.json"
OUT_PATH_TEST = REPO / "sim" / "fleet_features_v1_test.json"

#: Split name read only by ``--split test``; ``hazard.fit_v1`` does not define it.
TEST_SPLIT = "test"

#: Seed of the row keys that drive the sampling; used for nothing else.
SAMPLING_SEED = 20220501
MAX_PER_CELL = 2000
#: Share of rows without a usable value at or above which a (model, column) is structurally missing.
STRUCTURAL_MISSING_FRACTION = 0.999

HEALTHY_LABEL = 0
FAILURE_LABEL = 1
CELLS = {"healthy": HEALTHY_LABEL, "failure": FAILURE_LABEL}


class _Pool:
    """The ``capacity`` smallest-key (key, value) pairs seen so far."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.keys = np.empty(0, dtype=np.float64)
        self.values = np.empty(0, dtype=np.float32)

    def add(self, keys: np.ndarray, values: np.ndarray) -> None:
        keys = np.concatenate([self.keys, keys])
        values = np.concatenate([self.values, values])
        if keys.shape[0] > self.capacity:
            keep = np.argpartition(keys, self.capacity - 1)[: self.capacity]
            keys, values = keys[keep], values[keep]
        self.keys, self.values = keys, values

    def sample(self) -> list[float]:
        order = np.argsort(self.keys, kind="stable")
        return [float(str(v)) for v in self.values[order]]


def _usable_values(column: pa.ChunkedArray) -> np.ndarray:
    """Return ``column`` as float32, with NaN where the value is not usable."""
    with np.errstate(over="ignore"):
        values = column.to_pandas().to_numpy(dtype=np.float64, na_value=np.nan).astype(np.float32)
    values[~np.isfinite(values)] = np.nan
    return values


def _select_test_rows(labels: pd.DataFrame, drive_split: pd.DataFrame) -> np.ndarray:
    """Return the row mask for ``TEST_SPLIT``: drive in the split and ``label != -1``.

    Same logic as ``fit_v1.select_rows`` (left merge on ``serial_number``, many-to-one),
    kept local because ``fit_v1.select_rows`` accepts the train and validation splits only.
    """
    joined = labels[["serial_number", "label"]].merge(
        drive_split[["serial_number", "split"]], on="serial_number", how="left", validate="many_to_one"
    )
    return ((joined["split"] == TEST_SPLIT) & (joined["label"] != fit_v1.CENSORED)).to_numpy()


def build(
    out_path: Path = OUT_PATH,
    max_per_cell: int = MAX_PER_CELL,
    seed: int = SAMPLING_SEED,
    *,
    split: str = fit_v1.TRAIN_SPLIT,
) -> dict:
    """Read the rows of ``split``, build the table, write it to ``out_path`` and return it."""
    started = time.time()
    drive_split = pd.read_parquet(fit_v1.DRIVE_SPLIT, columns=["serial_number", "model", "split"])
    if drive_split["serial_number"].duplicated().any():
        raise ValueError("drive split has more than one row for a drive")
    drive_split = drive_split[drive_split["split"] == split].copy()
    categories = sorted(drive_split[fit_v1.MODEL_COLUMN].unique().tolist())
    split_drives = drive_split[fit_v1.MODEL_COLUMN].value_counts()
    total_drives = int(split_drives.sum())
    values, _ = fit_v1.feature_columns(pq.ParquetFile(fit_v1.TIMELINES).schema_arrow.names)

    pools = {m: {c: {cell: _Pool(max_per_cell) for cell in CELLS} for c in values} for m in categories}
    rows = {m: 0 for m in categories}
    unusable = {m: {c: 0 for c in values} for m in categories}
    rng = np.random.default_rng(seed)

    read = [*fit_v1.KEY_COLUMNS, fit_v1.MODEL_COLUMN, *values]
    for left, right in fit_v1.paired_chunks(read):
        labels = right.select(["serial_number", "label"]).to_pandas()
        if split == TEST_SPLIT:
            mask = _select_test_rows(labels, drive_split)
        else:
            mask = fit_v1.select_rows(labels, drive_split, split)
        keys = rng.random(left.num_rows)
        if not mask.any():
            continue
        chosen = left.filter(pa.array(mask))
        chosen_keys = keys[mask]
        chosen_labels = labels["label"].to_numpy()[mask]
        chosen_model = chosen.column(fit_v1.MODEL_COLUMN).to_pandas().astype(str).to_numpy()
        columns = {c: _usable_values(chosen.column(c)) for c in values}
        for model in categories:
            in_model = chosen_model == model
            rows[model] += int(in_model.sum())
            for c in values:
                usable = ~np.isnan(columns[c])
                unusable[model][c] += int((in_model & ~usable).sum())
                for cell, label in CELLS.items():
                    pick = in_model & usable & (chosen_labels == label)
                    if pick.any():
                        pools[model][c][cell].add(chosen_keys[pick], columns[c][pick])

    models = {}
    for model in categories:
        columns = {}
        for c in values:
            share_missing = unusable[model][c] / rows[model] if rows[model] else 1.0
            structural = share_missing >= STRUCTURAL_MISSING_FRACTION
            columns[c] = {
                "structurally_missing": bool(structural),
                "missing_fraction": share_missing,
                **{cell: pools[model][c][cell].sample() for cell in CELLS},
            }
            print(
                f"model={model} column={c} structurally_missing={structural} "
                f"healthy={len(columns[c]['healthy'])} failure={len(columns[c]['failure'])}"
            )
        models[model] = {
            "drives": int(split_drives[model]),
            "share": int(split_drives[model]) / total_drives,
            "rows": rows[model],
            "columns": columns,
        }
    table = {
        "split": split,
        "sampling_seed": seed,
        "max_per_cell": max_per_cell,
        "structural_missing_fraction": STRUCTURAL_MISSING_FRACTION,
        "value_columns": values,
        "model_categories": categories,
        "models": models,
    }
    staging = out_path.with_suffix(".json.tmp")
    staging.write_text(json.dumps(table, separators=(",", ":")) + "\n", encoding="utf-8")
    staging.replace(out_path)
    print(f"output={out_path} bytes={out_path.stat().st_size} wall_clock_seconds={time.time() - started:.1f}")
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=("train", "test"), default="train")
    args = parser.parse_args(argv)
    split, out_path = (fit_v1.TRAIN_SPLIT, OUT_PATH) if args.split == "train" else (TEST_SPLIT, OUT_PATH_TEST)
    build(out_path, split=split)
    return 0


if __name__ == "__main__":
    sys.exit(main())
