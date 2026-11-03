"""Fit the v1 failure-hazard ensemble on the training split of the Backblaze tables.

Usage (from the repository root)::

    python -m hazard.fit_v1

Inputs (``data_pipeline/data/processed``): ``backblaze_timelines.parquet``,
``backblaze_labels.parquet`` (same rows, same order, plus ``label``) and
``backblaze_drive_split.parquet`` (one row per drive).

Row selection: label rows are joined to the drive split on ``serial_number``
and rows with ``label == -1`` are dropped. Rows of the training split are fitted
on; rows of the validation split are used for early stopping and for the
isotonic calibration. Only these two splits are ever selected
(``ALLOWED_SPLITS``). Labels and timelines are read in lockstep batches and the
``(serial_number, date)`` keys of every batch pair are compared before the
batches are combined.

Ensemble: ``N_MODELS`` XGBoost boosters, each fitted on a bootstrap resample of
the training rows (same row count, drawn with replacement). The resample is
carried as integer sample weights equal to the number of times each row is
drawn, which gives the same weighted loss as the materialised resample.
Class imbalance is handled by ``scale_pos_weight`` (negative rows / positive
rows of the training split); no rows are resampled by class.

Calibration: one isotonic map per booster, fitted on that booster's raw
validation predictions, stored as threshold arrays.

Outputs: ``hazard/artifacts/v1/`` with ``booster_<i>.ubj``, ``calibrators.npz``
and ``meta.json``.
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

REPO = Path(__file__).resolve().parents[1]
PROCESSED = REPO / "data_pipeline" / "data" / "processed"
TIMELINES = PROCESSED / "backblaze_timelines.parquet"
LABELS = PROCESSED / "backblaze_labels.parquet"
DRIVE_SPLIT = PROCESSED / "backblaze_drive_split.parquet"
ARTIFACT_DIR = REPO / "hazard" / "artifacts" / "v1"

CENSORED = -1
TRAIN_SPLIT = "train"
VAL_SPLIT = "val"
ALLOWED_SPLITS = (TRAIN_SPLIT, VAL_SPLIT)

MISSING_SUFFIX = "_missing"
MODEL_COLUMN = "model"
KEY_COLUMNS = ("serial_number", "date")
BATCH_ROWS = 1_000_000

N_MODELS = 25
SEED_BASE = 20220301
SEEDS = tuple(SEED_BASE + i for i in range(N_MODELS))

MAX_ROUNDS = 2000
EARLY_STOPPING_ROUNDS = 50
PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "learning_rate": 0.05,
    "max_depth": 6,
    "max_bin": 256,
}
PROBABILITY_FLOOR = 1e-15


def select_rows(labels: pd.DataFrame, drive_split: pd.DataFrame, split_name: str) -> np.ndarray:
    """Return a boolean mask over ``labels`` rows: drive in ``split_name`` and label != -1.

    ``labels`` has ``serial_number`` and ``label``; ``drive_split`` has one row
    per ``serial_number`` with ``split``. A drive absent from ``drive_split``
    is not selected.

    Raises:
        ValueError: if ``split_name`` is not in ``ALLOWED_SPLITS``.
    """
    if split_name not in ALLOWED_SPLITS:
        raise ValueError(f"split must be one of {ALLOWED_SPLITS}, got {split_name!r}")
    joined = labels[["serial_number", "label"]].merge(
        drive_split[["serial_number", "split"]], on="serial_number", how="left", validate="many_to_one"
    )
    return ((joined["split"] == split_name) & (joined["label"] != CENSORED)).to_numpy()


def feature_columns(schema_names: Sequence[str]) -> tuple[list[str], list[str]]:
    """Return ``(value_columns, flag_columns)`` of a timelines schema.

    Flag columns end in ``_missing``; the value columns are the same names
    without the suffix, in schema order.
    """
    flags = [n for n in schema_names if n.endswith(MISSING_SUFFIX)]
    values = [n[: -len(MISSING_SUFFIX)] for n in flags]
    absent = [n for n in values if n not in schema_names]
    if absent:
        raise ValueError(f"flag columns without a value column: {absent}")
    return values, flags


def _chunks(batches: Iterator[pa.RecordBatch], rows: int) -> Iterator[pa.Table]:
    """Regroup record batches into tables of ``rows`` rows (the last one may be shorter)."""
    pending: list[pa.RecordBatch] = []
    count = 0
    for batch in batches:
        pending.append(batch)
        count += batch.num_rows
        while count >= rows:
            table = pa.Table.from_batches(pending)
            yield table.slice(0, rows)
            rest = table.slice(rows)
            pending = rest.to_batches()
            count = rest.num_rows
    if count:
        yield pa.Table.from_batches(pending)


def paired_chunks(
    timeline_columns: Sequence[str], rows: int = BATCH_ROWS
) -> Iterator[tuple[pa.Table, pa.Table]]:
    """Yield ``(timelines chunk, labels chunk)`` pairs of equal length with equal keys.

    Raises:
        ValueError: if the two files differ in row count or a pair differs in
            ``serial_number`` or ``date``.
    """
    timelines = pq.ParquetFile(TIMELINES).iter_batches(batch_size=rows, columns=list(timeline_columns))
    labels = pq.ParquetFile(LABELS).iter_batches(batch_size=rows, columns=[*KEY_COLUMNS, "label"])
    for left, right in zip(_chunks(timelines, rows), _chunks(labels, rows), strict=True):
        if left.num_rows != right.num_rows:
            raise ValueError("timelines and labels differ in row count")
        for key in KEY_COLUMNS:
            same = left.column(key).cast(pa.string() if key == "serial_number" else pa.date32())
            if not same.equals(right.column(key).cast(same.type)):
                raise ValueError(f"timelines and labels differ in {key}")
        yield left, right


class SplitBatches(xgb.DataIter):
    """XGBoost data iterator over the selected rows of one split.

    Each batch holds float32 value columns, float32 0/1 flag columns and the
    drive model as a categorical column with fixed categories.
    """

    def __init__(self, split_name: str, drive_split: pd.DataFrame, values: list[str],
                 flags: list[str], categories: list[str]) -> None:
        super().__init__(cache_prefix=None)
        self.split_name = split_name
        self.drive_split = drive_split
        self.values = values
        self.flags = flags
        self.categories = categories
        self.columns = [*values, *flags, MODEL_COLUMN]
        self._source: Iterator | None = None

    def reset(self) -> None:
        self._source = None

    def next(self, input_data) -> bool:
        if self._source is None:
            self._source = self._batches()
        try:
            frame, label = next(self._source)
        except StopIteration:
            return False
        input_data(data=frame, label=label)
        return True

    def _batches(self) -> Iterator[tuple[pd.DataFrame, np.ndarray]]:
        read = [*KEY_COLUMNS, MODEL_COLUMN, *self.values, *self.flags]
        for left, right in paired_chunks(read):
            labels = right.select(["serial_number", "label"]).to_pandas()
            mask = select_rows(labels, self.drive_split, self.split_name)
            if not mask.any():
                continue
            chosen = left.filter(pa.array(mask)).to_pandas()
            frame = pd.DataFrame(index=chosen.index)
            for name, flag in zip(self.values, self.flags):
                value = chosen[name].to_numpy(dtype=np.float32)
                if not np.array_equal(chosen[flag].to_numpy(dtype=bool), np.isnan(value)):
                    raise ValueError(f"{flag} differs from the missing values of {name}")
                frame[name] = value
                frame[flag] = chosen[flag].to_numpy(dtype=np.float32)
            frame[MODEL_COLUMN] = pd.Categorical(chosen[MODEL_COLUMN], categories=self.categories)
            yield frame[self.columns], labels["label"].to_numpy()[mask].astype(np.float32)


def select_device() -> tuple[str, str]:
    """Return ``(device, note)``: ``cuda`` if a small CUDA fit succeeds, else ``cpu``."""
    try:
        probe = xgb.DMatrix(np.array([[0.0], [1.0]], dtype=np.float32), label=[0, 1])
        xgb.train({"device": "cuda", "tree_method": "hist"}, probe, num_boost_round=1)
        return "cuda", "cuda"
    except xgb.core.XGBoostError as err:
        return "cpu", f"cpu (cuda unavailable: {str(err).splitlines()[0]})"


def bootstrap_weights(seed: int, rows: int) -> np.ndarray:
    """Return how many times each of ``rows`` rows is drawn in one bootstrap resample."""
    draws = np.random.default_rng(seed).integers(0, rows, size=rows)
    return np.bincount(draws, minlength=rows).astype(np.float32)


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p.astype(np.float64), PROBABILITY_FLOOR, 1.0 - PROBABILITY_FLOOR)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p.astype(np.float64) - y) ** 2))


def fit(out_dir: Path = ARTIFACT_DIR, seeds: Sequence[int] = SEEDS) -> dict:
    """Run the full selection, fit, calibration and save; return the metadata written."""
    started = time.time()
    device, note = select_device()
    print(f"device={note} xgboost={xgb.__version__} rounds_max={MAX_ROUNDS} "
          f"early_stopping_rounds={EARLY_STOPPING_ROUNDS}")
    print(f"seeds={list(seeds)}")

    drive_split = pd.read_parquet(DRIVE_SPLIT, columns=["serial_number", "model", "split"])
    if drive_split["serial_number"].duplicated().any():
        raise ValueError("drive split has more than one row for a drive")
    categories = sorted(drive_split[MODEL_COLUMN].unique().tolist())
    values, flags = feature_columns(pq.ParquetFile(TIMELINES).schema_arrow.names)

    train_iter = SplitBatches(TRAIN_SPLIT, drive_split, values, flags, categories)
    dtrain = xgb.QuantileDMatrix(train_iter, max_bin=PARAMS["max_bin"], enable_categorical=True)
    y_train = dtrain.get_label()
    train_rows, train_pos = int(dtrain.num_row()), int(y_train.sum())
    print(f"train_rows={train_rows} train_positive={train_pos}")
    val_iter = SplitBatches(VAL_SPLIT, drive_split, values, flags, categories)
    dval = xgb.QuantileDMatrix(val_iter, ref=dtrain, enable_categorical=True)
    y_val = dval.get_label().astype(np.float64)
    print(f"val_rows={int(dval.num_row())} val_positive={int(y_val.sum())}")

    scale = (train_rows - train_pos) / train_pos
    params = {**PARAMS, "device": device, "scale_pos_weight": scale}
    boosters, raw_val, rounds = [], [], []
    for seed in seeds:
        dtrain.set_weight(bootstrap_weights(seed, train_rows))
        booster = xgb.train({**params, "seed": seed}, dtrain, num_boost_round=MAX_ROUNDS,
                            evals=[(dval, "val")], early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                            verbose_eval=False)
        best = booster.best_iteration + 1
        booster = booster[:best]
        boosters.append(booster)
        raw_val.append(booster.predict(dval))
        rounds.append(best)
        print(f"seed={seed} rounds={best} raw_val_logloss={log_loss(y_val, raw_val[-1]):.6f}")

    calibrators = []
    for raw in raw_val:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y_val)
        calibrators.append((iso.X_thresholds_.astype(np.float64), iso.y_thresholds_.astype(np.float64)))
    calibrated = [np.interp(raw, x, y) for raw, (x, y) in zip(raw_val, calibrators)]
    per_model = [{"seed": int(s), "rounds": int(r), "val_logloss": log_loss(y_val, c),
                  "val_brier": brier(y_val, c)} for s, r, c in zip(seeds, rounds, calibrated)]
    for row in per_model:
        print(f"seed={row['seed']} rounds={row['rounds']} val_logloss={row['val_logloss']:.6f} "
              f"val_brier={row['val_brier']:.6f}")
    mean = np.mean(calibrated, axis=0)
    ensemble = {"val_logloss": log_loss(y_val, mean), "val_brier": brier(y_val, mean)}
    print(f"ensemble val_logloss={ensemble['val_logloss']:.6f} val_brier={ensemble['val_brier']:.6f}")

    staging = out_dir.with_name(out_dir.name + ".tmp")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    names = [f"booster_{i:02d}.ubj" for i in range(len(boosters))]
    for name, booster in zip(names, boosters):
        booster.save_model(staging / name)
    np.savez(staging / "calibrators.npz",
             **{f"x_{i:02d}": x for i, (x, _) in enumerate(calibrators)},
             **{f"y_{i:02d}": y for i, (_, y) in enumerate(calibrators)})
    meta = {
        "horizon_hours": 720.0,
        "feature_columns": [*values, *flags, MODEL_COLUMN],
        "value_columns": values,
        "flag_columns": flags,
        "model_categories": categories,
        "seeds": [int(s) for s in seeds],
        "device": device,
        "xgboost_version": xgb.__version__,
        "params": {**params, "num_boost_round_max": MAX_ROUNDS,
                   "early_stopping_rounds": EARLY_STOPPING_ROUNDS},
        "train_rows": train_rows, "train_positive": train_pos,
        "val_rows": int(len(y_val)), "val_positive": int(y_val.sum()),
        "per_model": per_model, "ensemble": ensemble,
        "boosters": names, "calibrators": "calibrators.npz",
    }
    meta["wall_clock_seconds"] = round(time.time() - started, 1)
    (staging / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    staging.rename(out_dir)
    total = sum(p.stat().st_size for p in out_dir.iterdir())
    print(f"artifacts={out_dir} files={len(list(out_dir.iterdir()))} bytes={total}")
    print(f"wall_clock_seconds={meta['wall_clock_seconds']}")
    return meta


def main() -> int:
    fit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
