"""Out-of-vendor evaluation of the hazard model fitted on the Seagate models only.

Usage (from the repository root)::

    python -m hazard.evaluate_vendor_holdout --out agent/evidence/hazard-vendor-holdout-2026-11-20

Models: ``hazard/artifacts/v1_seagate_only`` (fitted on the train and val buckets of
``backblaze_drive_split_vendor_holdout.parquet``, Seagate drives only), the pooled ``hazard/artifacts/v1``
(A3, read only) and ``hazard.v0``. Nothing is fitted or recalibrated here.

Rows: label != -1 rows. Cells:

- ``toshiba_holdout``: every drive of TOSHIBA MG07ACA14TA (the vendor holdout; none of its drives was fitted on).
- ``toshiba_a2_test``: the Toshiba drives that A2 put in its test split, the cell A4 reports for the pooled model. The pooled
  model was not fitted on these drives either, so the two models and v0 are compared on the same rows.
- ``seagate_test``: the Seagate drives of the test bucket (unchanged from A2), the in-vendor control.

Metrics are those of ``hazard.evaluate_v1``: PR-AUC (average precision), Brier score with 95% drive-bootstrap intervals
(200 resamples, seed 20220401), recall at a false-positive rate of 1% and 0.1%, and a 20-bin reliability table. Thresholds are
chosen on validation rows and frozen: for the Toshiba cells on the Toshiba drives of A2's validation split, for the Seagate
cell on the validation bucket of the new split. The paired difference "new model minus pooled model" uses the same
resamples for both.

Sanity checks are printed and written: score range and number of distinct values, mean score against the base rate, and AUC
of the in-vendor control.
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

from hazard import evaluate_v1 as ev
from hazard import fit_v1
from hazard.v1 import FittedHazardV1, _load as load_ensemble

REPO = Path(__file__).resolve().parents[1]
PROCESSED = fit_v1.PROCESSED
NEW_SPLIT = PROCESSED / "backblaze_drive_split_vendor_holdout.parquet"
A2_SPLIT = fit_v1.DRIVE_SPLIT
NEW_DIR = REPO / "hazard" / "artifacts" / "v1_seagate_only"
POOLED_DIR = REPO / "hazard" / "artifacts" / "v1"
HOLDOUT_MODEL = "TOSHIBA MG07ACA14TA"
HOLDOUT_BUCKET = "vendor_holdout"

V0, POOLED, NEW = "hazard_v0", "hazard_v1_pooled", "hazard_v1_seagate_only"


def predict_batch(frame: pd.DataFrame, directory: Path, device: str) -> np.ndarray:
    """Calibrated ensemble mean of ``directory``'s model for every row (the arithmetic of ``FittedHazardV1.__call__``)."""
    import xgboost as xgb

    ensemble = load_ensemble(directory)
    meta = ensemble.meta
    values = frame[meta["value_columns"]].to_numpy(dtype=np.float64)
    with np.errstate(over="ignore"):
        narrowed = values.astype(np.float32)
    narrowed[~np.isfinite(narrowed)] = np.nan
    out = pd.DataFrame(narrowed, columns=meta["value_columns"], index=frame.index)
    flags = pd.DataFrame(np.isnan(narrowed).astype(np.float32), columns=meta["flag_columns"], index=frame.index)
    out = pd.concat([out, flags], axis=1)
    out[fit_v1.MODEL_COLUMN] = pd.Categorical(frame[fit_v1.MODEL_COLUMN], categories=meta["model_categories"])
    matrix = xgb.DMatrix(out[meta["feature_columns"]], enable_categorical=True)
    boosters = []
    for name in meta["boosters"]:
        booster = xgb.Booster()
        booster.load_model(directory / name)
        booster.set_param({"device": device})
        boosters.append(booster)
    calibrated = np.stack([np.interp(b.predict(matrix), x, y) for b, (x, y) in zip(boosters, ensemble.thresholds)])
    return np.clip(calibrated.mean(axis=0), 0.0, np.nextafter(1.0, 0.0))


class _Boosters:
    """Boosters of one directory loaded once per device (``predict_batch`` is called per chunk)."""

    cache: dict = {}

    @classmethod
    def get(cls, directory: Path, device: str):
        import xgboost as xgb

        key = (directory, device)
        if key not in cls.cache:
            ensemble = load_ensemble(directory)
            boosters = []
            for name in ensemble.meta["boosters"]:
                booster = xgb.Booster()
                booster.load_model(directory / name)
                booster.set_param({"device": device})
                boosters.append(booster)
            cls.cache[key] = (ensemble, boosters)
        return cls.cache[key]


def predict_cached(frame: pd.DataFrame, directory: Path, device: str) -> np.ndarray:
    import xgboost as xgb

    ensemble, boosters = _Boosters.get(directory, device)
    meta = ensemble.meta
    values = frame[meta["value_columns"]].to_numpy(dtype=np.float64)
    with np.errstate(over="ignore"):
        narrowed = values.astype(np.float32)
    narrowed[~np.isfinite(narrowed)] = np.nan
    out = pd.DataFrame(narrowed, columns=meta["value_columns"], index=frame.index)
    flags = pd.DataFrame(np.isnan(narrowed).astype(np.float32), columns=meta["flag_columns"], index=frame.index)
    out = pd.concat([out, flags], axis=1)
    out[fit_v1.MODEL_COLUMN] = pd.Categorical(frame[fit_v1.MODEL_COLUMN], categories=meta["model_categories"])
    matrix = xgb.DMatrix(out[meta["feature_columns"]], enable_categorical=True)
    calibrated = np.stack([np.interp(b.predict(matrix), x, y) for b, (x, y) in zip(boosters, ensemble.thresholds)])
    return np.clip(calibrated.mean(axis=0), 0.0, np.nextafter(1.0, 0.0))


def cross_check(frame: pd.DataFrame, directory: Path, device: str, n_rows: int = 200) -> float:
    """Largest absolute difference between the batch path and ``FittedHazardV1`` on ``n_rows`` random rows."""
    sample = frame.sample(n=min(n_rows, len(frame)), random_state=ev.BOOTSTRAP_SEED).reset_index(drop=True)
    batch = predict_cached(sample, directory, device)
    meta = load_ensemble(directory).meta
    model = FittedHazardV1(directory)
    rowwise = []
    for _, row in sample.iterrows():
        features = {c: (None if pd.isna(row[c]) else float(row[c])) for c in meta["value_columns"]}
        features[fit_v1.MODEL_COLUMN] = row[fit_v1.MODEL_COLUMN]
        rowwise.append(model(features)[0])
    return float(np.max(np.abs(batch - np.array(rowwise))))


def score_rows(device: str, max_chunks: int | None = None) -> pd.DataFrame:
    """Scores of v0, the pooled model (A2-test rows only, NaN elsewhere) and the new model for the rows of the three cells and the val buckets."""
    new_split = pd.read_parquet(NEW_SPLIT, columns=["serial_number", "model", "split"]).set_index("serial_number")
    a2_split = pd.read_parquet(A2_SPLIT, columns=["serial_number", "split"]).set_index("serial_number")["split"]
    codes = pd.Series(np.arange(len(new_split), dtype=np.int64), index=new_split.index)
    values = list(load_ensemble(NEW_DIR).meta["value_columns"])
    read = [*fit_v1.KEY_COLUMNS, fit_v1.MODEL_COLUMN, *values]
    parts = []
    started = time.time()
    for index, (left, right) in enumerate(fit_v1.paired_chunks(read)):
        if max_chunks is not None and index >= max_chunks:
            break
        labels = right.select(["serial_number", "label"]).to_pandas()
        serial = labels["serial_number"]
        bucket = new_split["split"].reindex(serial).to_numpy()
        a2 = a2_split.reindex(serial).to_numpy()
        label = labels["label"].to_numpy()
        keep = (label != fit_v1.CENSORED) & (np.isin(bucket, (HOLDOUT_BUCKET, "test", "val")))
        if not keep.any():
            continue
        chosen = left.filter(pa.array(keep)).to_pandas().reset_index(drop=True)
        chosen["label"] = label[keep].astype(np.int8)
        chosen["bucket"] = bucket[keep]
        chosen["a2"] = a2[keep]
        chosen["code"] = codes.reindex(chosen["serial_number"]).to_numpy()
        new_scores = predict_cached(chosen, NEW_DIR, device)
        pooled = np.full(len(chosen), np.nan)
        need = (chosen["a2"] == "test") | (chosen["a2"] == "val")
        if need.any():
            pooled[need.to_numpy()] = predict_cached(chosen[need.to_numpy()].reset_index(drop=True), POOLED_DIR, device)
        parts.append(pd.DataFrame({
            "serial_number": chosen["code"].to_numpy(),
            fit_v1.MODEL_COLUMN: chosen[fit_v1.MODEL_COLUMN].astype(str).to_numpy(),
            "label": chosen["label"].to_numpy(), "bucket": chosen["bucket"].to_numpy(), "a2": chosen["a2"].to_numpy(),
            V0: ev.predict_v0_batch(chosen), POOLED: pooled, NEW: new_scores}))
        if index % 5 == 0:
            print(f"chunk={index} rows={sum(len(p) for p in parts)} seconds={time.time() - started:.0f}", flush=True)
    scored = pd.concat(parts, ignore_index=True)
    scored["vendor"] = np.where(scored[fit_v1.MODEL_COLUMN] == HOLDOUT_MODEL, "Toshiba", "Seagate")
    scored[fit_v1.MODEL_COLUMN] = scored[fit_v1.MODEL_COLUMN].astype("category")
    return scored


def _pair(frame: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Frame whose ``V0``/``V1`` columns are the scores of ``a`` and ``b`` (so ``evaluate_cell`` reports ``b`` minus ``a``)."""
    out = frame[["serial_number", "label"]].copy()
    out[ev.V0] = frame[a].to_numpy()
    out[ev.V1] = frame[b].to_numpy()
    return out


def run_cell(test: pd.DataFrame, val: pd.DataFrame, a: str, b: str, index: int) -> dict:
    t, v = _pair(test, a, b), _pair(val, a, b)
    return ev.evaluate_cell(index, t, v, np.ones(len(t), dtype=bool), np.ones(len(v), dtype=bool))


def sanity(frame: pd.DataFrame, column: str) -> dict:
    s = frame[column].to_numpy()
    y = frame["label"].to_numpy()
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    n1 = int(y.sum())
    auc = float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))
    return {"rows": int(len(s)), "min": float(s.min()), "max": float(s.max()), "mean_score": float(s.mean()),
            "base_rate": float(y.mean()), "distinct_scores": int(len(np.unique(s))), "roc_auc": auc}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-chunks", type=int, default=None, help="restrict the rows read (smoke runs)")
    args = parser.parse_args(argv)
    started = time.time()
    device = fit_v1.select_device()[0]
    scored = score_rows(device, args.max_chunks)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"scored_rows={len(scored)} seconds={time.time() - started:.0f}")
    first = scored.head(1)
    cells = {
        "toshiba_holdout": (scored[(scored["bucket"] == HOLDOUT_BUCKET)], scored[(scored["a2"] == "val") & (scored["vendor"] == "Toshiba")]),
        "toshiba_a2_test": (scored[(scored["a2"] == "test") & (scored["vendor"] == "Toshiba")], scored[(scored["a2"] == "val") & (scored["vendor"] == "Toshiba")]),
        "seagate_test": (scored[(scored["bucket"] == "test")], scored[(scored["bucket"] == "val")]),
    }
    results, checks = {}, {}
    n = 0
    for name, (test, val) in cells.items():
        checks[name] = {NEW: sanity(test, NEW), V0: sanity(test, V0)}
        print(name, json.dumps(checks[name]), flush=True)
        spec = [(f"{name}: v0 vs new", V0, NEW)]
        if name != "toshiba_holdout":
            checks[name][POOLED] = sanity(test, POOLED)
            spec += [(f"{name}: v0 vs pooled", V0, POOLED), (f"{name}: pooled vs new", POOLED, NEW)]
        for label, a, b in spec:
            results[label] = run_cell(test.reset_index(drop=True), val.reset_index(drop=True), a, b, n)
            n += 1
            print(f"done {label} seconds={time.time() - started:.0f}", flush=True)
    serializable = json.loads(json.dumps({"checks": checks, "cells": results}, default=lambda o: float(o) if isinstance(o, np.floating) else int(o)))
    (args.out / "calibration_results.json").write_text(json.dumps(serializable, indent=1) + "\n", encoding="utf-8")
    print(f"wall_clock_seconds={time.time() - started:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
