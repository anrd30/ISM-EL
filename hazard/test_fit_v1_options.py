"""fit_v1's optional drive-split, output and manifest arguments leave the default fit unchanged."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

pytest.importorskip("xgboost")

from hazard import fit_v1  # noqa: E402
from hazard.v1 import FittedHazardV1  # noqa: E402

COLS = ["smart_5_raw", "smart_187_raw", "smart_5_raw_diff7"]
PROBES = [
    {"smart_5_raw": 5.0, "smart_187_raw": 5.0, "smart_5_raw_diff7": 5.0, "model": "MA"},
    {"smart_5_raw": 13.0, "smart_187_raw": 13.0, "smart_5_raw_diff7": 13.0, "model": "MB"},
    {"smart_5_raw": 9.0, "model": "MC"},
    {"smart_5_raw": 13.0, "smart_187_raw": None, "model": "ZZ"},
]
#: Values of the fit on this synthetic table measured with ``hazard/fit_v1.py`` before the options were added.
PRE_EDIT_REFERENCE = {
    "train_rows": 8092, "train_positive": 414, "val_rows": 2695, "val_positive": 117, "rounds": [40, 40, 40],
    "val_logloss": 0.08702111214949164, "val_brier": 0.02979089110934733,
    "preds": [[0.0, 0.0], [0.29564087589581806, 0.005848295734206585], [0.2925604283809662, 0.008359824293482018],
              [0.38987237215042114, 0.08189371000159763]],
}


def build_tables(tmp: Path) -> None:
    rng = np.random.default_rng(11)
    rows, splits = [], []
    for d in range(450):
        model = ["MA", "MB", "MC"][d % 3]
        serial = f"S{d:04d}"
        sp = "train" if d % 10 < 6 else ("val" if d % 10 < 8 else "test")
        splits.append({"serial_number": serial, "model": model, "ever_failed": False, "split": sp})
        risk = rng.random() < 0.15
        for day in range(30):
            v = rng.normal(5 + 8 * risk, 2, size=3)
            if d % 7 == 0:
                v[1] = np.nan
            lab = int(risk and day > 20)
            if day == 29 and not risk and rng.random() < 0.05:
                lab = -1
            rows.append({"serial_number": serial, "date": pd.Timestamp("2022-01-01") + pd.Timedelta(days=day), "model": model,
                         **{c: float(x) for c, x in zip(COLS, v)},
                         **{c + "_missing": bool(np.isnan(x)) for c, x in zip(COLS, v)}, "label": lab})
    df = pd.DataFrame(rows)
    timelines = df.drop(columns=["label"])
    timelines["date"] = timelines["date"].dt.date
    labels = df[["serial_number", "date", "label"]].copy()
    labels["date"] = labels["date"].dt.date
    pq.write_table(pa.Table.from_pandas(timelines, preserve_index=False), tmp / "timelines.parquet", row_group_size=2000)
    pq.write_table(pa.Table.from_pandas(labels, preserve_index=False), tmp / "labels.parquet", row_group_size=2000)
    pd.DataFrame(splits).to_parquet(tmp / "split.parquet")


@pytest.fixture
def small(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    build_tables(tmp_path)
    monkeypatch.setattr(fit_v1, "TIMELINES", tmp_path / "timelines.parquet")
    monkeypatch.setattr(fit_v1, "LABELS", tmp_path / "labels.parquet")
    monkeypatch.setattr(fit_v1, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(fit_v1, "MAX_ROUNDS", 40)
    monkeypatch.setattr(fit_v1, "EARLY_STOPPING_ROUNDS", 10)
    monkeypatch.setattr(fit_v1, "BATCH_ROWS", 4000)
    monkeypatch.setattr(fit_v1, "select_device", lambda: ("cpu", "cpu"))
    return tmp_path


def _check(meta: dict, out: Path) -> None:
    ref = PRE_EDIT_REFERENCE
    assert (meta["train_rows"], meta["train_positive"], meta["val_rows"], meta["val_positive"]) == (
        ref["train_rows"], ref["train_positive"], ref["val_rows"], ref["val_positive"])
    assert [r["rounds"] for r in meta["per_model"]] == ref["rounds"]
    assert meta["ensemble"]["val_logloss"] == pytest.approx(ref["val_logloss"], rel=1e-6)
    assert meta["ensemble"]["val_brier"] == pytest.approx(ref["val_brier"], rel=1e-6)
    model = FittedHazardV1(out)
    for probe, expected in zip(PROBES, ref["preds"]):
        assert model(probe) == pytest.approx(tuple(expected), rel=1e-6, abs=1e-9)


def test_the_default_call_reproduces_the_pre_edit_fit(small: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fit_v1, "DRIVE_SPLIT", small / "split.parquet")
    out = small / "default_art"
    _check(fit_v1.fit(out_dir=out, seeds=(1, 2, 3)), out)
    assert (small / "manifest.json").exists()


def test_the_options_reproduce_the_default_and_write_only_where_they_are_told(small: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_manifest = fit_v1.REPO / "hazard" / "manifest.json"
    before = hashlib.sha256(repo_manifest.read_bytes()).hexdigest() if repo_manifest.exists() else None
    out, manifest = small / "other_art", small / "other_manifest.json"
    meta = fit_v1.fit(out_dir=out, seeds=(1, 2, 3), drive_split_path=small / "split.parquet", manifest_path=manifest)
    _check(meta, out)
    assert manifest.exists() and not (small / "manifest.json").exists()
    after = hashlib.sha256(repo_manifest.read_bytes()).hexdigest() if repo_manifest.exists() else None
    assert before == after


def test_a_split_without_a_vendor_fits_only_the_remaining_categories(small: Path) -> None:
    split = pd.read_parquet(small / "split.parquet")
    split.loc[split["model"] == "MC", "split"] = "vendor_holdout"
    split.to_parquet(small / "split_holdout.parquet")
    out = small / "no_mc"
    meta = fit_v1.fit(out_dir=out, seeds=(1, 2), drive_split_path=small / "split_holdout.parquet", manifest_path=small / "m.json")
    assert meta["model_categories"] == ["MA", "MB"]
    assert meta["train_rows"] < PRE_EDIT_REFERENCE["train_rows"]
    assert FittedHazardV1(out)({"smart_5_raw": 9.0, "model": "MC"})[0] >= 0.0  # an unseen model is a missing category


def test_the_command_line_options_are_given_together(small: Path) -> None:
    with pytest.raises(SystemExit):
        fit_v1.main(["--out", str(small / "x")])
