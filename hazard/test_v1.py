"""Tests for the v1 fitted hazard model and the row selection of its fitting script.

The model tests run against a tiny, real (not mocked) 2-booster calibrated
ensemble built in ``_build_tiny_artifacts`` below, in the exact file layout
``hazard/v1.py`` reads. This exercises the real load/predict/calibrate path
without the 218 MB production artifacts of ``hazard/artifacts/v1/`` (gitignored,
produced only by a real, ~95-minute ``python -m hazard.fit_v1`` run), so this
file collects and passes on a bare clone. They check interface conformance,
bounds, missing-value handling and determinism -- none of them says anything
about predictive accuracy.

One test, marked skipped when ``hazard/artifacts/v1/`` is absent, checks the
shape of the real committed-but-gitignored production artifacts when they do
exist locally (after running the fit).

The selection tests run on small synthetic frames and read no data files.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression

from contracts.contracts import HORIZON_HOURS, ContractViolation, HazardModel
from hazard import fit_v1
from hazard.v1 import ARTIFACT_DIR, FittedHazardV1, build_feature_frame, feature_columns

SEAGATE_ONLY = ("smart_187_raw", "smart_187_raw_diff7d", "smart_187_raw_diff30d", "smart_188_raw_diff7d")
VALUE_COLUMNS = ("smart_5_raw", "smart_9_raw", "smart_12_raw", *SEAGATE_ONLY[:3], SEAGATE_ONLY[3],
                  "smart_193_raw", "smart_194_raw", "smart_194_raw_diff7d")
VALUE_COLUMNS = tuple(dict.fromkeys(VALUE_COLUMNS))  # de-duplicate, keep order
FLAG_COLUMNS = tuple(f"{name}_missing" for name in VALUE_COLUMNS)
MODEL_CATEGORIES = ("ST4000DM000", "ST12000NM0008", "TOSHIBA MG07ACA14TA")
TINY_FEATURE_COLUMNS = (*VALUE_COLUMNS, *FLAG_COLUMNS, "model")

_REAL_ARTIFACTS_PRESENT = (ARTIFACT_DIR / "meta.json").exists()


def _synthetic_frame(n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """``n`` synthetic rows with finite float32 values (never missing) and a 0/1 label."""
    data = {name: rng.normal(size=n).astype(np.float32) for name in VALUE_COLUMNS}
    data.update({name: np.zeros(n, dtype=np.float32) for name in FLAG_COLUMNS})
    data["model"] = pd.Categorical(rng.choice(MODEL_CATEGORIES, size=n), categories=MODEL_CATEGORIES)
    label = (rng.random(n) < 0.3).astype(np.float32)
    return pd.DataFrame(data)[list(TINY_FEATURE_COLUMNS)], label


def _build_tiny_artifacts(directory: Path) -> None:
    """Write a tiny 2-booster calibrated ensemble to ``directory`` in the real file layout.

    Not a mock: a real (if minuscule) ``xgb.train`` fit and a real
    ``IsotonicRegression`` calibration, so ``hazard/v1.py``'s load/predict/
    calibrate path is genuinely exercised.
    """
    rng = np.random.default_rng(0)
    train_frame, train_label = _synthetic_frame(60, rng)
    val_frame, val_label = _synthetic_frame(40, rng)
    dtrain = xgb.DMatrix(train_frame, label=train_label, enable_categorical=True)
    dval = xgb.DMatrix(val_frame, label=val_label, enable_categorical=True)
    params = {"objective": "binary:logistic", "tree_method": "hist", "max_depth": 2}
    boosters, calibrators = [], []
    for seed in (0, 1):
        booster = xgb.train({**params, "seed": seed}, dtrain, num_boost_round=5)
        raw = booster.predict(dval)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, val_label)
        boosters.append(booster)
        calibrators.append((iso.X_thresholds_.astype(np.float64), iso.y_thresholds_.astype(np.float64)))
    directory.mkdir(parents=True, exist_ok=True)
    names = [f"booster_{i:02d}.ubj" for i in range(len(boosters))]
    for name, booster in zip(names, boosters):
        booster.save_model(directory / name)
    np.savez(directory / "calibrators.npz",
             **{f"x_{i:02d}": x for i, (x, _) in enumerate(calibrators)},
             **{f"y_{i:02d}": y for i, (_, y) in enumerate(calibrators)})
    meta = {
        "value_columns": list(VALUE_COLUMNS), "flag_columns": list(FLAG_COLUMNS),
        "model_categories": list(MODEL_CATEGORIES), "feature_columns": list(TINY_FEATURE_COLUMNS),
        "boosters": names, "calibrators": "calibrators.npz",
    }
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("hazard_v1_tiny")
    _build_tiny_artifacts(directory)
    return directory


@pytest.fixture(scope="module")
def tiny_model(tiny_dir: Path) -> FittedHazardV1:
    return FittedHazardV1(directory=tiny_dir)


def realistic() -> dict:
    """All value columns present with plausible magnitudes and a Seagate model."""
    row = {name: 0.0 for name in VALUE_COLUMNS}
    row.update({"smart_9_raw": 21000.0, "smart_194_raw": 31.0, "smart_193_raw": 0.0,
                "smart_12_raw": 38.0, "smart_194_raw_diff7d": -1.0, "model": "ST12000NM0008"})
    return row


def toshiba_shaped() -> dict:
    row = {k: v for k, v in realistic().items() if k not in SEAGATE_ONLY}
    row["model"] = "TOSHIBA MG07ACA14TA"
    return row


def check_contract_output(out: tuple) -> None:
    assert isinstance(out, tuple) and len(out) == 2
    p, u = out
    assert type(p) is float and type(u) is float
    assert 0.0 <= p < 1.0
    assert 0.0 <= u <= 0.5


# ---- model -----------------------------------------------------------------


@pytest.mark.skipif(not _REAL_ARTIFACTS_PRESENT,
                     reason="hazard/artifacts/v1/ is gitignored; run `python -m hazard.fit_v1` to generate it")
def test_split_batches_class_subclasses_the_xgboost_iterator_and_is_cached() -> None:
    cls = fit_v1.split_batches_class()
    assert issubclass(cls, xgb.DataIter) and cls.__name__ == "SplitBatches"
    assert fit_v1.split_batches_class() is cls


def test_loads_from_committed_artifacts_and_satisfies_protocol() -> None:
    assert isinstance(FittedHazardV1(), HazardModel)
    assert len(feature_columns()) == 57
    assert feature_columns()[-1] == "model"


def test_realistic_input_satisfies_contract_bounds_and_types(tiny_model: FittedHazardV1) -> None:
    check_contract_output(tiny_model(realistic()))
    check_contract_output(tiny_model(realistic(), HORIZON_HOURS))


def test_all_missing_input_satisfies_contract_bounds_and_does_not_raise(tiny_model: FittedHazardV1) -> None:
    check_contract_output(tiny_model({}))


def test_toshiba_shaped_input_does_not_raise_and_absent_columns_stay_missing(
    tiny_model: FittedHazardV1, tiny_dir: Path
) -> None:
    row = toshiba_shaped()
    assert not set(SEAGATE_ONLY) & set(row)
    check_contract_output(tiny_model(row))
    frame = build_feature_frame(row, tiny_dir)
    for name in SEAGATE_ONLY:
        assert np.isnan(frame[name].iloc[0]), name
        assert frame[f"{name}_missing"].iloc[0] == 1.0, name
    assert frame["smart_5_raw"].iloc[0] == 0.0
    assert frame["smart_5_raw_missing"].iloc[0] == 0.0


def test_present_zero_differs_from_absent_in_the_feature_frame(tiny_dir: Path) -> None:
    present = build_feature_frame({"smart_187_raw": 0.0}, tiny_dir)
    absent = build_feature_frame({}, tiny_dir)
    assert present["smart_187_raw"].iloc[0] == 0.0 and present["smart_187_raw_missing"].iloc[0] == 0.0
    assert np.isnan(absent["smart_187_raw"].iloc[0]) and absent["smart_187_raw_missing"].iloc[0] == 1.0


@pytest.mark.parametrize("bad", [None, True, "x", float("nan"), float("inf"), float("-inf")])
def test_unusable_values_are_missing_not_zero(bad: object, tiny_model: FittedHazardV1, tiny_dir: Path) -> None:
    frame = build_feature_frame({"smart_5_raw": bad}, tiny_dir)
    assert np.isnan(frame["smart_5_raw"].iloc[0]) and frame["smart_5_raw_missing"].iloc[0] == 1.0
    check_contract_output(tiny_model({"smart_5_raw": bad}))


def test_value_that_overflows_float32_is_missing_not_infinite(tiny_dir: Path) -> None:
    # 1e100 is finite in float64 but overflows float32 (the frame's dtype) to
    # inf, which np.isnan would not catch as missing.
    frame = build_feature_frame({"smart_5_raw": 1e100}, tiny_dir)
    assert np.isnan(frame["smart_5_raw"].iloc[0])
    assert frame["smart_5_raw_missing"].iloc[0] == 1.0
    assert not np.isinf(frame["smart_5_raw"].iloc[0])


def test_negative_difference_is_a_value(tiny_dir: Path) -> None:
    frame = build_feature_frame({"smart_194_raw_diff7d": -3.0}, tiny_dir)
    assert frame["smart_194_raw_diff7d"].iloc[0] == -3.0
    assert frame["smart_194_raw_diff7d_missing"].iloc[0] == 0.0


def test_unknown_or_absent_model_is_a_missing_category(tiny_model: FittedHazardV1, tiny_dir: Path) -> None:
    for row in ({"model": "NOT A MODEL"}, {"model": 3}, {}):
        assert pd.isna(build_feature_frame(row, tiny_dir)["model"].iloc[0])
    assert build_feature_frame({"model": "ST4000DM000"}, tiny_dir)["model"].iloc[0] == "ST4000DM000"
    check_contract_output(tiny_model({"model": "NOT A MODEL"}))


def test_frame_columns_follow_the_training_order(tiny_dir: Path) -> None:
    assert tuple(build_feature_frame(realistic(), tiny_dir).columns) == feature_columns(tiny_dir)


def test_repeated_calls_give_identical_output(tiny_model: FittedHazardV1, tiny_dir: Path) -> None:
    for row in (realistic(), toshiba_shaped(), {}):
        assert tiny_model(row) == tiny_model(row)
    assert FittedHazardV1(directory=tiny_dir)(realistic()) == tiny_model(realistic())


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), True, "720", None])
def test_invalid_horizon_raises_contract_violation(bad: object, tiny_model: FittedHazardV1) -> None:
    with pytest.raises(ContractViolation):
        tiny_model(realistic(), bad)


def test_horizon_other_than_the_fitted_one_raises_contract_violation(tiny_model: FittedHazardV1) -> None:
    with pytest.raises(ContractViolation):
        tiny_model(realistic(), 24.0)


@pytest.mark.parametrize("bad", [None, [1.0], "x", 3])
def test_non_mapping_features_raise_contract_violation(bad: object, tiny_model: FittedHazardV1) -> None:
    with pytest.raises(ContractViolation):
        tiny_model(bad)


def test_calibrator_interpolation_matches_isotonic_regression() -> None:
    rng = np.random.default_rng(0)
    raw = rng.random(500)
    y = (rng.random(500) < raw).astype(float)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y)
    probe = np.concatenate([raw, [-0.5, 1.5]])
    assert np.allclose(iso.predict(probe), np.interp(probe, iso.X_thresholds_, iso.y_thresholds_))


# ---- fitting script: row selection ----------------------------------------


def synthetic() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Six drives, one per (split, kind), and their label rows."""
    drive_split = pd.DataFrame({
        "serial_number": ["a", "b", "c", "d", "e", "f"],
        "model": ["m"] * 6,
        "split": ["train", "train", "val", "val", "x", "x"],
    })
    labels = pd.DataFrame({
        "serial_number": ["a", "a", "a", "b", "b", "c", "c", "d", "e", "e", "f", "ghost"],
        "label":         [1,   0,  -1,   0,  -1,   1,  -1,   0,   1,   0,  -1,   0],
    })
    return labels, drive_split


def test_select_rows_keeps_exactly_the_train_rows_with_a_label() -> None:
    labels, drive_split = synthetic()
    mask = fit_v1.select_rows(labels, drive_split, fit_v1.TRAIN_SPLIT)
    kept = labels[mask]
    assert list(kept["serial_number"]) == ["a", "a", "b"]
    assert list(kept["label"]) == [1, 0, 0]


def test_select_rows_for_the_validation_split() -> None:
    labels, drive_split = synthetic()
    kept = labels[fit_v1.select_rows(labels, drive_split, fit_v1.VAL_SPLIT)]
    assert list(kept["serial_number"]) == ["c", "d"] and list(kept["label"]) == [1, 0]


def test_select_rows_is_aligned_with_the_input_rows() -> None:
    labels, drive_split = synthetic()
    mask = fit_v1.select_rows(labels.sample(frac=1.0, random_state=1).reset_index(drop=True), drive_split,
                              fit_v1.TRAIN_SPLIT)
    assert mask.dtype == bool and len(mask) == len(labels) and mask.sum() == 3


def test_select_rows_rejects_any_other_split_name() -> None:
    labels, drive_split = synthetic()
    for name in ("x", "", "TRAIN", "all"):
        with pytest.raises(ValueError):
            fit_v1.select_rows(labels, drive_split, name)
    assert fit_v1.ALLOWED_SPLITS == ("train", "val")


def test_fit_script_has_no_reference_to_the_held_out_split() -> None:
    source = Path(fit_v1.__file__).read_text(encoding="utf-8")
    assert not re.search(r"(?<![a-z])test(?![a-z])", source, flags=re.IGNORECASE)
    assert not re.search(r"(?<![a-z])testing(?![a-z])", source, flags=re.IGNORECASE)
    assert "train" in source and "val" in source


def test_fit_script_derives_categories_only_from_the_allowed_splits() -> None:
    source = Path(fit_v1.__file__).read_text(encoding="utf-8")
    assert 'drive_split["split"].isin(ALLOWED_SPLITS)' in source


def test_bootstrap_weights_are_draw_counts_and_seeded() -> None:
    w = fit_v1.bootstrap_weights(5, 1000)
    assert w.sum() == 1000 and (w >= 0).all() and (w == 0).any()
    assert np.array_equal(w, fit_v1.bootstrap_weights(5, 1000))
    assert not np.array_equal(w, fit_v1.bootstrap_weights(6, 1000))


def test_seeds_are_distinct_and_fixed() -> None:
    assert len(fit_v1.SEEDS) == fit_v1.N_MODELS == 25 and len(set(fit_v1.SEEDS)) == 25


def test_feature_columns_pairs_each_flag_with_its_value_column() -> None:
    names = ["serial_number", "a", "a_diff7d", "model", "a_missing", "a_diff7d_missing"]
    assert fit_v1.feature_columns(names) == (["a", "a_diff7d"], ["a_missing", "a_diff7d_missing"])
    with pytest.raises(ValueError):
        fit_v1.feature_columns(["b_missing"])


def test_log_loss_and_brier_on_known_values() -> None:
    y = np.array([1.0, 0.0])
    p = np.array([0.5, 0.5])
    assert math.isclose(fit_v1.log_loss(y, p), math.log(2.0))
    assert math.isclose(fit_v1.brier(y, p), 0.25)


def test_write_manifest_matches_data_pipelines_schema() -> None:
    # write_manifest records each file's path relative to the repository root
    # (matching data_pipeline/manifest.json), so the directory it hashes must
    # itself be under REPO -- a pytest tmp_path outside the repo would raise.
    out_dir = fit_v1.REPO / "hazard" / "_test_write_manifest_tmp"
    out_dir.mkdir(exist_ok=True)
    try:
        (out_dir / "a.bin").write_bytes(b"hello")
        (out_dir / "b.bin").write_bytes(b"world")
        manifest_path = out_dir / "manifest.json"
        manifest = fit_v1.write_manifest(out_dir, manifest_path)
        assert {e["file"] for e in manifest["files"]} == {
            "hazard/_test_write_manifest_tmp/a.bin",
            "hazard/_test_write_manifest_tmp/b.bin",
        }
        assert all(e["rows"] is None and len(e["sha256"]) == 64 for e in manifest["files"])
        assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    finally:
        for name in ("a.bin", "b.bin", "manifest.json"):
            (out_dir / name).unlink(missing_ok=True)
        out_dir.rmdir()
