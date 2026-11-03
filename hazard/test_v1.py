"""Tests for the v1 fitted hazard model and the row selection of its fitting script.

The model tests load the committed artifacts in ``hazard/artifacts/v1/`` and
check interface conformance, bounds, missing-value handling and determinism.
None of them says anything about predictive accuracy. The selection tests run
on small synthetic frames and read no data files.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from contracts.contracts import HORIZON_HOURS, ContractViolation, HazardModel
from hazard import fit_v1
from hazard.v1 import FittedHazardV1, build_feature_frame, feature_columns

MODEL = FittedHazardV1()
SEAGATE_ONLY = ("smart_187_raw", "smart_187_raw_diff7d", "smart_187_raw_diff30d", "smart_188_raw_diff7d")
VALUE_COLUMNS = tuple(c for c in feature_columns() if c != "model" and not c.endswith("_missing"))


def realistic() -> dict:
    """All value columns present with plausible magnitudes and a Seagate model."""
    row = {name: 0.0 for name in VALUE_COLUMNS}
    row.update({"smart_9_raw": 21000.0, "smart_194_raw": 31.0, "smart_193_raw": 4200.0,
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


def test_loads_from_committed_artifacts_and_satisfies_protocol() -> None:
    assert isinstance(FittedHazardV1(), HazardModel)
    assert len(feature_columns()) == 57
    assert feature_columns()[-1] == "model"


def test_realistic_input_satisfies_contract_bounds_and_types() -> None:
    check_contract_output(MODEL(realistic()))
    check_contract_output(MODEL(realistic(), HORIZON_HOURS))


def test_all_missing_input_satisfies_contract_bounds_and_does_not_raise() -> None:
    check_contract_output(MODEL({}))


def test_toshiba_shaped_input_does_not_raise_and_absent_columns_stay_missing() -> None:
    row = toshiba_shaped()
    assert not set(SEAGATE_ONLY) & set(row)
    check_contract_output(MODEL(row))
    frame = build_feature_frame(row)
    for name in SEAGATE_ONLY:
        assert np.isnan(frame[name].iloc[0]), name
        assert frame[f"{name}_missing"].iloc[0] == 1.0, name
    assert frame["smart_5_raw"].iloc[0] == 0.0
    assert frame["smart_5_raw_missing"].iloc[0] == 0.0


def test_present_zero_differs_from_absent_in_the_feature_frame() -> None:
    present = build_feature_frame({"smart_187_raw": 0.0})
    absent = build_feature_frame({})
    assert present["smart_187_raw"].iloc[0] == 0.0 and present["smart_187_raw_missing"].iloc[0] == 0.0
    assert np.isnan(absent["smart_187_raw"].iloc[0]) and absent["smart_187_raw_missing"].iloc[0] == 1.0


@pytest.mark.parametrize("bad", [None, True, "x", float("nan"), float("inf"), float("-inf")])
def test_unusable_values_are_missing_not_zero(bad: object) -> None:
    frame = build_feature_frame({"smart_5_raw": bad})
    assert np.isnan(frame["smart_5_raw"].iloc[0]) and frame["smart_5_raw_missing"].iloc[0] == 1.0
    check_contract_output(MODEL({"smart_5_raw": bad}))


def test_negative_difference_is_a_value() -> None:
    frame = build_feature_frame({"smart_194_raw_diff7d": -3.0})
    assert frame["smart_194_raw_diff7d"].iloc[0] == -3.0
    assert frame["smart_194_raw_diff7d_missing"].iloc[0] == 0.0


def test_unknown_or_absent_model_is_a_missing_category() -> None:
    for row in ({"model": "NOT A MODEL"}, {"model": 3}, {}):
        assert pd.isna(build_feature_frame(row)["model"].iloc[0])
    assert build_feature_frame({"model": "ST4000DM000"})["model"].iloc[0] == "ST4000DM000"
    check_contract_output(MODEL({"model": "NOT A MODEL"}))


def test_frame_columns_follow_the_training_order() -> None:
    assert tuple(build_feature_frame(realistic()).columns) == feature_columns()


def test_repeated_calls_give_identical_output() -> None:
    for row in (realistic(), toshiba_shaped(), {}):
        assert MODEL(row) == MODEL(row)
    assert FittedHazardV1()(realistic()) == MODEL(realistic())


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), True, "720", None])
def test_invalid_horizon_raises_contract_violation(bad: object) -> None:
    with pytest.raises(ContractViolation):
        MODEL(realistic(), bad)


def test_horizon_other_than_the_fitted_one_raises_contract_violation() -> None:
    with pytest.raises(ContractViolation):
        MODEL(realistic(), 24.0)


@pytest.mark.parametrize("bad", [None, [1.0], "x", 3])
def test_non_mapping_features_raise_contract_violation(bad: object) -> None:
    with pytest.raises(ContractViolation):
        MODEL(bad)


def test_calibrator_interpolation_matches_isotonic_regression() -> None:
    from sklearn.isotonic import IsotonicRegression

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
