"""Vendor-holdout split: drive-level partition, the held-out model entirely outside fitting, and no cross-drive leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from data_pipeline.backblaze import labels as a2
from data_pipeline.backblaze import timelines
from data_pipeline.backblaze import vendor_holdout_split as vh
from hazard import fit_v1

MODELS = ("ST12000NM0008", "ST4000DM000", "ST8000NM0055", vh.HOLDOUT_MODEL)
NEW_SPLIT = vh.OUT_SPLIT
A2_SPLIT = a2.OUT_SPLIT
needs_real = pytest.mark.skipif(
    not (NEW_SPLIT.exists() and A2_SPLIT.exists() and vh.IN_LABELS.exists() and vh.IN_INDEX.exists()),
    reason="processed Backblaze tables or the vendor-holdout split are not built",
)


def synthetic_index(n_per_stratum: int = 23) -> pd.DataFrame:
    rows = []
    for m, model in enumerate(MODELS):
        for failed in (False, True):
            for k in range(n_per_stratum + 3 * m):
                rows.append({"serial_number": f"{model[:4]}-{int(failed)}-{k:04d}", "model": model, "failed_in_window": failed})
    return pd.DataFrame(rows)


def test_every_drive_is_in_exactly_one_bucket_and_the_holdout_model_only_in_the_holdout() -> None:
    index = synthetic_index()
    split = vh.assign_vendor_holdout(index)
    assert len(split) == len(index) and split["serial_number"].is_unique
    assert set(split["serial_number"]) == set(index["serial_number"])
    toshiba = split[split["model"] == vh.HOLDOUT_MODEL]
    assert (toshiba["split"] == vh.HOLDOUT_BUCKET).all() and len(toshiba) > 0
    others = split[split["model"] != vh.HOLDOUT_MODEL]
    assert set(others["split"]) == set(a2.SPLITS)
    assert not (others["split"] == vh.HOLDOUT_BUCKET).any()


def test_the_other_models_are_split_exactly_as_a2_splits_them() -> None:
    index = synthetic_index()
    new = vh.assign_vendor_holdout(index).set_index("serial_number")
    a2_full = a2.assign_splits(index).set_index("serial_number")
    seagate = a2_full[a2_full["model"] != vh.HOLDOUT_MODEL]
    assert new.loc[seagate.index, "split"].equals(seagate["split"])


def test_the_split_is_deterministic_and_independent_of_the_row_order() -> None:
    index = synthetic_index()
    a = vh.assign_vendor_holdout(index)
    b = vh.assign_vendor_holdout(index.sample(frac=1.0, random_state=1))
    pd.testing.assert_frame_equal(a, b)


def test_strata_of_the_fitted_models_are_within_one_drive_of_the_a2_shares() -> None:
    split = vh.assign_vendor_holdout(synthetic_index(41))
    for (model, failed), g in split[split["model"] != vh.HOLDOUT_MODEL].groupby(["model", "ever_failed"]):
        for name, fraction in zip(a2.SPLITS, a2.FRACTIONS):
            assert abs((g["split"] == name).sum() - fraction * len(g)) < 1.0 + 1e-9, (model, failed, name)


def test_an_absent_holdout_model_is_an_error() -> None:
    with pytest.raises(ValueError):
        vh.assign_vendor_holdout(synthetic_index(), holdout_model="NOT A MODEL")


def test_the_held_out_bucket_cannot_be_selected_for_fitting() -> None:
    assert vh.HOLDOUT_BUCKET not in fit_v1.ALLOWED_SPLITS
    labels = pd.DataFrame({"serial_number": ["a", "b"], "label": [0, 1]})
    drive_split = pd.DataFrame({"serial_number": ["a", "b"], "split": [vh.HOLDOUT_BUCKET, "train"]})
    with pytest.raises(ValueError):
        fit_v1.select_rows(labels, drive_split, vh.HOLDOUT_BUCKET)
    assert fit_v1.select_rows(labels, drive_split, "train").tolist() == [False, True]


def _frame():
    from data_pipeline.tests import test_no_leakage as leak

    return leak.synthetic_frame(seed=5)


def test_removing_the_held_out_model_changes_no_feature_of_another_drive() -> None:
    """Timeline features are trailing per drive: the rows of the fitted models are identical with or without the holdout's rows."""
    df = _frame()
    full = timelines.build_timelines(df)
    kept = df[df["model"] != "TOSHIBA"]
    reduced = timelines.build_timelines(kept)
    expected = full[full["model"] != "TOSHIBA"].reset_index(drop=True)
    pd.testing.assert_frame_equal(reduced.reset_index(drop=True), expected, check_exact=True)


def test_prefix_identity_still_holds_for_the_fitted_models_alone() -> None:
    from data_pipeline.tests import test_no_leakage as leak

    df = _frame()
    df = df[df["model"] != "TOSHIBA"]
    leak.assert_prefix_identical(df, leak.cut_date(df, 0.5))


@needs_real
def test_the_real_split_matches_a2_for_the_fitted_models_and_holds_out_every_toshiba_drive() -> None:
    new = pd.read_parquet(NEW_SPLIT)
    old = pd.read_parquet(A2_SPLIT).set_index("serial_number")
    index = pd.read_parquet(vh.IN_INDEX)
    assert len(new) == len(index) and new["serial_number"].is_unique
    toshiba = new[new["model"] == vh.HOLDOUT_MODEL]
    assert (toshiba["split"] == vh.HOLDOUT_BUCKET).all()
    fitted = new[new["model"] != vh.HOLDOUT_MODEL].set_index("serial_number")
    assert fitted["split"].equals(old.loc[fitted.index, "split"])
    assert not set(fitted["split"]) & {vh.HOLDOUT_BUCKET}


@needs_real
def test_every_label_row_belongs_to_one_known_drive_and_no_fit_row_is_toshiba() -> None:
    new = pd.read_parquet(NEW_SPLIT).set_index("serial_number")
    pf = pq.ParquetFile(vh.IN_LABELS)
    seen_fit_models: set[str] = set()
    for i in range(0, pf.num_row_groups, max(1, pf.num_row_groups // 12)):
        frame = pf.read_row_group(i, columns=["serial_number", "model", "label"]).to_pandas()
        bucket = new["split"].reindex(frame["serial_number"]).to_numpy()
        assert not pd.isna(bucket).any()
        for name in ("train", "val"):
            mask = fit_v1.select_rows(frame[["serial_number", "label"]], new.reset_index()[["serial_number", "split"]], name)
            seen_fit_models |= set(frame.loc[mask, "model"].astype(str))
    assert vh.HOLDOUT_MODEL not in seen_fit_models and seen_fit_models


@needs_real
def test_the_recorded_counts_are_the_bucket_sizes() -> None:
    new = pd.read_parquet(NEW_SPLIT)
    sizes = new.groupby("split").size().to_dict()
    assert sizes == {"test": 8091, "train": 37763, "val": 8092, "vendor_holdout": 38579}
    assert int(new.loc[new["split"] == "vendor_holdout", "ever_failed"].sum()) == 385
