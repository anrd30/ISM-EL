"""``vendor_holdout_config`` and the held-out-model fleet generator; the default ``SimConfig`` is unchanged."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sim import config as cfg
from sim import scenario

TABLE = scenario.FLEET_FEATURES_VENDOR_HOLDOUT_PATH


def test_default_config_still_uses_the_train_fleet_and_the_pooled_artifacts() -> None:
    default = cfg.SimConfig()
    assert default.feature_fn is scenario.v1_drive_features
    assert default.hazard_model.directory.name == "v1"
    assert cfg.v1_eval_config().feature_fn is scenario.v1_eval_drive_features


@pytest.mark.skipif(not TABLE.exists(), reason="sim/fleet_features_vendor_holdout.json absent")
def test_the_vendor_holdout_table_holds_only_the_held_out_model() -> None:
    raw = json.loads(TABLE.read_text(encoding="utf-8"))
    assert raw["split"] == "vendor_holdout"
    assert raw["model_categories"] == ["TOSHIBA MG07ACA14TA"]
    assert raw["models"]["TOSHIBA MG07ACA14TA"]["share"] == 1.0
    rng = np.random.default_rng(0)
    drives = [scenario.v1_vendor_holdout_drive_features(rng, 0.3, 0.0, 0.0) for _ in range(20)]
    assert {d["model"] for d in drives} == {"TOSHIBA MG07ACA14TA"}


def test_vendor_holdout_config_replaces_only_the_hazard_model_and_the_fleet_generator() -> None:
    base, vendor = cfg.SimConfig(), cfg.vendor_holdout_config()
    assert vendor.feature_fn is scenario.v1_vendor_holdout_drive_features
    assert vendor.hazard_model.directory == cfg.VENDOR_HOLDOUT_ARTIFACT_DIR
    assert vendor.hazard_model.directory != base.hazard_model.directory
    import dataclasses

    for f in dataclasses.fields(base):
        if f.name not in {"hazard_model", "feature_fn"}:
            assert getattr(vendor, f.name) == getattr(base, f.name), f.name


def test_vendor_holdout_config_accepts_overrides() -> None:
    assert cfg.vendor_holdout_config(hazard_error_sigma=0.25).hazard_error_sigma == 0.25


@pytest.mark.skipif(not (cfg.VENDOR_HOLDOUT_ARTIFACT_DIR / "meta.json").exists(), reason="hazard/artifacts/v1_seagate_only absent")
def test_the_seagate_only_model_scores_the_held_out_model_as_an_unseen_category() -> None:
    meta = json.loads((cfg.VENDOR_HOLDOUT_ARTIFACT_DIR / "meta.json").read_text(encoding="utf-8"))
    assert "TOSHIBA MG07ACA14TA" not in meta["model_categories"]
    rng = np.random.default_rng(1)
    drive = scenario.v1_vendor_holdout_drive_features(rng, 0.0, 0.0, 0.0)
    p, u = cfg.vendor_holdout_config().hazard_model(drive)
    assert 0.0 <= p < 1.0 and u >= 0.0
