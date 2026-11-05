"""Tests for the pluggable fleet feature generators in ``sim.scenario``.

``golden_v0_fleet.json`` holds the output of ``sample_fleet`` with the v0 hazard
model, captured before the feature generation was moved behind the
``feature_fn`` parameter. Floats are stored as JSON round-trip representations,
so equality below is exact.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hazard.v0 import FEATURE_NAMES, HeuristicHazardV0
from hazard.v1 import ARTIFACT_DIR as V1_ARTIFACT_DIR
from hazard.v1 import FittedHazardV1, feature_columns
from sim.config import SimConfig, v0_config
from sim.recovery_env import RecoveryEnv
from sim.scenario import (
    FLEET_FEATURES_PATH,
    N_SURVIVORS,
    draw_v1_drive_features,
    load_fleet_table,
    sample_fleet,
    v0_drive_features,
    v1_drive_features,
)
from sim.stress import ScaledHazard, stress_config

GOLDEN = Path(__file__).resolve().parent / "golden_v0_fleet.json"


def _golden() -> list[dict]:
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def _assert_matches(case: dict, fleet) -> None:
    assert fleet.level == case["fleet_level"]
    assert list(fleet.features) == case["features"]
    assert fleet.pairs.tolist() == case["pairs"]
    assert fleet.interval_fail_prob.tolist() == case["interval_fail_prob"]
    assert np.array_equal(fleet.pairs, np.array(case["pairs"], dtype=np.float64))
    assert np.array_equal(fleet.interval_fail_prob, np.array(case["interval_fail_prob"], dtype=np.float64))


def test_v0_default_path_is_unchanged() -> None:
    cases = _golden()
    assert len(cases) == 40
    for case in cases:
        fleet = sample_fleet(
            np.random.default_rng(case["seed"]),
            HeuristicHazardV0(),
            level=case["level"],
            drive_spread_sd=case["drive_spread_sd"],
        )
        _assert_matches(case, fleet)


def test_v0_explicit_feature_fn_is_unchanged() -> None:
    for case in _golden():
        fleet = sample_fleet(
            np.random.default_rng(case["seed"]),
            HeuristicHazardV0(),
            level=case["level"],
            drive_spread_sd=case["drive_spread_sd"],
            feature_fn=v0_drive_features,
        )
        _assert_matches(case, fleet)


def test_v0_features_have_no_model_key() -> None:
    fleet = sample_fleet(np.random.default_rng(0), HeuristicHazardV0(), level=0.5)
    assert all(set(drive) == set(FEATURE_NAMES) for drive in fleet.features)


# ---------------------------------------------------------------------------
# v1 generator
# ---------------------------------------------------------------------------

_V1_PRESENT = (V1_ARTIFACT_DIR / "meta.json").exists()
_TABLE_PRESENT = FLEET_FEATURES_PATH.exists()


def _column(healthy: list[float], failure: list[float], structural: bool = False) -> dict:
    return {
        "structurally_missing": structural,
        "missing_fraction": 1.0 if structural else 0.0,
        "healthy": healthy,
        "failure": failure,
    }


def _tiny_table(tmp_path: Path) -> Path:
    """Two models; B has column ``y`` structurally missing and no failure values for ``x``."""
    table = {
        "value_columns": ["x", "y"],
        "model_categories": ["A", "B"],
        "models": {
            "A": {"share": 0.25, "columns": {"x": _column([1.0, 2.0], [101.0, 102.0]),
                                             "y": _column([3.0], [103.0])}},
            "B": {"share": 0.75, "columns": {"x": _column([4.0], []),
                                             "y": _column([], [], structural=True)}},
        },
    }
    path = tmp_path / "table.json"
    path.write_text(json.dumps(table), encoding="utf-8")
    return path


def test_tiny_table_pools_follow_the_level(tmp_path: Path) -> None:
    table = load_fleet_table(_tiny_table(tmp_path))
    rng = np.random.default_rng(0)
    for _ in range(200):
        low = draw_v1_drive_features(table, rng, 0.0)
        high = draw_v1_drive_features(table, rng, 1.0)
        assert low["x"] < 100.0
        assert high["x"] > 100.0 if high["model"] == "A" else high["x"] == 4.0


def test_tiny_table_structural_missing_and_empty_pool_fallback(tmp_path: Path) -> None:
    table = load_fleet_table(_tiny_table(tmp_path))
    rng = np.random.default_rng(1)
    seen = set()
    for _ in range(100):
        drive = draw_v1_drive_features(table, rng, 1.0)
        seen.add(drive["model"])
        if drive["model"] == "B":
            assert set(drive) == {"x", "model"} and drive["x"] == 4.0
        else:
            assert set(drive) == {"x", "y", "model"}
    assert seen == {"A", "B"}


def test_same_seed_gives_the_same_draw(tmp_path: Path) -> None:
    table = load_fleet_table(_tiny_table(tmp_path))
    first = [draw_v1_drive_features(table, np.random.default_rng(5), 0.5) for _ in range(3)]
    again = [draw_v1_drive_features(table, np.random.default_rng(5), 0.5) for _ in range(3)]
    assert first == again


def test_model_assignment_follows_the_shares(tmp_path: Path) -> None:
    table = load_fleet_table(_tiny_table(tmp_path))
    rng = np.random.default_rng(2)
    n = 20_000
    share_a = sum(draw_v1_drive_features(table, rng, 0.5)["model"] == "A" for _ in range(n)) / n
    assert abs(share_a - 0.25) < 0.02


def test_load_fleet_table_is_cached(tmp_path: Path) -> None:
    path = _tiny_table(tmp_path)
    assert load_fleet_table(path) is load_fleet_table(path)


@pytest.mark.skipif(not _TABLE_PRESENT, reason="sim/fleet_features_v1.json absent")
def test_committed_table_shares_and_structural_missing() -> None:
    table = load_fleet_table(FLEET_FEATURES_PATH)
    raw = json.loads(FLEET_FEATURES_PATH.read_text(encoding="utf-8"))
    assert table.shares.sum() == pytest.approx(1.0)
    rng = np.random.default_rng(3)
    counts = {m: 0 for m in table.categories}
    n = 5_000
    for _ in range(n):
        drive = draw_v1_drive_features(table, rng, float(rng.random()))
        counts[drive["model"]] += 1
        columns = raw["models"][drive["model"]]["columns"]
        expected = {c for c, e in columns.items() if not e["structurally_missing"]}
        assert set(drive) == expected | {"model"}
    for model, share in zip(table.categories, table.shares):
        assert abs(counts[model] / n - share) < 0.03
    toshiba = "TOSHIBA MG07ACA14TA"
    missing = {c for c, e in raw["models"][toshiba]["columns"].items() if e["structurally_missing"]}
    assert missing and all(c.startswith(("smart_187", "smart_188")) for c in missing)


@pytest.mark.skipif(not (_TABLE_PRESENT and _V1_PRESENT), reason="fleet table or hazard/artifacts/v1 absent")
def test_v1_generator_keys_match_the_model_inputs() -> None:
    meta = json.loads((V1_ARTIFACT_DIR / "meta.json").read_text(encoding="utf-8"))
    table = load_fleet_table(FLEET_FEATURES_PATH)
    assert list(table.value_columns) == meta["value_columns"]
    assert list(table.categories) == meta["model_categories"]
    assert set(feature_columns()) == set(meta["value_columns"]) | set(meta["flag_columns"]) | {"model"}
    rng = np.random.default_rng(4)
    for _ in range(50):
        drive = v1_drive_features(rng, 0.5, 40_000.0, 0.0)
        assert set(drive) <= set(meta["value_columns"]) | {"model"}
        assert not set(drive) & set(meta["flag_columns"])
        assert drive["model"] in meta["model_categories"]


@pytest.mark.skipif(not (_TABLE_PRESENT and _V1_PRESENT), reason="fleet table or hazard/artifacts/v1 absent")
def test_sample_fleet_with_v1_scores_every_drive() -> None:
    model = FittedHazardV1()
    fleet = sample_fleet(np.random.default_rng(0), model, level=0.5, feature_fn=v1_drive_features)
    assert len(fleet.features) == N_SURVIVORS
    assert np.all(np.isfinite(fleet.pairs)) and np.all(fleet.interval_fail_prob > 0.0)
    again = sample_fleet(np.random.default_rng(0), model, level=0.5, feature_fn=v1_drive_features)
    assert list(again.features) == list(fleet.features)
    assert np.array_equal(again.pairs, fleet.pairs)


# ---------------------------------------------------------------------------
# Configuration pairing
# ---------------------------------------------------------------------------


def test_v0_config_pairs_hazard_and_features() -> None:
    cfg = v0_config(fleet_degradation=0.2)
    assert isinstance(cfg.hazard_model, HeuristicHazardV0)
    assert cfg.feature_fn is v0_drive_features
    assert cfg.fleet_degradation == 0.2


def test_stress_config_default_stays_on_v0() -> None:
    cfg = stress_config()
    assert isinstance(cfg.hazard_model, ScaledHazard)
    assert isinstance(cfg.hazard_model.base, HeuristicHazardV0)
    assert cfg.feature_fn is v0_drive_features


@pytest.mark.skipif(not _V1_PRESENT, reason="hazard/artifacts/v1 absent")
def test_default_config_is_v1_with_the_v1_generator() -> None:
    cfg = SimConfig()
    assert isinstance(cfg.hazard_model, FittedHazardV1)
    assert cfg.feature_fn is v1_drive_features


@pytest.mark.skipif(not (_TABLE_PRESENT and _V1_PRESENT), reason="fleet table or hazard/artifacts/v1 absent")
def test_default_env_episode_runs_on_v1() -> None:
    env = RecoveryEnv(SimConfig(workload_profile="idle"))
    obs, _ = env.reset(seed=0)
    assert np.all(np.isfinite(obs))
    _, _, _, _, info = env.step(0)
    assert info["config_label"] == "default"
