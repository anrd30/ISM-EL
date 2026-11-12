"""The held-out evaluation entry point.

Three things are worth testing here and the rest is ``baselines.evaluate``'s
own suite: that the configuration label matches what ``baselines.evaluate``
builds under the same flags (so rows from the two scripts are comparable), that
the strict-holdout guard refuses an in-sample evaluation, and that a learned
policy actually runs inside the paired harness next to the baselines.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from agent.checkpoint import save_checkpoint
from agent.evaluate_policy import (
    REFERENCE_POLICY,
    _assert_held_out,
    build_config,
    evaluate_checkpoints,
    main,
)
from agent.ppo import PPOAgent, PPOConfig
from baselines.evaluate import _build_config as baselines_build_config
from hazard.v1 import ARTIFACT_DIR as V1_ARTIFACT_DIR
from sim.config import SimConfig

_V1_PRESENT = (V1_ARTIFACT_DIR / "meta.json").exists()
_needs_v1 = pytest.mark.skipif(not _V1_PRESENT, reason="hazard/artifacts/v1 absent")

PROFILES = Path(__file__).resolve().parents[1] / "data_pipeline" / "profiles"
TRAIN_CSV = PROFILES / "profile_diurnal_long_train.csv"
HOLDOUT_CSV = PROFILES / "profile_diurnal_long_holdout.csv"
_SPLIT_PRESENT = TRAIN_CSV.exists() and HOLDOUT_CSV.exists()
_needs_split = pytest.mark.skipif(not _SPLIT_PRESENT, reason="long profile halves absent")


class _Args:
    """The two fields ``baselines.evaluate._build_config`` reads."""

    def __init__(self, eval_fleet: bool, hazard_error_sigma: float) -> None:
        self.eval_fleet = eval_fleet
        self.hazard_error_sigma = hazard_error_sigma


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("eval_fleet", "sigma"),
    [(False, 0.0), (True, 0.0), (False, 0.25), (True, 0.25), (True, 1.0)],
)
def test_config_label_matches_the_baselines_script_under_the_same_flags(
    eval_fleet: bool, sigma: float
) -> None:
    """Otherwise rows from the two scripts would not be comparable by label."""
    mine = build_config(eval_fleet=eval_fleet, hazard_error_sigma=sigma)
    theirs = baselines_build_config(_Args(eval_fleet, sigma))
    assert mine.label == theirs.label
    assert mine.hazard_error_sigma == theirs.hazard_error_sigma == sigma
    assert mine.feature_fn is theirs.feature_fn


def test_default_flags_leave_the_label_alone() -> None:
    assert build_config(eval_fleet=False, hazard_error_sigma=0.0).label == SimConfig().label


# ---------------------------------------------------------------------------
# The strict-holdout guard
# ---------------------------------------------------------------------------


def test_guard_refuses_a_checkpoint_trained_on_the_evaluation_workload(tmp_path: Path) -> None:
    provenance = [{"checkpoint": "ckpt_a.pt", "trained_on": str(TRAIN_CSV)}]
    with pytest.raises(ValueError, match="in-sample"):
        _assert_held_out(provenance, TRAIN_CSV)


def test_guard_accepts_the_other_half_of_the_split() -> None:
    provenance = [{"checkpoint": "ckpt_a.pt", "trained_on": str(TRAIN_CSV)}]
    _assert_held_out(provenance, HOLDOUT_CSV)  # does not raise


def test_guard_compares_resolved_paths_not_strings() -> None:
    """A relative and an absolute spelling of one file must not look held out."""
    relative = Path("data_pipeline") / "profiles" / "profile_diurnal_long_train.csv"
    provenance = [{"checkpoint": "ckpt_a.pt", "trained_on": str(relative)}]
    with pytest.raises(ValueError, match="in-sample"):
        _assert_held_out(provenance, TRAIN_CSV)


def test_guard_names_every_offender_not_just_the_first() -> None:
    provenance = [
        {"checkpoint": "a.pt", "trained_on": str(TRAIN_CSV)},
        {"checkpoint": "b.pt", "trained_on": str(HOLDOUT_CSV)},
        {"checkpoint": "c.pt", "trained_on": str(TRAIN_CSV)},
    ]
    with pytest.raises(ValueError) as exc:
        _assert_held_out(provenance, TRAIN_CSV)
    assert "a.pt" in str(exc.value) and "c.pt" in str(exc.value)
    assert "b.pt" not in str(exc.value)


def test_guard_ignores_a_checkpoint_with_no_recorded_workload() -> None:
    """A generator-profile checkpoint records a profile name, not a path."""
    _assert_held_out([{"checkpoint": "a.pt", "trained_on": "mixed"}], TRAIN_CSV)
    _assert_held_out([{"checkpoint": "a.pt", "trained_on": None}], TRAIN_CSV)


# ---------------------------------------------------------------------------
# Running inside the paired harness
# ---------------------------------------------------------------------------


def _untrained_checkpoint(path: Path, seed: int, trained_on: Path) -> Path:
    agent = PPOAgent(PPOConfig(), seed=seed)
    save_checkpoint(
        path,
        {
            "agent": agent.state(),
            "meta": {
                "iteration": 1,
                "seed": seed,
                "profile": "mixed",
                "workload_csv": str(trained_on),
                "hazard_error_sigma": 0.25,
                "eval_fleet": True,
                "fleet_degradation": -1.0,
                "ablate_hazard": False,
            },
        },
    )
    return path


@_needs_v1
@_needs_split
def test_learned_policies_run_beside_the_baselines_on_identical_scenarios(tmp_path: Path) -> None:
    checkpoints = [
        _untrained_checkpoint(tmp_path / f"ckpt{s}.pt", s, TRAIN_CSV) for s in (0, 1)
    ]
    config = build_config(eval_fleet=True, hazard_error_sigma=0.25)
    episodes, learned = evaluate_checkpoints(
        checkpoints, workload=HOLDOUT_CSV, config=config, seeds=[0, 1]
    )

    assert learned == ["ppo_lagrangian seed0", "ppo_lagrangian seed1"]
    assert set(learned) <= set(episodes["policy"])
    assert REFERENCE_POLICY in set(episodes["policy"])
    # evaluate_checkpoints calls check_pairing, so reaching here means every
    # policy saw the same scenario per seed. Assert the fingerprint directly too.
    for seed, block in episodes.groupby("seed"):
        assert block["fleet_level"].nunique() == 1
        assert block["trace_start_row"].nunique() == 1
    assert episodes["steps"].min() > 0


@_needs_v1
@_needs_split
def test_main_writes_both_outputs_and_the_summary_records_provenance(tmp_path: Path) -> None:
    ckpt = _untrained_checkpoint(tmp_path / "ckpt0.pt", 0, TRAIN_CSV)
    out = tmp_path / "out"
    code = main(
        [
            "--checkpoint", str(ckpt),
            "--workload-csv", str(HOLDOUT_CSV),
            "--eval-fleet", "--hazard-error-sigma", "0.25",
            "--seeds", "2", "--out", str(out),
        ]
    )
    assert code == 0
    assert (out / "episodes.csv").is_file()
    text = (out / "summary.md").read_text(encoding="utf-8")
    # The setting a reader needs in order to trust the table.
    assert "eval-fleet,sigma=0.25" in text
    assert "profile_diurnal_long_train.csv" in text, "what it trained on"
    assert "profile_diurnal_long_holdout.csv" in text, "what it was evaluated on"
    assert "their mode (argmax)" in text
    assert f"minus {REFERENCE_POLICY}" in text


@_needs_split
def test_main_refuses_an_in_sample_evaluation(tmp_path: Path) -> None:
    """The guard has to fire before any episode runs, so this needs no artifacts."""
    ckpt = _untrained_checkpoint(tmp_path / "ckpt0.pt", 0, TRAIN_CSV)
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--checkpoint", str(ckpt),
                "--workload-csv", str(TRAIN_CSV),
                "--seeds", "1", "--out", str(tmp_path / "out"),
            ]
        )
    assert exc.value.code != 0
    assert not (tmp_path / "out").exists(), "nothing is written when the guard fires"


@_needs_split
def test_no_strict_holdout_permits_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The escape hatch exists; check it reaches the harness rather than exiting."""
    ckpt = _untrained_checkpoint(tmp_path / "ckpt0.pt", 0, TRAIN_CSV)

    reached: dict[str, object] = {}

    def _stop(*args: object, **kwargs: object) -> None:
        reached["workload"] = kwargs.get("workload")
        raise RuntimeError("stop before the hazard model is touched")

    monkeypatch.setattr("agent.evaluate_policy.evaluate_checkpoints", _stop)
    with pytest.raises(RuntimeError, match="stop before"):
        main(
            [
                "--checkpoint", str(ckpt),
                "--workload-csv", str(TRAIN_CSV), "--no-strict-holdout",
                "--seeds", "1", "--out", str(tmp_path / "out"),
            ]
        )
    assert reached["workload"] == TRAIN_CSV


@pytest.mark.parametrize("bad", [["--hazard-error-sigma", "-1"], ["--seeds", "0"]])
def test_main_rejects_out_of_range_flags(tmp_path: Path, bad: list[str]) -> None:
    ckpt = _untrained_checkpoint(tmp_path / "ckpt0.pt", 0, TRAIN_CSV)
    with pytest.raises(SystemExit) as exc:
        main(["--checkpoint", str(ckpt), "--workload-csv", str(HOLDOUT_CSV),
              "--out", str(tmp_path / "out"), *bad])
    assert exc.value.code != 0


def test_main_rejects_a_missing_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--checkpoint", str(tmp_path / "absent.pt"),
              "--workload-csv", str(HOLDOUT_CSV), "--out", str(tmp_path / "out")])
    assert exc.value.code != 0
