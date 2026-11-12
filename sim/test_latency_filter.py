"""Tests for the latency ceiling (``sim/latency_filter.py``) and its wiring in ``RecoveryEnv.step``.

Every test uses ``v0_config`` so none needs ``hazard/artifacts/v1/``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest

from contracts.contracts import ACTIONS_MBPS, BW_TOTAL_MBPS, LATENCY_LIMIT_FACTOR, N_ACTIONS, validate_info
from sim.config import SimConfig, latency_mean_ms, v0_config, violation_rho_threshold
from sim.latency_filter import apply_latency_filter, latency_ceiling_index
from sim.recovery_env import RecoveryEnv
from hazard.v1 import ARTIFACT_DIR as V1_ARTIFACT_DIR
from sim.stress import stress_config

_V1_PRESENT = (V1_ARTIFACT_DIR / "meta.json").exists()

LATENCY_ONLY_KEYS = (
    "latency_filter_active",
    "latency_filter_fg_source",
    "latency_filter_fg_mbps",
    "alloc_after_reliability_filter_mbps",
    "latency_ceiling_mbps",
    "latency_ceiling_feasible",
    "alloc_clipped_by_latency_filter",
    "latency_filter_overridden_by_reliability",
    "reliability_filter_clipped",
    "both_filters_fired",
)


def _run(cfg: SimConfig, seed: int, actions: list[int] | None = None, steps: int = 200) -> list[tuple[Any, ...]]:
    env = RecoveryEnv(cfg)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    out: list[tuple[Any, ...]] = []
    for t in range(steps):
        a = int(actions[t % len(actions)]) if actions else int(rng.integers(0, N_ACTIONS))
        obs, reward, term, trunc, info = env.step(a)
        out.append((obs.copy(), reward, term, trunc, info))
        if term or trunc:
            break
    return out


def test_default_config_has_the_filter_off() -> None:
    assert v0_config().latency_filter is False


def test_flag_off_is_bit_identical_to_the_unchanged_simulator() -> None:
    """An explicit ``latency_filter=False`` and the default produce identical trajectories, with no new info keys."""
    default = v0_config(workload_profile="peak")
    explicit = v0_config(workload_profile="peak", latency_filter=False)
    for seed in (0, 1, 2):
        a, b = _run(default, seed), _run(explicit, seed)
        assert len(a) == len(b) > 0
        for (oa, ra, ta, tra, ia), (ob, rb, tb, trb, ib) in zip(a, b):
            np.testing.assert_array_equal(oa, ob)
            assert (ra, ta, tra) == (rb, tb, trb)
            assert ia == ib
            assert not any(k in ia for k in LATENCY_ONLY_KEYS)


def test_ceiling_index_matches_the_utilisation_threshold() -> None:
    cfg = v0_config()
    threshold = violation_rho_threshold(cfg)
    assert threshold == pytest.approx(0.9111, abs=1e-3)
    for fg in (0.0, 20.0, 62.0, 63.0, 100.0, 130.0, 150.0, 162.0, 163.0, 180.0, 190.0):
        index, feasible = latency_ceiling_index(fg, cfg)
        under = [i for i in range(N_ACTIONS) if (fg + ACTIONS_MBPS[i]) / BW_TOTAL_MBPS <= threshold]
        assert feasible == bool(under)
        assert index == (max(under) if under else 0)
    assert latency_ceiling_index(0.0, cfg) == (N_ACTIONS - 1, True)
    assert latency_ceiling_index(190.0, cfg) == (0, False)
    with pytest.raises(ValueError):
        latency_ceiling_index(-1.0, cfg)


def test_filter_only_moves_an_action_down_and_respects_the_floor() -> None:
    cfg = v0_config()
    for fg in (0.0, 70.0, 110.0, 150.0, 185.0):
        for floor in range(N_ACTIONS):
            for requested in range(floor, N_ACTIONS):
                d = apply_latency_filter(requested, fg, cfg, floor)
                assert d.applied_index <= requested
                assert d.applied_index >= floor
                assert d.applied_mbps == ACTIONS_MBPS[d.applied_index]
                assert d.clipped == (d.applied_index < requested)
                if d.reliability_overrode:
                    assert d.applied_index == floor > d.ceiling_index
    with pytest.raises(ValueError):
        apply_latency_filter(N_ACTIONS, 0.0, cfg)


def test_floor_above_ceiling_is_recorded_as_reliability_overriding() -> None:
    cfg = v0_config()
    ceiling, _ = latency_ceiling_index(150.0, cfg)
    assert ceiling < N_ACTIONS - 1
    d = apply_latency_filter(N_ACTIONS - 1, 150.0, cfg, floor_index=N_ACTIONS - 1)
    assert d.applied_index == N_ACTIONS - 1 and not d.clipped and d.reliability_overrode
    d = apply_latency_filter(N_ACTIONS - 1, 150.0, cfg, floor_index=0)
    assert d.applied_index == ceiling and d.clipped and not d.reliability_overrode


def test_filter_on_is_inert_when_the_request_is_already_under_the_ceiling() -> None:
    """Requesting the smallest action everywhere leaves the trajectory unchanged (extra info keys aside)."""
    base = v0_config(workload_profile="peak")
    on = dataclasses.replace(base, latency_filter=True)
    for seed in (0, 1):
        a, b = _run(base, seed, actions=[0]), _run(on, seed, actions=[0])
        assert len(a) == len(b) > 0
        for (oa, ra, ta, tra, ia), (ob, rb, tb, trb, ib) in zip(a, b):
            np.testing.assert_array_equal(oa, ob)
            assert (ra, ta, tra) == (rb, tb, trb)
            assert {k: v for k, v in ib.items() if k not in LATENCY_ONLY_KEYS} == ia
            assert not ib["alloc_clipped_by_latency_filter"]


def test_ceiling_keeps_latency_under_the_limit_on_the_peak_profile() -> None:
    """Always requesting 120 MB/s on the peak profile violates latency without the filter and not with it."""
    base = v0_config(workload_profile="peak", fleet_degradation=0.0)
    on = dataclasses.replace(base, latency_filter=True)
    limit_ms = LATENCY_LIMIT_FACTOR * on.latency_target_ms
    n_off_violations = n_clipped = n_checked = 0
    for seed in range(3):
        for _, _, _, _, info in _run(base, seed, actions=[N_ACTIONS - 1], steps=250):
            n_off_violations += int(info["violated_latency"])
        for _, _, _, _, info in _run(on, seed, actions=[N_ACTIONS - 1], steps=250):
            validate_info(info)
            n_clipped += int(info["alloc_clipped_by_latency_filter"])
            if info["latency_ceiling_feasible"] and not info["latency_filter_overridden_by_reliability"]:
                n_checked += 1
                assert info["alloc_applied_mbps"] <= info["latency_ceiling_mbps"]
                rho = (info["fg_bw_mbps"] + info["alloc_effective_mbps"]) / BW_TOTAL_MBPS
                assert rho <= violation_rho_threshold(on)
                assert latency_mean_ms(rho, on) <= limit_ms
                assert info["latency_now"] <= limit_ms
                assert info["violated_latency"] is False
    assert n_off_violations > 0, "the unfiltered peak run should violate, otherwise the test shows nothing"
    assert n_clipped > 0 and n_checked > 0


def test_applied_exposure_and_flags_describe_the_final_action() -> None:
    cfg = v0_config(workload_profile="peak", latency_filter=True)
    for seed in range(2):
        for _, _, _, _, info in _run(cfg, seed, actions=[N_ACTIONS - 1], steps=100):
            assert info["alloc_applied_mbps"] <= info["alloc_after_reliability_filter_mbps"]
            assert info["alloc_clipped_by_filter"] == (
                info["alloc_applied_mbps"] != info["alloc_requested_mbps"]
            )
            assert info["both_filters_fired"] == (
                info["alloc_clipped_by_latency_filter"] and info["reliability_filter_clipped"]
            )


def test_reliability_floor_wins_over_the_ceiling_in_the_environment() -> None:
    """With a strongly scaled hazard the safety floor is high; the ceiling cannot take the action below it."""
    cfg = stress_config(100.0, dataclasses.replace(v0_config(workload_profile="peak"), latency_filter=True))
    seen_override = False
    for seed in range(6):
        for _, _, _, _, info in _run(cfg, seed, actions=[0], steps=150):
            if info["latency_filter_overridden_by_reliability"]:
                seen_override = True
                assert info["alloc_applied_mbps"] > info["latency_ceiling_mbps"]
                assert info["alloc_applied_mbps"] == info["alloc_after_reliability_filter_mbps"]
                assert info["alloc_clipped_by_latency_filter"] is False
    assert seen_override, "no step with the reliability floor above the latency ceiling was constructed"


@pytest.mark.skipif(not _V1_PRESENT, reason="hazard/artifacts/v1 absent")
def test_cli_flag_reaches_the_simulator_configuration() -> None:
    import argparse

    from agent import evaluate_policy
    from agent.train import _parser as train_parser
    from baselines import evaluate as baselines_evaluate

    assert train_parser().parse_args([]).latency_filter is False
    assert train_parser().parse_args(["--latency-filter"]).latency_filter is True

    off = baselines_evaluate._build_config(argparse.Namespace(eval_fleet=False, hazard_error_sigma=0.0))
    assert off.latency_filter is False and off.label == "default"
    on = baselines_evaluate._build_config(
        argparse.Namespace(eval_fleet=False, hazard_error_sigma=0.25, latency_filter=True)
    )
    assert on.latency_filter is True and on.label == "train-fleet,sigma=0.25,latency-filter"

    assert evaluate_policy.build_config(eval_fleet=False, hazard_error_sigma=0.0).latency_filter is False
    cfg = evaluate_policy.build_config(eval_fleet=False, hazard_error_sigma=0.25, latency_filter=True)
    assert cfg.latency_filter is True and cfg.label.endswith(",latency-filter")


def _stepper(cfg: SimConfig, seed: int):
    env = RecoveryEnv(cfg)
    env.reset(seed=seed)
    return env


def test_causal_and_oracle_flags_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError):
        v0_config(latency_filter=True, latency_filter_causal=True)
    assert v0_config().latency_filter_causal is False


def test_causal_variant_uses_the_previous_intervals_load() -> None:
    """The load the ceiling reads at interval t is the load the observation carried at t (interval t-1's)."""
    cfg = v0_config(workload_profile="peak", latency_filter_causal=True)
    env = RecoveryEnv(cfg)
    obs, reset_info = env.reset(seed=3)
    previous_fg = reset_info["fg_bw_mbps"]
    for _ in range(40):
        obs, _, term, trunc, info = env.step(N_ACTIONS - 1)
        assert info["latency_filter_fg_source"] == "causal"
        assert info["latency_filter_fg_mbps"] == pytest.approx(previous_fg)
        previous_fg = info["fg_bw_mbps"]
        if term or trunc:
            break


def test_causal_decision_does_not_depend_on_the_current_intervals_load() -> None:
    """Altering only the row of the interval being decided leaves the causal ceiling unchanged and moves the oracle one."""
    for field_name, expect_equal in (("latency_filter_causal", True), ("latency_filter", False)):
        cfg = v0_config(workload_profile="peak", **{field_name: True})
        a, b = _stepper(cfg, 5), _stepper(cfg, 5)
        row = (b._offset + 1) % len(b._trace)
        altered = np.array(b._trace.fg_bw_mbps, dtype=np.float64).copy()
        altered[row] = 0.0 if altered[row] > 100.0 else 190.0
        b._trace = dataclasses.replace(b._trace, fg_bw_mbps=altered)
        _, _, _, _, ia = a.step(N_ACTIONS - 1)
        _, _, _, _, ib = b.step(N_ACTIONS - 1)
        assert ia["fg_bw_mbps"] != ib["fg_bw_mbps"]
        same = ia["latency_ceiling_mbps"] == ib["latency_ceiling_mbps"] and ia["alloc_applied_mbps"] == ib["alloc_applied_mbps"]
        assert same == expect_equal, field_name


@pytest.mark.skipif(not _V1_PRESENT, reason="hazard/artifacts/v1 absent")
def test_causal_flag_reaches_the_cli_configurations() -> None:
    import argparse

    from agent.train import _parser as train_parser
    from baselines import evaluate as baselines_evaluate

    assert train_parser().parse_args(["--latency-filter-causal"]).latency_filter_causal is True
    assert train_parser().parse_args([]).latency_filter_causal is False
    cfg = baselines_evaluate._build_config(
        argparse.Namespace(eval_fleet=False, hazard_error_sigma=0.25, latency_filter=False, latency_filter_causal=True)
    )
    assert cfg.latency_filter_causal is True and cfg.latency_filter is False
    assert cfg.label == "train-fleet,sigma=0.25,latency-filter-causal"
