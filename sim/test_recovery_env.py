"""Tests for the recovery simulator.

Nothing here validates the simulator against a real system: the latency model
is an idealisation and every constant is a design decision. The tests check
that the code does what it is documented to do, that it honours the contract,
and that the scenario generator exercises the range of the risk input.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from contracts.contracts import (
    ACTIONS_MBPS,
    BW_TOTAL_MBPS,
    CONTROL_INTERVAL_S,
    GROUP_TOLERATED_FAILURES,
    LATENCY_LIMIT_FACTOR,
    MAX_EPISODE_STEPS,
    N_ACTIONS,
    OBS_DIM,
    OBS_FIELDS,
    OBS_SCALES,
    REBUILD_WORK_GB,
    RISK_BUDGET_EPS,
    ContractViolation,
    Observation,
    TerminationReason,
    compute_reward,
    latency_degradation,
    latency_violation,
    termination_flags,
    validate_info,
)
from contracts.test_adapter_conformance import ALL_CHECKS
from sim.config import SimConfig, latency_mean_ms, latency_percentiles_ms, v0_config, violation_rho_threshold
from sim.recovery_env import RecoveryEnv, make_env
from sim.safety_filter import apply_safety_filter, exposure_probability
from sim.stress import STRESS_HAZARD_MULTIPLIER, ScaledHazard, stress_config
from hazard.v1 import ARTIFACT_DIR as V1_ARTIFACT_DIR
from workload.generate import N_ROWS, PROFILE_NAMES

_V1_PRESENT = (V1_ARTIFACT_DIR / "meta.json").exists()
REPO_ROOT = Path(__file__).resolve().parent.parent
FIELD_5 = OBS_FIELDS.index("hazard")

# Per-drive baseline hazard of the contract's risk arithmetic, per hour.
BASE_DRIVE_HAZARD = 0.0211 / 8760.0
BASE_GROUP_HAZARD = 9 * BASE_DRIVE_HAZARD

#: A fleet exactly at the model's baseline: no degradation, no per-drive spread.
HEALTHY = dict(fleet_degradation=0.0, drive_spread_sd=0.0)


class StressHazard:
    """Hazard stand-in with a near-certain 30-day failure, to reach data loss.

    With the v0 model the probability of two further failures inside a rebuild
    is of order 1e-5 per episode, so the loss path needs an artificially
    hostile risk model to be exercised at all.
    """

    def __call__(self, features: dict[str, float], horizon_hours: float = 720.0) -> tuple[float, float]:
        return 0.999999, 0.0


def make_v0_env(config: SimConfig | None = None) -> RecoveryEnv:
    """Environment factory pinned to the v0 hazard and its feature function."""
    return make_env(v0_config() if config is None else config)


def write_constant_csv(path: Path, total_mbps: float, rows: int = N_ROWS) -> Path:
    """Write a replay CSV of constant foreground load with int64 columns."""
    kbps = int(round(total_mbps * 1000.0))
    read = kbps // 2
    frame = pd.DataFrame(
        {
            "t_seconds": np.arange(rows, dtype=np.int64) * CONTROL_INTERVAL_S,
            "read_iops": np.full(rows, 1000, dtype=np.int64),
            "write_iops": np.full(rows, 500, dtype=np.int64),
            "read_bw_kbps": np.full(rows, read, dtype=np.int64),
            "write_bw_kbps": np.full(rows, kbps - read, dtype=np.int64),
            "disk_usage_mb": np.full(rows, 1_000_000, dtype=np.int64),
        }
    )
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def assert_valid_obs(obs: Any) -> None:
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32 and obs.shape == (OBS_DIM,)
    assert np.all(np.isfinite(obs)) and np.all(obs >= 0.0) and np.all(obs <= 1.0)


def run_episode(env: RecoveryEnv, policy: Any, seed: int | None = None) -> tuple[int, dict[str, Any], list[dict[str, Any]]]:
    """Run one episode, checking every observation and info; return (steps, final_info, infos)."""
    obs, info = env.reset(seed=seed)
    assert_valid_obs(obs)
    validate_info(info)
    infos: list[dict[str, Any]] = []
    for t in range(MAX_EPISODE_STEPS + 1):
        obs, reward, terminated, truncated, info = env.step(policy(t))
        assert_valid_obs(obs)
        validate_info(info)
        assert type(terminated) is bool and type(truncated) is bool and math.isfinite(reward)
        infos.append(info)
        if terminated or truncated:
            return t + 1, info, infos
    raise AssertionError("episode did not end within MAX_EPISODE_STEPS + 1 steps")


# ---------------------------------------------------------------------------
# Latency model
# ---------------------------------------------------------------------------


class TestLatencyModel:
    def test_formula_and_unloaded_limit(self) -> None:
        """latency = unloaded / (1 - rho): equals the unloaded latency at rho = 0."""
        cfg = v0_config()
        assert latency_mean_ms(0.0, cfg) == pytest.approx(2.0)
        assert latency_mean_ms(0.5, cfg) == pytest.approx(4.0)
        assert latency_mean_ms(0.9, cfg) == pytest.approx(20.0)

    def test_convex_and_increasing(self) -> None:
        """Convexity is what makes the trade-off non-trivial: strictly positive second differences."""
        cfg = v0_config()
        rhos = np.linspace(0.0, cfg.rho_max, 50)
        values = np.array([latency_mean_ms(r, cfg) for r in rhos])
        assert np.all(np.diff(values) > 0.0)
        assert np.all(np.diff(values, n=2) > 0.0)

    def test_utilisation_is_capped_at_rho_max(self) -> None:
        """A foreground load above the ceiling cannot send latency to infinity."""
        cfg = v0_config()
        cap = 2.0 / (1.0 - cfg.rho_max)
        for rho in (0.98, 0.99, 1.0, 1.5):
            assert latency_mean_ms(rho, cfg) == pytest.approx(cap)

    def test_violation_threshold_is_0_9111(self) -> None:
        """L_max = 1.5 * 15 = 22.5 ms, so a violation begins at rho = 1 - 2/22.5 ~ 0.9111.

        The threshold is found by bisection on the actual latency model and the
        contract's own ``latency_violation``, then compared with the closed form.
        """
        cfg = v0_config()
        closed_form = 1.0 - 2.0 / (LATENCY_LIMIT_FACTOR * 15.0)
        assert closed_form == pytest.approx(0.91111, abs=1e-4)
        assert cfg.latency_target_ms == 15.0
        assert violation_rho_threshold(cfg) == pytest.approx(closed_form, abs=1e-12)

        lo, hi = 0.0, cfg.rho_max
        assert latency_violation(latency_mean_ms(lo, cfg), cfg.latency_target_ms) == 0.0
        assert latency_violation(latency_mean_ms(hi, cfg), cfg.latency_target_ms) == 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if latency_violation(latency_mean_ms(mid, cfg), cfg.latency_target_ms) == 1.0:
                hi = mid
            else:
                lo = mid
        assert hi == pytest.approx(closed_form, abs=1e-9)
        assert hi == pytest.approx(0.9111, abs=1e-3)

    def test_degradation_term_is_silent_until_rho_0_867(self) -> None:
        """D = 0 while latency <= target (rho <= 1 - 2/15 ~ 0.867); informative only up to the 0.911 cliff."""
        cfg = v0_config()
        silent_until = 1.0 - cfg.latency_unloaded_ms / cfg.latency_target_ms
        assert silent_until == pytest.approx(0.8667, abs=1e-4)
        for rho in (0.0, 0.5, 0.8, silent_until - 1e-6):
            assert latency_degradation(latency_mean_ms(rho, cfg), cfg.latency_target_ms) == 0.0
        band = latency_degradation(latency_mean_ms(0.90, cfg), cfg.latency_target_ms)
        assert 0.0 < band < LATENCY_LIMIT_FACTOR - 1.0  # positive, and below the violation cliff
        assert silent_until < violation_rho_threshold(cfg)

    def test_target_above_unloaded_so_light_load_is_not_degraded(self) -> None:
        """With target 15 ms > unloaded 2 ms an idle system has no violation and no degradation."""
        cfg = v0_config()
        idle = latency_mean_ms(0.0, cfg)
        assert latency_violation(idle, cfg.latency_target_ms) == 0.0
        assert latency_degradation(idle, cfg.latency_target_ms) == 0.0

    def test_target_not_above_unloaded_is_rejected(self) -> None:
        """If the target equalled the unloaded latency, any load would register as degradation."""
        with pytest.raises(ValueError):
            v0_config(latency_target_ms=2.0)
        with pytest.raises(ValueError):
            v0_config(latency_target_ms=1.0)

    def test_percentiles_are_exponential_quantiles(self) -> None:
        p95, p99 = latency_percentiles_ms(10.0)
        assert p95 == pytest.approx(10.0 * math.log(20.0))
        assert p99 == pytest.approx(10.0 * math.log(100.0))
        # P(T > p) = exp(-p / mean) for an exponential response time.
        assert math.exp(-p95 / 10.0) == pytest.approx(0.05)
        assert math.exp(-p99 / 10.0) == pytest.approx(0.01)

    def test_config_validation(self) -> None:
        for bad in (
            dict(latency_unloaded_ms=0.0),
            dict(rho_max=1.0),
            dict(rho_max=0.0),
            dict(workload_profile="nope"),
            dict(workload_profile="idle", workload_csv="x.csv"),
            dict(fleet_degradation=1.5),
            dict(drive_spread_sd=-0.1),
        ):
            with pytest.raises(ValueError):
                v0_config(**bad)


# ---------------------------------------------------------------------------
# Safety filter
# ---------------------------------------------------------------------------


class TestSafetyFilter:
    def test_exposure_formula(self) -> None:
        """P = 1 - exp(-hazard * T) with T = work * 1000 / alloc / 3600 hours."""
        hours = 12000.0 * 1000.0 / 80.0 / 3600.0
        assert exposure_probability(1e-4, 12000.0, 80.0) == pytest.approx(1.0 - math.exp(-1e-4 * hours))
        assert exposure_probability(0.0, 12000.0, 20.0) == 0.0
        assert exposure_probability(1e-4, 0.0, 20.0) == 0.0

    def test_exposure_falls_with_bandwidth(self) -> None:
        for hazard in (1e-6, 2e-5, 1e-4, 5e-4):
            values = [exposure_probability(hazard, REBUILD_WORK_GB, a) for a in ACTIONS_MBPS]
            assert all(x > y for x, y in zip(values, values[1:]))

    @pytest.mark.parametrize(
        "multiple, expected_feasible_count",
        [(1.0, 4), (1.3, 4), (2.0, 3), (3.0, 3), (4.0, 2), (5.0, 2), (6.0, 1), (8.0, 1), (10.0, 0)],
    )
    def test_feasible_set_matches_contract_table(self, multiple: float, expected_feasible_count: int) -> None:
        """The contract's risk table: the constraint binds at ~1.4x, 3.5x, 5.5x and 8.3x baseline."""
        hazard = BASE_GROUP_HAZARD * multiple
        feasible = [
            exposure_probability(hazard, REBUILD_WORK_GB, a) < RISK_BUDGET_EPS for a in ACTIONS_MBPS
        ]
        assert sum(feasible) == expected_feasible_count
        # Feasible set is a suffix of the sorted action set.
        assert feasible == sorted(feasible)

    def test_feasible_request_is_applied_unchanged(self) -> None:
        d = apply_safety_filter(0, BASE_GROUP_HAZARD, REBUILD_WORK_GB)
        assert (d.applied_index, d.clipped, d.violated_risk_budget) == (0, False, False)

    def test_infeasible_request_is_clipped_up_to_smallest_feasible(self) -> None:
        for multiple, expected_index in ((2.0, 1), (4.0, 2), (6.0, 3)):
            d = apply_safety_filter(0, BASE_GROUP_HAZARD * multiple, REBUILD_WORK_GB)
            assert d.applied_index == expected_index
            assert d.clipped and not d.violated_risk_budget and d.any_feasible
            assert d.p_failure < RISK_BUDGET_EPS

    def test_nothing_feasible_selects_maximum_bandwidth(self) -> None:
        hazard = BASE_GROUP_HAZARD * 10.0
        for requested in range(N_ACTIONS):
            d = apply_safety_filter(requested, hazard, REBUILD_WORK_GB)
            assert d.applied_index == N_ACTIONS - 1
            assert d.applied_mbps == max(ACTIONS_MBPS)
            assert d.violated_risk_budget and not d.any_feasible
            assert d.clipped == (requested != N_ACTIONS - 1)

    def test_never_clips_down_and_applies_smallest_feasible_above(self) -> None:
        """Property check over a grid of hazards, remaining work and requests."""
        for hazard in np.geomspace(1e-7, 1e-3, 40):
            for work in (REBUILD_WORK_GB, 6000.0, 600.0, 10.0):
                probs = [exposure_probability(float(hazard), work, a) for a in ACTIONS_MBPS]
                for requested in range(N_ACTIONS):
                    d = apply_safety_filter(requested, float(hazard), work)
                    assert d.applied_index >= requested
                    if probs[requested] < RISK_BUDGET_EPS:
                        assert d.applied_index == requested and not d.clipped
                    else:
                        higher = [i for i in range(requested, N_ACTIONS) if probs[i] < RISK_BUDGET_EPS]
                        assert d.applied_index == (higher[0] if higher else N_ACTIONS - 1)
                    assert d.clipped == (d.applied_index != requested)
                    assert d.violated_risk_budget == (not probs[d.applied_index] < RISK_BUDGET_EPS)

    def test_env_filter_flags_are_consistent_with_info(self) -> None:
        """Across many scenarios: applied >= requested, clip flag matches, budget flag matches P."""
        env = RecoveryEnv(v0_config())
        seen_clip = seen_ok = False
        for seed in range(80):
            env.reset(seed=seed)
            _, _, _, _, info = env.step(0)
            assert info["alloc_applied_mbps"] >= info["alloc_requested_mbps"]
            assert info["alloc_clipped_by_filter"] == (info["alloc_applied_mbps"] != info["alloc_requested_mbps"])
            assert info["violated_risk_budget"] == (info["p_failure_before_complete"] >= RISK_BUDGET_EPS)
            assert info["p_failure_at_delivered_rate"] >= info["p_failure_before_complete"] - 1e-15
            seen_clip |= info["alloc_clipped_by_filter"]
            seen_ok |= not info["alloc_clipped_by_filter"]
        assert seen_clip and seen_ok

    def test_env_applies_maximum_when_nothing_feasible(self, tmp_path: Path) -> None:
        csv = write_constant_csv(tmp_path / "load.csv", 20.0)
        env = RecoveryEnv(v0_config(workload_csv=csv, hazard_model=StressHazard()))
        env.reset(seed=0)
        for action in range(N_ACTIONS):
            env.reset(seed=0)
            _, _, _, _, info = env.step(action)
            assert info["alloc_applied_mbps"] == max(ACTIONS_MBPS)
            assert info["violated_risk_budget"] is True


# ---------------------------------------------------------------------------
# Conformance harness
# ---------------------------------------------------------------------------


class TestConformance:
    def test_all_checks_pass_in_process(self) -> None:
        for check in ALL_CHECKS:
            check(make_v0_env)

    @pytest.mark.skipif(not _V1_PRESENT, reason="hazard/artifacts/v1 absent")
    def test_all_checks_pass_with_default_config(self) -> None:
        for check in ALL_CHECKS:
            check(make_env)

    @pytest.mark.parametrize("profile", PROFILE_NAMES)
    def test_all_checks_pass_for_each_pinned_profile(self, profile: str) -> None:
        def factory() -> RecoveryEnv:
            return make_env(v0_config(workload_profile=profile))

        for check in ALL_CHECKS:
            check(factory)

    def test_external_environment_tests_run_and_pass_not_skip(self) -> None:
        """With ISM_ENV_FACTORY set, the harness's 7 external-environment tests run and pass."""
        env = dict(os.environ)
        env["ISM_ENV_FACTORY"] = "sim.test_recovery_env:make_v0_env"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "contracts/test_adapter_conformance.py",
                "-k",
                "TestExternalEnvironment",
                "-q",
                "-p",
                "no:cacheprovider",
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )
        output = result.stdout + result.stderr
        assert result.returncode == 0, output
        assert "7 passed" in output, output
        assert "skipped" not in output, output


# ---------------------------------------------------------------------------
# Observation, info, termination
# ---------------------------------------------------------------------------


class TestObservationAndInfo:
    def test_reset_observation_fields(self) -> None:
        env = RecoveryEnv(v0_config())
        obs, info = env.reset(seed=3)
        assert_valid_obs(obs)
        v = dict(zip(OBS_FIELDS, obs.tolist()))
        assert v["progress"] == 0.0
        assert v["work_remaining"] == 1.0
        assert v["redundancy_left"] == pytest.approx(1.0 / GROUP_TOLERATED_FAILURES)
        assert v["alloc_current"] == 0.0
        assert v["latency_target"] == pytest.approx(15.0 / OBS_SCALES["latency_target"])
        # Headroom (field 11) equals rho_max * BW_TOTAL - fg_bw, as a fraction of its scale.
        expected = max(0.0, 0.98 * BW_TOTAL_MBPS - info["fg_bw_mbps"]) / OBS_SCALES["bw_available"]
        assert v["bw_available"] == pytest.approx(expected, abs=1e-6)

    def test_valid_observation_and_info_over_full_episodes(self) -> None:
        """Every reset and step of several full episodes passes the contract checks."""
        env = RecoveryEnv(v0_config())
        ends = []
        for seed, policy in (
            (0, lambda t: 3),
            (1, lambda t: 0),
            (2, lambda t: t % N_ACTIONS),
            (3, lambda t: (t // 50) % N_ACTIONS),
            (4, lambda t: 1),
        ):
            steps, final, infos = run_episode(env, policy, seed=seed)
            assert 1 <= steps <= MAX_EPISODE_STEPS
            assert all("termination_reason" not in i for i in infos[:-1])
            ends.append(final["termination_reason"])
        assert set(ends) <= {r.value for r in TerminationReason}

    def test_diagnostic_keys_and_types(self) -> None:
        env = RecoveryEnv(v0_config())
        _, info = env.reset(seed=1)
        _, _, _, _, step_info = env.step(1)
        for i in (info, step_info):
            for key in (
                "rho",
                "alloc_requested_mbps",
                "alloc_effective_mbps",
                "bw_available_mbps",
                "fg_bw_mbps",
                "n_survivors",
                "trace_wraps",
                "workload_profile",
                "workload_synthetic",
            ):
                assert key in i, key
            assert 0.0 <= i["rho"] <= 0.98
            assert i["n_survivors"] == 9
            assert i["workload_profile"] in PROFILE_NAMES
            assert i["workload_synthetic"] is True  # parametric generator
        assert step_info["alloc_requested_mbps"] == 50.0
        assert step_info["alloc_effective_mbps"] <= step_info["bw_available_mbps"] + 1e-12
        assert step_info["alloc_effective_mbps"] <= step_info["alloc_applied_mbps"] + 1e-12

    def test_percentiles_and_reward_follow_the_documented_formulas(self) -> None:
        env = RecoveryEnv(v0_config())
        env.reset(seed=5)
        for _ in range(20):
            _, reward, _, _, info = env.step(2)
            lat = info["latency_now"]
            assert info["latency_p95"] == pytest.approx(lat * math.log(20.0))
            assert info["latency_p99"] == pytest.approx(lat * math.log(100.0))
            expected = compute_reward(
                latency_degradation(lat, 15.0), 1.0, latency_violation(lat, 15.0)
            )
            assert reward == pytest.approx(expected)
            assert info["violated_latency"] == (lat > LATENCY_LIMIT_FACTOR * 15.0)

    def test_p_failure_matches_contract_formula_on_observed_hazard(self) -> None:
        """P is computed from the group hazard shown in field 5 (here unsaturated)."""
        env = RecoveryEnv(v0_config(fleet_degradation=0.2, drive_spread_sd=0.0))
        obs, _ = env.reset(seed=11)
        assert obs[FIELD_5] < 1.0
        hazard = float(obs[FIELD_5]) * OBS_SCALES["hazard"]
        _, _, _, _, info = env.step(3)
        hours = REBUILD_WORK_GB * 1000.0 / info["alloc_applied_mbps"] / 3600.0
        assert info["p_failure_before_complete"] == pytest.approx(1.0 - math.exp(-hazard * hours), rel=1e-4)

    def test_bad_actions_raise_contract_violation(self) -> None:
        env = RecoveryEnv(v0_config())
        env.reset(seed=0)
        for bad in (-1, N_ACTIONS, 99, 1.5, True, "a", None):
            with pytest.raises(ContractViolation):
                env.step(bad)  # type: ignore[arg-type]

    def test_step_without_running_episode_raises(self) -> None:
        env = RecoveryEnv(v0_config())
        with pytest.raises(RuntimeError):
            env.step(0)
        env.reset(seed=0)
        while True:
            _, _, terminated, truncated, _ = env.step(3)
            if terminated or truncated:
                break
        with pytest.raises(RuntimeError):
            env.step(3)
        env.reset(seed=1)
        env.step(3)

    @pytest.mark.parametrize("bad_seed", [-1, 1.5, True, "3"])
    def test_bad_seed_is_rejected(self, bad_seed: Any) -> None:
        with pytest.raises(ValueError):
            RecoveryEnv(v0_config()).reset(seed=bad_seed)


class TestTermination:
    def _final(self, env: RecoveryEnv, action: int) -> tuple[bool, bool, dict[str, Any], int]:
        obs, _ = env.reset(seed=0)
        for t in range(1, MAX_EPISODE_STEPS + 1):
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                return terminated, truncated, info, t
            assert (terminated, truncated) == (False, False)
            assert info.get("termination_reason") is None
        raise AssertionError("no end")

    def test_all_three_reasons_are_reachable_and_flags_come_from_termination_flags(
        self, tmp_path: Path
    ) -> None:
        seen: dict[str, tuple[bool, bool]] = {}

        # RECOVERY_COMPLETE: default scenario, fastest rebuild.
        t_, tr_, info, _ = self._final(RecoveryEnv(v0_config(**HEALTHY)), 3)
        seen[info["termination_reason"]] = (t_, tr_)
        assert info["recovery_complete"] is True

        # TERMINAL_FAILURE: hostile hazard model, so that two further failures occur.
        loss_cfg = v0_config(hazard_model=StressHazard())
        for seed in range(30):
            env = RecoveryEnv(loss_cfg)
            env.reset(seed=seed)
            for _ in range(MAX_EPISODE_STEPS):
                _, _, terminated, truncated, info = env.step(3)
                if terminated or truncated:
                    break
            if info["termination_reason"] == TerminationReason.TERMINAL_FAILURE.value:
                seen[info["termination_reason"]] = (terminated, truncated)
                assert info["recovery_complete"] is False
                assert info["n_survivors"] <= 7
                break
        else:
            pytest.fail("TERMINAL_FAILURE not reached under the hostile hazard model")

        # HORIZON_EXPIRED: foreground load leaves no headroom, so no progress is possible.
        stalled = write_constant_csv(tmp_path / "stalled.csv", 199.0)
        t_, tr_, info, steps = self._final(RecoveryEnv(v0_config(workload_csv=stalled, **HEALTHY)), 3)
        seen[info["termination_reason"]] = (t_, tr_)
        assert steps == MAX_EPISODE_STEPS and info["recovery_complete"] is False

        assert set(seen) == {r.value for r in TerminationReason}
        for value, flags in seen.items():
            assert flags == termination_flags(TerminationReason(value))
        assert seen[TerminationReason.HORIZON_EXPIRED.value] == (False, True)

    def test_episode_never_exceeds_max_steps(self, tmp_path: Path) -> None:
        stalled = write_constant_csv(tmp_path / "stalled.csv", 199.0)
        env = RecoveryEnv(v0_config(workload_csv=stalled))
        steps, final, _ = run_episode(env, lambda t: 0, seed=0)
        assert steps <= MAX_EPISODE_STEPS
        for seed in range(6):
            steps, _, _ = run_episode(RecoveryEnv(v0_config()), lambda t: t % 2, seed=seed)
            assert steps <= MAX_EPISODE_STEPS

    def test_loss_requires_more_failures_than_remaining_redundancy(self) -> None:
        """Redundancy goes 1 -> 0 on the first failure; hazard drops with the lost drive."""
        env = RecoveryEnv(v0_config(hazard_model=StressHazard()))
        for seed in range(30):
            env.reset(seed=seed)
            hazard_before = env._hazard  # noqa: SLF001 - checking the aggregate drops
            for _ in range(MAX_EPISODE_STEPS):
                obs, _, terminated, truncated, info = env.step(3)
                if info["n_survivors"] < 9:
                    assert env._hazard < hazard_before or terminated  # noqa: SLF001
                    if not terminated:
                        assert obs[OBS_FIELDS.index("redundancy_left")] == 0.0
                        assert info["n_survivors"] == 8
                    return
                if terminated or truncated:
                    break
        pytest.fail("no survivor failure observed under the hostile hazard model")


# ---------------------------------------------------------------------------
# Rebuild behaviour
# ---------------------------------------------------------------------------


class TestRebuild:
    @pytest.mark.parametrize("profile", PROFILE_NAMES)
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_constant_120_and_constant_20_both_complete_within_horizon(self, profile: str, seed: int) -> None:
        """The 2400-step horizon clears the slowest action; neither fixed policy is truncated."""
        env = RecoveryEnv(v0_config(workload_profile=profile, **HEALTHY))
        fast_steps, fast, _ = run_episode(env, lambda t: 3, seed=seed)
        slow_steps, slow, _ = run_episode(env, lambda t: 0, seed=seed)
        assert fast["termination_reason"] == TerminationReason.RECOVERY_COMPLETE.value
        assert slow["termination_reason"] == TerminationReason.RECOVERY_COMPLETE.value
        assert fast_steps >= math.ceil(REBUILD_WORK_GB * 1000.0 / 120.0 / CONTROL_INTERVAL_S)
        assert slow_steps == math.ceil(REBUILD_WORK_GB * 1000.0 / 20.0 / CONTROL_INTERVAL_S) == 2000
        assert slow_steps <= MAX_EPISODE_STEPS
        assert fast_steps < slow_steps

    def test_healthy_group_is_never_clipped_at_baseline_hazard(self) -> None:
        """At baseline hazard all four actions are feasible, so the filter stays out of the way."""
        env = RecoveryEnv(v0_config(workload_profile="diurnal", **HEALTHY))
        _, _, infos = run_episode(env, lambda t: 0, seed=0)
        assert not any(i["alloc_clipped_by_filter"] for i in infos)
        assert not any(i["violated_risk_budget"] for i in infos)

    def test_more_bandwidth_means_faster_completion_and_higher_latency(self, tmp_path: Path) -> None:
        """Under constant foreground load, completion time falls and latency rises with the action."""
        csv = write_constant_csv(tmp_path / "load.csv", 60.0)
        steps, mean_latency = [], []
        for action in range(N_ACTIONS):
            env = RecoveryEnv(v0_config(workload_csv=csv, **HEALTHY))
            n, final, infos = run_episode(env, lambda t, a=action: a, seed=0)
            assert final["termination_reason"] == TerminationReason.RECOVERY_COMPLETE.value
            assert not any(i["alloc_clipped_by_filter"] for i in infos)
            steps.append(n)
            mean_latency.append(float(np.mean([i["latency_now"] for i in infos])))
        assert steps == [2000, 800, 500, 334]
        assert all(a > b for a, b in zip(steps, steps[1:]))
        assert all(a < b for a, b in zip(mean_latency, mean_latency[1:]))
        for action, lat in enumerate(mean_latency):
            rho = (60.0 + ACTIONS_MBPS[action]) / BW_TOTAL_MBPS
            assert lat == pytest.approx(2.0 / (1.0 - rho))

    def test_monotone_on_a_synthetic_profile(self) -> None:
        """Same scenario, different fixed action: completion never slower, first-step latency never lower."""
        steps, first_latency = [], []
        for action in range(N_ACTIONS):
            env = RecoveryEnv(v0_config(workload_profile="diurnal", **HEALTHY))
            n, _, infos = run_episode(env, lambda t, a=action: a, seed=7)
            steps.append(n)
            first_latency.append(infos[0]["latency_now"])
        assert all(a >= b for a, b in zip(steps, steps[1:])) and steps[0] > steps[-1]
        assert all(a <= b for a, b in zip(first_latency, first_latency[1:]))

    def test_progress_and_work_are_consistent(self) -> None:
        env = RecoveryEnv(v0_config(**HEALTHY))
        env.reset(seed=4)
        previous = 0.0
        for _ in range(50):
            obs, *_ = env.step(2)
            progress = obs[OBS_FIELDS.index("progress")]
            work = obs[OBS_FIELDS.index("work_remaining")]
            assert progress >= previous
            assert progress + work == pytest.approx(1.0, abs=1e-6)
            previous = float(progress)

    def test_violation_threshold_in_the_env(self, tmp_path: Path) -> None:
        """With 20 MB/s of rebuild, the latency limit is crossed between fg 162 and 162.5 MB/s.

        rho = (fg + 20) / 200 crosses ~0.9111 at fg ~ 162.22 MB/s.
        """
        outcomes = {}
        for fg in (100.0, 162.0, 162.5, 170.0):
            csv = write_constant_csv(tmp_path / f"fg_{fg}.csv", fg)
            env = RecoveryEnv(v0_config(workload_csv=csv, **HEALTHY))
            env.reset(seed=0)
            _, _, _, _, info = env.step(0)
            outcomes[fg] = info["violated_latency"]
            assert info["rho"] == pytest.approx((fg + 20.0) / BW_TOTAL_MBPS)
        assert outcomes == {100.0: False, 162.0: False, 162.5: True, 170.0: True}

    def test_alloc_is_limited_by_headroom(self, tmp_path: Path) -> None:
        """alloc_effective = min(applied, headroom): at fg 150 only 46 MB/s are available."""
        csv = write_constant_csv(tmp_path / "fg150.csv", 150.0)
        env = RecoveryEnv(v0_config(workload_csv=csv, **HEALTHY))
        env.reset(seed=0)
        _, _, _, _, info = env.step(3)
        assert info["bw_available_mbps"] == pytest.approx(0.98 * 200.0 - 150.0)
        assert info["alloc_effective_mbps"] == pytest.approx(46.0)
        assert info["rho"] == pytest.approx(0.98)
        assert info["p_failure_at_delivered_rate"] > info["p_failure_before_complete"]


# ---------------------------------------------------------------------------
# Workload handling
# ---------------------------------------------------------------------------


class TestWorkload:
    def test_kbps_converted_once_to_mbps(self, tmp_path: Path) -> None:
        csv = write_constant_csv(tmp_path / "load.csv", 128.0)
        obs, info = RecoveryEnv(v0_config(workload_csv=csv)).reset(seed=0)
        assert info["fg_bw_mbps"] == pytest.approx(128.0)
        assert obs[OBS_FIELDS.index("fg_bw")] == pytest.approx(128.0 / BW_TOTAL_MBPS)

    def test_trace_is_tiled_cyclically_and_wraps_are_recorded(self, tmp_path: Path) -> None:
        stalled = write_constant_csv(tmp_path / "stalled.csv", 199.0)
        env = RecoveryEnv(v0_config(workload_csv=stalled, random_trace_offset=False))
        _, info = env.reset(seed=0)
        assert info["trace_wraps"] == 0
        for t in range(1, 700):
            _, _, _, _, info = env.step(0)
            assert info["trace_wraps"] == t // N_ROWS
        assert info["trace_wraps"] == 2

    def test_trace_wraps_with_random_offset(self, tmp_path: Path) -> None:
        stalled = write_constant_csv(tmp_path / "stalled.csv", 199.0)
        env = RecoveryEnv(v0_config(workload_csv=stalled))
        env.reset(seed=3)
        offset = env._offset  # noqa: SLF001
        _, _, _, _, info = env.step(0)
        assert info["trace_wraps"] == (offset + 1) // N_ROWS
        steps, final, infos = run_episode(env, lambda t: 0, seed=3)
        assert final["trace_wraps"] == (offset + steps) // N_ROWS >= 8

    def test_pinned_profile(self) -> None:
        for name in PROFILE_NAMES:
            _, info = RecoveryEnv(v0_config(workload_profile=name)).reset(seed=2)
            assert info["workload_profile"] == name

    def test_profile_sampling_covers_all_four_profiles(self) -> None:
        env = RecoveryEnv(v0_config())
        seen = {env.reset(seed=s)[1]["workload_profile"] for s in range(60)}
        assert seen == set(PROFILE_NAMES)

    @pytest.mark.parametrize(
        "sidecar, expected",
        [
            ({"synthetic": True, "profile": "peak"}, True),
            ({"synthetic": False, "profile": "peak"}, False),
            ({"profile": "peak"}, None),
            ({"synthetic": "yes"}, None),
            (None, None),
            ("not json", None),
        ],
    )
    def test_synthetic_flag_comes_from_sidecar_or_is_none(
        self, tmp_path: Path, sidecar: Any, expected: bool | None
    ) -> None:
        csv = write_constant_csv(tmp_path / "profile_x.csv", 50.0)
        if sidecar is not None:
            text = sidecar if isinstance(sidecar, str) else json.dumps(sidecar)
            (tmp_path / "profile_x.meta.json").write_text(text, encoding="utf-8")
        env = RecoveryEnv(v0_config(workload_csv=csv))
        _, info = env.reset(seed=0)
        assert info["workload_synthetic"] is expected
        _, _, _, _, step_info = env.step(0)
        assert step_info["workload_synthetic"] is expected
        validate_info(info)

    def test_csv_with_wrong_schema_is_rejected(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.csv"
        pd.DataFrame({"t_seconds": [0, 300], "read_iops": [1, 2]}).to_csv(bad, index=False)
        with pytest.raises(ContractViolation):
            RecoveryEnv(v0_config(workload_csv=bad)).reset(seed=0)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    @staticmethod
    def _trajectory(seed: int, config: SimConfig | None = None) -> list[tuple[Any, ...]]:
        env = RecoveryEnv(v0_config() if config is None else config)
        rng = np.random.default_rng(99)  # actions: independent of the env's own randomness
        obs, info = env.reset(seed=seed)
        out: list[tuple[Any, ...]] = [(obs.copy(), None, False, False, dict(info))]
        for _ in range(MAX_EPISODE_STEPS):
            obs, r, te, tr, info = env.step(int(rng.integers(0, N_ACTIONS)))
            out.append((obs.copy(), r, te, tr, dict(info)))
            if te or tr:
                break
        return out

    @staticmethod
    def _assert_equal(a: list[tuple[Any, ...]], b: list[tuple[Any, ...]]) -> None:
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert np.array_equal(x[0], y[0])
            assert x[1:4] == y[1:4]
            assert x[4] == y[4]

    @pytest.mark.parametrize("seed", [0, 5, 123])
    def test_same_seed_reproduces_identical_trajectory(self, seed: int) -> None:
        self._assert_equal(self._trajectory(seed), self._trajectory(seed))

    def test_same_seed_on_a_reused_env(self) -> None:
        env = RecoveryEnv(v0_config())
        runs = []
        for _ in range(2):
            obs, _ = env.reset(seed=8)
            traj = [obs.copy()]
            for t in range(100):
                traj.append(env.step(t % N_ACTIONS)[0].copy())
            runs.append(traj)
        assert all(np.array_equal(x, y) for x, y in zip(*runs))

    def test_same_seed_with_failures_is_identical(self) -> None:
        cfg = v0_config(hazard_model=StressHazard())
        self._assert_equal(self._trajectory(2, cfg), self._trajectory(2, cfg))

    def test_different_seeds_give_different_scenarios(self) -> None:
        a, b = self._trajectory(1), self._trajectory(2)
        assert not np.array_equal(a[0][0], b[0][0])


# ---------------------------------------------------------------------------
# Scenario coverage of the risk input
# ---------------------------------------------------------------------------


class TestRiskInputCoverage:
    N_RESETS = 600

    @pytest.fixture(scope="class")
    def field5_and_feasible(self) -> tuple[np.ndarray, np.ndarray]:
        env = RecoveryEnv(v0_config())
        f5, feasible = [], []
        for seed in range(self.N_RESETS):
            obs, _ = env.reset(seed=seed)
            f5.append(float(obs[FIELD_5]))
            feasible.append(
                sum(
                    exposure_probability(env._hazard, REBUILD_WORK_GB, a) < RISK_BUDGET_EPS  # noqa: SLF001
                    for a in ACTIONS_MBPS
                )
            )
        return np.array(f5), np.array(feasible)

    def test_normalised_field_5_spans_the_unit_interval(
        self, field5_and_feasible: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Field 5 must not cluster near its baseline of ~0.12.

        Required over 600 seeded resets: the minimum is within 0.03 of the
        healthy baseline (<= 0.15); the maximum saturates at 1.0 with at least
        4% of resets saturated; each quarter of [0, 1] holds at least 12% of
        resets; the standard deviation is at least 0.2. A policy only learns to
        use the hazard input if it sees every part of the range, and the
        upper range is reachable only through shared degradation across drives,
        because one degraded drive of nine moves the group value by ~2x
        baseline (~0.24).
        """
        f5, _ = field5_and_feasible
        assert f5.min() <= 0.15
        assert f5.max() == 1.0 and np.mean(f5 >= 1.0) >= 0.04
        edges = [0.0, 0.25, 0.5, 0.75, 1.0000001]
        for lo, hi in zip(edges, edges[1:]):
            share = float(np.mean((f5 >= lo) & (f5 < hi)))
            assert share >= 0.12, f"only {share:.3f} of resets in [{lo}, {hi})"
        assert f5.std() >= 0.2

    def test_every_filter_regime_is_represented(self, field5_and_feasible: tuple[np.ndarray, np.ndarray]) -> None:
        """Scenarios must reach each regime of the contract's risk table, not just the easy one.

        The regimes are the number of feasible actions at the start of the
        episode (4, 3, 2, 1, 0). Each must hold at least 5% of resets, so the
        filter clips in some scenarios and has no feasible action in others.
        """
        _, feasible = field5_and_feasible
        for count in range(N_ACTIONS + 1):
            assert np.mean(feasible == count) >= 0.05, f"regime with {count} feasible actions under-represented"

    def test_one_degraded_drive_alone_only_reaches_about_0_24(self) -> None:
        """Documents why correlation is needed: a lone worst-case drive is far from saturation."""
        from hazard.v0 import MULTIPLIER_CAP

        group = (8 * 1.0 + MULTIPLIER_CAP) / 9.0 * BASE_GROUP_HAZARD
        assert group / OBS_SCALES["hazard"] == pytest.approx(0.24, abs=0.01)

    def test_drives_in_an_episode_are_correlated(self) -> None:
        """Within-episode spread of drive severity is smaller than the spread across episodes."""
        from hazard.v0 import severity_multiplier

        env = RecoveryEnv(v0_config())
        within, means = [], []
        for seed in range(200):
            env.reset(seed=seed)
            m = [severity_multiplier(f)[0] for f in env._fleet.features]  # noqa: SLF001
            within.append(np.var(m))
            means.append(np.mean(m))
        assert np.mean(within) < np.var(means)

    def test_pinned_fleet_level_is_respected(self) -> None:
        env = RecoveryEnv(v0_config(fleet_degradation=1.0, drive_spread_sd=0.0))
        _, info = env.reset(seed=0)
        assert info["fleet_degradation_level"] == 1.0
        env = RecoveryEnv(v0_config(fleet_degradation=0.0, drive_spread_sd=0.0))
        obs, info = env.reset(seed=0)
        assert info["fleet_degradation_level"] == 0.0
        # Healthy fleet: only the age term raises the hazard above baseline (<= 1.12x).
        assert obs[FIELD_5] <= 0.12 * 1.13


# ---------------------------------------------------------------------------
# Latency budget by profile, and the stress configuration
# ---------------------------------------------------------------------------


class TestProfileViolationFractions:
    """Violation fraction per constant action on each profile, healthy fleet.

    Simulated and uncalibrated: these pin down the consequences of the
    documented constants, they are not findings about a real system.
    """

    BUDGET = 0.05  # LATENCY_VIOLATION_FRACTION_MAX

    #: Number of actions whose violation fraction is within the budget, per
    #: profile. A graded ladder: heavier workloads admit fewer actions.
    EXPECTED_FEASIBLE = {"idle": 4, "diurnal": 3, "bursty": 2, "peak": 1}

    @pytest.fixture(scope="class")
    def fractions(self) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {}
        for profile in PROFILE_NAMES:
            row = []
            for action in range(N_ACTIONS):
                per_seed = []
                for seed in range(3):
                    env = RecoveryEnv(v0_config(workload_profile=profile, **HEALTHY))
                    _, _, infos = run_episode(env, lambda t, a=action: a, seed=seed)
                    assert not any(i["alloc_clipped_by_filter"] for i in infos)
                    per_seed.append(float(np.mean([i["violated_latency"] for i in infos])))
                row.append(float(np.mean(per_seed)))
            out[profile] = row
        return out

    def test_every_profile_has_at_least_one_feasible_action(self, fractions: dict[str, list[float]]) -> None:
        """The latency constraint must be satisfiable everywhere, or a policy comparison there is meaningless."""
        for profile, row in fractions.items():
            assert any(f <= self.BUDGET for f in row), f"{profile}: no action within the latency budget: {row}"

    def test_feasible_action_count_per_profile(self, fractions: dict[str, list[float]]) -> None:
        counts = {p: sum(f <= self.BUDGET for f in row) for p, row in fractions.items()}
        assert counts == self.EXPECTED_FEASIBLE, fractions

    def test_feasible_set_is_a_prefix_of_the_action_set(self, fractions: dict[str, list[float]]) -> None:
        """Faster rebuilds only ever add violations: feasible actions are the smallest ones."""
        for profile, row in fractions.items():
            flags = [f <= self.BUDGET for f in row]
            assert flags == sorted(flags, reverse=True), f"{profile}: {row}"

    def test_idle_never_violates(self, fractions: dict[str, list[float]]) -> None:
        assert fractions["idle"] == [0.0] * N_ACTIONS


class TestStressConfiguration:
    def test_default_label(self) -> None:
        _, info = RecoveryEnv(v0_config()).reset(seed=0)
        assert info["config_label"] == "default"

    def test_stress_config_is_labelled_in_every_info(self) -> None:
        cfg = stress_config()
        assert cfg.label == f"stress_hazard_x{STRESS_HAZARD_MULTIPLIER:g}"
        env = RecoveryEnv(cfg)
        _, info = env.reset(seed=0)
        assert info["config_label"] == cfg.label
        _, _, _, _, step_info = env.step(0)
        assert step_info["config_label"] == cfg.label

    def test_stress_config_keeps_other_fields(self) -> None:
        cfg = stress_config(base=v0_config(workload_profile="idle", latency_target_ms=9.0))
        assert cfg.workload_profile == "idle" and cfg.latency_target_ms == 9.0
        assert isinstance(cfg.hazard_model, ScaledHazard)

    def test_scaled_hazard_properties(self) -> None:
        from hazard.v0 import HeuristicHazardV0

        base = HeuristicHazardV0()
        features = {"smart_187_raw": 3.0, "smart_197_raw": 2.0, "smart_9_raw": 20000.0}
        p0, u0 = base(features)
        assert ScaledHazard(base, 1.0)(features) == (p0, u0)
        previous = p0
        for k in (2.0, 10.0, 100.0, 1e3, 1e4):
            p, u = ScaledHazard(base, k)(features)
            assert previous < p < 1.0 and 0.0 <= u <= 0.5
            previous = p
        p_huge, u_huge = ScaledHazard(base, 1e12)(features)  # saturates below 1, never reaches it
        assert p_huge < 1.0 and 0.0 <= u_huge <= 0.5
        # Per-hour rate scales by the multiplier (small-p regime).
        from contracts.contracts import hazard_p_to_per_hour

        p100, _ = ScaledHazard(base, 100.0)(features)
        assert hazard_p_to_per_hour(p100) == pytest.approx(100.0 * hazard_p_to_per_hour(p0), rel=1e-3)
        with pytest.raises(ValueError):
            ScaledHazard(base, 0.5)

    def test_stress_run_makes_terminal_failures_measurable(self) -> None:
        """Losses are frequent enough to count in a couple of hundred episodes.

        At this hazard every action breaches the risk budget, so the filter
        applies the largest allocation whatever is requested.
        """
        env = RecoveryEnv(stress_config())
        losses = 0
        n = 200
        for seed in range(n):
            env.reset(seed=seed)
            _, _, _, _, info = env.step(0)
            assert info["alloc_applied_mbps"] == max(ACTIONS_MBPS)
            assert info["violated_risk_budget"] is True
            while not (info.get("termination_reason")):
                _, _, _, _, info = env.step(0)
            losses += info["termination_reason"] == TerminationReason.TERMINAL_FAILURE.value
        assert losses >= 4, f"only {losses} losses in {n} stress episodes"


def test_observation_class_is_used_for_every_vector() -> None:
    """The vector is built via Observation.to_vector: it round-trips through from_vector."""
    env = RecoveryEnv(v0_config())
    obs, _ = env.reset(seed=0)
    restored = Observation.from_vector(obs)
    restored.validate()
    assert np.array_equal(restored.to_vector(), obs)
