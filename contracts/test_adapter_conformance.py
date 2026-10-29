"""Conformance harness for environment implementations.

Any environment that is to enter the run path must pass the checks in this
module.  The environment is located through ``ISM_ENV_FACTORY``, a
``module:callable`` path whose callable returns a fresh environment instance.
When the variable is unset or does not resolve, the external-environment tests
skip; they never fail for the absence of an environment.

The checks are duck-typed against the Gymnasium calling convention
(``reset() -> (obs, info)``; ``step(action) -> (obs, reward, terminated,
truncated, info)``) and deliberately do not import gymnasium, so the harness
runs in an environment where only numpy and pytest are installed.

The harness is itself verified: a minimal in-file environment must pass every
check, and a set of deliberately broken variants must each be rejected, so a
check that silently accepts everything is detected.
"""

import importlib
import math
import os

import numpy as np
import pytest

from contracts.contracts import (
    BW_TOTAL_MBPS,
    CONTROL_INTERVAL_S,
    MAX_EPISODE_STEPS,
    N_ACTIONS,
    OBS_DIM,
    REBUILD_WORK_GB,
    ContractViolation,
    Observation,
    TerminationReason,
    action_to_mbps,
    termination_flags,
    validate_info,
)

ENV_FACTORY_VAR = "ISM_ENV_FACTORY"
_VALID_REASONS = {m.value for m in TerminationReason}


# ---------------------------------------------------------------------------
# Factory discovery
# ---------------------------------------------------------------------------


def load_env_factory(spec=None):
    """Resolve ``module:callable`` from ``spec`` or ``ISM_ENV_FACTORY``.

    Skips the calling test when the variable is unset, malformed, or the target
    cannot be imported or is not callable.
    """
    if spec is None:
        spec = os.environ.get(ENV_FACTORY_VAR, "").strip()
    if not spec:
        pytest.skip(f"{ENV_FACTORY_VAR} is not set; no environment to check")
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        pytest.skip(f"{ENV_FACTORY_VAR}={spec!r} is not of the form module:callable")
    try:
        module = importlib.import_module(module_name)
        factory = getattr(module, attr)
    except Exception as exc:  # unimportable module or missing attribute
        pytest.skip(f"cannot import {spec!r}: {type(exc).__name__}: {exc}")
    if not callable(factory):
        pytest.skip(f"{spec!r} is not callable")
    return factory


# ---------------------------------------------------------------------------
# Checks.  Each takes a zero-argument environment factory and raises
# AssertionError on a violation.  Contract exceptions are converted so that
# every failure mode of an implementation surfaces uniformly.
# ---------------------------------------------------------------------------


def _check_obs(obs, where):
    assert isinstance(obs, np.ndarray), f"{where}: observation is {type(obs).__name__}, not ndarray"
    assert obs.dtype == np.float32, f"{where}: observation dtype is {obs.dtype}, not float32"
    assert obs.shape == (OBS_DIM,), f"{where}: observation shape is {obs.shape}, not ({OBS_DIM},)"
    assert np.all(np.isfinite(obs)), f"{where}: observation contains NaN or inf"
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0), (
        f"{where}: observation outside [0, 1]: min={obs.min()}, max={obs.max()}"
    )


def _check_info(info, where):
    assert isinstance(info, dict), f"{where}: info is {type(info).__name__}, not dict"
    try:
        validate_info(info)
    except ContractViolation as exc:
        raise AssertionError(f"{where}: info violates the contract: {exc}") from exc


def _unpack_reset(result):
    assert isinstance(result, tuple) and len(result) == 2, (
        f"reset() must return (obs, info), got {type(result).__name__}"
        + (f" of length {len(result)}" if isinstance(result, tuple) else "")
    )
    return result


def _unpack_step(result):
    assert isinstance(result, tuple) and len(result) == 5, (
        "step() must return (obs, reward, terminated, truncated, info), got "
        + (f"a tuple of length {len(result)}" if isinstance(result, tuple) else type(result).__name__)
    )
    return result


def _check_step_outputs(result, where):
    obs, reward, terminated, truncated, info = _unpack_step(result)
    _check_obs(obs, where)
    assert isinstance(reward, (float, np.floating)) and not isinstance(reward, (bool, np.bool_)), (
        f"{where}: reward is {type(reward).__name__}, not a float"
    )
    assert math.isfinite(float(reward)), f"{where}: reward is not finite ({reward})"
    assert type(terminated) is bool, f"{where}: terminated is {type(terminated).__name__}, not bool"
    assert type(truncated) is bool, f"{where}: truncated is {type(truncated).__name__}, not bool"
    _check_info(info, where)
    _check_reason_and_flags(terminated, truncated, info, where)
    return terminated, truncated, info


def _check_reason_and_flags(terminated, truncated, info, where):
    """The reported reason and the (terminated, truncated) pair must agree.

    ``validate_info`` has already enforced that a present reason is the ``.value``
    string of a ``TerminationReason``.  The expected flags come from
    ``termination_flags``, so an environment cannot report, for example,
    HORIZON_EXPIRED as ``terminated`` (value-bootstrapping bias in the learner).
    """
    ended = terminated or truncated
    reason_value = info.get("termination_reason")
    if reason_value is None:
        assert not ended, f"{where}: episode ended without termination_reason"
        return
    assert ended, f"{where}: termination_reason {reason_value!r} reported on a step that did not end"
    expected = termination_flags(TerminationReason(reason_value))
    assert (terminated, truncated) == expected, (
        f"{where}: termination_reason {reason_value!r} requires "
        f"(terminated, truncated) == {expected}, got {(terminated, truncated)}"
    )


def check_reset(factory):
    obs, info = _unpack_reset(factory().reset())
    _check_obs(obs, "reset")
    _check_info(info, "reset")


def check_reset_is_repeatable(factory):
    env = factory()
    for i in range(3):
        obs, info = _unpack_reset(env.reset())
        _check_obs(obs, f"reset #{i}")
        _check_info(info, f"reset #{i}")


def check_step_outputs(factory):
    env = factory()
    env.reset()
    for action in range(N_ACTIONS):
        _check_step_outputs(env.step(action), f"step(action={action})")
        env.reset()


def check_out_of_range_action_raises(factory):
    for bad in (-1, N_ACTIONS, 99):
        env = factory()
        env.reset()
        try:
            env.step(bad)
        except Exception:
            continue
        raise AssertionError(f"step({bad}) was accepted; out-of-range actions must raise")


def run_episode(factory, policy=lambda t: t % N_ACTIONS):
    """Run one episode, checking every step; return (steps, final_info, terminated, truncated)."""
    env = factory()
    obs, info = _unpack_reset(env.reset())
    _check_obs(obs, "reset")
    _check_info(info, "reset")
    final_info = None
    terminated = truncated = False
    steps = 0
    for t in range(MAX_EPISODE_STEPS):
        steps += 1
        terminated, truncated, final_info = _check_step_outputs(env.step(policy(t)), f"step {steps}")
        if terminated or truncated:
            break
    return steps, final_info, terminated, truncated


def check_episode_ends_with_reason(factory):
    steps, info, terminated, truncated = run_episode(factory)
    assert terminated or truncated, (
        f"episode did not end within MAX_EPISODE_STEPS={MAX_EPISODE_STEPS} steps"
    )
    assert info.get("termination_reason") is not None, (
        f"final step ({steps}): episode ended without termination_reason"
    )


def check_episode_length_bounded(factory):
    steps, _, terminated, truncated = run_episode(factory, policy=lambda t: 0)
    assert steps <= MAX_EPISODE_STEPS
    assert terminated or truncated, (
        f"episode still running after MAX_EPISODE_STEPS={MAX_EPISODE_STEPS} steps"
    )


def check_reset_after_episode(factory):
    env = factory()
    env.reset()
    for t in range(MAX_EPISODE_STEPS):
        _, _, terminated, truncated, _ = _unpack_step(env.step(t % N_ACTIONS))
        if terminated or truncated:
            break
    obs, info = _unpack_reset(env.reset())
    _check_obs(obs, "reset after episode")
    _check_info(info, "reset after episode")
    _check_step_outputs(env.step(0), "first step after reset")


ALL_CHECKS = (
    check_reset,
    check_reset_is_repeatable,
    check_step_outputs,
    check_out_of_range_action_raises,
    check_episode_ends_with_reason,
    check_episode_length_bounded,
    check_reset_after_episode,
)


# ---------------------------------------------------------------------------
# Minimal reference environment and deliberately broken variants
# ---------------------------------------------------------------------------


class FakeEnv:
    """Smallest environment satisfying the contract.

    ``speed`` scales rebuild progress per step (0.0 never completes, so the
    episode ends by horizon expiry); ``fail_at`` forces a terminal failure at
    that step; ``horizon_steps`` is the truncation step.
    """

    def __init__(self, speed=1.0, fail_at=None, horizon_steps=MAX_EPISODE_STEPS):
        self.speed = speed
        self.fail_at = fail_at
        self.horizon_steps = horizon_steps
        self._t = 0
        self._progress = 0.0
        self._alloc = 0.0

    def _obs(self):
        obs = Observation(
            fg_read_iops=8000.0,
            fg_write_iops=3000.0,
            fg_bw=90.0,
            latency_now=22.0,
            latency_target=20.0,
            hazard=5e-5,
            hazard_uncertainty=2e-5,
            redundancy_left=1.0,
            progress=self._progress,
            work_remaining=REBUILD_WORK_GB * (1.0 - self._progress),
            alloc_current=self._alloc,
            bw_available=BW_TOTAL_MBPS,
        )
        obs.validate()
        return obs.to_vector(normalized=True)

    def _info(self, reason=None, complete=False):
        info = {
            "latency_p95": 40.0,
            "latency_p99": 55.0,
            "violated_latency": False,
            "p_failure_before_complete": 1e-4,
            "violated_risk_budget": False,
            "alloc_clipped_by_filter": False,
            "recovery_complete": complete,
        }
        if reason is not None:
            info["termination_reason"] = reason.value
        return info

    def reset(self, *, seed=None, options=None):
        self._t = 0
        self._progress = 0.0
        self._alloc = 0.0
        return self._obs(), self._info()

    def step(self, action):
        mbps = action_to_mbps(action)  # raises ContractViolation when invalid
        self._t += 1
        self._alloc = mbps
        gb = self.speed * mbps * CONTROL_INTERVAL_S / 1000.0
        self._progress = min(1.0, self._progress + gb / REBUILD_WORK_GB)
        reason = None
        if self._progress >= 1.0:
            reason = TerminationReason.RECOVERY_COMPLETE
        elif self.fail_at is not None and self._t >= self.fail_at:
            reason = TerminationReason.TERMINAL_FAILURE
        terminated = reason is not None
        truncated = False
        if not terminated and self._t >= self.horizon_steps:
            reason = TerminationReason.HORIZON_EXPIRED
            truncated = True
        info = self._info(reason, complete=reason is TerminationReason.RECOVERY_COMPLETE)
        return self._obs(), -0.1, terminated, truncated, info


class BrokenEnv(FakeEnv):
    """``FakeEnv`` with exactly one contract breach selected by ``mutation``."""

    def __init__(self, mutation, **kwargs):
        super().__init__(**kwargs)
        self.mutation = mutation

    def _mutate_obs(self, obs):
        m = self.mutation
        if m == "obs_float64":
            return obs.astype(np.float64)
        if m == "obs_above_one":
            bad = obs.copy()
            bad[0] = 1.5
            return bad
        if m == "obs_short":
            return obs[:-1]
        if m == "obs_nan":
            bad = obs.copy()
            bad[3] = np.nan
            return bad
        if m == "obs_list":
            return obs.tolist()
        return obs

    def _mutate_info(self, info, ended):
        m = self.mutation
        info = dict(info)
        if m == "info_missing_key":
            info.pop("latency_p99")
        elif m == "info_np_bool":
            info["violated_latency"] = np.bool_(False)
        elif m == "info_prob_above_one":
            info["p_failure_before_complete"] = 1.5
        elif m == "no_reason" and ended:
            info.pop("termination_reason", None)
        elif m == "bad_reason" and ended:
            info["termination_reason"] = "banana"
        return info

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return self._mutate_obs(obs), self._mutate_info(info, ended=False)

    def step(self, action):
        if self.mutation == "accepts_bad_action":
            try:
                result = super().step(action)
            except ContractViolation:
                result = super().step(0)
        else:
            result = super().step(action)
        obs, reward, terminated, truncated, info = result
        m = self.mutation
        if m == "reward_nan":
            reward = float("nan")
        elif m == "reward_int":
            reward = 0
        elif m == "terminated_np_bool":
            terminated = np.bool_(terminated)
        elif m == "truncated_int":
            truncated = int(truncated)
        elif m == "wrong_arity":
            return obs, reward, terminated, info
        elif m == "horizon_terminated" and truncated:
            terminated, truncated = True, False
        elif m == "recovery_truncated" and terminated:
            terminated, truncated = False, True
        elif m == "both_flags" and (terminated or truncated):
            terminated, truncated = True, True
        elif m == "reason_without_end" and not (terminated or truncated):
            info = dict(info, termination_reason=TerminationReason.RECOVERY_COMPLETE.value)
        return (
            self._mutate_obs(obs),
            reward,
            terminated,
            truncated,
            self._mutate_info(info, ended=bool(terminated) or bool(truncated)),
        )


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class _ConformanceTests:
    """Tests shared by every environment under check; subclasses supply ``factory``."""

    def test_reset_returns_valid_observation_and_info(self, factory):
        """reset() yields a float32 (12,) vector in [0, 1] and a contract-valid info
        dict; a policy network and the constraint accounting both depend on this."""
        check_reset(factory)

    def test_reset_is_repeatable(self, factory):
        """Training reuses one instance across episodes, so reset must work repeatedly."""
        check_reset_is_repeatable(factory)

    def test_step_outputs_follow_the_contract(self, factory):
        """Every step returns a clean 5-tuple: valid observation, finite float reward,
        genuine bool flags, and a valid info dict, for each of the four actions."""
        check_step_outputs(factory)

    def test_out_of_range_action_raises(self, factory):
        """An invalid action must raise rather than be clipped, otherwise a policy bug
        is masked as a legal allocation."""
        check_out_of_range_action_raises(factory)

    def test_ended_episode_carries_termination_reason(self, factory):
        """Evaluation attributes each episode to a cause, so the final step's info must
        name one of the TerminationReason values."""
        check_episode_ends_with_reason(factory)

    def test_episode_cannot_exceed_max_steps(self, factory):
        """The training loop sizes buffers by MAX_EPISODE_STEPS; an environment that
        keeps running past it would overflow them."""
        check_episode_length_bounded(factory)

    def test_reset_works_after_episode_end(self, factory):
        """The state left by a finished episode must not leak into the next one."""
        check_reset_after_episode(factory)


class TestExternalEnvironment(_ConformanceTests):
    """Runs the shared checks against the environment named by ISM_ENV_FACTORY."""

    @pytest.fixture
    def factory(self):
        return load_env_factory()


class TestReferenceEnvironment(_ConformanceTests):
    """Runs the shared checks against the in-file reference environment.

    The three parametrisations end an episode by recovery completion, by terminal
    failure, and by horizon expiry, covering every TerminationReason path.
    """

    @pytest.fixture(
        params=[
            pytest.param(lambda: FakeEnv(), id="recovery_complete"),
            pytest.param(lambda: FakeEnv(fail_at=50), id="terminal_failure"),
            pytest.param(lambda: FakeEnv(speed=0.0), id="horizon_expired"),
        ]
    )
    def factory(self, request):
        return request.param


class TestHarnessIsNotVacuous:
    def test_reference_env_passes_every_check(self):
        """Proves the harness can pass something: all checks accept the reference env."""
        for check in ALL_CHECKS:
            check(FakeEnv)

    def test_reference_env_covers_all_termination_reasons(self):
        """The reference environment produces each reason, so the reason check
        is exercised against all three legal values."""
        seen = set()
        for make in (FakeEnv, lambda: FakeEnv(fail_at=50), lambda: FakeEnv(speed=0.0)):
            _, info, _, _ = run_episode(make)
            seen.add(info["termination_reason"])
        assert seen == _VALID_REASONS

    def test_reference_env_flags_come_from_termination_flags(self):
        """For each ending path the reference environment's flags equal
        ``termination_flags`` of its reported reason."""
        for make in (FakeEnv, lambda: FakeEnv(fail_at=50), lambda: FakeEnv(speed=0.0)):
            _, info, terminated, truncated = run_episode(make)
            reason = TerminationReason(info["termination_reason"])
            assert (terminated, truncated) == termination_flags(reason)

    def test_horizon_expiry_takes_exactly_max_steps(self):
        """With no progress, the episode runs the full MAX_EPISODE_STEPS and then truncates."""
        steps, info, terminated, truncated = run_episode(lambda: FakeEnv(speed=0.0))
        assert steps == MAX_EPISODE_STEPS
        assert truncated and not terminated
        assert info["termination_reason"] == TerminationReason.HORIZON_EXPIRED.value

    @pytest.mark.parametrize(
        "mutation, kwargs",
        [
            ("obs_float64", {}),
            ("obs_above_one", {}),
            ("obs_short", {}),
            ("obs_nan", {}),
            ("obs_list", {}),
            ("info_missing_key", {}),
            ("info_np_bool", {}),
            ("info_prob_above_one", {}),
            ("reward_nan", {}),
            ("reward_int", {}),
            ("terminated_np_bool", {}),
            ("truncated_int", {}),
            ("wrong_arity", {}),
            ("no_reason", {}),
            ("no_reason", {"speed": 0.0}),
            ("bad_reason", {}),
            ("horizon_terminated", {"speed": 0.0}),
            ("recovery_truncated", {}),
            ("both_flags", {}),
            ("reason_without_end", {}),
            ("accepts_bad_action", {}),
            ("never_ends", {"speed": 0.0, "horizon_steps": 10 ** 9}),
        ],
        ids=lambda v: v if isinstance(v, str) else None,
    )
    def test_broken_environment_is_rejected(self, mutation, kwargs):
        """Each single deviation from the contract must make at least one check fail;
        a harness that accepts a broken environment would let it into the run path."""

        def factory():
            return BrokenEnv(mutation, **kwargs)

        with pytest.raises(AssertionError):
            for check in ALL_CHECKS:
                check(factory)


class TestFactoryDiscovery:
    def test_unset_variable_skips(self, monkeypatch):
        """No configured environment is a skip, never a failure."""
        monkeypatch.delenv(ENV_FACTORY_VAR, raising=False)
        with pytest.raises(pytest.skip.Exception):
            load_env_factory()

    @pytest.mark.parametrize("spec", ["", "   ", "no_colon", ":attr", "module:"])
    def test_malformed_spec_skips(self, spec):
        """A spec that is not module:callable cannot name an environment."""
        with pytest.raises(pytest.skip.Exception):
            load_env_factory(spec)

    def test_unimportable_module_skips(self):
        """A path to a module that does not exist is reported as a skip."""
        with pytest.raises(pytest.skip.Exception):
            load_env_factory("ism_no_such_module_xyz:make_env")

    def test_missing_attribute_skips(self):
        """An importable module without the named attribute is reported as a skip."""
        with pytest.raises(pytest.skip.Exception):
            load_env_factory(f"{__name__}:no_such_factory")

    def test_non_callable_target_skips(self):
        """The target must be callable to act as a factory."""
        with pytest.raises(pytest.skip.Exception):
            load_env_factory(f"{__name__}:ENV_FACTORY_VAR")

    def test_valid_spec_resolves_to_the_callable(self):
        """A well-formed spec returns the named object, resolved through importlib."""
        assert load_env_factory(f"{__name__}:FakeEnv") is FakeEnv

    def test_environment_variable_is_honoured(self, monkeypatch):
        """The default source of the spec is ISM_ENV_FACTORY."""
        monkeypatch.setenv(ENV_FACTORY_VAR, f"{__name__}:FakeEnv")
        assert load_env_factory() is FakeEnv
