"""Arena tests. None need the fitted hazard artifacts."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from arena.generate import extract_policy
from arena.score import aggregate, score_scenario
from arena.submission import SubmissionError, SubmittedPolicy, load_policy_fn

TEMPLATE = Path(__file__).resolve().parent / "submissions" / "_template.py"


# --------------------------------------------------------------------------
# Score
# --------------------------------------------------------------------------


def test_g_is_one_at_the_optimum_and_zero_at_the_rule() -> None:
    at_opt = score_scenario(0, j_policy=10.0, j_rule=0.0, j_star=10.0, violation_fraction=0.0,
                            completion_hours=1.0)
    at_rule = score_scenario(0, j_policy=0.0, j_rule=0.0, j_star=10.0, violation_fraction=0.0,
                             completion_hours=1.0)
    assert at_opt.g == pytest.approx(1.0)
    assert at_rule.g == pytest.approx(0.0)


def test_g_is_negative_below_the_rule() -> None:
    s = score_scenario(0, j_policy=-5.0, j_rule=0.0, j_star=10.0, violation_fraction=0.0,
                       completion_hours=1.0)
    assert s.g == pytest.approx(-0.5)


def test_g_is_invariant_to_rescaling_the_reward() -> None:
    """The point of the metric: assumed coefficients cannot flatter a submission."""
    base = score_scenario(0, 3.0, 0.0, 10.0, 0.0, 1.0).g
    for k in (0.1, 7.0, 1000.0):
        scaled = score_scenario(0, 3.0 * k, 0.0, 10.0 * k, 0.0, 1.0).g
        assert scaled == pytest.approx(base)


def test_scenario_without_headroom_is_excluded_not_scored() -> None:
    s = score_scenario(0, j_policy=1.0, j_rule=0.0, j_star=0.0, violation_fraction=0.0,
                       completion_hours=1.0)
    assert not s.controllable and math.isnan(s.g)
    agg = aggregate([s])
    assert agg.n_controllable == 0 and agg.n_excluded == 1


def test_optimum_below_rule_is_rejected() -> None:
    """A bad reference must fail loudly rather than produce a plausible number."""
    with pytest.raises(ValueError, match="below the rule"):
        score_scenario(0, j_policy=0.0, j_rule=10.0, j_star=0.0, violation_fraction=0.0,
                       completion_hours=1.0)


def test_g_above_one_is_reported_not_clipped() -> None:
    agg = aggregate([score_scenario(7, 20.0, 0.0, 10.0, 0.0, 1.0)])
    assert agg.impossible == [7]
    assert agg.mean_g > 1.0


def test_sem_is_across_scenarios() -> None:
    scores = [score_scenario(i, float(v), 0.0, 10.0, 0.0, 1.0)
              for i, v in enumerate([0.0, 5.0, 10.0])]
    agg = aggregate(scores)
    g = np.array([0.0, 0.5, 1.0])
    assert agg.mean_g == pytest.approx(g.mean())
    assert agg.sem_g == pytest.approx(g.std(ddof=1) / math.sqrt(3))


# --------------------------------------------------------------------------
# Submission interface
# --------------------------------------------------------------------------


def test_template_loads_and_returns_a_valid_action() -> None:
    policy = SubmittedPolicy(load_policy_fn(TEMPLATE), "template")
    obs = np.zeros(12, dtype=np.float32)
    assert policy(obs) in (0, 1, 2, 3)


def test_out_of_range_action_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.py"
    path.write_text("def policy(obs, state):\n    return 9\n", encoding="utf-8")
    with pytest.raises(SubmissionError, match="outside"):
        SubmittedPolicy(load_policy_fn(path), "bad")(np.zeros(12, dtype=np.float32))


def test_non_integer_action_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.py"
    path.write_text("def policy(obs, state):\n    return 1.5\n", encoding="utf-8")
    with pytest.raises(SubmissionError, match="int action"):
        SubmittedPolicy(load_policy_fn(path), "bad")(np.zeros(12, dtype=np.float32))


def test_raising_policy_is_a_submission_error(tmp_path: Path) -> None:
    path = tmp_path / "bad.py"
    path.write_text("def policy(obs, state):\n    raise RuntimeError('boom')\n", encoding="utf-8")
    with pytest.raises(SubmissionError, match="policy raised"):
        SubmittedPolicy(load_policy_fn(path), "bad")(np.zeros(12, dtype=np.float32))


def test_missing_policy_function_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.py"
    path.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(SubmissionError, match="no callable"):
        load_policy_fn(path)


def test_state_is_cleared_between_episodes(tmp_path: Path) -> None:
    path = tmp_path / "counter.py"
    path.write_text(
        "def policy(obs, state):\n"
        "    state['n'] = state.get('n', 0) + 1\n"
        "    return 0 if state['n'] < 3 else 1\n",
        encoding="utf-8",
    )
    policy = SubmittedPolicy(load_policy_fn(path), "counter")
    obs = np.zeros(12, dtype=np.float32)
    assert [policy(obs) for _ in range(4)] == [0, 0, 1, 1]
    policy.reset()
    assert policy(obs) == 0, "reset must clear per-episode memory"


def test_wrong_observation_shape_is_rejected() -> None:
    policy = SubmittedPolicy(load_policy_fn(TEMPLATE), "template")
    with pytest.raises(SubmissionError, match="observation must be"):
        policy(np.zeros(5, dtype=np.float32))


def test_policy_cannot_mutate_the_harness_observation() -> None:
    """The policy receives a copy, so writing to it cannot corrupt the simulator."""
    source = "def policy(obs, state):\n    obs[:] = 999.0\n    return 0\n"
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(source)
        path = Path(handle.name)
    policy = SubmittedPolicy(load_policy_fn(path), "mutator")
    obs = np.zeros(12, dtype=np.float32)
    policy(obs)
    assert np.all(obs == 0.0)


# --------------------------------------------------------------------------
# Generation helper
# --------------------------------------------------------------------------


def test_extract_policy_takes_the_last_fenced_block_defining_policy() -> None:
    reply = (
        "Here is a first attempt:\n```python\ndef policy(obs, state):\n    return 0\n```\n"
        "On reflection:\n```python\ndef policy(obs, state):\n    return 3\n```\n"
    )
    assert "return 3" in extract_policy(reply)


def test_extract_policy_ignores_fences_without_policy() -> None:
    reply = "```python\nimport numpy\n```\n```python\ndef policy(obs, state):\n    return 1\n```"
    assert "return 1" in extract_policy(reply)


def test_extract_policy_rejects_a_reply_with_no_policy() -> None:
    with pytest.raises(ValueError, match="no 'def policy'"):
        extract_policy("I am unable to help with that.")
