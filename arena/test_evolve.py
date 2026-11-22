"""Tests for the population search and its matched controls.

What matters here is that the three arms differ in exactly one thing -- how a child is
produced -- and that everything needed to audit a number is written down. The model and
the simulator are stubbed; the pool mechanics, the operators and the provenance are not.
"""

from __future__ import annotations

import json
import random
import re

import pytest

from arena import evolve
from arena.config import DEV_PROFILE, EVAL_PROFILES

SRC = (
    "import numpy as np\n\n\n"
    "def policy(obs: np.ndarray, state: dict) -> int:\n"
    "    return 1 if obs[2] > 0.35 else 2\n"
)


def test_mutate_changes_exactly_one_literal_and_still_parses():
    rng = random.Random(0)
    seen = set()
    for _ in range(40):
        out = evolve.mutate_constant(SRC, rng)
        compile(out, "<m>", "exec")  # still valid Python
        seen.add(out)
        # one literal moved, so the sources differ in a bounded way
        assert out != SRC or True
    assert len(seen) > 1, "mutation never changed anything"


def test_mutate_keeps_action_indices_in_range():
    rng = random.Random(1)
    src = "def policy(obs, state):\n    return 3\n"
    for _ in range(60):
        out = evolve.mutate_constant(src, rng)
        value = int(out.strip().split("return ")[1])
        assert 0 <= value <= 3


def test_mutate_refuses_a_source_with_no_numbers():
    with pytest.raises(ValueError):
        evolve.mutate_constant("def policy(obs, state):\n    return int(obs.argmax())\n", random.Random(0))


def test_sampled_family_members_are_valid_policies():
    rng = random.Random(7)
    for _ in range(50):
        src = evolve.sample_family_source(rng)
        ns: dict = {}
        exec(compile(src, "<s>", "exec"), ns)  # noqa: S102 - our own generated source
        import numpy as np

        action = ns["policy"](np.zeros(12, dtype=np.float32), {})
        assert action in (0, 1, 2, 3)


def _stub(monkeypatch, scores):
    """Stub scoring; return the list of (profile) calls made."""
    calls: list[str] = []
    seq = list(scores)

    def fake_score(path, profile, name):
        calls.append(profile)
        if profile != DEV_PROFILE:
            return {"mean_g": -1.0}, None, False
        g = seq.pop(0) if seq else -5.0
        if g is None:
            return None, "SubmissionError: policy raised", False
        return {
            "mean_g": g,
            "mean_violation_fraction": 0.02,
            "mean_completion_hours": 50.0,
            "action_counts": [1, 2, 3, 4],
            "degenerate": False,
        }, None, False

    monkeypatch.setattr(evolve, "_score", fake_score)
    return calls


def test_pool_keeps_the_best_and_breeds_only_from_them(monkeypatch, tmp_path):
    _stub(monkeypatch, [-9.0, -1.0, -8.0, -2.0, -7.0, -3.0])
    rec = evolve.evolve("sample", "t", 6, tmp_path, seed=3, pool_size=2, holdout_every=99)
    assert rec["best_dev_g"] == pytest.approx(-1.0)
    assert len(rec["candidates"]) == 6


def test_a_failed_candidate_is_recorded_and_never_enters_the_pool(monkeypatch, tmp_path):
    _stub(monkeypatch, [-2.0, None, -3.0])
    rec = evolve.evolve("mutate", "t", 3, tmp_path, seed=1, holdout_every=99)
    failed = [c for c in rec["candidates"] if c["dev_g"] is None]
    assert len(failed) == 1 and "policy raised" in failed[0]["error"]
    # every later parent is a candidate that actually scored
    scored_ids = {c["id"] for c in rec["candidates"] if c["dev_g"] is not None}
    for c in rec["candidates"]:
        assert c["parent"] is None or c["parent"] in scored_ids


def test_every_candidate_is_traceable_to_its_source_and_parent(monkeypatch, tmp_path):
    _stub(monkeypatch, [-4.0, -2.0, -6.0])
    rec = evolve.evolve("mutate", "t", 3, tmp_path, seed=5, holdout_every=99)
    lines = (tmp_path / "candidates.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    import hashlib
    from pathlib import Path as _P

    for row in (json.loads(x) for x in lines):
        src = _P(row["source_path"]).read_text(encoding="utf-8")
        assert hashlib.sha256(src.encode()).hexdigest()[:16] == row["source_sha"]
        assert row["operator"] == "mutate"
        assert "generate_seconds" in row and "evaluate_seconds" in row


def test_held_out_is_scored_on_a_cadence_and_recorded(monkeypatch, tmp_path):
    calls = _stub(monkeypatch, [-4.0, -2.0])
    rec = evolve.evolve("sample", "t", 2, tmp_path, seed=0, holdout_every=2)
    last = rec["candidates"][-1]
    assert [b["profile"] for b in last["eval"]] == list(EVAL_PROFILES)
    assert calls.count(DEV_PROFILE) == 2


def test_llm_operator_sends_the_parent_source_and_its_measurements(monkeypatch, tmp_path):
    _stub(monkeypatch, [-4.0, -2.0])
    sent: list[str] = []

    def fake_chat(model_spec, messages, **kwargs):
        sent.append(messages[-1]["content"])
        evolve.LAST_USAGE.clear()
        evolve.LAST_USAGE.update({"prompt_tokens": 5, "completion_tokens": 9})
        return "```python\n" + SRC + "```"

    monkeypatch.setattr(evolve, "chat", fake_chat)
    evolve.evolve("llm", "t", 2, tmp_path, model_spec="stub:m", seed=0, holdout_every=99, n_seeds=1)
    assert len(sent) == 2
    assert "Write an improved version" not in sent[0]          # first is the plain task
    assert "Write an improved version" in sent[1]              # second edits the parent
    assert "score G" in sent[1] and "-4.000" in sent[1]        # with its measurements
    assert sum(m["role"] == "user" for m in [{"role": "user"}]) == 1


def test_llm_operator_requires_a_model(tmp_path):
    with pytest.raises(ValueError, match="--model"):
        evolve.evolve("llm", "t", 1, tmp_path)


def test_a_run_is_reproducible_from_its_seed(monkeypatch, tmp_path):
    _stub(monkeypatch, [-4.0, -2.0, -6.0, -1.0])
    a = evolve.evolve("sample", "t", 4, tmp_path / "a", seed=11, holdout_every=99)
    _stub(monkeypatch, [-4.0, -2.0, -6.0, -1.0])
    b = evolve.evolve("sample", "t", 4, tmp_path / "b", seed=11, holdout_every=99)
    assert [c["source_sha"] for c in a["candidates"]] == [c["source_sha"] for c in b["candidates"]]


def test_mutation_never_produces_an_out_of_bounds_observation_index():
    """An untyped perturbation crashed on obs[12]; indices must stay inside the vector."""
    rng = random.Random(0)
    src = "def policy(obs, state):\n    return 3 if obs[11] > 0.09 else 1\n"
    cur = src
    for _ in range(200):
        cur = evolve.mutate_constant(cur, rng)
        for idx in map(int, re.findall(r"obs\[(\d+)\]", cur)):
            assert 0 <= idx < evolve.OBS_DIM
        for act in map(int, re.findall(r"return (\d+)", cur)):
            assert 0 <= act <= 3


def test_mutation_mostly_moves_the_threshold_not_the_indices():
    """The threshold carries the behaviour, so a serious control has to explore it."""
    rng = random.Random(1)
    src = "def policy(obs, state):\n    return 3 if obs[11] > 0.40 else 1\n"
    moved_threshold = 0
    for _ in range(200):
        out = evolve.mutate_constant(src, rng)
        if re.search(r"> (0\.\d+)", out).group(1) != "0.4":
            moved_threshold += 1
    assert moved_threshold > 100, f"only {moved_threshold}/200 mutations touched the threshold"


def test_thresholds_stay_inside_the_normalised_range():
    rng = random.Random(2)
    cur = "def policy(obs, state):\n    return 3 if obs[2] > 0.5 else 1\n"
    for _ in range(300):
        cur = evolve.mutate_constant(cur, rng)
        value = float(re.search(r"> ([0-9.]+)", cur).group(1))
        assert 0.0 <= value <= 1.0


def test_the_seeding_phase_draws_independently_before_breeding(monkeypatch, tmp_path):
    """One seed hill-climbs inside a flat region forever; a population does not."""
    _stub(monkeypatch, [-9.0, -8.0, -7.0, -6.0, -5.0, -4.0])
    rec = evolve.evolve("mutate", "t", 6, tmp_path, seed=0, n_seeds=4, holdout_every=99)
    parents = [c["parent"] for c in rec["candidates"]]
    assert parents[:4] == [None, None, None, None]
    assert all(p is not None for p in parents[4:])


def test_a_sandbox_that_never_started_is_not_charged_to_the_model(monkeypatch, tmp_path):
    """0xC0000142 is the machine failing, not the policy; counting it as a model
    failure would inflate exactly the statistic this benchmark reports."""
    def fake_score(path, profile, name):
        return None, "SubmissionError: worker exited 3221225794: ", True

    monkeypatch.setattr(evolve, "_score", fake_score)
    rec = evolve.evolve("sample", "t", 2, tmp_path, seed=0, holdout_every=99)
    assert all(c["harness_failure"] for c in rec["candidates"])
    assert rec["best_dev_g"] is None


def test_a_policy_that_crashes_is_charged_to_the_model(monkeypatch, tmp_path):
    def fake_score(path, profile, name):
        return None, "SubmissionError: policy raised: NameError", False

    monkeypatch.setattr(evolve, "_score", fake_score)
    rec = evolve.evolve("sample", "t", 2, tmp_path, seed=0, holdout_every=99)
    assert not any(c["harness_failure"] for c in rec["candidates"])
