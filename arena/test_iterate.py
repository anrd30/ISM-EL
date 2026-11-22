"""Tests for the iterative track and its search control.

The expensive parts (calling a model, running the simulator) are stubbed. What is
checked is the protocol: that feedback carries only development measurements, that
memory is carried or dropped as asked, that a failing round is recorded and fed back
rather than stopping the run, and that the held-out environments never reach the model.
"""

from __future__ import annotations

import json

import pytest

from arena import iterate, search_baseline
from arena.config import DEV_PROFILE, EVAL_PROFILES

DEV_RESULT = {
    "mean_g": -1.25,
    "mean_violation_fraction": 0.0231,
    "mean_completion_hours": 54.1,
    "max_step_ms": 0.42,
    "action_counts": [10, 90, 0, 0],
    "degenerate": False,
}

POLICY = "```python\nimport numpy as np\n\ndef policy(obs, state):\n    return 1\n```"


def test_feedback_reports_development_measurements_only():
    text = iterate.feedback_message(DEV_RESULT, None)
    assert "-1.250" in text
    assert "0.0231" in text
    assert "54.1" in text
    for profile in EVAL_PROFILES:
        assert profile not in text
    # Nothing about the held-out set, by name or by hint.
    assert "held-out" not in text.lower()
    assert "altB" not in text and "popC" not in text


def test_feedback_on_a_crash_asks_for_a_repair_and_quotes_the_error():
    text = iterate.feedback_message(None, "policy took 12.1 ms, budget is 10 ms")
    assert "12.1 ms" in text
    assert "did not run" in text


def test_degenerate_policy_is_called_out():
    result = dict(DEV_RESULT, degenerate=True, action_counts=[100, 0, 0, 0])
    assert "same action on every decision" in iterate.feedback_message(result, None)


def _stub(monkeypatch, tmp_path, replies, scores):
    """Stub the model and the simulator; return the list of conversations sent."""
    sent: list[list[dict]] = []

    def fake_chat(model_spec, messages, **kwargs):
        sent.append([dict(m) for m in messages])
        iterate.LAST_USAGE.clear()
        iterate.LAST_USAGE.update({"prompt_tokens": 10, "completion_tokens": 20})
        return replies[len(sent) - 1]

    calls: list[str] = []

    def fake_score(path, profile, name):
        calls.append(profile)
        return (scores.pop(0) if profile == DEV_PROFILE else dict(DEV_RESULT)), None

    monkeypatch.setattr(iterate, "chat", fake_chat)
    monkeypatch.setattr(iterate, "_score", fake_score)
    return sent, calls


def test_full_memory_carries_every_earlier_round(monkeypatch, tmp_path):
    sent, _ = _stub(monkeypatch, tmp_path, [POLICY] * 3, [dict(DEV_RESULT) for _ in range(3)])
    iterate.run_repeat("stub:model", "m", 1, 3, tmp_path, memory="full", holdout_every=99)
    assert len(sent[0]) == 1
    assert len(sent[1]) == 3  # task, assistant, feedback
    assert len(sent[2]) == 5  # ... and the second pair
    assert sent[2][1]["role"] == "assistant"


def test_no_memory_keeps_only_the_task_and_the_latest_feedback(monkeypatch, tmp_path):
    sent, _ = _stub(monkeypatch, tmp_path, [POLICY] * 3, [dict(DEV_RESULT) for _ in range(3)])
    iterate.run_repeat("stub:model", "m", 1, 3, tmp_path, memory="none", holdout_every=99)
    assert len(sent[1]) == 2 and len(sent[2]) == 2
    assert all(m["role"] == "user" for m in sent[2])


def test_a_reply_without_a_policy_is_recorded_and_the_run_continues(monkeypatch, tmp_path):
    sent, _ = _stub(monkeypatch, tmp_path, ["sorry, no code", POLICY], [dict(DEV_RESULT)])
    record = iterate.run_repeat("stub:model", "m", 1, 2, tmp_path, holdout_every=99)
    assert record["rounds"][0]["dev"] is None
    assert "no policy" in record["rounds"][0]["error"]
    assert record["rounds"][1]["dev"]["mean_g"] == pytest.approx(-1.25)
    assert "did not run" in sent[1][-1]["content"]


def test_held_out_environments_are_scored_but_never_sent_to_the_model(monkeypatch, tmp_path):
    sent, calls = _stub(monkeypatch, tmp_path, [POLICY], [dict(DEV_RESULT)])
    record = iterate.run_repeat("stub:model", "m", 1, 1, tmp_path, holdout_every=1)
    assert [b["profile"] for b in record["rounds"][0]["eval"]] == list(EVAL_PROFILES)
    assert calls[0] == DEV_PROFILE
    whole_conversation = json.dumps(sent)
    for profile in EVAL_PROFILES:
        assert profile not in whole_conversation


def test_the_record_is_written_with_tokens_and_timings(monkeypatch, tmp_path):
    _stub(monkeypatch, tmp_path, [POLICY], [dict(DEV_RESULT)])
    iterate.run_repeat("stub:model", "m", 2, 1, tmp_path, holdout_every=99)
    written = json.loads((tmp_path / "repeat2.json").read_text(encoding="utf-8"))
    assert written["memory"] == "full" and written["track"] == "iterative"
    assert written["rounds"][0]["completion_tokens"] == 20
    assert "generation_seconds" in written["rounds"][0]


def test_search_family_is_constants_plus_one_threshold_rules():
    names = search_baseline.family()
    assert len(names) == len(set(names))
    assert names[:4] == ["const0", "const1", "const2", "const3"]
    # 2 fields x 19 thresholds x 12 ordered action pairs, plus the 4 constants.
    assert len(names) == 4 + 2 * 19 * 12


@pytest.mark.parametrize(
    "name,obs,expected",
    [
        ("const3", [0.0] * 12, 3),
        ("f2_t0.30_hi1_lo2", [0, 0, 0.9] + [0] * 9, 1),
        ("f2_t0.30_hi1_lo2", [0, 0, 0.1] + [0] * 9, 2),
        ("f11_t0.50_hi0_lo3", [0] * 11 + [0.9], 0),
    ],
)
def test_a_member_is_fully_determined_by_its_name(name, obs, expected):
    import numpy as np

    assert search_baseline._policy(name)(np.array(obs, dtype=np.float32), {}) == expected


def test_the_sine_floor_is_a_member_of_this_family():
    """``1 if sin(obs[2]) > 0.3 else 2`` is a threshold at 0.305 on obs[2], not nonsense."""
    import math

    import numpy as np

    threshold = search_baseline._policy("f2_t0.30_hi1_lo2")
    for x in (0.0, 0.1, 0.2, 0.31, 0.5, 0.9):
        obs = np.array([0, 0, x] + [0] * 9, dtype=np.float32)
        sine = 1 if math.sin(obs[2]) > 0.3 else 2
        assert threshold(obs, {}) == sine, f"differ at obs[2]={x}"


def test_context_guard_stops_before_the_backend_would_truncate(monkeypatch, tmp_path):
    """Ollama clips a long conversation silently; a clipped memory arm measures nothing."""
    sent, _ = _stub(monkeypatch, tmp_path, [POLICY] * 3, [dict(DEV_RESULT) for _ in range(3)])
    record = iterate.run_repeat(
        "stub:model", "m", 1, 3, tmp_path, holdout_every=99, num_ctx=2048, max_new_tokens=2048
    )
    assert sent == []  # the very first round already does not fit
    assert record["rounds"][0]["context_exhausted"]["num_ctx"] == 2048
    assert record["num_ctx"] == 2048


def test_a_reasoning_model_that_never_reaches_code_is_a_reply_not_a_transport_failure(monkeypatch):
    """qwen3 returns empty `content` and a populated `thinking` when the budget runs out."""
    import json as _json
    from arena import generate as g

    payload = {
        "message": {"role": "assistant", "content": "", "thinking": "Okay, let me think..."},
        "prompt_eval_count": 7,
        "eval_count": 600,
        "done_reason": "length",
    }

    class _Response:
        def read(self):
            return _json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: _Response())
    reply = g._call_ollama("m", [{"role": "user", "content": "x"}], "http://h", 0.0, 1, 600)
    assert reply == "Okay, let me think..."
    assert g.LAST_USAGE["completion_tokens"] == 600
    with pytest.raises(ValueError):
        g.extract_policy(reply)  # recorded as a reply without a policy


def test_num_ctx_is_sent_to_ollama_when_given(monkeypatch):
    from arena import generate as g

    captured = {}

    class _Response:
        def read(self):
            return b'{"message": {"content": "ok"}}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(request, *a, **k):
        import json as _json

        captured.update(_json.loads(request.data))
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    g._call_ollama("m", [{"role": "user", "content": "x"}], "http://h", 0.5, 2, 99, num_ctx=4096)
    assert captured["options"]["num_ctx"] == 4096
    assert captured["options"]["num_predict"] == 99
