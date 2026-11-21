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


# --------------------------------------------------------------------------
# Sandbox: a policy must not be able to read the future
# --------------------------------------------------------------------------

CHEATS = {
    "open_trace": 'def policy(obs, state):\n'
                  '    open("data_pipeline/profiles/profile_diurnal_long_holdout.csv").readline()\n'
                  '    return 2\n',
    "module_scope_open": 'T = open("data_pipeline/profiles/profile_diurnal_long_holdout.csv").read()\n'
                         'def policy(obs, state):\n    return 2\n',
    "import_os": 'def policy(obs, state):\n    import os\n    os.listdir(".")\n    return 2\n',
    "import_sim": 'def policy(obs, state):\n'
                  '    from sim.scenario import load_replay_csv\n    return 2\n',
    "import_arena_reference": 'def policy(obs, state):\n'
                              '    from arena.reference import reference_table\n    return 2\n',
    "import_pandas": 'def policy(obs, state):\n    import pandas\n    return 2\n',
    "importlib_escape": 'def policy(obs, state):\n'
                        '    import importlib\n    importlib.import_module("os")\n    return 2\n',
    "socket": 'def policy(obs, state):\n    import socket\n    return 2\n',
}


@pytest.mark.parametrize("name", sorted(CHEATS))
def test_sandbox_blocks_reaching_outside_the_observation(name: str, tmp_path: Path) -> None:
    """Each of these is a route to the workload trace or the reference optimum.

    A policy that reads the trace is the lookahead oracle, so this is a validity
    control, not only a security one.
    """
    from arena.sandbox import run_sandboxed

    path = tmp_path / f"{name}.py"
    path.write_text(CHEATS[name], encoding="utf-8")
    result = run_sandboxed(path, "profile_diurnal_long_holdout.csv", name, timeout_s=300)
    assert "error" in result, f"{name} was NOT blocked"
    assert "Restriction" in result["error"], result["error"]


def test_sandbox_allows_a_legitimate_policy(tmp_path: Path) -> None:
    path = tmp_path / "ok.py"
    path.write_text(
        "import numpy as np\nimport math\n"
        "def policy(obs, state):\n"
        "    state['n'] = state.get('n', 0) + 1\n"
        "    return int(np.clip(int(obs[11] * 4), 0, 3))\n",
        encoding="utf-8",
    )
    from arena.sandbox import run_sandboxed

    result = run_sandboxed(path, "profile_diurnal_long_holdout.csv", "ok", timeout_s=600)
    assert "error" not in result, result.get("error")
    assert result["n_decisions"] > 0


def test_constant_policy_is_flagged_degenerate(tmp_path: Path) -> None:
    """Two models produced constant policies that only identical scores revealed."""
    from arena.sandbox import run_sandboxed

    path = tmp_path / "const.py"
    path.write_text("def policy(obs, state):\n    return 3\n", encoding="utf-8")
    result = run_sandboxed(path, "profile_diurnal_long_holdout.csv", "const", timeout_s=600)
    assert "error" not in result, result.get("error")
    assert result["degenerate"] is True
    assert result["distinct_actions"] == 1
    assert result["modal_action_share"] == pytest.approx(1.0)



# --------------------------------------------------------------------------
# Environments (spec 1.5.0)
# --------------------------------------------------------------------------


def test_default_arena_config_is_unchanged_by_variants() -> None:
    from arena.config import SPEC_VERSION, arena_config
    from contracts.contracts import RewardConfig

    base = arena_config()
    assert base.label == f"arena-{SPEC_VERSION}"
    assert base.reward == RewardConfig()
    assert (base.latency_unloaded_ms, base.latency_target_ms) == (2.0, 15.0)
    assert arena_config(None).label == base.label


def test_every_environment_resolves_and_variants_differ_from_the_default() -> None:
    from arena.config import ENV_VARIANTS, EVAL_PROFILES, PROFILES, arena_config, reference_stem, split_env

    assert EVAL_PROFILES[0] == "profile_diurnal_long_holdout.csv"
    assert len(set(EVAL_PROFILES)) == len(EVAL_PROFILES)
    stems = set()
    for env in EVAL_PROFILES:
        base, variant = split_env(env)
        assert (PROFILES / base).is_file()
        stems.add(reference_stem(env))
        if variant:
            cfg, default = arena_config(variant), arena_config()
            assert (cfg.reward, cfg.latency_unloaded_ms, cfg.latency_target_ms) != (
                default.reward, default.latency_unloaded_ms, default.latency_target_ms)
    assert len(stems) == len(EVAL_PROFILES)
    assert set(ENV_VARIANTS) == {split_env(e)[1] for e in EVAL_PROFILES if "@" in e}


def test_unknown_variant_is_rejected() -> None:
    from arena.config import split_env

    with pytest.raises(ValueError):
        split_env("profile_diurnal_long_holdout.csv@nope")


def test_reference_values_exist_for_every_environment_and_are_valid() -> None:
    import pandas as pd
    from arena.config import EVAL_PROFILES, REFERENCE, reference_stem

    for env in EVAL_PROFILES:
        path = REFERENCE / f"{reference_stem(env)}.csv"
        assert path.is_file(), path
        ref = pd.read_csv(path)
        assert len(ref) == 20 and (ref["j_star"] >= ref["j_rule"] - 1e-9).all()


# --------------------------------------------------------------------------
# Cost ledger and API providers (no network)
# --------------------------------------------------------------------------


def test_cost_guard_requires_an_estimate_and_respects_the_ceiling(tmp_path) -> None:
    from arena import cost_ledger as L

    ledger = tmp_path / "ledger.json"
    with pytest.raises(L.CostCeilingExceeded, match="estimate"):
        L.check_before_call("claude-haiku-5-5", "p" * 300, 8192, ledger)
    data = L.load(ledger)
    data["estimate_printed"] = True
    L._save(data, ledger)
    L.check_before_call("claude-haiku-5-5", "p" * 300, 8192, ledger)
    L.record("openai", "gpt-6.2-sol", 1_000_000, 1_500_000, ledger)  # 17 USD
    with pytest.raises(L.CostCeilingExceeded, match="exceed"):
        L.check_before_call("claude-haiku-5-5", "p" * 300, 8192, ledger)
    with pytest.raises(L.CostCeilingExceeded, match="no price"):
        L.check_before_call("unknown-model", "p", 10, ledger)


def test_ledger_records_reported_usage_at_the_model_price(tmp_path) -> None:
    from arena import cost_ledger as L

    ledger = tmp_path / "ledger.json"
    cost = L.record("anthropic", "claude-opus-5-5", 1000, 2000, ledger)
    assert cost == pytest.approx((1000 * 4.0 + 2000 * 20.0) / 1e6)
    data = L.load(ledger)
    assert L.total_usd(data) == pytest.approx(cost)
    assert L.by_model(data)["claude-opus-5-5"]["calls"] == 1


def test_deepseek_off_peak_is_half_the_peak_rate() -> None:
    from datetime import datetime, timezone
    from arena import cost_ledger as L

    peak = datetime(2026, 11, 23, 2, 0, tzinfo=timezone.utc)  # Monday 02:00 UTC
    off = datetime(2026, 11, 22, 2, 0, tzinfo=timezone.utc)  # Sunday
    assert L.deepseek_is_peak(peak) and not L.deepseek_is_peak(off)
    assert L.price_per_token_usd("deepseek-v4-pro", off)[1] == pytest.approx(
        L.price_per_token_usd("deepseek-v4-pro", peak)[1] / 2)
    assert L.price_per_token_usd("claude-opus-5-5", off) == L.price_per_token_usd("claude-opus-5-5", peak)


def test_worst_case_estimate_of_the_planned_batch_is_under_the_ceiling() -> None:
    from arena import cost_ledger as L
    from arena.config import MAX_NEW_TOKENS
    from arena.generate import PROMPT

    est = L.estimate_batch(PROMPT.read_text(encoding="utf-8"), MAX_NEW_TOKENS)
    assert len(est["models"]) == 8 and est["total_usd"] < L.MAX_TOTAL_USD


def test_missing_api_key_is_named(monkeypatch) -> None:
    from arena import generate as g

    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY or GEMINI_API_KEY"):
        g._api_key("GOOGLE_API_KEY", "GEMINI_API_KEY")
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        g._api_key("DEEPSEEK_API_KEY")


def test_provider_branches_call_the_sdk_and_record_usage(monkeypatch, tmp_path) -> None:
    """Each branch records the reported usage in the ledger and returns text only."""
    import sys
    import types

    from arena import cost_ledger as L
    from arena import generate as g

    ledger = tmp_path / "ledger.json"
    L._save({"estimate_printed": True, "calls": []}, ledger)
    monkeypatch.setattr(L, "LEDGER_PATH", ledger)
    for name in ("OPENAI_API_KEY", "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(name, "test")
    seen: dict[str, dict] = {}

    class _Msg:
        def __init__(self, text):
            self.content = text

    class _Choice:
        def __init__(self, text):
            self.message = _Msg(text)

    class _Completions:
        def __init__(self, init):
            self.init = init

        def create(self, **kw):
            seen[self.init.get("base_url") or "openai"] = kw
            return types.SimpleNamespace(choices=[_Choice("def policy(o, s): return 1")],
                                         usage=types.SimpleNamespace(prompt_tokens=10, completion_tokens=20))

    class _OpenAI:
        def __init__(self, **init):
            self.chat = types.SimpleNamespace(completions=_Completions(init))

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_OpenAI))

    class _Anthropic:
        def __init__(self, **init):
            self.messages = types.SimpleNamespace(create=self.create)

        def create(self, **kw):
            seen["anthropic"] = kw
            return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text="def policy(o, s): return 2")],
                                         usage=types.SimpleNamespace(input_tokens=11, output_tokens=22))

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_Anthropic))

    class _Models:
        def generate_content(self, **kw):
            seen["google"] = kw
            return types.SimpleNamespace(
                text="def policy(o, s): return 3",
                usage_metadata=types.SimpleNamespace(prompt_token_count=12, candidates_token_count=20,
                                                     thoughts_token_count=30))

    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    genai.Client = lambda **init: types.SimpleNamespace(models=_Models())
    genai.types = types.SimpleNamespace(GenerateContentConfig=lambda **kw: kw)
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)

    for provider, model in (("openai", "gpt-6.2-sol"), ("deepseek", "deepseek-v4-pro"),
                            ("anthropic", "claude-haiku-5-5"), ("google", "gemini-3.1-pro-preview")):
        assert "def policy" in g._call_api(provider, model, "prompt", 0.7, 8192)
    assert seen["openai"]["max_completion_tokens"] == 8192 and seen["openai"]["temperature"] == 0.7
    assert seen["https://api.deepseek.com"]["max_tokens"] == 8192
    assert seen["anthropic"]["max_tokens"] == 8192
    assert seen["google"]["config"]["max_output_tokens"] == 8192
    calls = L.load(ledger)["calls"]
    assert [(c["model"], c["input_tokens"], c["output_tokens"]) for c in calls] == [
        ("gpt-6.2-sol", 10, 20), ("deepseek-v4-pro", 10, 20), ("claude-haiku-5-5", 11, 22),
        ("gemini-3.1-pro-preview", 12, 50)]
    with pytest.raises(ValueError):
        g._call_api("nope", "m", "p")
