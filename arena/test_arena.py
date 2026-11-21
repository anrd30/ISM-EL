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



def _id(var: str) -> str:
    from arena.models import model_id

    return model_id(var)


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
        L.check_before_call(_id("ARENA_MODEL_ANTHROPIC_HAIKU"), "p" * 300, 8192, ledger)
    data = L.load(ledger)
    data["estimate_printed"] = True
    L._save(data, ledger)
    L.check_before_call(_id("ARENA_MODEL_ANTHROPIC_HAIKU"), "p" * 300, 8192, ledger)
    L.record("openai", _id("ARENA_MODEL_OPENAI_SOL"), 1_000_000, 1_500_000, ledger)  # 17 USD
    with pytest.raises(L.CostCeilingExceeded, match="exceed"):
        L.check_before_call(_id("ARENA_MODEL_ANTHROPIC_HAIKU"), "p" * 300, 8192, ledger)
    with pytest.raises(L.CostCeilingExceeded, match="no price"):
        L.check_before_call("unknown-model", "p", 10, ledger)


def test_ledger_records_reported_usage_at_the_model_price(tmp_path) -> None:
    from arena import cost_ledger as L

    ledger = tmp_path / "ledger.json"
    cost = L.record("anthropic", _id("ARENA_MODEL_ANTHROPIC_OPUS"), 1000, 2000, ledger)
    assert cost == pytest.approx((1000 * 4.0 + 2000 * 20.0) / 1e6)
    data = L.load(ledger)
    assert L.total_usd(data) == pytest.approx(cost)
    assert L.by_model(data)[_id("ARENA_MODEL_ANTHROPIC_OPUS")]["calls"] == 1


def test_worst_case_estimate_of_the_planned_batch_is_under_the_ceiling() -> None:
    from arena import cost_ledger as L
    from arena.config import MAX_NEW_TOKENS
    from arena.generate import PROMPT

    from arena.models import batch_roster

    PRICES_IN_BATCH = list(batch_roster().values())
    est = L.estimate_batch(PROMPT.read_text(encoding="utf-8"), MAX_NEW_TOKENS, PRICES_IN_BATCH)
    assert len(est["models"]) == len(PRICES_IN_BATCH) and est["total_usd"] < L.MAX_TOTAL_USD


def test_missing_api_key_is_named_and_model_ids_come_from_the_env_file(monkeypatch) -> None:
    from arena import models

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        models.key_for(models.API_PROVIDER)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    assert models.key_for("openrouter") == "k"
    roster = models.roster()
    assert all(v.startswith("ARENA_MODEL_") and "/" in m for v, m in roster.items())
    assert models.provider_of("ARENA_MODEL_GOOGLE_PRO") == "google"
    assert all(models.provider_of(v) not in models.DEFERRED_PROVIDERS for v in models.batch_roster())
    monkeypatch.setenv("ARENA_MODEL_OPENAI_SOL", "other/id")
    assert models.model_id("ARENA_MODEL_OPENAI_SOL") == "other/id"


def test_env_example_has_one_empty_key_and_filled_model_slugs() -> None:
    from dotenv import dotenv_values

    values = dotenv_values(__import__("arena.models", fromlist=["x"]).EXAMPLE_FILE)
    keys = {k: v for k, v in values.items() if k.endswith("_API_KEY")}
    assert set(keys) == {"OPENROUTER_API_KEY"} and not keys["OPENROUTER_API_KEY"]
    assert all(v for k, v in values.items() if k.startswith("ARENA_MODEL_"))


def test_openrouter_branch_calls_the_sdk_with_a_price_cap_and_records_usage(monkeypatch, tmp_path) -> None:
    """The branch sends slug, temperature, seed and token limit, caps the price, and records the reported cost."""
    import sys
    import types

    from arena import cost_ledger as L
    from arena import generate as g

    ledger = tmp_path / "ledger.json"
    L._save({"estimate_printed": True, "calls": []}, ledger)
    monkeypatch.setattr(L, "LEDGER_PATH", ledger)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    seen: dict = {}

    class _Completions:
        def create(self, **kw):
            seen.update(kw)
            usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=20, cost=0.0123)
            message = types.SimpleNamespace(content="def policy(o, s): return 1")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=usage)

    class _OpenAI:
        def __init__(self, **init):
            seen["init"] = init
            self.chat = types.SimpleNamespace(completions=_Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_OpenAI))
    slug = _id("ARENA_MODEL_OPENAI_SOL")
    assert "def policy" in g._call_api("openrouter", slug, [{"role": "user", "content": "prompt"}], 0.7, 8192, 3)
    assert seen["init"]["base_url"] == "https://openrouter.ai/api/v1"
    assert (seen["model"], seen["temperature"], seen["max_tokens"], seen["seed"]) == (slug, 0.7, 8192, 3)
    cap = seen["extra_body"]["provider"]["max_price"]
    assert (cap["prompt"], cap["completion"]) == L.PRICES[slug]
    (call,) = L.load(ledger)["calls"]
    assert (call["model"], call["input_tokens"], call["output_tokens"], call["cost_usd"]) == (slug, 10, 20, 0.0123)
    for gone in ("openai", "google", "deepseek"):
        with pytest.raises(ValueError):
            g._call_api(gone, "m", "p")


def test_frontier_driver_makes_no_call_while_a_key_is_missing(monkeypatch) -> None:
    from arena import run_frontier

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def _boom(args):
        raise AssertionError(f"driver ran {args}")

    monkeypatch.setattr(run_frontier, "_run", _boom)
    assert len(run_frontier.missing_keys()) == 1
    assert run_frontier.main() == 2


def test_pinned_model_is_sent_to_one_provider_without_fallback_and_the_host_is_recorded(monkeypatch, tmp_path) -> None:
    import sys
    import types

    from arena import cost_ledger as L
    from arena import generate as g
    from arena import models

    ledger = tmp_path / "ledger.json"
    L._save({"estimate_printed": True, "calls": []}, ledger)
    monkeypatch.setattr(L, "LEDGER_PATH", ledger)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test")
    monkeypatch.setenv("ARENA_PROVIDER_DEEPSEEK_PRO", "HostX")
    seen: dict = {}

    class _Completions:
        def create(self, **kw):
            seen.update(kw)
            usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=2)
            message = types.SimpleNamespace(content="def policy(o, s): return 1")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)], usage=usage,
                                         provider="HostX", model="vendor/served")

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(
        OpenAI=lambda **init: types.SimpleNamespace(chat=types.SimpleNamespace(completions=_Completions()))))
    slug = _id("ARENA_MODEL_DEEPSEEK_PRO")
    g._call_api("openrouter", slug, [{"role": "user", "content": "prompt"}], 0.7, 100, 1)
    prefs = seen["extra_body"]["provider"]
    assert prefs["only"] == ["HostX"] and prefs["allow_fallbacks"] is False
    assert g.LAST_CALL["served_by"] == "HostX" and g.LAST_CALL["pinned_provider"] == "HostX"
    unpinned = _id("ARENA_MODEL_OPENAI_SOL")
    monkeypatch.delenv("ARENA_PROVIDER_OPENAI_SOL", raising=False)
    g._call_api("openrouter", unpinned, [{"role": "user", "content": "prompt"}], 0.7, 100, 1)
    assert "only" not in seen["extra_body"]["provider"]
    assert models.display_label(slug).startswith("DeepSeek V4 Pro (0423")


def test_generation_budget_is_one_value_across_config_and_generate() -> None:
    from arena import generate
    from arena.config import MAX_NEW_TOKENS, SPEC_VERSION

    assert MAX_NEW_TOKENS == generate.DEFAULT_MAX_NEW_TOKENS == 32768
    assert SPEC_VERSION == "1.6.0"


def test_every_batch_model_is_pinned_to_one_host() -> None:
    from pathlib import Path

    from arena import models

    lines = (Path(models.__file__).parent / ".env.example").read_text(encoding="utf-8").splitlines()
    pinned = {l.split("=")[0] for l in lines if l.startswith("ARENA_PROVIDER_")}
    for var in models.batch_roster():
        assert "ARENA_PROVIDER_" + var[len("ARENA_MODEL_"):] in pinned, var


def test_headline_report_is_the_median_over_environments_with_a_beat_sine_count(tmp_path, monkeypatch, capsys) -> None:
    import json

    from arena import run
    from arena.config import EVAL_PROFILES

    def result(name, gs):
        return {"name": name, "error": None, "dev": {"mean_g": 0.0},
                "eval": [{"profile": e, "mean_g": g, "mean_violation_fraction": 0.0} for e, g in zip(EVAL_PROFILES, gs)]}

    n = len(EVAL_PROFILES)
    for r in (result(run.SINE_RESULT, [0.0] * n), result("m#1", [1.0] * 3 + [-5.0] * (n - 3)),
              result("m#2", [3.0] * 3 + [-5.0] * (n - 3))):
        (tmp_path / f"{r['name']}.json").write_text(json.dumps(r), encoding="utf-8")
    (tmp_path / "bad#1.json").write_text(json.dumps({"name": "bad#1", "error": "no policy", "eval": []}), encoding="utf-8")
    monkeypatch.setattr(run, "RESULTS", tmp_path)
    run._report_headline()
    out = capsys.readouterr().out
    line = next(l for l in out.splitlines() if l.split() and l.split()[0] == "m")
    assert "-5.0" in line and "3/8" in line and "2/2" in line
    assert next(l for l in out.splitlines() if l.split() and l.split()[0] == "bad").split()[-1] == "-"
