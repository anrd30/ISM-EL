"""Cost ledger and spending guard for paid API generation calls.

Every call's cost is the provider's own reported token usage times the per-token price of
that exact model, appended to a JSON file. Before each call ``check_before_call`` refuses
to proceed if the running total plus the worst-case cost of that call would exceed
``MAX_TOTAL_USD``. No paid call is allowed until ``estimate`` has printed the worst-case
estimate of the planned batch.

Usage::

    python -m arena.cost_ledger estimate     # worst case for the planned roster; exit 1 above the ceiling
    python -m arena.cost_ledger report       # running total and per-model breakdown
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from arena import models

#: Hard spending ceiling in USD across the whole task. One-line change to adjust.
MAX_TOTAL_USD: float = 15.0

LEDGER_PATH = Path(__file__).resolve().parent / "cost_ledger.json"

#: Planned batch: models and repeats per model.
REPEATS_PER_MODEL: int = 5

#: USD per million tokens (input, output) at the standard synchronous rate, read from each
#: provider's pricing page on PRICES_AS_OF, keyed by roster variable of ``arena/.env``. The
#: price belongs to the model that variable held then: changing a model id in ``arena/.env``
#: requires updating its price here. Thinking and reasoning tokens bill as output and arrive
#: inside the provider's reported output count.
#:
#: Sources: platform.claude.com/docs/en/about-claude/pricing (Haiku up to 100k prompt
#: tokens), developers.openai.com/api/docs/pricing (short context, Standard),
#: aistudio.google.com/docs/pricing (Standard, prompts up to 200k tokens),
#: api-docs.deepseek.com/quick_start/pricing (cache-miss input, PEAK rate; off-peak is half).
PRICES_AS_OF = "2026-11-21"
PRICES_BY_VAR: dict[str, tuple[float, float]] = {
    "ARENA_MODEL_ANTHROPIC_HAIKU": (0.10, 0.50),
    "ARENA_MODEL_ANTHROPIC_SONNET": (2.00, 10.00),
    "ARENA_MODEL_ANTHROPIC_OPUS": (4.00, 20.00),
    "ARENA_MODEL_OPENAI_SOL": (2.00, 10.00),
    "ARENA_MODEL_GOOGLE_FLASH": (0.25, 1.50),
    "ARENA_MODEL_GOOGLE_PRO": (2.00, 12.00),
    "ARENA_MODEL_DEEPSEEK_FLASH": (0.30, 1.20),
    "ARENA_MODEL_DEEPSEEK_PRO": (1.32, 3.96),
}

#: Model id -> (input, output) and model id -> provider, resolved from ``arena/.env``.
PRICES: dict[str, tuple[float, float]] = {models.model_id(v): p for v, p in PRICES_BY_VAR.items()}
PROVIDERS: dict[str, str] = {models.model_id(v): models.provider_of(v) for v in PRICES_BY_VAR}

#: Characters per token assumed when estimating the prompt's length before a call (a low
#: value, so the estimate is high).
CHARS_PER_TOKEN_ESTIMATE = 3.0

__all__ = [
    "MAX_TOTAL_USD",
    "CostCeilingExceeded",
    "price_per_token_usd",
    "worst_case_call_usd",
    "estimate_batch",
    "check_before_call",
    "record",
    "load",
]


class CostCeilingExceeded(RuntimeError):
    """The next call could take the running total above ``MAX_TOTAL_USD``."""


def deepseek_is_peak(moment: datetime) -> bool:
    """True in DeepSeek's peak window: 01:00-04:00 and 06:00-10:00 UTC, Monday to Friday.

    Chinese public holidays are off-peak on the provider's side; they are treated as peak
    here, so the recorded cost is never below the billed one.
    """
    moment = moment.astimezone(timezone.utc)
    return moment.weekday() < 5 and (1 <= moment.hour < 4 or 6 <= moment.hour < 10)


def price_per_token_usd(model: str, moment: datetime | None = None) -> tuple[float, float]:
    """(input, output) USD per token for ``model`` at ``moment`` (now when omitted).

    Raises:
        KeyError: if the model has no entry in ``PRICES``.
    """
    inp, out = PRICES[model]
    if PROVIDERS.get(model) == "deepseek":
        # PRICES holds the peak rate; off-peak is half of it.
        now = moment or datetime.now(timezone.utc)
        if not deepseek_is_peak(now):
            inp, out = inp / 2.0, out / 2.0
    return inp / 1e6, out / 1e6


def worst_case_call_usd(model: str, input_tokens: int, max_new_tokens: int) -> float:
    """Cost of one call that reads ``input_tokens`` and writes the full ``max_new_tokens``, at the peak rate."""
    inp, out = PRICES[model]
    return (input_tokens * inp + max_new_tokens * out) / 1e6


def load(path: Path | None = None) -> dict:
    path = LEDGER_PATH if path is None else path
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"prices_as_of": PRICES_AS_OF, "ceiling_usd": MAX_TOTAL_USD, "estimate_printed": False,
            "calls": []}


def _save(data: dict, path: Path | None = None) -> None:
    path = LEDGER_PATH if path is None else path
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def total_usd(data: dict) -> float:
    return float(sum(c["cost_usd"] for c in data["calls"]))


def by_model(data: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for c in data["calls"]:
        row = out.setdefault(c["model"], {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
        row["calls"] += 1
        row["input_tokens"] += c["input_tokens"]
        row["output_tokens"] += c["output_tokens"]
        row["cost_usd"] += c["cost_usd"]
    return out


def prompt_tokens_estimate(prompt: str) -> int:
    return int(len(prompt) / CHARS_PER_TOKEN_ESTIMATE) + 1


def estimate_batch(prompt: str, max_new_tokens: int, models: Sequence[str] | None = None,
                   repeats: int = REPEATS_PER_MODEL) -> dict:
    """Worst-case cost of the planned batch: every call reads the whole prompt and writes ``max_new_tokens``."""
    tokens = prompt_tokens_estimate(prompt)
    rows = {}
    for model in models or list(PRICES):
        per_call = worst_case_call_usd(model, tokens, max_new_tokens)
        rows[model] = {"per_call_usd": per_call, "calls": repeats, "total_usd": per_call * repeats}
    return {"prompt_tokens_assumed": tokens, "max_new_tokens": max_new_tokens, "models": rows,
            "total_usd": sum(r["total_usd"] for r in rows.values())}


def check_before_call(model: str, prompt: str, max_new_tokens: int, path: Path | None = None) -> None:
    """Raise before a paid call that is unpriced, unestimated, or could take the total above the ceiling."""
    if model not in PRICES:
        raise CostCeilingExceeded(f"no price recorded for {model!r}; add it to PRICES before calling")
    data = load(path)
    if not data.get("estimate_printed"):
        raise CostCeilingExceeded("run `python -m arena.cost_ledger estimate` before any paid call")
    worst = worst_case_call_usd(model, prompt_tokens_estimate(prompt), max_new_tokens)
    running = total_usd(data)
    if running + worst > MAX_TOTAL_USD:
        raise CostCeilingExceeded(
            f"running total ${running:.4f} + worst case ${worst:.4f} of this call would exceed "
            f"${MAX_TOTAL_USD:.2f}; per model: {json.dumps(by_model(data))}"
        )


def record(provider: str, model: str, input_tokens: int, output_tokens: int,
           path: Path | None = None) -> float:
    """Append one call from the provider's reported usage and return its cost in USD."""
    now = datetime.now(timezone.utc)
    inp, out = price_per_token_usd(model, now)
    cost = input_tokens * inp + output_tokens * out
    data = load(path)
    data["calls"].append({
        "timestamp": now.isoformat(), "provider": provider, "model": model,
        "input_tokens": int(input_tokens), "output_tokens": int(output_tokens),
        "cost_usd": cost,
    })
    data["total_usd"] = total_usd(data)
    _save(data, path)
    return cost


def _estimate_cli() -> int:
    from arena.config import MAX_NEW_TOKENS

    prompt = (Path(__file__).resolve().parent / "prompts" / "task_prompt.md").read_text(encoding="utf-8")
    est = estimate_batch(prompt, MAX_NEW_TOKENS)
    print(f"prices as of {PRICES_AS_OF}; prompt tokens assumed {est['prompt_tokens_assumed']} "
          f"(at {CHARS_PER_TOKEN_ESTIMATE} chars/token), output {est['max_new_tokens']} per call, "
          f"{REPEATS_PER_MODEL} repeats per model")
    for model, row in est["models"].items():
        print(f"  {model:26s} per call ${row['per_call_usd']:.4f}   x{row['calls']} = ${row['total_usd']:.4f}")
    print(f"WORST CASE {sum(r['calls'] for r in est['models'].values())} calls: ${est['total_usd']:.4f} "
          f"(ceiling ${MAX_TOTAL_USD:.2f})")
    if est["total_usd"] > MAX_TOTAL_USD:
        print("ABOVE CEILING")
        return 1
    data = load()
    data["estimate_printed"] = True
    data["worst_case_estimate_usd"] = est["total_usd"]
    data["ceiling_usd"] = MAX_TOTAL_USD
    _save(data)
    return 0


def _report_cli() -> int:
    data = load()
    print(f"calls={len(data['calls'])} total=${total_usd(data):.4f} ceiling=${MAX_TOTAL_USD:.2f}")
    for model, row in sorted(by_model(data).items()):
        print(f"  {model:26s} calls={row['calls']} in={row['input_tokens']} out={row['output_tokens']} "
              f"${row['cost_usd']:.4f}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv if argv is not None else sys.argv[1:])
    if args == ["estimate"]:
        return _estimate_cli()
    if args == ["report"]:
        return _report_cli()
    print("usage: python -m arena.cost_ledger estimate|report")
    return 2


if __name__ == "__main__":
    sys.exit(main())
