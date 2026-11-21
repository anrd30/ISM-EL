"""Ask a model for a policy and write it to a submission file.

Two backends, chosen by the ``--model`` prefix:

    ollama:<model>            a local Ollama server (no extra dependency)
    api:<provider>/<model>    an HTTP API: openrouter/<vendor>/<model> (every frontier model) or anthropic/<model>
    hf:<org>/<model>          a local transformers model, e.g. on a GPU

Examples::

    python -m arena.generate --model ollama:qwen2.5-coder:7b --out arena/submissions/qwen-7b
    python -m arena.generate --model api:openai/gpt-4o --out arena/submissions/gpt-4o
    python -m arena.generate --model hf:Qwen/Qwen2.5-Coder-7B-Instruct --out arena/submissions/qwen-7b

Neither backend is a dependency of the arena itself: scoring a submission needs only
numpy and the simulator, so a participant who writes a policy by hand, or pastes one
from a chat window, never imports this module.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Sequence

PROMPT = Path(__file__).resolve().parent / "prompts" / "task_prompt.md"

#: Generation budget. Must be generous: a reasoning model spends most of its output
#: thinking before it writes any code, and a budget that truncates mid-thought records
#: a capable model as producing no policy. DeepSeek-R1-Distill-32B exhausted 2048
#: tokens reasoning and never reached the function on all five repeats, which is a
#: property of this constant and not of the model.
DEFAULT_MAX_NEW_TOKENS = 32768

#: Exit codes of ``main``: 1 is a reply without a policy (the reply is kept); the others
#: mean no reply was obtained.
EXIT_NO_POLICY = 1
EXIT_GUARD = 3
EXIT_NO_KEY = 4

__all__ = ["extract_policy", "generate", "chat", "main", "EXIT_NO_POLICY", "EXIT_GUARD", "EXIT_NO_KEY"]

_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_policy(text: str) -> str:
    """Pull the policy source out of a model reply.

    Takes the last fenced block that defines ``policy``, or the whole reply if it
    is bare code.

    Raises:
        ValueError: if no block defines ``policy``.
    """
    blocks = [b for b in _FENCE.findall(text) if "def policy" in b]
    if blocks:
        return blocks[-1].strip() + "\n"
    if "def policy" in text:
        return text.strip() + "\n"
    raise ValueError("model reply contains no 'def policy'")


def _call_ollama(model: str, messages: list[dict], host: str, temperature: float, seed: int,
                 max_new_tokens: int) -> str:
    """One chat completion from a local Ollama server.

    ``messages`` is the whole conversation, so the iterative track can hand back
    earlier rounds. The zero-shot track passes a single user turn.

    Uses urllib so the arena gains no dependency: a participant with Ollama
    already running needs nothing else installed.

    Raises:
        RuntimeError: if the server is unreachable or returns no message.
    """
    import json as _json
    import urllib.error
    import urllib.request

    body = _json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            # Greedy at temperature 0, so a rerun reproduces the submission exactly.
            # Above 0 the seed is what makes a given repeat reproducible.
            "options": {"temperature": temperature, "seed": seed, "num_predict": max_new_tokens},
        }
    ).encode()
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            payload = _json.loads(response.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"cannot reach Ollama at {host}: {exc}. Is `ollama serve` running?"
        ) from exc
    message = payload.get("message", {}).get("content")
    if not message:
        raise RuntimeError(f"Ollama returned no message: {payload}")
    LAST_USAGE.clear()
    LAST_USAGE.update(
        {
            "prompt_tokens": int(payload.get("prompt_eval_count") or 0),
            "completion_tokens": int(payload.get("eval_count") or 0),
        }
    )
    return message


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

#: Routing details of the last ``_call_api`` call (the serving host, from the response), written next to the reply.
LAST_CALL: dict = {}

#: Token usage of the last call, on every backend. The iterative track reports cost per
#: round in tokens, which is the currency a search baseline does not spend at all.
LAST_USAGE: dict = {}


def _call_api(provider: str, model: str, messages: list[dict], temperature: float = 0.0,
              max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, seed: int | None = None) -> str:
    """One chat completion from a provider's API; returns the reply text only.

    ``openrouter`` serves every frontier model by its slug (``<vendor>/<model>``); the
    ``anthropic`` branch is the direct fallback and is not used by the current batch.

    Before the call ``arena.cost_ledger.check_before_call`` refuses to proceed if the
    spending ceiling could be exceeded; after it the provider's reported usage is
    recorded. The output budget counts reasoning tokens, so a reasoning model that
    exhausts it returns no code, which is a recorded result.
    """
    from arena import cost_ledger, models

    if provider not in ("openrouter", "anthropic"):
        raise ValueError(f"unknown provider {provider!r}; add it to _call_api")
    # The ledger estimates from character count, so the whole conversation is what
    # gets charged on an iterative round, not just the newest turn.
    cost_ledger.check_before_call(
        model, "\n".join(str(m.get("content", "")) for m in messages), max_new_tokens
    )

    if provider == "openrouter":
        from openai import OpenAI  # imported lazily; not an arena dependency

        client = OpenAI(api_key=models.key_for("openrouter"), base_url=OPENROUTER_BASE_URL)
        price_in, price_out = cost_ledger.PRICES[model]
        provider_prefs: dict = {"max_price": {"prompt": price_in, "completion": price_out}}
        pin = models.pinned_provider(model)
        if pin:
            # One host, no fallback: every repeat of the model lands on the same endpoint.
            provider_prefs.update({"only": [pin], "allow_fallbacks": False})
        reply = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_new_tokens,
            **({"seed": seed} if seed is not None else {}),
            # Refuse any endpoint dearer than the price the ledger assumes (USD per million tokens).
            extra_body={"provider": provider_prefs},
        )
        LAST_CALL.clear()
        LAST_CALL.update({"api": "openrouter", "requested_slug": model, "pinned_provider": pin,
                          "served_by": getattr(reply, "provider", None),
                          "served_model": getattr(reply, "model", None)})
        usage = reply.usage
        cost_ledger.record(
            provider, model, usage.prompt_tokens, usage.completion_tokens,
            reported_cost_usd=getattr(usage, "cost", None),
        )
        LAST_USAGE.clear()
        LAST_USAGE.update(
            {"prompt_tokens": usage.prompt_tokens, "completion_tokens": usage.completion_tokens}
        )
        return reply.choices[0].message.content or ""

    import anthropic

    client = anthropic.Anthropic(api_key=models.key_for("anthropic"))
    reply = client.messages.create(
        model=model,
        max_tokens=max_new_tokens,
        temperature=temperature,
        messages=messages,
    )
    cost_ledger.record(provider, model, reply.usage.input_tokens, reply.usage.output_tokens)
    LAST_USAGE.clear()
    LAST_USAGE.update(
        {
            "prompt_tokens": reply.usage.input_tokens,
            "completion_tokens": reply.usage.output_tokens,
        }
    )
    return "".join(b.text for b in reply.content if b.type == "text")


#: Loaded ``hf:`` models, keyed by model id. The iterative track calls the same model
#: tens of times in one process; without this each round reloads tens of GiB of weights
#: from disk and the loop spends most of its wall-clock in ``from_pretrained``.
_LOCAL_CACHE: dict = {}


def _call_local(model_id: str, messages: list[dict], temperature: float, seed: int,
                max_new_tokens: int) -> str:
    """Decode from a local transformers model, reusing the loaded weights."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_id in _LOCAL_CACHE:
        tok, model = _LOCAL_CACHE[model_id]
        return _generate_local(tok, model, messages, temperature, seed, max_new_tokens)

    tok = AutoTokenizer.from_pretrained(model_id)

    # Place the whole model on one GPU rather than letting accelerate shard it.
    # "auto" spreads layers across every visible device and silently falls back to
    # CPU offload, which on a shared machine means either a non-Ampere card that
    # cannot run FlashAttention, or a 32B model generating on CPU for hours. Both
    # happened. Pin to a single device and fail loudly if it does not fit.
    #
    # Select the device with CUDA_VISIBLE_DEVICES; index 0 is then the only one the
    # process can see.
    if torch.cuda.is_available():
        device_map: object = {"": 0}
        free, total = torch.cuda.mem_get_info(0)
        print(
            f"[arena] loading {model_id} onto {torch.cuda.get_device_name(0)}, "
            f"{free / 2**30:.1f} GiB free of {total / 2**30:.1f} GiB",
            flush=True,
        )
    else:
        device_map = None
        print("[arena] no CUDA device visible; loading on CPU (slow)", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map=device_map
    )
    _LOCAL_CACHE[model_id] = (tok, model)
    return _generate_local(tok, model, messages, temperature, seed, max_new_tokens)


def _generate_local(tok, model, messages: list[dict], temperature: float, seed: int,
                    max_new_tokens: int) -> str:
    """One decode from an already-loaded local model."""
    import torch

    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0.0,
        **({"temperature": temperature, "top_p": 0.95} if temperature > 0.0 else {}),
    )
    n_prompt = inputs["input_ids"].shape[1]
    LAST_USAGE.clear()
    LAST_USAGE.update(
        {"prompt_tokens": int(n_prompt), "completion_tokens": int(out.shape[1] - n_prompt)}
    )
    return tok.decode(out[0][n_prompt:], skip_special_tokens=True)


def generate(
    model_spec: str,
    prompt: str,
    host: str = "http://localhost:11434",
    temperature: float = 0.0,
    seed: int = 0,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> str:
    """Dispatch to the right backend and return the raw reply.

    ``temperature = 0`` is greedy and reproducible, which is right for a single
    submission. Repeats need ``temperature > 0``, where ``seed`` makes each repeat
    individually reproducible while letting the set of repeats measure how much a
    model's output varies.
    """
    return chat(
        model_spec,
        [{"role": "user", "content": prompt}],
        host=host,
        temperature=temperature,
        seed=seed,
        max_new_tokens=max_new_tokens,
    )


def chat(
    model_spec: str,
    messages: list[dict],
    host: str = "http://localhost:11434",
    temperature: float = 0.0,
    seed: int = 0,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> str:
    """Dispatch a whole conversation to the right backend and return the reply text.

    ``generate`` is the single-turn case. The iterative track passes every earlier
    round back, so the model can see what it already tried and what that scored,
    rather than reacting to a number with no memory of how it got there.

    Token counts for the call land in ``LAST_USAGE``.
    """
    kind, _, rest = model_spec.partition(":")
    if kind == "ollama":
        return _call_ollama(rest, messages, host, temperature, seed, max_new_tokens)
    if kind == "api":
        provider, _, name = rest.partition("/")
        return _call_api(provider, name, messages, temperature, max_new_tokens, seed)
    if kind == "hf":
        return _call_local(rest, messages, temperature, seed, max_new_tokens)
    raise ValueError(f"--model must start with 'ollama:', 'api:' or 'hf:', got {model_spec!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arena.generate")
    parser.add_argument(
        "--model", required=True, help="ollama:<model>, api:<provider>/<model> or hf:<org>/<model>"
    )
    parser.add_argument(
        "--host", default="http://localhost:11434", help="Ollama server, if not local"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="0 is greedy and reproducible; above 0 is needed for repeats to differ",
    )
    parser.add_argument("--seed", type=int, default=0, help="makes one repeat reproducible")
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
        help="generation budget; reasoning models need far more than chat models",
    )
    parser.add_argument("--out", required=True, type=Path, help="submission directory")
    parser.add_argument("--prompt", type=Path, default=PROMPT)
    args = parser.parse_args(argv)

    prompt = args.prompt.read_text(encoding="utf-8")
    from arena.cost_ledger import CostCeilingExceeded

    try:
        reply = generate(
            args.model, prompt, host=args.host,
            temperature=args.temperature, seed=args.seed,
            max_new_tokens=args.max_new_tokens,
        )
    except CostCeilingExceeded as exc:
        print(f"spending guard refused the call: {exc}")
        return EXIT_GUARD
    except RuntimeError as exc:
        if "is not set" in str(exc):
            print(str(exc))
            return EXIT_NO_KEY
        raise

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "raw_reply.md").write_text(reply, encoding="utf-8")
    if LAST_CALL:
        import json as _json

        (args.out / "generation.json").write_text(_json.dumps(LAST_CALL, indent=2), encoding="utf-8")
    try:
        source = extract_policy(reply)
    except ValueError as exc:
        print(f"no policy in reply: {exc}; raw reply kept at {args.out / 'raw_reply.md'}")
        return 1

    target = args.out / "policy.py"
    target.write_text(source, encoding="utf-8")
    print(f"wrote {target}")
    print(f"next: python -m arena.run eval --submission {target} --name {args.out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
