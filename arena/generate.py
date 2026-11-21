"""Ask a model for a policy and write it to a submission file.

Two backends, chosen by the ``--model`` prefix:

    ollama:<model>            a local Ollama server (no extra dependency)
    api:<provider>/<model>    an HTTP API (openai, anthropic, google, deepseek), key from the environment
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
DEFAULT_MAX_NEW_TOKENS = 8192

#: Exit codes of ``main``: 1 is a reply without a policy (the reply is kept); the others
#: mean no reply was obtained.
EXIT_NO_POLICY = 1
EXIT_GUARD = 3
EXIT_NO_KEY = 4

__all__ = ["extract_policy", "generate", "main", "EXIT_NO_POLICY", "EXIT_GUARD", "EXIT_NO_KEY"]

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


def _call_ollama(model: str, prompt: str, host: str, temperature: float, seed: int,
                 max_new_tokens: int) -> str:
    """One chat completion from a local Ollama server.

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
            "messages": [{"role": "user", "content": prompt}],
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
    return message


def _call_api(provider: str, model: str, prompt: str, temperature: float = 0.0,
              max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> str:
    """One chat completion from a provider's API; returns the reply text only.

    Before the call ``arena.cost_ledger.check_before_call`` refuses to proceed if the
    spending ceiling could be exceeded; after it the provider's reported token usage is
    recorded. The output budget counts thinking tokens, so a reasoning model that exhausts
    it returns no code, which is a recorded result.
    """
    from arena import cost_ledger, models

    if provider not in ("openai", "deepseek", "anthropic", "google"):
        raise ValueError(f"unknown provider {provider!r}; add it to _call_api")
    cost_ledger.check_before_call(model, prompt, max_new_tokens)

    if provider in ("openai", "deepseek"):
        from openai import OpenAI  # imported lazily; not an arena dependency

        if provider == "openai":
            client = OpenAI(api_key=models.key_for("openai"))
            limit = {"max_completion_tokens": max_new_tokens}
        else:
            # DeepSeek serves the OpenAI chat-completions protocol.
            client = OpenAI(api_key=models.key_for("deepseek"), base_url="https://api.deepseek.com")
            limit = {"max_tokens": max_new_tokens}
        reply = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=temperature, **limit,
        )
        usage = reply.usage
        cost_ledger.record(provider, model, usage.prompt_tokens, usage.completion_tokens)
        return reply.choices[0].message.content or ""

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=models.key_for("anthropic"))
        reply = client.messages.create(
            model=model,
            max_tokens=max_new_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        cost_ledger.record(provider, model, reply.usage.input_tokens, reply.usage.output_tokens)
        return "".join(b.text for b in reply.content if b.type == "text")

    if provider == "google":
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=models.key_for("google"))
        reply = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=temperature, max_output_tokens=max_new_tokens),
        )
        usage = reply.usage_metadata
        # Thinking tokens bill as output and are reported apart from the candidates' count.
        out_tokens = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        cost_ledger.record(provider, model, usage.prompt_token_count or 0, out_tokens)
        return reply.text or ""

    raise ValueError(f"unknown provider {provider!r}; add it to _call_api")


def _call_local(model_id: str, prompt: str, temperature: float, seed: int,
                max_new_tokens: int) -> str:
    """Greedy decode from a local transformers model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

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
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    inputs = tok(text, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0.0,
        **({"temperature": temperature, "top_p": 0.95} if temperature > 0.0 else {}),
    )
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


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
    kind, _, rest = model_spec.partition(":")
    if kind == "ollama":
        return _call_ollama(rest, prompt, host, temperature, seed, max_new_tokens)
    if kind == "api":
        provider, _, name = rest.partition("/")
        return _call_api(provider, name, prompt, temperature, max_new_tokens)
    if kind == "hf":
        return _call_local(rest, prompt, temperature, seed, max_new_tokens)
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
