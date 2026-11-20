"""Ask a model for a policy and write it to a submission file.

Two backends, chosen by the ``--model`` prefix:

    api:<provider>/<model>    an HTTP API, key from the environment
    hf:<org>/<model>          a local transformers model, e.g. on a GPU

Examples::

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

__all__ = ["extract_policy", "generate", "main"]

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


def _call_api(provider: str, model: str, prompt: str) -> str:
    """One chat completion. Providers are added here as needed."""
    if provider == "openai":
        from openai import OpenAI  # imported lazily; not an arena dependency

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        reply = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}], temperature=0
        )
        return reply.choices[0].message.content or ""

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        reply = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in reply.content if b.type == "text")

    raise ValueError(f"unknown provider {provider!r}; add it to _call_api")


def _call_local(model_id: str, prompt: str) -> str:
    """Greedy decode from a local transformers model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    inputs = tok(text, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def generate(model_spec: str, prompt: str) -> str:
    """Dispatch to the right backend and return the raw reply."""
    kind, _, rest = model_spec.partition(":")
    if kind == "api":
        provider, _, name = rest.partition("/")
        return _call_api(provider, name, prompt)
    if kind == "hf":
        return _call_local(rest, prompt)
    raise ValueError(f"--model must start with 'api:' or 'hf:', got {model_spec!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arena.generate")
    parser.add_argument("--model", required=True, help="api:<provider>/<model> or hf:<org>/<model>")
    parser.add_argument("--out", required=True, type=Path, help="submission directory")
    parser.add_argument("--prompt", type=Path, default=PROMPT)
    args = parser.parse_args(argv)

    prompt = args.prompt.read_text(encoding="utf-8")
    reply = generate(args.model, prompt)

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
