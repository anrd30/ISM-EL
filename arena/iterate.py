"""Iterative track: let a model see how its policy scored and try again.

The zero-shot track asks once. This one runs a conversation. Each round the model
writes a policy, the policy is scored on the development workload, and the measurement
goes back as the next turn. With ``--memory full`` the whole exchange is carried
forward, so the model can see what it already tried and what that did, rather than
reacting to a number with no memory of how it got there.

    python -m arena.iterate --model hf:Qwen/Qwen2.5-Coder-7B-Instruct \
        --name qwen2.5-coder-7b --rounds 10 --repeats 5 --temperature 0.7

Three things make the result interpretable, and all three are deliberate:

Feedback comes only from ``DEV_PROFILE``. The eight evaluation environments are scored
every round and written to the record, but never shown to the model and never mentioned
in the conversation. Watching the held-out score while the model optimises the
development score is the point: if development keeps improving after held-out turns
over, the loop is fitting the feedback signal rather than learning control.

``--memory none`` is the ablation. Same budget, same feedback, but each round starts
from a fresh conversation holding only the previous policy and its score. The gap
between the two settings is what memory is worth; without it an improvement cannot be
told apart from simply re-rolling with a hint.

Cost is recorded in rounds, tokens and wall-clock. Tokens matter because the search
baseline in ``arena.search_baseline`` spends none at all, and a fair comparison of
``best score after k simulator evaluations`` has to say what else each method consumed.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from arena.config import DEV_PROFILE, EVAL_PROFILES, SPEC_VERSION
from arena.generate import LAST_USAGE, PROMPT, chat, extract_policy
from arena.submission import SubmissionError

__all__ = ["feedback_message", "run_repeat", "main"]

#: Where a run writes its rounds. One directory per model, one file per repeat.
ITERATIONS = Path(__file__).resolve().parent / "iterations"


def feedback_message(result: dict | None, error: str | None) -> str:
    """The turn handed back to the model after a round.

    Reports only what was measured on the development workload: the normalised score,
    the constraint it is held to, how long the rebuild took, and which actions it
    actually chose. No hint about what a better policy looks like, nothing from the
    held-out environments, and no mention of the reference policies beyond the score's
    own definition, which the task prompt already gives.
    """
    if error is not None:
        return (
            f"Your policy did not run. The harness reported:\n\n    {error}\n\n"
            "Write a corrected version of the whole function."
        )

    counts = result.get("action_counts") or []
    total = sum(counts) or 1
    shares = ", ".join(f"{i}: {c / total:.1%}" for i, c in enumerate(counts))
    lines = [
        "Your policy ran on the development workload. Measurements:",
        "",
        f"  score G                 {result['mean_g']:+.3f}  "
        f"(1.0 is optimal, 0.0 matches the reference controller)",
        f"  latency violations      {result['mean_violation_fraction']:.4f} of intervals",
        f"  mean rebuild time       {result['mean_completion_hours']:.1f} hours",
        f"  action usage            {shares}",
        f"  slowest decision        {result['max_step_ms']:.2f} ms of the 10 ms budget",
    ]
    if result.get("degenerate"):
        lines.append("  note: you returned the same action on every decision.")
    lines += ["", "Write your next version of the whole function."]
    return "\n".join(lines)


def _score(path: Path, profile: str, name: str) -> tuple[dict | None, str | None]:
    """Score one policy file on one environment. Returns (result, error)."""
    from arena.run import evaluate_submission

    try:
        return evaluate_submission(path, profile, name), None
    except SubmissionError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 - a crash is a recorded round, not a stop
        return None, f"{type(exc).__name__}: {exc}"


def run_repeat(
    model_spec: str,
    name: str,
    repeat: int,
    rounds: int,
    out_dir: Path,
    *,
    memory: str = "full",
    temperature: float = 0.7,
    max_new_tokens: int = 8192,
    host: str = "http://localhost:11434",
    holdout_every: int = 1,
) -> dict:
    """One conversation of ``rounds`` rounds. Returns the record written to disk."""
    task = PROMPT.read_text(encoding="utf-8")
    history: list[dict] = [{"role": "user", "content": task}]
    record: dict = {
        "name": name,
        "model": model_spec,
        "repeat": repeat,
        "spec_version": SPEC_VERSION,
        "track": "iterative",
        "memory": memory,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "dev_profile": DEV_PROFILE,
        "python": platform.python_version(),
        "started": datetime.now(timezone.utc).isoformat(),
        "rounds": [],
    }
    round_dir = out_dir / f"repeat{repeat}"
    round_dir.mkdir(parents=True, exist_ok=True)

    best_dev = None
    for k in range(1, rounds + 1):
        t0 = time.time()
        LAST_USAGE.clear()
        try:
            reply = chat(
                model_spec, history, host=host, temperature=temperature,
                seed=repeat * 1000 + k, max_new_tokens=max_new_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - a dead backend ends the repeat, recorded
            record["rounds"].append({"round": k, "fatal": f"{type(exc).__name__}: {exc}"})
            break
        gen_s = time.time() - t0
        usage = dict(LAST_USAGE)
        (round_dir / f"round{k}_reply.md").write_text(reply, encoding="utf-8")

        row: dict = {
            "round": k,
            "generation_seconds": round(gen_s, 2),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }

        try:
            source = extract_policy(reply)
        except ValueError as exc:
            row.update({"dev": None, "error": f"no policy in reply: {exc}"})
            record["rounds"].append(row)
            history += [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": feedback_message(None, row["error"])},
            ]
            if memory == "none":
                history = [{"role": "user", "content": task}, history[-1]]
            print(f"  round {k}: no policy in reply", flush=True)
            continue

        policy_path = round_dir / f"round{k}_policy.py"
        policy_path.write_text(source, encoding="utf-8")

        t1 = time.time()
        dev, dev_err = _score(policy_path, DEV_PROFILE, f"{name}#{repeat}r{k}")
        row["dev"] = dev
        row["error"] = dev_err
        row["eval_seconds"] = round(time.time() - t1, 2)

        if dev is not None and (k % holdout_every == 0 or k == rounds):
            # Scored every round and recorded, never shown to the model: this is the
            # generalisation curve, and showing it would make it a training signal.
            row["eval"] = []
            for profile in EVAL_PROFILES:
                block, err = _score(policy_path, profile, f"{name}#{repeat}r{k}")
                row["eval"].append({"profile": profile, "result": block, "error": err})

        record["rounds"].append(row)
        if dev is not None:
            best_dev = dev["mean_g"] if best_dev is None else max(best_dev, dev["mean_g"])
            print(
                f"  round {k}: dev G {dev['mean_g']:+.3f} (best {best_dev:+.3f}) "
                f"{usage.get('completion_tokens', '?')} out-tokens, {gen_s:.0f}s",
                flush=True,
            )
        else:
            print(f"  round {k}: failed -- {dev_err}", flush=True)

        history += [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": feedback_message(dev, dev_err)},
        ]
        if memory == "none":
            # Same feedback, no recollection of how it got here.
            history = [{"role": "user", "content": task}, history[-1]]

    record["finished"] = datetime.now(timezone.utc).isoformat()
    out = out_dir / f"repeat{repeat}.json"
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"  wrote {out}", flush=True)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arena.iterate")
    parser.add_argument("--model", required=True,
                        help="ollama:<model>, api:<provider>/<model> or hf:<org>/<model>")
    parser.add_argument("--name", default=None, help="leaderboard name; defaults to the model id")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--memory", choices=["full", "none"], default="full",
                        help="'full' carries the conversation; 'none' is the no-memory ablation")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=8192,
                        help="per round; the conversation grows, so this is not the zero-shot budget")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--holdout-every", type=int, default=1,
                        help="score the 8 held-out environments every N rounds (always on the last)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    name = args.name or args.model.partition(":")[2].replace("/", "-")
    out_dir = args.out or (ITERATIONS / f"{name}-{args.memory}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"arena spec {SPEC_VERSION} | iterative | {name} | memory={args.memory} | "
          f"{args.repeats} repeats x {args.rounds} rounds", flush=True)

    for repeat in range(1, args.repeats + 1):
        print(f"--- repeat {repeat}/{args.repeats} ---", flush=True)
        run_repeat(
            args.model, name, repeat, args.rounds, out_dir,
            memory=args.memory, temperature=args.temperature,
            max_new_tokens=args.max_new_tokens, host=args.host,
            holdout_every=args.holdout_every,
        )
    print(f"done. records in {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
