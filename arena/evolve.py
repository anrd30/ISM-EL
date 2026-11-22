"""Population search over policies, with the mutation operator as the variable.

The conversational track in ``arena.iterate`` asks one model to keep revising its own
work. It turned out to perform exactly as well as blind sampling, which is a result
about *that* loop, not about search. The published benchmarks in this area
(CO-Bench, HeuriGym) do not use a chat loop; they run agentic search for tens of
iterations. This module matches that setting, and isolates the only part that is in
question.

The loop is the same for every arm: keep a pool of the best candidates so far, draw a
parent, produce a child from it, score the child on the development workload, insert.
What changes between arms is **how the child is produced**:

    llm       ask a language model to improve the parent, given its measurements
    mutate    perturb one numeric constant in the parent's own source
    sample    draw a fresh policy from the simple family (constants, one threshold)

``llm`` is the treatment. ``mutate`` is the matched control that nobody in this
literature runs: identical pool, identical selection pressure, identical evaluation
budget, with the model replaced by a random number. If ``llm`` does not beat
``mutate``, the model contributed nothing to the search and that is measurable rather
than arguable. ``sample`` is the memoryless floor -- search with no inheritance at all.

Budget is counted in **scored candidates**, because evaluation is the resource that
supplies information. Tokens and wall-clock are logged separately; the non-``llm`` arms
spend none.

Every candidate is written to disk with its parent, its operator, its source, its score
and its cost, so any number in a paper can be traced back to the exact program that
produced it::

    python -m arena.evolve --operator llm --model api:openrouter/deepseek/deepseek-v4.1-flash-1017 \
        --budget 64 --name deepseek-flash
    python -m arena.evolve --operator mutate --budget 64 --name control-mutate
    python -m arena.evolve --operator sample --budget 64 --name control-sample
"""

from __future__ import annotations

import argparse
import ast
import json
import platform
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from arena.config import DEV_PROFILE, EVAL_PROFILES, SPEC_VERSION
from arena.generate import LAST_USAGE, PROMPT, chat, extract_policy
from arena.submission import SubmissionError

__all__ = ["Candidate", "mutate_constant", "sample_family_source", "evolve", "main"]

RUNS = Path(__file__).resolve().parent / "evolve_runs"

#: Candidates kept as breeding stock. Small enough to stay selective, large enough that
#: the search does not collapse onto one lineage in the first few rounds.
POOL_SIZE = 8


@dataclass
class Candidate:
    """One scored program, with everything needed to reproduce and audit it."""

    id: int
    round: int
    operator: str
    parent: int | None
    parent_dev_g: float | None
    dev_g: float | None
    error: str | None
    source_sha: str
    source_path: str
    prompt_tokens: int | None
    completion_tokens: int | None
    generate_seconds: float
    evaluate_seconds: float
    eval: list = field(default_factory=list)


#: Observation width, so an index mutation stays in bounds.
OBS_DIM = 12


def _classify(tree: ast.AST) -> list[tuple[ast.Constant, str]]:
    """Every mutable numeric literal, tagged by the role it plays.

    Role decides how it may move, which is what keeps the control competent instead of
    merely destructive. An untyped perturbation spends most of its budget turning
    ``obs[11]`` into an out-of-bounds index or editing a branch that never executes.
    """
    indices = {
        id(node.slice)
        for node in ast.walk(tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "obs"
    }
    out: list[tuple[ast.Constant, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant):
            continue
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            continue
        if node.end_lineno != node.lineno:  # offsets only usable on one line
            continue
        if id(node) in indices:
            out.append((node, "index"))
        elif isinstance(node.value, float):
            out.append((node, "threshold"))
        else:
            out.append((node, "action"))
    return out


def mutate_constant(source: str, rng: random.Random) -> str:
    """Perturb one numeric literal in a policy's source, respecting what it means.

    The matched control for the language model: it edits the program where a thoughtful
    change would go, by a random amount rather than a reasoned one. Thresholds are
    preferred two times out of three when any exist, because on this task the threshold
    is the parameter that carries the behaviour -- an exhaustive sweep of the
    single-threshold family reaches G = -0.377, so a control that rarely touches a
    threshold is not a serious baseline and beating it would prove nothing.

    Thresholds move by a factor in [0.5, 2], clamped to [0, 1] since every observation
    is normalised. Actions step within [0, 3]. Indices step within the observation
    width, so a mutation never crashes on an out-of-bounds read.

    Raises:
        ValueError: if the source holds no numeric literal to perturb.
    """
    tree = ast.parse(source)
    spots = _classify(tree)
    if not spots:
        raise ValueError("no numeric literal to mutate")

    thresholds = [s for s in spots if s[1] == "threshold"]
    if thresholds and rng.random() < 2 / 3:
        node, role = rng.choice(thresholds)
    else:
        node, role = rng.choice(spots)

    old = node.value
    if role == "threshold":
        new: object = round(min(1.0, max(0.0, float(old) * rng.uniform(0.5, 2.0))), 6)
    elif role == "index":
        new = max(0, min(OBS_DIM - 1, int(old) + rng.choice((-1, 1))))
    else:
        new = max(0, min(3, int(old) + rng.choice((-1, 1))))

    lines = source.splitlines(keepends=True)
    line = lines[node.lineno - 1]
    lines[node.lineno - 1] = line[: node.col_offset] + repr(new) + line[node.end_col_offset :]
    return "".join(lines)


def sample_family_source(rng: random.Random) -> str:
    """A fresh member of the simple family: a constant, or one threshold on one field.

    The same family ``arena.search_baseline`` sweeps, emitted as source so it travels
    through the identical sandbox and scorer as a model's submission.
    """
    if rng.random() < 0.1:
        return (
            "import numpy as np\n\n\n"
            f"def policy(obs: np.ndarray, state: dict) -> int:\n    return {rng.randrange(4)}\n"
        )
    field_index = rng.choice((2, 11))
    threshold = round(rng.uniform(0.05, 0.95), 2)
    high, low = rng.sample(range(4), 2)
    return (
        "import numpy as np\n\n\n"
        "def policy(obs: np.ndarray, state: dict) -> int:\n"
        f"    return {high} if obs[{field_index}] > {threshold} else {low}\n"
    )


IMPROVE_PROMPT = """\
Here is a control policy that was measured on the development workload.

```python
{source}
```

Measurements:

{measurements}

Write an improved version of this policy. Reply with the complete function in a single
Python code block, following the same rules as before.
"""


def _measurements(dev: dict | None, error: str | None) -> str:
    if error is not None:
        return f"It did not run. The harness reported:\n\n    {error}"
    counts = dev.get("action_counts") or []
    total = sum(counts) or 1
    shares = ", ".join(f"{i}: {c / total:.1%}" for i, c in enumerate(counts))
    out = [
        f"  score G             {dev['mean_g']:+.3f}  (1.0 is optimal, 0.0 matches the reference)",
        f"  latency violations  {dev['mean_violation_fraction']:.4f} of intervals",
        f"  mean rebuild time   {dev['mean_completion_hours']:.1f} hours",
        f"  action usage        {shares}",
    ]
    if dev.get("degenerate"):
        out.append("  note: one action on every decision.")
    return "\n".join(out)


def _score(path: Path, profile: str, name: str) -> tuple[dict | None, str | None]:
    from arena.run import evaluate_submission

    try:
        return evaluate_submission(path, profile, name), None
    except SubmissionError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 - a crash is a recorded candidate
        return None, f"{type(exc).__name__}: {exc}"


def evolve(
    operator: str,
    name: str,
    budget: int,
    out_dir: Path,
    *,
    model_spec: str | None = None,
    seed: int = 0,
    temperature: float = 0.7,
    max_new_tokens: int = 32768,
    host: str = "http://localhost:11434",
    holdout_every: int = 8,
    pool_size: int = POOL_SIZE,
    n_seeds: int = 4,
) -> dict:
    """Run ``budget`` scored candidates and return the run record.

    The first ``n_seeds`` candidates are drawn independently, before any breeding. One
    seed is not enough: a lineage that starts inside a flat region of the objective
    makes no progress, because every mutation of its threshold leaves the branch on the
    same side and scores identically. A population starts the search in several places
    at once, and it is also what the search frameworks this arm is matched against do.
    """
    import hashlib

    rng = random.Random(seed)
    task = PROMPT.read_text(encoding="utf-8")
    sources = out_dir / "candidates"
    sources.mkdir(parents=True, exist_ok=True)

    record: dict = {
        "name": name,
        "operator": operator,
        "model": model_spec,
        "spec_version": SPEC_VERSION,
        "track": "evolve",
        "budget": budget,
        "pool_size": pool_size,
        "seed": seed,
        "temperature": temperature,
        "max_new_tokens": max_new_tokens,
        "dev_profile": DEV_PROFILE,
        "python": platform.python_version(),
        "started": datetime.now(timezone.utc).isoformat(),
        "candidates": [],
    }

    pool: list[Candidate] = []
    best: Candidate | None = None

    for i in range(1, budget + 1):
        # Seeding phase: independent draws, no parent, for every arm alike.
        parent = rng.choice(pool) if (pool and i > n_seeds) else None
        t0 = time.time()
        LAST_USAGE.clear()
        error_pre: str | None = None

        if operator == "sample" or (operator == "mutate" and parent is None):
            source = sample_family_source(rng)
        elif operator == "mutate":
            try:
                source = mutate_constant(Path(parent.source_path).read_text(encoding="utf-8"), rng)
            except (ValueError, SyntaxError):
                source = sample_family_source(rng)
        elif operator == "llm":
            if model_spec is None:
                raise ValueError("--model is required for --operator llm")
            if parent is None:
                content = task
            else:
                # One user turn, not two: some providers reject consecutive user
                # messages, and the task is context for the edit rather than a
                # separate request.
                content = task + "\n\n---\n\n" + IMPROVE_PROMPT.format(
                    source=Path(parent.source_path).read_text(encoding="utf-8"),
                    measurements=_measurements(parent.dev_full, parent.error),
                )
            messages = [{"role": "user", "content": content}]
            try:
                reply = chat(
                    model_spec, messages, host=host, temperature=temperature,
                    seed=seed * 10000 + i, max_new_tokens=max_new_tokens,
                )
                source = extract_policy(reply)
                (sources / f"cand{i:04d}_reply.md").write_text(reply, encoding="utf-8")
            except Exception as exc:  # noqa: BLE001 - recorded, the search continues
                error_pre = f"{type(exc).__name__}: {exc}"
                source = ""
        else:
            raise ValueError(f"unknown operator {operator!r}")

        gen_s = time.time() - t0
        usage = dict(LAST_USAGE)

        path = sources / f"cand{i:04d}.py"
        path.write_text(source, encoding="utf-8")
        sha = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

        t1 = time.time()
        if error_pre is not None:
            dev, err = None, error_pre
        else:
            dev, err = _score(path, DEV_PROFILE, f"{name}-c{i}")
        eval_s = time.time() - t1

        cand = Candidate(
            id=i,
            round=i,
            operator=operator,
            parent=parent.id if parent else None,
            parent_dev_g=parent.dev_g if parent else None,
            dev_g=dev["mean_g"] if dev else None,
            error=err,
            source_sha=sha,
            source_path=str(path),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            generate_seconds=round(gen_s, 2),
            evaluate_seconds=round(eval_s, 2),
        )
        cand.dev_full = dev  # type: ignore[attr-defined] - kept in memory for the next prompt

        if dev is not None:
            if best is None or dev["mean_g"] > (best.dev_g or -float("inf")):
                best = cand
            pool.append(cand)
            pool.sort(key=lambda c: c.dev_g or -float("inf"), reverse=True)
            del pool[pool_size:]

        if dev is not None and (i % holdout_every == 0 or i == budget):
            # Recorded, never shown to the model: the generalisation curve.
            for profile in EVAL_PROFILES:
                block, berr = _score(path, profile, f"{name}-c{i}")
                cand.eval.append({"profile": profile, "result": block, "error": berr})

        row = {k: v for k, v in asdict(cand).items()}
        record["candidates"].append(row)
        (out_dir / "candidates.jsonl").open("a", encoding="utf-8").write(json.dumps(row) + "\n")

        print(
            f"  [{i}/{budget}] {operator}"
            + (f" parent={parent.id}" if parent else " seed")
            + (f" dev {dev['mean_g']:+.3f}" if dev else f" FAILED {err}")
            + (f"  best {best.dev_g:+.3f}" if best and best.dev_g is not None else "")
            + (f"  {usage.get('completion_tokens')}tok" if usage else "")
            + f"  {gen_s:.0f}s/{eval_s:.0f}s",
            flush=True,
        )

    record["finished"] = datetime.now(timezone.utc).isoformat()
    record["best_id"] = best.id if best else None
    record["best_dev_g"] = best.dev_g if best else None
    (out_dir / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"  wrote {out_dir / 'run.json'}", flush=True)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arena.evolve")
    parser.add_argument("--operator", required=True, choices=["llm", "mutate", "sample"])
    parser.add_argument("--model", default=None, help="required for --operator llm")
    parser.add_argument("--name", required=True)
    parser.add_argument("--budget", type=int, default=64, help="scored candidates")
    parser.add_argument("--pool-size", type=int, default=POOL_SIZE)
    parser.add_argument("--seeds", type=int, default=4,
                        help="independent draws before breeding starts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--holdout-every", type=int, default=8)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    out_dir = args.out or (RUNS / f"{args.name}-{args.operator}-s{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"arena spec {SPEC_VERSION} | evolve | {args.name} | operator={args.operator} | "
          f"budget={args.budget} | pool={args.pool_size} | seeds={args.seeds} | seed={args.seed}", flush=True)
    evolve(
        args.operator, args.name, args.budget, out_dir,
        model_spec=args.model, seed=args.seed, temperature=args.temperature,
        max_new_tokens=args.max_new_tokens, host=args.host,
        holdout_every=args.holdout_every, pool_size=args.pool_size,
        n_seeds=args.seeds,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
