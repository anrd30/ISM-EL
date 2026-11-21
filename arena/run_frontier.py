"""Frontier batch: generate and evaluate every roster model, five repeats each.

Reads keys and model ids from ``arena/.env`` (via ``arena.models``) before anything else.
Preconditions, each checked before the first call: every provider key is set, the worst-case
cost estimate has been printed and is under the ceiling, and ``arena.run verify`` passes.

For each model and each seed in ``SEEDS``::

    python -m arena.generate --model api:<provider>/<id> --out arena/submissions/<id>#<seed> \\
        --seed <seed> --temperature 0.7
    python -m arena.run eval --submission arena/submissions/<id>#<seed>/policy.py --name <id>#<seed>

A reply without ``def policy`` is recorded as a result with an error (0 valid), its
``raw_reply.md`` kept, and is never retried. A repeat whose ``raw_reply.md`` already exists is
skipped, so a rerun cannot spend twice on the same repeat. Any other generation failure (an
unknown model id, an API error, the spending guard) stops the batch.

Usage::

    python -m arena.run_frontier
"""

from __future__ import annotations

from arena import models  # loads arena/.env before anything else runs

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from arena import cost_ledger
from arena.config import RESULTS, SPEC_VERSION, SUBMISSIONS
from arena.generate import EXIT_NO_POLICY

SEEDS = (1, 2, 3, 4, 5)
TEMPERATURE = 0.7


def _run(args: list[str]) -> int:
    return subprocess.run([sys.executable, "-m", *args], check=False).returncode


def _record_no_policy(name: str, out: Path) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    result = {
        "name": name, "spec_version": SPEC_VERSION, "track": "zero-shot",
        "submission": str(out / "policy.py"), "timestamp": datetime.now(timezone.utc).isoformat(),
        "dev": None, "eval": [], "error": f"no policy: reply kept at {out / 'raw_reply.md'}",
    }
    (RESULTS / f"{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def missing_keys() -> list[str]:
    """Messages for every provider of the roster whose key is not set."""
    out: list[str] = []
    for var in models.roster():
        try:
            models.key_for(models.provider_of(var))
        except RuntimeError as exc:
            if str(exc) not in out:
                out.append(str(exc))
    return out


def main() -> int:
    roster = models.roster()
    missing = missing_keys()
    if missing:
        print("\n".join(missing))
        return 2
    if cost_ledger._estimate_cli() != 0:
        return 2
    if _run(["arena.run", "verify"]) != 0:
        print("verify failed")
        return 2

    for var, model in roster.items():
        provider = models.provider_of(var)
        for seed in SEEDS:
            name = f"{model}#{seed}"
            out = SUBMISSIONS / name
            if (out / "raw_reply.md").exists():
                print(f"skip {name}: reply already kept")
                continue
            code = _run(["arena.generate", "--model", f"api:{provider}/{model}", "--out", str(out),
                         "--seed", str(seed), "--temperature", str(TEMPERATURE)])
            if code == EXIT_NO_POLICY and (out / "raw_reply.md").exists():
                _record_no_policy(name, out)
                continue
            if code != 0:
                print(f"stop: generation of {name} exited {code}")
                return code
            _run(["arena.run", "eval", "--submission", str(out / "policy.py"), "--name", name])
    _run(["arena.cost_ledger", "report"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
