"""Arena CLI: verify, run, report.

    python -m arena.run verify
    python -m arena.run eval --submission arena/submissions/_template.py --name baseline
    python -m arena.run report
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from arena.config import (
    DEV_PROFILE,
    EVAL_PROFILES,
    RESULTS,
    SEEDS,
    SPEC_VERSION,
    arena_config,
)
from arena.reference import reference_table
from arena.sandbox import run_sandboxed
from arena.submission import SubmissionError

__all__ = ["evaluate_submission", "main"]


def evaluate_submission(path: str | Path, profile: str, name: str) -> dict:
    """Score one submission on one profile, in a sandboxed subprocess.

    Raises:
        SubmissionError: if the submission could not be evaluated. The message is
            the worker's, so a restriction violation says which import was refused.
    """
    result = run_sandboxed(path, profile, name)
    if "error" in result:
        raise SubmissionError(result["error"])
    return result


def _verify() -> int:
    """Check the environment, the reference values and determinism."""
    ok = True
    print(f"arena spec {SPEC_VERSION} | python {platform.python_version()}")

    for profile in (DEV_PROFILE, *EVAL_PROFILES):
        try:
            ref = reference_table(profile)
        except Exception as exc:  # noqa: BLE001 - verification reports, does not raise
            print(f"  FAIL  {profile}: {exc}")
            ok = False
            continue
        headroom = ref["j_star"] - ref["j_rule"]
        n_ctl = int((headroom > 0).sum())
        print(
            f"  ok    {profile}: {len(ref)} scenarios, "
            f"headroom mean {headroom.mean():.2f}, {n_ctl} with headroom > 0"
        )

    template = Path(__file__).resolve().parent / "submissions" / "_template.py"
    if template.is_file():
        try:
            a = evaluate_submission(template, DEV_PROFILE, "verify-a")
            b = evaluate_submission(template, DEV_PROFILE, "verify-b")
            same = abs(a["mean_g"] - b["mean_g"]) < 1e-12
            print(f"  {'ok   ' if same else 'FAIL '} determinism: template reproduces its score")
            ok = ok and same
        except Exception as exc:  # noqa: BLE001
            print(f"  FAIL  template: {exc}")
            ok = False

    print("VERIFY PASS" if ok else "VERIFY FAIL -- results from this checkout are not comparable")
    return 0 if ok else 1


def _eval(args: argparse.Namespace) -> int:
    name = args.name or Path(args.submission).parent.name
    result = {
        "name": name,
        "spec_version": SPEC_VERSION,
        "track": args.track,
        "submission": str(args.submission),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "dev": None,
        "eval": [],
        "error": None,
    }
    try:
        result["dev"] = evaluate_submission(args.submission, DEV_PROFILE, name)
        for profile in EVAL_PROFILES:
            result["eval"].append(evaluate_submission(args.submission, profile, name))
    except SubmissionError as exc:
        result["error"] = f"SubmissionError: {exc}"
    except Exception as exc:  # noqa: BLE001 - record and keep going; a crash is a result
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{name}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    if result["error"]:
        print(f"{name}: FAILED -- {result['error']}")
        print(f"written to {out}")
        return 1

    dev_g = result["dev"]["mean_g"]
    for block in result["eval"]:
        print(
            f"{name} [{block['profile']}]  G = {block['mean_g']:+.3f} "
            f"+/- {block['sem_g']:.3f}   "
            f"({block['n_controllable']} controllable, {block['n_excluded']} excluded, "
            f"{block['n_better_than_rule']} better than rule)   "
            f"violations {block['mean_violation_fraction']:.4f}   "
            f"gap to dev {block['mean_g'] - dev_g:+.3f}"
        )
        if block.get("degenerate"):
            print(
                f"  DEGENERATE: one action on all {block['n_decisions']} decisions "
                "-- this is a constant policy, not a controller"
            )
        elif block.get("modal_action_share", 0.0) > 0.95:
            print(f"  near-constant: modal action on {block['modal_action_share']:.1%} of decisions")
        if block["impossible_seeds"]:
            print(f"  WARNING G > 1 on seeds {block['impossible_seeds']} -- reference is wrong")
    print(f"written to {out}")
    return 0


def _display_names() -> dict[str, str]:
    """Result-name prefix -> report label for the frontier roster (``ARENA_LABEL_*``), when one is set."""
    from arena import models

    out = {}
    for slug in models.roster().values():
        label = models.display_label(slug)
        if label != slug:
            out[slug.replace("/", "-")] = label
    return out


def _block(result: dict, env: str | None) -> dict | None:
    """The evaluation block of ``result`` for environment ``env`` (the first block when ``env`` is None)."""
    if env is None:
        return result["eval"][0]
    for block in result["eval"]:
        if block["profile"] == env:
            return block
    return None


SINE_RESULT = "_sine-nonsense"


def _per_env_mean_g(results: list[dict]) -> dict[str, float]:
    """Environment name -> mean over the runnable results of each run's mean G in that environment."""
    out: dict[str, float] = {}
    for env in EVAL_PROFILES:
        gs = [b["mean_g"] for b in (_block(r, env) for r in results if not r.get("error") and r.get("eval"))
              if b is not None]
        if gs:
            out[env] = sum(gs) / len(gs)
    return out


def _report_headline() -> int:
    """One row per model: the median over the evaluation environments of its per-environment G.

    A model's G in an environment is the mean over its runnable repeats of each repeat's mean G there.
    ``beat_sine`` counts the environments in which that G exceeds the sine baseline's G
    (``arena/results/_sine-nonsense.json``). A model with no runnable repeat has no G.
    """
    import statistics
    from collections import defaultdict

    labels = _display_names()
    runs: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(RESULTS.glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        runs[r["name"].split("#")[0]].append(r)
    sine = _per_env_mean_g(runs.get(SINE_RESULT, []))
    n_env = len(EVAL_PROFILES)

    rows = []
    for model, results in runs.items():
        per_env = _per_env_mean_g(results)
        valid = sum(1 for r in results if not r.get("error"))
        beat = sum(1 for e, g in per_env.items() if e in sine and g > sine[e])
        rows.append({
            "model": labels.get(model, model),
            "valid": f"{valid}/{len(results)}",
            "G_median": round(statistics.median(per_env.values()), 2) if len(per_env) == n_env else None,
            "G_min": round(min(per_env.values()), 2) if per_env else None,
            "G_max": round(max(per_env.values()), 2) if per_env else None,
            "beat_sine": f"{beat}/{n_env}" if per_env and sine and model != SINE_RESULT else "-",
        })
    if not rows:
        print("no results yet")
        return 0
    frame = pd.DataFrame(rows).sort_values("G_median", ascending=False, na_position="last")
    print(f"headline: median over {n_env} environments of the per-environment G")
    print(frame.to_string(index=False))
    print()
    print("per-environment G = mean over runnable repeats; beat_sine = environments where it exceeds the sine baseline's G")
    return 0


def _report_by_model(env: str | None = None) -> int:
    """Aggregate repeats per model.

    ``env`` selects the evaluation block by environment name; ``None`` reads the
    first block, which is the original held-out profile.

    A result name of the form ``<model>#<k>`` is treated as repeat ``k`` of
    ``<model>``. With repeats the interesting quantities are not one score but
    three: how often the model produced runnable code at all, how good its runs
    were on average, and how much they varied. A model whose best run is strong
    and whose mean is poor has not solved the task, it has been lucky once.
    """
    import math
    from collections import defaultdict

    labels = _display_names()
    runs: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(RESULTS.glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        runs[r["name"].split("#")[0]].append(r)

    rows = []
    for model, results in sorted(runs.items()):
        ok = [r for r in results if not r.get("error")]
        blocks = [_block(r, env) for r in ok if r.get("eval")]
        blocks = [b for b in blocks if b is not None]
        gs = [b["mean_g"] for b in blocks]
        viols = [b["mean_violation_fraction"] for b in blocks]
        degen = sum(1 for b in blocks if b.get("degenerate"))
        breaches = sum(1 for v in viols if v > 0.05)
        rows.append(
            {
                "model": labels.get(model, model),
                "runs": len(results),
                "valid": len(gs),
                "G_mean": round(sum(gs) / len(gs), 2) if gs else None,
                "G_sd": round(
                    math.sqrt(sum((g - sum(gs) / len(gs)) ** 2 for g in gs) / (len(gs) - 1)), 2
                ) if len(gs) > 1 else None,
                "G_best": round(max(gs), 2) if gs else None,
                "G_worst": round(min(gs), 2) if gs else None,
                "viol_mean": round(sum(viols) / len(viols), 4) if viols else None,
                "over_budget": f"{breaches}/{len(gs)}" if gs else "-",
                "degenerate": f"{degen}/{len(gs)}" if gs else "-",
            }
        )
    if not rows:
        print("no results yet")
        return 0
    frame = pd.DataFrame(rows).sort_values("G_mean", ascending=False, na_position="last")
    print(frame.to_string(index=False))
    print()
    print("valid = produced runnable code; over_budget = violation fraction above 0.05")
    return 0


def _report() -> int:
    rows = []
    for path in sorted(RESULTS.glob("*.json")):
        r = json.loads(path.read_text(encoding="utf-8"))
        if r.get("error"):
            rows.append({"model": r["name"], "track": r.get("track"), "G": None,
                         "status": "error"})
            continue
        for block in r["eval"]:
            rows.append(
                {
                    "model": r["name"],
                    "track": r.get("track"),
                    "profile": block["profile"],
                    "G": round(block["mean_g"], 3),
                    "sem": round(block["sem_g"], 3),
                    "better": block["n_better_than_rule"],
                    "viol": round(block["mean_violation_fraction"], 4),
                    "hours": round(block["mean_completion_hours"], 2),
                    "dev_gap": round(block["mean_g"] - r["dev"]["mean_g"], 3),
                    "max_ms": round(block["max_step_ms"], 2),
                    "acts": block.get("distinct_actions"),
                    "modal": round(block.get("modal_action_share", 0.0), 2),
                    "status": "degenerate" if block.get("degenerate") else "ok",
                }
            )
    if not rows:
        print("no results yet")
        return 0
    frame = pd.DataFrame(rows).sort_values("G", ascending=False, na_position="last")
    print(frame.to_string(index=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m arena.run")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("verify", help="check references and determinism before running anything")

    ev = sub.add_parser("eval", help="score one submission")
    ev.add_argument("--submission", required=True, type=Path)
    ev.add_argument("--name", default=None, help="leaderboard name; defaults to the parent folder")
    ev.add_argument("--track", default="zero-shot", choices=["zero-shot", "iterative"])

    rep = sub.add_parser("report", help="print the leaderboard from arena/results/")
    rep.add_argument(
        "--by-model", action="store_true",
        help="aggregate repeats named <model>#<k> into one row per model",
    )
    rep.add_argument(
        "--env", default=None,
        help="with --by-model: environment name (a workload file, optionally @variant) for one table, or "
             "'all' for the headline table followed by one table per environment; default is the headline table",
    )

    args = parser.parse_args(argv)
    if args.cmd == "verify":
        return _verify()
    if args.cmd == "eval":
        return _eval(args)
    if not args.by_model:
        return _report()
    if args.env is None:
        return _report_headline()
    if args.env == "all":
        _report_headline()
        print()
        for env in EVAL_PROFILES:
            print(f"=== {env}")
            _report_by_model(env)
            print()
        return 0
    return _report_by_model(args.env)


if __name__ == "__main__":
    sys.exit(main())
