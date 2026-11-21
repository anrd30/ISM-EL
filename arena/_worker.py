"""Subprocess entry point: evaluate one submission under restrictions.

Reads a JSON request on stdin, writes a JSON response on stdout. Never raises to
the caller: a failing submission is a result, reported as ``{"error": ...}``.

Run by ``arena.sandbox``; not intended to be called directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Launched with -I (isolated), so PYTHONPATH is ignored by design and the repo must be
# put on the path explicitly. Done before any project import, and before restrictions.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _limit_resources(memory_mb: int, cpu_seconds: int) -> None:
    """Cap address space and CPU time. No-op on platforms without ``resource``."""
    try:
        import resource
    except ImportError:  # Windows
        return
    soft_mem = memory_mb * 1024 * 1024
    for which, limit in ((resource.RLIMIT_AS, soft_mem), (resource.RLIMIT_CPU, cpu_seconds)):
        try:
            _, hard = resource.getrlimit(which)
            resource.setrlimit(which, (limit, hard if hard != resource.RLIM_INFINITY else limit))
        except (ValueError, OSError):
            pass


def main() -> int:
    request = json.loads(sys.stdin.read())
    # Restore the parent's import path before any project import: -I dropped user
    # site-packages and PYTHONPATH, so numpy and pandas may be invisible without this.
    for entry in reversed(request.get("sys_path", [])):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    _limit_resources(request.get("memory_mb", 4096), request.get("cpu_seconds", 1800))

    response: dict[str, object]
    try:
        # Import everything the arena needs BEFORE restrictions go up: the allowlist
        # applies to the whole interpreter once active, and the harness legitimately
        # reads the workload CSV and the reference table.
        from arena.config import PROFILES, SEEDS, arena_config, split_env
        from arena.reference import reference_table
        from arena.restrict import install, preload_allowed, submission_code
        from arena.score import ScenarioScore, aggregate, score_scenario
        from arena.submission import SubmittedPolicy, load_policy_source
        from baselines.evaluate import evaluate

        profile = request["profile"]
        ref = reference_table(profile).set_index("seed")
        base_profile, variant = split_env(profile)
        config = arena_config(variant)
        workload = PROFILES / base_profile
        seeds = list(SEEDS)

        # Touch the workload once so the trace is in memory before the lockdown.
        from baselines.evaluate import _pinned_config

        _pinned_config(config, workload)

        # Read the source now: inside the restrictions there is no file access, and
        # the import machinery opens the file itself.
        source = Path(request["submission"]).read_text(encoding="utf-8")
        preload_allowed()
        install()

        with submission_code():
            fn = load_policy_source(source, request["name"])
        policy = SubmittedPolicy(fn, name=request["name"])
        # evaluate() itself opens and hashes the workload, so it runs unrestricted;
        # SubmittedPolicy raises the flag around each call into submission code.
        frame = evaluate(policy, profile=workload, seeds=seeds, config=config)

        scores: list[ScenarioScore] = []
        for _, row in frame.iterrows():
            seed = int(row["seed"])
            scores.append(
                score_scenario(
                    seed=seed,
                    j_policy=float(row["total_reward"]),
                    j_rule=float(ref.loc[seed, "j_rule"]),
                    j_star=float(ref.loc[seed, "j_star"]),
                    violation_fraction=float(row["latency_violation_fraction"]),
                    completion_hours=float(row["completion_hours"]),
                )
            )

        agg = aggregate(scores)
        response = {
            "profile": profile,
            **agg.as_dict(),
            "max_step_ms": policy.max_step_s * 1e3,
            "mean_step_ms": (policy.total_s / policy.n_calls * 1e3) if policy.n_calls else 0.0,
            "n_decisions": policy.n_calls,
            "action_counts": policy.action_counts,
            "distinct_actions": policy.distinct_actions,
            "modal_action_share": policy.modal_action_share,
            "degenerate": policy.distinct_actions <= 1,
            "per_scenario": [s.__dict__ for s in scores],
        }
    except BaseException as exc:  # noqa: BLE001 - any failure is a reportable result
        response = {"error": f"{type(exc).__name__}: {exc}"}

    sys.stdout.write(json.dumps(response))
    return 0


if __name__ == "__main__":
    sys.exit(main())
