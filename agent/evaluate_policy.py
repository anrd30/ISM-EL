"""Evaluate trained checkpoints against the baselines on a held-out workload.

``baselines.evaluate`` compares the fixed rates, the throttle, the risk-aware
rule and the oracle. It has no way to add a learned policy, because a
checkpoint is the agent lane's artifact. This module is that entry point: it
wraps each checkpoint in :class:`agent.policy_adapter.LearnedPolicy` and hands
it to ``baselines.evaluate.compare`` alongside the baselines, so every policy
sees the identical paired scenarios.

What makes a result reportable. Three things have to line up, and the command
line makes each of them explicit rather than implicit:

- ``--workload-csv`` points at a window the policy did not train on. The four
  generator profiles are single 24-hour windows tiled to fill an episode, so
  the next interval's load sits exactly 288 steps back in the agent's own
  history; evaluating on one measures memorisation of the period.
- ``--eval-fleet`` draws drives from the held-out-split feature table, matching
  what ``baselines.evaluate`` does under the same flag, so the fleet is not the
  one the hazard model was fitted on.
- ``--hazard-error-sigma`` is recorded in the configuration label and so in
  every row of ``episodes.csv``.

A checkpoint's own ``meta`` carries the workload and fleet it trained against.
Those are printed next to the evaluation setting and, with ``--strict-holdout``
(the default), a checkpoint whose training workload is the evaluation workload
is refused: the point of this script is the held-out number, and quietly
producing the in-sample one instead is the mistake it exists to prevent.

Usage::

    python -m agent.evaluate_policy \
        --checkpoint runs/ppo-clean/seed0/ckpt_final.pt \
        --checkpoint runs/ppo-clean/seed1/ckpt_final.pt \
        --checkpoint runs/ppo-clean/seed2/ckpt_final.pt \
        --workload-csv data_pipeline/profiles/profile_diurnal_long_holdout.csv \
        --eval-fleet --hazard-error-sigma 0.25 --seeds 20 \
        --out runs/ppo-clean-eval
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from agent.checkpoint import load_checkpoint
from agent.policy_adapter import LearnedPolicy
from baselines.evaluate import (
    _pinned_config,
    _workload_section,
    check_pairing,
    compare,
    paired_delta,
    summarise,
    to_markdown,
)
from baselines.policies import Policy, baseline_policies, diagnostic_policies
from contracts.contracts import ContractViolation
from sim.config import SimConfig, v1_eval_config

__all__ = ["build_config", "evaluate_checkpoints", "main"]

#: The baseline a learned policy has to beat to be worth deploying. The oracle
#: is an upper reference, not a comparator.
REFERENCE_POLICY = "risk_aware_rule_based"


def build_config(
    *, eval_fleet: bool, hazard_error_sigma: float, latency_filter: bool = False, latency_filter_causal: bool = False
) -> SimConfig:
    """Simulator configuration, labelled with the fleet split and sigma.

    Mirrors ``baselines.evaluate._build_config`` so that a run of this script
    and a run of that one on the same flags produce the same ``config.label``
    and so comparable rows.
    """
    fields: dict[str, Any] = {"hazard_error_sigma": hazard_error_sigma}
    if eval_fleet or hazard_error_sigma != 0.0 or latency_filter or latency_filter_causal:
        fleet = "eval-fleet" if eval_fleet else "train-fleet"
        fields["label"] = (
            f"{fleet},sigma={hazard_error_sigma:g}"
            + (",latency-filter" if latency_filter else "")
            + (",latency-filter-causal" if latency_filter_causal else "")
        )
    if latency_filter:
        fields["latency_filter"] = True
    if latency_filter_causal:
        fields["latency_filter_causal"] = True
    return v1_eval_config(**fields) if eval_fleet else SimConfig(**fields)


def _checkpoint_provenance(path: Path) -> dict[str, Any]:
    """The training setting a checkpoint records, for the report's header."""
    meta = load_checkpoint(path)["meta"]
    return {
        "checkpoint": str(path),
        "seed": meta.get("seed"),
        "iterations": meta.get("iteration"),
        # ``workload_csv`` wins when set: ``profile`` keeps its default in that
        # case and would otherwise misreport what the policy trained on.
        "trained_on": meta.get("workload_csv") or meta.get("profile"),
        "eval_fleet": meta.get("eval_fleet"),
        "hazard_error_sigma": meta.get("hazard_error_sigma"),
        "ablate_hazard": meta.get("ablate_hazard"),
    }


def _assert_held_out(provenance: Sequence[dict[str, Any]], workload: Path) -> None:
    """Refuse a checkpoint that trained on the workload it is being evaluated on.

    Raises:
        ValueError: naming every offending checkpoint.
    """
    target = workload.resolve()
    offenders = [
        row["checkpoint"]
        for row in provenance
        if row["trained_on"] and Path(str(row["trained_on"])).resolve() == target
    ]
    if offenders:
        raise ValueError(
            "these checkpoints trained on the evaluation workload, so the result would be "
            f"in-sample rather than held out: {offenders}. Point --workload-csv at the other "
            "half of the split, or pass --no-strict-holdout if an in-sample number is what "
            "you actually want."
        )


def evaluate_checkpoints(
    checkpoints: Sequence[Path],
    *,
    workload: Path,
    config: SimConfig,
    seeds: Sequence[int],
    diagnostics: bool = False,
    stochastic: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """Run every baseline and every checkpoint on the identical paired scenarios.

    Returns:
        The per-episode frame and the names of the learned policies, in the
        order the checkpoints were given.

    Raises:
        ContractViolation: if ``workload`` is not a valid replay file.
        RuntimeError: if the policies did not all see the same scenarios.
    """
    policies: list[Policy] = baseline_policies(config)
    if diagnostics:
        policies += diagnostic_policies(config)

    learned_names: list[str] = []
    for path in checkpoints:
        seed = load_checkpoint(path)["meta"].get("seed")
        name = f"ppo_lagrangian seed{seed}" if seed is not None else f"ppo_lagrangian {path.parent.name}"
        policies.append(
            LearnedPolicy.from_checkpoint(path, name=name, deterministic=not stochastic)
        )
        learned_names.append(name)

    episodes = compare(policies, profiles=[workload], seeds=list(seeds), config=config)
    check_pairing(episodes)
    return episodes, learned_names


def _provenance_section(provenance: Sequence[dict[str, Any]], config: SimConfig) -> str:
    """What each checkpoint trained on, against what it is now being evaluated on."""
    lines = [
        "## Checkpoints",
        "",
        f"Evaluated under configuration `{config.label}`.",
        "",
        to_markdown(pd.DataFrame(list(provenance))),
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; returns the process exit status."""
    parser = argparse.ArgumentParser(
        prog="python -m agent.evaluate_policy",
        description="Paired evaluation of trained checkpoints against the baselines.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=[],
        required=True,
        help="a checkpoint written by agent.train (repeatable; one policy per checkpoint)",
    )
    parser.add_argument(
        "--workload-csv",
        type=Path,
        required=True,
        help="replay CSV to evaluate on; should be a window no checkpoint trained against",
    )
    parser.add_argument("--seeds", type=int, default=20, help="number of paired scenario seeds")
    parser.add_argument("--seed-start", type=int, default=0, help="first seed")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--diagnostics", action="store_true", help="also run the diagnostic policies")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="sample from the policy instead of taking its mode",
    )
    parser.add_argument(
        "--hazard-error-sigma",
        type=float,
        default=0.0,
        help="log-space sigma of the multiplicative error between reported and true per-drive hazard",
    )
    parser.add_argument(
        "--eval-fleet",
        action="store_true",
        help="draw the simulated fleet from the held-out-split feature table instead of the train-split table",
    )
    parser.add_argument(
        "--latency-filter",
        action="store_true",
        help="apply the latency ceiling (sim/latency_filter.py) after the safety filter, from the interval's own load; off by default",
    )
    parser.add_argument(
        "--latency-filter-causal",
        action="store_true",
        help="apply the latency ceiling from the previous interval's load; off by default; exclusive with --latency-filter",
    )
    parser.add_argument(
        "--no-strict-holdout",
        dest="strict_holdout",
        action="store_false",
        help="permit evaluating a checkpoint on the workload it trained on (the result is in-sample)",
    )
    parser.set_defaults(strict_holdout=True)
    args = parser.parse_args(argv)

    if args.seeds < 1:
        parser.error("--seeds must be >= 1")
    if args.latency_filter and args.latency_filter_causal:
        parser.error("--latency-filter and --latency-filter-causal are mutually exclusive")
    if not (math.isfinite(args.hazard_error_sigma) and args.hazard_error_sigma >= 0.0):
        parser.error(f"--hazard-error-sigma must be a finite number >= 0, got {args.hazard_error_sigma!r}")
    for path in args.checkpoint:
        if not path.is_file():
            parser.error(f"--checkpoint {path} does not exist")
    try:
        _pinned_config(None, args.workload_csv)
    except (FileNotFoundError, ContractViolation) as exc:
        parser.error(f"{args.workload_csv}: {exc}")

    provenance = [_checkpoint_provenance(p) for p in args.checkpoint]
    if args.strict_holdout:
        try:
            _assert_held_out(provenance, args.workload_csv)
        except ValueError as exc:
            parser.error(str(exc))

    config = build_config(
        eval_fleet=args.eval_fleet,
        hazard_error_sigma=args.hazard_error_sigma,
        latency_filter=args.latency_filter,
        latency_filter_causal=args.latency_filter_causal,
    )
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    try:
        episodes, learned = evaluate_checkpoints(
            args.checkpoint,
            workload=args.workload_csv,
            config=config,
            seeds=seeds,
            diagnostics=args.diagnostics,
            stochastic=args.stochastic,
        )
    except ValueError as exc:
        parser.error(str(exc))

    summary = summarise(episodes)
    args.out.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(args.out / "episodes.csv", index=False)

    parts = [
        "# Trained policy against the baselines (simulated)",
        "",
        f"Simulator configuration `{config.label}`; latency model uncalibrated "
        "(sim/LIMITATIONS.md). "
        f"{len(seeds)} paired scenario seeds ({seeds[0]}..{seeds[-1]}), identical for every policy. "
        f"Learned policies act by {'sampling' if args.stochastic else 'their mode (argmax)'}. "
        "`completion_h` is the mean over completed episodes only.",
        "",
        _provenance_section(provenance, config),
        "",
        _workload_section(episodes),
        "",
        to_markdown(summary),
    ]
    names = set(episodes["policy"])
    for name in learned:
        if REFERENCE_POLICY in names:
            parts += [
                "",
                f"## Paired difference: {name} minus {REFERENCE_POLICY}",
                "",
                to_markdown(paired_delta(episodes, name, REFERENCE_POLICY)),
            ]
    text = "\n".join(parts) + "\n"
    (args.out / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
