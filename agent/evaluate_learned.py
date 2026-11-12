"""Paired evaluation of learned controllers against the baselines.

Usage::

    python -m agent.evaluate_learned \\
        --search-result runs/search/seasonal/seed0/search.json runs/search/rule/seed0/search.json \\
        --workload-csv data_pipeline/profiles/profile_diurnal_long_holdout.csv \\
        --seeds 20 --eval-fleet --hazard-error-sigma 0.25 --diagnostics --out runs/eval_search

``--ppo-checkpoint`` takes ``.pt`` files from ``agent.forecast_ppo``
(``ckpt_best.pt`` is labelled ``forecast_ppo_<init>``, ``ckpt_init.pt`` of a
``bc`` run ``forecast_bc_only``); ``--search-result`` takes ``search.json``
files from ``agent.search_seasonal``; ``--checkpoint`` takes ``.pt`` files
from ``agent.train_residual``. Any combination may be given.

Every policy runs on the identical scenarios through
``baselines.evaluate.compare``, which checks the pairing. Writes
``episodes.csv`` and ``summary.md`` and prints the summary.

Guards. A checkpoint whose training workload has the same SHA-256 as the
evaluation workload is refused unless ``--no-strict-holdout`` is given. A
checkpoint trained with a different risk model or ``hazard_error_sigma`` from
the evaluation configuration is refused.

Strata. The scenario set is reported whole and split two ways:

- ``initial_budget_infeasible``: the reset-time reliability flag, computed
  before any action. Fixed in advance of any result from this harness.
- reward magnitude: scenarios on which ``risk_aware_rule_based`` scores below
  ``EXTREME_REWARD`` are ``extreme``. This reproduces the split used in
  ``agent/PPO_COLLAPSE_DECISION.md``, which was chosen after seeing results;
  it is kept only so the two records can be compared.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import pandas as pd

from agent.seasonal_rule import SeasonalForecastRule
from agent.forecast_ppo import ForecastPPOPolicy
from agent.residual_policy import ResidualPolicy, load_residual_checkpoint
from agent.search_seasonal import make_policy
from agent.train_residual import build_sim_config
from baselines.evaluate import compare, paired_delta, summarise, to_markdown
from baselines.policies import RiskAwareRuleBased, baseline_policies

__all__ = ["EXTREME_REWARD", "REFERENCE", "strata_table", "main"]

#: Rule reward below which a scenario counts as extreme in the magnitude split.
EXTREME_REWARD: float = -500.0
REFERENCE: str = "risk_aware_rule_based"


def _gap_row(df: pd.DataFrame, treatment: str, stratum: str, seeds: Sequence[int]) -> dict[str, object]:
    a = df[df["policy"] == treatment].set_index("seed").loc[list(seeds), "total_reward"].astype(float)
    b = df[df["policy"] == REFERENCE].set_index("seed").loc[list(seeds), "total_reward"].astype(float)
    diff = (a - b).to_numpy()
    n = len(diff)
    return {
        "policy": treatment,
        "stratum": stratum,
        "n": n,
        "gap": float(diff.mean()) if n else float("nan"),
        "sem": float(diff.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        "n_better": int((diff > 0).sum()),
        "n_worse": int((diff < 0).sum()),
    }


def strata_table(df: pd.DataFrame, treatments: Sequence[str]) -> pd.DataFrame:
    """Paired total-reward gap against the rule, whole set and per stratum."""
    ref = df[df["policy"] == REFERENCE].set_index("seed")
    seeds = list(ref.index)
    infeasible = [s for s in seeds if bool(ref.loc[s, "initial_budget_infeasible"])]
    feasible = [s for s in seeds if s not in infeasible]
    extreme = [s for s in seeds if float(ref.loc[s, "total_reward"]) < EXTREME_REWARD]
    typical = [s for s in seeds if s not in extreme]
    rows = []
    for name in treatments:
        for label, subset in (
            ("all", seeds),
            ("initial_budget_infeasible=False (pre-registered)", feasible),
            ("initial_budget_infeasible=True (pre-registered)", infeasible),
            (f"typical, rule reward >= {EXTREME_REWARD:g} (post hoc)", typical),
            (f"extreme, rule reward < {EXTREME_REWARD:g} (post hoc)", extreme),
        ):
            if subset:
                rows.append(_gap_row(df, name, label, subset))
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_forms(path: Path) -> frozenset[str]:
    """SHA-256 of a file's bytes as stored, with LF line endings, and with CRLF line endings.

    A recorded hash is matched against all three, so the same text content
    compares equal whichever line-ending convention the recorded file or the
    checked-out file carries. Two files that differ in anything but line
    endings share none of the three.
    """
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    return frozenset(hashlib.sha256(variant).hexdigest() for variant in (raw, lf, lf.replace(b"\n", b"\r\n")))


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    p = argparse.ArgumentParser(prog="python -m agent.evaluate_learned", description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", type=Path, nargs="*", default=[])
    p.add_argument("--search-result", type=Path, nargs="*", default=[])
    p.add_argument("--ppo-checkpoint", type=Path, nargs="*", default=[])
    p.add_argument("--workload-csv", type=Path, required=True)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--hazard", choices=["v1", "v0"], default="v1")
    p.add_argument("--eval-fleet", action="store_true", help="held-out fleet table (v1 only)")
    p.add_argument("--hazard-error-sigma", type=float, default=0.25)
    p.add_argument("--diagnostics", action="store_true", help="add the hand-built seasonal-forecast rule")
    p.add_argument("--no-strict-holdout", action="store_true")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    config = build_sim_config(args.hazard, args.eval_fleet, args.hazard_error_sigma)
    eval_sha = _sha256(args.workload_csv)
    eval_forms = _sha256_forms(args.workload_csv)
    residuals: list[ResidualPolicy] = []
    ckpt_rows = []
    for path in args.checkpoint:
        meta = load_residual_checkpoint(path)["meta"]
        if meta["trained_on_sha256"] in eval_forms and not args.no_strict_holdout:
            p.error(f"{path} was trained on the evaluation workload (sha256 {eval_sha[:12]}); see --no-strict-holdout")
        if meta["hazard"] != args.hazard:
            p.error(f"{path} was trained with hazard={meta['hazard']}, evaluation uses {args.hazard}")
        if float(meta["hazard_error_sigma"]) != args.hazard_error_sigma:
            p.error(f"{path} was trained with sigma={meta['hazard_error_sigma']}, evaluation uses {args.hazard_error_sigma}")
        policy = ResidualPolicy.from_checkpoint(path, config)
        residuals.append(policy)
        ckpt_rows.append(
            {
                "checkpoint": path.as_posix(),
                "policy": policy.name,
                "iteration": meta["iteration"],
                "trained_on": meta["trained_on"],
                "hazard": meta["hazard"],
                "train_fleet": "eval-fleet" if meta["eval_fleet"] else "train-fleet",
                "sigma": meta["hazard_error_sigma"],
                "val_gap": round(float(meta["val_gap"]), 3),
                "val_gap_sem": round(float(meta["val_gap_sem"]), 3),
                "val_seeds": meta["val_seeds"],
            }
        )
    searched = []
    for path in args.search_result:
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta["trained_on_sha256"] in eval_forms and not args.no_strict_holdout:
            p.error(f"{path} was searched on the evaluation workload (sha256 {eval_sha[:12]}); see --no-strict-holdout")
        if meta["hazard"] != args.hazard:
            p.error(f"{path} was searched with hazard={meta['hazard']}, evaluation uses {args.hazard}")
        if float(meta["hazard_error_sigma"]) != args.hazard_error_sigma:
            p.error(f"{path} was searched with sigma={meta['hazard_error_sigma']}, evaluation uses {args.hazard_error_sigma}")
        label = {"seasonal": "seasonal_search", "rule": "rule_progress_search"}[meta["base"]]
        policy = make_policy(meta["base"], meta["params"], config, f"{label} seed{int(meta['seed'])}")
        searched.append(policy)
        ckpt_rows.append(
            {
                "checkpoint": path.as_posix(),
                "policy": policy.name,
                "iteration": meta["generation"],
                "trained_on": meta["trained_on"],
                "hazard": meta["hazard"],
                "train_fleet": "eval-fleet" if meta["eval_fleet"] else "train-fleet",
                "sigma": meta["hazard_error_sigma"],
                "val_gap": round(float(meta["val_gap"]), 3),
                "val_gap_sem": round(float(meta["val_gap_sem"]), 3),
                "val_seeds": meta["val_seeds"],
                "params": ", ".join(f"{k}={v:.3f}" for k, v in meta["params"].items()),
            }
        )
    ppo_policies = []
    for path in args.ppo_checkpoint:
        meta = torch.load(path, map_location="cpu", weights_only=True)["meta"]
        if meta["trained_on_sha256"] in eval_forms and not args.no_strict_holdout:
            p.error(f"{path} was trained on the evaluation workload (sha256 {eval_sha[:12]}); see --no-strict-holdout")
        for split in ("fit_csv_sha256", "val_csv_sha256"):
            if meta[split] in eval_forms and not args.no_strict_holdout:
                p.error(f"{path} used the evaluation workload as its {split[:3]} split (sha256 {eval_sha[:12]}); see --no-strict-holdout")
        if meta["hazard"] != args.hazard:
            p.error(f"{path} was trained with hazard={meta['hazard']}, evaluation uses {args.hazard}")
        if float(meta["hazard_error_sigma"]) != args.hazard_error_sigma:
            p.error(f"{path} was trained with sigma={meta['hazard_error_sigma']}, evaluation uses {args.hazard_error_sigma}")
        if path.name == "ckpt_init.pt":
            if meta["init"] != "bc":
                p.error(f"{path} is an untrained network")
            label = f"forecast_bc_only seed{int(meta['seed'])}"
        else:
            label = f"forecast_ppo_{meta['init']} seed{int(meta['seed'])}"
        policy = ForecastPPOPolicy.from_checkpoint(path, config, label)
        ppo_policies.append(policy)
        ckpt_rows.append(
            {
                "checkpoint": path.as_posix(),
                "policy": label,
                "iteration": meta["iteration"],
                "trained_on": meta["trained_on"],
                "hazard": meta["hazard"],
                "train_fleet": "train-fleet",
                "sigma": meta["hazard_error_sigma"],
                "val_gap": round(float(meta["val_gap"]), 3),
                "val_gap_sem": round(float(meta["val_gap_sem"]), 3),
                "val_seeds": meta["val_seeds"],
                "params": f"bc_accuracy={float(meta['bc_accuracy']):.3f}" if meta["init"] == "bc" else "",
            }
        )
    if not residuals and not searched and not ppo_policies:
        p.error("give at least one --ppo-checkpoint, --checkpoint or --search-result")
    names = [r.name for r in residuals] + [q.name for q in searched] + [q.name for q in ppo_policies]
    if len(set(names)) != len(names):
        p.error(f"duplicate policy names {names}; checkpoints must come from different seeds")

    policies = list(baseline_policies(config))
    if args.diagnostics:
        policies.append(SeasonalForecastRule(config))
    policies += residuals + searched + ppo_policies
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    episodes = compare(policies, profiles=[args.workload_csv], seeds=seeds, config=config)
    summary = summarise(episodes)

    treatments = [n for n in (["rule_seasonal_forecast (hand-built)"] if args.diagnostics else []) + names]
    strata = strata_table(episodes, treatments)
    changed = pd.DataFrame(
        [{"policy": r.name, "steps": r.n_steps, "frac_changed_from_rule": r.n_changed / max(1, r.n_steps)} for r in residuals]
    )

    parts = [
        "# Learned controllers against the baselines (simulated)",
        "",
        f"Simulator configuration `{config.label}`; latency model uncalibrated (sim/LIMITATIONS.md). "
        f"Workload `{args.workload_csv.as_posix()}` (sha256 `{eval_sha}`). "
        f"{len(seeds)} paired scenario seeds ({seeds[0]}..{seeds[-1]}), identical for every policy. "
        "Learned policies act deterministically. Reward is the contract reward with contract coefficients.",
        "",
        "## Checkpoints",
        "",
        to_markdown(pd.DataFrame(ckpt_rows)),
        "",
        "## Summary",
        "",
        to_markdown(summary),
        "",
        f"## Paired total-reward gap against {REFERENCE}, by stratum",
        "",
        "`gap` is treatment minus rule, mean over paired scenarios; `sem` its standard error; "
        "`n_better` / `n_worse` count scenarios. Strata are defined in the module docstring.",
        "",
        to_markdown(strata),
        "",
    ]
    if residuals:
        parts += ["", "## Share of intervals on which the residual policy changed the rule's action", "",
                  to_markdown(changed)]
    pairs = [
        (q.name, q.name.replace("seasonal_search", "rule_progress_search"))
        for q in searched
        if q.name.startswith("seasonal_search") and q.name.replace("seasonal_search", "rule_progress_search") in names
    ]
    ppo_pairs = [
        (q.name, q.name.replace("forecast_ppo_bc", "forecast_bc_only"))
        for q in ppo_policies
        if q.name.startswith("forecast_ppo_bc") and q.name.replace("forecast_ppo_bc", "forecast_bc_only") in names
    ]
    if ppo_pairs:
        rows = []
        for treatment, comparator in ppo_pairs:
            d = paired_delta(episodes, treatment, comparator, metrics=["total_reward"]).iloc[0]
            rows.append({"treatment": treatment, "comparator": comparator, "n": int(d["n"]),
                         "gap": float(d["mean_diff"]), "sem": float(d["sem"])})
        parts += [
            "",
            "## PPO-trained policy against its own imitation-only starting point",
            "",
            "Same network before and after PPO, so this gap is what PPO training added.",
            "",
            to_markdown(pd.DataFrame(rows)),
        ]
    if pairs:
        rows = []
        for treatment, comparator in pairs:
            for stratum, frame in (("all", episodes),):
                d = paired_delta(frame, treatment, comparator, metrics=["total_reward"]).iloc[0]
                rows.append({"treatment": treatment, "comparator": comparator, "stratum": stratum, "n": int(d["n"]),
                             "gap": float(d["mean_diff"]), "sem": float(d["sem"])})
        parts += [
            "",
            "## Seasonal search against the progress-price search of the same seed",
            "",
            "The comparator retunes only the rule's progress price on the same training days, so this gap is "
            "what the forecast adds beyond retuning the baseline.",
            "",
            to_markdown(pd.DataFrame(rows)),
        ]
    for name in treatments:
        parts += ["", f"## Paired difference: {name} minus {REFERENCE}", "", to_markdown(paired_delta(episodes, name, REFERENCE))]
    text = "\n".join(parts) + "\n"
    args.out.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(args.out / "episodes.csv", index=False)
    strata.to_csv(args.out / "strata.csv", index=False)
    (args.out / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
