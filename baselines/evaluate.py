"""Paired evaluation harness for rebuild-bandwidth policies.

Paired design. Every policy is run on the identical set of scenarios: for a
given ``(profile, seed)`` the environment draws the same workload trace, trace
start offset and surviving-drive fleet whatever the policy does, because the
scenario comes from a stream seeded by ``reset(seed=...)`` that the policy
never touches, and the failure stream draws a fixed number of values per step.
Differences between policies are then differences caused by the policy rather
than by scenario luck. This matters here because the events that separate
policies are rare (a second drive failing during a rebuild has a probability of
order 1e-3 per episode), so an unpaired comparison would be dominated by noise.
``compare`` checks the pairing on every run by comparing a fingerprint of each
scenario across policies and raises if they differ.

Metric sources. Every per-episode quantity is taken from the ``info`` mapping
the environment returns, which the contract defines as the authoritative
record, and from the step reward. Nothing is re-derived from the observation
or from the reward. The one quantity ``info`` does not carry is the latency
degradation ``D``; it is computed from ``info["latency_now"]`` with the
contract's own ``latency_degradation`` and the configured latency target, the
same call the environment makes.

Everything measured here comes from the simulator in ``sim/``: its constants
are design decisions, its latency model is uncalibrated, and its workloads are
synthetic (``sim/LIMITATIONS.md``). Constraint satisfaction is read from the
``info`` flags, never inferred from reward.

Command line::

    python -m baselines.evaluate --profiles all --seeds 20 --out <dir>
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baselines.policies import (
    NeedsEnv,
    Policy,
    baseline_policies,
    diagnostic_policies,
)
from contracts.contracts import (
    BW_TOTAL_MBPS,
    CONTROL_INTERVAL_S,
    LATENCY_VIOLATION_FRACTION_MAX,
    MAX_EPISODE_STEPS,
    TerminationReason,
    latency_degradation,
)
from sim.config import SimConfig
from sim.recovery_env import RecoveryEnv
from workload.generate import PROFILE_NAMES

__all__ = [
    "EPISODE_COLUMNS",
    "evaluate",
    "compare",
    "summarise",
    "paired_delta",
    "to_markdown",
    "check_pairing",
    "main",
]

EPISODE_COLUMNS: tuple[str, ...] = (
    "profile",
    "policy",
    "seed",
    "termination_reason",
    "completed",
    "steps",
    "completion_hours",
    "latency_degradation_mean",
    "latency_degradation_p95",
    "latency_violation_fraction",
    "latency_budget_met",
    "p_failure_mean",
    "p_failure_max",
    "risk_budget_violated",
    "clip_fraction",
    "mean_requested_mbps",
    "mean_delivered_mbps",
    "stall_fraction",
    "p_failure_delivered_mean",
    "p_failure_delivered_max",
    "total_reward",
    "fleet_level",
    "initial_fg_bw_mbps",
    "initial_budget_infeasible",
)

#: Columns that fingerprint a scenario and so must agree across policies.
_SCENARIO_COLUMNS: tuple[str, ...] = ("fleet_level", "initial_fg_bw_mbps", "initial_budget_infeasible")

#: Metrics reported by ``paired_delta`` by default.
DEFAULT_DELTA_METRICS: tuple[str, ...] = (
    "total_reward",
    "latency_degradation_mean",
    "latency_violation_fraction",
    "completion_hours",
    "mean_delivered_mbps",
    "p_failure_mean",
    "clip_fraction",
)


def _check_policy_config(policy: Policy, config: SimConfig) -> None:
    """Reject a policy whose latency model differs from the environment's."""
    policy_config = getattr(policy, "config", None)
    if policy_config is None:
        return
    for attr in ("latency_unloaded_ms", "latency_target_ms", "rho_max"):
        if getattr(policy_config, attr) != getattr(config, attr):
            raise ValueError(
                f"policy {policy.name!r} was built with {attr}={getattr(policy_config, attr)!r} "
                f"but the environment uses {getattr(config, attr)!r}"
            )


def _run_episode(env: RecoveryEnv, policy: Policy, profile: str, seed: int, config: SimConfig) -> dict[str, Any]:
    """Run one episode and reduce its per-step records to one row."""
    obs, reset_info = env.reset(seed=seed)
    policy.reset()
    initial_fg = float(obs[2]) * BW_TOTAL_MBPS  # field 2 (fg_bw) is normalised by BW_TOTAL_MBPS

    latency_now: list[float] = []
    violated_latency: list[bool] = []
    p_failure: list[float] = []
    p_delivered: list[float] = []
    risk_violated: list[bool] = []
    clipped: list[bool] = []
    requested: list[float] = []
    delivered: list[float] = []
    total_reward = 0.0
    final_info: dict[str, Any] = reset_info

    for _ in range(MAX_EPISODE_STEPS + 1):
        action = int(policy(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        latency_now.append(float(info["latency_now"]))
        violated_latency.append(bool(info["violated_latency"]))
        p_failure.append(float(info["p_failure_before_complete"]))
        p_delivered.append(float(info["p_failure_at_delivered_rate"]))
        risk_violated.append(bool(info["violated_risk_budget"]))
        clipped.append(bool(info["alloc_clipped_by_filter"]))
        requested.append(float(info["alloc_requested_mbps"]))
        delivered.append(float(info["alloc_effective_mbps"]))
        final_info = info
        if terminated or truncated:
            break
    else:  # pragma: no cover - the environment truncates at MAX_EPISODE_STEPS
        raise RuntimeError("episode did not end within MAX_EPISODE_STEPS + 1 steps")

    degradation = np.array(
        [latency_degradation(x, config.latency_target_ms) for x in latency_now], dtype=np.float64
    )
    steps = len(latency_now)
    reason = str(final_info["termination_reason"])
    completed = reason == TerminationReason.RECOVERY_COMPLETE.value
    violation_fraction = float(np.mean(violated_latency))
    return {
        "profile": profile,
        "policy": policy.name,
        "seed": int(seed),
        "termination_reason": reason,
        "completed": completed,
        "steps": steps,
        "completion_hours": steps * CONTROL_INTERVAL_S / 3600.0 if completed else float("nan"),
        "latency_degradation_mean": float(degradation.mean()),
        "latency_degradation_p95": float(np.percentile(degradation, 95)),
        "latency_violation_fraction": violation_fraction,
        "latency_budget_met": violation_fraction <= LATENCY_VIOLATION_FRACTION_MAX,
        "p_failure_mean": float(np.mean(p_failure)),
        "p_failure_max": float(np.max(p_failure)),
        "risk_budget_violated": bool(any(risk_violated)),
        "clip_fraction": float(np.mean(clipped)),
        "mean_requested_mbps": float(np.mean(requested)),
        "mean_delivered_mbps": float(np.mean(delivered)),
        "stall_fraction": float(np.mean(np.asarray(delivered) == 0.0)),
        "p_failure_delivered_mean": float(np.mean(p_delivered)),
        "p_failure_delivered_max": float(np.max(p_delivered)),
        "total_reward": total_reward,
        "fleet_level": float(final_info["fleet_degradation_level"]),
        "initial_fg_bw_mbps": initial_fg,
        "initial_budget_infeasible": bool(reset_info["violated_risk_budget"]),
    }


def _pinned_config(config: SimConfig | None, profile: str) -> SimConfig:
    """Return ``config`` with the workload pinned to ``profile``."""
    if profile not in PROFILE_NAMES:
        raise ValueError(f"profile must be one of {PROFILE_NAMES}, got {profile!r}")
    base = config if config is not None else SimConfig()
    if base.workload_csv is not None:
        raise ValueError("config.workload_csv is set; a named profile cannot be pinned on top of it")
    return dataclasses.replace(base, workload_profile=profile)


def _check_seeds(seeds: Sequence[int]) -> list[int]:
    out = [int(s) for s in seeds]
    if not out:
        raise ValueError("seeds is empty")
    if len(set(out)) != len(out):
        raise ValueError("seeds contains duplicates")
    return out


def evaluate(
    policy: Policy,
    *,
    profile: str,
    seeds: Sequence[int],
    config: SimConfig | None = None,
) -> pd.DataFrame:
    """Run ``policy`` for one episode per seed on one workload profile.

    Args:
        policy: Any object satisfying :class:`baselines.policies.Policy`. A
            policy that also satisfies ``NeedsEnv`` (the oracle) is bound to
            the environment first.
        profile: One of ``workload.generate.PROFILE_NAMES``.
        seeds: Scenario seeds, one episode each. Pass the same seeds for every
            policy being compared.
        config: Simulator configuration; the workload is pinned to ``profile``
            on a copy. Defaults to ``SimConfig()``.

    Returns:
        One row per seed with the columns of ``EPISODE_COLUMNS``.

    Raises:
        ValueError: on an unknown profile, empty or duplicate seeds, a config
            with ``workload_csv`` set, or a policy whose latency model differs
            from the environment's.
    """
    cfg = _pinned_config(config, profile)
    seed_list = _check_seeds(seeds)
    _check_policy_config(policy, cfg)
    env = RecoveryEnv(cfg)
    if isinstance(policy, NeedsEnv):
        policy.bind(env)
    rows = [_run_episode(env, policy, profile, s, cfg) for s in seed_list]
    return pd.DataFrame(rows, columns=list(EPISODE_COLUMNS))


def check_pairing(df: pd.DataFrame) -> None:
    """Raise unless every policy saw the same scenario for each ``(profile, seed)``.

    Compares the scenario fingerprint columns (fleet degradation level, first
    foreground bandwidth, whether the budget is infeasible at the start) across
    policies. It also requires every policy to have the same set of seeds in
    every profile.

    Raises:
        RuntimeError: on any mismatch.
    """
    for profile, block in df.groupby("profile", sort=False):
        seed_sets = {name: frozenset(g["seed"]) for name, g in block.groupby("policy", sort=False)}
        reference = next(iter(seed_sets.values()))
        for name, seeds in seed_sets.items():
            if seeds != reference:
                raise RuntimeError(f"policy {name!r} was evaluated on a different seed set for profile {profile!r}")
        for column in _SCENARIO_COLUMNS:
            spread = block.groupby("seed")[column].nunique(dropna=False)
            if (spread > 1).any():
                bad = int(spread[spread > 1].index[0])
                raise RuntimeError(
                    f"scenario column {column!r} differs across policies for profile {profile!r}, seed {bad}"
                )


def compare(
    policies: Sequence[Policy],
    *,
    profiles: Sequence[str],
    seeds: Sequence[int],
    config: SimConfig | None = None,
) -> pd.DataFrame:
    """Evaluate every policy on every profile with identical seeds.

    Returns the concatenated per-episode rows (``EPISODE_COLUMNS``), ordered by
    profile, then policy in the order given, then seed. The pairing is verified
    with :func:`check_pairing`.

    Raises:
        ValueError: on empty ``policies`` or ``profiles``, or duplicate policy
            names; see :func:`evaluate` for the rest.
    """
    if not policies:
        raise ValueError("policies is empty")
    if not profiles:
        raise ValueError("profiles is empty")
    names = [p.name for p in policies]
    if len(set(names)) != len(names):
        raise ValueError(f"policy names must be unique, got {names}")
    seed_list = _check_seeds(seeds)
    frames = [
        evaluate(policy, profile=profile, seeds=seed_list, config=config)
        for profile in profiles
        for policy in policies
    ]
    out = pd.concat(frames, ignore_index=True)
    check_pairing(out)
    return out


def _first_appearance_order(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per-policy, per-profile means over episodes.

    Rows are ordered by profile then policy, both in order of first appearance
    in ``df``. Column meanings:

    - ``n``: episodes. ``completed``, ``horizon_expired``, ``terminal_failure``:
      episode counts per termination reason (each reported separately).
    - ``completion_h``: mean completion time over completed episodes only.
    - ``lat_deg_mean``, ``lat_deg_p95``: mean over episodes of the per-episode
      mean and 95th percentile of ``latency_degradation``.
    - ``viol_frac``: mean fraction of intervals with ``violated_latency``.
    - ``lat_budget_met``: fraction of episodes with ``viol_frac <= 0.05``.
    - ``p_fail_mean``, ``p_fail_max``: mean over episodes of the per-episode
      mean and maximum ``p_failure_before_complete``.
    - ``risk_viol``: fraction of episodes in which ``violated_risk_budget``
      fired at least once.
    - ``clip_frac``: mean fraction of steps with ``alloc_clipped_by_filter``.
    - ``deliv_mbps``: mean delivered allocation (``alloc_effective_mbps``),
      which differs from the request when the filter raises it or the
      foreground load leaves less headroom than the request.
    - ``stall_frac``: mean fraction of intervals in which the delivered rate
      was zero (foreground load at the utilisation ceiling).

    Exposure at the delivered rate is in the episode file
    (``p_failure_delivered_*``) but not summarised: the filter defines it as 1
    for a zero rate, so one stalled interval sets an episode's maximum to 1.
    - ``reward``, ``reward_sem``: mean total reward and its standard error.
    """
    profiles = _first_appearance_order(df["profile"])
    policies = _first_appearance_order(df["policy"])
    rows: list[dict[str, Any]] = []
    for profile in profiles:
        for policy in policies:
            g = df[(df["profile"] == profile) & (df["policy"] == policy)]
            if g.empty:
                continue
            n = len(g)
            reward = g["total_reward"].to_numpy(dtype=np.float64)
            rows.append(
                {
                    "profile": profile,
                    "policy": policy,
                    "n": n,
                    "completed": int((g["termination_reason"] == TerminationReason.RECOVERY_COMPLETE.value).sum()),
                    "horizon_expired": int((g["termination_reason"] == TerminationReason.HORIZON_EXPIRED.value).sum()),
                    "terminal_failure": int((g["termination_reason"] == TerminationReason.TERMINAL_FAILURE.value).sum()),
                    "completion_h": float(g["completion_hours"].mean()) if g["completed"].any() else float("nan"),
                    "lat_deg_mean": float(g["latency_degradation_mean"].mean()),
                    "lat_deg_p95": float(g["latency_degradation_p95"].mean()),
                    "viol_frac": float(g["latency_violation_fraction"].mean()),
                    "lat_budget_met": float(g["latency_budget_met"].mean()),
                    "p_fail_mean": float(g["p_failure_mean"].mean()),
                    "p_fail_max": float(g["p_failure_max"].mean()),
                    "risk_viol": float(g["risk_budget_violated"].mean()),
                    "clip_frac": float(g["clip_fraction"].mean()),
                    "deliv_mbps": float(g["mean_delivered_mbps"].mean()),
                    "stall_frac": float(g["stall_fraction"].mean()),
                    "reward": float(reward.mean()),
                    "reward_sem": float(reward.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def paired_delta(
    df: pd.DataFrame,
    treatment: str,
    reference: str,
    *,
    metrics: Sequence[str] = DEFAULT_DELTA_METRICS,
) -> pd.DataFrame:
    """Paired difference ``treatment - reference`` per profile, matched by seed.

    Returns one row per profile and metric with the mean difference, its
    standard error over the paired seeds, and the number of pairs. Episodes
    where either side's metric is NaN (``completion_hours`` of an episode that
    did not complete) are dropped from that metric's pairs, and ``n`` counts
    the pairs that remain.

    Raises:
        ValueError: if either policy is absent from ``df``.
    """
    present = set(df["policy"])
    for name in (treatment, reference):
        if name not in present:
            raise ValueError(f"policy {name!r} is not in the data; have {sorted(present)}")
    rows: list[dict[str, Any]] = []
    for profile in _first_appearance_order(df["profile"]):
        block = df[df["profile"] == profile]
        a = block[block["policy"] == treatment].set_index("seed")
        b = block[block["policy"] == reference].set_index("seed")
        common = a.index.intersection(b.index)
        for metric in metrics:
            diff = (a.loc[common, metric].astype(float) - b.loc[common, metric].astype(float)).dropna()
            n = len(diff)
            rows.append(
                {
                    "profile": profile,
                    "metric": metric,
                    "treatment": treatment,
                    "reference": reference,
                    "mean_diff": float(diff.mean()) if n else float("nan"),
                    "sem": float(diff.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
                    "n": n,
                }
            )
    return pd.DataFrame(rows)


def _format_cell(value: Any, floatfmt: str) -> str:
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return "n/a" if math.isnan(value) else format(float(value), floatfmt)
    return str(value)


def to_markdown(df: pd.DataFrame, *, floatfmt: str = ".4g") -> str:
    """Render a frame as a GitHub-flavoured markdown table (no index).

    NaN is written as ``n/a``. A policy whose name contains ``ORACLE`` is
    followed by a note, because an oracle is an upper reference and not a
    comparator.
    """
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[c], floatfmt) for c in columns) + " |")
    text = "\n".join(lines)
    if "policy" in df.columns and df["policy"].astype(str).str.contains("ORACLE").any():
        text += "\n\nORACLE rows read the next interval's foreground load from the trace; no deployable controller can."
    return text


def _resolve_profiles(arg: str) -> list[str]:
    if arg == "all":
        return list(PROFILE_NAMES)
    names = [p.strip() for p in arg.split(",") if p.strip()]
    for name in names:
        if name not in PROFILE_NAMES:
            raise argparse.ArgumentTypeError(f"unknown profile {name!r}; choose from {PROFILE_NAMES} or 'all'")
    return names


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; returns the process exit status."""
    parser = argparse.ArgumentParser(
        prog="python -m baselines.evaluate",
        description="Paired evaluation of the baseline policies on the recovery simulator.",
    )
    parser.add_argument("--profiles", default="all", help="'all' or a comma-separated list of profile names")
    parser.add_argument("--seeds", type=int, default=20, help="number of paired scenario seeds (0 .. N-1 from --seed-start)")
    parser.add_argument("--seed-start", type=int, default=0, help="first seed")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--diagnostics", action="store_true", help="also run the diagnostic policies")
    args = parser.parse_args(argv)
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")
    try:
        profiles = _resolve_profiles(args.profiles)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    config = SimConfig()
    policies: list[Policy] = baseline_policies(config)
    if args.diagnostics:
        policies += diagnostic_policies(config)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    episodes = compare(policies, profiles=profiles, seeds=seeds, config=config)
    summary = summarise(episodes)

    args.out.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(args.out / "episodes.csv", index=False)

    parts = [
        "# Baseline comparison (simulated)",
        "",
        f"Simulator configuration `{config.label}`; synthetic workloads; latency model uncalibrated "
        "(sim/LIMITATIONS.md). "
        f"{len(seeds)} paired scenario seeds ({seeds[0]}..{seeds[-1]}), identical for every policy. "
        "`completion_h` is the mean over completed episodes only. "
        "`lat_budget_met` and `risk_viol` are fractions of episodes; `viol_frac`, `clip_frac` and "
        "`stall_frac` are fractions of intervals. Latency terms are the mean-latency model of the "
        "simulator; `p_fail_*` are the filter's exposure for the applied allocation (nominal rate).",
        "",
        to_markdown(summary),
    ]
    names = {p.name for p in policies}
    pairs = [
        ("risk_aware_rule_based", "workload_only_throttle"),
        ("ORACLE_lookahead (not deployable)", "risk_aware_rule_based"),
        ("risk_aware_rule_based", "slowest_risk_feasible (literal spec)"),
        ("risk_aware_rule_based", "risk_aware_ceiling_persistence (literal spec)"),
        ("risk_aware_rule_based k=1", "risk_aware_rule_based"),
    ]
    for treatment, reference in pairs:
        if treatment in names and reference in names:
            parts += [
                "",
                f"## Paired difference: {treatment} minus {reference}",
                "",
                to_markdown(paired_delta(episodes, treatment, reference)),
            ]
    text = "\n".join(parts) + "\n"
    (args.out / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
