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
record, and from the step reward. Nothing is re-derived from the reward. The
quantities ``info`` does not carry are the latency degradation ``D``, computed
from ``info["latency_now"]`` with the contract's own ``latency_degradation``
and the configured latency target (the call the environment makes), and the
``exogenous_violations`` count, which applies the environment's own safety
filter and latency formulas to the observation the policy saw and the realised
load (see ``_run_episode``).

Scenario flags. ``filter_forced`` marks a scenario in which the safety filter
raises a ``fixed_20`` rollout's applied action on at least 10% of intervals
(``FILTER_FORCED_CLIP_FRACTION``). It is a property of the scenario, identical
for every policy, and is computed once per workload, seed set and configuration
by a ``fixed_20`` rollout. ``summarise`` blocks are ``scenario_blocks``: all,
controllable (not forced) and filter-forced scenarios.

Everything measured here comes from the simulator in ``sim/``: its constants
are design decisions and its latency model is uncalibrated
(``sim/LIMITATIONS.md``). Constraint satisfaction is read from the ``info``
flags, never inferred from reward.

Workload sources. A workload is either a generator profile name (a ``str`` in
``workload.generate.PROFILE_NAMES``) or a replay CSV (an ``os.PathLike``,
conventionally a ``pathlib.Path``). The distinction is by type, not by
inspecting the string: a ``str`` is always a profile name and a ``PathLike`` is
always a file, so a file called ``idle`` cannot be mistaken for the profile of
that name and a mistyped profile name is never opened as a path. The command
line accepts strings only and converts file arguments to ``Path`` itself.

Provenance. Every episode row carries ``workload_source`` (the profile name, or
the path exactly as given, not resolved, so a relative path stays portable) and
``workload_synthetic`` (the ``info`` value: ``True`` for the generator, the
sidecar's flag for a CSV, ``None`` when there is no usable sidecar). ``None`` is
reported as unknown and is never replaced by a guess. A CSV run is paired with
a generator run of the same seed only when the CSV has as many rows as the
generator's frame (``N_ROWS``): the start row is drawn uniformly from
``range(len(trace))`` on the same stream, so a different length changes it.

A CSV row also carries ``workload_sha256``, the digest of the file's bytes (``None`` for
the generator), because a path alone does not identify data that is regenerated under
the same name. To compare a synthetic and a real profile of one label pairwise, call
:func:`compare` once per source and join the two frames on ``seed`` (and policy).

Command line::

    python -m baselines.evaluate --profiles all --seeds 20 --out <dir>
    python -m baselines.evaluate --workload-dir data_pipeline/profiles --seeds 20 --out <dir>
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baselines.policies import (
    FixedRate,
    NeedsEnv,
    Policy,
    baseline_policies,
    diagnostic_policies,
)
from contracts.contracts import (
    ACTIONS_MBPS,
    BW_TOTAL_MBPS,
    CONTROL_INTERVAL_S,
    LATENCY_VIOLATION_FRACTION_MAX,
    MAX_EPISODE_STEPS,
    ContractViolation,
    Observation,
    TerminationReason,
    latency_degradation,
    latency_violation,
)
from sim.config import SimConfig, latency_mean_ms, v1_eval_config
from sim.recovery_env import RecoveryEnv
from sim.safety_filter import apply_safety_filter
from sim.scenario import load_replay_csv
from workload.generate import PROFILE_NAMES

__all__ = [
    "EPISODE_COLUMNS",
    "Workload",
    "evaluate",
    "compare",
    "summarise",
    "scenario_blocks",
    "forced_masks",
    "forced_seed_sets",
    "controllable_sensitivity",
    "pareto_table",
    "recovery_time_distribution",
    "paired_delta",
    "to_markdown",
    "check_pairing",
    "main",
]

#: A workload: a generator profile name, or a replay CSV path.
Workload = str | os.PathLike[str]

EPISODE_COLUMNS: tuple[str, ...] = (
    "profile",
    "workload_source",
    "workload_synthetic",
    "workload_sha256",
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
    "trace_start_row",
    "latency_cost",
    "exogenous_violations",
    "data_loss_probability",
    "fixed20_clip_fraction",
    "fixed20_latency_cost",
    "filter_forced",
)

#: A scenario is filter-forced when the safety filter raises a ``fixed_20`` rollout's applied action on at
#: least this share of the intervals (primary definition; fixed before any result was seen).
FILTER_FORCED_CLIP_FRACTION: float = 0.10
#: Sensitivity definitions reported next to the primary one: other clip-fraction thresholds, and a
#: ``fixed_20`` latency cost above ``FIXED20_LATENCY_COST_THRESHOLD``.
SENSITIVITY_CLIP_FRACTIONS: tuple[float, ...] = (0.01, 0.25)
FIXED20_LATENCY_COST_THRESHOLD: float = 50.0

#: Columns that fingerprint a scenario and so must agree across policies. The
#: provenance columns are included so that a policy run on a different file, or
#: on a file whose sidecar changed, is detected.
_SCENARIO_COLUMNS: tuple[str, ...] = (
    "workload_source",
    "workload_synthetic",
    "workload_sha256",
    "fleet_level",
    "initial_fg_bw_mbps",
    "initial_budget_infeasible",
    "trace_start_row",
    "fixed20_clip_fraction",
    "fixed20_latency_cost",
    "filter_forced",
)

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


def _minimum_action_violates(fg_mbps: float, floor_index: int, config: SimConfig) -> bool:
    """Whether the smallest action the safety filter allows would violate latency at ``fg_mbps``.

    The delivered rate is ``min(ACTIONS_MBPS[floor_index], max(0, rho_max * BW_TOTAL_MBPS - fg))``, the
    utilisation ``min(rho_max, (fg + delivered) / BW_TOTAL_MBPS)``, the latency ``latency_mean_ms`` of it:
    the formulas ``RecoveryEnv.step`` applies, with the contract's violation test.
    """
    available = max(0.0, config.rho_max * BW_TOTAL_MBPS - fg_mbps)
    delivered = min(float(ACTIONS_MBPS[floor_index]), available)
    rho = min(config.rho_max, (fg_mbps + delivered) / BW_TOTAL_MBPS)
    return latency_violation(latency_mean_ms(rho, config), config.latency_target_ms) == 1.0


def _run_episode(
    env: RecoveryEnv, policy: Policy, source: str, sha256: str | None, seed: int, config: SimConfig
) -> dict[str, Any]:
    """Run one episode and reduce its per-step records to one row.

    ``source`` is the caller's name for the workload (profile name or path as
    given) and ``sha256`` the digest of the CSV bytes (``None`` for the generator); the label, the
    synthetic flag and the start row come from the reset ``info``.

    ``exogenous_violations`` counts the intervals in which the smallest action the safety filter allows
    would also have violated latency. The floor is ``apply_safety_filter(0, hazard, work_remaining)``;
    ``hazard`` is the reported hazard the environment's filter uses at the start of the interval
    (``info["hazard_reported_per_hour"]`` of the reset, then of the previous step; the observation's
    hazard field is clipped at ``OBS_SCALES["hazard"]`` and is not used), ``work_remaining`` comes from
    the observation the policy saw (de-normalised with ``Observation.from_vector``), and the load is
    ``info["fg_bw_mbps"]`` of the step. Hazard and remaining work are those of the policy's own
    trajectory. ``data_loss_probability`` is ``P(N >= 2)`` for ``N`` Poisson with mean the
    sum of ``info["hazard_true_per_hour"]`` over the realised intervals times the interval length in hours.
    """
    obs, reset_info = env.reset(seed=seed)
    synthetic = reset_info["workload_synthetic"]
    if synthetic is not None and not isinstance(synthetic, bool):
        raise RuntimeError(f"info['workload_synthetic'] must be a bool or None, got {synthetic!r}")
    start_row = int(reset_info["trace_start_row"])  # part of the scenario fingerprint
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
    exogenous: list[bool] = []
    true_hazard: list[float] = []
    hazard_now = float(reset_info["hazard_reported_per_hour"])
    total_reward = 0.0
    final_info: dict[str, Any] = reset_info

    for _ in range(MAX_EPISODE_STEPS + 1):
        action = int(policy(obs))
        seen = Observation.from_vector(obs)
        floor_index = apply_safety_filter(0, hazard_now, seen.work_remaining).applied_index
        obs, reward, terminated, truncated, info = env.step(action)
        hazard_now = float(info["hazard_reported_per_hour"])
        total_reward += float(reward)
        exogenous.append(_minimum_action_violates(float(info["fg_bw_mbps"]), floor_index, config))
        true_hazard.append(float(info["hazard_true_per_hour"]))
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
    latency_cost = float(
        np.sum(config.reward.alpha * degradation + config.reward.gamma * np.asarray(violated_latency, dtype=np.float64))
    )
    expected_failures = float(np.sum(true_hazard)) * CONTROL_INTERVAL_S / 3600.0
    data_loss = -math.expm1(-expected_failures) - expected_failures * math.exp(-expected_failures)
    return {
        "profile": str(reset_info["workload_profile"]),
        "workload_source": source,
        "workload_synthetic": synthetic,
        "workload_sha256": sha256,
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
        "trace_start_row": start_row,
        "latency_cost": latency_cost,
        "exogenous_violations": int(sum(exogenous)),
        "data_loss_probability": data_loss,
    }


_SCENARIO_FLAG_CACHE: dict[Any, tuple[Any, dict[int, tuple[float, float]]]] = {}
_SCENARIO_FLAG_CACHE_SIZE = 16


def _config_key(config: SimConfig) -> tuple[Any, ...]:
    """Hashable description of a config; the hazard model, which cannot be described, enters by identity."""
    fields = [(f.name, repr(getattr(config, f.name))) for f in dataclasses.fields(config) if f.name != "hazard_model"]
    return (id(config.hazard_model), tuple(fields))


def _scenario_flags(
    config: SimConfig, source: str, sha256: str | None, seeds: Sequence[int]
) -> dict[int, tuple[float, float]]:
    """``fixed_20`` clip fraction and latency cost per seed: the scenario's forcing by the safety filter.

    Cached per (workload, seeds, config); the entry holds a reference to the hazard model so that its
    identity cannot be reused while the entry exists.
    """
    key = (source, sha256, tuple(seeds), _config_key(config))
    hit = _SCENARIO_FLAG_CACHE.get(key)
    if hit is not None:
        return hit[1]
    env = RecoveryEnv(config)
    policy = FixedRate(int(ACTIONS_MBPS[0]))
    flags = {}
    for seed in seeds:
        row = _run_episode(env, policy, source, sha256, int(seed), config)
        flags[int(seed)] = (float(row["clip_fraction"]), float(row["latency_cost"]))
    if len(_SCENARIO_FLAG_CACHE) >= _SCENARIO_FLAG_CACHE_SIZE:
        _SCENARIO_FLAG_CACHE.pop(next(iter(_SCENARIO_FLAG_CACHE)))
    _SCENARIO_FLAG_CACHE[key] = (config.hazard_model, flags)
    return flags


def _file_sha256(path: Path) -> str:
    """Hex SHA-256 of the file's bytes, so a row names the exact data that produced it."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sidecar_disclosure(csv_path: Path) -> str:
    """The sidecar's ``disclosure`` (else ``label``) text, or a statement that it is absent.

    Read directly from ``<stem>.meta.json`` beside the CSV. A missing,
    unreadable or malformed sidecar, or one with no non-empty string in either
    field, yields an "absent" message and never raises.
    """
    try:
        meta = json.loads(csv_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "absent (no readable sidecar)"
    if not isinstance(meta, dict):
        return "absent (sidecar is not a JSON object)"
    for key in ("disclosure", "label"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "absent (sidecar has no disclosure or label field)"


def _pinned_config(config: SimConfig | None, workload: Workload) -> tuple[SimConfig, str]:
    """Return ``config`` with the workload pinned, and the workload's source string.

    A ``str`` must be a generator profile name; any other ``os.PathLike`` is a
    replay CSV, which must be an existing file that passes the replay schema
    (it is loaded once here so a bad file fails before any episode runs). The
    source string is the profile name, or the path as given in POSIX form.

    Raises:
        ValueError: on an unknown profile name, a ``str`` that looks like a
            file (so a path passed without ``Path`` is not silently treated as
            a bad name), or a config whose ``workload_csv`` is already set.
        FileNotFoundError: if a CSV path is not an existing file.
        ContractViolation: if the CSV breaks the replay schema.
    """
    base = config if config is not None else SimConfig()
    if base.workload_csv is not None:
        raise ValueError("config.workload_csv is set; the workload is given by the profile argument instead")
    if isinstance(workload, str):
        if workload not in PROFILE_NAMES:
            hint = " (to use a file, pass a pathlib.Path)" if workload.lower().endswith(".csv") else ""
            raise ValueError(f"profile must be one of {PROFILE_NAMES}, got {workload!r}{hint}")
        return dataclasses.replace(base, workload_profile=workload), workload
    if not isinstance(workload, os.PathLike):
        raise TypeError(f"workload must be a profile name or a path, got {type(workload).__name__}")
    path = Path(workload)
    if not path.is_file():
        raise FileNotFoundError(f"workload CSV not found: {path}")
    load_replay_csv(path)  # raises ContractViolation on a schema failure
    return dataclasses.replace(base, workload_profile=None, workload_csv=path), path.as_posix()


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
    profile: Workload,
    seeds: Sequence[int],
    config: SimConfig | None = None,
) -> pd.DataFrame:
    """Run ``policy`` for one episode per seed on one workload.

    Args:
        policy: Any object satisfying :class:`baselines.policies.Policy`. A
            policy that also satisfies ``NeedsEnv`` (the oracle) is bound to
            the environment first.
        profile: A ``str`` in ``workload.generate.PROFILE_NAMES`` (synthetic
            generator), or an ``os.PathLike`` naming a replay CSV. See the
            module docstring for why the two are separated by type.
        seeds: Scenario seeds, one episode each. Pass the same seeds for every
            policy being compared.
        config: Simulator configuration; the workload is pinned on a copy.
            Defaults to ``SimConfig()``.

    Returns:
        One row per seed with the columns of ``EPISODE_COLUMNS``. ``profile``
        is the workload label reported by the environment (the profile name,
        or the sidecar's ``profile`` / the file stem for a CSV);
        ``workload_source`` is the name or path given; ``workload_synthetic``
        is ``True``, ``False`` or ``None`` (unknown) as reported in ``info``.

    Raises:
        ValueError: on an unknown profile name, empty or duplicate seeds, a
            config with ``workload_csv`` set, or a policy whose latency model
            differs from the environment's.
        FileNotFoundError: if a CSV path is not an existing file.
        ContractViolation: if a CSV breaks the replay schema.
    """
    cfg, source = _pinned_config(config, profile)
    sha256 = None if isinstance(profile, str) else _file_sha256(Path(profile))
    seed_list = _check_seeds(seeds)
    _check_policy_config(policy, cfg)
    env = RecoveryEnv(cfg)
    if isinstance(policy, NeedsEnv):
        policy.bind(env)
    rows = [_run_episode(env, policy, source, sha256, s, cfg) for s in seed_list]
    flags = _scenario_flags(cfg, source, sha256, seed_list)
    for row in rows:
        clip_fraction, cost = flags[row["seed"]]
        row["fixed20_clip_fraction"] = clip_fraction
        row["fixed20_latency_cost"] = cost
        row["filter_forced"] = clip_fraction >= FILTER_FORCED_CLIP_FRACTION
    frame = pd.DataFrame(rows, columns=list(EPISODE_COLUMNS))
    # Always object dtype (True / False / None), so frames from sources with and
    # without a sidecar have one column type and None is never coerced to NaN.
    frame["workload_synthetic"] = pd.Series(
        [row["workload_synthetic"] for row in rows], index=frame.index, dtype=object
    )
    return frame


def check_pairing(df: pd.DataFrame) -> None:
    """Raise unless every policy saw the same scenario for each ``(profile, seed)``.

    Compares the scenario fingerprint columns (workload source and synthetic
    flag, fleet degradation level, first foreground bandwidth, whether the
    budget is infeasible at the start, trace start row) across policies. It
    also requires every policy to have the same set of seeds in every profile.
    A profile label that was produced by more than one workload source fails
    the source column, because the sources disagree within a seed.

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
    profiles: Sequence[Workload],
    seeds: Sequence[int],
    config: SimConfig | None = None,
) -> pd.DataFrame:
    """Evaluate every policy on every workload with identical seeds.

    Each entry of ``profiles`` is a profile name or a replay CSV path, as in
    :func:`evaluate`. Returns the concatenated per-episode rows
    (``EPISODE_COLUMNS``), ordered by workload, then policy in the order given,
    then seed. The pairing is verified with :func:`check_pairing`.

    Two workloads in one call must have different labels (``profile`` column),
    because the summaries and paired differences are keyed by label. To compare
    a generator profile with a CSV of the same label, call this once per source
    and compare the frames by seed.

    Raises:
        ValueError: on empty ``policies`` or ``profiles``, duplicate policy
            names, the same workload twice, or two workloads with one label;
            see :func:`evaluate` for the rest.
    """
    if not policies:
        raise ValueError("policies is empty")
    if not profiles:
        raise ValueError("profiles is empty")
    names = [p.name for p in policies]
    if len(set(names)) != len(names):
        raise ValueError(f"policy names must be unique, got {names}")
    seed_list = _check_seeds(seeds)
    frames: list[pd.DataFrame] = []
    label_source: dict[str, str] = {}
    for workload in profiles:
        block = [evaluate(policy, profile=workload, seeds=seed_list, config=config) for policy in policies]
        label = str(block[0]["profile"].iloc[0])
        source = str(block[0]["workload_source"].iloc[0])
        if label in label_source:
            raise ValueError(
                f"workloads {label_source[label]!r} and {source!r} both have the label {label!r}; "
                "results are keyed by label, so evaluate them in separate calls"
            )
        label_source[label] = source
        frames += block
    out = pd.concat(frames, ignore_index=True)
    check_pairing(out)
    return out


def _first_appearance_order(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _parse_flag(value: Any) -> bool | None:
    """A ``workload_synthetic`` cell as ``True``, ``False`` or ``None`` (unknown).

    Accepts the in-memory values and the spellings the episode CSV uses
    (``True``/``False``/``unknown``), so a file written by :func:`main` reads
    back with the same meaning. Anything else raises rather than being coerced.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "false"):
            return text == "true"
        if text == "unknown":
            return None
    raise ValueError(f"cannot read {value!r} as a workload_synthetic flag")


def _provenance(block: pd.DataFrame, profile: str) -> str:
    """``"true"``, ``"false"`` or ``"unknown"`` for one profile's rows.

    Raises:
        ValueError: if the rows come from more than one workload source or
            disagree on the synthetic flag, since a mean over them would mix
            workloads under one label.
    """
    sources = block["workload_source"].unique().tolist()
    flags = {_parse_flag(v) for v in block["workload_synthetic"]}
    if len(sources) != 1 or len(flags) != 1:
        raise ValueError(f"profile {profile!r} mixes workload sources {sources} / synthetic flags {sorted(map(str, flags))}")
    flag = next(iter(flags))
    return "unknown" if flag is None else ("true" if flag else "false")


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per-policy, per-profile means over episodes.

    Rows are ordered by profile then policy, both in order of first appearance
    in ``df``. Column meanings:

    - ``workload_synthetic``: ``true``, ``false`` or ``unknown`` (no usable
      sidecar), from the episode rows; a profile label whose rows mix sources
      raises ``ValueError``.
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
                    "workload_synthetic": _provenance(df[df["profile"] == profile], profile),
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


def scenario_blocks(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Rows split by the scenario's ``filter_forced`` flag: ``all``, ``controllable`` and ``filter_forced``."""
    forced = df["filter_forced"].astype(bool)
    return {"all": df, "controllable": df[~forced], "filter_forced": df[forced]}


def forced_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Boolean "filter-forced" masks, row-aligned with ``df``, for the primary and the sensitivity definitions.

    ``primary``: ``fixed20_clip_fraction >= FILTER_FORCED_CLIP_FRACTION``; ``clip_<p>``: the same with another
    threshold of ``SENSITIVITY_CLIP_FRACTIONS``; ``fixed20_cost``: ``fixed20_latency_cost >
    FIXED20_LATENCY_COST_THRESHOLD``.
    """
    clip = df["fixed20_clip_fraction"].astype(float)
    masks = {"primary": clip >= FILTER_FORCED_CLIP_FRACTION}
    for threshold in SENSITIVITY_CLIP_FRACTIONS:
        masks[f"clip_{threshold:g}"] = clip >= threshold
    masks["fixed20_cost"] = df["fixed20_latency_cost"].astype(float) > FIXED20_LATENCY_COST_THRESHOLD
    return masks


def controllable_sensitivity(
    df: pd.DataFrame, treatment: str, reference: str, metric: str = "total_reward"
) -> pd.DataFrame:
    """Paired ``treatment - reference`` on ``metric`` over the scenarios that are controllable under each definition.

    One row per profile and definition: the number of controllable scenarios, the mean paired difference,
    its standard error across scenarios, ``t`` = mean / sem and the counts better and worse.
    """
    rows: list[dict[str, Any]] = []
    for profile in _first_appearance_order(df["profile"]):
        block = df[df["profile"] == profile]
        a = block[block["policy"] == treatment].set_index("seed")
        b = block[block["policy"] == reference].set_index("seed")
        common = a.index.intersection(b.index)
        masks = forced_masks(b.loc[common])
        diff = a.loc[common, metric].astype(float) - b.loc[common, metric].astype(float)
        for name, forced in masks.items():
            d = diff[~forced]
            n = len(d)
            sem = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
            rows.append({"profile": profile, "definition": name, "n_controllable": n,
                         "mean_diff": float(d.mean()) if n else float("nan"), "sem": sem,
                         "t": float(d.mean() / sem) if n > 1 and sem > 0 else float("nan"),
                         "better": int((d > 0).sum()), "worse": int((d < 0).sum())})
    return pd.DataFrame(rows)


def forced_seed_sets(df: pd.DataFrame) -> pd.DataFrame:
    """Seeds each definition of "filter-forced" flags, per profile.

    ``primary`` is ``fixed20_clip_fraction >= FILTER_FORCED_CLIP_FRACTION``; ``clip_<p>`` uses another
    threshold from ``SENSITIVITY_CLIP_FRACTIONS``; ``fixed20_cost`` is ``fixed20_latency_cost >
    FIXED20_LATENCY_COST_THRESHOLD``. One row per profile and definition.
    """
    rows: list[dict[str, Any]] = []
    for profile in _first_appearance_order(df["profile"]):
        block = df[df["profile"] == profile].drop_duplicates("seed").sort_values("seed")
        definitions = {"primary": block["fixed20_clip_fraction"] >= FILTER_FORCED_CLIP_FRACTION}
        for threshold in SENSITIVITY_CLIP_FRACTIONS:
            definitions[f"clip_{threshold:g}"] = block["fixed20_clip_fraction"] >= threshold
        definitions["fixed20_cost"] = block["fixed20_latency_cost"] > FIXED20_LATENCY_COST_THRESHOLD
        for name, mask in definitions.items():
            seeds = [int(s) for s in block.loc[mask, "seed"]]
            rows.append({"profile": profile, "definition": name, "n_forced": len(seeds), "n": len(block),
                         "forced_seeds": " ".join(map(str, seeds))})
    return pd.DataFrame(rows)


def recovery_time_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Completion hours per profile and policy: n, mean, p50, p90, p95, p99 and max, completed episodes only."""
    rows: list[dict[str, Any]] = []
    for profile in _first_appearance_order(df["profile"]):
        for policy in _first_appearance_order(df["policy"]):
            g = df[(df["profile"] == profile) & (df["policy"] == policy)]
            hours = g.loc[g["completed"].astype(bool), "completion_hours"].to_numpy(dtype=np.float64)
            if g.empty:
                continue
            row: dict[str, Any] = {"profile": profile, "policy": policy, "n": len(g), "n_completed": len(hours)}
            for name, q in (("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99)):
                row[name] = float(np.percentile(hours, q)) if len(hours) else float("nan")
            row["mean"] = float(hours.mean()) if len(hours) else float("nan")
            row["max"] = float(hours.max()) if len(hours) else float("nan")
            rows.append(row)
    return pd.DataFrame(rows, columns=["profile", "policy", "n", "n_completed", "mean", "p50", "p90", "p95", "p99", "max"])


def pareto_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per profile and policy: the completion-time tail against latency, exposure and data-loss figures.

    ``completion_p50/p95/p99`` are over completed episodes; ``viol_frac`` and ``lat_budget_met`` as in
    :func:`summarise`; ``p_fail_mean`` is the filter's exposure of the applied allocation and
    ``p_fail_delivered`` the exposure at the delivered rate; ``exog_viol`` is the mean number of intervals
    per episode in which the smallest allowed action also violates latency; ``data_loss_prob`` is the mean
    simulated data-loss probability per episode (the probability of two or more further failures under
    the true group hazard over the realised episode).
    """
    rows: list[dict[str, Any]] = []
    for profile in _first_appearance_order(df["profile"]):
        for policy in _first_appearance_order(df["policy"]):
            g = df[(df["profile"] == profile) & (df["policy"] == policy)]
            if g.empty:
                continue
            hours = g.loc[g["completed"].astype(bool), "completion_hours"].to_numpy(dtype=np.float64)
            nan = float("nan")
            rows.append(
                {
                    "profile": profile,
                    "policy": policy,
                    "n": len(g),
                    "completion_mean_h": float(hours.mean()) if len(hours) else nan,
                    "completion_p50_h": float(np.percentile(hours, 50)) if len(hours) else nan,
                    "completion_p95_h": float(np.percentile(hours, 95)) if len(hours) else nan,
                    "completion_p99_h": float(np.percentile(hours, 99)) if len(hours) else nan,
                    "viol_frac": float(g["latency_violation_fraction"].mean()),
                    "lat_budget_met": float(g["latency_budget_met"].mean()),
                    "latency_cost": float(g["latency_cost"].mean()),
                    "exog_viol": float(g["exogenous_violations"].mean()),
                    "p_fail_mean": float(g["p_failure_mean"].mean()),
                    "p_fail_delivered": float(g["p_failure_delivered_mean"].mean()),
                    "data_loss_prob": float(g["data_loss_probability"].mean()),
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
        _provenance(block, profile)  # raises if the label mixes workload sources
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


def _workload_section(episodes: pd.DataFrame) -> str:
    """One entry per workload: label, source, synthetic flag, file digest and disclosure.

    The disclosure is the sidecar's own text, repeated here so the caveat
    travels with the numbers it qualifies; an absent one is stated as absent.
    """
    lines = ["## Workloads", ""]
    seen = episodes[["profile", "workload_source", "workload_synthetic", "workload_sha256"]].drop_duplicates()
    for _, row in seen.iterrows():
        flag = _parse_flag(row["workload_synthetic"])
        text = "unknown (no usable sidecar)" if flag is None else ("true" if flag else "false")
        digest = row["workload_sha256"]
        is_file = isinstance(digest, str) and bool(digest)
        lines.append(f"- `{row['profile']}`: source `{row['workload_source']}`, synthetic: {text}")
        if is_file:
            lines.append(f"  - sha256: `{digest}`")
            lines.append(f"  - disclosure: {_sidecar_disclosure(Path(str(row['workload_source'])))}")
        else:
            lines.append("  - sha256: not applicable (generator profile)")
            lines.append("  - disclosure: not applicable (generator profile)")
    return "\n".join(lines)


def _csv_workloads(parser: argparse.ArgumentParser, directory: Path | None, files: Sequence[Path]) -> list[Path]:
    """Replay CSVs named on the command line, validated; exits via ``parser.error``."""
    paths: list[Path] = []
    if directory is not None:
        if not directory.is_dir():
            parser.error(f"--workload-dir {directory} is not a directory")
        paths += sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
        if not paths:
            parser.error(f"--workload-dir {directory} contains no .csv files")
    paths += list(files)
    for path in paths:
        try:
            _pinned_config(None, path)
        except (FileNotFoundError, ContractViolation) as exc:
            parser.error(f"{path}: {exc}")
    return paths


def _build_config(args: argparse.Namespace) -> SimConfig:
    """Base config for the command line: train-split fleet, or the held-out-split fleet with ``--eval-fleet``.

    The label is ``"<fleet>,sigma=<value>"``, with ``",latency-filter"`` appended under ``--latency-filter``;
    it stays ``"default"`` when all flags are at their defaults.
    """
    latency_filter = bool(getattr(args, "latency_filter", False))
    latency_filter_causal = bool(getattr(args, "latency_filter_causal", False))
    fields: dict[str, Any] = {"hazard_error_sigma": args.hazard_error_sigma}
    if args.eval_fleet or args.hazard_error_sigma != 0.0 or latency_filter or latency_filter_causal:
        fleet = "eval-fleet" if args.eval_fleet else "train-fleet"
        fields["label"] = (
            f"{fleet},sigma={args.hazard_error_sigma:g}"
            + (",latency-filter" if latency_filter else "")
            + (",latency-filter-causal" if latency_filter_causal else "")
        )
    if latency_filter:
        fields["latency_filter"] = True
    if latency_filter_causal:
        fields["latency_filter_causal"] = True
    if args.eval_fleet:
        return v1_eval_config(**fields)
    return SimConfig(**fields)


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; returns the process exit status."""
    parser = argparse.ArgumentParser(
        prog="python -m baselines.evaluate",
        description="Paired evaluation of the baseline policies on the recovery simulator.",
    )
    parser.add_argument(
        "--profiles",
        default=None,
        help="'all' or a comma-separated list of generator profile names (default 'all' when no CSV is given)",
    )
    parser.add_argument(
        "--workload-dir",
        type=Path,
        default=None,
        help="directory of replay CSVs; every *.csv in it is evaluated, in name order",
    )
    parser.add_argument(
        "--workload-csv",
        type=Path,
        action="append",
        default=[],
        help="one replay CSV (repeatable); a profile_<name>.meta.json beside it supplies the synthetic flag",
    )
    parser.add_argument("--seeds", type=int, default=20, help="number of paired scenario seeds (0 .. N-1 from --seed-start)")
    parser.add_argument("--seed-start", type=int, default=0, help="first seed")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--diagnostics", action="store_true", help="also run the diagnostic policies")
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
    args = parser.parse_args(argv)
    if args.latency_filter and args.latency_filter_causal:
        parser.error("--latency-filter and --latency-filter-causal are mutually exclusive")
    if args.seeds < 1:
        parser.error("--seeds must be >= 1")
    if not (math.isfinite(args.hazard_error_sigma) and args.hazard_error_sigma >= 0.0):
        parser.error(f"--hazard-error-sigma must be a finite number >= 0, got {args.hazard_error_sigma!r}")
    csv_given = args.workload_dir is not None or bool(args.workload_csv)
    if csv_given and args.profiles is not None:
        parser.error("--profiles cannot be combined with --workload-dir or --workload-csv")
    workloads: list[Workload]
    if csv_given:
        workloads = list(_csv_workloads(parser, args.workload_dir, args.workload_csv))
    else:
        try:
            workloads = list(_resolve_profiles(args.profiles if args.profiles is not None else "all"))
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))

    config = _build_config(args)
    policies: list[Policy] = baseline_policies(config)
    if args.diagnostics:
        policies += diagnostic_policies(config)
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))

    try:
        episodes = compare(policies, profiles=workloads, seeds=seeds, config=config)
    except ValueError as exc:
        parser.error(str(exc))
    summary = summarise(episodes)
    blocks = scenario_blocks(episodes)

    args.out.mkdir(parents=True, exist_ok=True)
    on_disk = episodes.copy()
    # An unknown flag is written as the word, not as an empty cell.
    on_disk["workload_synthetic"] = [
        "unknown" if (flag := _parse_flag(v)) is None else str(flag) for v in episodes["workload_synthetic"]
    ]
    on_disk.to_csv(args.out / "episodes.csv", index=False)

    parts = [
        "# Baseline comparison (simulated)",
        "",
        f"Simulator configuration `{config.label}`; latency model uncalibrated "
        "(sim/LIMITATIONS.md); workload provenance is listed under Workloads. "
        f"{len(seeds)} paired scenario seeds ({seeds[0]}..{seeds[-1]}), identical for every policy. "
        "`completion_h` is the mean over completed episodes only. "
        "`lat_budget_met` and `risk_viol` are fractions of episodes; `viol_frac`, `clip_frac` and "
        "`stall_frac` are fractions of intervals. Latency terms are the mean-latency model of the "
        "simulator; `p_fail_*` are the filter's exposure for the applied allocation (nominal rate).",
        "",
        _workload_section(episodes),
        "",
        "## Scenario blocks",
        "",
        "`filter_forced`: the safety filter raises a `fixed_20` rollout's applied action on at least "
        f"{FILTER_FORCED_CLIP_FRACTION:.0%} of the intervals of that scenario. Controllable scenarios are the rest. "
        "Both are properties of the scenario, identical for every policy.",
        "",
        to_markdown(forced_seed_sets(episodes)),
    ]
    for title, key in (
        ("All scenarios", "all"),
        ("Controllable scenarios", "controllable"),
        ("Filter-forced scenarios", "filter_forced"),
    ):
        block = blocks[key]
        n_seeds = block.drop_duplicates(["profile", "seed"]).shape[0]
        parts += ["", f"## {title} ({n_seeds} profile-seed scenarios)", ""]
        parts.append(to_markdown(summarise(block)) if len(block) else "No scenarios.")
    parts += [
        "",
        "## Pareto table: completion time, latency, exposure and simulated data-loss probability",
        "",
        "`exog_viol` is the mean number of intervals per episode in which the smallest action the safety "
        "filter allows also violates latency, computed on the policy's own trajectory. `data_loss_prob` is "
        "the simulated data-loss probability: the probability of two or more further failures for a Poisson "
        "count with mean the true group hazard integrated over the realised episode.",
        "",
        to_markdown(pareto_table(episodes)),
        "",
        "## Recovery-time distribution (hours, completed episodes)",
        "",
        to_markdown(recovery_time_distribution(episodes)),
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
            if len(blocks["controllable"]):
                parts += [
                    "",
                    f"## Paired difference, controllable scenarios: {treatment} minus {reference}",
                    "",
                    to_markdown(paired_delta(blocks["controllable"], treatment, reference)),
                    "",
                    f"## Total reward difference by definition of controllable: {treatment} minus {reference}",
                    "",
                    to_markdown(controllable_sensitivity(episodes, treatment, reference)),
                ]
    pareto_table(episodes).to_csv(args.out / "pareto.csv", index=False, lineterminator="\n")
    recovery_time_distribution(episodes).to_csv(args.out / "recovery_time_distribution.csv", index=False, lineterminator="\n")
    text = "\n".join(parts) + "\n"
    (args.out / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
