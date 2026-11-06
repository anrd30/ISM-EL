"""Training loop and command line for the constrained PPO agent.

Usage::

    python -m agent.train --iterations 200 --rollout-steps 8192 --seeds 0 1 2 \\
        --profile bursty --out runs/ppo [--ablate-hazard]

For each seed, ``<out>/seed<k>/`` receives ``log.csv`` (one row per iteration),
``ckpt_latest.pt`` and ``ckpt_final.pt``. Checkpoints are written atomically
(:mod:`agent.checkpoint`).

Log columns, per iteration:

- ``mean_return`` / ``mean_episode_length``: over episodes that finished in the
  iteration's batch (NaN when none finished). Return is the unscaled
  environment reward, without penalties.
- ``latency_violation_fraction`` and ``max_exposure``: the two constraint
  measurements, means over those episodes, read from ``info``.
- ``lambda_latency`` and ``lambda_risk``: the multipliers after this
  iteration's update.
- ``lambda_latency_capped`` and ``lambda_risk_capped``: 1 when this iteration's
  update was clipped by ``lambda_max`` (the multiplier is pinned by the bound,
  not by the constraint), else 0. The mean of the column is the share of
  iterations on which the bound was active; ``train`` prints it at the end.
- ``entropy``, ``approx_kl``, ``clip_fraction``, ``policy_loss``,
  ``value_loss``: means over the update's minibatches.
- ``n_recovery_complete``, ``n_terminal_failure``, ``n_horizon_expired``:
  finished episodes by termination reason.

The ablation flag masks observation fields 5 and 6 and changes nothing else.
Nothing in this module implies a trained policy is any good; it records what
the loop measured.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from agent.checkpoint import save_checkpoint
from agent.ppo import PPOAgent, PPOConfig
from agent.rollout import RolloutCollector
from sim.config import SimConfig, v1_eval_config
from workload.generate import PROFILE_NAMES

__all__ = ["DEFAULT_ROLLOUT_STEPS", "LOG_COLUMNS", "TrainConfig", "train", "main"]

#: Control intervals per rollout. Both the multiplier update and the episodic
#: part of the advantage estimate see only episodes that finish inside a
#: rollout, so the rollout has to span several of them. Episode length under
#: the default configuration (all four profiles, randomised fleet), measured
#: over 200 episodes per policy: uniform-random allocation 434 to 984 steps
#: (median 582, mean 639); always the smallest allocation 716 to 2000 (mean
#: 1366); always the largest 334 to 864 (mean 464). Episodes run to
#: ``MAX_EPISODE_STEPS = 2400`` only if the rebuild never completes. A
#: rollout of 2048 steps therefore held about 3 episodes at uniform-random
#: allocation and 1 to 2 at the slowest allocation. 8192 steps hold about 13
#: episodes at uniform-random allocation and about 6 at the slowest, so each
#: batch contains several full rebuilds and the multiplier measurement is a
#: mean over several scenario draws rather than over one or two. Larger values
#: (16384 gives about 25 and 12) add episodes but halve the number of policy
#: updates in a fixed step budget; 8192 is a compromise between the two, not
#: a tuned value. A rollout can still end mid-episode: the collector carries
#: the running episode into the next batch.
DEFAULT_ROLLOUT_STEPS: int = 8192

LOG_COLUMNS: tuple[str, ...] = (
    "iteration",
    "env_steps",
    "n_episodes",
    "mean_return",
    "mean_episode_length",
    "latency_violation_fraction",
    "max_exposure",
    "lambda_latency",
    "lambda_risk",
    "lambda_latency_capped",
    "lambda_risk_capped",
    "entropy",
    "approx_kl",
    "clip_fraction",
    "policy_loss",
    "value_loss",
    "n_recovery_complete",
    "n_terminal_failure",
    "n_horizon_expired",
)


@dataclass(frozen=True)
class TrainConfig:
    """Run configuration.

    Attributes:
        iterations: Number of rollout/update cycles.
        rollout_steps: Control intervals per rollout.
        seed: Run seed (network init, sampling, episode seeds).
        profile: Workload profile name, or ``None`` to sample per episode.
            Ignored when ``workload_csv`` is set.
        workload_csv: Path of a replay CSV to drive foreground load from,
            instead of the synthetic generator. A run whose training and
            evaluation windows are the same file measures an upper bound on
            what the policy could achieve, not a generalisable result.
        hazard_error_sigma: Log-space spread between the hazard the policy is
            shown and the hazard failures are drawn from. 0.0 means the
            estimate is exact, which makes the safety filter perfectly
            protective and the reliability constraint inert.
        eval_fleet: Draw the fleet from the held-out-split feature table
            (``v1_eval_config``) instead of the train-split table. A run whose
            fleets come from drives the hazard model was fitted on is
            contaminated on the fleet side independently of the workload.
        fleet_degradation: Pinned fleet degradation level, or ``None`` to sample.
        ablate_hazard: Mask observation fields 5 and 6.
        out_dir: Directory for ``log.csv`` and checkpoints.
        checkpoint_every: Write ``ckpt_latest.pt`` every this many iterations.
        ppo: PPO hyperparameters.
    """

    iterations: int = 100
    rollout_steps: int = DEFAULT_ROLLOUT_STEPS
    seed: int = 0
    profile: str | None = None
    workload_csv: str | None = None
    hazard_error_sigma: float = 0.0
    eval_fleet: bool = False
    fleet_degradation: float | None = None
    ablate_hazard: bool = False
    out_dir: str = "runs/ppo"
    checkpoint_every: int = 10
    ppo: PPOConfig = field(default_factory=PPOConfig)


def _nan_if_empty(values: Sequence[float]) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def _checkpoint_payload(agent: PPOAgent, cfg: TrainConfig, iteration: int) -> dict:
    return {
        "agent": agent.state(),
        "meta": {
            "iteration": iteration,
            "seed": cfg.seed,
            "profile": cfg.profile if cfg.profile is not None else "mixed",
            "workload_csv": cfg.workload_csv if cfg.workload_csv is not None else "",
            "hazard_error_sigma": cfg.hazard_error_sigma,
            "eval_fleet": cfg.eval_fleet,
            "fleet_degradation": cfg.fleet_degradation if cfg.fleet_degradation is not None else -1.0,
            "ablate_hazard": cfg.ablate_hazard,
        },
    }


def train(cfg: TrainConfig, echo: bool = True) -> Path:
    """Run training and return the run directory.

    Each iteration collects ``rollout_steps`` intervals, performs the PPO
    update with the multiplier-penalised reward, then updates both multipliers
    from the episodes that finished in the batch, appends a row to ``log.csv``
    (flushed immediately) and prints the same facts.
    """
    if cfg.iterations < 1 or cfg.rollout_steps < 1 or cfg.checkpoint_every < 1:
        raise ValueError("iterations, rollout_steps and checkpoint_every must be >= 1")
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fields: dict[str, object] = {
        "workload_profile": cfg.profile,
        "workload_csv": cfg.workload_csv,
        "hazard_error_sigma": cfg.hazard_error_sigma,
        "fleet_degradation": cfg.fleet_degradation,
    }
    if cfg.eval_fleet or cfg.hazard_error_sigma != 0.0:
        fleet = "eval-fleet" if cfg.eval_fleet else "train-fleet"
        fields["label"] = f"{fleet},sigma={cfg.hazard_error_sigma:g}"
    sim_cfg = v1_eval_config(**fields) if cfg.eval_fleet else SimConfig(**fields)  # type: ignore[arg-type]
    agent = PPOAgent(cfg.ppo, seed=cfg.seed)
    collector = RolloutCollector(sim_cfg, seed=cfg.seed, ablate_hazard=cfg.ablate_hazard)

    env_steps = 0
    started = time.monotonic()
    with open(out / "log.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LOG_COLUMNS))
        writer.writeheader()
        for iteration in range(1, cfg.iterations + 1):
            batch = collector.collect(agent, cfg.rollout_steps)
            env_steps += len(batch)
            stats = agent.update(batch, progress=(iteration - 1) / max(1, cfg.iterations - 1))
            constraint = agent.update_multipliers(batch.episodes)
            counts = batch.termination_counts()
            row = {
                "iteration": iteration,
                "env_steps": env_steps,
                "n_episodes": len(batch.episodes),
                "mean_return": _nan_if_empty([e.episode_return for e in batch.episodes]),
                "mean_episode_length": _nan_if_empty([e.length for e in batch.episodes]),
                "latency_violation_fraction": constraint["latency_violation_fraction"],
                "max_exposure": constraint["max_exposure"],
                "lambda_latency": constraint["lambda_latency"],
                "lambda_risk": constraint["lambda_risk"],
                "lambda_latency_capped": int(constraint["lambda_latency_capped"]),
                "lambda_risk_capped": int(constraint["lambda_risk_capped"]),
                "entropy": stats["entropy"],
                "approx_kl": stats["approx_kl"],
                "clip_fraction": stats["clip_fraction"],
                "policy_loss": stats["policy_loss"],
                "value_loss": stats["value_loss"],
                "n_recovery_complete": counts["recovery_complete"],
                "n_terminal_failure": counts["terminal_failure"],
                "n_horizon_expired": counts["horizon_expired"],
            }
            writer.writerow(row)
            handle.flush()
            if echo:
                print(
                    f"seed={cfg.seed} iter={iteration}/{cfg.iterations} steps={env_steps} "
                    f"episodes={row['n_episodes']} mean_return={row['mean_return']:.2f} "
                    f"latency_violation_fraction={row['latency_violation_fraction']:.4f} "
                    f"max_exposure={row['max_exposure']:.3e} "
                    f"lambda_latency={row['lambda_latency']:.4f} lambda_risk={row['lambda_risk']:.4f} "
                    f"entropy={row['entropy']:.4f} approx_kl={row['approx_kl']:.5f} "
                    f"clip_fraction={row['clip_fraction']:.3f} elapsed_s={time.monotonic() - started:.1f}"
                )
            if iteration % cfg.checkpoint_every == 0:
                save_checkpoint(out / "ckpt_latest.pt", _checkpoint_payload(agent, cfg, iteration))
    save_checkpoint(out / "ckpt_final.pt", _checkpoint_payload(agent, cfg, cfg.iterations))
    if echo:
        lat, risk = agent.lambda_latency, agent.lambda_risk
        print(
            f"seed={cfg.seed} multiplier bound lambda_max={cfg.ppo.lambda_max:g} clipped the update on "
            f"{lat.n_capped}/{lat.n_updates} latency and {risk.n_capped}/{risk.n_updates} reliability updates"
        )
    return out


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m agent.train", description=__doc__.split("\n\n")[0])
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--rollout-steps", type=int, default=DEFAULT_ROLLOUT_STEPS)
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--profile", default="mixed", choices=[*PROFILE_NAMES, "mixed"], help="'mixed' samples per episode")
    p.add_argument(
        "--workload-csv",
        default=None,
        help="replay CSV to drive foreground load; overrides --profile",
    )
    p.add_argument(
        "--eval-fleet",
        action="store_true",
        help="draw the fleet from the held-out-split feature table instead of the train-split table",
    )
    p.add_argument(
        "--hazard-error-sigma",
        type=float,
        default=0.0,
        help="log-space spread between shown and true hazard; 0.0 makes the filter exact",
    )
    p.add_argument("--fleet-degradation", type=float, default=None)
    p.add_argument("--ablate-hazard", action="store_true", help="mask observation fields 5 and 6")
    p.add_argument("--out", default="runs/ppo")
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--gamma", type=float, default=PPOConfig.gamma)
    p.add_argument("--lr", type=float, default=PPOConfig.lr)
    p.add_argument("--lr-lambda-latency", type=float, default=PPOConfig.lr_lambda_latency)
    p.add_argument("--lr-lambda-risk", type=float, default=PPOConfig.lr_lambda_risk)
    p.add_argument("--epochs", type=int, default=PPOConfig.epochs)
    p.add_argument("--minibatch-size", type=int, default=PPOConfig.minibatch_size)
    p.add_argument("--lambda-max", type=float, default=PPOConfig.lambda_max, help="upper bound on both multipliers")
    p.add_argument("--ent-coef", type=float, default=PPOConfig.ent_coef)
    p.add_argument("--ent-coef-final", type=float, default=PPOConfig.ent_coef_final)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    args = _parser().parse_args(argv)
    ppo = dataclasses.replace(
        PPOConfig(),
        gamma=args.gamma,
        lr=args.lr,
        lr_lambda_latency=args.lr_lambda_latency,
        lr_lambda_risk=args.lr_lambda_risk,
        epochs=args.epochs,
        minibatch_size=args.minibatch_size,
        lambda_max=args.lambda_max,
        ent_coef=args.ent_coef,
        ent_coef_final=args.ent_coef_final,
    )
    for seed in args.seeds:
        cfg = TrainConfig(
            iterations=args.iterations,
            rollout_steps=args.rollout_steps,
            seed=seed,
            profile=None if args.profile == "mixed" else args.profile,
            workload_csv=args.workload_csv,
            hazard_error_sigma=args.hazard_error_sigma,
            eval_fleet=args.eval_fleet,
            fleet_degradation=args.fleet_degradation,
            ablate_hazard=args.ablate_hazard,
            out_dir=os.path.join(args.out, f"seed{seed}"),
            checkpoint_every=args.checkpoint_every,
            ppo=ppo,
        )
        run_dir = train(cfg)
        print(f"seed={seed} wrote {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
