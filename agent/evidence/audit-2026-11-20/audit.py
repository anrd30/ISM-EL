"""Measured audits of the evaluation setup, per population (A holdout, B altB, C popC, D popD).

Evaluation fleet, sigma 0.25, history rows 576, seeds 0 to 19, run with the rule and with fixed_20.
Usage: python agent/evidence/audit-2026-11-20/audit.py <out_dir>
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from baselines.evaluate import _pinned_config  # noqa: E402
from baselines.policies import FixedRate, RiskAwareRuleBased  # noqa: E402
from contracts.contracts import (  # noqa: E402
    ACTIONS_MBPS, BW_TOTAL_MBPS, LATENCY_LIMIT_FACTOR, OBS_SCALES, MAX_EPISODE_STEPS, Observation,
    latency_degradation, latency_violation,
)
from sim.config import v1_eval_config, violation_rho_threshold  # noqa: E402
from sim.recovery_env import RecoveryEnv  # noqa: E402
from sim.safety_filter import apply_safety_filter  # noqa: E402

POPS = {"A": "profile_diurnal_long_holdout.csv", "B": "profile_diurnal_long_altB.csv",
        "C": "profile_diurnal_long_popC.csv", "D": "profile_diurnal_long_popD.csv"}
HAZARD_SCALE = OBS_SCALES["hazard"]


def episode_steps(config, policy, seed: int):
    env = RecoveryEnv(config)
    obs, reset_info = env.reset(seed=seed)
    if hasattr(policy, "set_history"):
        policy.set_history(*(reset_info[k] for k in ("history_fg_bw_mbps", "history_read_iops", "history_write_iops")))
    policy.reset()
    hazard = float(reset_info["hazard_reported_per_hour"])
    rows = []
    for step in range(MAX_EPISODE_STEPS + 1):
        seen = Observation.from_vector(obs)
        action = int(policy(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        rows.append({"step": step, "reward": float(reward), "latency": info["latency_now"], "fg": info["fg_bw_mbps"],
                     "alloc": info["alloc_effective_mbps"], "hazard_reported": hazard, "work": seen.work_remaining,
                     "applied_mbps": info["alloc_applied_mbps"], "reason": info.get("termination_reason")})
        hazard = float(info["hazard_reported_per_hour"])
        if terminated or truncated:
            break
    return pd.DataFrame(rows), float(reset_info["hazard_reported_per_hour"])


def main(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    base = dataclasses.replace(v1_eval_config(hazard_error_sigma=0.25), history_rows=576)
    limit_ms = LATENCY_LIMIT_FACTOR * base.latency_target_ms
    rho_thr = violation_rho_threshold(base)
    combined_thr = rho_thr * BW_TOTAL_MBPS
    thresholds = pd.DataFrame([{"latency_limit_ms": limit_ms, "rho_threshold": rho_thr, "combined_threshold_mbps": combined_thr,
                                "rho_max": base.rho_max, "combined_ceiling_mbps": base.rho_max * BW_TOTAL_MBPS}])
    thresholds.to_csv(out / "latency_violation_threshold.csv", index=False, lineterminator="\n", float_format="%.6g")
    near, degraded, clip, terminal = [], [], [], []
    for pop, name in POPS.items():
        config, _ = _pinned_config(base, Path("data_pipeline/profiles") / name)
        trace_fg = pd.read_csv(Path("data_pipeline/profiles") / name)
        fg_series = (trace_fg["read_bw_kbps"] + trace_fg["write_bw_kbps"]).to_numpy(float) / 1000.0
        near.append({"population": pop, "policy": "trace (fg only)", "intervals": len(fg_series),
                     "within_5mbps_of_combined_threshold": float((np.abs(fg_series - combined_thr) <= 5).mean()),
                     "above_threshold": float((fg_series > combined_thr).mean())})
        for label, policy in (("risk_aware_rule_based", RiskAwareRuleBased(config)), ("fixed_20", FixedRate(20))):
            steps_all, scen_exceed, scen_exceed_any, wrong_floor_steps, n_steps = [], 0, 0, 0, 0
            for seed in range(20):
                df, h0 = episode_steps(config, policy, seed)
                steps_all.append(df)
                scen_exceed += h0 > HAZARD_SCALE
                scen_exceed_any += bool((df["hazard_reported"] > HAZARD_SCALE).any())
                truth = [apply_safety_filter(0, r.hazard_reported, r.work).applied_index for r in df.itertuples()]
                seen = [apply_safety_filter(0, min(r.hazard_reported, HAZARD_SCALE), r.work).applied_index for r in df.itertuples()]
                wrong_floor_steps += int(sum(a != b for a, b in zip(truth, seen)))
                n_steps += len(df)
                terminal.append({"population": pop, "policy": label, "seed": seed, "reason": df["reason"].iloc[-1]})
            steps = pd.concat(steps_all, ignore_index=True)
            combined = steps["fg"] + steps["alloc"]
            near.append({"population": pop, "policy": label + " (fg + delivered)", "intervals": len(steps),
                         "within_5mbps_of_combined_threshold": float((np.abs(combined - combined_thr) <= 5).mean()),
                         "above_threshold": float((combined > combined_thr).mean())})
            d = latency_degradation
            c_part = steps["reward"] + steps.apply(lambda r: base.reward.alpha * d(r.latency, base.latency_target_ms)
                                                   + base.reward.gamma * latency_violation(r.latency, base.latency_target_ms), axis=1)
            degraded.append({"population": pop, "policy": label, "intervals": len(steps),
                             "all_intervals_charge_beta": bool(np.allclose(c_part, -base.reward.beta, atol=1e-9)),
                             "min_c": float((-c_part / base.reward.beta).min()), "max_c": float((-c_part / base.reward.beta).max())})
            clip.append({"population": pop, "policy": label, "scenarios": 20, "scenarios_reset_hazard_above_scale": int(scen_exceed),
                         "scenarios_any_step_hazard_above_scale": int(scen_exceed_any),
                         "steps": n_steps, "steps_floor_from_clipped_hazard_differs": wrong_floor_steps,
                         "share_steps_floor_differs": wrong_floor_steps / n_steps})
    pd.DataFrame(near).to_csv(out / "intervals_near_threshold.csv", index=False, lineterminator="\n", float_format="%.6g")
    pd.DataFrame(degraded).to_csv(out / "degraded_cost.csv", index=False, lineterminator="\n", float_format="%.6g")
    pd.DataFrame(clip).to_csv(out / "hazard_clip.csv", index=False, lineterminator="\n", float_format="%.6g")
    pd.DataFrame(terminal).to_csv(out / "termination_reasons.csv", index=False, lineterminator="\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
