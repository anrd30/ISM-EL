"""Diagnostics of the residual-Q zero-shot result on profile_diurnal_long_altB. No training.

Writes CSV/markdown tables to the directory given as argv[1].
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from agent.residual_features import FEATURE_DIM, HISTORY, SEASONAL_AHEAD, ResidualFeatures
from agent.residual_q import ResidualQPolicy, collect
from agent.seasonal_rule import SeasonalForecastRule
from agent.train_residual import build_sim_config
from baselines.evaluate import _pinned_config, evaluate
from contracts.contracts import (
    BW_TOTAL_MBPS, MAX_EPISODE_STEPS, OBS_FIELDS, ACTIONS_MBPS, degraded_cost, latency_degradation, latency_violation,
)
from agent.search_seasonal import CachedHazard
from sim.config import violation_rho_threshold
from sim.recovery_env import RecoveryEnv
from sim.scenario import load_replay_csv, trace_from_frame

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
PROF = Path("data_pipeline/profiles")
ALTB = PROF / "profile_diurnal_long_altB.csv"
TRAIN = PROF / "profile_diurnal_long_train.csv"
HOLD = PROF / "profile_diurnal_long_holdout.csv"
SEEDS = list(range(20))
MODELS = [Path(f"agent/evidence/residual-q-2026-11-14/training/main/seed{k}") for k in range(3)]

eval_config = build_sim_config("v1", True, 0.25)
cfg_altB, _ = _pinned_config(eval_config, ALTB)

# ---------------------------------------------------------------- (d) harness sanity
print("== (d) unreachable margin on altB ==", flush=True)
models0 = ResidualQPolicy.from_dir(MODELS[0], cfg_altB, "m").models


def rollout(policy, config, seed):
    env = RecoveryEnv(config)
    obs, _ = env.reset(seed=seed)
    policy.reset()
    actions, rewards, infos = [], [], []
    for _ in range(MAX_EPISODE_STEPS + 1):
        a = int(policy(obs))
        obs, r, term, trunc, info = env.step(a)
        actions.append(a); rewards.append(float(r)); infos.append(info)
        if term or trunc:
            break
    return actions, rewards, infos


same = True
for s in SEEDS:
    qa, qr, _ = rollout(ResidualQPolicy(models0, cfg_altB, margin=1e9), cfg_altB, s)
    ra, rr, _ = rollout(SeasonalForecastRule(cfg_altB), cfg_altB, s)
    if qa != ra or qr != rr:
        same = False
        print("MISMATCH seed", s)
print("direct rollouts, actions and per-step rewards identical on 20 seeds:", same, flush=True)
f_q = evaluate(ResidualQPolicy(models0, cfg_altB, margin=1e9, name="x"), profile=ALTB, seeds=SEEDS, config=eval_config)
f_r = evaluate(SeasonalForecastRule(cfg_altB), profile=ALTB, seeds=SEEDS, config=eval_config)
cols = [c for c in f_q.columns if c != "policy"]
harness_same = f_q[cols].reset_index(drop=True).equals(f_r[cols].reset_index(drop=True))
print("evaluate() frames identical (all columns except policy name):", harness_same, flush=True)
pd.DataFrame([{"check": "direct rollouts identical (actions, per-step rewards), 20 seeds", "result": same},
              {"check": "evaluate() frames identical except policy name, 20 seeds", "result": harness_same}]
             ).to_csv(OUT / "d_harness_sanity.csv", index=False, lineterminator="\n")

# ---------------------------------------------------------------- (a) load scale
print("== (a) load scale ==", flush=True)
thr_fg = violation_rho_threshold(cfg_altB) * BW_TOTAL_MBPS
rows = []
for name, path in [("train (fit data)", TRAIN), ("holdout", HOLD), ("altB", ALTB)]:
    fg = trace_from_frame(pd.read_csv(path), name, False).fg_bw_mbps
    row = {"profile": name, "rows": len(fg), "mean": fg.mean(), "p50": np.percentile(fg, 50), "p90": np.percentile(fg, 90),
           "p99": np.percentile(fg, 99), "max": fg.max(), "frac_above_182.2": float((fg > thr_fg).mean())}
    for a in ACTIONS_MBPS:
        row[f"frac_fg+{int(a)}_above_182.2"] = float(((fg + a) > thr_fg).mean())
    rows.append(row)
scale = pd.DataFrame(rows)
scale.to_csv(OUT / "a_load_scale.csv", index=False, lineterminator="\n", float_format="%.4f")
print(scale.round(3).to_string(index=False), "\nthreshold fg-only MB/s:", thr_fg, flush=True)

# ---------------------------------------------------------------- (b) feature coverage
print("== (b) feature coverage ==", flush=True)
names = list(OBS_FIELDS)
names += [f"rule_action_onehot_{k}" for k in range(4)] + ["risk_floor_over_3"]
names += [f"tanh_score_gap_{k}" for k in range(4)]
names += [f"load_change_lag{k}" for k in range(1, HISTORY + 1)]
names += ["yesterday_available"] + [f"yday_change_ahead{j}" for j in range(1, SEASONAL_AHEAD + 1)] + ["level_change_vs_yday"]
names += ["elapsed_fraction"]
for grp in ("thr_minus_last_load", "thr_minus_yday_forecast", "p_violation"):
    names += [f"{grp}_delta{d}" for d in (-1, 0, 1)]
assert len(names) == FEATURE_DIM, (len(names), FEATURE_DIM)

train_cfg_base = build_sim_config("v1", False, 0.25)
train_cfg = dataclasses.replace(train_cfg_base, hazard_model=CachedHazard(train_cfg_base.hazard_model))
x_train, _, _ = collect(train_cfg, str(TRAIN), range(400_000, 400_000 + 80), 0.1, np.random.default_rng(0))
print("training-data samples (80 episodes of the whole train profile, explore 0.1):", len(x_train), flush=True)
lo, hi = x_train.min(axis=0), x_train.max(axis=0)
p01, p99 = np.percentile(x_train, 1, axis=0), np.percentile(x_train, 99, axis=0)


def rule_features(path, seeds):
    cfg, _ = _pinned_config(eval_config, path)
    xs = []
    for s in seeds:
        env = RecoveryEnv(cfg)
        rule_feats = ResidualFeatures(SeasonalForecastRule(cfg))
        obs, _ = env.reset(seed=s)
        rule_feats.reset()
        while True:
            x, step = rule_feats.update(obs)
            xs.append(x)
            obs, _, term, trunc, _ = env.step(step.action)
            if term or trunc:
                break
    return np.array(xs)


cov = {"feature": names, "train_min": lo, "train_max": hi}
any_out = {}
for label, path in [("altB", ALTB), ("holdout", HOLD)]:
    x = rule_features(path, SEEDS)
    out_range = (x < lo - 1e-9) | (x > hi + 1e-9)
    out_p = (x < p01 - 1e-9) | (x > p99 + 1e-9)
    cov[f"{label}_min"], cov[f"{label}_max"] = x.min(axis=0), x.max(axis=0)
    cov[f"{label}_frac_outside_minmax"] = out_range.mean(axis=0)
    cov[f"{label}_frac_outside_p1p99"] = out_p.mean(axis=0)
    any_out[label] = {"states": len(x), "any_feature_outside_minmax": float(out_range.any(axis=1).mean()),
                      "mean_features_outside_minmax_per_state": float(out_range.sum(axis=1).mean()),
                      "any_feature_outside_p1p99": float(out_p.any(axis=1).mean())}
    print(label, any_out[label], flush=True)
covdf = pd.DataFrame(cov)
covdf.to_csv(OUT / "b_feature_coverage.csv", index=False, lineterminator="\n", float_format="%.5g")
pd.DataFrame(any_out).T.to_csv(OUT / "b_feature_coverage_any.csv", lineterminator="\n", float_format="%.5g")
print(covdf[(covdf.altB_frac_outside_minmax > 0) | (covdf.holdout_frac_outside_minmax > 0)][
    ["feature", "train_min", "train_max", "altB_min", "altB_max", "altB_frac_outside_minmax", "holdout_frac_outside_minmax"]
].to_string(index=False), flush=True)

# ---------------------------------------------------------------- (c) where the loss is
print("== (c) decomposition ==", flush=True)
CUT = 288
reward_cfg = cfg_altB.reward
target = cfg_altB.latency_target_ms


def terms(rewards, infos):
    d = np.array([latency_degradation(i["latency_now"], target) for i in infos])
    v = np.array([latency_violation(i["latency_now"], target) for i in infos])
    lat = -(reward_cfg.alpha * d + reward_cfg.gamma * v)
    r = np.array(rewards)
    return r, lat, r - lat, v  # r - lat = -beta*C (degraded-state term)


def episode_stats(policy, seed, rule_step_tracker=None):
    actions, rewards, infos = rollout(policy, cfg_altB, seed)
    r, lat, deg, v = terms(rewards, infos)
    return actions, r, lat, deg, v, infos


rows = []
rule_cache = {}
for s in SEEDS:
    a, r, lat, deg, v, infos = episode_stats(SeasonalForecastRule(cfg_altB), s)
    rule_cache[s] = (a, r, lat, deg, v)

for m_dir in MODELS:
    policy = ResidualQPolicy.from_dir(m_dir, cfg_altB, m_dir.name)
    last = {}
    orig = policy.features.update

    def wrapped(obs, _orig=orig, _last=last):
        x, step = _orig(obs)
        _last["rule_action"] = step.action
        return x, step

    policy.features.update = wrapped
    for s in SEEDS:
        env = RecoveryEnv(cfg_altB)
        obs, _ = env.reset(seed=s)
        policy.reset()
        acts, rule_acts, rewards, infos = [], [], [], []
        for _ in range(MAX_EPISODE_STEPS + 1):
            a = int(policy(obs))
            acts.append(a); rule_acts.append(last["rule_action"])
            obs, rw, term, trunc, info = env.step(a)
            rewards.append(float(rw)); infos.append(info)
            if term or trunc:
                break
        r, lat, deg, v = terms(rewards, infos)
        ra, rr, rlat, rdeg, rv = rule_cache[s]
        n = max(len(r), len(rr))
        pad = lambda z: np.concatenate([z, np.zeros(n - len(z))])
        d_tot, d_lat, d_deg = pad(r) - pad(rr), pad(lat) - pad(rlat), pad(deg) - pad(rdeg)
        acts_a, rule_a = np.array(acts), np.array(rule_acts)
        rows.append({
            "model": m_dir.name, "seed": s,
            "gap_total": d_tot.sum(), "gap_latency_terms": d_lat.sum(), "gap_degraded_term": d_deg.sum(),
            "gap_before288": d_tot[:CUT].sum(), "gap_after288": d_tot[CUT:].sum(),
            "lat_before288": d_lat[:CUT].sum(), "lat_after288": d_lat[CUT:].sum(),
            "deg_before288": d_deg[:CUT].sum(), "deg_after288": d_deg[CUT:].sum(),
            "up_moves": int((acts_a > rule_a).sum()), "down_moves": int((acts_a < rule_a).sum()),
            "up_before288": int((acts_a[:CUT] > rule_a[:CUT]).sum()), "down_before288": int((acts_a[:CUT] < rule_a[:CUT]).sum()),
            "violation_frac": float(v.mean()), "rule_violation_frac": float(rv.mean()),
            "steps": len(r), "rule_steps": len(rr),
            "violations": float(v.sum()), "rule_violations": float(rv.sum()),
            "delivered_mean": float(np.mean([i["alloc_effective_mbps"] for i in infos])),
        })
df = pd.DataFrame(rows)
df.to_csv(OUT / "c_per_episode.csv", index=False, lineterminator="\n", float_format="%.6g")
scen = df.drop(columns="model").groupby("seed").mean()
summ = []
for col in ["gap_total", "gap_latency_terms", "gap_degraded_term", "gap_before288", "gap_after288", "lat_before288",
            "lat_after288", "deg_before288", "deg_after288", "up_moves", "down_moves", "up_before288", "down_before288",
            "violations", "rule_violations", "violation_frac", "rule_violation_frac", "steps", "rule_steps", "delivered_mean"]:
    v_ = scen[col].to_numpy()
    sem = v_.std(ddof=1) / np.sqrt(len(v_))
    summ.append({"quantity": col, "mean_over_20_scenarios": v_.mean(), "sem": sem,
                 "t": v_.mean() / sem if sem > 0 else float("nan")})
summ = pd.DataFrame(summ)
summ.to_csv(OUT / "c_decomposition_summary.csv", index=False, lineterminator="\n", float_format="%.5g")
print(summ.to_string(index=False), flush=True)
