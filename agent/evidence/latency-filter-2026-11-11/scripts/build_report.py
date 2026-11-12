"""Build report.md and combined episode CSVs from run_condition.py outputs.

Usage: python build_report.py RUN_DIR COMMITTED_EPISODES_CSV OUT_DIR

Variants: ``none`` (no filter), ``oracle`` (ceiling_oracle_fg: the interval's own foreground load) and
``causal`` (ceiling_causal_fg: the previous interval's foreground load).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

run_dir, committed, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ORDER = ["home", "peak", "bursty", "diurnal_generic", "idle", "sigma_0", "sigma_0.5", "degraded_fleet"]
POLICIES = ["risk_aware_rule_based", "rule_seasonal_forecast (hand-built)",
            "forecast_ppo_bc seed0", "forecast_ppo_bc seed1", "forecast_ppo_bc seed2"]
REF = "risk_aware_rule_based"
VARIANTS = [("none", 0, "no filter"), ("oracle", 1, "oracle-fg"), ("causal", 2, "causal-fg")]
frames = {}
for name, idx, _ in VARIANTS:
    parts = [pd.read_csv(run_dir / f"{c}_filter{idx}.csv") for c in ORDER if (run_dir / f"{c}_filter{idx}.csv").exists()]
    df = pd.concat(parts)
    df["latency_filter"] = name
    frames[name] = df
out_dir.mkdir(parents=True, exist_ok=True)
for name, idx, _ in VARIANTS:
    frames[name].to_csv(out_dir / f"episodes_filter{idx}.csv", index=False)


def series(variant, cond, pol, col="total_reward"):
    df = frames[variant]
    return df[(df.condition == cond) & (df.policy == pol)].set_index("seed")[col].astype(float)


def gap(variant, cond, pol):
    d = series(variant, cond, pol) - series(variant, cond, REF)
    n = len(d)
    return d.mean(), (d.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")), int((d > 0).sum()), n


L = ["# Latency ceiling: every condition, no filter / oracle-fg / causal-fg", "",
     "`none` is `latency_filter=False`. `oracle-fg` (`ceiling_oracle_fg`, `latency_filter=True`) reads the foreground load of the "
     "interval being decided. `causal-fg` (`ceiling_causal_fg`, `latency_filter_causal=True`) reads the previous interval's load, "
     "the value the policy's own observation carries. Same scenario seeds (0..19), same policies and checkpoints in every column. "
     "`gap` is the paired total-reward difference against `risk_aware_rule_based` under the same variant (policy minus rule; "
     "standard error in brackets); `better` counts scenarios with a positive gap. `viol` is the mean fraction of intervals violating "
     "latency; `lat_ok` the fraction of episodes within the latency budget.", ""]
if committed.exists():
    c = pd.read_csv(committed)
    L += ["## Harness reproduction (no filter, home, against the committed forecast-ppo evidence)", "",
          "| policy | max abs per-scenario difference | mean reward here | mean reward committed |", "|---|---|---|---|"]
    for pol in POLICIES:
        a = series("none", "home", pol); b = c[c.policy == pol].set_index("seed")["total_reward"].astype(float)
        L.append(f"| {pol} | {float(np.abs(a.reindex(b.index) - b).max()):.6g} | {a.mean():.2f} | {b.mean():.2f} |")
    L.append("")
for cond in ORDER:
    if not all((frames[v].condition == cond).any() for v, _, _ in VARIANTS):
        continue
    L += [f"## Condition `{cond}`", "",
          "| policy | reward none | reward oracle-fg | reward causal-fg | gap none | gap oracle-fg | gap causal-fg | better none | better oracle-fg | better causal-fg |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for pol in POLICIES:
        r = [series(v, cond, pol).mean() for v, _, _ in VARIANTS]
        if pol == REF:
            g = ["-"] * 3; b = ["-"] * 3
        else:
            res = [gap(v, cond, pol) for v, _, _ in VARIANTS]
            g = [f"{m:+.2f} ({s:.2f})" for m, s, _, _ in res]; b = [f"{k}/{n}" for _, _, k, n in res]
        L.append(f"| {pol} | {r[0]:.1f} | {r[1]:.1f} | {r[2]:.1f} | " + " | ".join(g) + " | " + " | ".join(b) + " |")
    L += ["", "| policy | viol none | viol oracle-fg | viol causal-fg | lat_ok none / oracle-fg / causal-fg | completion h none / oracle-fg / causal-fg |", "|---|---|---|---|---|---|"]
    for pol in POLICIES:
        v = [series(x, cond, pol, "latency_violation_fraction").mean() for x, _, _ in VARIANTS]
        l = [series(x, cond, pol, "latency_budget_met").mean() for x, _, _ in VARIANTS]
        h = [series(x, cond, pol, "completion_hours").mean() for x, _, _ in VARIANTS]
        L.append(f"| {pol} | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} | {l[0]:.2f} / {l[1]:.2f} / {l[2]:.2f} | {h[0]:.1f} / {h[1]:.1f} / {h[2]:.1f} |")
    w = [min(series(x, cond, p).min() for p in POLICIES) for x, _, _ in VARIANTS]
    L += ["", f"Worst single-episode reward over these policies: none {w[0]:.1f}, oracle-fg {w[1]:.1f}, causal-fg {w[2]:.1f}.", ""]
(out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
