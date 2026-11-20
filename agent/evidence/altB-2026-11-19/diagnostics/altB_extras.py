"""Predictability of the foreground load by the same time yesterday and by the last value, per profile.

Writes e_predictability.csv next to this file's outputs and prints the paired violation split.
Run from the repository root: python altB_extras.py
"""
import numpy as np
import pandas as pd

from sim.scenario import trace_from_frame

OUT = "runs/altB_diagnostics"
P = "data_pipeline/profiles/profile_diurnal_long_"
rows = []
for name in ("train", "holdout", "altB"):
    fr = pd.read_csv(P + name + ".csv")
    fg = trace_from_frame(fr, name, False).fg_bw_mbps
    t = np.arange(288, len(fg))
    d = np.diff(fg)
    rows.append(dict(
        profile=name,
        rmse_yesterday=np.sqrt(np.mean((fg[t] - fg[t - 288]) ** 2)),
        rmse_last_value=np.sqrt(np.mean((fg[t] - fg[t - 1]) ** 2)),
        step_change_corr_1day=np.corrcoef(d[288:], d[:-288])[0, 1],
        mean_read_iops=fr.read_iops.mean(), mean_write_iops=fr.write_iops.mean(),
        p99_read_iops=np.percentile(fr.read_iops, 99),
        mean_bw=fg.mean(), std_bw=fg.std(), step_change_sd=d.std()))
pd.DataFrame(rows).to_csv(f"{OUT}/e_predictability.csv", index=False, lineterminator="\n", float_format="%.5g")
print(pd.DataFrame(rows).round(3).to_string(index=False))

df = pd.read_csv(f"{OUT}/c_per_episode.csv")
sc = df.drop(columns="model").groupby("seed").mean()
dv = sc.violations - sc.rule_violations
sem = dv.std(ddof=1) / np.sqrt(len(dv))
print(f"paired violations per episode, residual_q minus seasonal: {dv.mean():.3f} sem {sem:.3f} t={dv.mean() / sem:.2f}")
print(f"violation term (gamma=10) contribution to gap: {-10 * dv.mean():.2f}; remainder: {sc.gap_latency_terms.mean() + 10 * dv.mean():.2f}")
print("scenarios with more violations than the rule:", int((dv > 0).sum()), "/", len(dv))
