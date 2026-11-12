import sys; sys.path.insert(0, ".")
import numpy as np
from agent.train_residual import build_sim_config
from baselines.policies import RiskAwareRuleBased
from sim.recovery_env import RecoveryEnv
from agent.forecast_ppo import ForecastPPOPolicy
from pathlib import Path
import dataclasses
for variant in ("oracle", "causal"):
  for name, csv, seeds in (("home extremes", "data_pipeline/profiles/profile_diurnal_long_holdout.csv", [9, 17, 19]), ("peak", "data_pipeline/profiles/profile_peak.csv", [0, 1, 2])):
      cfg = dataclasses.replace(build_sim_config("v1", True, 0.25, latency_filter=variant == "oracle", latency_filter_causal=variant == "causal"), workload_csv=csv)
      for label, mk in (("rule", lambda c: RiskAwareRuleBased(c)),
                        ("forecast_ppo_bc seed0", lambda c: ForecastPPOPolicy.from_checkpoint(Path("agent/evidence/forecast-ppo-2026-11-11/training/bc/seed0/ckpt_best.pt"), c, "p"))):
          pol = mk(cfg); env = RecoveryEnv(cfg)
          for sd in seeds:
              obs, _ = env.reset(seed=sd); pol.reset(); n = clip = over = infeas = viol = 0; done = False; mix = np.zeros(4)
              while not done:
                  a = int(pol(obs)); obs, r, te, tr, info = env.step(a)
                  n += 1; clip += info["alloc_clipped_by_latency_filter"]; over += info["latency_filter_overridden_by_reliability"]; infeas += (not info["latency_ceiling_feasible"]); viol += info["violated_latency"]
                  done = te or tr
              print(f"{variant:6s} {name:14s} {label:22s} seed {sd:2d}: steps {n} latency-clipped {clip/n:.3f} reliability-overrode {over/n:.3f} ceiling-infeasible {infeas/n:.3f} violated {viol/n:.3f}")
