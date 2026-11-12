# Episode-weighting measurement bias in pooled PPO evaluation

Status: proposed. Not yet reviewed by the other lanes.

## Finding

Pooling episode-level reward across scenarios (as in `agent/evidence/b2-3seed-2026-11-07/summary.md`'s unweighted aggregate) is not the same quantity as the scenario-level mean of per-episode means, whenever the number of steps per episode (`n`) correlates with the per-step reward (`r`).

Exact identity, for any set of episodes with per-episode step count `n_i` and per-episode mean reward `r_i`:

    rho_ep - rho_pooled = -Cov_N(n, r) / mean(n)

where `rho_ep` is the simple (episode-weighted) mean of `r_i` and `rho_pooled` is the step-weighted pooled mean (total reward / total steps). Verified numerically against `episodes.csv` to 1e-16.

## Where it bites

Applied to the `b2_244` / `ppo_lagrangian` evidence: pooling inflates the measured gap vs. the rule by **+0.0213**, which is **42.6%** of the `latency_limit` contract constant (0.05). Longer episodes here correlate with worse per-step reward, so naive pooling overstates the worst episodes relative to a correctly scenario-weighted estimate.

## Consequence

Any aggregate computed by summing reward and steps across episodes and dividing once (pooled), rather than averaging per-scenario means first, carries this bias whenever that correlation exists. `agent/evidence/b2-3seed-2026-11-07/summary.md` and the forecast-PPO evidence in `agent/FORECAST_PPO_DECISION.md` both pool this way; neither currently corrects for it.

## Not covered here

Seed-to-seed aggressiveness on extreme scenarios and the reset-time infeasibility question are already documented directly, not in this file: see `agent/evidence/b2-3seed-2026-11-07/README.md` ("Seed 1 is the best of the three seeds on representative load and the worst by far on the extremes") and `agent/T_DECISION.md` (commit `193c4ec`) respectively.

## Rejected alternatives

None recorded.

## Limits

- Derived and checked against one evidence set (`b2-3seed-2026-11-07`); not yet re-checked against the forecast-PPO evidence, where seeds sharing the same 20 scenarios is a separate non-independence problem handled by a different fix (a sem correction, not this pooling correction) — the two corrections address different statistics and may both be needed on the same evidence.
