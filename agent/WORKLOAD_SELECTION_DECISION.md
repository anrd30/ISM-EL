# Workload selection bias and real-data confirmation of the forecast-PPO OOD finding

Status: proposed, not yet reviewed by the other lanes.

Generated from `agent/workload_selection_check.py` and two `agent.evaluate_learned`
runs against real Tencent-trace workloads. Source data and full output in
`agent/evidence/workload-selection-check-2026-11-12/`. Reproduction gate
(Step 0 of `workload_selection_check.py`) passed: `same_uids=True`,
`same_rows=True` against the committed `data_pipeline/profiles/profile_diurnal_long*`.

## Background

`agent/FORECAST_PPO_DECISION.md` ("Out-of-distribution results", added
2026-11-11) showed the forecast-PPO policy losing badly to
`risk_aware_rule_based` on synthetic non-diurnal workloads (bursty, phase-shifted,
degraded fleet), and flagged as an open question whether this was a real
generalization failure or an artifact of the synthetic profiles being
unrealistically harsh. This document answers that question with real data.

## 1. How extreme is the training workload's selection?

`profile_diurnal_long` (the only workload the agent lane has ever trained or
tuned against) was built by ranking all real candidate volumes in the
14-day window by `daily_strength` (the fraction of a volume's variance
explained by a 288-bin time-of-day mean) and keeping the 44
highest-scoring, most clearly cyclical ones.

Against the full ranked pool (6,054 volumes after excluding 392 with
implausible I/O):

| | daily_strength |
|---|---|
| pool median | 0.0270 |
| pool p90 | 0.1551 |
| pool p99 | 0.7608 |
| selected 44 — min | 0.8321 |
| selected 44 — median | 0.9310 |
| selected 44 — max | 0.9965 |

The least-cyclical volume the agent was ever trained or evaluated on sits
at the **99.29th percentile** of the real pool's cyclicality distribution.
Only 0.73% of the pool is at or above that minimum. The training workload
is not a representative sample of real storage traffic — it is the most
cyclical slice of it.

## 2. Real non-cyclical workload: does the OOD loss reproduce?

`profile_bursty_long` was built from the same 14-day window and candidate
pool, using the `bursty` selection rule instead of `diurnal`
(`daily_strength=0.0451`, just below the pool median; zero overlap with
the diurnal selection), split at the same point as `profile_diurnal_long`.

`agent.evaluate_learned` against `profile_bursty_long_holdout.csv`
(20 paired scenarios, `--eval-fleet`, `sigma=0.25`):

| policy | reward | gap vs rule | n_better | n_worse |
|---|---|---|---|---|
| risk_aware_rule_based (reference) | -223.9 | — | — | — |
| rule_seasonal_forecast (hand-built) | -227.8 | -3.9 | 5 | 14 |
| forecast_ppo_bc seed0 | -573.4 | -349.5 | 1 | 19 |
| forecast_ppo_bc seed1 | -516.4 | -292.6 | 1 | 18 |
| forecast_ppo_bc seed2 | -519.3 | -295.4 | 0 | 20 |

The forecast-PPO policy loses to the rule baseline in 18-20 of 20 paired
scenarios, by -293 to -350 reward on average. This is **larger** than the
synthetic bursty gap reported in `FORECAST_PPO_DECISION.md` (-174 to -196):
real data does not soften the finding, it sharpens it.

What the policy is trading away: vs. the rule baseline it delivers ~7%
more throughput (84.9-85.8 vs 79.2 Mbps) and finishes ~10% faster
(40.6-41.3h vs 45.6h), with a marginally *lower* mean failure probability
(-0.00025). But its latency-violation fraction is 2.5-2.8x higher
(0.068-0.077 vs 0.028), and the contract reward's penalty for that
dominates the trade.

Why: `agent/evidence/workload-selection-check-2026-11-12/daily_strength_candidates.csv`
and `selection_check.md` show, for this same bursty workload,
`rmse_last_value_mbps=8.33` vs `rmse_same_time_yesterday_mbps=35.12` — a
naive "last value" guess beats the "same time yesterday" cyclical forecast
by more than 4x. The forecast-PPO policy's inputs are built on exactly that
cyclical assumption, so on non-cyclical real load the signal it was trained
to exploit is actively misleading, not just uninformative.

## 3. Real near-idle workload: the idle condition stays uninformative

`profile_idle_long` (`daily_strength=0.380`, built the same way with the
`idle` rule): every policy, including every forecast-PPO seed, produces
**identical** results to `risk_aware_rule_based` (gap = 0, all strata, all
seeds) — see `eval_real_idle_summary.md`. This matches the synthetic idle
result in `FORECAST_PPO_DECISION.md` (all gaps 0). At near-zero load there
is no throttling decision to make — running at max rate is trivially
optimal for every controller — so this condition cannot discriminate
between policies either way. It is not evidence for or against
generalization.

## Conclusion

The out-of-distribution weakness reported in `FORECAST_PPO_DECISION.md` is
real, not a synthetic-data artifact. It is explained by where the training
workload sits in the real traffic distribution: the forecast-PPO agent has
only ever been trained and evaluated on the top 0.73% most cyclical real
volumes, and its advantage over the hand-built rule disappears into a
large, consistent loss as soon as the load stops being cyclical — on real
data as much as on synthetic data. The agent should not be described as
generalizing beyond near-diurnal load without this caveat.

## Open questions (unchanged by this document)

- D2 (forecast-error-rate gate) — status not re-checked here.
- Rolling-origin folds across the full Tencent trace — not run.
- Whether a forecast input that falls back to "last value" under low
  cyclicality (rather than always assuming a daily cycle) would close part
  of this gap — not attempted; out of scope for a verification document.
