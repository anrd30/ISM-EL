# L3 marginal-violation dual-update cost: real 3-seed result

Status: proposed, not yet reviewed by the other lanes.

Generated from `agent/min_action_counterfactual.py` and `agent/ppo.py
--marginal-violation-cost` (`agent/train.py --marginal-violation-cost`), 3
seeds, 244 iterations, identical protocol to the `b2_244` clean baseline
(`agent/evidence/b2-244-clean-2026-11-12/`) except the Lagrangian dual
update's cost input. Full output in `agent/evidence/l3-2026-11-12/`.

## Background

Proposed by the RL-agent lane: charge the latency dual update only for the
marginal violation an agent's own action adds beyond what the minimum (20
MB/s) action would have caused, computed online from the same live
hazard/work state already available at that training step (no hindsight).
Motivation checked beforehand, independently, on our own data:

- `agent/GAMMA_SWEEP_DECISION.md` -- reweighting the fixed violation term
  (gamma, 1.0 to 10.0) does not close PPO's gap to `risk_aware_rule_based`
  at any value tested.
- `agent/EXOGENOUS_VIOLATION_DECISION.md` -- a load-only exogenous
  definition reads zero everywhere in this simulator; 8 of `fixed_20`'s 20
  holdout episodes are forced above the floor by D1's safety filter, not by
  load, so any usable definition needs live state, not a load-only
  post-hoc check.
- `runs/b2_244_clean/seed{0,1,2}/log.csv` -- `lambda_latency`/`lambda_risk`
  never hit their cap of 2.0 in the clean baseline, in any seed, so the
  "lambda diverging into its cap" premise did not hold for this run to
  begin with.

Prerequisite checks on the implementation itself, before any real training
run: `apply_safety_filter` confirmed to mutate nothing (returns a frozen
dataclass from plain floats); the default (unflagged) training path proven
byte-exact unchanged against a 15-value pre-edit reference; the per-interval
counterfactual confirmed correct; 13 unit tests passing; a 5-iteration
smoke run sane before any real-scale run.

## Protocol

Training, three seeds, gamma held at 2.0 to match the clean baseline:

    python -m agent.train --iterations 244 --rollout-steps 8192 --seeds 0 1 2 \
        --workload-csv data_pipeline/profiles/profile_diurnal_long_train.csv \
        --hazard-error-sigma 0.25 --reward-gamma 2.0 --marginal-violation-cost \
        --out runs/l3_244

Evaluation, all three checkpoints against every baseline on identical scenarios:

    python -m agent.evaluate_policy \
        --checkpoint runs/l3_244/seed0/ckpt_final.pt \
        --checkpoint runs/l3_244/seed1/ckpt_final.pt \
        --checkpoint runs/l3_244/seed2/ckpt_final.pt \
        --workload-csv data_pipeline/profiles/profile_diurnal_long_holdout.csv \
        --seeds 20 --out runs/eval244/l3_3seed --diagnostics \
        --hazard-error-sigma 0.25

Run on 2026-11-12. Workload sha256
`94fe6bd6f8a5cb163cba7a9fafac1274d4ec6511113920a63d4993f03295a53d` (same
holdout file as the clean baseline).

## Result

Paired difference against `risk_aware_rule_based`, total reward, n=20
scenarios (from `summary.md`):

| seed | L3 mean_diff | L3 sem | clean-baseline mean_diff | clean-baseline sem |
|---|---|---|---|---|
| 0 | -23.96 | 8.102 | -20.13 | 1.764 |
| 1 | -20.95 | 1.521 | -16.98 | 1.471 |
| 2 | -22.68 | 2.129 | -39.50 | 3.463 |

Direct comparison, L3 minus the clean baseline (same gamma, same seeds, same
20 holdout scenarios), sem computed by averaging the three training seeds
within each scenario first, then taking sem across the 20 resulting
scenario-level means (the three seeds share scenarios, so 60 raw pairs are
not 60 independent samples -- same correction as
`agent/FORECAST_PPO_DECISION.md`):

| metric | mean diff | sem | t |
|---|---|---|---|
| total_reward | +3.01 | 3.80 | 0.79 |
| completion_hours | -6.23 | 0.62 | -10.04 |
| mean_delivered_mbps | +4.18 | 0.23 | 18.32 |
| latency_violation_fraction | +0.0023 | 0.0013 | 1.73 |
| p_failure_mean | +0.0000 | 0.0000 | -0.10 |
| clip_fraction | +0.0014 | 0.0137 | 0.10 |

Per-training-seed total_reward diff, L3 minus clean baseline: seed0
-3.83±8.61, seed1 -3.97±1.42, seed2 +16.82±3.25.

Lambda trajectories (`agent/evidence/l3-2026-11-12/seed{0,1,2}/log.csv`):
`lambda_latency` ends at 0.0000 in all three seeds (down from a max of
0.15-0.17 early in training), against 1.20-1.68 at the end of the clean
baseline's training. `lambda_risk` is essentially unchanged (0.37-0.61
final, vs. the baseline's 0.30-0.61). Neither multiplier hit its cap of 2.0
in either run, in any seed.

## Finding

L3 does what it mechanically should: once the dual update stops charging
the policy for violations the minimum action would also have incurred,
`lambda_latency` has almost nothing left to react to and relaxes toward
zero -- `lambda_risk`, untouched by the change, stays essentially where it
was, which is the expected control. With that constraint pressure removed,
the policy completes significantly faster (-6.23h, t=-10.04) and at
significantly higher throughput (+4.18 Mbps, t=18.32), with no significant
change in failure probability (t=-0.10) or clip fraction (t=0.10).
Latency violation fraction ticks up but not significantly (t=1.73).

Total reward does not improve significantly in aggregate (t=0.79) and is
uneven across seeds: a large, likely real gain on the one seed whose
baseline was a clear outlier against `risk_aware_rule_based` (seed2,
-39.50 baseline vs. -20.13/-16.98 for seeds 0/1), bringing it in line with
the other two (-22.68 vs. -23.96/-20.95); small, non-significant
regressions on the other two seeds. L3 changes only the dual update's cost
input, not the reward function itself, so the throughput/speed gain the
mechanism produces is not guaranteed to show up as a reward win, and here
it mostly doesn't.

This matches the honest outcome anticipated for L3 going in: lambda
converges rather than diverging into its cap, and this does not create
reward headroom. It produces a real, significant behavioral change (faster,
higher-throughput policy at unchanged risk) that the reward metric alone
does not reward.

## Falsification check

Lambda saturating in low-exogenous-fraction conditions would have
falsified the premise. It did not saturate in either the clean baseline or
the L3 run, in any seed -- consistent with the earlier, separate finding
that the "lambda diverging into its cap" premise did not hold for this
run to begin with. Across the full gamma sweep (`agent/GAMMA_SWEEP_DECISION.md`,
gamma 1.0-10.0, single seed each), `lambda_latency` never caps either:
final values 1.27-1.68 at every gamma tested. The premise that motivated
this check -- lambda saturating at the 2.0 bound -- does not hold anywhere
in this project's own data; the general claim it was narrowed to (lambda
inflated by exogenous violations, whatever level it settles at) is what
the evidence here supports.

## Reclassification (2026-11-12)

Proposed by the RL-agent lane and accepted here: this result is better
read as a mechanism demonstration than a performance result. `lambda_risk`
is an unexercised control -- nothing about it changed -- while
`lambda_latency`, the one input L3 touches, collapsed from 1.20-1.68 to
0.0000 in all three seeds. One targeted change, one multiplier moved, the
untargeted one did not: that is a controlled intervention showing the
latency multiplier was being driven almost entirely by violations the
policy's own action did not cause, not a correlational inference from it.

A specific figure proposed alongside this (roughly 60% of violations
occurring at load levels where even the slowest allocation violates) does
not match what this run's own training-time logs show. Computed directly
from `runs/l3_244/seed{0,1,2}/l3_stats.csv` (`real_violations -
marginal_cost_steps`, summed over all 244 iterations and all three seeds):
398,170 real violation-steps, of which 335,685 would also have violated
under the minimum action -- **84.3%** exogenous, not ~60%. This is the
training workload under the policy's own evolving action distribution
across all of training, not the holdout; a holdout-specific figure under a
fixed evaluation policy could differ and has not been computed here. Either
way the qualitative claim is unchanged and, on this number, stronger: the
large majority of recorded latency violations in this environment are not
attributable to the policy's choice of action.

The seed-level pattern (`+16.82 +/- 3.25` on the one outlier seed, small
non-significant moves on the other two) is also better read as a variance
observation than an established effect, per the same caution raised
alongside it: three seeds is thin evidence for a variance claim.

The oracle-ceiling figure used to interpret the flat reward is now
cross-verified independently in two places rather than asserted by one
lane: `agent/evidence/forecast-ppo-2026-11-11/episodes.csv` gives
`ORACLE_lookahead` beating `risk_aware_rule_based` by 11.850 +/- 1.496 on
20/20 scenarios against a rule-based mean of -330.196, i.e. 3.59% on the
holdout half; `baselines/evidence/diurnal-long-sigma0.25/summary.md` gives
-278.1 vs. -293.3 on the full long profile, i.e. 5.19%. Both numbers come
from committed evidence already in this repo, independent of this L3 run.
The two windows differ (holdout is the harder one), and both represent a
narrow reward ceiling for any controller change, L3 included, to work
inside of.

## Caveat

Only one gamma value (2.0) was tested with L3, matching the clean
baseline. Whether the uneven per-seed effect (seed2 vs. seeds 0/1)
replicates at other gammas, or whether L3 interacts with gamma, is
untested.
