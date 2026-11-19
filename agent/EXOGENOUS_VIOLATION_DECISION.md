# Load-only exogenous-violation check does not explain latency violations here

Status: proposed, not yet reviewed by the other lanes.

Generated from `agent/exogenous_fraction.py` run against
`agent/evidence/b2-244-clean-2026-11-12/episodes.csv`
(`data_pipeline/profiles/profile_diurnal_long_holdout.csv`). Full output in
`agent/evidence/exogenous-violation-check-2026-11-12/`.

## Background

L3's marginal-cost proposal needs a definition of "exogenous": a violation
the policy's own action did not cause, versus one the minimum action (20
MB/s) would also have caused. The simplest candidate is load-only:
`min_action_violates`, evaluating `latency_violation` at the smallest action
in `ACTIONS_MBPS` against the interval's foreground load alone
(`agent/exogenous_fraction.py`). `fixed_20` is a direct validation case for
this: it always requests the literal minimum action, so its own recorded
`latency_violation_fraction` is ground truth for how often that action
actually violates, in this run.

## Result

The load-only `exogenous_fraction` is 0.0 for every policy and every
episode checked, including `fixed_20` itself. `fixed_20`'s own recorded
`latency_violation_fraction` is 0.043559 aggregate; the worst single-episode
mismatch between the load-only estimate and the recorded rate is 0.4212.
The load-only definition predicts zero exogenous violations everywhere in
this simulator.

## Mechanism

Cross-referencing `clip_fraction` in `fixed_20`'s own episodes: 8 of its 20
holdout episodes have `clip_fraction` > 0 (ranging 0.056 to 0.808); the
other 12 are exactly 0 and match the load-only prediction exactly. The 8
clipped episodes are where `sim/safety_filter.py` (`apply_safety_filter`,
lines 93-125) raised the delivered rate above the requested 20 MB/s floor
-- up to 86 MB/s observed -- for reliability reasons. That forced delivery,
not the raw foreground load, is what pushes those 8 episodes into a
latency violation; the holdout trace's peak foreground load alone (116
MB/s) never reaches the threshold (162 MB/s) needed to force a violation at
a literal 20 MB/s floor. A load-only definition has no way to see this,
because the filter's override depends on the episode's accumulated hazard
and remaining work, not on the current interval's load alone.

## Implication

A load-only exogenous-violation check will always read zero in this
simulator and understates the true floor-forced share by exactly the
clipped episodes' contribution. This is why the post-hoc, CSV-only version
of this check (`agent/exogenous_fraction.py`, evaluation-time, load only)
was not extended into a per-interval training-time cost function. The L3
implementation (`agent/min_action_counterfactual.py`) avoids this by
calling `apply_safety_filter` a second time with the counterfactual minimum
action, using the same live hazard/work state already computed for the
real step at that point in training -- so it does inherit the filter's
override behavior and is not subject to this limitation.

## Mechanistic prediction (2026-11-12)

Proposed by the RL-agent lane: the exogenous fraction should fall as a
policy acts more aggressively, and rise toward 100% as it retreats to the
floor. At the minimum action, violations are exactly the unavoidable set
(call it X); any faster policy adds a self-inflicted set Y on top, so its
exogenous fraction is X/(X+Y). As a policy slows toward the floor, Y
shrinks and the fraction climbs toward 100%, which is also offered as the
reason a penalty-constrained policy has no incentive to stop retreating:
slowing down removes Y but cannot touch X, so the latency penalty never
fully disappears and the policy converges to maximum timidity.

Checked directly against `runs/l3_244/seed{0,1,2}/l3_stats.csv` and
`log.csv` (732 iteration-seed points, excluding iterations with zero real
violations): correlation between the per-iteration exogenous fraction and
a weighted-mean action-level proxy (`20*action_frac_20 +
50*action_frac_50 + 80*action_frac_80 + 120*action_frac_120`) is -0.235 --
the predicted direction, but modest, not strong. Bucketing by action level
shows the exogenous fraction roughly flat around 0.80-0.87 for most of the
run (this policy spent most of training between 43 and 54 Mbps) and
dropping to 0.69 only in the top decile (~58 Mbps) -- consistent with the
predicted direction but better described as a threshold near the top of
the range this run explored than a smooth relationship across it. The
direction holds; the strength of the effect is not established by this
check alone, and this run never explored a genuinely aggressive policy
(it collapsed toward the floor early, which is itself consistent with the
proposed trap).

## Caveat

Checked against evaluation-time `episodes.csv` output, using `fixed_20` as
its own ground truth (it has no other baseline to compare the load-only
estimate against). Says nothing about whether a post-hoc, non-training-time
exogenous check is possible some other way -- only that this specific
load-only one is not. The mechanistic-prediction check above is likewise
limited to the narrow action range one training run actually visited.

## Correction (2026-11-19)

- **Superseded definition.** The load-only `min_action_violates` helper is
  non-functional here (above). The harness now counts, per episode, the
  intervals in which the smallest action the safety filter allows would also
  have violated latency (`exogenous_violations`, `baselines/evaluate.py`): the
  floor is `apply_safety_filter(0, hazard, work_remaining)` on the observation
  the policy saw, the load is the realised `fg_bw_mbps`, and the delivered rate
  and latency use the environment's formulas.
- **The 8 clipped episodes.** The 8 of 20 `fixed_20` holdout episodes with
  `clip_fraction` > 0 are scenarios where the safety filter, not the load,
  sets the allocation. The harness now flags them per scenario
  (`filter_forced`: the filter raises the `fixed_20` applied action on at least
  10% of the intervals). The flag is 4 scenarios on the holdout (seeds 3, 9, 17,
  19) and 4 on altB (3, 9, 17, 19). Seeds 9, 17 and 19 carry 84.2% of the
  rule's total reward cost on the holdout (80.4% of its latency cost on altB).
  Evidence: `agent/evidence/scenario-split-2026-11-19/`.
- **Limit of the count.** The observation's hazard field is clipped at 1.8e-4
  per hour. On the two scenarios where the reported hazard exceeds that (seeds
  9 and 17) the recovered floor is lower than the environment's, and the count is
  a lower bound: for `fixed_20`, whose own violations equal the exogenous ones,
  68 and 20 counted against 179 and 91 realised on the holdout (66 and 9
  against 91 and 42 on altB). On every other scenario the count equals the
  realised violations of `fixed_20`. The counterfactual uses the policy's own
  trajectory for the hazard and the remaining work.
