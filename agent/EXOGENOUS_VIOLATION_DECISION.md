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

## Caveat

Checked against evaluation-time `episodes.csv` output, using `fixed_20` as
its own ground truth (it has no other baseline to compare the load-only
estimate against). Says nothing about whether a post-hoc, non-training-time
exogenous check is possible some other way -- only that this specific
load-only one is not.
