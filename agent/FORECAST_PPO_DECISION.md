# T-lane decision record: a PPO agent that beats the rule

Status: proposed. Not yet reviewed by the other lanes. **See the correction
dated 2026-11-12 below the Evidence section before reading the headline
number as support for the title of this document.**

Follows `agent/PPO_COLLAPSE_DECISION.md` (constrained PPO trailed the rule) and
`agent/SEASONAL_SEARCH_DECISION.md` (a same-time-yesterday load forecast is the
exploitable signal). Evidence: `agent/evidence/forecast-ppo-2026-11-11/`.

## Decision

- **Adopted controller:** the PPO agent in `agent/forecast_ppo.py`.
  - A neural network chooses the rebuild allocation directly, one of the four
    `ACTIONS_MBPS`, with actions below the safety filter's risk floor masked out.
  - No rule runs inside the policy.
  - It is trained with clipped PPO on the contract reward, with contract coefficients;
    unlike `b2_244` there is no `gamma` override.
- **Inputs:**
  - the 12 contract observation fields;
  - a 7-value load-forecast block (same-time-yesterday forecast, recent level and
    spread, forecast error spread);
  - the risk floor.
  - Everything is built from observations of the current episode, the same information
    the rule has.
- **Warm start:** the network is first fitted by behaviour cloning to the hand-built
  `SeasonalForecastRule` (96% action agreement on the training days), then trained with
  PPO with a KL penalty towards that start.
- **Training conditions:** train-split fleets; the first 7 training days for training
  and the last 3 for validation and checkpoint selection; nothing from the holdout.

## Why these choices

- **Forecast inputs:** the 12 contract fields carry no time-of-day information, but
  `profile_diurnal_long` repeats day to day. The next-interval RMSE is 2.0 MB/s from
  the forecast against 10.1 MB/s from the last value. The earlier PPO could not see this.
- **Warm start:** PPO from scratch with the same inputs collapsed on all three seeds
  (-18 to -20 against the rule), through the starvation mechanism in
  `PPO_COLLAPSE_DECISION.md`. Starting from a sensible policy avoids that.
- **Same networks before and after PPO:** this isolates what PPO training adds from
  what imitation provides.

## Evidence

Configuration `v1,eval-fleet,sigma=0.25`, holdout `profile_diurnal_long_holdout.csv`,
20 paired scenarios. Rule -330.2, oracle -318.3, so the oracle headroom is +11.9. The
baseline harness reproduced the `b2_244` evaluation row for row.

| Policy | Gap vs rule | sem | Scenarios better |
| --- | --- | --- | --- |
| PPO agent, seed 0 | +6.24 | 1.42 | 19 / 20 |
| PPO agent, seed 1 | +6.32 | 1.42 | 19 / 20 |
| PPO agent, seed 2 | +6.71 | 1.26 | 20 / 20 |
| pooled, 60 pairs | **+6.43** | **1.35**[^corr] | 58 / 60 |
| same networks before PPO (imitation only) | -2.83 / +1.07 / +4.44 | 1.8 to 2.6 | 11 to 18 / 20 |
| PPO from scratch, 3 seeds | -18.1 to -20.4 | 1.5 to 2.4 | 0 / 20 |
| hand-built `SeasonalForecastRule` (the imitation target) | +3.68 | 0.28 | 20 / 20 |
| `seasonal_search`, 3 seeds (`SEASONAL_SEARCH_DECISION.md`) | +2.86 to +3.68 | ~0.25 | 20 / 20 |

- **Headroom:** the pooled gain is 54% of the oracle's.
- **PPO adds to its starting point:** +5.17, +1.88 and +9.54 against the same network
  before PPO.
- **PPO beats the heuristic it imitated:** +2.56, +2.64 and +3.03 (sem 1.2 to 1.4), on
  8 or 9 of 20 scenarios. The edge is concentrated, not uniform.
- **Latency:** violation fraction 0.045 against the rule's 0.044. Share of episodes
  within the latency budget 0.85 for both.
- **Completion:** every episode completed, in 41.4 to 41.6 h against the rule's 43.6 h.
- **Tails:** all seeds gain on the three extreme scenarios (+0.42 to +2.33). The one
  losing scenario for two seeds (-6.5) is a single extra violation interval.
- **Validation gaps** (training days 8 to 10, best checkpoint, biased upward by
  selection): +4.4, +6.5 and +6.8. Before PPO they were -3.6, +3.8 and -0.6.

## Correction (2026-11-12): the headline number is an argmax artifact, and it reverses

`ForecastPPOPolicy` acted by hardcoded argmax of the logits with no sampling
path at all (`agent/forecast_ppo.py`, before this date). PPO's own training
objective is expected return under *sampling* from the policy, not under its
mode -- so +6.43 was never a measurement of what PPO was trained to
optimise. It was flagged as a live risk (entropy at this checkpoint: 0.09-0.12
nats, not near zero, so argmax and sampling are genuinely different
policies here) and checked directly, rather than assumed.

`ForecastPPOPolicy` now supports `deterministic=False` (sampling, seeded and
reproducible; `agent/evaluate_learned.py --stochastic`). Same 3 checkpoints,
same holdout, same protocol, evaluated honestly:

| | argmax (original) | stochastic (honest) |
|---|---|---|
| seed 0 vs. rule | +6.24±1.42 | **-12.81±2.58** |
| seed 1 vs. rule | +6.32±1.42 | **-5.21±2.59** |
| seed 2 vs. rule | +6.71±1.26 | **-4.91±2.69** |
| pooled (scenario-paired sem) | +6.43±1.35, t=4.77 | **-7.64±2.31, t=-3.31** |

This does not shrink the gain, it reverses it: a significant win over the
rule becomes a significant loss. All three seeds flip sign individually.
The per-stratum breakdown (`agent/evidence/forecast-ppo-2026-11-12-stochastic/strata.csv`)
shows the loss is not confined to one stratum -- it holds on the
pre-registered "typical" split and on the extreme-scenario split alike.

**What does and doesn't survive:** the hand-built `rule_seasonal_forecast`
comparator in the same evaluation run is unaffected (it's deterministic by
construction, so there's no argmax/stochastic distinction for it), and it
still beats `risk_aware_rule_based` by +3.69±0.28 -- consistent with its
own earlier reported number. So the underlying forecast *signal* (same-time-
yesterday load) is still real and still exploitable by a hand-built rule.
What doesn't survive is the specific claim that *this trained PPO network*,
run the way its own objective was trained, exploits that signal better than
the rule does. It does not; under honest evaluation it loses to the rule it
was warm-started from and KL-penalised toward.

**Why this one checkpoint was the most exposed in the project:** every other
positive result evaluated so far (`L3_DECISION.md`, `BARRIER_DECISION.md`)
showed the same general argmax-flatters-the-policy pattern, but none of
them reversed sign -- they lost significance or lost a seed's apparent
win, not the whole comparator relationship. This network's residual entropy
(0.09-0.12 nats) is far lower than the `agent/ppo.py`-family policies
(0.9-1.1 nats) checked for barrier/L3/baseline, which made it look like the
*safer* case for an argmax/stochastic gap going in (closer to fully
deterministic). It was not: a near-peaked-but-not-peaked categorical policy
can still have its probability mass split closely enough between two
actions that argmax and the true sampling distribution disagree on which
one is actually favoured on net, and that is what happened here. Entropy
alone, without measuring the actual gap, was not a reliable guide, for the
project's own prior intuition or for the one given above when this
correction was scoped -- both read a low entropy number as reassuring and
both were wrong to.

**Status implication:** the title and Decision section above ("a PPO agent
that beats the rule") are not supported by this evidence as it now stands.
This needs the other lanes' review before anything is adopted on the
strength of it. The hand-built `rule_seasonal_forecast` and
`seasonal_search` comparators, both deterministic and both still showing a
real (if smaller) gain over the rule in this same evaluation, remain the
better-supported claims in this document.

(Full evidence: `agent/evidence/forecast-ppo-2026-11-12-stochastic/`.)

## Rolling-origin check (2026-11-12): the reversal replicates, it is not one unlucky split

The single train/holdout split above is the same split every positive result
in this document was designed and selected against (see Limits, "Holdout
reuse"). Before concluding anything further from -7.64, it was checked
against 3 additional sliding train/holdout folds of the same 14-day source
(`data_pipeline/tencent/sliding_rolling_folds.py`): 8-day train / 4-day
holdout / 1-day step. With only 14 days of source data, consecutive folds
overlap by about 75% -- **this is a boundary-sensitivity check, not
independent-fold cross-validation**, and is reported as such, not as CV.

Same protocol throughout (bc-init, 3 seeds, `--stochastic`), pooled gap vs.
`risk_aware_rule_based` (scenario-paired, 20 scenarios per fold):

| fold | train window | pooled gap | sem | t |
|---|---|---|---|---|
| 1 | days 0-8 | -11.926 | 2.754 | -4.330 |
| 2 | days 1-9 | -7.389 | 2.895 | -2.553 |
| 3 | days 2-10 | -7.008 | 1.967 | -3.564 |
| original split | days 0-10 | -7.641 | 2.308 | -3.311 |

All four partitions -- the original split and all three folds -- show a
significant loss to the rule. Across-fold mean -8.775, sd 2.736 (n=3,
sem 1.580). Fold 3's holdout window is the same calendar days as the
original split's holdout (only the train window is shifted earlier by 2
days), and its gap (-7.008) lands close to the original (-7.641) as a
direct consistency check.

Per-seed, the sign is consistently negative in 8 of 9 seed-by-fold
combinations; the one exception is seed 2 on fold 3 (-1.029±2.247, not
distinguishable from zero) -- a wash, not a reversal back to a win. Seed 1
is the most consistently negative across folds (-9.41, -9.27, -16.82).

**Conclusion:** the reversal found above is not an artifact of one
particular train/holdout partition. It replicates, with some spread in
magnitude (-7 to -12), across every partition checked. The loss to
`risk_aware_rule_based` is the better-supported reading of this
checkpoint's real behaviour than the original +6.43.

(Full evidence: `agent/evidence/forecast-ppo-2026-11-13-folds/`.)

## Rejected alternatives

- **PPO from scratch with the same inputs:** collapsed, as above.
- **Per-interval correction networks** on top of the rule (PPO and counterfactual action
  values, `agent/train_residual.py`): did not generalise to later days.
  See `SEASONAL_SEARCH_DECISION.md`.
- **Three-parameter policy search over the forecast rule** (`agent/search_seasonal.py`):
  beats the rule (+3.27 pooled) but does not beat its own untuned version. Kept as the
  interpretable comparator.

## Limits

- **Holdout reuse:** this configuration was evaluated on the holdout once. Its design
  (forecast inputs, imitation target) followed the earlier holdout finding that the
  forecast carries the gain, so the holdout is not untouched by the design process.
- **Workload:** one workload family, selected for its daily cycle. Outside it the agent
  loses to the rule, badly on bursty and high load; see "Out-of-distribution results"
  below.
- **Forecast warm-up:** the forecast needs a day plus an hour of the current episode,
  less than half of a typical 44-hour episode.
- **Training budget:** 3 seeds, 120 iterations of 4,096 steps, one architecture, untuned
  hyperparameters. Training scenarios cycle through a pool of 300 per seed.
- **Untested route:** whether PPO alone could reach this result without the warm start
  under some other setting is not tested here.
- **Simulator:** the latency model is uncalibrated (`sim/LIMITATIONS.md`). Every figure
  is a property of the model.

## Out-of-distribution results (added 2026-11-11)

`agent/evaluate_ood.py`; evidence `agent/evidence/forecast-ppo-ood-2026-11-11/`.
Reproduced exactly from the committed checkpoints. Gap against `risk_aware_rule_based`,
3 seeds:

| Condition | PPO agent | Hand-built forecast rule |
| --- | --- | --- |
| holdout, all fleets degraded (level 0.9) | +4.2 to +4.8 | +3.5 |
| holdout, days shifted 6 h per day | -10 to -35 | -3.2 |
| synthetic `diurnal`, days differ | -2.5 to -23 | -4.5 |
| synthetic `bursty` | -174 to -196 | -38 |
| synthetic `peak` (load above the training range) | -9,283 | -0.8 |

- **Scope of the headline gain:** the +6.43 holds on `profile_diurnal_long`, which was
  selected for the strongest daily cycle. It also holds under a risk-side shift.
- **Where it inverts:** where the daily pattern is absent or changes, or where load leaves
  the training range.
- **Status:** this is a limit of the adopted controller, not of the evaluation.
- **Open question:** whether a real Tencent workload not selected for its daily cycle
  behaves like the synthetic ones. `agent/workload_selection_check.py` answers it; it
  needs the Tencent pipeline outputs in `data_pipeline/data/processed/`, and it has not
  been run.

## Statistical correction (2026-11-12)

The pooled-row sem in the table above was **1.35**, not the 0.78 it originally
read. The three seeds are evaluated on the same 20 scenarios, so the 60
pairs are not 60 independent samples: mean within-scenario, across-seed sd
is 0.48, against a between-scenario sd of 6.03 -- near-replicates, not
independent draws. Averaging the three seeds within each scenario first,
then taking the sem across the resulting 20 scenario-level means, gives sem
1.35 (t = 4.77). The mean is unchanged at +6.43; only the uncertainty
estimate was wrong. Verified independently from
`agent/evidence/forecast-ppo-2026-11-11/episodes.csv`.
