# T-lane decision record: the near-constant-action collapse, its causes, and the three-seed confirmation

Status: proposed. Not yet reviewed by the other lanes.

This record covers the investigation of 2026-11-07, which started from a trained
`ppo_lagrangian` policy that had collapsed to a near-constant allocation and lost to
`risk_aware_rule_based` on the held-out window. It supersedes nothing in
`agent/T_DECISION.md`; it is the follow-on that record asked for, and it satisfies that
record's publication condition ("no number in this record may be published until the
clean re-run on a held-out window with committed `sigma` provenance replaces the
upper-bound run"). Every number below comes from a run on
`profile_diurnal_long_holdout.csv` with `v1_eval_config()` held-out fleets and
`hazard_error_sigma=0.25`, which is that clean re-run.

The decisive artifacts are committed under
`agent/evidence/b2-3seed-2026-11-07/`, because `runs/` is gitignored and does not
travel with a clone.

Two provenance classes are marked throughout, because they were produced by different
operators on the same machine and verified to different depths:

- **[verified]** — read directly out of the committed or on-disk artifact named, and
  recomputed from `episodes.csv` where a derived statistic is quoted.
- **[run-reported]** — produced by a scripted analysis during the session and reported
  in its output, not independently recomputed. Reproduction commands are given, but
  these should be re-derived before publication.

## Decision

- **`RewardConfig.gamma` is the lever that resolves the collapse, not the Lagrange
  multiplier.** Training with `gamma=2.0` instead of the contract default `10.0`
  produces a load-adaptive policy instead of a near-constant one, and roughly halves
  the loss against `risk_aware_rule_based`. `gamma` remains at its contract value
  (`contracts/contracts.py`, `RewardConfig`, marked `[ASSUMED]`) for every consumer;
  the training-time override is a CLI flag. **This is a training-time deviation from a
  contract default and must be stated wherever a number from these runs is quoted.**
- **The latency measurement feeding the dual update was biased, and is fixed.**
  `update_multipliers()` averaged `latency_violation_fraction` with equal weight per
  finished episode, regardless of episode length. Episode length correlates with
  violation fraction at -0.82 to -0.90 [run-reported], so short, aggressive episodes
  were systematically over-represented. The measurement is now pooled over steps. The
  fix is unconditional and changes the measured rate by about 0.025 to 0.03
  [run-reported]; it is not behind a flag.
- **`--latency-window`, `--action-prob-floor`, `--action-prob-floor-hold`,
  `--action-prob-floor-end` and `--latency-limit` are experimental knobs, not
  configuration.** Each defaults to off or to the contract constant, each says
  "experimental" in `--help`, and **none of them may be presented as part of the
  working configuration.** They exist so the experiments below are reproducible.
  `--latency-window` defaults to 1, which is exactly the unsmoothed behaviour.
- **Episodes that are latency-infeasible at reset are excluded from the dual update,
  the per-step latency cost, and the pooled measurement.** An episode whose latency
  already violates at zero recovery bandwidth cannot be improved by any action, so
  charging it to the multiplier adds a constant the policy cannot act on. The signal is
  `reset_info["violated_latency"]`, computed in `sim/recovery_env.py` `_info()` from the
  latency at zero recovery bandwidth. This is implemented, carried on `EpisodeRecord`
  as `infeasible`, logged as `n_infeasible_episodes`, and tested. Masking is **on by
  default**; `--no-mask-infeasible` restores the old behaviour of counting these
  episodes in the multiplier and the per-step cost. **It is not the cause
  of the collapse** — see the falsified hypothesis below — but it is correct on its own
  terms and is kept.
- **`baselines/evaluate.py`'s `initial_budget_infeasible` column tracks the wrong
  constraint for this purpose** and was not reused. It is
  `bool(reset_info["violated_risk_budget"])` (line 255) — the reliability constraint,
  not the latency one. It is load-bearing: it is in `EPISODE_COLUMNS` and in
  `_SCENARIO_COLUMNS`, the cross-policy fingerprint tuple. It was left alone.
- **Forced exploration is rejected as a fix.** Two independent action-probability-floor
  designs both made results worse than no floor. See "Rejected alternatives".

## Evidence

### The symptom

The policy under investigation had converged to a near-constant allocation and lost to
`risk_aware_rule_based` on the held-out window. On that window the entire anticipatory
headroom available to any controller is small: `ORACLE_lookahead` scores -318.3 against
the rule's -330.2, a difference of 11.9 reward, or **3.6% of the rule's magnitude**
[verified, `runs/eval244/b2_3seed/summary.md`]. `agent/T_DECISION.md` records 5.2% on
the full `profile_diurnal_long` window; the 3.6% figure here is the time-forward holdout
half of that window, so the two are different windows and not in conflict.

The oracle is not deployable (it reads the next interval's foreground load from the
trace). So the deployable headroom is at most 3.6%, and the policy was moving the wrong
way inside it.

### Falsified hypothesis: reset-time latency infeasibility

The first hypothesis was that unavoidable violations — episodes already in latency
violation at zero recovery bandwidth — were inflating the dual variable and dragging the
policy to the slowest allocation. The mechanism is real and the machinery was built
(above), but **the hypothesis does not generalise: 0 of 250 resets tripped reset-time
latency infeasibility** across train, holdout and synthetic profiles [run-reported].

This sits against a separate report, from the same day and a different operator, that 3
of 20 holdout episodes were latency-infeasible from the start. The two measurements used
different profile sets and configurations and have not been reconciled. **This is an
open discrepancy, not a settled number**, and it is the first thing to re-measure if the
infeasibility path is revisited.

### Diagnosis: the largest action was starved, not avoided

`ACTIONS_MBPS`'s largest entry (120 MB/s) had its probability driven to near zero early
in training and never recovered, which is why the policy could not express a
load-conditional rule [run-reported]:

- Early exploration sampled 120 MB/s across all load levels, not only the low ones,
  giving it a 67% to 72% overall violation rate at a point when the policy was still
  effectively state-blind.
- The pressure doing this was the **base reward's `gamma` term**, not the multiplier:
  `lambda_latency` was only 0.14 to 0.33 over the same interval.
- The probability did not recover after the pressure relaxed.

This is action starvation — the action was eliminated before the policy could learn
*when* it is safe — rather than a learned judgement that the action is bad.

### The two levers, and what they gave

Two independent changes each roughly halved the loss against the rule, evaluated on the
held-out window with 20 paired seeds [run-reported for the gap figures,
[verified] for `b2_244` seed 0 which is in the committed evidence]:

| Run | Change | Paired reward gap vs `risk_aware_rule_based` |
| --- | --- | --- |
| collapsed baseline run | none | roughly -20 to -23 (the 5.4% to 6.9% loss reported that morning) |
| `limit008` | `--latency-limit 0.08` at training time | -9.6 +- 1.3 |
| `b2_244` | `RewardConfig.gamma` 10.0 -> 2.0 | -11.1 +- 1.3 (seed 0) |

Both remained under the real 0.05 latency budget on evaluation, and both produced
genuinely load-adaptive policies rather than a single repeated action.

`--latency-limit` is the **riskier** of the two and is not recommended as a default. An
offline replay of the dual update at limits 0.08 and 0.10 showed `lambda_latency`
collapsing to approximately 0 almost immediately, which leaves the constraint inactive
for most of training [run-reported]. Nothing else in the system enforces latency: the
safety filter polices only the reliability constraint
(`sim/safety_filter.py`, and `sim/LIMITATIONS.md` section 4). For a policy already near
the real boundary this is a live hazard. `b2_244`'s own evaluation violation fraction is
0.047 against a 0.05 budget [verified, `runs/eval244/b2_3seed/summary.md` row
`ppo_lagrangian seed0`, `viol_frac` 0.0332; the 0.047 figure is from the training log's
final iterations], i.e. inside the budget but not far inside it.

`gamma=2.0` is therefore the adopted configuration, and `--latency-limit` stays an
experimental knob.

### The three-seed confirmation

`b2_244` configuration — `gamma=2.0`, latency limit at the contract constant 0.05, no
action floor, 244 iterations, 8192 rollout steps, seeds 0, 1 and 2 — each run to
1,998,848 environment steps [verified, `runs/b2_244/*/log.csv`, 244 logged iterations
each]. The multiplier bound never bound: `lambda_max=2` clipped 0 of 244 latency and 0
of 244 reliability updates on seed 0 [verified, `runs/b2_244/train.txt`].

Evaluated with `python -m agent.evaluate_policy` on
`profile_diurnal_long_holdout.csv`, 20 paired scenario seeds, `--eval-fleet`,
`--hazard-error-sigma 0.25`, policies acting by their mode
[verified, `runs/eval244/b2_3seed/summary.md`]:

| Policy | Reward | Paired gap vs `risk_aware_rule_based` | sem |
| --- | --- | --- | --- |
| `risk_aware_rule_based` | -330.2 | — | — |
| `ORACLE_lookahead` (not deployable) | -318.3 | +11.9 | — |
| `ppo_lagrangian` seed 0 | -341.3 | -11.12 | 1.294 |
| `ppo_lagrangian` seed 1 | -414.3 | -84.07 | 48.23 |
| `ppo_lagrangian` seed 2 | -346.3 | -16.13 | 1.74 |

Seed 1's gap and its standard error are both an order of magnitude off the other two.
**That is not seed noise, and the aggregate of these three numbers must not be
reported as-is.** The cause is in the next section.

### The 20 evaluation scenarios are not homogeneous, and the split matters

The holdout profile's sidecar discloses the scenario set as "selected extremes, not a
representative sample". Recomputed per-episode from
`runs/eval244/b2_3seed/episodes.csv` [verified], the set is bimodal: 17 scenarios carry
a `risk_aware_rule_based` reward between -54.46 and -69.89, and 3 scenarios (seeds 9,
17 and 19) carry -2782.69, -1389.94 and -1385.13 — a 20x to 50x difference in
magnitude, so those 3 dominate any unweighted mean.

Splitting on that line [verified, recomputed from `episodes.csv`]:

| | typical, 17 scenarios | extreme, 3 scenarios |
| --- | --- | --- |
| seed 0 | **-10.98 +- 1.51** | -11.93 +- 1.44 |
| seed 1 | **-8.33 +- 1.67** | **-513.31 +- 197.03** |
| seed 2 | **-15.18 +- 1.94** | -21.53 +- 2.03 |

**On the 17 representative scenarios the result is reproducible across all three
seeds**: pooled -11.50 +- 1.05 over 51 pairs, mean-of-seed-means -11.50 with a
3.45 standard deviation across seeds. This matches seed 0's original -11.12 +- 1.294
and is the number that confirms the configuration.

On the 3 extreme scenarios, seed 1 alone fails badly: its per-episode gaps there are
-142.41 (scenario 9), -814.02 (17) and -583.50 (19), against seed 0's -13.64, -13.08,
-9.07 and seed 2's -17.74, -22.18, -24.67 [verified].

### Mechanism of the seed-1 tail failure

On those same 3 scenarios, seed 1 allocates far more aggressively than the other two
seeds, and pays for it in latency degradation [verified, `episodes.csv`]:

| Scenario | metric | rule | seed 0 | seed 1 | seed 2 |
| --- | --- | --- | --- | --- | --- |
| 9 | `mean_requested_mbps` | 92.6 | 27.6 | **78.6** | 20.4 |
| 17 | `mean_requested_mbps` | 92.3 | 30.8 | **71.2** | 21.4 |
| 19 | `mean_requested_mbps` | 86.3 | 37.7 | **66.0** | 23.6 |
| 9 | `latency_degradation_mean` | 2.445 | 1.795 | **2.133** | 1.667 |
| 17 | `latency_degradation_mean` | 1.109 | 0.828 | **1.543** | 0.707 |
| 19 | `latency_degradation_mean` | 0.972 | 0.791 | **1.139** | 0.623 |

Every policy completed every episode: `terminal_failure` and `horizon_expired` are 0
throughout, so this is accumulated per-step cost, not a termination event [verified].
`risk_budget_violated` is `True` on scenarios 9 and 17 for **every** policy including
the rule, and `False` on 19 — so the extremity of these scenarios is a property of the
environment, not of any controller [verified].

Seed 1 is the best of the three seeds on representative load (-8.33) and
catastrophically the worst on the extremes (-513.31). The same slightly-more-aggressive
policy character produces both. **Typical-case performance is reproducible across
seeds; tail robustness is not**, and nothing in the training setup selects for it.

### Settling whether the starved action was worth recovering

Rather than train again, the question was answered by replaying a fixed low-load rule
through the real `compute_reward` on `b2_244`'s greedy policy, holding everything else
constant. Low load is the lowest tercile of `fg_bw` over that policy's holdout states
(observation value 0.3289). 20 scenarios, `episode_seed(5, i)` — **a different seed set
from the main evaluation**, so the "as-is" rows below are not comparable with the
-341.3 above [run-reported]:

| Low-load rule | reward, gamma=2 | delta vs as-is | reward, gamma=10 | delta vs as-is | violation rate | delivered MB/s |
| --- | --- | --- | --- | --- | --- | --- |
| `b2_244` as is | -201.4 | — | -353.8 | — | 0.0375 | 60.3 |
| 80 MB/s at low load | -197.4 | **+3.95 +- 0.36** | -349.8 | **+3.95 +- 0.36** | 0.0383 | 63.8 |
| 120 MB/s at low load | -259.9 | -58.5 +- 3.7 | -523.1 | -169.3 +- 10.3 | **0.0682** | 76.0 |
| 50 MB/s at low load | -209.0 | -7.6 +- 0.6 | -361.4 | -7.6 +- 0.6 | 0.0360 | 54.5 |

A **blanket** 120 MB/s across the low-load tercile loses at both `gamma` values, and
pushes the violation rate to 0.0682 — over the 0.05 budget even in the safest third of
states. The reward disfavours it; this is not a training artifact.

**The scope of that conclusion matters and is narrower than it looks.** The rule-based
baseline also uses 120 MB/s on roughly 60% of low-load steps, gated by its own risk
read, and scores -330 under the same reward. So the reward does pay for *selective* 120
MB/s use; what this replay rules out is only the *unconditional* version. A selective
policy was not replayed, so **no cause is assigned** for PPO's failure to find one, and
a credit-assignment or observation-information limit remains consistent with the
evidence without being demonstrated by it.

One rigor gap in the table above: the -330 figure it is compared against comes from the
main evaluation's seed set, not these 20 replay seeds. Recomputing
`risk_aware_rule_based` on `episode_seed(5, i)` would make that comparison exact and is
cheap. It has not been done.

The usable finding here is the **80 MB/s row**: +3.95 +- 0.36, about a third of the
remaining gap, available from using an action the policy already selects — it chooses 80
MB/s on only about 60% of low-load steps [run-reported]. This has not been acted on.

## Rejected alternatives

- **Smoothing the latency measurement** (`--latency-window` 10 and 20, plus an offline
  replay). Smoothing *raises* `lambda_latency` rather than lowering it when the
  underlying signal is persistently above the limit, because a trailing window removes
  the occasional favourable dips that let the multiplier decay. Confirmed by offline
  replay and by three live runs at windows 1, 10 and 20 [run-reported]. The flag is
  kept at its default of 1 (no smoothing).
- **Delaying the multiplier / warm-up.** Does not address the cause: the pressure that
  starved the largest action was the base reward's `gamma` term, with the multiplier at
  0.14 to 0.33 at the time. Not run.
- **A flat 2% action-probability floor.** -14.8 +- 1.4, worse than `b2_244`'s
  -11.1 +- 1.3 [run-reported].
- **A front-loaded decaying floor**, 8% held over iterations 1 to 25 and decaying to 0
  by iteration 70. -17.1 +- 1.4, worse still [run-reported].
  Both floors were implemented consistently in the sampled action *and* in the stored
  log-probabilities used by the clipped objective, so neither is a sampling-only
  artifact. The shared mechanism of their failure: forcing exploration into an action
  that is genuinely unsafe at high load creates a persistent violation tax, which
  tightens the multiplier and makes everything else more conservative, and the lesson
  does not stick once the floor relaxes. Reproduced twice in two different designs;
  not pursued further.
- **Recalibrating `--latency-limit` on the strength of a stochastic-versus-greedy
  evaluation gap.** This was recommended and then **withdrawn**. The gap was large for
  one earlier run that had collapsed to 20 MB/s — an action with an outlier-low
  intrinsic violation rate — and absent for `b2_244`, which collapsed to 50 MB/s, whose
  intrinsic rate matches the stochastic average [run-reported]. The gap is an artifact
  of which action a given run collapses onto, not a general training-versus-deployment
  mismatch, so it does not support a global recalibration.
- **A per-action violation-rate floor argument** was raised and then corrected: the
  original table (20 MB/s at about 4.8%, 50 MB/s at about 5.1%) was a selection effect,
  because the policy picks 20 MB/s specifically under high load where violations are
  likelier regardless of action. Unconditional 20 MB/s violates 2.3% to 2.9%
  [run-reported]. The floor argument built on the first table does not hold.

## Verdict

Constrained PPO still does not beat `risk_aware_rule_based` on this problem. The
collapse to a near-constant allocation was a real and fixable defect, and fixing it
roughly halved the loss — from the 5.4% to 6.9% reported that morning to a reproducible
-11.50 +- 1.05 on representative scenarios, against a rule magnitude of -330.2 and a
total deployable headroom of 3.6%. **The remaining gap is larger than the headroom the
method is competing for**, so this is not a near-miss.

Three findings stand on their own and do not depend on PPO winning:

1. A measurement defect in the dual update (episode-weighted instead of step-pooled),
   found, explained and fixed.
2. Action starvation as the collapse mechanism, with the base reward's `gamma` term —
   not the Lagrange multiplier — identified as the pressure responsible, and forced
   exploration shown to be the wrong remedy in two independent designs.
3. Typical-case performance is reproducible across seeds while tail robustness is not,
   demonstrated concretely on 3 extreme scenarios where one of three seeds fails by
   -513 reward through a traceable mechanism (2x to 3x more aggressive allocation).

Finding 3 is the one most worth reporting, because it is a property of the method in
exactly the dimension this project's framing is about, and it is invisible to the
average-reward comparison that the rest of the investigation optimised.

This is reported as a finding, not a failure, on the same basis as
`agent/T_DECISION.md`: the Phase-1 proposal commits to removing the risk input,
retraining, and reporting the outcome rather than reframing it if the central claim
fails.

## Limits

- **Training drew its fleets from the held-out feature table, and this is unresolved.**
  `agent/train.py` line 209 selects `v1_eval_config()` when `--eval-fleet` is passed,
  which uses `v1_eval_drive_features` — the held-out-split table — while the default
  `SimConfig` uses the train-split one. The `b2_244` runs passed `--eval-fleet`, and so
  did evaluation. The workload split is clean and enforced; the fleet pool was not
  split. Fleets are sampled fresh per episode from per-(model, column) pools, so this is
  a shared distribution rather than shared instances, but
  `agent/evidence/upperbound-2026-11-06/README-DO-NOT-REPORT.md` lists the mirror-image
  mistake as one of three disqualifying reasons for that run. **No number in this record
  may be published until the lane that owns the flag and that record rules on whether
  this is contamination.** If it is, the fix is a re-run without `--eval-fleet` and a
  re-evaluation, about 28 minutes per seed.
- `gamma=2.0` is a single-lever, untuned choice, selected because it worked, with
  `{1.0, 2.0}` and the default `10.0` the only values tried at this budget. It is a
  deviation from a contract default marked `[ASSUMED]`, and no sweep establishes it as
  an optimum.
- 244 iterations at 8192 steps is approximately 2M environment steps per seed, one
  architecture, one profile half, three seeds. None of that is a basis for a general
  claim about PPO.
- The three-seed confirmation is reported on 17 of 20 evaluation scenarios. The split is
  justified by the sidecar's own disclosure and by a 20x magnitude difference, and both
  halves are reported above rather than one being dropped — but **the subset was chosen
  after seeing the results**, and a pre-registered scenario set would be stronger.
- Every `[run-reported]` figure above was produced by a scripted analysis and reported
  in its output, not independently recomputed. The analysis scripts were scratchpad
  files and are **not** in the repository, so these figures are reproducible only from
  the commands described, not by re-running a committed script.
- The seed-1 tail failure is characterised on 3 scenarios and 1 seed. Whether it is a
  general property of this configuration, a property of that seed, or a property of
  those scenarios cannot be separated at this sample size.
- The reset-time infeasibility counts (0 of 250 here, 3 of 20 reported separately) are
  unreconciled.
- `--latency-limit`'s effect of driving `lambda_latency` to approximately 0 means a run
  using it has no active latency enforcement from either the multiplier or the safety
  filter. No run using it should be reported without that stated.
- The full test suite has not been re-measured since these changes. The README's pinned
  count (1,695 passed, 16 skipped) predates them, and `agent/T_DECISION.md` already
  notes that a checkout without `hazard/artifacts/v1/` reports a different count
  (1,673 passed, 29 skipped). The count must be re-measured on a checkout that has the
  artifacts before any README update.
