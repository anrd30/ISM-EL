# T decision record: constrained PPO, the estimation-error parameter, and a negative upper bound

Status: proposed. Not yet reviewed by the other lanes.

This is T-lane's first decision record. It covers the three PPO defaults changed after
the pilot runs, the choice of `hazard_error_sigma`, the clean held-out training run and
its result, the earlier contaminated run that is explicitly **not reportable**, and a
finding that invalidates the hazard-input ablation as originally specified.

## Decision

- `PPOConfig.lambda_max`: `inf` to `2.0`. The latency constraint is not satisfiable on
  `peak` under a degraded fleet (`sim/LIMITATIONS.md` sections 13 and 14). An unbounded
  multiplier on an unsatisfiable constraint grows for as long as training runs, and once
  large it dominates the reward on the profiles where the constraint *is* satisfiable,
  pushing the policy to the slowest allocation there too. The bound makes the penalty
  finite. Where it binds, the constraint is no longer enforced by the multiplier, so
  every binding update is recorded: `lambda_latency_capped` and `lambda_risk_capped` in
  `log.csv`, `cap_counts` in the checkpoint, and a summary line at the end of each run.
  The value 2.0 is in units of the scaled reward and is an untuned design choice; the
  rationale for bounding at all is the argument above.
- `PPOConfig.ent_coef_final`: `0.001` to `0.01`, which disables the anneal. The policy
  collapsed to about 0.15 nats within 100 iterations of 2048 steps (uniform over four
  actions is 1.386), concentrating on the smallest allocation. The same collapse
  occurred with the latency penalty switched off (`--lambda-max 0`), so it is not caused
  by the multiplier: on three of the four profiles one of the two smallest allocations
  has the best fixed-allocation return, so the gradient points the same way in most
  states. A constant 0.01 held entropy between about 0.75 and 1.06 nats over 60
  iterations of 8192 steps. Single-seed comparison, untuned.
- `DEFAULT_ROLLOUT_STEPS`: `4096` to `8192`. The multiplier update and the episodic part
  of the advantage estimate see only episodes that finish inside a rollout. Measured
  episode lengths under the default configuration, 200 episodes per policy:
  uniform-random allocation 434 to 984 steps (median 582, mean 639); always the smallest
  allocation 716 to 2000 (mean 1366); always the largest 334 to 864 (mean 464). At 4096
  a batch held one or two episodes at the slowest allocation. At 8192 it holds about
  six, so the multiplier measurement is a mean over several scenario draws.
- `hazard_error_sigma = 0.25`, swept over `{0, 0.25, 0.5, 1.0}` in real runs rather than
  fixed. Derivation and its open provenance problem below.
- Two new CLI flags on `agent/train.py`: `--workload-csv` (replay trace instead of the
  synthetic generator) and `--hazard-error-sigma`. `--lambda-max`, `--ent-coef` and
  `--ent-coef-final` are also exposed so the three changes above can be swept.

## Evidence

### Why `hazard_error_sigma` has to be nonzero

At `sigma = 0` the hazard the policy is shown is the hazard failures are drawn from,
which makes the safety filter exactly protective and the reliability constraint inert.
Three independent confirmations at `sigma = 0`:

1. The risk-blind and risk-aware rule-based baselines differ by -0.004 (sem 0.010) over
   40 episodes, differing on 2 of them.
2. The filter clips the risk-blind policy on 4.5%, 11.9% and 28.5% of steps depending on
   profile, with identical episode outcomes.
3. PPO's reliability multiplier sits pinned at 0 for whole runs.

### The value 0.25

Source: `hazard/calibration_spread.py`, a drive-bootstrap of the v1 reliability table's
log residuals on the held-out split, 16 bins, 228,217 rows per bin, 200 resamples,
reusing A4's `BOOTSTRAP_SEED` and `BOOTSTRAP_RESAMPLES`. Output committed as
`hazard/calibration_spread.json`, a sidecar to `A4_DECISION.md`.

    Log ratio ln(observed / mean_pred), mean absolute (16 bins)   0.234   [0.156, 0.390]
    Log ratio, mean per-bin SE                                    0.183
    Log ratio, weighted mean                                     -0.164   SE 0.085

Read back from the committed file: `headline.log_ratio.mean_abs.point` =
0.23416631224744355, `headline.log_ratio.mean_bin_se` = 0.1829225719885486,
`headline.log_ratio.weighted_mean.point` = -0.16362054367462528.

The mean absolute value of a zero-mean normal is `sigma * sqrt(2/pi)`, so a mean
absolute log residual of 0.234 implies a total spread of `0.234 / 0.798 = 0.293`. That
spread mixes real miscalibration with bootstrap sampling noise. Removing the sampling
component in quadrature:

    sqrt(0.293^2 - 0.183^2) = 0.23

so the noise-corrected miscalibration is about 0.23, which is why 0.25 is used.

Provenance, resolved. The script was written during cross-lane discussion and run once
locally, but sat uncommitted in a working tree, which is why neither lane could point to
it for a time. A-lane located it, re-ran it and committed both script and output (PR
#17). The re-run reproduces the two numbers this record depends on to five significant
figures, so the derivation above now cites a file rather than a remembered value.

### The clean held-out run: PPO loses by more than the whole available headroom

This is the reportable result. It replaces the upper-bound run below, which predicted its
direction and is kept only as that prediction.

Setup, with each contamination source in the upper-bound run closed:

- **Window.** Trained on `profile_diurnal_long_train.csv` (2,880 rows, 10 days), evaluated
  on `profile_diurnal_long_holdout.csv` (1,152 rows, 4 days). One continuous 14-day
  window cut forward in time, so the evaluation half is strictly after the training half
  and neither is the 288-step tiled period of the original profiles.
- **Fleet.** `--eval-fleet` on both sides, so drives are drawn from the held-out-split
  feature table rather than the one v1 was fitted on.
- **Estimation error.** `--hazard-error-sigma 0.25`, provenance in
  `hazard/calibration_spread.json`.
- **Budget.** 244 iterations x 8192 steps = 2.0M environment steps per seed, three seeds,
  on current `main`. Chosen so the result cannot be dismissed as an undertrained agent.

Evaluation is 20 paired scenario seeds, every policy on identical episodes, learned
policies acting by their mode. Evidence in `agent/evidence/clean-holdout-2026-11-06/`;
reproduce with:

    python -m agent.evaluate_policy \
        --checkpoint runs/ppo-clean/seed0/ckpt_final.pt \
        --checkpoint runs/ppo-clean/seed1/ckpt_final.pt \
        --checkpoint runs/ppo-clean/seed2/ckpt_final.pt \
        --workload-csv data_pipeline/profiles/profile_diurnal_long_holdout.csv \
        --eval-fleet --hazard-error-sigma 0.25 --seeds 20 --out runs/ppo-clean-eval

| policy                              | reward    | completion (h) | delivered (MB/s) | lat_deg_mean | violation fraction |
|-------------------------------------|-----------|----------------|------------------|--------------|--------------------|
| ORACLE_lookahead (not deployable)   |  -318.345 | 36.86          | 90.95            | 0.2462       | 0.0468             |
| risk_aware_rule_based               |  -330.196 | 43.60          | 77.33            | 0.2315       | 0.0444             |
| workload_only_throttle              |  -330.196 | 43.60          | 77.33            | 0.2315       | 0.0444             |
| fixed_50                            |  -348.271 | 63.08          | 54.12            | 0.1814       | 0.0352             |
| ppo_lagrangian seed1                |  -348.171 | 63.00          | 54.29            | 0.1845       | 0.0358             |
| ppo_lagrangian seed0                |  -349.431 | 64.05          | 53.44            | 0.1816       | 0.0352             |
| ppo_lagrangian seed2                |  -353.059 | 64.46          | 53.80            | 0.1965       | 0.0381             |
| fixed_20                            |  -446.361 | 144.80         | 26.90            | 0.1431       | 0.0277             |
| fixed_80                            |  -498.635 | 40.82          | 82.00            | 0.2702       | 0.0730             |
| fixed_120                           | -3630.000 | 29.27          | 114.00           | 3.4980       | 0.6726             |

Paired against `risk_aware_rule_based`, 20 seeds, reward:

| treatment                           | mean diff | sem   | as % of the rule |
|-------------------------------------|-----------|-------|------------------|
| ORACLE_lookahead (not deployable)   |   +11.850 | 1.496 | **+3.59%**       |
| ppo_lagrangian seed1                |   -17.975 | 1.492 | -5.44%           |
| fixed_50                            |   -18.075 | 1.459 | -5.47%           |
| ppo_lagrangian seed0                |   -19.235 | 1.568 | -5.83%           |
| ppo_lagrangian seed2                |   -22.863 | 2.201 | -6.92%           |

**PPO loses to the rule by 5.4% to 6.9%, at 8 to 12 standard errors.** The entire
anticipatory headroom on this window, measured by the oracle that reads the next
interval's load from the trace, is +3.6%. PPO therefore sits below the rule by about 1.5
to 1.9 times the whole gap any controller could win. The direction the upper-bound run
predicted holds, with the margin slightly wider on clean data than on the contaminated
window: -2% to -6% there, -5.4% to -6.9% here.

**The learned policy is a constant-rate policy.** This is measured from its actions, not
inferred from its reward. Over five holdout episodes per seed (~4,000 intervals each), the
action it selects is 50 MB/s on 97.1% / 99.9% / 80.9% of intervals, out of the four
available rates {20, 50, 80, 120}. Seed 1 uses only two distinct actions in 3,948 steps.
Consistently, its reward is statistically indistinguishable from the `fixed_50` baseline:
paired reward difference against `fixed_50` is +0.100 (sem 0.065) for seed 1, -1.160 (sem
0.270) for seed 0, -4.788 (sem 2.159) for seed 2. The three seeds did not find three
different policies; they found the same constant, twice exactly and once approximately.

**Mechanism: the latency multiplier saturates.** `lambda_latency` reaches the
`lambda_max = 2.0` bound in 25 / 30 / 41 of 244 iterations and averages 1.98 / 1.95 /
1.98 over the final 50, while the terminal latency violation fraction is 0.051 to 0.066
against a 0.05 budget. The constraint is not satisfied, so the multiplier grows until the
bound stops it, and the best response to a latency price that large is the conservative
constant rate. `lambda_risk` stays near zero throughout (maximum 0.049 to 0.118): the
reliability budget is slack on this window and the safety filter handles it.

**Why the constraint is not satisfiable: the violations are exogenous.** The same 3 of
20 evaluation episodes (seeds 9, 17, 19) fail the latency budget under *every* policy:
`fixed_20` at 26.9 MB/s, which is the slowest rate available, the risk-aware rule, the
lookahead oracle, and all three PPO seeds. The failing sets are identical, not merely
the same size. `latency_budget_met` is 0.85 for every policy in the table except
`fixed_120`. So on those episodes the foreground load alone exhausts the latency budget
and no action in `ACTIONS_MBPS` changes the outcome.

This is what the multiplier cannot see. The dual update observes violation > 0 and
raises the price, with no way to separate "violating because the allocation is too fast"
from "violating because the foreground load leaves no headroom". The agent's only lever
against a rising latency price is to slow down, so it slows down permanently and pays
about 20 hours of completion time and 24 MB/s of throughput for violations it was not
causing. That is the -18 reward almost exactly.

A correction, since an earlier draft of this record's commit message got it wrong:
**there is no latency-infeasibility flag in the simulator.**
`initial_budget_infeasible`, which `baselines/evaluate.py` records per episode, comes
from `violated_risk_budget`, defined at `sim/recovery_env.py:206` as
`not p_best < RISK_BUDGET_EPS`. That is the *reliability* budget being unreachable at
the slowest allocation the filter can reach, a different quantity. Seeds 9 and 17 carry
it and seed 19 does not, so its agreement with the latency failures is a coincidence
across 20 episodes. The exogeneity argument above does not depend on it: every policy
failing the identical episodes is sufficient on its own.

The consequence for the fix is that a feasibility-aware multiplier cannot be built by
reading an existing field. A latency-feasibility signal has to be computed, roughly
"is the budget reachable at the slowest allocation given this interval's foreground
load", which is the latency-model analogue of `safety_filter`'s `p_best`. That is new
simulator surface and is being agreed with A-lane before it is written.

Note which way the bound pushes. The multiplier is pressing *against* `lambda_max` from
below, so removing the bound raises the latency price further and makes the policy more
conservative, not less. The cap did not cause the collapse. But it is binding rather than
slack, which the record previously only anticipated, so `lambda_max` is now a measured
load-bearing choice and not a safe default. See Limits.

**A caveat about what PPO lost to.** On this window `risk_aware_rule_based` and
`workload_only_throttle` are identical on every reported metric, on all 20 seeds
individually: same reward, completion, delivered bandwidth, latency degradation and
exposure. They differ only in `clip_fraction` (0.0296 against 0.0645), meaning the rule's
risk term anticipates the safety filter and so is clipped less often, but never changes
the allocation that is actually applied. On the real held-out window the risk-aware
baseline reduces to its workload throttle. The baseline PPO fails to beat is therefore a
pure workload throttle, which strengthens the negative result and is the same degeneracy
family as the dead `sigma` knob below: the reliability term is inert whenever the fleet is
healthy enough that the budget is slack.

### An upper-bound training run, and why its numbers are not reportable

**Superseded by the clean run above, and kept only as the prediction it made.** It called
the direction and roughly the margin before held-out data existed; that is the only claim
it supports.

Three seeds, 1M environment steps each, `--workload-csv
data_pipeline/profiles/profile_diurnal.csv --hazard-error-sigma 0.25`. Per-seed logs in
`agent/evidence/upperbound-2026-11-06/`.

Training behaved correctly. Latency violation fraction fell from 0.217 / 0.249 / 0.215
at the first iteration to 0.025 / 0.073 / 0.046 at the last, against a 0.05 budget.
`lambda_latency` rose and then decayed towards 0, which is the expected Lagrangian
signature of a constraint becoming satisfied. The multiplier bound never bound: 0 of 122
updates on both multipliers for all three seeds. Entropy held between 1.02 and 1.07.

Evaluation on identical episodes:

| policy                 | reward    | completion (h) | violation fraction | latency budget met |
|------------------------|-----------|----------------|--------------------|--------------------|
| risk_aware_rule_based  |  -215.256 |  40.69         | 0.0259             | 0.850              |
| workload_only_throttle |  -215.256 |  40.69         | 0.0259             | 0.850              |
| ppo_seed2              |  -220.097 |  60.15         | 0.0176             | 0.850              |
| fixed_50               |  -223.499 |  62.99         | 0.0172             | 0.850              |
| ppo_seed1              |  -225.357 |  62.98         | 0.0174             | 0.850              |
| ppo_seed0              |  -228.641 |  66.10         | 0.0183             | 0.850              |
| fixed_20               |  -289.984 | 118.39         | 0.0126             | 0.875              |
| fixed_80               |  -317.961 |  41.10         | 0.0434             | 0.850              |
| fixed_120              | -3699.659 |  29.14         | 0.6958             | 0.000              |

PPO loses to the rule-based baseline on all three seeds and to a fixed 50 MB/s policy on
two of three. It converged to approximately `fixed_50` behaviour: it carries a *better*
violation fraction than the rule (0.017 to 0.018 against 0.026) and still loses, because
it takes 60 to 66 hours against the rule's 40.7 and the reward weights completion
heavily. It never found the faster, riskier operating point the rule exploits.

**This run is contaminated three ways and its numbers must not be published.**

1. Training and evaluation use the same workload window, deliberately. This was the
   design: a negative result under memorisation is conclusive without waiting for
   held-out data, because a clean run can only be worse.
2. The evaluation used a default `SimConfig`, so train-split fleets, not
   `v1_eval_config()`'s held-out split. Accidental.
3. The checkout was 10 commits behind `main`, predating changes to `sim/scenario.py`
   (fleet sampling), `sim/config.py`, `hazard/v1.py` and `sim/stress.py`. Accidental.

The ranking is internally valid because every policy ran against the same contaminated
simulator on identical episodes. The absolute values are not comparable with anything
produced on current `main`. `agent/evidence/upperbound-2026-11-06/README-DO-NOT-REPORT`
repeats this next to the data.

### The hazard-input ablation is degenerate, and `sigma` does not fix it

At `sigma = 0.25` the risk-blind and risk-aware baselines are still identical to six
significant figures (-215.255935 both).

The reason is structural rather than statistical. Both arms observe the *reported*
hazard, and the safety filter computes its floor from the *reported* hazard, so the
filter's output is the same function of state for both and their emitted actions are
identical wherever it binds. `sigma` only changes which failures are actually drawn, and
at a per-episode data-loss probability of order 1e-5 none are drawn in 40 episodes.
There is nothing left for the difference to appear in.

The filter override is the dominant mechanism, not failure rarity: at `sigma = 0` the
filter already clipped the risk-blind policy on 4.5% to 28.5% of steps with identical
outcomes. Unlike rarity, the override does not weaken as `sigma` grows.

A-lane's baseline sweep (`baselines/evidence/`, four runs over `sigma` in
{0, 0.25, 0.5, 1.0} on `profile_diurnal_long` with held-out fleets) establishes this more
strongly than the argument above, on a clean setup and across the whole range rather than
at one value: `episodes.csv` is identical at full floating-point precision across all
four, for every policy and every seed. Their code trace names the reason exactly.
`sim/scenario.py` computes `true_hazard_per_hour = estimated_hazard_per_hour *
multiplier`, and `multiplier` is the only place `hazard_error_sigma` enters.
`true_hazard_per_hour` then feeds exactly one consumer, the per-interval stochastic
failure draw. Observation building (`sim/recovery_env.py::_observation`) and the filter's
exposure and clip computation (`sim/safety_filter.py`) both read
`estimated_hazard_per_hour`. So no policy and no filter can observe the noise at all, and
the single remaining channel, a failure actually firing, produced `terminal_failure = 0`
in every one of the four sweeps.

This means the degeneracy is stronger than "the ablation shows nothing": `sigma` is
currently unobservable through the entire pipeline by construction, not merely
underpowered. The one paired difference that survives is `clip_fraction`
(-0.0456, sem 0.023): the risk-aware policy does request differently, and the filter
erases the difference before it reaches any outcome.

So an input-side ablation cannot measure what it was specified to measure, for any
`sigma`. Replacement, in priority order:

- **(a)** Report the degeneracy itself: under a correctness-preserving safety filter,
  the hazard observation contributes nothing measurable. Costs nothing and is a real
  result about the architecture.
- **(b)** Ablate the *filter* rather than the input, so the policy must learn the
  constraint instead of being handed it. This isolates where the safety comes from. New
  code; `--ablate-hazard` masks observation fields 5 and 6 only.
- **(c)** The stress configuration at 100x hazard, where failures are frequent enough to
  score. Deprioritised: it measures a fleet condition that is not the real one.

### Workload window contamination

All four profiles are single 24-hour windows at the 300 s sampling interval, per the
Phase-2 plan's B5. Episodes run to `MAX_EPISODE_STEPS = 2400`, which is 200 hours, so
the simulator tiles one day more than eight times in a long episode. A 24-hour window is
288 steps, so the tiling has exact period 288 and the next interval's load sits 288
steps back in the agent's own history. Any anticipation result on these traces is
contaminated by construction, and a second 24-hour window would not fix it.

The fix is one continuous 7 to 14 day window per profile. 14 days rather than 7 because
the longest episodes are about 8.3 days. It also yields a time-forward train/test split
for free, which A2 could not provide on the drive data.

B-lane has since built this: `data_pipeline/profiles/profile_diurnal_long.csv`, 4,032
rows covering 2020-07-07 to 2020-07-21, aggregated across the 44 disks sharing that
window, with `profile_diurnal_long.meta.json` carrying the disk list, window bounds and
a suggested time-forward split (first 2,880 rows train, last 1,152 held out; nothing
reads the split automatically). Scale matches the existing 24-hour profile to within 1%
(mean 69,917.9 against 69,075.5 KB/s), and p95 99.79 MB/s and max 116.76 MB/s both sit
inside `ACTIONS_MBPS`. The four existing profiles are byte-identical. The real re-run
uses this file, not `profile_diurnal.csv`.

## Verdict

Constrained PPO, as currently configured, does not beat a risk-aware rule-based baseline
on the recovery-bandwidth problem. This is now measured on a held-out window with
held-out fleets, not inferred from a setting rigged in its favour.

Two numbers frame it, both on `--eval-fleet` with `sigma=0.25` and 20 paired seeds:

- The **headroom**: the lookahead oracle beats `risk_aware_rule_based` by 3.59% on the
  held-out half (+11.85, sem 1.50, against the rule's -330.2;
  `agent/evidence/clean-holdout-2026-11-06/`) and by 5.2% on the full long profile
  (+15.25, sem 1.90, against -293.3; `baselines/evidence/diurnal-long-sigma0.25/`).
  These are two windows, not a contradiction: the 4-day holdout half is the harder of
  the two for an anticipator. Either way the ceiling is single-digit percent, and the
  oracle that reaches it reads the future and is not deployable. An earlier figure of
  about 10.4% came from the tiled 24-hour window and is superseded by both.
- The **result**: PPO comes in 5.4% to 6.9% *worse* than the rule, at 8 to 12 standard
  errors, having converged to a near-constant 50 MB/s allocation indistinguishable from
  the `fixed_50` baseline.

So the method is below the baseline by more than the entire gap any controller could
win, and the policy it learned is not a controller. On the real held-out window the
baseline it fails to beat is itself only a workload throttle, since the rule's risk term
never changes an applied allocation there.

This is reported as a finding, not a failure. It is not a general claim about PPO:
hyperparameters are untuned, one architecture, one profile family, and the saturating
latency multiplier identifies a specific fixable cause rather than an exhausted search.
What it does establish is that the problem as specified has little room for learned
anticipation and that a tuned constrained-RL controller has to clear a bar set by a
throttle, which the Phase-1 deck commits to reporting honestly either way.

Every number above comes from committed evidence under `agent/evidence/` and
`baselines/evidence/` and is reproducible from the commands recorded beside it. The
contaminated upper-bound run remains excluded.

## Limits

- `lambda_max = 2.0` **binds in the clean run and is therefore load-bearing, not a safe
  default.** It is reached in 25 / 30 / 41 of 244 iterations and `lambda_latency` averages
  1.95 to 1.98 over the final 50, so for roughly the last third of training the latency
  constraint is enforced by the bound rather than by the multiplier. The direction is
  unfavourable to a rescue: the multiplier presses against the bound from below, so
  raising or removing it makes the latency price larger and the policy more conservative.
  The interesting sweep is therefore *downward*, and the honest reading is that the
  reward's latency term and the 0.05 violation budget are jointly too expensive on this
  window for any multiplier value to produce a non-degenerate policy. Untested.
- `ent_coef_final = 0.01` is still an untuned single-seed choice. Terminal entropy is 0.85
  to 0.94 nats against 1.386 for uniform, so the anneal is not what froze the policy: it
  retains stochasticity and still chooses one action almost always, which locates the
  cause in the value of the actions and not in exploration pressure. Both knobs are
  CLI-exposed; neither has been swept.
- **The result is one window and one action space.** `ACTIONS_MBPS` offers four rates and
  the rule's mean delivered bandwidth is 77.3 MB/s, between 50 and 80, so the baseline
  wins partly by alternating across a discretisation the policy collapsed inside. A finer
  action space is an untested confound, not a dismissal: it would change the margin
  without changing the direction, since the oracle on the same four actions only reaches
  +3.6%.
- `DEFAULT_ROLLOUT_STEPS = 8192` is a compromise, not an optimum: 16384 would hold about
  25 episodes at uniform-random allocation but halves the number of policy updates in a
  fixed step budget.
- The derivation uses the bootstrap's *point* estimate of the mean absolute log residual
  (0.2342). The bootstrap *mean* is higher (0.2648), which through the same two steps
  gives `sqrt((0.2648/0.798)^2 - 0.183^2) = 0.28`. So the defensible range is about 0.23
  to 0.28 depending on which estimator is used, and 0.25 sits inside it rather than
  falling out of it uniquely. This is a further reason to sweep rather than fix.
- The 0.234 and 0.183 figures are bin-level. Per-drive estimation error is unmeasurable
  in principle, because each drive contributes one Bernoulli draw, so its residual is
  dominated by outcome noise rather than model error. 0.25 is a bin-level number used as
  a per-drive proxy, which is why real runs sweep it.
- v1's calibration error is asymmetric in the dangerous direction: it over-predicts in
  the low bins (conservative) and under-predicts in the top bin, which holds 4,051 of
  the positives (mean prediction 0.015493 against an observed rate of 0.017751). A
  symmetric lognormal `hazard_error_sigma` does not represent this.
- The latency budget is met on only 85% of evaluation episodes for nearly every policy,
  including the baselines. Whether the 5% budget is satisfiable under the default
  profile mix has not been settled, and it is a stated limit of the environment rather
  than a property of any policy.
- The full suite on a checkout without `hazard/artifacts/v1/` reports 1,673 passed and
  29 skipped. The README's pinned count must be re-measured on a checkout that has the
  artifacts; it cannot be re-pinned from this environment.
