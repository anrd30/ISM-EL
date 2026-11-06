# T decision record: constrained PPO, the estimation-error parameter, and a negative upper bound

Status: proposed. Not yet reviewed by the other lanes.

This is T-lane's first decision record. It covers the three PPO defaults changed after
the pilot runs, the choice of `hazard_error_sigma`, one training run that is explicitly
**not reportable**, and a finding that invalidates the hazard-input ablation as
originally specified.

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

### An upper-bound training run, and why its numbers are not reportable

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
on the recovery-bandwidth problem, in a setting rigged in its favour.

How much headroom exists at all is now measured on a clean setup rather than inferred
from the contaminated one. On `profile_diurnal_long` with held-out fleets, 20 paired
seeds (`baselines/evidence/diurnal-long-sigma0.25/`), the lookahead oracle beats
`risk_aware_rule_based` by 15.25 reward (sem 1.90) against the rule's -293.3, which is
**5.2%**. An earlier measurement on the tiled 24-hour window gave about 10.4%; that
window is contaminated by the 288-step period, so 5.2% supersedes it.

So the entire anticipatory headroom available to any deployable controller is about 5%,
and the oracle reaching it is not deployable. PPO came in 2% to 6% *worse* than the rule.

This is reported as a finding, not a failure. It is not yet a general claim about PPO:
hyperparameters are untuned, the budget was 1M steps, one architecture, one profile. But
closing a 10% gap that the method currently moves the wrong way is a high bar, and the
Phase-1 deck commits to reporting a falsification honestly if it occurs.

No number in this record may be published until the clean re-run on a held-out window
with committed `sigma` provenance replaces the upper-bound run.

## Limits

- `lambda_max = 2.0` and `ent_coef_final = 0.01` are untuned single-seed choices. Both
  are now CLI-exposed so they can be swept, and neither has been.
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
