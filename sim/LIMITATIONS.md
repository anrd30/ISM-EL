# Simulator limitations

This file is the single place the report's limitations section draws from for
the recovery simulator in `sim/`. Every number in the simulator is a design
decision, not a measurement. Everything the simulator produces (rebuild times,
allocated bandwidth, latency during a rebuild, outcomes under alternative
actions) is simulated and must be labelled so wherever it is reported. Module
docstrings in `sim/` point here.

## 1. The latency model is uncalibrated

Latency is `latency_unloaded_ms / (1 - rho)` with
`rho = (fg_bw + alloc_effective) / BW_TOTAL_MBPS`. This is a single-queue
idealisation. It ignores queue-depth limits, caching, read/write asymmetry,
RAID geometry, device garbage collection and burstiness within a control
interval, and it has not been compared with any measurement of any device.
Its convexity is a deliberate modelling choice (a linear model removes the
load dependence of the trade-off); the numbers it produces (where latency
becomes unacceptable, how steeply it rises, which profile violates under which
action) are properties of the formula, not findings about a real system.

The constants `latency_unloaded_ms = 2.0`, `latency_target_ms = 15.0` and
`rho_max = 0.98` are design decisions. With them the latency limit is
`1.5 * 15.0 = 22.5 ms` and a violation begins at `rho = 1 - 2/22.5`, about
0.911. Whether a given action satisfies the latency budget on a given profile
depends directly on that threshold, so every violation-rate figure is
conditional on these constants. The target was raised from 5 ms and then 8 ms
because at those values the `peak` profile had no action inside the budget;
15 ms is the smallest round value at which every profile admits one *at a
healthy fleet*. Under the default randomised fleet `peak` admits none (section
13).

## 2. The service-level target constrains the mean, not the tail

The target, the limit and the reward terms are defined on mean latency. The
response time is modelled as exponential about its mean, so `p95 = 3.0 x mean`
and `p99 = 4.6 x mean`. The p95 passes the 15 ms target once the mean latency
is above 5 ms (`rho` 0.6), and at the violation threshold (mean 22.5 ms) the
p99 is about 104 ms. A mean-based target can be met while the tail is poor. The
percentiles are reported in `info` only and are not constrained.

## 3. Rebuild speed is unchanged when a survivor is lost

When a surviving drive fails, `redundancy_left` falls and the failed drive
stops contributing to the hazard aggregate, but the rebuild continues at the
same rate and the second failed drive is not itself rebuilt. In a real array a
rebuild reads from fewer sources and slows down. Rebuild time after a second
failure is therefore optimistic, and so is the exposure window.

## 4. The safety filter acts on nominal, not delivered, exposure

The filter computes the contract's `P = 1 - exp(-hazard * T_remaining)` with
`T_remaining` taken at the candidate allocation. The rebuild actually receives
`min(applied, headroom)`, which is lower whenever foreground load leaves less
headroom than the allocation, so the exposure at the delivered rate is higher.

This is an information constraint, not an approximation. The simulator has no
look-ahead (section 5): at decision time the foreground load of the interval
about to run is genuinely unknown, so the delivered rate cannot be known and
the nominal allocation is the only thing the filter can bound. The gap is made
visible rather than hidden: `info["p_failure_at_delivered_rate"]` reports the
exposure at the delivered rate for each step. On the `peak` profile the gap is
large, because the headroom is far below the largest allocation.

## 5. No look-ahead: the observation lags by one interval

The reset observation shows the trace row at the episode's start offset with no
rebuild allocation. Step `t` executes the next row, and everything in the
observation it returns (foreground load, latency, headroom, delivered
allocation) was measured over the interval just executed. The controller never
sees the load of an interval before choosing its allocation. This is a stated
information constraint of the environment, not a modelling oversight. It
bounds what any policy can learn: a policy cannot anticipate a burst that
begins in the interval it is about to act in, however good its function
approximator. The reward and the `info` latency terms use the same interval as
the observation.

## 6. Utilisation is clamped at `rho_max`; the rebuild stalls when foreground saturates

`alloc_effective = min(applied, bw_available)` with
`bw_available = max(0, rho_max * BW_TOTAL_MBPS - fg_bw)`. When foreground
traffic alone is at or above `rho_max * BW_TOTAL_MBPS` (196 MB/s) the headroom
is zero, the rebuild receives no bandwidth and makes no progress, and `rho` is
clamped at `rho_max` so latency stays finite (100 ms with the defaults). The
unclamped formula would reach or exceed `rho = 1`. In that regime the episode
can run to the horizon without completing. Nothing in the simulator models
foreground throttling to relieve it.

## 7. `C` is constant, so `beta * C` is a per-step time cost

`C` (degraded-state cost) is 1 in every interval of every episode: the group is
degraded from the first interval until the rebuild completes. Within a step
`beta * C` is therefore a constant and carries no information about the action
chosen in that step. It does work through episode length: a shorter rebuild
accumulates fewer degraded intervals, so over an episode `beta * C` is the
incentive to finish sooner, which is the exposure incentive of the reward. It
is not a per-action signal.

## 8. Data loss is unobservable in default-configuration runs

With the v0 hazard the probability of two further failures inside one rebuild
is of order 1e-5 per episode. In 3000 default-configuration episodes under each
of constant-20 and constant-120 there were zero terminal failures, which is not
an estimate of the loss rate. The reward has no data-loss term, so reliability
enters the controller only through the safety filter.

Any reported loss figure must come from a stress run or a rare-event method
and must be labelled as such, never quoted as a default-configuration result.
`sim.stress.stress_config()` multiplies every drive's cumulative hazard by
`STRESS_HAZARD_MULTIPLIER` (100) and sets `info["config_label"]` to
`stress_hazard_x100`; its only purpose is to make terminal failures frequent
enough to count. At that hazard every action violates the risk budget, so the
filter applies the largest allocation whatever the controller requests: a
stress run measures loss under the filter's degradation rule, not how loss
depends on the controller's choices. The multiplier is not derived from any
fleet.

`stress_config()` with no `base` stays pinned to the v0 hazard and its feature
function (`sim.config.v0_config()`) on purpose, independent of the `SimConfig`
default (now the fitted v1): the figures above were measured with v0, and the
stress configuration is defined on that hazard.

## 9. Workload

Foreground load is a synthetic aggregate from `workload.generate` (or a replay
CSV whose provenance is read from its sidecar; `workload_synthetic` is `None`
when there is no sidecar). The trace is tiled cyclically, so the same day
repeats back to back within an episode; real load would not repeat. Episodes
start at a random trace row by default.

The `peak` profile loads the budget heavily (mean foreground `rho` about 0.74,
p95 about 0.80). With the defaults and a healthy fleet, the fraction of
intervals violating the latency limit by constant action (mean over 20 seeds;
simulated, uncalibrated), measured under the v1 default
(`python -m sim.measure_defaults`; `--hazard v0` reproduces the figures this
table replaces):

| Profile | 20 MB/s | 50 MB/s | 80 MB/s | 120 MB/s | Actions within the 5% budget |
|---|---|---|---|---|---|
| idle | 0.000 | 0.000 | 0.000 | 0.000 | 4 |
| diurnal | 0.000 | 0.000 | 0.024 | 0.497 | 3 |
| bursty | 0.004 | 0.026 | 0.084 | 0.123 | 2 |
| peak | 0.027 | 0.966 | 1.000 | 1.000 | 1 |

The mean table is almost unchanged from v0 (only `peak` at 20 MB/s moves,
0.026 to 0.027): this measurement forces a constant action, and the latency
model itself never reads the hazard model. The worst seed does move: `peak`
at 20 MB/s reaches 5.3% under v1, over the 5% budget on that one seed (3.8%
under v0); `bursty` at 50 MB/s stays 4.4% under either. The cause is the
safety filter, not the latency model -- under v1 it clips 0.1% to 0.2% of
steps at 20 MB/s on every profile (v0 clips none at these seeds), raising the
delivered allocation above the nominally requested one on the affected
episodes, consistent with v1's generated fleets spreading wider in hazard
than v0's did at the same level (`hazard/A5_DECISION.md`).

Every profile admits at least one action inside
`LATENCY_VIOLATION_FRACTION_MAX = 0.05` on the mean, and the count falls as
workload intensity rises. The margins are not large, and are tighter under
v1 specifically on `peak`'s worst seed, so the latency budget is sensitive to
`latency_target_ms`, to the profile shapes (synthetic, p95 set by scaling),
and now also to which hazard model is in use.

The table and the "actions within the budget" column describe a **healthy
fleet**. They do not hold under the default randomised-fleet configuration; see
section 13.

**Real profiles come from one 24-hour window and repeat with period 288.** The
real-derived profiles are all drawn from a single 24-hour window (2020-08-06
04:00 UTC) and, as the trace is tiled cyclically, repeat with exact period 288
rows (one day at 300 s). A policy evaluated on them could in principle recover
next-interval load from the same time one day earlier in its own history. Any
advantage measured over a forecasting oracle on these profiles therefore bounds
what *memorisation* of that window could achieve, not what *generalisation*
could. A held-out window is needed to separate the two; until one is used, no
claim of generalisation to unseen workloads may rest on the real profiles. The window and period are
measured: all four profiles in `data_pipeline/profiles/` have 288 rows with
`t_seconds` starting at 0 and a uniform 300 s step, and the sidecar
`profile_bursty.meta.json` records `window_start_utc = 2020-08-06T04:00:00Z`
(`window_start_ts = 1596686400`), `synthetic = false`, `n_disks = 36`.

## 10. The latency term of the reward is silent over most of the operating range

With `latency_target_ms = 15.0` and an unloaded latency of 2 ms,
`D = max(0, (L - L0) / L0)` is zero until `rho` exceeds `1 - 2/15`, about
0.867, and the violation threshold is at 0.911. The latency term is therefore
zero over most of the operating range and informative only in the narrow band
between 0.867 and 0.911, followed by the `gamma * V` cliff.

This is a shaping consequence of the chosen constants, not a defect. The
constant `beta * C` supplies time pressure to finish the rebuild, so the latency
term only has to push back near saturation. The controller this shapes
rebuilds as fast as possible while staying below roughly 0.87 utilisation,
which is a sensible policy. It also means that the region in which the hazard
input decides anything (the choice of how fast to rebuild when the risk is
elevated) is largely the region in which latency is not binding, which is where
the risk signal earns its place. A reader should not take the flat latency
reward for an error, and should not expect a learned policy to show graded
latency awareness below 0.87.

## 11. Fleet and failure process

- Drive features are fixed for the whole episode; a real drive's counters
  would change over the 28 to 167 hours of a rebuild.
- The default hazard model is the fitted v1 (`hazard/v1.py`), fed by an
  empirical generator (`sim.scenario.v1_drive_features`) that draws a drive's
  model name and its 28 SMART value columns from real training-split values
  stored in `sim/fleet_features_v1.json`. The heuristic v0 remains available
  and reproducible by passing it explicitly with its own feature function
  (`sim.config.v0_config()`). The generator samples each column independently
  from that column's real marginal distribution (healthy or failure-day
  values, chosen with probability equal to the drive's degradation level), not
  jointly, so it does not reproduce the correlation between a drive's columns
  on the same day (for example between a level and its own 7- and 30-day
  differences); a drawn feature vector need not match any real drive. The
  model of each drive in a group is drawn independently, so a group mixes drive
  models, which a real group usually does not.
- Per-drive failures are independent given the features. Correlation enters
  only through the shared fleet degradation level that sets the features, so
  the contract's independence limitation (it makes the group look safer than
  it is) applies to the hazard aggregate and the uncertainty.
- The hazard is constant across the episode (the contract's constant-rate
  assumption).
- When a loss-causing failure and rebuild completion fall in the same interval,
  the failure is resolved first and the episode ends as data loss.
- The fleet degradation level distribution and per-drive spread are design
  decisions chosen so that the group-hazard observation spans its range; they
  are not fitted to any fleet.

## 12. Observation fields 0 and 1 (foreground IOPS) are inert

`fg_read_iops` and `fg_write_iops` are in the observation vector, but the
latency model is `rho = (fg_bw + alloc) / BW_TOTAL_MBPS`: bandwidth only. IOPS
never enters the dynamics, the reward, the termination rule or the safety
filter. Those two fields therefore cannot affect any decision the environment
makes, and a policy cannot learn to use them in any way that changes an outcome.

**No claim may be made that the controller responds to foreground IOPS.** A
policy that nominally receives them is, as far as the environment's consequences
go, ignoring them.

Two consequences:

- Their normalisation scale (20,000, `contracts.OBS_SCALES`) is far below real
  aggregate values. A measurement on the real profiles found read IOPS of 88k to
  118k and write IOPS of 95k to 265k on the bursty profile, and the idle profile
  exceeding the scale on 13% of rows. Both fields therefore sit clipped at 1.0
  much of the time.
- That clipping currently costs nothing precisely because the fields are inert.
  Re-scaling would not help until something consumes them: a model in which
  IOPS (for example through request size or queue depth) affects latency would
  have to be added first, and the scale chosen with real aggregates in view.

## 13. The feasibility ladder is a healthy-fleet statement

The documented 4/3/2/1 count of latency-feasible actions for
idle/diurnal/bursty/peak (section 9) holds when the drive fleet is healthy.
Under the default randomised-fleet configuration, measured over 40 seeds on
both synthetic and real-derived profiles, **`peak` has zero feasible actions**:
the safety filter raises the allocation on degraded fleets, and the latency
limit is then exceeded whatever the policy requests.

This is not a synthetic-versus-real divergence: both sources behave the same
way. It also means the statement in section 1 that "every profile admits one"
action inside the budget is conditional on a healthy fleet and on the constants
there; it is not a property of the default configuration. All figures here are
simulated and conditional on the uncalibrated latency model of section 1.

## 14. Latency and reliability are jointly infeasible on `peak` with a degraded fleet

This follows from section 13 and from the contract's stated precedence:
reliability (the risk budget, enforced by the safety filter) is the hard
constraint and latency is the objective. Under the default fleet, no policy met
the 5% latency budget (`LATENCY_VIOLATION_FRACTION_MAX`) in more than 20% of
`peak` episodes.

This is the precedence rule behaving as specified, not a defect. Its
consequence for reporting: **`peak` results under a degraded fleet measure the
safety filter more than the policy**, because the filter overrides the
requested allocation. They must not be read as a comparison between policies.
A comparison of policies on `peak` needs a healthy or pinned-healthy fleet
(`SimConfig.fleet_degradation`), reported as such.

## 15. Hazard estimation error is implemented but switched off

**What it is.** `SimConfig.hazard_error_sigma` is the log-space standard
deviation of a multiplicative error between the risk model's reported per-drive
hazard (the estimate) and the hazard failures are drawn from (the truth). The
true hazard of a drive is `estimate * exp(N(-sigma^2/2, sigma^2))`, drawn once
per drive per episode from its own random stream. The observation, the safety
filter and the exposure it bounds see the estimate; only the failure draw uses
the truth. The multiplier is lognormal because hazards are positive and span
orders of magnitude, and its mean is exactly 1 (the `-sigma^2/2` term), so the
error adds dispersion without shifting the fleet's mean hazard. It is drawn per
drive per episode because a model's error on a drive is a property of that
drive's features, which are fixed for the episode; fresh per-interval noise
would average out over a rebuild's hundreds of intervals. `info` carries
`hazard_reported_per_hour`, `hazard_true_per_hour`, `hazard_realised_ratio`,
`hazard_error_sigma`, and the exposure at the true hazard
(`p_failure_true_at_applied_rate`, `p_failure_true_at_delivered_rate`,
`violated_risk_budget_true`). The contract keys `p_failure_before_complete` and
`violated_risk_budget` remain defined on the estimate.

**Why it exists.** With truth equal to the estimate, a filter acting on the
estimate is exactly protective, so there is nothing left for a policy to get
right or wrong about the reliability constraint. Measured on the simulator with
the parameter at zero:

- A policy built to ignore the hazard fields entirely produced outcomes
  indistinguishable from a risk-aware one: paired difference -0.004 (sem 0.010)
  over 40 seeds on real workload profiles, with 2 of 40 episodes differing at
  all.
- The filter clipped the risk-blind policy on 4.5%, 11.9% and 28.5% of steps on
  three profiles, and the outcome was unchanged regardless.
- In a PPO smoke run, episode exposure sat just below the budget (4.78e-3 to
  4.98e-3 against 5e-3) whatever the policy did, so the reliability Lagrange
  multiplier never left zero and the learner received no gradient on that
  constraint.

**Current state: off.** The default is `0.0`, which reproduces the simulator
without estimation error bit for bit (checked against recorded trajectories,
including survivor failures and a terminal failure). No non-zero value is
chosen anywhere in the code. Its tag is `[ASSUMED]`, the only tag that fits a
constant that has no source.

**Consequence of leaving it at zero.** The hazard ablation cannot detect
anything, because the hazard input cannot matter to an outcome the filter
already guarantees, and the reliability constraint is inert, because the
filtered policy meets the budget by construction. A result showing "no effect of
the hazard input" or "constraint satisfied" from a zero-sigma run is a property
of the simulator, not evidence about a risk model or a policy.

**Do not pick a value.** The right sigma is the dispersion of the log ratio
between a fitted risk model's estimated and realised hazard, and that has to be
derived from a measured calibration curve of that model. Switching the
parameter on with a round number would substitute one invented figure for
another, and every result downstream would inherit it. Even a derived value
describes one model on one data window, and the multiplier family (lognormal,
independent across drives, identical for every drive and hazard level) is itself
an assumption: real calibration error is typically larger at the extremes of
the predicted range and correlated across drives that share a batch. Nothing in
this section makes the simulator calibrated, and no result obtained with
a positive sigma is validated by it.
