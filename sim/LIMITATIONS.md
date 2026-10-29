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
15 ms is the smallest round value at which every profile admits one.

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

## 9. Workload

Foreground load is a synthetic aggregate from `workload.generate` (or a replay
CSV whose provenance is read from its sidecar; `workload_synthetic` is `None`
when there is no sidecar). The trace is tiled cyclically, so the same day
repeats back to back within an episode; real load would not repeat. Episodes
start at a random trace row by default.

The `peak` profile loads the budget heavily (mean foreground `rho` about 0.74,
p95 about 0.80). With the defaults and a healthy fleet, the fraction of
intervals violating the latency limit by constant action (mean over 20 seeds;
simulated, uncalibrated) is:

| Profile | 20 MB/s | 50 MB/s | 80 MB/s | 120 MB/s | Actions within the 5% budget |
|---|---|---|---|---|---|
| idle | 0.000 | 0.000 | 0.000 | 0.000 | 4 |
| diurnal | 0.000 | 0.000 | 0.024 | 0.497 | 3 |
| bursty | 0.004 | 0.026 | 0.084 | 0.123 | 2 |
| peak | 0.026 | 0.966 | 1.000 | 1.000 | 1 |

Every profile admits at least one action inside
`LATENCY_VIOLATION_FRACTION_MAX = 0.05`, and the count falls as workload
intensity rises. The margins are not large: `peak` at 20 MB/s reaches 3.8% on
its worst seed and `bursty` at 50 MB/s 4.4%, so the latency budget is
sensitive to `latency_target_ms` and to the profile shapes, which are
synthetic and whose p95 is set by scaling.

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
- The scenario generator emits only the five SMART features read by the v0
  hazard model. A replacement risk model that reads other attributes would see
  them as missing.
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
