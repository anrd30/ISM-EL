# Frozen Interface Contract

## 1. Purpose and sign-off rule

This document is the human-readable spec for `contracts/contracts.py`. It fixes the interfaces that four workstreams code against: the RL agent and simulator, the statistical risk model, the data pipeline, and the evaluation/report lane.

| Item | Value |
|---|---|
| `CONTRACT_VERSION` | `1.5.0` |
| Freeze point | End of week 1 |
| Change rule | Changeable only with the sign-off of all four teammates, recorded in the changelog at the bottom of this file |

Nothing described here has been measured except where a constant is tagged `[MEASURED]` below. Every other numeric value is a **design decision**, and every statement about behaviour describes what the code is required to do, not what it has been shown to do.

**Provenance convention.** Every numeric constant in `contracts/contracts.py` (and in `sim/config.py`) carries exactly one tag in its comment: `[MEASURED]` (taken from data, source named), `[DERIVED]` (computed from other constants or a stated model, derivation shown or referenced) or `[ASSUMED]` (chosen by design, never checked against data). Several of this project's constants were set by reasoning before any data existed, and a wrong constant does not fail a test: it sits in a file looking plausible while the suite passes over it. The tag turns "which numbers have never been checked against data" into a grep. A tag records origin, not correctness; nothing here implies the simulator or its latency model has been validated.

## 2. Observation vector

Twelve fields, in this order, never reordered. The Python class is `Observation`; the tuple of names is `OBS_FIELDS`; the dimension is `OBS_DIM = 12`.

| Index | Name | Unit | Normalisation scale | Supplied by |
|---|---|---|---|---|
| 0 | `fg_read_iops` | IOPS | 20000.0 | Data pipeline |
| 1 | `fg_write_iops` | IOPS | 20000.0 | Data pipeline |
| 2 | `fg_bw` | MB/s | 200.0 (= `BW_TOTAL_MBPS`) | Data pipeline |
| 3 | `latency_now` | ms | 50.0 | Simulator |
| 4 | `latency_target` | ms | 50.0 | Simulator |
| 5 | `hazard` | per hour, group-level aggregate | 1.8e-4 (= `(GROUP_SIZE - 1) * 2e-5`) | Risk model |
| 6 | `hazard_uncertainty` | per hour, group-level aggregate | 1.8e-4 (= `(GROUP_SIZE - 1) * 2e-5`) | Risk model |
| 7 | `redundancy_left` | count | 2.0 (= `GROUP_TOLERATED_FAILURES`) | Simulator |
| 8 | `progress` | fraction | 1.0 | Simulator |
| 9 | `work_remaining` | GB | 12000.0 | Simulator |
| 10 | `alloc_current` | MB/s | 120.0 (= `max(ACTIONS_MBPS)`) | Simulator |
| 11 | `bw_available` | MB/s | 200.0 | Simulator |

**Fields 0 and 1 are inert.** The simulator's latency model is bandwidth-only (`rho = (fg_bw + alloc) / BW_TOTAL_MBPS`), so `fg_read_iops` and `fg_write_iops` never enter the dynamics, the reward, the termination rule or the safety filter, and a policy cannot learn to use them. Their scale (20000, `[ASSUMED]`) is also far below real aggregate values (a measurement on real profiles found 88k to 118k read and 95k to 265k write IOPS on the bursty profile, and the idle profile above the scale on 13% of rows), so both fields are clipped at 1.0 much of the time; that costs nothing only because the fields are inert. **No claim may be made that the controller responds to foreground IOPS.** See `sim/LIMITATIONS.md` section 12.

**Starting value of `redundancy_left`.** A rebuild episode begins after one drive has failed, so it **starts at `redundancy_left = 1`**. A second failure takes it to 0, and a third failure is data loss. `redundancy_left = 2` is the healthy state and **never occurs during an episode**; it exists in the valid range `[0, 2]` and as the normalisation scale so that the field means the same thing outside a rebuild.

`Observation.to_vector(normalized=True)` returns a `float32` array of shape `(12,)`. Exact semantics:

| Call | Result |
|---|---|
| `normalized=False` | Raw values in index order, **unclipped** |
| `normalized=True` | `clip(value / scale, 0, 1)`: a negative value becomes `0.0`, a value above its scale saturates at `1.0` |

`to_vector` **does not validate**. NaN, inf and negative values pass through it (NaN stays NaN, inf saturates under clipping or passes through when raw). `Observation.validate()` is the only gate for them and must be called explicitly.

`Observation.from_vector` multiplies back by the scales and raises `ContractViolation` unless the input has exactly 12 entries. Because of the clip, the round trip is lossy for raw values above a scale, which is intended. Values are computed in float64 and cast to float32, so an exact round trip holds only for float32-representable values; realistic decimals match to about 1e-6 relative.

`Observation.validate()` raises `ContractViolation` on: NaN or inf in any field; any negative value; `progress` outside `[0, 1]`; `redundancy_left` outside `[0, 2]`; `latency_target <= 0`.

## 3. Why normalisation scales are frozen constants

The scales are literal constants in `contracts.py`. They are never computed from a dataset. A scale fitted to data (a maximum, a percentile, a standard deviation) would carry statistics of the training traces into the policy input, so a policy evaluated on a different trace split would be seeing inputs scaled by information from the training split. A frozen constant removes that channel, and keeps the input scaling identical across the agent, the simulator, evaluation and any later retraining. Where a scale equals another constant (`BW_TOTAL_MBPS`, `GROUP_TOLERATED_FAILURES`, `max(ACTIONS_MBPS)`) it is defined from that constant.

The cost is that values above a scale saturate at 1.0. That is accepted.

**Hazard scale `1.8e-4`, and what saturation means.** Fields 5 and 6 are **group-level aggregates**: the sum of the per-hour hazards of the surviving drives (section 8). The scale is chosen from the design, not from data. The baseline per-drive hazard is 2.41e-6 per hour (section 9), so with nine survivors the baseline group hazard is 9 x 2.41e-6 = 2.17e-5 per hour. The budget `RISK_BUDGET_EPS = 5e-3` stops being satisfiable by the fastest action (120 MB/s) at about 8.3x baseline, which is 8.3 x 2.17e-5 = 1.8e-4 per hour for the group; per drive that is 8.3 x 2.41e-6 = 2.0e-5, and `(GROUP_SIZE - 1) * 2e-5 = 1.8e-4`. Setting the scale there makes field 5 saturate exactly where every action violates the budget and the controller has no choice left to make. `contracts.py` checks at import that the scale equals `(GROUP_SIZE - 1) * 2e-5`, so a change to the group geometry fails loudly instead of silently invalidating it.

**Stale inputs.** The baseline hazard (2.41e-6 per hour), the 8.3x multiple and the scale arithmetic above all rest on the 2.11% AFR from the earlier Q1-2022 three-model extract. The project's data window has since moved to full-year 2022 with four models, and the rate there is 1.95%. These figures are stale pending re-derivation, which has not been done in this document; nothing above has been recomputed.

| Hazard multiple of baseline | Group hazard (per hour, 9 survivors) | Normalised field 5 | Meaning |
|---|---|---|---|
| 1x | 2.17e-5 | 0.12 | Baseline; all four actions feasible |
| 1.4x | 3.0e-5 | 0.17 | Constraint first binds (20 MB/s infeasible) |
| 3.5x | 7.5e-5 | 0.42 | 50 MB/s becomes infeasible |
| 5.5x | 1.2e-4 | 0.66 | 80 MB/s becomes infeasible |
| 8.3x and above | 1.8e-4 and above | 1.00 (saturated) | All-actions-violate regime |

The decision region therefore spans roughly 0.12 to 1.0 of the field's range. Because the 9x group aggregation moves the baseline and the infeasibility thresholds together, the normalised values are the same as for a per-drive field with scale `2e-5`.

**Why not a looser scale.** A looser scale such as `1e-4` per drive-equivalent (`9e-4` for the group) would put the same decision region at about 0.024 to 0.20 of the field's range, so most of the input's resolution would sit in the all-actions-violate region where the controller has no choice. The scale is tight so that the decision region occupies most of the range and saturation coincides with the all-actions-violate threshold (about 8.3x baseline hazard). This is design arithmetic, not a measurement.

**Accepted limitation.** A group hazard at or above 1.8e-4 per hour reads as 1.0, and the safety filter is then in its all-actions-violate regime, so the policy cannot distinguish degrees of severity beyond that point. This is a limitation of the fixed scale and belongs in the report's limitations section. The multiples above depend on the baseline AFR and the 9-survivor arithmetic of section 9; they are design arithmetic, not measurements. With fewer than nine survivors the aggregate is a sum over fewer terms, so the same normalised value corresponds to a higher per-drive hazard.

## 4. Action set

`ACTIONS_MBPS = (20, 50, 80, 120)`, `N_ACTIONS = 4`. `action_to_mbps(index)` maps an index to MB/s and raises `ContractViolation` for a non-integer (including `bool`) or out-of-range index. Any environment that receives an out-of-range or wrong-typed action must raise `ContractViolation` (by routing the action through `action_to_mbps`), never a bare `ValueError` or `IndexError`, so a conformance harness can require the named exception instead of accepting any failure.

**This is a prototype set.** The values are illustrative and are **not measured device limits**. Any result reported against this set must say so.

## 5. Reward

```
reward = -(alpha * D + beta * C + gamma * V)
```

| Term | Meaning | Source of the value |
|---|---|---|
| `D` | Latency degradation, `max(0, (L - L0) / L0)` with `L = latency_now`, `L0 = latency_target` (`latency_degradation`) | Simulator |
| `C` | `1.0` for each control interval in which the group is degraded (at least one drive failed, rebuild not complete; `redundancy_left < GROUP_TOLERATED_FAILURES`), else `0.0` (`degraded_cost(is_degraded)`). Summed over an episode it is the count of degraded intervals, a dimensionless exposure proxy with no hidden scaling | Simulator |
| `V` | `1.0` if `latency_now > LATENCY_LIMIT_FACTOR * latency_target` for that interval, else `0.0` (`latency_violation(latency_now, latency_target)`). **Latency only.** The risk budget is enforced by the safety filter and checked at episode level in `info`, not through `V` | Simulator |

`C` and `V` are defined once, by the two helper functions in `contracts.py`. No lane defines its own. `degraded_cost` accepts a built-in `bool` or `numpy.bool_` (so `degraded_cost(redundancy_left < GROUP_TOLERATED_FAILURES)` works directly under numpy) and rejects `int`, `float`, `str` and `None`. This is deliberately more permissive than `validate_info`: the helper is a compute step inside the simulator loop, whereas `info` is the authoritative record that is logged and analysed, where a numpy bool would hide a type error and serialise inconsistently.

**Non-finite latencies raise.** `latency_degradation` and `latency_violation` require `latency_now` and `latency_target` to be finite real numbers and raise `ContractViolation` otherwise. The failure direction is what makes this dangerous. Every comparison with NaN is false, so `max(0.0, nan)` is `0.0` and `nan > limit` is `False`: a NaN `latency_now`, which a simulator's latency average can produce for a control interval with no completed requests, would read as zero degradation and no violation. A broken measurement would look like a healthy system, and the latency constraint record in `info` would show it satisfied for the wrong reason. Rejecting the value makes the broken measurement visible instead of being scored as the best possible state.

Coefficients are `RewardConfig(alpha=1.0, beta=0.1, gamma=10.0)`. They are design defaults and live in configuration (`RewardConfig.from_dict` rejects unknown keys); a call site must never hardcode them.

**Reliability and QoS are constraints, not reward weights.** A weighted sum lets a large enough latency gain buy back any amount of reliability loss: for some weights the reward-optimal policy will accept a higher chance of permanent data loss to shave latency. The risk budget (`RISK_BUDGET_EPS`) and the latency limit (`LATENCY_LIMIT_FACTOR`, `LATENCY_VIOLATION_FRACTION_MAX`) are therefore stated as explicit constraints of a constrained MDP and recorded in `info` (section 6). The reward terms only shape behaviour inside the feasible region. The `gamma * V` term is shaping, not enforcement: `info` is authoritative for whether a constraint held, and a policy's reward must never be quoted as evidence of constraint satisfaction.

## 6. Required `info` keys

`validate_info(info)` raises `ContractViolation` naming the offending key.

| Key | Type | Rule |
|---|---|---|
| `latency_p95` | float | finite, >= 0 |
| `latency_p99` | float | finite, >= 0 |
| `violated_latency` | `bool` | built-in `bool` only |
| `p_failure_before_complete` | float | finite, in `[0, 1]` |
| `violated_risk_budget` | `bool` | built-in `bool` only |
| `alloc_clipped_by_filter` | `bool` | built-in `bool` only |
| `recovery_complete` | `bool` | built-in `bool` only |

`numpy.bool_` and `int` are rejected for the four flag keys. Numeric keys reject `bool`.

**Independence rule.** These keys are computed independently of the reward. A reward penalty is therefore never evidence that a constraint held: whether the latency limit or the risk budget was met is read from `violated_latency` and `violated_risk_budget`, not inferred from reward values or reward curves.

**Extra keys are permitted.** `validate_info` checks that the required keys are present and well-typed; it never rejects additional keys. A lane may log its own diagnostics and must not strip them to satisfy the check.

**Optional reset history (v1.5.0).** With `SimConfig.history_rows > 0` the reset `info` (and only the reset `info`) additionally carries `history_fg_bw_mbps`, `history_read_iops` and `history_write_iops`, float64 `numpy` arrays of length `history_rows` holding the trace rows before `trace_start_row`, oldest first, ending at row `trace_start_row - 1`; and `history_rows`, `trace_start_row_drawn` and `trace_start_row_remapped`. They are `numpy` arrays, not JSON-serialisable; a logger must convert them. With `history_rows = 0` (default) none of these keys is present.

**`termination_reason`.** When an episode ends, `info` additionally carries `termination_reason` (section 7), whose value is the enum's `.value` **string**, not the enum member, so `info` stays JSON-serialisable for logging and offline analysis. If that key is present and not `None`, `validate_info` requires one of the three strings and rejects the enum member itself. `None` (or absence) is accepted for steps where the episode has not ended.

## 7. Termination

`TerminationReason` has three members. All three are distinguishable in `info`: at episode end the key `termination_reason` carries the enum's `.value`.

| Member | `.value` | Meaning |
|---|---|---|
| `RECOVERY_COMPLETE` | `"recovery_complete"` | Rebuild finished without data loss |
| `TERMINAL_FAILURE` | `"terminal_failure"` | **Data loss**: redundancy was already exhausted (`redundancy_left == 0`) and a further drive failure occurred before the rebuild finished |
| `HORIZON_EXPIRED` | `"horizon_expired"` | `MAX_EPISODE_STEPS` elapsed with neither of the above |

`info["recovery_complete"]` is `True` only for `RECOVERY_COMPLETE`. The three outcomes must be reported separately; a truncated episode (`HORIZON_EXPIRED`) is not a success and not a failure and must not be folded into either count.

### `terminated` and `truncated`

The mapping is frozen and exposed as `termination_flags(reason) -> (terminated, truncated)`. No lane derives it independently.

| Reason | `terminated` | `truncated` |
|---|---|---|
| `RECOVERY_COMPLETE` | `True` | `False` |
| `TERMINAL_FAILURE` | `True` | `False` |
| `HORIZON_EXPIRED` | `False` | `True` |

This is not a convention detail. The episode horizon is an artificial cutoff, not an absorbing state. If horizon expiry were reported as `terminated=True`, a value-based learner would bootstrap `V(s') = 0` at the cutoff and systematically under-value states that lead to long rebuilds. That biases the agent against slow allocations, which is exactly the trade-off the project exists to measure, so the bug would corrupt the headline result while looking like a legitimate finding. Horizon expiry must be truncation.

`termination_flags` raises `ContractViolation` if given anything other than a `TerminationReason`.

### Exposure and loss

**Exposure and loss are different quantities.** An 8+2 group with one failed drive survives a second failure (it continues with `redundancy_left = 0`) and loses data on a third. Ending episodes at the second failure would end episodes the real system survives and would conflate exposure with loss. So:

| Quantity | Event | Where it appears |
|---|---|---|
| Exposure | At least one additional drive failure before rebuild completes | `p_failure_before_complete`; bounded by `RISK_BUDGET_EPS`; flagged by `violated_risk_budget` |
| Loss | A failure while `redundancy_left == 0` | `TerminationReason.TERMINAL_FAILURE` ends the episode |

Loss requires an exposure event followed by another failure, so loss is a strictly rarer event than the one `RISK_BUDGET_EPS` bounds. The two must not be reported under one name.

## 8. Hazard interface

```python
class HazardModel(Protocol):
    def __call__(self, features: Mapping[str, float],
                 horizon_hours: float = HORIZON_HOURS) -> tuple[float, float]: ...
```

The return value is `(p_horizon, uncertainty)`. Both are **over the horizon H = 30 days = 720 h**: `p_horizon` is a probability, not a rate. `uncertainty` is **one standard deviation of the estimate of `p_horizon`**, over the same horizon, or `0.0` when the risk model provides none. It is a dispersion measure, not a probability, and must lie in `[0, 0.5]`. Observation field 5 (`hazard`) is a **per-hour rate**, and a **group-level aggregate**: the sum of the per-hour hazards of the surviving drives. The one sanctioned bridge between a horizon probability and a per-hour rate is

```
hazard_per_hour = 1 - (1 - p_horizon) ** (1 / horizon_hours)      # hazard_p_to_per_hour
```

which is a **discrete per-hour probability**, not the continuous rate `-ln(1 - p) / H`; the two are nearly equal for small `p`. It raises `ContractViolation` unless `p` is a real number with `0 <= p < 1` and `horizon_hours` is a real number that is **finite** and `> 0`. No other code converts between them.

**Every boundary failure raises `ContractViolation`.** Nothing in the hazard path leaks an interpreter exception such as `TypeError`: a caller that catches `ContractViolation` to report a clean boundary error sees every failure. This covers a malformed `per_drive` element (see below), a non-numeric or `bool` argument, and out-of-range values.

**Return types and precision.** Every function in the contract annotated `-> float` (`action_to_mbps`, `latency_degradation`, `degraded_cost`, `latency_violation`, `compute_reward`, `hazard_p_to_per_hour`, `hazard_uncertainty_to_per_hour`, `kbps_to_mbps`) returns a builtin `float`, and `aggregate_group_hazard` returns a tuple of builtin `float`. Inputs are converted to `float` before any arithmetic, so computation is done in float64 regardless of input dtype. A numpy-native lane can therefore pass its own scalars (`float32`, `float64`, integer) without changing the result or the type anything downstream sees, and nobody needs a defensive `float()` around a contract call. Non-numbers and `bool` are still rejected with `ContractViolation`.

**`horizon_hours` must be finite and positive** in both `hazard_p_to_per_hour` and `hazard_uncertainty_to_per_hour`. An infinite horizon makes the exponent `1 / H` zero and silently returns a zero hazard (and a zero uncertainty). That is the most dangerous failure mode for this field: a broken input would read as a perfectly safe group instead of raising.

**Group aggregation.** The risk model is called per surviving drive. `aggregate_group_hazard(per_drive, horizon_hours=HORIZON_HOURS) -> (hazard, hazard_uncertainty)` takes one `(p_horizon, uncertainty)` pair per surviving drive, converts each `p_horizon` with `hazard_p_to_per_hour` and each `uncertainty` with `hazard_uncertainty_to_per_hour` (evaluated at that drive's own `p_horizon`), and returns the values for fields 5 and 6 in per-hour units:

```
hazard             = sum_i rate_i
hazard_uncertainty = sqrt( sum_i uncertainty_rate_i ** 2 )
```

**Accepted shapes of `per_drive`.** Either a sequence (list or tuple) of pairs, or an `(n, 2)` `numpy.ndarray` whose rows are the pairs. Each pair is a sized, indexable object of exactly two real numbers: a tuple, a list or a 1-D `numpy.ndarray` of length 2. The element test is structural (sized, indexable, length 2, not a `str`, `bytes` or mapping), not a nominal `Sequence` test. Numpy rows are accepted deliberately: the risk model is numpy-native and its natural output for the surviving drives is an `(n, 2)` array, so the contract takes it directly instead of leaving each lane to write its own unpacking shim. The `GROUP_SIZE - 1` limit applies to the number of rows of an array as it does to the length of a list. An array passed as `per_drive` that is not 2-D with two columns (a 1-D or 3-D array, or `(n, 3)`) raises `ContractViolation` stating the required shape.

It raises `ContractViolation` on an empty sequence, on more than `GROUP_SIZE - 1` entries, on a malformed element, on a `per_drive` that is not iterable, or on any value outside its bounds (`0 <= p < 1`; `0 <= uncertainty <= 0.5`). `None`, a scalar, a string, a 2-D array row, a sequence of any other length, and a pair with a non-numeric member (`None`, a string, a `bool` or `numpy.bool_`) are all rejected with `ContractViolation`, never `TypeError`. `bool` members are rejected on type so that a truth value never reaches a probability field. The error names the pair index and what was wrong: not a pair, wrong length, a non-numeric member with its type, or which of `p_horizon` or `uncertainty` failed and the bound it broke. No lane aggregates any other way.

**Propagating the uncertainty.** `hazard_p_to_per_hour` is a probability transform and must not be applied to a standard deviation: a dispersion does not convert the way the quantity it disperses does. `hazard_uncertainty_to_per_hour(sigma_horizon, p_horizon, horizon_hours=HORIZON_HOURS)` uses first-order error propagation (the delta method). With `h = f(p) = 1 - (1 - p) ** (1 / H)`:

```
dh/dp   = (1 / H) * (1 - p) ** (1 / H - 1)
sigma_h = sigma_p * dh/dp          # evaluated at that drive's own p
```

For small `p` the derivative tends to `1 / H`, so the delta method and the (incorrect) probability map both give roughly `sigma / 720` and the numbers are nearly identical in the expected regime. The delta method is still the correct propagation, stays correct when `p` is not small, and survives the question of why a standard deviation was pushed through a probability map. It is a first-order approximation; it ignores curvature of `f` and assumes the horizon-probability estimate is locally linear in its error.

**Why `sigma_horizon <= 0.5`.** The standard deviation of a quantity bounded in `[0, 1]` cannot exceed 0.5. A larger value indicates a broken or mis-scaled estimate rather than genuine uncertainty, and silently accepting it would let a miscalibrated risk model inflate field 6 without anything noticing. Non-finite values are rejected.

**Why group-level.** The constraint and the observation must be about the same quantity. `RISK_BUDGET_EPS` bounds the probability of at least one additional failure anywhere in the group before the rebuild completes. If field 5 carried one drive's hazard, the agent would have no information about the other eight while the safety filter computed exposure over all nine, and it could not learn to satisfy a constraint whose driving variable it cannot see. With the aggregate in field 5, the exposure the filter bounds is

```
P = 1 - exp(-hazard * T_remaining)
T_remaining (h) = work_remaining (GB) * 1000 / alloc_mbps / 3600
```

which is computable from the observation alone. This is the definition the safety filter implements. Here `hazard` is the observation's group rate (field 5), `alloc_mbps` the candidate allocation and `work_remaining` field 9. The continuous form is used here; it differs negligibly from the discrete per-hour probability of the conversion at these magnitudes.

**Limitation: independence (to be reported, not hidden).** Summing per-drive hazards treats drive failures as independent. Correlated failure is well documented in fleet data: shared batch, temperature, vibration and age all couple drives in one enclosure. The aggregate therefore **understates tail risk**; the direction of the error is that it makes the group look **safer than it is**. The `sqrt(sum of squares)` uncertainty rests on the same independence assumption, so field 6 is also **optimistic** under correlation. Results must state this wherever group hazard or exposure is reported.

**Limitation (to be reported, not hidden).** The conversion assumes a **constant hazard rate across the 720 h horizon**. That assumption is what lets a 30-day probability be used as a per-hour rate at rebuild timescales of hours. It is a modelling assumption, not a measurement. If the true hazard varies within the horizon, the per-hour value is an average and can understate the instantaneous risk during a rebuild. Results must state this assumption wherever hazard-derived quantities appear.

## 9. Frozen numeric constants

Values are design decisions unless the tag says otherwise. The arithmetic is shown so that each can be re-derived and challenged. The first column of tags follows the provenance convention of section 1; `[DERIVED]` inherits the weakest tag among its inputs.

| Constant | Value | Tag | Justification |
|---|---|---|---|
| `CONTRACT_VERSION` | `"1.5.0"` | n/a (string) | Semantic version of this contract |
| `CONTROL_INTERVAL_S` | 300 | `[MEASURED]` | Matches the Tencent trace sampling interval (dataset README), so one control step is one trace sample and no resampling is needed |
| `HORIZON_DAYS` | 30 | `[ASSUMED]` | Hazard prediction window |
| `HORIZON_HOURS` | 720.0 | `[DERIVED]` | 30 d x 24 h/d. Horizon H of the hazard model. A 30-day window is standard in the SMART failure-prediction literature, which keeps published PR-AUC figures comparable |
| `RISK_BUDGET_EPS` | 5e-3 | `[ASSUMED]` | Bounds P(at least one additional failure before rebuild completes). Slack at baseline hazard, binding from about 1.4x baseline; derivation below. The baseline rests on the measured AFR note in the risk-budget subsection, which is stale |
| `LATENCY_LIMIT_FACTOR` | 1.5 | `[ASSUMED]` | `L_max = 1.5 x latency_target` |
| `LATENCY_VIOLATION_FRACTION_MAX` | 0.05 | `[ASSUMED]` | At most 5% of control steps may exceed `L_max` |
| `MAX_EPISODE_STEPS` | 2400 | `[DERIVED]` | 2400 x 300 s = 720,000 s = 200 h. Slowest action (20 MB/s) needs 2000 steps; see below. Import-time check enforces only the lower bound; the 400-step margin is a design choice and the inputs (work, slowest action) are `[ASSUMED]` |
| `ACTIONS_MBPS` | (20, 50, 80, 120) | `[ASSUMED]` | Prototype set, illustrative, not measured device limits |
| `N_ACTIONS` | 4 | `[DERIVED]` | `len(ACTIONS_MBPS)` |
| `GROUP_DATA_DRIVES` | 8 | `[ASSUMED]` | Group geometry 8+2 |
| `GROUP_PARITY_DRIVES` | 2 | `[ASSUMED]` | Group geometry 8+2 |
| `GROUP_TOLERATED_FAILURES` | 2 | `[DERIVED]` | Equals the parity count |
| `GROUP_SIZE` | 10 | `[DERIVED]` | `GROUP_DATA_DRIVES + GROUP_PARITY_DRIVES`. Survivors at episode start: `GROUP_SIZE - 1 = 9`, used in the risk arithmetic and in the hazard scale |
| `DRIVE_CAPACITY_GB` | 12000.0 | `[ASSUMED]` | 12 TB drive (decimal) |
| `REBUILD_WORK_GB` | 12000.0 | `[DERIVED]` | One drive's worth of reconstruction; equals `DRIVE_CAPACITY_GB` (import-time check) |
| `BW_TOTAL_MBPS` | 200.0 | `[ASSUMED]` | Shared bandwidth budget per group; the largest action (120) leaves 80 MB/s for foreground |

Observation scales: the hazard scale `1.8e-4` is `[DERIVED]` from `(GROUP_SIZE - 1) * 2e-5` (section 3); the IOPS scales (20000) and latency scales (50.0) are `[ASSUMED]`; the remaining scales are `[DERIVED]` from the constant they equal. Reward coefficients `alpha`, `beta`, `gamma` are `[ASSUMED]`. The simulator's own constants (unloaded latency 2 ms, target 15 ms, `rho_max` 0.98, drive spread 0.15) are `[ASSUMED]` in `sim/config.py`.

### Rebuild duration (why the action set is meaningful)

Rebuild work is 12,000 GB = 12,000,000 MB (decimal). Duration is work divided by allocated bandwidth, assuming the allocation is sustained.

| Allocation (MB/s) | Duration (s) | Duration (h) | Control steps at 300 s |
|---|---|---|---|
| 20 | 600,000 | 166.7 | 2000 |
| 50 | 240,000 | 66.7 | 800 |
| 80 | 150,000 | 41.7 | 500 |
| 120 | 100,000 | 27.8 | 333 |

At 80 MB/s a rebuild runs about 41.7 h. The span of 27.8 h to 166.7 h across the action set is wide enough that the choice matters for both exposure time and foreground impact.

### Risk budget `RISK_BUDGET_EPS = 5e-3`

`RISK_BUDGET_EPS` bounds `p_failure_before_complete`: the probability of **at least one additional drive failure before the rebuild completes** (exposure, not data loss; see section 7).

> **WARNING: `RISK_BUDGET_EPS` is a bound on exposure, not on data loss.** A rebuild episode starts at `redundancy_left = 1`, so `p_failure_before_complete` is the probability of **one** additional failure before the rebuild completes. Data loss needs **two** further failures before the rebuild completes, and its probability is smaller than that of exposure by roughly another factor of the same order (about the square of the exposure probability, up to a combinatorial factor; an order-of-magnitude statement, not a figure computed by this contract). `5e-3` therefore bounds exposure. **It must never be described in results, the report or a slide as a bound on the probability of data loss.** If a data-loss probability is ever quoted, it must be computed separately as the probability of two further failures within the degraded window, and labelled as such.

Inputs:

- Pooled Seagate AFR near 2.11%. This figure is derived from the project's own Q1 2022 Backblaze inspection (276 failures across 4.78M drive-days for the three pooled Seagate models: 276 / 4.78M x 365 = 2.11% per drive-year), and is consistent with the vendor's published AFRs for those models. It is **not independently verified by this contract**. Provenance: `[MEASURED]`, but from that earlier Q1-2022 extract only. The data window has since changed and the figure has **not been re-derived against the current window**; the baseline hazard (2.41e-6 per hour), the multiples in the table below, the hazard scale and the claim that `5e-3` is slack at baseline are all downstream of it and stale until it is.
- 8,760 h per year.
- 9 surviving drives in a 10-drive group after one failure.

```
per-drive-hour hazard   lambda = 0.0211 / 8760 ~= 2.41e-6 per hour
exposure over a rebuild P(k) = 1 - exp(-9 x k x lambda x T)
```

`k` is a multiplier on the per-drive hazard (`k = 1` is the fleet baseline) and `T` is the rebuild duration at the allocation. All values below are design arithmetic for a constant rate, not outputs of the risk model. **Stale inputs:** the 2.11% AFR behind `lambda` and every probability and multiple in the table below comes from the earlier Q1-2022 three-model extract; the data window has since moved to full-year 2022 with four models, where the rate is 1.95%. The table is stale pending re-derivation and has not been recomputed here.

| Allocation (MB/s) | T (h) | 1x baseline | 2x | 10x | Constraint binds (P > 5e-3) above `k` = |
|---|---|---|---|---|---|
| 120 | 27.8 | 6.0e-4 | 1.2e-3 | 6.0e-3 | 8.3 |
| 80 | 41.7 | 9.0e-4 | 1.8e-3 | 9.0e-3 | 5.5 |
| 50 | 66.7 | 1.4e-3 | 2.9e-3 | 1.4e-2 | 3.5 |
| 20 | 166.7 | 3.6e-3 | 7.2e-3 | 3.5e-2 | 1.4 |

Reading the table against `5e-3`:

- **At baseline (1x) all four actions are feasible.** The worst case is 3.6e-3 at 20 MB/s. The safety filter therefore does not remove any action on the strength of the fleet-average hazard alone.
- **The constraint begins to bind near 1.4x baseline**, where 20 MB/s first exceeds the budget.
- **At 2x**, 20 MB/s is infeasible; 50, 80 and 120 are feasible.
- **From about 3.5x**, 50 MB/s is also infeasible; only 80 and 120 remain. From about 5.5x only 120 remains.
- **At 10x** every action exceeds the budget (even 120 MB/s gives 6.0e-3). This is the all-actions-violate case, below.

**Why 5e-3 rather than 1e-3.** At `1e-3`, the 50 MB/s (1.4e-3) and 20 MB/s (3.6e-3) actions already violate at baseline hazard. The filter would remove half the action set statically, before the agent saw any hazard estimate, and the controller would be clipped on workload alone. The hazard input would then carry no decision-relevant information, and the planned ablation (remove the hazard input, retrain, check whether the policy changes) would fail by construction rather than by evidence. The budget must be slack at baseline hazard and binding at elevated per-drive hazard; that gap is the reason the risk input exists. `5e-3` is the design choice that satisfies this for the prototype action set.

**All-actions-violate rule.** When every action exceeds `RISK_BUDGET_EPS`, the safety filter does not fail or stall: it selects the minimum-risk action, which is the largest bandwidth (the shortest exposure window). This is a **deliberate degradation rule**. It sets `alloc_clipped_by_filter` when it changes the agent's choice, and `violated_risk_budget` is `True` for the step, so the breach stays on record in `info`.

### Episode length versus rebuild duration

`MAX_EPISODE_STEPS = 2400` covers 200 h (2400 x 300 s = 720,000 s). The slowest action needs 2000 steps (166.7 h), so the horizon clears it with a margin of 400 steps, about 17%.

**The episode horizon must exceed the slowest action's completion time, or baseline comparisons are invalid.** At a smaller horizon a fixed-rate 20 MB/s baseline could never complete a rebuild and would always end as `HORIZON_EXPIRED`; a learned controller would then beat it for a spurious reason. `contracts.py` checks this invariant at import. The margin assumes the allocation is actually delivered; foreground contention that lowers the delivered rebuild rate below the allocation consumes that margin.

## 10. Resolved contradictions

Five conflicts in the source instructions were found and resolved before freeze.

| # | Topic | Conflict | Resolution |
|---|---|---|---|
| 1 | Hazard unit | The observation field was specified as "per hour", while the hazard signature returns a probability over H. | The model returns `p` over H. The adapter converts with `hazard_p_to_per_hour`. Field 5 is a per-hour rate, aggregated over the surviving drives with `aggregate_group_hazard`. |
| 2 | Replay columns | One instruction said replay files carry exactly six columns; another said to attach five metadata fields. | The six-column replay files stay untouched (`REPLAY_COLUMNS`). Metadata lives in a separate disk-index file (`DISK_INDEX_COLUMNS`), joined on `disk_uid` only where needed. *Historical note: "five metadata fields" records the original instruction. `DISK_INDEX_COLUMNS` has since carried six entries, and seven from 1.2.0. The decision stands; only the count is stale.* |
| 3 | Bandwidth units | Traces report KB/s; the observation needs MB/s. | The pipeline stores source KB/s unconverted so its files remain a faithful record of the source. `kbps_to_mbps` converts once, at the adapter boundary, using the decimal convention (divide by 1000). |
| 4 | `t_seconds` origin | Source timestamps are Unix epoch; the grid test assumed a 0-based grid. | Replay files carry episode-relative seconds starting at 0 with a 300 s step. Rebasing happens in the pipeline and is recorded in its preprocessing log. `validate_replay_grid` enforces this. |
| 5 | Foreground load scaling | A single real virtual-disk trace peaks in the tens of KB/s while the action set is 20-120 MB/s, so one disk cannot load a simulated group. | Foreground load for one group is the **aggregate of N real disk traces**, with N chosen per intensity profile so that the peak profile's p95 aggregate bandwidth lands near 0.8 x `BW_TOTAL_MBPS` (160 MB/s). N and the resulting p95 are recorded per profile in the pipeline's preprocessing log and manifest. |

On decision 5: the aggregation is a **modelling assumption**. The resulting load is a **simulated aggregate**, not an observed group workload, and it must be labelled as such wherever it appears in results.

## 11. What this contract does not provide

No dataset used by this project supplies:

- rebuild start or end times;
- the rebuild bandwidth that was allocated;
- application latency during a rebuild;
- counterfactual outcomes under alternative actions.

Everything of that kind comes from the simulator and must be labelled **simulated** wherever it is reported.

## Appendix A. Recorded behaviour decisions

These are decisions, not omissions. Each is exercised by the implementation as stated.

| Item | Decision |
|---|---|
| `RewardConfig.from_dict` | Missing keys are filled with the defaults; unknown keys raise `ContractViolation` |
| `hazard_p_to_per_hour` with NaN | Raises `ContractViolation` (NaN fails `0 <= p < 1`) |
| `latency_degradation` / `latency_violation` with NaN or inf | Raise `ContractViolation` (a NaN would otherwise read as zero degradation or no violation) |
| Numeric `info` keys | Accept numpy floating scalars (and any real number except `bool`); the four boolean keys reject `numpy.bool_` and `int` |
| `validate_replay_grid` | Rejects an empty sequence; accepts `[0]` (a single-sample grid) |
| `to_vector` round trip | Exact only for float32-representable values; realistic decimals match to about 1e-6 relative |
| `termination_reason` | `None` or absent is accepted by `validate_info`; a present non-`None` value must be one of the three strings |

### Considered and rejected: log-scaled hazard input

A log transform of field 5 would condition the input better across the several orders of magnitude a SMART-based hazard can span. It was rejected because it would make field 5 no longer a plain normalised rate (so `Observation.to_vector` and `from_vector` would no longer be the simple `clip(value / scale, 0, 1)` pair) and would put a non-uniform rule into a contract whose value is being simple to check. The scheme stays uniform: one linear scale per field, no special cases. This was a considered decision, not an oversight.

## 12. Replay file schema

A replay file is the foreground-load trace the simulator consumes. It has exactly the six columns of `REPLAY_COLUMNS`, in this order, and each column has the dtype pinned by `REPLAY_DTYPES`.

| Column | Required dtype | Meaning |
|---|---|---|
| `t_seconds` | `int64` | Episode-relative seconds, starting at 0, step `CONTROL_INTERVAL_S` (section 10, decision 4) |
| `read_iops` | `int64` | Read operations per second |
| `write_iops` | `int64` | Write operations per second |
| `read_bw_kbps` | `int64` | Read bandwidth in KB/s, unconverted (section 10, decision 3) |
| `write_bw_kbps` | `int64` | Write bandwidth in KB/s, unconverted |
| `disk_usage_mb` | `int64` | Disk usage in MB |

**Why the dtypes are pinned.** Column names and order alone do not fix a format. The Tencent Cloud source fields are integer counts as distributed (for example `read_IOPS=11`, `read_bandwidth=50` KB/s, `disk_usage=3360` MB), and the documented aggregation of N disk traces into one group load (section 10, decision 5) is a sum of integers, so it stays integral. Pinning `int64` keeps a synthetic fallback generator and the real pipeline byte-comparable, which is what lets either feed the simulator interchangeably. Without it a consumer can behave differently depending on which source produced its input (a float column from one source and an integer column from the other), which is the divergence this contract exists to prevent. The aggregation of N traces is a modelling assumption, and a replay file built from it is a simulated aggregate (section 10).

**`validate_replay_frame(df) -> None`** is the single check a producer should run before writing a replay file and a consumer before reading one. It takes a pandas `DataFrame` and raises `ContractViolation` unless:

1. the columns equal `REPLAY_COLUMNS` exactly, in order, not merely as a set;
2. no column contains a null (checked before the dtype, because a null forces an integer column to float and the null is the more useful thing to report);
3. every column's dtype equals `REPLAY_DTYPES`;
4. no value is negative;
5. `validate_replay_grid` passes on the `t_seconds` column (non-empty, starts at 0, no duplicates, strictly increasing by exactly `CONTROL_INTERVAL_S`).

Every error names the offending column, and a dtype mismatch names both the expected and the actual dtype. pandas is imported inside the function, not at module top level, so `contracts.contracts` keeps its numpy-and-stdlib-only import surface for lanes that do not use pandas. `validate_replay_grid` remains available for a time grid held outside a frame.

**`to_replay_frame(window)`** builds a replay frame from a window of the cleaned Tencent load table. The window is a pandas `DataFrame` with a `timestamp` column (source Unix epoch seconds, exactly `CONTROL_INTERVAL_S` apart, no gap) and the five value columns of `REPLAY_COLUMNS` after `t_seconds`; other columns are ignored. It returns the six `REPLAY_COLUMNS` in order, all `int64`, with `t_seconds` rebased to start at 0 (section 10, decision 4) and bandwidth left in KB/s (decision 3). Timestamps and values must be finite whole numbers: a whole-number float such as `300.0` is accepted and converted to `int64`, while a NaN, an infinity or a fractional value is rejected before any cast, with the column named. The result is passed through `validate_replay_frame` before it is returned, so negative values, dtypes and the time grid are enforced by the same function consumers use. It raises `ContractViolation` if the argument is not a DataFrame, a required column is absent or non-numeric, a required column holds a non-finite or non-integral value, the timestamps are not a gap-free grid of at least one row, or the result breaks the replay schema. It lives in the contract so that the rebasing rule is defined once; it was previously a function of the data pipeline's extractor.

**Disk index (`DISK_INDEX_COLUMNS`).** Per-disk metadata lives in a separate disk-index file, joined to replay data on `disk_uid` only where needed (section 10, decision 2). Its seven columns, in this order, are:

| Position | Column |
|---|---|
| 0 | `disk_uid` |
| 1 | `disk_capacity` |
| 2 | `disk_attr` |
| 3 | `disk_type` |
| 4 | `user_type` |
| 5 | `vm_cpu` |
| 6 | `vm_memory` |

**Why the order is source order.** The order is the field order of the Tencent Cloud Volume Workload Traces README for the subscription table: cloud disk ID, cloud disk capacity (GB), product type, cloud disk type, user type, number of virtual machine CPU cores, virtual machine memory size (GB). The README's own example row, `a2049546-cd8b-49ff-b863-c503baa79429,50,1,1,0,2.0,4.0`, confirms that capacity is the second field. `disk_attr` is the README's "product type" and `disk_type` is its "cloud disk type". The subscription files are headerless and the data pipeline reads them positionally, so keeping the contract's order identical to the source order means a producer can map columns without a lookup table, whereas any other order would make a positional reader silently assign values to the wrong column: it does not raise, it mis-maps every later field. `user_type` distinguishes customer classes, which plausibly explains differences in load pattern between volumes, so it is available as a stratifier when selecting workload windows. Nothing requires it today, and that plausibility is a hypothesis, not a measured effect.

**Rule: verify this order against the dataset's README only.** If this order is ever changed again, it must be verified against the dataset's own README and its example row, never against a project document. The earlier order was taken from a secondhand schema description in a project document, not from the dataset, and was wrong (see the latest changelog row). A wrong position does not raise an error; it silently mis-maps every later field.

**`validate_disk_index_frame(df) -> None`** is the single check a producer should run before writing the disk-index file and a consumer before reading it. It takes a pandas `DataFrame` and raises `ContractViolation` unless:

1. the argument is a `DataFrame`;
2. the columns equal `DISK_INDEX_COLUMNS` exactly, in order, not merely as a set;
3. no column contains a null (checked before the dtype, for the same reason as in `validate_replay_frame`);
4. `disk_uid` is a string column (NumPy `object` holding only `str`, or a pandas string dtype) and the other six columns have dtype `int64`;
5. `disk_uid` has no duplicate value, the error naming the first repeated value and its row;
6. no value in the six numeric columns is negative, the error naming the column and the row.

Every error names the offending column, and a dtype mismatch names both the expected and the actual dtype. pandas string dtypes are accepted for `disk_uid` because the default string dtype of current pandas is not NumPy `object`; a frame read back from Parquet carries it. A frame with the right columns and dtypes but no rows passes. pandas is imported inside the function, as in `validate_replay_frame`.

## 13. Changelog

| Version | Date | Change |
|---|---|---|
| 1.0.0 | 2026-10-29 | Initial freeze. |
| 1.1.0 | 2026-10-29 | Pinned replay-file dtypes via `REPLAY_DTYPES` (all `int64`) and added `validate_replay_frame`; the schema previously fixed column names and order but not types. |
| 1.2.0 | 2026-10-30 | Added `user_type` to `DISK_INDEX_COLUMNS` at source-order position 3, as a stratifier for workload-window selection. Note: inserting rather than appending is a **breaking change for any positional consumer** of this tuple, so by strict semver this warrants a major bump; it is recorded as minor because the only consumer is the data pipeline and nothing had been built against the six-column form. **Superseded:** that position was wrong — the order came from a project document, not the dataset. See the next row. |
| 1.3.0 | 2026-10-30 | Corrected `DISK_INDEX_COLUMNS` to the dataset README's field order, moving `disk_capacity` from last position to index 1. The 1.2.0 order was taken from a project document rather than the dataset and was wrong; the subscription file is headerless, so a positional reader using the 1.2.0 order would have mis-mapped every field after the first. Caught by the data-pipeline lane, which had the correct order in its own extractor code. |
| 1.4.0 | 2026-10-31 | Added `validate_disk_index_frame` and moved `to_replay_frame` into the contract. The data pipeline's disk-index output (section 12, disk index) had no contract-level validator while its replay frames did, so `build_disk_index.py` validated itself with ad-hoc checks instead of a shared, tested contract function; the new function checks columns, nulls, dtypes, duplicate `disk_uid` and negative values in the same order and with the same error conventions as `validate_replay_frame`. `to_replay_frame`, which rebases a cleaned Tencent window to the replay schema, previously lived in the pipeline's extractor; it is the producer-side counterpart of `validate_replay_frame` and its only users were inside the data pipeline, so it now sits beside `REPLAY_COLUMNS` with `ContractViolation` as its error type. It rejects non-finite, non-integral and non-numeric timestamps and values before any integer cast, and validates its own result with `validate_replay_frame` before returning it. Both are additions; no existing name, column or value changed. |
| 1.5.0 | 2026-11-19 | **Proposal, sign-off of all four teammates pending.** Added optional pre-failure load history to the reset `info`. `PRE_FAILURE_HISTORY_ROWS` (576, two days at the 300 s control interval) and `RESET_HISTORY_INFO_KEYS` (`history_fg_bw_mbps`, `history_read_iops`, `history_write_iops`) are new names. When `SimConfig.history_rows > 0` the reset `info` carries those three float64 arrays, the `history_rows` trace rows immediately before `trace_start_row`, oldest first and ending at row `trace_start_row - 1` (the reset observation carries row `trace_start_row`), together with `history_rows`, `trace_start_row_drawn` and `trace_start_row_remapped`; a drawn start row below `history_rows` is mapped onto `[history_rows, len(trace))` so the history never wraps, and every other start row is unchanged. `history_rows` defaults to 0, which leaves every reset, step, observation, `info` key and value unchanged. The observation vector, the action set, the reward and the required `info` keys do not change. Rationale: a deployed controller has monitored the load for days before a drive fails; without the history a controller must start every rebuild with an empty forecast, which the simulator otherwise imposes. |

The rationale for each design decision lives in the numbered sections above, not in this table. This contract is frozen at the first row; every subsequent row requires the sign-off of all four teammates. The second row was added without that sign-off: the gap was found by the first consumer built against the contract, before any other workstream had built against it, and it is recorded here rather than amended silently. The third row also lacks the full sign-off: it was proposed by the data-pipeline lane, which owns the only consumer of the disk-index schema, and confirmed by the contract owner. It was not ratified by all four teammates, and it is recorded here rather than amended silently. The fourth row has the same caveat: the correction was raised by the data-pipeline lane and confirmed by the contract owner, and was not ratified by all four teammates. The fifth row has the same caveat: it was raised by the data-pipeline lane, which owns the only consumers of both functions, and ratification by all four teammates is not recorded in this repository; it is listed here rather than amended silently.
