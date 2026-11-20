# Task prompt (spec 1.1.0)

Give this to the model verbatim. It is the experimental stimulus: changing it changes
every score, so it is versioned with the spec and must not be edited per model.

---

You are writing a control policy for a storage system.

## The situation

A disk has failed. The system is rebuilding its data onto a spare. The rebuild shares
device bandwidth with live user traffic, so rebuilding faster finishes sooner but
degrades user-visible latency.

Every 300 seconds you choose a rebuild rate from four options:

```
ACTIONS_MBPS = [20, 50, 80, 120]      # you return the index 0, 1, 2 or 3
```

An episode ends when the rebuild completes, a second disk fails, or 2400 intervals pass.

## What you are optimising

Per interval the system charges you:

```
cost = 1.0 * latency_degradation + 0.1 * degraded_state + 10.0 * latency_violation
```

- **latency_degradation** rises as the device gets busier.
- **degraded_state** is 1 for every interval the rebuild is unfinished, so finishing
  sooner is worth something.
- **latency_violation** is 1 when mean latency exceeds the target. It is weighted
  heavily.

You want to minimise total cost. Roughly: **finish quickly without pushing latency over
the limit.**

## A safety filter you do not control

Before your action is applied, a safety filter may **raise** it if the remaining
redundancy exposure would be too high. It never lowers it. The filter runs on every
policy identically, including the baseline you are compared against, so you cannot gain
or lose by trying to work around it. On some intervals it will override you entirely.

## Your observation

A 12-element float32 vector. **Every field is normalised to roughly [0, 1].** None of
them are in physical units: there are no milliseconds, no IOPS and no per-hour rates in
what you receive. Compare fields against each other and against the ranges below, never
against a physical constant.

Observed ranges, measured over the development workload:

| index | field | min | median | max | notes |
|---|---|---|---|---|---|
| 0 | `fg_read_iops` | 0.06 | 0.22 | 0.59 | foreground read load |
| 1 | `fg_write_iops` | 0.07 | 0.21 | 1.00 | foreground write load |
| 2 | `fg_bw` | 0.12 | 0.40 | 0.58 | foreground bandwidth share |
| 3 | `latency_now` | 0.07 | 0.20 | 1.00 | current latency |
| 4 | `latency_target` | 0.30 | 0.30 | 0.30 | **constant**; a violation is `latency_now > latency_target` |
| 5 | `hazard` | 0.16 | 0.25 | 0.33 | failure-rate estimate; saturated, so it never reaches the top of its range |
| 6 | `hazard_uncertainty` | 0.01 | 0.02 | 0.03 | |
| 7 | `redundancy_left` | 0.50 | 0.50 | 0.50 | **constant** in this configuration |
| 8 | `progress` | 0.00 | 0.40 | 0.80 | fraction rebuilt; **never reaches 1.0 during an episode**, the episode ends instead |
| 9 | `work_remaining` | 0.20 | 0.60 | 1.00 | |
| 10 | `alloc_current` | 0.00 | 0.67 | 0.67 | the previously applied rate, as a fraction of 120 MB/s |
| 11 | `bw_available` | 0.40 | 0.58 | 0.86 | headroom: capacity minus foreground load |

Three traps worth avoiding:

- `latency_target` and `redundancy_left` are constants here. Branching on them does
  nothing.
- `progress < 1` is true on every step of every episode. A branch on it is a constant
  policy.
- `hazard` never goes below 0.16. A threshold like `hazard > 0.001` fires always.

You see only the present interval -- **nothing about future load.**

## What to write

Exactly this function, in one file, nothing else at module scope except imports:

```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    ...
```

Constraints:

- **Deterministic.** The same inputs must always give the same output.
- **Under 10 ms per call.** It is called thousands of times per episode.
- **`state`** is a dict you may write to in order to remember things within one episode.
  It is cleared at the start of every episode.
- **Only `numpy` and the Python standard library.** No file access, no network.
- Return a Python `int` in `[0, 3]`.

## How you are scored

Against the **exact optimal policy**, computed by dynamic programming with full
knowledge of the future, and against a tuned rule-based controller:

```
G = (your_return - rule_return) / (optimal_return - rule_return)
```

`G = 1` means you matched the optimum. `G = 0` means you matched the rule. Negative
means you did worse than the rule.

You are scored on held-out workloads you have not seen, so prefer a policy that reasons
from the observation over one tuned to any particular pattern.

Write the function.
