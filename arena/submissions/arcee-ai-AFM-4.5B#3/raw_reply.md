Let me think through how to approach this problem step by step.

First, I need to understand the key variables and the goal:
- We're optimizing a cost function with three components: latency degradation, degraded state, and latency violations
- We have 12 normalized observation fields
- We need to choose one of four possible rebuild rates every 300 seconds
- There's a safety filter that may override our choice

Given these constraints, the optimal approach will likely involve:
1. Evaluating the current state and deciding whether to increase, decrease, or maintain the rebuild rate
2. Considering the safety filter's effect
3. Balancing between finishing quickly and not violating latency

Let's think about what each observation field might represent:
- `fg_read_iops` and `fg_write_iops` are foreground loads
- `fg_bw` is the share of available bandwidth for foreground
- `latency_now` is current latency
- `latency_target` is the limit
- `hazard` and `hazard_uncertainty` are failure rates
- `redundancy_left` is remaining redundancy
- `progress` is rebuild progress
- `work_remaining` is rebuild work left
- `alloc_current` is the current rate
- `bw_available` is headroom

Given the safety filter always raises (never lowers) rates, we should:
1. Only increase the rate if it's safe to do so
2. Prefer higher rates if we're close to finishing
3. Avoid increasing if it would cause a violation

The optimal policy should generally increase the rate when:
- We're close to finishing the rebuild
- The safety filter hasn't raised the rate yet
- We're not already at the maximum rate
- Increasing doesn't cause a violation

For the specific rate choices:
- Higher rates (80 and 120) are riskier but finish faster
- Lower rates (20, 50) are safer but slower

Based on this reasoning, the optimal policy is to:
1. Check if we can increase the rate
2. If so, prefer to increase to 80 or 120 if possible
3. Otherwise, stay at the current rate

So the function should return:
- 0: stay at current rate
- 1: increase to 80
- 2: increase to 120
- 3: increase to max (120)

The function implementation would check if we can increase, and if so, choose between 80 and 120 based on some criteria (e.g., how close we are to finishing).

Here's the code:

```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    current_rate = state['alloc_current']
    fg_read_iops = obs[0]
    fg_write_iops = obs[1]
    fg_bw = obs[2]
    latency_now = obs[3]
    latency_target = obs[4]
    hazard = obs[5]
    hazard_uncertainty = obs[6]
    redundancy_left = obs[7]
    progress = obs[8]
    work_remaining = obs[9]
    bw_available = obs[11]
    actions = [0, 1, 2, 3]

    if current_rate == 0:
        # Start with lowest risk rate
        return 0

    # Check if we can increase
    if current_rate < 3:
        # Prefer higher rates when close to finishing
        if progress + 1 >= 1:
            if current_rate < 2:
                return 1
            elif current_rate < 3:
                return 2
        else:
            # If we're not close to finishing, prefer lower risk
            return 0

    # If we can't increase, stay at current rate
    return current_rate
```

This function:
- Starts with the lowest risk rate (0)
- Checks if we can increase
- If so, increases to 80 if we're close to finishing, or 120 otherwise
- Otherwise, stays at current rate

This should match the optimal policy as reasoned above.