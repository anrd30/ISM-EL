To create an effective policy for the given storage system scenario, we need to balance the trade-off between rebuilding the disk as quickly as possible and keeping the latency within acceptable limits. The policy should take into account various observations such as the current foreground IOPS, bandwidth usage, and the current latency.

Here's a possible implementation of the policy function:

```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
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
    alloc_current = obs[10]
    bw_available = obs[11]

    # Determine the threshold for latency violation
    latency_violation_threshold = 0.1  # Adjust this threshold based on desired sensitivity

    # Calculate the potential actions
    actions_mbps = [20, 50, 80, 120]

    # Initialize variables to track the best action
    best_action_index = 0
    min_cost = float('inf')

    # Iterate through each possible action
    for i, action_mbps in enumerate(actions_mbps):
        # Calculate the potential new allocation
        new_alloc = alloc_current + action_mbps

        # Check if the new allocation exceeds available bandwidth
        if new_alloc > bw_available:
            continue

        # Calculate the estimated cost for this action
        # Latency degradation is proportional to the new allocation
        latency_degradation = new_alloc / bw_available

        # Degraded state is 1 for every interval the rebuild is unfinished
        degraded_state = 1 if progress < 1 else 0

        # Calculate latency violation
        latency_violation = 1 if latency_now > latency_target else 0

        # Calculate the total cost
        cost = 1.0 * latency_degradation + 0.1 * degraded_state + 10.0 * latency_violation

        # Update the best action if this one has a lower cost
        if cost < min_cost:
            min_cost = cost
            best_action_index = i

    return best_action_index
```

### Explanation:

1. **Observations**: The function extracts all relevant observations from the input `obs` array.
2. **Latency Violation Threshold**: A threshold is set to determine when latency violation occurs.
3. **Actions**: The possible actions (rebuild rates) are defined.
4. **Cost Calculation**: For each possible action, the function calculates the potential new allocation and checks if it exceeds the available bandwidth. If not, it estimates the cost based on latency degradation, degraded state, and latency violation.
5. **Best Action Selection**: The function selects the action with the lowest estimated cost.

This policy aims to balance the rebuild speed with latency constraints, ensuring that the system operates efficiently while maintaining performance.