To solve this problem, we need to create a policy function that determines the optimal rebuild rate based on the given observations. The goal is to minimize the total cost, which is influenced by latency degradation, degraded state, and latency violations.

Here's the implementation of the policy function:

```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    
    # Extract relevant observations
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
    
    # Define the actions
    ACTIONS_MBPS = [20, 50, 80, 120]
    
    # Calculate the cost for each action
    costs = []
    for action in ACTIONS_MBPS:
        # Calculate latency degradation
        if action > bw_available:
            latency_degradation = 1.0
        else:
            latency_degradation = action / bw_available
        
        # Calculate degraded state
        degraded_state = 1 if progress < 1 else 0
        
        # Calculate latency violation
        if latency_now > latency_target:
            latency_violation = 1.0
        else:
            latency_violation = 0.0
        
        # Calculate total cost
        cost = 1.0 * latency_degradation + 0.1 * degraded_state + 10.0 * latency_violation
        costs.append(cost)
    
    # Choose the action with the minimum cost
    best_action_index = np.argmin(costs)
    
    return best_action_index
```

### Explanation:

1. **Observation Extraction**: We extract the relevant observations from the input array `obs`.
2. **Action Definition**: We define the possible rebuild rates (`ACTIONS_MBPS`).
3. **Cost Calculation**: For each possible action, we calculate the cost based on the given formula:
   - **Latency Degradation**: This is calculated based on how much of the available bandwidth is used by the rebuild.
   - **Degraded State**: This is 1 if the rebuild is not yet complete, otherwise 0.
   - **Latency Violation**: This is 1 if the current latency exceeds the target, otherwise 0.
4. **Cost Comparison**: We compare the costs for each action and choose the one with the minimum cost.
5. **Return Action**: The function returns the index of the action with the minimum cost.

This policy aims to balance the trade-off between finishing the rebuild quickly and keeping the latency below the target, while also considering the available bandwidth and the current state of the system.