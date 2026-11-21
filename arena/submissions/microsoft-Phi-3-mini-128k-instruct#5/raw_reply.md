```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    # Extracting the necessary fields from the observation vector
    fg_read_iops, fg_write_iops, fg_bw, latency_now, latency_target, hazard, hazard_uncertainty, redundancy_left, progress, work_remaining, alloc_current, bw_available = obs
    
    # Constants for the cost function
    latency_cost_factor = 1.0
    degraded_state_cost_factor = 0.1
    latency_violation_cost_factor = 10.0
    
    # Calculating the cost for each action
    costs = []
    for action_index in range(len(ACTIONS_MBPS)):
        # Update the alloc_current based on the chosen action
        new_alloc_current = alloc_current + ACTIONS_MBPS[action_index]
        
        # Estimate the new bandwidth available after applying the action
        new_bw_available = bw_available - new_alloc_current
        
        # Estimate the new latency based on the new bandwidth available
        # This is a simplistic linear estimation and should be replaced with a proper model
        latency_estimate = latency_now + (fg_bw * (1 - new_bw_available / bw_available))
        
        # Calculate cost components
        latency_degradation = (latency_estimate - latency_target) * latency_cost_factor if latency_estimate > latency_target else 0
        degraded_state = (1 - progress) * degraded_state_cost_factor
        latency_violation = max(latency_target - latency_estimate, 0) * latency_violation_cost_factor
        
        # Calculate total cost for this action
        cost = latency_degradation + degraded_state + latency_violation
        costs.append(cost)
    
    # Choose the action with the lowest cost
    chosen_action_index = np.argmin(costs)
    
    return chosen_action_index

# Note: The constants ACTIONS_MBPS and the function ACTIONS_MBPS[action_index]
# are used here as placeholders to indicate the action's MBPS rate.
# These should be replaced with the actual values from the system's configuration.
ACTIONS_MBPS = [20, 50, 80, 120]
```

This function implements a deterministic policy that chooses the action with the lowest estimated cost based on the current observation. It uses a simplistic linear model to estimate the impact of each action on latency, which should be replaced with a more accurate model for real-world applications. The function also assumes that the `state` dictionary is not used, as the policy is fully determined by the current observation and does not require any stateful computation.