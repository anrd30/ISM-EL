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
    
    # Define the action space
    ACTIONS_MBPS = [20, 50, 80, 120]
    
    # Calculate the cost of each action
    costs = []
    for action in ACTIONS_MBPS:
        # Calculate latency degradation
        latency_degradation = max(0, action - bw_available)
        
        # Calculate degraded state
        degraded_state = 1 if progress < 1 else 0
        
        # Calculate latency violation
        latency_violation = 1 if latency_now > latency_target else 0
        
        # Calculate total cost
        cost = 1.0 * latency_degradation + 0.1 * degraded_state + 10.0 * latency_violation
        costs.append(cost)
    
    # Choose the action with the minimum cost
    best_action_index = np.argmin(costs)
    
    return best_action_index
```