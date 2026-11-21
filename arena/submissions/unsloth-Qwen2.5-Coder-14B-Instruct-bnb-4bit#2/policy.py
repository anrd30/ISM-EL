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
    
    # Define the possible actions
    ACTIONS_MBPS = [20, 50, 80, 120]
    
    # Calculate the latency violation potential
    latency_violation_potential = max(0, latency_now - latency_target)
    
    # Calculate the cost for each action
    costs = []
    for action in ACTIONS_MBPS:
        # Calculate the new latency degradation
        new_latency_degradation = action / bw_available
        
        # Calculate the cost
        cost = 1.0 * new_latency_degradation + 0.1 * (1 - progress) + 10.0 * latency_violation_potential
        
        costs.append(cost)
    
    # Return the action with the minimum cost
    return np.argmin(costs)
