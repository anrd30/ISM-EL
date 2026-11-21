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
    
    # Define actions
    ACTIONS_MBPS = [20, 50, 80, 120]
    
    # Calculate potential new allocations
    new_allocations = [min(alloc_current + d, bw_available) for d in [0, 30, 60, 90]]
    
    # Initialize variables to track the best action
    min_cost = float('inf')
    best_action = 0
    
    # Evaluate each action
    for i, new_alloc in enumerate(new_allocations):
        # Calculate latency degradation
        latency_degradation = max(0, new_alloc - bw_available)
        
        # Calculate degraded state
        degraded_state = 1 if progress < 1 else 0
        
        # Calculate latency violation
        latency_violation = 1 if latency_now > latency_target else 0
        
        # Calculate cost
        cost = 1.0 * latency_degradation + 0.1 * degraded_state + 10.0 * latency_violation
        
        # Update best action if this one is better
        if cost < min_cost:
            min_cost = cost
            best_action = i
    
    # Return the index of the best action
    return best_action
