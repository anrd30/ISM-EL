import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    
    # Extract observations
    fg_bw = obs[2]
    latency_now = obs[3]
    latency_target = obs[4]
    hazard = obs[5]
    redundancy_left = obs[7]
    progress = obs[8]
    work_remaining = obs[9]
    alloc_current = obs[10]
    
    # Define thresholds and penalties
    latency_threshold = 0.9 * latency_target
    latency_violation = int(latency_now > latency_threshold)
    degradation_penalty = min(fg_bw, 1)  # Assuming fg_bw is normalized between 0 and 1
    
    # Determine the state of the rebuild
    degraded_state = 1 if progress < 1 else 0
    
    # Calculate cost components
    cost_latency_degradation = degradation_penalty
    cost_degraded_state = degraded_state
    cost_latency_violation = latency_violation * 10.0
    
    total_cost = cost_latency_degradation + cost_degraded_state + cost_latency_violation
    
    # Adjust rebuild rate based on current conditions
    if redundancy_left < 0.2:
        desired_rate = 3  # Highest priority to finish quickly
    elif latency_now > latency_target and progress < 0.8:
        desired_rate = 1  # Balance between speed and latency
    else:
        desired_rate = np.argmax([cost_latency_degradation, cost_degraded_state, cost_latency_violation])
    
    return int(desired_rate)
