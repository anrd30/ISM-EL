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
    
    # Determine the current state
    if progress >= 0.95:
        return 3  # Fastest rebuild rate if close to completion
    
    if latency_now > latency_target:
        return 0  # Slowest rebuild rate if latency is already violated
    
    if hazard > 0.1:
        return 2  # Moderate rebuild rate if high hazard
    
    if redundancy_left < 0.5:
        return 1  # Intermediate rebuild rate if low redundancy left
    
    return 1  # Default to intermediate rebuild rate
