import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    fg_bw = obs[2]
    latency_now = obs[3]
    latency_target = obs[4]
    hazard = obs[5]
    progress = obs[8]
    work_remaining = obs[9]
    
    # Determine if we are close to the end of the rebuild
    if progress > 0.7:
        return 3  # Fastest rebuild rate
    
    # Check if latency is already above the target
    if latency_now > latency_target:
        return 2  # Moderate rebuild rate
    
    # Balance between finishing quickly and maintaining low latency
    if fg_bw > 0.5:
        return 1  # Slower rebuild rate to maintain lower bandwidth usage
    else:
        return 0  # Faster rebuild rate to finish sooner
