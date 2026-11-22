import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """
    Return 0, 1, 2 or 3.
    
    Strategy:
    - Prioritize avoiding latency_violation (index 4).
    - Use bw_available (index 11) to gauge safe headroom.
    - Balance progress (index 8) vs latency risks.
    - If latency is near target, throttle back to 0 or 1.
    - If latency is low and progress is slow, increase rate.
    """
    
    latency_now = obs[3]
    latency_target = obs[4]
    bw_available = obs[11]
    progress = obs[8]
    work_remaining = obs[9]
    
    # Calculate headroom buffer
    # latency_target is the threshold for the 10.0 penalty.
    # We want a safety margin.
    latency_ratio = latency_now / (latency_target + 1e-6)
    
    # If we are already violating or very close (90%+), throttle aggressively
    if latency_ratio > 0.95:
        return 0
    
    # If latency is moderate (70-95%), be conservative
    if latency_ratio > 0.75:
        return 1
        
    # If we have plenty of bandwidth headroom and low latency, 
    # we can afford higher rebuild rates.
    # bw_available is normalized [0, 1].
    if bw_available > 0.6:
        # High headroom, low latency: speed up
        if progress < 0.8:
            return 3
        else:
            return 2
    elif bw_available > 0.3:
        # Moderate headroom
        return 2
    
    # Default fallback to conservative speed
    return 1
