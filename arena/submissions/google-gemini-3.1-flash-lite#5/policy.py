import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """
    Return 0, 1, 2 or 3.
    
    Strategy:
    - Prioritize avoiding latency violations (index 4 > 3).
    - If latency is safe, accelerate rebuild based on remaining work (index 9)
      and available headroom (index 11).
    - Use the current applied rate (index 10) to maintain stability.
    """
    
    latency_now = obs[3]
    latency_target = obs[4]
    bw_available = obs[11]
    work_remaining = obs[9]
    
    # 1. Safety First: If we are close to or exceeding the latency target, 
    # throttle back immediately to minimize the heavy 10.0 penalty.
    # We use a buffer (0.9) to be conservative.
    if latency_now > (0.9 * latency_target):
        return 0
    
    # 2. Progress-based acceleration: 
    # If the system is healthy, we want to finish the rebuild.
    # Higher bw_available allows for higher rebuild rates.
    
    # Calculate a score for how much we can push the rebuild.
    # We want higher rates if work is high and headroom is high.
    if work_remaining > 0.6:
        if bw_available > 0.7:
            action = 3
        elif bw_available > 0.4:
            action = 2
        else:
            action = 1
    elif work_remaining > 0.3:
        if bw_available > 0.5:
            action = 2
        else:
            action = 1
    else:
        # Near completion, be cautious to avoid late-stage latency spikes.
        action = 1
        
    # 3. Final check: Ensure we don't exceed a reasonable rate if latency is rising.
    # If latency is trending towards 80% of target, limit to index 1.
    if latency_now > (0.8 * latency_target):
        return min(action, 1)
        
    return int(action)
