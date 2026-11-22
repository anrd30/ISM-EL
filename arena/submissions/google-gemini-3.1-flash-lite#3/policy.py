import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """
    Return 0, 1, 2 or 3.
    
    Strategy:
    1. Assess latency headroom: If latency is already near or above the target, 
       minimize rebuild rate to avoid heavy latency_violation penalties.
    2. Assess urgency: If progress is low and work remaining is high, prioritize speed.
    3. Use bw_available (headroom) as a primary signal for safe rebuild rates.
    """
    
    latency_now = obs[3]
    latency_target = obs[4]
    bw_available = obs[11]
    work_remaining = obs[9]
    
    # 1. Safety First: If latency is at or above target, minimize rebuild load.
    # We use a small buffer (0.95) to be conservative.
    if latency_now > (latency_target * 0.95):
        return 0
    
    # 2. Heuristic: Map available bandwidth to speed
    # bw_available is [0, 1]. 
    # High availability allows for higher rebuild rates.
    if bw_available > 0.7:
        rate_idx = 3
    elif bw_available > 0.4:
        rate_idx = 2
    elif bw_available > 0.2:
        rate_idx = 1
    else:
        rate_idx = 0
        
    # 3. Urgency Adjustment:
    # If we have a lot of work remaining and latency is comfortable, 
    # boost the rate slightly.
    if work_remaining > 0.5 and rate_idx < 3:
        if latency_now < (latency_target * 0.7):
            rate_idx += 1
            
    # 4. Final check against latency risk
    # If latency is trending high, throttle back
    if latency_now > (latency_target * 0.8) and rate_idx > 1:
        rate_idx = 1
        
    return int(max(0, min(3, rate_idx)))
