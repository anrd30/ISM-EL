import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """
    Return 0, 1, 2 or 3.
    
    Strategy:
    1. Assess latency headroom: If latency is already near or above target,
       prioritize safety/latency by choosing the lowest rate.
    2. Assess progress/urgency: If the rebuild is nearly finished or work is low,
       increase the rate to finish quickly.
    3. Assess available capacity: If bw_available is high, we can afford a higher
       rebuild rate without significant latency degradation.
    """
    
    latency_now = obs[3]
    latency_target = obs[4]
    bw_available = obs[11]
    work_remaining = obs[9]
    hazard = obs[5]
    
    # Calculate a safety margin for latency
    # If latency_now > latency_target, we are already violating.
    if latency_now > (latency_target * 0.95):
        return 0
    
    # If we have very little headroom, stay conservative
    if bw_available < 0.2:
        return 0
    
    # If we have high urgency (hazard is high or work is almost done),
    # push harder if latency allows.
    if hazard > 0.6 or work_remaining < 0.1:
        if bw_available > 0.6:
            return 3
        else:
            return 2
            
    # Default behavior based on available headroom
    if bw_available > 0.7:
        return 3
    elif bw_available > 0.4:
        return 2
    elif bw_available > 0.2:
        return 1
    else:
        return 0
