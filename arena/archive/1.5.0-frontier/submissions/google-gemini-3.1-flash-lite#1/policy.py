import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """
    Control policy for storage rebuild.
    
    Strategy:
    1. Prioritize latency safety: if current latency is approaching the target,
       aggressively throttle the rebuild (index 0 or 1).
    2. Use headroom: if bw_available is high, we can afford higher rebuild rates.
    3. Urgency: if work_remaining is high or redundancy_left is low, increase
       the rebuild rate, provided latency permits.
    4. Smoothing: use the state dict if necessary (not strictly needed here, 
       but available).
    """
    
    # Extract relevant observations
    # Index 3: latency_now, 4: latency_target, 11: bw_available
    # Index 9: work_remaining, 7: redundancy_left
    latency_now = obs[3]
    latency_target = obs[4]
    bw_available = obs[11]
    work_remaining = obs[9]
    redundancy_left = obs[7]
    
    # Calculate safety margin for latency
    # If latency_now is close to latency_target, we must be conservative.
    latency_ratio = latency_now / (latency_target + 1e-6)
    
    # Decision logic:
    # If latency is already at or near the limit, we MUST minimize impact.
    if latency_ratio > 0.95:
        return 0
    
    # If latency is moderately high, use a medium-low rate
    if latency_ratio > 0.8:
        return 1
        
    # If latency is low, we check our urgency
    # High work remaining or low redundancy justifies higher rates
    urgency = (1.0 - redundancy_left) + work_remaining
    
    # If we have plenty of bandwidth and low latency, go fast
    if bw_available > 0.6 and urgency > 0.4:
        return 3
    
    # Default to medium-high if latency is safe
    if bw_available > 0.3:
        return 2
        
    return 1
