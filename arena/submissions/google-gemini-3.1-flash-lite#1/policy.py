import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """
    Return 0, 1, 2 or 3.
    
    Strategy:
    1. Assess latency headroom: If latency is already near or above target,
       prioritize safety by reducing throughput.
    2. Assess urgency: If work remains high and hazard is increasing, 
       increase throughput.
    3. Use bw_available (headroom) as a primary constraint to avoid latency_violation.
    """
    
    # Unpack observation for readability
    latency_now = obs[3]
    latency_target = obs[4]
    hazard = obs[5]
    work_remaining = obs[9]
    bw_available = obs[11]
    
    # Calculate a safety margin for latency
    # If latency_now > latency_target, latency_ratio > 1
    latency_ratio = latency_now / (latency_target + 1e-6)
    
    # Determine risk profile
    # High hazard or high work remaining suggests we want faster rebuilds
    urgency = (hazard + work_remaining) / 2.0
    
    # Decision logic:
    # We want to be aggressive (higher index) if:
    # - Latency is low (latency_ratio is small)
    # - Headroom is high (bw_available is large)
    # - Urgency is high
    
    if latency_ratio > 0.95:
        # Critical: minimize impact immediately
        return 0
    
    if latency_ratio > 0.8:
        # Caution: moderate impact
        return 1
    
    # If latency is healthy, scale based on urgency and available bandwidth
    # Map urgency (0-1) and bw_available (0-1) to actions
    score = (urgency * 0.6) + (bw_available * 0.4)
    
    if score > 0.7:
        return 3
    elif score > 0.4:
        return 2
    else:
        return 1
