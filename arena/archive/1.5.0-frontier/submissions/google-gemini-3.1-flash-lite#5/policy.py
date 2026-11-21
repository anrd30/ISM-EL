import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """
    Control policy for disk rebuild rate.
    
    Strategy:
    1. Check for latency violations: If latency is already near or exceeding 
       the target, aggressively minimize rebuild rate to recover.
    2. Assess headroom: Use 'bw_available' and 'fg_bw' to determine if the 
       system can handle a higher rebuild rate without pushing latency.
    3. Urgency: Increase rebuild rate as 'work_remaining' is high or 
       'hazard' (failure risk) increases.
    """
    
    # Extract key observations
    latency_now = obs[3]
    latency_target = obs[4]
    hazard = obs[5]
    work_remaining = obs[9]
    bw_available = obs[11]
    
    # 1. Safety First: If latency is close to or exceeding target, 
    # prioritize reducing load immediately.
    # We use a buffer (0.9) to be conservative against the target.
    if latency_now > (latency_target * 0.9):
        return 0
    
    # 2. Heuristic for throughput:
    # If we have significant headroom, we can afford a higher rebuild rate.
    # If latency is healthy, we balance progress vs risk.
    
    # Thresholds based on normalized observations
    # High headroom allows for higher rebuild rates
    if bw_available > 0.7:
        rate_choice = 3
    elif bw_available > 0.4:
        rate_choice = 2
    else:
        rate_choice = 1
        
    # 3. Increase urgency if failure hazard is high or work is still significant
    # and we aren't currently violating latency.
    if hazard > 0.6 or work_remaining > 0.5:
        rate_choice = min(rate_choice + 1, 3)
        
    # 4. Final safety check: if latency is moderate, don't push to max (3)
    if latency_now > (latency_target * 0.7) and rate_choice == 3:
        rate_choice = 2
        
    return int(rate_choice)
