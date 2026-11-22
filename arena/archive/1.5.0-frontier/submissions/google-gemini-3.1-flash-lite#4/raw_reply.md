```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """
    Control policy for disk rebuild rate.
    
    Strategy:
    1. Check for latency violations: If latency is already exceeding or very close
       to the target, prioritize reducing load immediately (select 0 or 1).
    2. Assess urgency: If redundancy is low or hazard is high, prioritize 
       finishing the rebuild faster (select 2 or 3).
    3. Use headroom: Use available bandwidth (bw_available) to decide if the 
       system can handle a higher rebuild rate without violating latency.
    """
    
    # Extract key observations
    latency_now = obs[3]
    latency_target = obs[4]
    hazard = obs[5]
    redundancy_left = obs[7]
    work_remaining = obs[9]
    bw_available = obs[11]
    
    # 1. Safety Check: If latency is near or above target, minimize rebuild impact.
    # We use a buffer (0.9) to stay proactive before hitting the hard limit.
    if latency_now > (latency_target * 0.9):
        return 0
    
    # 2. Urgency Check: If redundancy is critically low or hazard is high, 
    # we must push the rebuild faster regardless of minor latency impact.
    # High hazard or low redundancy suggests we should favor higher rates.
    urgency = (1.0 - redundancy_left) + hazard
    
    # 3. Decision Logic based on headroom and urgency
    # If we have plenty of headroom, we can afford higher rates.
    # If urgency is high, we take higher rates even with moderate headroom.
    
    if urgency > 0.6:
        # High danger: push hard if possible
        if bw_available > 0.4:
            return 3
        return 2
    elif urgency > 0.3:
        # Moderate danger
        if bw_available > 0.6:
            return 3
        if bw_available > 0.3:
            return 2
        return 1
    else:
        # Low danger: prioritize latency and finish at a moderate pace
        if bw_available > 0.7:
            return 2
        return 1

    return 1
```