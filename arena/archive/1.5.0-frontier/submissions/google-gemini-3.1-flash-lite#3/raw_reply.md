```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """
    Return 0, 1, 2 or 3 (rebuild rates).
    
    Strategy:
    1. Check for latency violations. If we are violating the target, 
       reduce rebuild rate immediately to minimize the heavy penalty.
    2. Assess headroom (bw_available). If we have plenty of bandwidth 
       and no violation, we can afford a higher rebuild rate.
    3. Factor in urgency: as progress is low or hazard is high, 
       increase pressure to rebuild, but cap it based on current latency.
    """
    
    # Extract relevant metrics
    latency_now = obs[3]
    latency_target = obs[4]
    hazard = obs[5]
    progress = obs[8]
    bw_available = obs[11]
    
    # 1. Safety check: If latency is near or above target, minimize rebuild impact
    # We use a buffer (0.95) to be proactive before hitting the hard penalty
    if latency_now > (latency_target * 0.95):
        return 0
    
    # 2. Heuristic: Determine base aggressiveness
    # If hazard is high or we are early in the process, we want to go faster.
    # If progress is near completion, we can slow down to ensure no late violations.
    urgency = (hazard * 0.6) + ((1.0 - progress) * 0.4)
    
    # 3. Determine potential rate based on available headroom
    # bw_available is capacity minus foreground load
    if bw_available > 0.7:
        # High headroom: allow high rates
        if urgency > 0.6:
            return 3
        return 2
    elif bw_available > 0.4:
        # Medium headroom: moderate rates
        if urgency > 0.7:
            return 2
        return 1
    else:
        # Low headroom: conservative
        return 0
```