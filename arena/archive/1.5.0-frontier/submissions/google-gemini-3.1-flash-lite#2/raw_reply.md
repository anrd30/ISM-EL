```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """
    Return 0, 1, 2 or 3 (rebuild rates).
    
    Strategy:
    1. Check for latency violations. If latency_now > latency_target, prioritize
       reducing load immediately.
    2. Assess headroom (bw_available). If there is significant headroom, we can
       afford higher rebuild rates.
    3. Balance progress vs. risk. As progress increases, we can afford slightly
       more aggressive rates, but if hazard is high, prioritize safety.
    """
    
    # Extract key observations
    latency_now = obs[3]
    latency_target = obs[4]
    hazard = obs[5]
    progress = obs[8]
    bw_available = obs[11]
    
    # 1. Hard constraint: If latency violation is imminent or occurring,
    # minimize rebuild impact.
    if latency_now >= latency_target:
        return 0
    
    # 2. Safety constraint: If hazard is high, we want to finish faster,
    # but only if the system can handle it without violating latency.
    # If latency is close to target, stay conservative.
    latency_margin = latency_target - latency_now
    
    # 3. Decision logic based on headroom and progress
    # We want to use bw_available to drive the rebuild.
    # High bw_available means we can afford higher rates.
    
    if latency_margin < 0.2:
        # Latency is tight, stay low
        return 0 if bw_available < 0.3 else 1
    
    if bw_available > 0.6:
        # Plenty of headroom, push for speed
        return 3
    elif bw_available > 0.3:
        # Moderate headroom
        return 2
    else:
        # Tight headroom
        return 1
```