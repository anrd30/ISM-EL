```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    # Extract observations
    fg_bw = obs[2]
    latency_now = obs[3]
    latency_target = obs[4]
    progress = obs[8]
    
    # Determine the current rebuild rate
    alloc_current = obs[10]
    
    # Calculate the cost components
    if latency_now > latency_target:
        latency_violation = 1.0
    else:
        latency_violation = 0.0
    
    degraded_state = 1 - progress
    
    # Adjust for latency degradation and remaining work
    if fg_bw < 0.5:
        latency_degradation = 0.5 * (1 - fg_bw)
    elif fg_bw < 0.75:
        latency_degradation = 0.8 * (1 - fg_bw)
    else:
        latency_degradation = 1.2 * (1 - fg_bw)
    
    # Calculate the total cost
    cost = latency_degradation + degraded_state + 10 * latency_violation
    
    # Adjust the rebuild rate based on the current cost
    if cost < 5.0:
        target_rate = min(120, int(alloc_current * 1.2))
    elif cost < 8.0:
        target_rate = min(120, int(alloc_current * 1.1))
    else:
        target_rate = min(120, alloc_current)
    
    # Map the target rate to an index
    if target_rate <= 50:
        return 0
    elif target_rate <= 80:
        return 1
    elif target_rate <= 120:
        return 2
    else:
        return 3
```