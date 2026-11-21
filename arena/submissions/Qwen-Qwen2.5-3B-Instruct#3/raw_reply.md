```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    
    # Extract relevant fields from the observation
    fg_read_iops = obs[0]
    fg_write_iops = obs[1]
    fg_bw = obs[2]
    latency_now = obs[3]
    hazard = obs[5]
    redundancy_left = obs[7]
    progress = obs[8]
    work_remaining = obs[9]
    alloc_current = obs[10]

    # Safety Filter Handling
    if hazard > 0.9:
        return alloc_current  # Raise the filter
    
    # Determine the best action based on the current state
    if redundancy_left < 0.1:  # If redundancy is low, prioritize recovery
        return 3  # Choose the highest possible rate to finish quickly
    elif progress > 0.95:  # If nearly finished, reduce the rate to avoid degradation
        return 0
    elif fg_bw > 0.5:  # If there's enough spare capacity, choose a moderate rate
        return 1
    else:  # Otherwise, choose the lowest rate to avoid excessive latency
        return 2
```

This function attempts to balance between finishing the rebuild quickly while minimizing degradation and avoiding latency violations, taking into account the given constraints and the provided observation fields.