```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    actions_mbps = [20, 50, 80, 120]
    
    # Extract relevant fields from obs
    fg_read_iops = obs[0]
    fg_write_iops = obs[1]
    fg_bw = obs[2]
    redundancy_left = obs[7]
    progress = obs[8]
    work_remaining = obs[9]
    alloc_current = obs[10]

    # Determine the appropriate action based on the observed state
    if redundancy_left < 0.1:  # If redundancy is critically low, stop all activity to avoid further risk
        return 0
    elif progress > 0.95:  # If almost done, keep the current allocation to avoid sudden slowdowns
        return alloc_current
    elif fg_bw < 0.2:  # If foreground bandwidth is very low, increase speed to catch up
        return min(alloc_current + 1, 3)
    else:  # Otherwise, use the current allocation as long as it's not excessive
        return alloc_current

    # Return default if none of the conditions apply, which should not happen due to deterministic nature
    return 0
```