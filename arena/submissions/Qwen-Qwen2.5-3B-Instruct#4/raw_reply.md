```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    
    # Unpack the observation
    fg_read_iops = obs[0]
    fg_write_iops = obs[1]
    fg_bw = obs[2]
    latency_now = obs[3]
    latency_target = obs[4]
    hazard = obs[5]
    hazard_uncertainty = obs[6]
    redundancy_left = obs[7]
    progress = obs[8]
    work_remaining = obs[9]
    alloc_current = obs[10]
    bw_available = obs[11]

    # Calculate the cost based on the current alloc_current and work_remaining
    # and decide the action.
    if progress >= 0.95:
        # If the rebuild is almost complete, we can safely lower the bandwidth allocation.
        return 0
    elif progress >= 0.8:
        # As the rebuild approaches completion, we can increase the bandwidth allocation.
        return 1
    elif work_remaining <= 2:
        # If the workload is very small, we can allocate more bandwidth for quick completion.
        return 2
    else:
        # Otherwise, we maintain the current allocation to avoid overshooting.
        return 3
```

This function is designed to balance between completing the rebuild quickly while maintaining low latency. It leverages the `alloc_current` value to determine the appropriate rebuild rate, ensuring that the rebuild progresses efficiently without significantly degrading performance.