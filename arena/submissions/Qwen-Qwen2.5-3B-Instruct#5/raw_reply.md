```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    
    # Extract relevant observables
    fg_read_iops = obs[0]
    fg_write_iops = obs[1]
    fg_bw = obs[2]
    progress = obs[8]
    work_remaining = obs[9]
    alloc_current = obs[10]

    # Determine the action based on the state
    if progress < 0.1:  # Early stages of rebuild, prioritize speed
        return ACTIONS_MBPS.index(3)
    elif progress > 0.9:  # Near completion, prioritize stability
        return ACTIONS_MBPS.index(0)
    else:
        # Balance between speed and stability
        if fg_bw > 0.5:  # If the foreground load is heavy, slow down
            return ACTIONS_MBPS.index(0)
        else:  # Otherwise, prioritize speed
            return ACTIONS_MBPS.index(3)

    # Default to the highest possible rate
    return ACTIONS_MBPS.index(3)
```

This policy attempts to balance between minimizing latency degradation and ensuring the rebuild does not violate the latency target. It prioritizes higher speeds in early stages and near completion, and balances between the two states based on the current foreground bandwidth and progress.