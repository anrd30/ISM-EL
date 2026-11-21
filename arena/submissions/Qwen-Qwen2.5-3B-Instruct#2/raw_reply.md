```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    
    # Define the target bandwidth allocation
    target_bw = 80  # MBps
    
    # Calculate available bandwidth
    bw_available = obs[10]
    
    # If there's enough available bandwidth, use the highest possible rate
    if bw_available >= target_bw:
        alloc_current = 3
    else:
        # Otherwise, choose the next highest rate
        alloc_current = min(3, max(0, obs[7] - 1))  # Subtract 1 to ensure non-negative
        
    # Store the allocated rate for later reference
    state['alloc_current'] = alloc_current
    
    return alloc_current
```