Here is a possible implementation of the control policy for the storage system:

```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    fg_bw = obs[2]
    latency_now = obs[3]
    latency_target = obs[4]
    hazard = obs[5]
    progress = obs[8]
    work_remaining = obs[9]
    
    # Determine if we are close to the end of the rebuild
    if progress > 0.7:
        return 3  # Fastest rebuild rate
    
    # Check if latency is already above the target
    if latency_now > latency_target:
        return 2  # Moderate rebuild rate
    
    # Balance between finishing quickly and maintaining low latency
    if fg_bw > 0.5:
        return 1  # Slower rebuild rate to maintain lower bandwidth usage
    else:
        return 0  # Faster rebuild rate to finish sooner
```

### Explanation:

1. **Progress Check**: If the rebuild progress is greater than 70%, it selects the fastest rebuild rate (index 3) to complete the rebuild quickly.

2. **Latency Check**: If the current latency exceeds the target, it selects a moderate rebuild rate (index 2) to keep the system responsive.

3. **Bandwidth Check**: If the foreground bandwidth is high (more than 50%), it selects a slower rebuild rate (index 1) to avoid overwhelming the system and causing higher latency.

4. **Default Action**: For other cases, it selects the fastest rebuild rate (index 0) to balance between finishing quickly and maintaining low latency.

This policy aims to find a balance between completing the rebuild as soon as possible and keeping the system responsive by not overloading the disk with too much bandwidth.