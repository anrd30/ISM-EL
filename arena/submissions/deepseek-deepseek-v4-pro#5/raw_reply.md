```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    # unpack relevant observations
    latency_now = obs[3]
    latency_target = obs[4]
    bw_available = obs[11]

    margin = latency_target - latency_now

    # If we are already violating or right at the limit, drop to minimum
    if margin < 0.0:
        return 0

    # Near the latency target: very conservative
    if margin < 0.05:
        if bw_available < 0.3:
            return 0
        else:
            return 1

    # Moderate margin: allow some rebuild, but not the maximum rate
    if margin < 0.15:
        if bw_available >= 0.7:
            return 2
        elif bw_available >= 0.4:
            return 1
        else:
            return 0

    # Safe margin: use higher rates when headroom permits
    if bw_available >= 0.8:
        return 3
    elif bw_available >= 0.55:
        return 2
    elif bw_available >= 0.3:
        return 1
    else:
        return 0
```