```python
import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """
    Deterministic control policy for disk rebuild rate.
    Returns an action index 0..3, corresponding to 20, 50, 80, 120 MBps.
    """
    # Extract relevant normalised fields
    latency_now = obs[3]
    latency_target = obs[4]
    bw_available = obs[11]
    alloc_current = obs[10]

    # Avoid division by zero
    target = max(latency_target, 1e-6)
    margin_ratio = (latency_target - latency_now) / target

    # Decide desired normalised rebuild rate
    if margin_ratio < 0.0:
        # Violation – reduce to zero immediately
        desired_norm = 0.0
    elif margin_ratio < 0.1:
        # Close to violation – back off a little
        desired_norm = alloc_current * 0.8
        desired_norm = max(desired_norm, 0.0)
    elif margin_ratio > 0.5:
        # Plenty of headroom – use all available bandwidth
        desired_norm = bw_available
    else:
        # Moderate margin – stay at current rate
        desired_norm = alloc_current

    # Normalised values for the discrete actions (20, 50, 80, 120)
    # Assumed linear mapping from MBps to [0, 1]
    actions_norm = np.array([0.0, 0.3, 0.6, 1.0], dtype=np.float32)

    # Pick the largest action whose normalised rate does not exceed the desired rate
    action = 0
    for i in range(4):
        if actions_norm[i] <= desired_norm:
            action = i
    return int(action)
```