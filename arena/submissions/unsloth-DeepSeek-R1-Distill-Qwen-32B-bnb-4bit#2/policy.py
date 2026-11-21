import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    if obs[3] > obs[4]:  # Current latency exceeds target
        return 0
    progress = obs[8]
    if progress > 0.8:
        return 0
    elif progress > 0.6:
        return 1
    else:
        bw_available = obs[11]
        if bw_available > 0.8:
            return 3
        elif bw_available > 0.6:
            return 2
        elif bw_available > 0.4:
            return 1
        else:
            return 0
