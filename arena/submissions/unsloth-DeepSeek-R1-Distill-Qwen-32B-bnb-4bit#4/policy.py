import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    if obs[3] > obs[4]:
        return 0
    if obs[5] > 0.7:  # High hazard
        if obs[11] > 0.4:
            return 3
        else:
            return 2
    else:
        if obs[11] > 0.75:
            return 3
        elif obs[11] > 0.5:
            return 2
        elif obs[11] > 0.25:
            return 1
        else:
            return 0
