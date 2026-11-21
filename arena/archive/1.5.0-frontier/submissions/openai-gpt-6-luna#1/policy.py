import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    x = np.asarray(obs, dtype=float)
    latency = float(x[3])
    target = float(x[4])
    headroom = float(x[11])

    if not np.isfinite(latency) or not np.isfinite(target):
        return 0

    margin = target - latency

    if margin <= 0.02:
        action = 0
    elif margin <= 0.06:
        action = 1
    elif margin <= 0.12:
        action = 2
    else:
        action = 3

    if np.isfinite(headroom):
        if headroom < 0.10:
            action = min(action, 0)
        elif headroom < 0.22:
            action = min(action, 1)
        elif headroom < 0.38:
            action = min(action, 2)

    return int(action)
