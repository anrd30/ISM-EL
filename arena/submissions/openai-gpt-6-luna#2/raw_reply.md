import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    latency = float(obs[3])
    target = float(obs[4])
    available = float(obs[11])

    if not np.isfinite(latency) or not np.isfinite(target) or not np.isfinite(available):
        return 0

    margin = (target - latency) / max(abs(target), 1e-6)

    if margin <= 0.0 or available < 0.12:
        return 0
    if margin < 0.10 or available < 0.25:
        return 1
    if margin < 0.22 or available < 0.45:
        return 2
    return 3