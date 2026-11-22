import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    latency = float(obs[3])
    target = float(obs[4])

    if latency > target:
        return 0

    if target > 1e-8:
        latency_risk = latency / target
    else:
        latency_risk = 1.0

    foreground_load = float(obs[2])
    headroom = float(obs[11])
    load_risk = 0.5 * foreground_load + 0.5 * (1.0 - headroom)

    risk = max(latency_risk, 0.75 * load_risk + 0.25 * latency_risk)

    if risk >= 0.97:
        return 0
    if risk >= 0.88:
        return 1
    if risk >= 0.78:
        return 2
    return 3