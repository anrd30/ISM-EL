import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    latency_now = float(obs[3])
    latency_target = float(obs[4])
    bw_available = float(obs[11])
    work_remaining = float(obs[9])

    margin = (latency_target - latency_now) / max(abs(latency_target), 0.05)

    if margin <= 0.0:
        latency_cap = 0
    elif margin < 0.04:
        latency_cap = 0
    elif margin < 0.12:
        latency_cap = 1
    elif margin < 0.25:
        latency_cap = 2
    else:
        latency_cap = 3

    if bw_available < 0.12:
        bandwidth_cap = 0
    elif bw_available < 0.25:
        bandwidth_cap = 1
    elif bw_available < 0.42:
        bandwidth_cap = 2
    else:
        bandwidth_cap = 3

    action = min(latency_cap, bandwidth_cap)

    if work_remaining < 0.02 and margin > 0.0:
        action = min(action + 1, latency_cap, bandwidth_cap)

    return int(action)