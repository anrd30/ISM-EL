import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    latency_now = float(obs[3])
    latency_target = float(obs[4])
    bw_available = float(obs[11])

    if not np.isfinite(latency_now):
        latency_now = 1.0
    if not np.isfinite(latency_target):
        latency_target = 0.0
    if not np.isfinite(bw_available):
        bw_available = 0.0

    margin = (latency_target - latency_now) / max(abs(latency_target), 1e-6)

    if margin < 0.025:
        latency_cap = 0
    elif margin < 0.07:
        latency_cap = 1
    elif margin < 0.14:
        latency_cap = 2
    else:
        latency_cap = 3

    if bw_available < 0.12:
        bandwidth_cap = 0
    elif bw_available < 0.28:
        bandwidth_cap = 1
    elif bw_available < 0.48:
        bandwidth_cap = 2
    else:
        bandwidth_cap = 3

    return int(min(latency_cap, bandwidth_cap))