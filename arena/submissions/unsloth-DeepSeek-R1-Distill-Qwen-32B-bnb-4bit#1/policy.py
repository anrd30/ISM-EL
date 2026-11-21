import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    if obs[3] > obs[4]:  # Latency now exceeds target
        if obs[11] < 0.3:  # Low available bandwidth
            return 0
        else:
            return 1
    else:
        available_bandwidth = obs[11]
        work_remaining = obs[9]
        hazard = obs[5]
        priority = work_remaining * hazard * available_bandwidth
        if priority > 0.8:
            return 3
        elif priority > 0.6:
            return 2
        elif priority > 0.4:
            return 1
        else:
            return 0
