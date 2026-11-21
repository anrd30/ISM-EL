import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    
    latency_now = obs[3]
    latency_target = obs[4]
    fg_bw = obs[2]
    progress = obs[8]
    alloc_current = obs[10]
    
    if progress >= 0.95:
        return 3  # Full speed to finish quickly
    
    if latency_now > latency_target:
        return 0  # Minimum bandwidth to reduce latency impact
    
    if fg_bw < 0.25:
        return 2  # Moderate speed when load is low
    
    return 1  # Balanced speed for normal conditions
