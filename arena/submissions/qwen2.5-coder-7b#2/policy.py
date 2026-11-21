import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """Return 0, 1, 2 or 3."""
    # Extract observations
    fg_bw = obs[2]
    latency_now = obs[3]
    latency_target = obs[4]
    hazard = obs[5]
    redundancy_left = obs[7]
    progress = obs[8]
    work_remaining = obs[9]
    alloc_current = obs[10]
    
    # Define thresholds
    latency_threshold = 1.2 * latency_target
    hazard_threshold = 0.1
    
    # Determine the action based on observations and thresholds
    if redundancy_left < 0.2:
        return 3  # Set high rebuild rate to complete quickly
    elif latency_now > latency_threshold or hazard > hazard_threshold:
        return 0  # Reduce rebuild rate to avoid latency issues
    else:
        return np.argmax(ACTIONS_MBPS)  # Use the highest available rate for balance

# Note: The policy assumes that the `ACTIONS_MBPS` list is defined somewhere in the environment.
