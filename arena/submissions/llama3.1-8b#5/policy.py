import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """
    Return 0, 1, 2 or 3.
    
    Args:
        obs (np.ndarray): A 12-element float32 vector containing various system metrics.
        state (dict): A dictionary to store information between intervals.

    Returns:
        int: The index of the rebuild rate action to take in [0, 3].
    """
    # Extract relevant observations
    fg_read_iops = obs[0]
    fg_write_iops = obs[1]
    fg_bw = obs[2]
    latency_now = obs[3]
    latency_target = obs[4]
    hazard = obs[5]
    redundancy_left = obs[7]
    progress = obs[8]
    
    # Calculate the rebuild rate based on the observations
    if latency_now > latency_target:
        # If we're close to violating the latency target, reduce the rebuild rate
        action_index = 0
    elif fg_bw < 0.5 and redundancy_left > 0.2:
        # If foreground bandwidth is low and there's still some redundancy left,
        # try to finish rebuilding faster
        action_index = 1 if progress < 0.7 else 3
    elif hazard > 0.5 or redundancy_left < 0.1:
        # If the failure rate is high or there's not much redundancy left, slow down
        action_index = 2
    else:
        # Otherwise, choose a moderate rebuild rate
        action_index = 2
    
    return action_index
