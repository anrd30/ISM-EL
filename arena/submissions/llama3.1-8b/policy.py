import numpy as np

def policy(obs: np.ndarray, state: dict) -> int:
    """
    Return 0, 1, 2 or 3.
    
    Parameters:
    obs (np.ndarray): A 12-element float32 vector with normalised observations.
    state (dict): A dictionary to store episode-specific information.
    
    Returns:
    int: The index of the rebuild rate from ACTIONS_MBPS.
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
    
    # Calculate the rebuild rate based on observations
    if latency_now > latency_target:
        # If we're over the limit, try to reduce load
        return 0
    elif fg_read_iops + fg_write_iops > 0.5 * bw_available:
        # If foreground load is high, reduce rebuild rate
        return 1
    elif hazard < 0.01 and redundancy_left > 0.2:
        # If failure rate is low and we have some redundancy left, try to finish quickly
        return 3
    else:
        # Otherwise, choose a moderate rebuild rate
        return 2
    
    # Store the chosen rebuild rate in state for future reference
    state['rebuild_rate'] = ACTIONS_MBPS[policy]
    
    return policy
